"""Loading, integrity checks, the locked split and the scalers.

The test set is locked. This module builds the test indices and writes them to disk, but its
loaders hand out training and validation rows only. The one script allowed to read test rows
is src/final_eval.py, and src/tests/test_no_test_leak.py enforces that.

Usage:  python -m src.data --model rbergomi [--quick]
"""
import argparse
import gzip
import json

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (DATA_DIR, DATASETS, MATURITIES, N_PARAMS, N_ROWS, N_VOLS, QUICK, SPLIT,
                        SPLIT_FILE, STRIKES, results_dir)


def load_raw(model: str) -> np.ndarray:
    """Read one data file as an (80000, 99) float64 array: 11 parameters, then 88 implied vols.

    Despite the .txt.gz name the file is a gzipped NumPy binary, so it needs np.load, not np.loadtxt.
    Only this module and src/final_eval.py may call this function, because it returns every row.
    """
    path = DATA_DIR / DATASETS[model]["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python data/get_data.py")
    with gzip.open(path, "rb") as f:
        return np.load(f)


def check_raw(model: str, raw: np.ndarray) -> dict:
    """Assert the file-level facts we rely on and return them for results/<model>/data_facts.json.

    These are integrity checks on the whole file (shape, type, missing values, ranges). They
    feed no modelling decision, which is why they may look at all 80,000 rows.
    """
    cfg = DATASETS[model]
    lower, upper = np.array(cfg["lower"]), np.array(cfg["upper"])
    params, vols = raw[:, :N_PARAMS], raw[:, N_PARAMS:]

    assert raw.shape == (N_ROWS, N_PARAMS + N_VOLS), raw.shape
    assert raw.dtype == np.float64, raw.dtype
    assert np.isfinite(raw).all(), "file contains NaN or inf"

    tolerance = 0.01 * (upper - lower)
    assert (params.min(0) >= lower - 1e-12).all() and (params.max(0) <= upper + 1e-12).all()
    assert (params.min(0) <= lower + tolerance).all() and (params.max(0) >= upper - tolerance).all()

    vol_range = (round(float(vols.min()), 3), round(float(vols.max()), 3))
    assert vol_range == tuple(cfg["vol_range_3dp"]), vol_range

    return {
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "n_nan_or_inf": int((~np.isfinite(raw)).sum()),
        "param_names": cfg["param_names"],
        "param_min": params.min(0).tolist(),
        "param_max": params.max(0).tolist(),
        "vol_min": float(vols.min()),
        "vol_max": float(vols.max()),
    }


def check_orientation(model: str, vols_train: np.ndarray) -> dict:
    """Guard the reshape convention: vols.reshape(8, 11) must give rows = maturities, columns = strikes.

    With the right orientation the mean smile is steepest at the shortest maturity and flattens
    as maturity grows, and its short-maturity minimum sits at a known strike. A transposed or
    strike-major layout fails both checks. Uses training rows only.
    """
    mean_surface = vols_train.mean(0).reshape(len(MATURITIES), len(STRIKES))
    smile_range = mean_surface.max(1) - mean_surface.min(1)
    assert (np.diff(smile_range) < 0).all(), "smile does not flatten with maturity: wrong orientation?"
    row0_min_strike = STRIKES[int(mean_surface[0].argmin())]
    assert row0_min_strike == DATASETS[model]["row0_min_strike"], row0_min_strike
    return {"mean_surface_train": mean_surface.tolist(), "row0_min_strike": row0_min_strike}


def edge_shares(model: str, theta_train: np.ndarray) -> dict:
    """Share of training rows in the lowest and highest 5% of each parameter's range.

    Uniform sampling would put 5% of rows in each band. A smaller share means the network saw
    fewer examples there, which matters for the edge-of-box experiment.
    """
    cfg = DATASETS[model]
    lower, upper = np.array(cfg["lower"]), np.array(cfg["upper"])
    band = 0.05 * (upper - lower)
    low = (theta_train < lower + band).mean(0)
    high = (theta_train > upper - band).mean(0)
    return {"share_in_lowest_5pct": dict(zip(cfg["param_names"], low.tolist())),
            "share_in_highest_5pct": dict(zip(cfg["param_names"], high.tolist()))}


def build_split() -> dict:
    """Return the three index sets: 60,000 train, 8,000 validation, 12,000 test.

    Step 1 repeats the authors' call, train_test_split(test_size=0.15, random_state=42), on row
    indices. The shuffle depends only on the number of rows and the seed, so it selects exactly
    the rows their notebook tests on, in the same order (they calibrate the first 5,000).
    Step 2 carves the validation rows out of their 68,000 training rows with our own seed.
    """
    train68, test = train_test_split(np.arange(N_ROWS), test_size=SPLIT["test_size"],
                                     random_state=SPLIT["authors_random_state"])
    order = np.random.default_rng(SPLIT["val_seed"]).permutation(len(train68))
    val = train68[order[:SPLIT["n_val"]]]
    train = train68[order[SPLIT["n_val"]:]]
    return {"train_idx": train, "val_idx": val, "test_idx": test}


def save_or_verify_split() -> None:
    """Write the split once. If the file exists, insist that it equals a fresh rebuild.

    Verifying instead of overwriting means a changed seed or library can never silently move
    rows between the training and the test set.
    """
    split = build_split()
    if SPLIT_FILE.exists():
        with np.load(SPLIT_FILE) as saved:
            for name, idx in split.items():
                assert np.array_equal(saved[name], idx), f"{name} in {SPLIT_FILE} differs from a rebuild"
        return
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLIT_FILE, **split)


def load_train_val_idx(quick: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for training and validation. Quick mode keeps the first rows of each set."""
    with np.load(SPLIT_FILE) as saved:
        train, val = saved["train_idx"], saved["val_idx"]
    if quick:
        train, val = train[:QUICK["n_train"]], val[:QUICK["n_val"]]
    return train, val


def scale_params(theta: np.ndarray, scalers: dict) -> np.ndarray:
    """Map parameters from their sampling box to [-1, 1] (paper Section 3.2.2, item 2)."""
    lower, upper = scalers["lower"], scalers["upper"]
    return (2.0 * theta - (upper + lower)) / (upper - lower)


def unscale_params(x: np.ndarray, scalers: dict) -> np.ndarray:
    """Inverse of scale_params."""
    lower, upper = scalers["lower"], scalers["upper"]
    return 0.5 * (x * (upper - lower) + (upper + lower))


def scale_vols(vols: np.ndarray, scalers: dict) -> np.ndarray:
    """Standardize each of the 88 grid cells with the training mean and standard deviation (item 3)."""
    return (vols - scalers["vol_mean"]) / scalers["vol_std"]


def unscale_vols(y: np.ndarray, scalers: dict) -> np.ndarray:
    """Inverse of scale_vols: back to implied-vol units."""
    return y * scalers["vol_std"] + scalers["vol_mean"]


def fit_scalers(model: str, vols_train: np.ndarray) -> dict:
    """Parameter bounds come from the configuration; vol statistics from training rows only.

    Fitting on training rows only keeps validation and test information out of the inputs the
    network is trained on. np.std uses ddof=0, the same convention as scikit-learn's StandardScaler.
    """
    return {"lower": np.array(DATASETS[model]["lower"]), "upper": np.array(DATASETS[model]["upper"]),
            "vol_mean": vols_train.mean(0), "vol_std": vols_train.std(0)}


def load_scalers(model: str, quick: bool = False) -> dict:
    """Read the scalers saved by prepare()."""
    with np.load(results_dir(model, quick) / "scalers.npz") as saved:
        return {name: saved[name] for name in saved.files}


def prepare(model: str, quick: bool = False) -> dict:
    """Check the file, fix the split, fit and save the scalers, and write the data facts."""
    raw = load_raw(model)
    facts = check_raw(model, raw)
    save_or_verify_split()
    train_idx, val_idx = load_train_val_idx(quick)
    theta_train, vols_train = raw[train_idx, :N_PARAMS], raw[train_idx, N_PARAMS:]

    scalers = fit_scalers(model, vols_train)
    out = results_dir(model, quick)
    np.savez(out / "scalers.npz", **scalers)

    with np.load(SPLIT_FILE) as saved:
        facts["split_sizes"] = {name: int(saved[name].size) for name in saved.files}
    facts["n_train_rows_used"] = int(train_idx.size)
    facts["n_val_rows_used"] = int(val_idx.size)
    if not quick:
        facts.update(check_orientation(model, vols_train))
    facts.update(edge_shares(model, theta_train))
    with open(out / "data_facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    return facts


def load_train_val(model: str, quick: bool = False) -> dict:
    """Training and validation rows, raw and scaled, plus the scalers. Never returns test rows."""
    raw = load_raw(model)
    train_idx, val_idx = load_train_val_idx(quick)
    scalers = load_scalers(model, quick)
    data = {"scalers": scalers}
    for name, idx in (("train", train_idx), ("val", val_idx)):
        theta, vols = raw[idx, :N_PARAMS], raw[idx, N_PARAMS:]
        data[f"theta_{name}"], data[f"vols_{name}"] = theta, vols
        data[f"x_{name}"], data[f"y_{name}"] = scale_params(theta, scalers), scale_vols(vols, scalers)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    facts = prepare(args.model, args.quick)
    print(f"{args.model}: checks passed, split {facts['split_sizes']}, "
          f"scalers fitted on {facts['n_train_rows_used']} training rows")
