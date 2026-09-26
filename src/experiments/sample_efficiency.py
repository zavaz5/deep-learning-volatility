"""Upgrade 3a (extension): how many Monte Carlo samples does the network need?

Every training sample costs a Monte Carlo run, so the offline cost is proportional to the size
of the training set. We train on nested subsets of our 60,000 training rows and measure the
error on the same validation rows. (The guide suggests going up to 68,000; our honest training
set has 60,000 because 8,000 rows were set aside for validation.)

Each subset gets its own vol scaler, fitted on that subset only, as someone who owned only that
many samples would have to do. The 60,000-row point reuses the networks trained in src/train.py.

Usage:  python -m src.experiments.sample_efficiency --model rbergomi [--quick]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibrate import forward
from src.config import DATASETS, QUICK, RECIPE, results_dir
from src.data import fit_scalers, load_train_val, scale_params, scale_vols, unscale_vols
from src.model import load_weights, numpy_weights
from src.plots import INK_SECONDARY, SERIES, SPLIT_NAMES, SURFACE, apply_style, save
from src.train import fit

SIZES = [5_000, 10_000, 20_000, 40_000, 60_000]
SEEDS = [0, 1, 2]


def error_on(weights, scalers: dict, theta: np.ndarray, vols: np.ndarray) -> dict:
    """Average and maximum relative error on the given rows, in percent, using that run's own scalers."""
    errors = np.abs(unscale_vols(forward(weights, scale_params(theta, scalers)), scalers) - vols) / vols
    return {"avg_rel_error_pct": 100 * errors.mean(), "max_rel_error_pct": 100 * errors.max()}


def score_saved_runs(model: str, quick: bool, theta: np.ndarray, vols: np.ndarray) -> pd.DataFrame:
    """Score every saved subset network on other rows (used for the test-set version of the figure)."""
    rows = []
    for path in sorted((results_dir(model, quick) / "sample_efficiency").glob("n*_seed*.npz")):
        n, seed = (int(part.lstrip("nsed")) for part in path.stem.split("_"))
        rows.append({"n_train": n, "seed": seed, **error_on(*load_run(path), theta, vols)})
    return pd.DataFrame(rows).sort_values(["n_train", "seed"])


def one_run(job: tuple) -> dict:
    """Train on the first n training rows with one seed. Runs in its own process, one thread."""
    model, quick, n, seed, max_epochs = job
    data = load_train_val(model, quick)
    full = len(data["theta_train"])
    if n == full and not quick:
        weights, scalers, epochs, seconds = load_weights(results_dir(model) / f"weights_seed{seed}.npz"), data["scalers"], None, None
    else:
        theta, vols = data["theta_train"][:n], data["vols_train"][:n]
        scalers = fit_scalers(model, vols)
        net, _, summary = fit(scale_params(theta, scalers), scale_vols(vols, scalers),
                              scale_params(data["theta_val"], scalers), scale_vols(data["vols_val"], scalers),
                              seed, max_epochs, verbose=False)
        weights, epochs, seconds = numpy_weights(net), summary["epochs_run"], summary["seconds"]
    save_run(results_dir(model, quick) / "sample_efficiency" / f"n{n}_seed{seed}.npz", weights, scalers)
    return {"n_train": n, "seed": seed, **error_on(weights, scalers, data["theta_val"], data["vols_val"]),
            "epochs_run": epochs, "train_seconds": seconds}


def save_run(path, weights, scalers: dict) -> None:
    """Keep each subset network with its own vol scaler, so it can be scored on the test rows later without retraining."""
    path.parent.mkdir(exist_ok=True)
    arrays = {"vol_mean": scalers["vol_mean"], "vol_std": scalers["vol_std"], "lower": scalers["lower"], "upper": scalers["upper"]}
    for k, (W, b) in enumerate(weights):
        arrays[f"W{k}"], arrays[f"b{k}"] = W, b
    np.savez(path, **arrays)


def load_run(path) -> tuple:
    """(weights, scalers) saved by save_run."""
    with np.load(path) as saved:
        scalers = {name: saved[name] for name in ("vol_mean", "vol_std", "lower", "upper")}
        weights = [(saved[f"W{k}"], saved[f"b{k}"]) for k in range((len(saved.files) - 4) // 2)]
    return weights, scalers


def draw(table: pd.DataFrame, model: str, quick: bool, split: str = "val") -> None:
    """Error against training-set size on log axes: one dot per seed, a line through the means."""
    apply_style()
    means = table.groupby("n_train")["avg_rel_error_pct"].mean()
    slope = np.polyfit(np.log(means.index), np.log(means.to_numpy()), 1)[0]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(means.index, means.to_numpy(), color=SERIES[0], zorder=2, label="Mean over seeds")
    ax.plot(table["n_train"], table["avg_rel_error_pct"], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none", zorder=3, label="One seed")
    lowest = table.groupby("n_train")["avg_rel_error_pct"].min()
    for n, value in means.items():
        ax.annotate(f"mean {value:.2f}%", (n, lowest[n]), xytext=(0, -18), textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(means.index), [f"{n:,}" for n in means.index])
    ticks = [t for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 3, 5, 7, 10) if means.min() * 0.8 <= t <= means.max() * 1.25]
    ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(means.min() * 0.8, means.max() * 1.25)
    ax.set_xlabel("Training rows = Monte Carlo runs needed (log scale)")
    ax.set_ylabel(f"Average relative error on {SPLIT_NAMES[split]} rows (log scale)")
    ax.set_title(f"{DATASETS[model]['title']}: {means.index.min():,} training rows give {means.iloc[0]:.2f}%, "
                 f"{means.index.max():,} give {means.iloc[-1]:.2f}%\n{means.index.max():,} rows is our full training set "
                 f"(the paper trains on 68,000). Log-log slope over the whole range: {slope:.2f}", loc="left", fontsize=11)
    ax.legend(loc="upper right")
    save(fig, model, f"sample_efficiency_{split}", quick)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    sizes = [400, 800, QUICK["n_train"]] if quick else SIZES
    seeds = [0] if quick else SEEDS
    max_epochs = QUICK["max_epochs"] if quick else RECIPE["max_epochs"]
    jobs = [(model, quick, n, seed, max_epochs) for n in sizes for seed in seeds]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one_run, jobs))
    table = pd.DataFrame(rows).sort_values(["n_train", "seed"])
    table.to_csv(results_dir(model, quick) / "sample_efficiency_val.csv", index=False)
    draw(table, model, quick)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(4).to_string(index=False))
