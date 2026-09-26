"""Stretch: the authors' own calibrations of the same test surfaces, set beside ours.

The authors' repository publishes, for the first 5,000 test rows, the parameters their notebook fitted
(NNParameters*.txt) and those parameters re-priced by their external pricer (surfacesFromNN*.txt). They allow
two checks the paper alone does not:

1. how far apart the two calibrations land on the same surface: their fit against ours from the same SciPy
   call (least_squares with its default method) and from Levenberg-Marquardt, relative to the true value;
2. the surface RMSE of the paper's Figure 9 exactly as the notebook computes it: the root of the mean squared
   gap between the re-priced surface and the target, over the rows whose re-pricing did not fail. Ours is the
   network surface at our fit against the target; a third variant pushes their parameters through our network,
   which separates the calibration from the pricer.

This module never reads test rows itself. src/final_eval.py, the only reader of the test set,
hands it the rows and our fitted parameters:  python -m src.final_eval --model rbergomi --authors-calibration
"""
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.calibrate import forward
from src.config import DATA_DIR, DATASETS, N_PARAMS, N_VOLS
from src.data import scale_params, unscale_vols
from src.plots import INK, INK_MUTED, SERIES, SURFACE, apply_style, save

SENTINEL = -1e300
OURS = {"lm": "Levenberg-Marquardt", "trf": "least_squares default (TRF)"}


def load_authors_files(model: str) -> tuple[np.ndarray, np.ndarray]:
    """The authors' fitted parameters (full precision) and their re-priced vols, one row per test surface, in test order."""
    files = DATASETS[model]["authors_calibration"]
    params = np.loadtxt(DATA_DIR / files["params"][0])
    surfaces = np.loadtxt(DATA_DIR / files["surfaces"][0])
    assert params.shape[1] == N_PARAMS and surfaces.shape == (len(params), N_PARAMS + N_VOLS)
    assert np.allclose(params, surfaces[:, :N_PARAMS], rtol=1e-4, atol=1e-6), "the two files should hold the same fit at two precisions"
    return params, surfaces[:, N_PARAMS:]


def quantile_row(values: np.ndarray) -> dict:
    """Summary of one RMSE definition, in percent of implied vol."""
    v = 100 * np.asarray(values)
    return {"n_surfaces": len(v), "mean_pct": v.mean(), "median_pct": float(np.median(v)), "q95_pct": float(np.quantile(v, 0.95)),
            "q99_pct": float(np.quantile(v, 0.99)), "max_pct": v.max()}


def compare(model: str, weights, theta: np.ndarray, vols: np.ndarray, scalers: dict, ours: pd.DataFrame):
    """Per-parameter table, RMSE table and per-surface table for the first len(theta) test surfaces.

    theta, vols: true parameters and target surfaces of those rows. ours: our calibration table of the test set
    (every optimizer); its rows 0 to n-1 are the same surfaces, in the same order, because both we and the authors
    calibrate the first test rows in the authors' order.
    """
    n, names = len(theta), DATASETS[model]["param_names"]
    authors_params, authors_vols = (a[:n] for a in load_authors_files(model))
    fits, rmse_ours = {}, {}
    for short, optimizer in OURS.items():
        table = ours[ours["optimizer"] == optimizer].sort_values("row")
        assert (table["row"].to_numpy()[:n] == np.arange(n)).all(), "our calibration rows are not the first test surfaces"
        fits[short] = table[[f"fit_{p}" for p in names]].to_numpy()[:n]
        rmse_ours[short] = table["rmse"].to_numpy()[:n]

    def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """|a - b| relative to the true parameter, the error measure of the paper's Figure 9."""
        return np.abs(a - b) / np.abs(theta)

    errors = {"authors": relative(authors_params, theta), "lm": relative(fits["lm"], theta), "trf": relative(fits["trf"], theta)}
    if np.median(errors["authors"][:, names.index("nu")]) > 0.05:
        raise ValueError("the authors' calibrations do not line up with our test rows")
    gaps = {short: relative(fits[short], authors_params) for short in OURS}
    rows = []
    for k, name in enumerate(names + ["all"]):
        pick = (lambda m: m[:, k]) if name != "all" else (lambda m: m.ravel())
        rows.append({"parameter": name, "n_surfaces": n,
                     "authors_mean_pct": 100 * pick(errors["authors"]).mean(), "ours_lm_mean_pct": 100 * pick(errors["lm"]).mean(),
                     "ours_trf_mean_pct": 100 * pick(errors["trf"]).mean(),
                     "authors_median_pct": 100 * np.median(pick(errors["authors"])), "ours_lm_median_pct": 100 * np.median(pick(errors["lm"])),
                     "ours_trf_median_pct": 100 * np.median(pick(errors["trf"])),
                     "gap_trf_mean_pct": 100 * pick(gaps["trf"]).mean(), "gap_trf_median_pct": 100 * np.median(pick(gaps["trf"])),
                     "gap_lm_mean_pct": 100 * pick(gaps["lm"]).mean(), "gap_lm_median_pct": 100 * np.median(pick(gaps["lm"]))})
    per_param = pd.DataFrame(rows)

    failed = (authors_vols < SENTINEL).any(axis=1)
    zero_vols = ~failed & (authors_vols <= 0).any(axis=1)
    kept, clean = ~failed, ~failed & ~zero_vols
    rmse_paper = np.sqrt(((np.where(failed[:, None], np.nan, authors_vols) - vols) ** 2).mean(axis=1))
    predicted = unscale_vols(forward(weights, scale_params(authors_params, scalers)), scalers)
    rmse_theirs_our_network = np.sqrt(((predicted - vols) ** 2).mean(axis=1))
    rmse = pd.DataFrame([
        {"definition": "paper_as_coded", "description": "authors' re-priced surface against the target, root of the mean; rows whose re-pricing failed dropped, as in their notebook",
         "n_dropped": int(failed.sum()), "n_rows_with_zero_vol": int(zero_vols.sum()), **quantile_row(rmse_paper[kept])},
        {"definition": "paper_as_coded_no_zero_vols", "description": "as above, also dropping the rows whose re-priced surface holds a zero implied vol (failed inversions the notebook keeps)",
         "n_dropped": int(failed.sum() + zero_vols.sum()), "n_rows_with_zero_vol": 0, **quantile_row(rmse_paper[clean])},
        {"definition": "ours_lm", "description": "our network surface at our Levenberg-Marquardt fit against the target", "n_dropped": 0, "n_rows_with_zero_vol": 0,
         **quantile_row(rmse_ours["lm"])},
        {"definition": "ours_trf", "description": "our network surface at our fit from the authors' actual call (least_squares default) against the target", "n_dropped": 0,
         "n_rows_with_zero_vol": 0, **quantile_row(rmse_ours["trf"])},
        {"definition": "authors_params_our_network", "description": "our network surface at the authors' fitted parameters against the target", "n_dropped": 0,
         "n_rows_with_zero_vol": 0, **quantile_row(rmse_theirs_our_network)}])

    per_surface = pd.DataFrame({"row": np.arange(n), "repricing_failed": failed, "zero_vols": (authors_vols <= 0).sum(axis=1),
                                "rmse_paper_as_coded": rmse_paper, "rmse_ours_lm": rmse_ours["lm"], "rmse_ours_trf": rmse_ours["trf"],
                                "rmse_authors_params_our_network": rmse_theirs_our_network,
                                **{f"gap_trf_{p}": gaps["trf"][:, k] for k, p in enumerate(names)}})
    return per_param, rmse, per_surface


def draw(per_param: pd.DataFrame, per_surface: pd.DataFrame, model: str, quick: bool = False, label: str = "test") -> None:
    """Left: mean relative error of their fit and ours per parameter, and the median gap between the two fits.
    Right: quantile curves of the surface RMSE under four definitions."""
    apply_style()
    params = per_param[per_param["parameter"] != "all"]
    x, n = np.arange(len(params)), int(params["n_surfaces"].iloc[0])
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    marker = {"markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 1.5, "linestyle": "none"}
    left.plot(x, params["authors_mean_pct"], "o", color=SERIES[1], label="Authors' published fit (error recomputed by us)", **marker)
    left.plot(x, params["ours_lm_mean_pct"], "o", color=SERIES[0], label="Our fit", **marker)
    left.plot(x, params["gap_trf_median_pct"], "D", color=INK_MUTED, label="Median gap between the two fits", **marker)
    left.set_yscale("log")
    left.set_ylim(top=6 * params["ours_lm_mean_pct"].max())
    left.set_xticks(x, DATASETS[model]["param_labels"])
    left.set_ylabel("Relative parameter error (%, log scale)")
    left.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    left.set_title(f"Mean relative parameter error on the same {n:,} surfaces", loc="left")
    left.legend(loc="upper left")

    q = np.linspace(0, 1, 400)
    kept = per_surface["rmse_paper_as_coded"].notna()
    clean = kept & (per_surface["zero_vols"] == 0)
    curves = [(per_surface.loc[kept, "rmse_paper_as_coded"], SERIES[1], "-", "Paper as coded: their re-priced surfaces"),
              (per_surface.loc[clean, "rmse_paper_as_coded"], SERIES[1], "--", "Same, without the surfaces holding a zero vol"),
              (per_surface["rmse_ours_lm"], SERIES[0], "-", "Ours: network surface at our fit"),
              (per_surface["rmse_authors_params_our_network"], SERIES[2], "-", "Their parameters through our network")]
    for values, colour, style, text in curves:
        right.plot(100 * q, 100 * np.quantile(values.to_numpy(), q), color=colour, linestyle=style, label=f"{text} ({len(values):,})")
    right.axvline(99, color=INK, linestyle=":", linewidth=1.2, label="99% quantile")
    right.set_xlabel("Quantile")
    right.set_ylabel("Surface RMSE in implied vol")
    right.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    right.set_title("Surface RMSE after calibration, by definition (surfaces in brackets)", loc="left")
    right.legend(loc="upper left")
    fig.suptitle(f"{DATASETS[model]['title']}: the authors' published calibrations of the first {n:,} test surfaces, against ours", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, model, f"authors_calibration_{label}", quick)
