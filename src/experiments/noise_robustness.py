"""Stretch: how much do noisy quotes move the calibrated parameters?

Market quotes are not exact. We add 0.5% relative Gaussian noise (half the paper's bid-ask yardstick) to each
validation surface, calibrate with Levenberg-Marquardt, and compare parameter errors with and without the noise.

Usage:  python -m src.experiments.noise_robustness --model rbergomi [--quick]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibrate import GRADIENT_BASED, Problem
from src.config import DATASETS, QUICK, results_dir
from src.data import load_train_val, scale_vols, unscale_params
from src.model import load_weights
from src.plots import INK_MUTED, SERIES, SURFACE, apply_style, save

NOISE = 0.005
N_SURFACES = 1_000
NOISE_SEED = 0


def parameter_errors(weights, targets_scaled: np.ndarray, theta: np.ndarray, scalers: dict) -> np.ndarray:
    """Relative error of every parameter after Levenberg-Marquardt, one row per surface."""
    solve = GRADIENT_BASED["Levenberg-Marquardt"]
    fitted = np.array([unscale_params(solve(Problem(weights, y)).x, scalers) for y in targets_scaled])
    return np.abs(fitted - theta) / np.abs(theta)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    data, folder, names = load_train_val(model, quick), results_dir(model, quick), DATASETS[model]["param_names"]
    n = QUICK["n_calibrations"] if quick else N_SURFACES
    weights, theta, vols = load_weights(folder / "weights_seed0.npz"), data["theta_val"][:n], data["vols_val"][:n]
    noisy = vols * (1 + NOISE * np.random.default_rng(NOISE_SEED).standard_normal(vols.shape))
    clean_errors = parameter_errors(weights, scale_vols(vols, data["scalers"]), theta, data["scalers"])
    noisy_errors = parameter_errors(weights, scale_vols(noisy, data["scalers"]), theta, data["scalers"])
    table = pd.DataFrame({"parameter": names, "noise_rel_std": NOISE, "n_surfaces": n,
                          "clean_mean_pct": 100 * clean_errors.mean(0), "noisy_mean_pct": 100 * noisy_errors.mean(0),
                          "clean_median_pct": 100 * np.median(clean_errors, 0), "noisy_median_pct": 100 * np.median(noisy_errors, 0)})
    table.to_csv(folder / "noise_robustness_val.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.vlines(x, table["clean_median_pct"], table["noisy_median_pct"], color=INK_MUTED, linewidth=1.5, zorder=1)
    ax.plot(x, table["clean_median_pct"], "o", color=SERIES[0], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label="Exact surfaces")
    ax.plot(x, table["noisy_median_pct"], "o", color=SERIES[1], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
            label=f"With {NOISE:.1%} relative noise on every vol")
    ax.set_yscale("log")
    ax.set_xticks(x, DATASETS[model]["param_labels"])
    ax.set_ylabel("Median relative parameter error (%, log scale)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title(f"{DATASETS[model]['title']}: Levenberg-Marquardt calibration of {n:,} validation surfaces, with and without noise", loc="left")
    ax.legend(loc="upper left")
    save(fig, model, "noise_robustness_val", quick)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(3).to_string(index=False))
