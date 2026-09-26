"""Accuracy of the trained networks on training and validation rows.

Relative error per grid cell is |network - Monte Carlo| / Monte Carlo, in implied-vol units,
exactly as in the authors' notebook. "Monte Carlo" is the vol stored in their data file.
Test rows are not touched here; src/final_eval.py does that once, at the end.

Usage:  python -m src.evaluate --model rbergomi [--quick]
"""
import argparse

import numpy as np
import pandas as pd

from src.calibrate import forward
from src.config import DATASETS, MATURITIES, STRIKES, results_dir
from src.data import load_train_val, unscale_vols
from src.model import load_weights


def relative_errors(weights, x_scaled: np.ndarray, vols: np.ndarray, scalers: dict) -> np.ndarray:
    """Relative error for every row and every one of the 88 cells, as a fraction."""
    predicted = unscale_vols(forward(weights, x_scaled), scalers)
    return np.abs(predicted - vols) / vols


def grid_table(errors: np.ndarray) -> pd.DataFrame:
    """Mean, standard deviation and maximum over rows for each grid cell, in percent (the three panels of Figure 6)."""
    grid = pd.MultiIndex.from_product([MATURITIES, STRIKES], names=["maturity", "strike"]).to_frame(index=False)
    grid["mean_pct"] = 100 * errors.mean(0)
    grid["std_pct"] = 100 * errors.std(0)
    grid["max_pct"] = 100 * errors.max(0)
    return grid


def summary_row(errors: np.ndarray) -> dict:
    """Headline numbers for one network on one set of rows, in percent."""
    cell_means = errors.mean(0)
    return {"n_rows": len(errors),
            "avg_rel_error_pct": 100 * errors.mean(),
            "largest_cell_mean_pct": 100 * cell_means.max(),
            "smallest_cell_mean_pct": 100 * cell_means.min(),
            "largest_cell_std_pct": 100 * errors.std(0).max(),
            "max_rel_error_pct": 100 * errors.max(),
            "p95_rel_error_pct": 100 * np.quantile(errors, 0.95),
            "p99_rel_error_pct": 100 * np.quantile(errors, 0.99),
            "share_of_cells_above_1pct": float((errors > 0.01).mean())}


def evaluate(model: str, quick: bool = False) -> pd.DataFrame:
    """Evaluate every trained seed on training and validation rows; write one grid file per seed and split."""
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    seeds = sorted(int(p.stem.split("seed")[1]) for p in folder.glob("weights_seed*.npz"))
    rows = []
    for seed in seeds:
        weights = load_weights(folder / f"weights_seed{seed}.npz")
        for split in ("train", "val"):
            errors = relative_errors(weights, data[f"x_{split}"], data[f"vols_{split}"], data["scalers"])
            grid_table(errors).to_csv(folder / f"accuracy_{split}_seed{seed}.csv", index=False)
            rows.append({"seed": seed, "split": split, **summary_row(errors)})
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "accuracy_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(evaluate(args.model, args.quick).round(4).to_string(index=False))
