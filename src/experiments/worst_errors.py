"""Critique evidence: averages hide the tail. Where do the largest errors come from?

The paper's headline is an average below 0.5%, next to a maximum of 25% in its text and about 50%
on the colour bar of its Figure 6. For every row we take the largest relative error over the 88
grid cells, then map that against two model parameters and against the grid cell where it occurs.

Usage:  python -m src.experiments.worst_errors --model rbergomi [--quick]
"""
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.config import DATASETS, MATURITIES, STRIKES, results_dir
from src.data import load_train_val
from src.evaluate import relative_errors
from src.model import load_weights
from src.paper_numbers import PAPER
from src.plots import AXIS, SPLIT_NAMES, apply_style, save

N_BINS = 8
WORST_SHARE = 0.01


def compute(model: str, weights, x_scaled: np.ndarray, theta: np.ndarray, vols: np.ndarray, scalers: dict) -> pd.DataFrame:
    """Two analyses in one long table: row-maximum error on a parameter grid, and the grid cells of the worst rows."""
    errors = 100 * relative_errors(weights, x_scaled, vols, scalers)
    row_max, worst_cell = errors.max(1), errors.argmax(1)
    names, (first, second) = DATASETS[model]["param_names"], DATASETS[model]["worst_error_axes"]
    i, j = names.index(first), names.index(second)
    edges_i = np.linspace(DATASETS[model]["lower"][i], DATASETS[model]["upper"][i], N_BINS + 1)
    edges_j = np.linspace(DATASETS[model]["lower"][j], DATASETS[model]["upper"][j], N_BINS + 1)
    bin_i = np.clip(np.digitize(theta[:, i], edges_i) - 1, 0, N_BINS - 1)
    bin_j = np.clip(np.digitize(theta[:, j], edges_j) - 1, 0, N_BINS - 1)
    threshold = np.quantile(row_max, 1 - WORST_SHARE)

    rows = []
    for a in range(N_BINS):
        for b in range(N_BINS):
            mask = (bin_i == a) & (bin_j == b)
            rows.append({"analysis": "parameter_grid", "first": first, "first_low": edges_i[a], "first_high": edges_i[a + 1],
                         "second": second, "second_low": edges_j[b], "second_high": edges_j[b + 1], "n_rows": int(mask.sum()),
                         "mean_row_max_pct": row_max[mask].mean() if mask.any() else np.nan,
                         "share_of_rows_above_1pct": float((row_max[mask] > PAPER["bid_ask_rel_pct"][0]).mean()) if mask.any() else np.nan,
                         "n_worst_rows": int((row_max[mask] >= threshold).sum())})
    worst = row_max >= threshold
    for cell in range(errors.shape[1]):
        rows.append({"analysis": "worst_cell", "maturity": MATURITIES[cell // len(STRIKES)], "strike": STRIKES[cell % len(STRIKES)],
                     "n_worst_rows": int((worst_cell[worst] == cell).sum()), "n_rows": int(worst.sum()),
                     "worst_row_threshold_pct": threshold})
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame, model: str, quick: bool, split: str) -> None:
    """Left: mean row-maximum error on the two-parameter grid. Right: where on the 8 x 11 grid the worst rows peak.

    Magnitude is encoded with one hue, light to dark. (The Figure 6 heatmaps use viridis only to match the paper.)
    """
    apply_style()
    names, labels = DATASETS[model]["param_names"], DATASETS[model]["param_labels"]
    grid = table[table["analysis"] == "parameter_grid"]
    first, second = grid["first"].iloc[0], grid["second"].iloc[0]
    values = grid["mean_row_max_pct"].to_numpy().reshape(N_BINS, N_BINS).T
    cells = table[table["analysis"] == "worst_cell"]
    counts = cells["n_worst_rows"].to_numpy().reshape(len(MATURITIES), len(STRIKES))

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    extent = [grid["first_low"].min(), grid["first_high"].max(), grid["second_low"].min(), grid["second_high"].max()]
    image = left.imshow(values, origin="lower", extent=extent, aspect="auto", cmap="Blues")
    bar = fig.colorbar(image, ax=left, format=matplotlib.ticker.PercentFormatter(decimals=1))
    bar.outline.set_edgecolor(AXIS)
    bar.set_label("Largest relative error on a surface, mean over rows")
    left.set_xlabel(labels[names.index(first)])
    left.set_ylabel(labels[names.index(second)])
    left.set_title("Mean over rows of the largest error on the surface", loc="left")
    left.grid(False)

    image = right.imshow(counts, cmap="Blues", aspect="auto")
    bar = fig.colorbar(image, ax=right)
    bar.outline.set_edgecolor(AXIS)
    bar.set_label("Number of rows")
    right.set_xticks(range(len(STRIKES)), [f"{k:.1f}" for k in STRIKES], fontsize=9)
    right.set_yticks(range(len(MATURITIES)), [f"{t:.1f}" for t in MATURITIES])
    right.set_xlabel("Strike")
    right.set_ylabel("Maturity")
    n_worst, threshold = int(cells["n_rows"].iloc[0]), cells["worst_row_threshold_pct"].iloc[0]
    right.set_title(f"Grid cell of the largest error, worst {n_worst:,} rows (largest error above {threshold:.1f}%)", loc="left")
    right.grid(False)
    fig.suptitle(f"{DATASETS[model]['title']}: where the largest errors come from, {int(grid['n_rows'].sum()):,} "
                 f"{SPLIT_NAMES[split]} rows, seed 0", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, model, f"worst_errors_{split}", quick)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    weights = load_weights(folder / "weights_seed0.npz")
    table = compute(model, weights, data["x_val"], data["theta_val"], data["vols_val"], data["scalers"])
    table.to_csv(folder / "worst_errors_val.csv", index=False)
    draw(table, model, quick, "val")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = main(args.model, args.quick)
    print(result[result["analysis"] == "parameter_grid"].nlargest(5, "mean_row_max_pct").round(3).to_string(index=False))
