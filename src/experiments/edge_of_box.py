"""Upgrade 2 (robustness check): is the network worse near the edges of the training box?

The paper tests on parameters drawn from the same box as the training data, and its own SPX
calibration ends up near rho = -1 (its Figure 11). A network has fewer neighbours to learn from
near an edge, so we compare errors in the outer 5% of each parameter's range with the inside,
and look separately at strongly negative rho and at low H.

The error of one row is its relative error averaged over the 88 grid cells, in percent.

Usage:  python -m src.experiments.edge_of_box --model rbergomi [--quick]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATASETS, results_dir
from src.data import load_train_val
from src.evaluate import relative_errors
from src.model import load_weights
from src.plots import AXIS, INK_MUTED, INK_SECONDARY, SERIES, SPLIT_NAMES, SURFACE, apply_style, save

BAND = 0.9


def _stats(analysis, parameter, region, low, high, row_error, row_max, mask) -> dict:
    return {"analysis": analysis, "parameter": parameter, "region": region, "low": low, "high": high,
            "n_rows": int(mask.sum()), "mean_row_error_pct": row_error[mask].mean() if mask.any() else np.nan,
            "p95_row_error_pct": np.quantile(row_error[mask], 0.95) if mask.any() else np.nan,
            "mean_row_max_error_pct": row_max[mask].mean() if mask.any() else np.nan}


def compute(model: str, weights, x_scaled: np.ndarray, theta: np.ndarray, vols: np.ndarray, scalers: dict) -> pd.DataFrame:
    """Three analyses in one long table: distance from the centre, outer band per parameter, focus regions."""
    errors = 100 * relative_errors(weights, x_scaled, vols, scalers)
    row_error, row_max = errors.mean(1), errors.max(1)
    names = DATASETS[model]["param_names"]
    rows = []

    distance = np.abs(x_scaled).max(1)
    edges = np.quantile(distance, np.linspace(0, 1, 11))
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (distance >= low) & (distance <= high if high == edges[-1] else distance < high)
        rows.append(_stats("distance_decile", "all", "decile", low, high, row_error, row_max, mask))

    for j, name in enumerate(names):
        for region, mask in (("inside", np.abs(x_scaled[:, j]) <= BAND), ("lowest_5pct", x_scaled[:, j] < -BAND),
                             ("highest_5pct", x_scaled[:, j] > BAND)):
            rows.append(_stats("outer_band", name, region, np.nan, np.nan, row_error, row_max, mask))

    for name, threshold in DATASETS[model]["edge_focus"]:
        j = names.index(name)
        bins = np.linspace(DATASETS[model]["lower"][j], DATASETS[model]["upper"][j], 21)
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (theta[:, j] >= low) & (theta[:, j] <= high)
            rows.append(_stats("focus_bin", name, "bin", low, high, row_error, row_max, mask))
        rows.append(_stats("focus_region", name, f"below_{threshold:g}", np.nan, threshold, row_error, row_max, theta[:, j] < threshold))
        rows.append(_stats("focus_region", name, "rest", threshold, np.nan, row_error, row_max, theta[:, j] >= threshold))
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame, model: str, quick: bool, split: str) -> None:
    """Panel A: error by distance decile. Panel B: inside against the two outer bands. Panels C, D: focus regions."""
    apply_style()
    names, labels = DATASETS[model]["param_names"], DATASETS[model]["param_labels"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    a, b, c, d = axes.ravel()

    dist = table[table["analysis"] == "distance_decile"]
    a.plot(0.5 * (dist["low"] + dist["high"]), dist["mean_row_error_pct"], "o-", color=SERIES[0], markersize=8,
           markeredgecolor=SURFACE, markeredgewidth=2)
    a.set_xlabel("Distance from the box centre, by decile (1 = on the boundary)")
    a.set_ylabel("Mean row error (%)")
    a.set_title("A. Error against distance from the centre", loc="left")
    a.set_ylim(0, None)

    band = table[table["analysis"] == "outer_band"].pivot(index="parameter", columns="region", values="mean_row_error_pct").loc[names]
    x = np.arange(len(names))
    for region, colour, label, shift in (("inside", INK_MUTED, "Inside (middle 90% of the range)", 0.0),
                                         ("lowest_5pct", SERIES[0], "Lowest 5% of the range", -0.18),
                                         ("highest_5pct", SERIES[1], "Highest 5% of the range", 0.18)):
        b.plot(x + shift, band[region], "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
               linestyle="none", label=label)
    b.set_xticks(x, labels)
    b.set_ylabel("Mean row error (%)")
    b.set_title("B. Outer 5% of each parameter's range against the inside", loc="left")
    b.set_ylim(0, None)
    b.legend(loc="upper left")

    for ax, (name, threshold), letter in zip((c, d), DATASETS[model]["edge_focus"], "CD"):
        bins = table[(table["analysis"] == "focus_bin") & (table["parameter"] == name)]
        region = table[(table["analysis"] == "focus_region") & (table["parameter"] == name)].set_index("region")
        centre = 0.5 * (bins["low"] + bins["high"])
        ax.axvspan(bins["low"].min(), threshold, color=SERIES[1], alpha=0.10, linewidth=0)
        ax.plot(centre, bins["p95_row_error_pct"], color=AXIS, label="95% quantile of row error")
        ax.plot(centre, bins["mean_row_error_pct"], color=SERIES[0], label="Mean row error")
        inside, rest = region.loc[f"below_{threshold:g}"], region.loc["rest"]
        ax.set_title(f"{letter}. {labels[names.index(name)]} below {threshold:g}: mean {inside['mean_row_error_pct']:.2f}% "
                     f"({int(inside['n_rows']):,} rows) against {rest['mean_row_error_pct']:.2f}% elsewhere", loc="left")
        ax.set_xlabel(labels[names.index(name)])
        ax.set_ylabel("Row error (%)")
        ax.set_ylim(0, None)
        ax.legend(loc="upper right")
    for ax in (c, d)[len(DATASETS[model]["edge_focus"]):]:
        ax.set_visible(False)
    n_rows = int(table[table["analysis"] == "distance_decile"]["n_rows"].sum())
    fig.suptitle(f"{DATASETS[model]['title']}: edge-of-box test on {n_rows:,} {SPLIT_NAMES[split]} rows, seed 0. "
                 "Row error = relative error averaged over the 88 grid cells.", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Shaded: the focus region.", color=INK_SECONDARY, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, model, f"edge_of_box_{split}", quick)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    weights = load_weights(folder / "weights_seed0.npz")
    table = compute(model, weights, data["x_val"], data["theta_val"], data["vols_val"], data["scalers"])
    table.to_csv(folder / "edge_of_box_val.csv", index=False)
    draw(table, model, quick, "val")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = main(args.model, args.quick)
    print(result[result["analysis"].isin(["outer_band", "focus_region"])].round(3).to_string(index=False))
