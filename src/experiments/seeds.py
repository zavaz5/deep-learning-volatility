"""Upgrade 1 (fixed flaw): does the headline accuracy depend on the random seed?

The paper reports one run, stopped by watching the test set. We train five seeds with a separate
validation set and report the mean and the spread of the average relative error.

Usage:  python -m src.experiments.seeds --model rbergomi [--quick]
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATASETS, results_dir
from src.paper_numbers import PAPER
from src.plots import INK, INK_SECONDARY, SERIES, SPLIT_NAMES, SURFACE, apply_style, save


def seed_table(accuracy_summary: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per seed plus a mean row and a sample-standard-deviation row, all in percent."""
    columns = ["avg_rel_error_pct", "largest_cell_mean_pct", "p99_rel_error_pct", "max_rel_error_pct"]
    per_seed = accuracy_summary[accuracy_summary["split"] == split][["seed"] + columns].copy()
    per_seed["seed"] = per_seed["seed"].astype(str)
    stats = pd.DataFrame([{"seed": "mean", **per_seed[columns].mean()}, {"seed": "std", **per_seed[columns].std(ddof=1)}])
    table = pd.concat([per_seed, stats], ignore_index=True)
    table.insert(0, "split", split)
    return table


def draw(table: pd.DataFrame, model: str, quick: bool, split: str) -> None:
    """Each seed's average relative error as a dot, their mean as a line, the paper's bound as reference."""
    apply_style()
    seeds = table[~table["seed"].isin(["mean", "std"])]
    mean = float(table.loc[table["seed"] == "mean", "avg_rel_error_pct"].iloc[0])
    std = float(table.loc[table["seed"] == "std", "avg_rel_error_pct"].iloc[0])
    bound, source = PAPER["avg_rel_error_bound_pct"]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axvspan(mean - std, mean + std, color=SERIES[0], alpha=0.10, linewidth=0)
    ax.axvline(mean, color=SERIES[0], linewidth=2)
    heights = [0.3 * (2 * k / max(1, len(seeds) - 1) - 1) for k in range(len(seeds))]
    ax.plot(seeds["avg_rel_error_pct"], heights, "o", color=SERIES[0], markersize=9, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none")
    ax.axvline(bound, color=INK, linestyle="--", linewidth=1.2)
    ax.annotate(f"paper's bound: {bound:g}%", (bound, 0.62), xytext=(7, 0), textcoords="offset points", ha="left",
                color=INK_SECONDARY, fontsize=10)
    ax.annotate(f"mean {mean:.3f}%, std {std:.3f}%\nover {len(seeds)} seeds", (mean - std, -0.62), xytext=(-9, 0), textcoords="offset points",
                ha="right", va="center", color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(0, max(bound, seeds["avg_rel_error_pct"].max()) * 1.15)
    ax.set_xlabel("Average relative error over all rows and grid cells (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{DATASETS[model]['title']}: accuracy of five seeds on {SPLIT_NAMES[split]} rows", loc="left")
    save(fig, model, f"seeds_{split}", quick)


def main(model: str, quick: bool = False, split: str = "val") -> pd.DataFrame:
    folder = results_dir(model, quick)
    table = seed_table(pd.read_csv(folder / "accuracy_summary.csv"), split)
    table.to_csv(folder / f"seeds_{split}.csv", index=False)
    draw(table, model, quick, split)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(4).to_string(index=False))
