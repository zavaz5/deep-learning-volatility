"""Every figure in the README and the notebook, drawn from files in results/ (and, for the data
plot, from training rows). One function per figure; the command line picks one by name.

Usage:  python -m src.plots --model rbergomi --figure surfaces [--quick]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.config import DATASETS, MATURITIES, STRIKES, figures_dir, figures_root, results_dir
from src.data import load_train_val
from src.paper_numbers import PAPER, notebook

SURFACE = "#fcfcfb"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP_8 = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281", "#0d366b"]


def apply_style() -> None:
    """Set matplotlib defaults once per figure so every chart shares the same anatomy."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "regular", "axes.labelsize": 11,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 10,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, model: str, name: str, quick: bool) -> None:
    """Write the figure as PNG into results/figures/<model>/ (or results/quick/figures/<model>/) and close it."""
    path = figures_dir(model, quick) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_surfaces(model: str, quick: bool = False) -> None:
    """Five random training surfaces and the mean training surface, as smiles per maturity.

    This is the first sanity check of the data: the file is read with the right
    orientation if short maturities (light blue) show the steepest smiles. The five rows are
    drawn with a fixed seed so the figure is reproducible. Training rows only.
    """
    apply_style()
    data = load_train_val(model, quick)
    theta, vols = data["theta_train"], data["vols_train"]
    names = DATASETS[model]["param_labels"]
    rows = np.random.default_rng(0).choice(len(vols), size=5, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    panels = [(vols[r].reshape(8, 11), theta[r]) for r in rows] + [(vols.mean(0).reshape(8, 11), None)]
    for ax, (surface, params) in zip(axes.ravel(), panels):
        for k, maturity in enumerate(MATURITIES):
            ax.plot(STRIKES, surface[k], color=BLUE_RAMP_8[k], label=f"{maturity:g}")
        if params is None:
            ax.set_title(f"Mean of {len(vols):,} training surfaces")
        else:
            ax.set_title(", ".join(f"{n} = {v:.2f}" for n, v in zip(names[8:], params[8:])))
    for ax in axes[-1]:
        ax.set_xlabel("Strike")
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied volatility")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Maturity (years)", loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{DATASETS[model]['title']}: five random training surfaces and the mean surface", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    save(fig, model, "surfaces", quick)


def plot_learning_curves(model: str, quick: bool = False) -> None:
    """Training and validation loss per epoch for seed 0, validation loss of the other seeds behind it.

    Both curves use the Keras convention (mean of per-batch RMSE in scaled units), so the authors'
    final losses from their training log can sit on the same axes as two reference points.
    Evidence on overfitting: a network that memorised noise would show the validation
    curve turning up while the training curve keeps falling.
    """
    apply_style()
    folder = results_dir(model, quick)
    seeds = sorted(int(p.stem.split("seed")[1]) for p in folder.glob("history_seed*.csv"))
    histories = {s: pd.read_csv(folder / f"history_seed{s}.csv") for s in seeds}
    with open(folder / f"train_summary_seed{seeds[0]}.json") as f:
        summary = json.load(f)
    main = histories[seeds[0]]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for s in seeds[1:]:
        ax.plot(histories[s]["epoch"], histories[s]["val_loss_keras"], color=AXIS, linewidth=1.2, zorder=1)
    if len(seeds) > 1:
        ax.plot([], [], color=AXIS, linewidth=1.2, label=f"Validation, seeds {seeds[1]}–{seeds[-1]}")
    ax.plot(main["epoch"], main["train_loss_running"], color=SERIES[0], label=f"Training, seed {seeds[0]}", zorder=3)
    ax.plot(main["epoch"], main["val_loss_keras"], color=SERIES[1], label=f"Validation, seed {seeds[0]}", zorder=3)
    best = summary["best_epoch"]
    ax.plot([best], [summary["best_val_loss_keras"]], "o", color=SERIES[1], markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.annotate(f"best epoch {best}: weights kept", (best, summary["best_val_loss_keras"]),
                xytext=(0, -20), textcoords="offset points", ha="center", color=INK_SECONDARY, fontsize=10)

    stop = notebook(model, "stopped_epoch")
    ax.plot([stop], [notebook(model, "final_train_loss")], "D", color=INK, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, linestyle="none", zorder=5,
            label=f"Authors' training loss at their last epoch, {stop}: {notebook(model, 'final_train_loss'):.4f}")
    ax.plot([stop], [notebook(model, "final_val_loss")], "D", color=SURFACE, markersize=7, markeredgecolor=INK,
            markeredgewidth=1.5, linestyle="none", zorder=5,
            label=f"Authors' test-set loss at epoch {stop}: {notebook(model, 'final_val_loss'):.4f}")

    ax.set_yscale("log")
    lowest = min(main["val_loss_keras"].min(), notebook(model, "final_val_loss"))
    ax.set_ylim(0.82 * lowest, 1.1 * main["train_loss_running"].max())
    ticks = [t for t in (0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
             if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, max(main["epoch"].max(), stop) * 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE in scaled units (log scale)")
    ax.set_title(f"{DATASETS[model]['title']}: learning curves, {summary['n_train']:,} training rows", loc="left")
    ax.legend(loc="upper right")
    save(fig, model, "learning_curves", quick)


PAPER_STYLE = ["default", {"savefig.dpi": 200, "savefig.bbox": "tight"}]
SPLIT_NAMES = {"train": "training", "val": "validation", "test": "test", "dryrun": "dry-run (validation stand-in)"}


def add_footnote(fig, text: str) -> None:
    """One small grey line under the figure; constrained layout keeps it clear of the axes."""
    fig.supxlabel(text, x=0.01, ha="left", fontsize=8.5, color="0.35")


def write(fig, path: Path) -> Path:
    """Save the figure at the given path and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def shared_figures_dir(quick: bool = False) -> Path:
    """Folder of the one figure that covers both models (the paper's Figure 8): results/figures/ or results/quick/figures/."""
    path = figures_root(quick)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_error_heatmaps(model: str, quick: bool = False, split: str = "val", seed: int = 0, out: Path | None = None,
                        footnote: bool = True) -> Path:
    """Mean, standard deviation and maximum of the relative error on the 8 x 11 grid, drawn like the paper's Figure 6.

    Same three panels, viridis, square cells, one colour bar per panel fixed to the paper's limits, matplotlib's
    default style, so that the two figures can be compared colour by colour. Where our values leave the paper's
    scale the cell shows the end colour and the footnote counts it; nothing is rescaled silently. `out` and
    `footnote` draw the same figure elsewhere and without the footnote, for use at a smaller size.
    """
    grid = pd.read_csv(results_dir(model, quick) / f"accuracy_{split}_seed{seed}.csv")
    summary_file = "accuracy_summary.csv" if split in ("train", "val") else f"accuracy_summary_{split}.csv"
    summary = pd.read_csv(results_dir(model, quick) / summary_file)
    n_rows = int(summary[(summary.seed == seed) & (summary.split == split)]["n_rows"].iloc[0])
    limits, source = PAPER[DATASETS[model]["paper_heatmap"]]
    panels = [("mean_pct", "Average relative error", "mean"), ("std_pct", "Std relative error", "std"),
              ("max_pct", "Maximum relative error", "max")]

    with plt.style.context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        beyond = []
        for ax, (column, title, key) in zip(axes, panels):
            values = grid[column].to_numpy().reshape(len(MATURITIES), len(STRIKES))
            low, high = limits[key]
            image = ax.imshow(values, cmap="viridis", vmin=low, vmax=high)
            fig.colorbar(image, ax=ax, format=matplotlib.ticker.PercentFormatter())
            ax.set_title(title)
            ax.set_xticks(range(len(STRIKES)), [f"{k:.1f}" for k in STRIKES])
            ax.set_yticks(range(len(MATURITIES)), [f"{t:.1f}" for t in MATURITIES])
            ax.set_xlabel("Strike")
            ax.set_ylabel("Maturity")
            above, below = int((values > high).sum()), int((values < low).sum())
            if above:
                beyond.append(f"{key}: {above} above ({values.max():.2f}% against {high:.2f}%)")
            if below:
                beyond.append(f"{key}: {below} below ({values.min():.2f}% against {low:.2f}%)")
        if footnote:
            text = (f"Ours: {DATASETS[model]['title']}, {n_rows:,} {SPLIT_NAMES[split]} rows, seed {seed}; colour limits of the paper's "
                    f"{source.split(',')[0]}. Cells beyond the paper's scale: {'; '.join(beyond) if beyond else 'none'}.")
            add_footnote(fig, text)
        return write(fig, out or figures_dir(model, quick) / f"error_heatmaps_{split}.png")


OPTIMIZER_ORDER = ["L-BFGS-B", "SLSQP", "BFGS", "Levenberg-Marquardt", "COBYLA", "Differential Evolution",
                   "Nelder-Mead", "least_squares default (TRF)"]
PAPER_OPTIMIZERS = OPTIMIZER_ORDER[:7]
FIG8_LEGEND = {"rbergomi": "rBergomi piecewise forward variance", "onefactor": "1FBergomi piecewise forward variance"}


def plot_calibration_times(model: str, quick: bool = False, split: str = "val", out: Path | None = None,
                           footnote: bool = True) -> Path:
    """Mean time per calibration of the paper's seven optimizers, both models side by side, drawn like the paper's Figure 8.

    The paper's chart holds both models in one picture, so ours does too: it is written once, to
    results/figures/calibration_times_<split>.png, whichever model the command names, from every model whose calibration
    summary exists (in the full run the second model's call completes it). Grouped bars in matplotlib's default blue
    and orange, log axis, the paper's title and legend text. The hardware differs from the authors', so the comparison
    is about the ranking and the orders of magnitude. The authors' actual least-squares call (TRF) has no bar in the
    paper's chart, so it has none here; its time stays in the summary table, the README and the footnote.
    """
    summaries = {}
    for name in DATASETS:
        file = results_dir(name, quick) / f"calibration_summary_{split}.csv"
        if file.exists():
            summaries[name] = pd.read_csv(file).set_index("optimizer")
    if model not in summaries:
        raise FileNotFoundError(f"no calibration summary for {model}, split {split}")
    positions, width = np.arange(len(PAPER_OPTIMIZERS)), 0.8 / len(DATASETS)

    with plt.style.context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
        for k, name in enumerate(DATASETS):
            if name in summaries:
                offset = (k - (len(DATASETS) - 1) / 2) * width
                ax.bar(positions + offset, summaries[name].loc[PAPER_OPTIMIZERS, "mean_ms"], width=width, label=FIG8_LEGEND[name])
        ax.set_yscale("log")
        ax.set_xticks(positions, PAPER_OPTIMIZERS)
        ax.set_ylabel("Milliseconds")
        ax.set_title("Average calibration time with different optimisers", fontsize=16)
        ax.grid(True)
        ax.legend(loc="upper left")
        if footnote:
            first = summaries[model]
            n_based, n_free = int(first.loc["L-BFGS-B", "n_surfaces"]), int(first.loc["Nelder-Mead", "n_surfaces"])
            trf = ", ".join(f"{s.loc['least_squares default (TRF)', 'mean_ms']:.3g} ms ({DATASETS[name]['title']})" for name, s in summaries.items())
            text = (f"Ours: mean over the first {n_based:,} {SPLIT_NAMES[split]} surfaces for the gradient-based optimizers and {n_free:,} "
                    f"for the gradient-free ones. The paper's Levenberg-Marquardt bar came from SciPy's least_squares default (TRF), "
                    f"which takes {trf} here.")
            add_footnote(fig, text)
        return write(fig, out or shared_figures_dir(quick) / f"calibration_times_{split}.png")


def plot_calibration_errors(model: str, quick: bool = False, split: str = "val", optimizer: str = "Levenberg-Marquardt",
                            out: Path | None = None, footnote: bool = True) -> Path:
    """Quantile curves of relative parameter error and of surface RMSE, drawn like the paper's Figures 9 and 10.

    Two panels with the paper's titles and axis labels; matplotlib's default colour cycle, so that each parameter has
    the colour it has in the paper; the 95% and 99% quantiles as dashed lines; quantiles up to 99% on the left, as in
    the authors' notebook. The vertical axes start from the paper's ranges and grow only if our values need more.
    RMSE is between the network surface at the fitted parameters and the target, as the paper's formula says.
    """
    results = pd.read_csv(results_dir(model, quick) / f"calibration_{split}.csv.gz")
    results = results[results["optimizer"] == optimizer]
    names, labels = DATASETS[model]["param_names"], DATASETS[model]["param_labels"]
    tops, source = PAPER[DATASETS[model]["paper_error_cdf"]]

    with plt.style.context(PAPER_STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(18, 5), layout="constrained")
        q = np.linspace(0, 0.99, 200)
        highest = 0.0
        for name, label in zip(names, labels):
            curve = np.quantile(100 * results[f"relerr_{name}"], q)
            highest = max(highest, curve[-1])
            left.plot(100 * q, curve, label=label)
        left.axvline(95, color="black", linestyle="--", label="95% quantile")
        left.set_xlim(0, 100)
        left.set_ylim(0, max(tops["param_error"], 1.05 * highest))
        left.set_title("Empirical CDF of parameter relative error")
        left.set_ylabel("relative error")
        left.legend(loc="upper left")

        q_all = np.linspace(0, 1, 400)
        rmse = np.quantile(100 * results["rmse"], q_all)
        right.plot(100 * q_all, rmse, linewidth=3, label="RMSE")
        right.axvline(99, color="black", linestyle="--", label="99% quantile")
        right.set_ylim(0, max(tops["rmse"], 1.05 * rmse[-1]))
        right.set_title("Empirical CDF of implied vol surface RMSE")
        right.set_ylabel("RMSE")
        right.legend(loc="upper left")
        for ax in (left, right):
            ax.set_xlabel("quantiles")
            ax.set_xticks(range(0, 101, 10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
            ax.grid(True)
        if footnote:
            text = (f"Ours: {DATASETS[model]['title']}, {optimizer} on the first {len(results):,} {SPLIT_NAMES[split]} surfaces; RMSE between "
                    f"the network surface at the fitted parameters and the target; vertical axes at least as tall as the paper's {source.split(',')[0]}.")
            add_footnote(fig, text)
        return write(fig, out or figures_dir(model, quick) / f"calibration_errors_{split}.png")


FIGURES = {
    "surfaces": plot_surfaces,
    "learning_curves": plot_learning_curves,
    "calibration_times_val": plot_calibration_times,
    "calibration_errors_val": plot_calibration_errors,
    "error_heatmaps_val": lambda model, quick: plot_error_heatmaps(model, quick, split="val"),
    "error_heatmaps_train": lambda model, quick: plot_error_heatmaps(model, quick, split="train"),
}
TEST_FIGURES = {
    "error_heatmaps_test": lambda model, quick: plot_error_heatmaps(model, quick, split="test"),
    "calibration_times_test": lambda model, quick: plot_calibration_times(model, quick, split="test"),
    "calibration_errors_test": lambda model, quick: plot_calibration_errors(model, quick, split="test"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--figure", choices=list(FIGURES) + list(TEST_FIGURES) + ["all"], default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.figure in TEST_FIGURES:
        TEST_FIGURES[args.figure](args.model, args.quick)
    for name, draw in FIGURES.items():
        if args.figure in (name, "all"):
            draw(args.model, args.quick)
