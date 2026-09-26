"""Critique evidence: the speed-up in the paper's Table 2 leaves out the offline cost.

The network is fast because 80,000 Monte Carlo surfaces were computed beforehand. This script does
the arithmetic. Monte Carlo times are the PAPER'S numbers (Table 2); we have no simulator and did
not measure them. Evaluations per calibration and network times are OURS, from results/.

Assumption for the break-even: a brute-force Monte Carlo calibration with Levenberg-Marquardt
needs the same number of iterations as on the network, and each iteration prices 1 + 11 surfaces,
because without an exact Jacobian the 11 partial derivatives come from finite differences
(the paper says it uses finite differences for its Monte Carlo calibration, Section 4.2.2, p. 24).

Usage:  python -m src.experiments.offline_cost --model rbergomi [--quick]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATASETS, N_PARAMS, results_dir
from src.paper_numbers import PAPER
from src.plots import INK_SECONDARY, SERIES, SURFACE, apply_style, save


def compute(model: str, quick: bool) -> pd.DataFrame:
    folder = results_dir(model, quick)
    calibration = pd.read_csv(folder / "calibration_summary_val.csv").set_index("optimizer").loc["Levenberg-Marquardt"]
    with open(folder / "train_summary_seed0.json") as f:
        train_seconds = json.load(f)["seconds"]
    n_samples, n_source = PAPER["n_samples"]
    mc_us, mc_source = PAPER[f"mc_{model}_us"]
    mc_seconds = mc_us / 1e6

    offline_seconds = n_samples * mc_seconds
    surfaces_per_mc_calibration = calibration["mean_evaluations"] * (1 + N_PARAMS)
    mc_calibration_seconds = surfaces_per_mc_calibration * mc_seconds
    nn_calibration_seconds = calibration["mean_ms"] / 1e3
    break_even = (offline_seconds + train_seconds) / (mc_calibration_seconds - nn_calibration_seconds)
    rows = [
        ("training_samples", n_samples, "surfaces", f"paper, {n_source}"),
        ("mc_seconds_per_surface", mc_seconds, "s", f"paper, {mc_source}"),
        ("offline_mc_cpu_hours", offline_seconds / 3600, "h", "paper's numbers multiplied: samples x seconds per surface"),
        ("network_training_minutes", train_seconds / 60, "min", "ours, train_summary_seed0.json (one CPU thread)"),
        ("lm_evaluations_per_calibration", calibration["mean_evaluations"], "evaluations", "ours, calibration_summary_val.csv"),
        ("mc_surfaces_per_calibration", surfaces_per_mc_calibration, "surfaces", "assumption: 1 + 11 surfaces per evaluation (finite differences)"),
        ("mc_seconds_per_calibration", mc_calibration_seconds, "s", "assumption x paper's seconds per surface"),
        ("network_ms_per_calibration", calibration["mean_ms"], "ms", "ours, calibration_summary_val.csv"),
        ("break_even_calibrations", break_even, "calibrations", "offline cost / time saved per calibration"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit", "source"])


def draw(table: pd.DataFrame, model: str, quick: bool) -> None:
    """Cumulative computing time against the number of calibrations, with and without the network."""
    apply_style()
    v = table.set_index("quantity")["value"]
    offline_hours = v["offline_mc_cpu_hours"] + v["network_training_minutes"] / 60
    n = np.logspace(0, 6, 200)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(n, n * v["mc_seconds_per_calibration"] / 3600, color=SERIES[1], label="Brute-force Monte Carlo calibration")
    ax.plot(n, offline_hours + n * v["network_ms_per_calibration"] / 3.6e6, color=SERIES[0],
            label="Network: offline data and training, then calibration")
    ax.plot([v["break_even_calibrations"]], [offline_hours], "o", color=SERIES[0], markersize=9, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none")
    ax.annotate(f"break-even after about {v['break_even_calibrations']:,.0f} calibrations\n(offline cost {offline_hours:.1f} CPU-hours)",
                (v["break_even_calibrations"], offline_hours), xytext=(12, -34), textcoords="offset points",
                color=INK_SECONDARY, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:,.10g}"))
    ax.minorticks_off()
    ax.set_xlabel("Number of calibrations (log scale)")
    ax.set_ylabel("Cumulative CPU-hours (log scale)")
    ax.set_title(f"{DATASETS[model]['title']}: the offline cost pays off only after many calibrations\n"
                 "Monte Carlo times are the paper's (Table 2); evaluations per calibration are ours", loc="left", fontsize=11)
    ax.legend(loc="upper left")
    save(fig, model, "offline_cost", quick)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    table = compute(model, quick)
    table.to_csv(results_dir(model, quick) / "offline_cost.csv", index=False)
    draw(table, model, quick)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(3).to_string(index=False))
