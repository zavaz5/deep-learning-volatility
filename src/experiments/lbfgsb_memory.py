"""Stretch: why does L-BFGS-B need more steps for rough Bergomi than for 1-factor Bergomi, when the paper's Figure 8 shows the opposite?

A calibration costs (number of network evaluations) x (cost of one evaluation), and one evaluation costs the same for both models,
because both networks have the same size. So the question is the number of evaluations. On validation surfaces we count them for
L-BFGS-B as src/calibrate.py runs it (bounds, SciPy's default memory of 10 correction pairs, fewer than the 11 parameters), without
the bounds, with a memory of 20, and for BFGS, which keeps the full curvature matrix. We also measure how well conditioned each
calibration problem is: the condition number of the network's Jacobian at the Levenberg-Marquardt fit. Counting evaluations needs
no timing, so the result does not depend on the machine. Validation rows only.

Usage:  python -m src.experiments.lbfgsb_memory [--quick]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.calibrate import BOUNDS, GRADIENT_BASED, Problem, jacobian
from src.config import CALIBRATION, DATASETS, N_PARAMS, QUICK, RESULTS_DIR, results_dir
from src.data import load_train_val
from src.model import load_weights
from src.plots import INK_SECONDARY, SERIES, SURFACE, apply_style, shared_figures_dir, write

N_SURFACES = 500
SETTINGS = {
    "L-BFGS-B as run: bounds, memory 10": ("L-BFGS-B", True, 10),
    "L-BFGS-B without bounds, memory 10": ("L-BFGS-B", False, 10),
    "L-BFGS-B with bounds, memory 20": ("L-BFGS-B", True, 20),
    "BFGS: full curvature matrix": ("BFGS", False, None),
}


def evaluations(problem: Problem, method: str, bounded: bool, memory: int | None) -> int:
    """Network evaluations one calibration needs, with the notebook's tolerance and iteration cap, from the centre of the box."""
    options = {"maxiter": CALIBRATION["maxiter"]} | ({"maxcor": memory} if memory else {})
    result = minimize(problem.cost, np.zeros(N_PARAMS), jac=problem.gradient, method=method,
                      bounds=BOUNDS if bounded else None, tol=CALIBRATION["tol"], options=options)
    return result.nfev


def condition_number(weights, problem: Problem) -> float:
    """Largest over smallest singular value of the 88 x 11 Jacobian at the fitted parameters: a large value means some parameter
    directions barely move the surface, which is what slows a quasi-Newton method with a short memory."""
    fitted = GRADIENT_BASED["Levenberg-Marquardt"](problem).x
    singular = np.linalg.svd(jacobian(weights, fitted), compute_uv=False)
    return float(singular[0] / singular[-1])


def draw(table: pd.DataFrame, quick: bool) -> None:
    """Mean evaluations per setting, one bar per model: the gap between the models closes as soon as the memory is long enough."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    settings = list(SETTINGS)
    y = np.arange(len(settings))
    for k, model in enumerate(DATASETS):
        part = table[table.model == model].set_index("setting").loc[settings]
        bars = ax.barh(y + (k - 0.5) * 0.38, part["mean_evaluations"], height=0.36, color=SERIES[k], edgecolor=SURFACE,
                       label=f"{DATASETS[model]['title']} (Jacobian condition number, median: {part['jacobian_condition_median'].iloc[0]:.0f})")
        for bar, value in zip(bars, part["mean_evaluations"]):
            ax.annotate(f"{value:.0f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK_SECONDARY, fontsize=10)
    ax.set_yticks(y, settings)
    ax.invert_yaxis()
    ax.set_xlabel("Mean network evaluations per calibration")
    ax.set_title(f"Network evaluations per calibration on {table['n_surfaces'].iloc[0]:,} validation surfaces", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    write(fig, shared_figures_dir(quick) / "lbfgsb_memory_val.png")


def main(quick: bool = False) -> pd.DataFrame:
    n = QUICK["n_calibrations"] if quick else N_SURFACES
    rows = []
    for model in DATASETS:
        data = load_train_val(model, quick)
        weights = load_weights(results_dir(model, quick) / "weights_seed0.npz")
        problems = [Problem(weights, target) for target in data["y_val"][:n]]
        conditions = [condition_number(weights, problem) for problem in problems]
        for setting, spec in SETTINGS.items():
            counts = [evaluations(problem, *spec) for problem in problems]
            rows.append({"model": model, "setting": setting, "n_surfaces": n, "mean_evaluations": np.mean(counts),
                         "median_evaluations": np.median(counts), "jacobian_condition_median": np.median(conditions),
                         "jacobian_condition_q90": np.quantile(conditions, 0.9)})
    table = pd.DataFrame(rows)
    folder = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    table.to_csv(folder / "lbfgsb_memory_val.csv", index=False)
    print(f"wrote {folder / 'lbfgsb_memory_val.csv'}")
    draw(table, quick)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.quick).round(2).to_string(index=False))
