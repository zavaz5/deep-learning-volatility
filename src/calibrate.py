"""The network in plain NumPy, its exact Jacobian, and the calibration step.

Calibration evaluates the network thousands of times for single parameter vectors. For one
11-vector a few NumPy matrix products are faster than a deep-learning framework's call
overhead, which is also how the authors obtained their 0.0309 milliseconds. PyTorch is used
only to train the weights and to check the Jacobian below against autograd (src/tests/).

The optimizers work in scaled space: inputs in [-1, 1], outputs in standardized vol units.
The objective is the squared gap between the network's surface and the target surface, started
from the centre of the box (the zero vector), as in the authors' notebook.

Usage:  python -m src.calibrate --model rbergomi [--quick] [--n-gradient-based 200 --n-gradient-free 10]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize

from src.config import CALIBRATION, DATASETS, N_PARAMS, QUICK, results_dir
from src.data import load_train_val, unscale_params, unscale_vols
from src.model import load_weights

Weights = list[tuple[np.ndarray, np.ndarray]]


def elu(z: np.ndarray) -> np.ndarray:
    """ELU with alpha = 1: z for z > 0, exp(z) - 1 otherwise. Smooth, so its gradient is usable."""
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))


def elu_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of ELU: 1 for z > 0, exp(z) otherwise."""
    return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0.0)))


def forward(weights: Weights, x: np.ndarray) -> np.ndarray:
    """Network output for one scaled parameter vector (11,) or a batch (n, 11)."""
    a = x
    for W, b in weights[:-1]:
        a = elu(a @ W.T + b)
    W, b = weights[-1]
    return a @ W.T + b


def jacobian(weights: Weights, x: np.ndarray) -> np.ndarray:
    """Exact 88 x 11 matrix of d output_i / d x_j at one scaled parameter vector, by the chain rule.

    Each hidden layer maps a -> elu(W a + b), whose derivative is diag(elu'(z)) W. Multiplying
    these layer by layer, starting from the identity, gives the derivative of the whole network.
    """
    a, J = x, np.eye(len(x))
    for W, b in weights[:-1]:
        z = W @ a + b
        J = elu_prime(z)[:, None] * (W @ J)
        a = elu(z)
    return weights[-1][0] @ J


BOUNDS = [(-1.0, 1.0)] * N_PARAMS


class Problem:
    """One calibration: find scaled parameters x whose network surface matches `target` (scaled vols)."""

    def __init__(self, weights: Weights, target: np.ndarray):
        self.weights, self.target = weights, target

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """The 88 gaps between network and target; least-squares solvers want these, not their sum."""
        return forward(self.weights, x) - self.target

    def residual_jacobian(self, x: np.ndarray) -> np.ndarray:
        """The target is constant, so the Jacobian of the residuals is the network's Jacobian."""
        return jacobian(self.weights, x)

    def cost(self, x: np.ndarray) -> float:
        r = self.residuals(x)
        return float(r @ r)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """d/dx of sum(r^2) = 2 J^T r, exact because J is exact."""
        return 2.0 * self.residual_jacobian(x).T @ self.residuals(x)


def _minimize(method: str, bounded: bool, use_gradient: bool):
    """A scipy.optimize.minimize call with the notebook's tolerance and iteration cap."""
    def solve(problem: Problem):
        return minimize(problem.cost, np.zeros(N_PARAMS), jac=problem.gradient if use_gradient else None,
                        method=method, bounds=BOUNDS if bounded else None, tol=CALIBRATION["tol"],
                        options={"maxiter": CALIBRATION["maxiter"]})
    return solve


def _levenberg_marquardt(problem: Problem):
    """MINPACK's Levenberg-Marquardt, the optimizer the paper names. It cannot take bounds."""
    return least_squares(problem.residuals, np.zeros(N_PARAMS), jac=problem.residual_jacobian, method="lm",
                         gtol=CALIBRATION["gtol"])


def _least_squares_default(problem: Problem):
    """The authors' exact call: no `method`, so SciPy runs trust-region reflective ('trf'), without bounds."""
    return least_squares(problem.residuals, np.zeros(N_PARAMS), jac=problem.residual_jacobian, gtol=CALIBRATION["gtol"])


def _differential_evolution(problem: Problem):
    """Global, gradient-free, SciPy defaults; seeded because the method is random."""
    return differential_evolution(problem.cost, BOUNDS, seed=CALIBRATION["de_seed"])


GRADIENT_BASED = {
    "L-BFGS-B": _minimize("L-BFGS-B", bounded=True, use_gradient=True),
    "SLSQP": _minimize("SLSQP", bounded=True, use_gradient=True),
    "BFGS": _minimize("BFGS", bounded=False, use_gradient=True),
    "Levenberg-Marquardt": _levenberg_marquardt,
    "least_squares default (TRF)": _least_squares_default,
}
GRADIENT_FREE = {
    "COBYLA": _minimize("COBYLA", bounded=True, use_gradient=False),
    "Differential Evolution": _differential_evolution,
    "Nelder-Mead": _minimize("Nelder-Mead", bounded=True, use_gradient=False),
}


def calibrate_surfaces(weights: Weights, optimizers: dict, theta: np.ndarray, vols: np.ndarray, y_scaled: np.ndarray,
                       scalers: dict, param_names: list[str], verbose: bool = True) -> pd.DataFrame:
    """Calibrate every optimizer to every given surface; one output row per (surface, optimizer).

    Only the optimizer call is timed, with time.perf_counter, one calibration after another in a
    single process. RMSE follows the paper: network surface at the fitted parameters against the
    target surface, in implied-vol units. `rmse` is the root of the mean square (what the name
    says and what the authors' notebook computes); `rmse_root_sum` is the formula as printed in
    the paper, larger by sqrt(88). Relative parameter error is |fitted - true| / |true|.
    """
    records = []
    for row in range(len(theta)):
        problem = Problem(weights, y_scaled[row])
        for name, solve in optimizers.items():
            start = time.perf_counter()
            result = solve(problem)
            seconds = time.perf_counter() - start
            fitted = unscale_params(result.x, scalers)
            gap = unscale_vols(forward(weights, result.x), scalers) - vols[row]
            record = {"row": row, "optimizer": name, "seconds": seconds, "n_evaluations": result.nfev,
                      "success": bool(result.success), "in_box": bool(np.all(np.abs(result.x) <= 1.0 + 1e-9)),
                      "cost_scaled": problem.cost(result.x), "rmse": np.sqrt(np.mean(gap ** 2)),
                      "rmse_root_sum": np.sqrt(np.sum(gap ** 2))}
            record.update({f"fit_{p}": v for p, v in zip(param_names, fitted)})
            record.update({f"relerr_{p}": v for p, v in zip(param_names, np.abs(fitted - theta[row]) / np.abs(theta[row]))})
            records.append(record)
        if verbose and (row + 1) % max(1, len(theta) // 10) == 0:
            print(f"  {row + 1}/{len(theta)} surfaces", flush=True)
    return pd.DataFrame(records)


def summarise(results: pd.DataFrame, param_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-optimizer timing and fit quality, and per-optimizer, per-parameter relative errors (percent)."""
    rows, param_rows = [], []
    for name, group in results.groupby("optimizer", sort=False):
        ms, rmse_pct = 1e3 * group["seconds"], 100 * group["rmse"]
        rows.append({"optimizer": name, "n_surfaces": len(group), "mean_ms": ms.mean(), "median_ms": ms.median(),
                     "p05_ms": ms.quantile(0.05), "p95_ms": ms.quantile(0.95),
                     "mean_evaluations": group["n_evaluations"].mean(), "share_success": group["success"].mean(),
                     "share_in_box": group["in_box"].mean(), "rmse_median_pct": rmse_pct.median(),
                     "rmse_q95_pct": rmse_pct.quantile(0.95), "rmse_q99_pct": rmse_pct.quantile(0.99),
                     "rmse_max_pct": rmse_pct.max()})
        for p in param_names:
            err = 100 * group[f"relerr_{p}"]
            param_rows.append({"optimizer": name, "parameter": p, "mean_pct": err.mean(), "median_pct": err.median(),
                               "q95_pct": err.quantile(0.95), "max_pct": err.max()})
    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def run(model: str, quick: bool = False, n_gradient_based: int | None = None, n_gradient_free: int | None = None,
        seed: int = 0) -> pd.DataFrame:
    """Calibrate to the first validation surfaces and write the three calibration files."""
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    weights, names = load_weights(folder / f"weights_seed{seed}.npz"), DATASETS[model]["param_names"]
    n_based = n_gradient_based or (QUICK["n_calibrations"] if quick else CALIBRATION["n_gradient_based"])
    n_free = n_gradient_free or (QUICK["n_calibrations_gradient_free"] if quick else CALIBRATION["n_gradient_free"])
    parts = []
    for optimizers, n in ((GRADIENT_BASED, n_based), (GRADIENT_FREE, n_free)):
        print(f"{', '.join(optimizers)}: {n} validation surfaces", flush=True)
        parts.append(calibrate_surfaces(weights, optimizers, data["theta_val"][:n], data["vols_val"][:n],
                                        data["y_val"][:n], data["scalers"], names))
    results = pd.concat(parts, ignore_index=True)
    results.to_csv(folder / "calibration_val.csv.gz", index=False, float_format="%.8g")
    summary, param_errors = summarise(results, names)
    summary.to_csv(folder / "calibration_summary_val.csv", index=False)
    param_errors.to_csv(folder / "calibration_param_errors_val.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-gradient-based", type=int, default=None)
    parser.add_argument("--n-gradient-free", type=int, default=None)
    args = parser.parse_args()
    table = run(args.model, args.quick, args.n_gradient_based, args.n_gradient_free)
    print(table.round(3).to_string(index=False))
