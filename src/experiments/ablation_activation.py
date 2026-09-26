"""Stretch: does the smooth activation matter? ELU against ReLU, in accuracy and in calibration.

The paper chooses ELU because calibration uses the network's gradient (its Remark 1 and Remark 6), but shows no
experiment. We train the same network with ReLU (same data, seed and recipe) and calibrate the same validation
surfaces with Levenberg-Marquardt on both networks, each with its own exact Jacobian.

Usage:  python -m src.experiments.ablation_activation --model rbergomi [--quick]
"""
import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.config import CALIBRATION, DATASETS, N_PARAMS, QUICK, RECIPE, results_dir
from src.data import load_train_val, unscale_params, unscale_vols
from src.model import load_weights, numpy_weights
from src.plots import SERIES, apply_style, save
from src.train import fit

N_SURFACES = 1_000

ACTIVATIONS = {
    "elu": (lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0.0))), lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0.0)))),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float)),
}


def forward(weights, x, act):
    a = x
    for W, b in weights[:-1]:
        a = act(a @ W.T + b)
    return a @ weights[-1][0].T + weights[-1][1]


def jacobian(weights, x, act, act_prime):
    a, J = x, np.eye(len(x))
    for W, b in weights[:-1]:
        z = W @ a + b
        J = act_prime(z)[:, None] * (W @ J)
        a = act(z)
    return weights[-1][0] @ J


def calibrate(weights, activation: str, data: dict, n: int, names: list[str]) -> pd.DataFrame:
    """Levenberg-Marquardt from the centre of the box on the first n validation surfaces; one row per surface."""
    act, act_prime = ACTIVATIONS[activation]
    rows = []
    for k in range(n):
        target = data["y_val"][k]
        start = time.perf_counter()
        result = least_squares(lambda x: forward(weights, x, act) - target, np.zeros(N_PARAMS),
                               jac=lambda x: jacobian(weights, x, act, act_prime), method="lm", gtol=CALIBRATION["gtol"])
        seconds = time.perf_counter() - start
        fitted = unscale_params(result.x, data["scalers"])
        gap = unscale_vols(forward(weights, result.x, act), data["scalers"]) - data["vols_val"][k]
        relerr = np.abs(fitted - data["theta_val"][k]) / np.abs(data["theta_val"][k])
        rows.append({"activation": activation, "row": k, "seconds": seconds, "n_evaluations": result.nfev, "rmse": np.sqrt(np.mean(gap ** 2)),
                     "in_box": bool(np.all(np.abs(result.x) <= 1 + 1e-9)), **{f"relerr_{p}": v for p, v in zip(names, relerr)}})
    return pd.DataFrame(rows)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    data, folder, names = load_train_val(model, quick), results_dir(model, quick), DATASETS[model]["param_names"]
    n = QUICK["n_calibrations"] if quick else N_SURFACES
    max_epochs = QUICK["max_epochs"] if quick else RECIPE["max_epochs"]
    relu_net, _, relu_summary = fit(data["x_train"], data["y_train"], data["x_val"], data["y_val"], 0, max_epochs, activation="relu", verbose=False)
    networks = {"elu": load_weights(folder / "weights_seed0.npz"), "relu": numpy_weights(relu_net)}

    summary, details = [], []
    for activation, weights in networks.items():
        act = ACTIVATIONS[activation][0]
        predicted = unscale_vols(forward(weights, data["x_val"], act), data["scalers"])
        errors = np.abs(predicted - data["vols_val"]) / data["vols_val"]
        fits = calibrate(weights, activation, data, n, names)
        details.append(fits)
        summary.append({"activation": activation, "val_avg_rel_error_pct": 100 * errors.mean(), "val_max_rel_error_pct": 100 * errors.max(),
                        "n_surfaces": n, "lm_mean_ms": 1e3 * fits["seconds"].mean(), "lm_mean_evaluations": fits["n_evaluations"].mean(),
                        "rmse_median_pct": 100 * fits["rmse"].median(), "rmse_q99_pct": 100 * fits["rmse"].quantile(0.99),
                        "share_in_box": fits["in_box"].mean(),
                        "mean_param_error_pct": 100 * fits[[f"relerr_{p}" for p in names]].mean().mean(),
                        "median_param_error_pct": 100 * fits[[f"relerr_{p}" for p in names]].median().mean(),
                        "epochs_run": relu_summary["epochs_run"] if activation == "relu" else np.nan})
    table = pd.DataFrame(summary)
    table.to_csv(folder / "ablation_activation_val.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    q = np.linspace(0, 1, 400)
    for k, fits in enumerate(details):
        ax.plot(100 * q, np.quantile(100 * fits["rmse"], q), color=SERIES[k], label=fits["activation"].iloc[0].upper())
    ax.set_yscale("log")
    ax.set_xlabel("Quantile of validation surfaces")
    ax.set_ylabel("Surface RMSE after Levenberg-Marquardt (log scale)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_title(f"{DATASETS[model]['title']}: calibration with an ELU and a ReLU network, {n:,} validation surfaces", loc="left")
    ax.legend(loc="upper left")
    save(fig, model, "ablation_activation_val", quick)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(4).T.to_string())
