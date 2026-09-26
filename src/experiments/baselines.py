"""Upgrade 3b (missing baseline): could a simpler regressor replace the network?

The paper compares the network only with Monte Carlo. We fit three standard regressors on the
same 60,000 training rows and score them on the same validation rows, in the same units:
ridge regression on polynomial features, k-nearest neighbours, and gradient boosting.
Each maps the 11 scaled parameters to the 88 scaled vols. The few hyperparameters tried are
chosen on the validation rows, so the validation scores of the baselines are slightly
flattering to them; the test-set table in src/final_eval.py is the clean comparison.

Prediction time is for one surface (one parameter vector), the unit that matters in calibration.
Run on an otherwise idle machine.

Usage:  python -m src.experiments.baselines --model rbergomi [--quick]
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.calibrate import forward
from src.config import DATASETS, results_dir
from src.data import load_train_val, unscale_vols
from src.ledger import us_to_ms
from src.model import load_weights
from src.plots import INK_SECONDARY, SERIES, SPLIT_NAMES, SURFACE, apply_style, save


def candidates(quick: bool) -> dict:
    """Method name -> list of (description, unfitted model). Small grids: the point is a fair look, not a contest."""
    boosting_rounds = 20 if quick else 300
    return {
        "Ridge on polynomial features": [(f"degree {d}, alpha {a:g}", make_pipeline(PolynomialFeatures(d), Ridge(alpha=a)))
                                         for d in ((2,) if quick else (2, 3, 4, 5)) for a in (1e-6, 1e-3)],
        "k-nearest neighbours": [(f"k = {k}, distance weights", KNeighborsRegressor(n_neighbors=k, weights="distance"))
                                 for k in (5, 10, 20)],
        "Gradient boosting": [(f"{boosting_rounds} rounds, learning rate 0.1, one model per cell",
                               MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=boosting_rounds, learning_rate=0.1,
                                                                                  random_state=0), n_jobs=8))],
    }


def score(predict, x: np.ndarray, vols: np.ndarray, scalers: dict, network_vols: np.ndarray | None = None) -> dict:
    """Relative error in implied-vol units, the same measure as for the network.

    `avg_rel_gap_to_network_pct` compares the method with the network instead of with the Monte Carlo
    labels (same denominator). If two unrelated regressors agree with each other much better than
    either agrees with the labels, most of their common error is noise in the labels. This is
    suggestive, not proof: both could share a bias. A cleaner Monte Carlo set would settle it (Tier C).
    """
    predicted = unscale_vols(predict(x), scalers)
    errors = np.abs(predicted - vols) / vols
    result = {"avg_rel_error_pct": 100 * errors.mean(), "p99_rel_error_pct": 100 * np.quantile(errors, 0.99),
              "max_rel_error_pct": 100 * errors.max()}
    if network_vols is not None:
        result["avg_rel_gap_to_network_pct"] = 100 * (np.abs(predicted - network_vols) / vols).mean()
    return result


def time_one_surface(predict, x_row: np.ndarray, n_calls: int) -> dict:
    """Microseconds to predict one surface: warm-up, then n_calls single calls; median and 5%-95% range."""
    for _ in range(max(3, n_calls // 10)):
        predict(x_row)
    samples = []
    for _ in range(n_calls):
        start = time.perf_counter()
        predict(x_row)
        samples.append((time.perf_counter() - start) * 1e6)
    return {"predict_us_median": np.median(samples), "predict_us_p05": np.quantile(samples, 0.05),
            "predict_us_p95": np.quantile(samples, 0.95)}


def fit_baselines(data: dict, quick: bool) -> dict:
    """Fit every candidate, keep the best of each method by validation error. Returns name -> (description, model, fit seconds)."""
    best = {}
    for method, options in candidates(quick).items():
        for description, model in options:
            start = time.perf_counter()
            model.fit(data["x_train"], data["y_train"])
            seconds = time.perf_counter() - start
            error = score(model.predict, data["x_val"], data["vols_val"], data["scalers"])["avg_rel_error_pct"]
            print(f"  {method}: {description}: validation error {error:.3f}%, fit {seconds:.1f} s", flush=True)
            if method not in best or error < best[method][3]:
                best[method] = (description, model, seconds, error)
    return {method: entry[:3] for method, entry in best.items()}


def table_for(models: dict, weights, x: np.ndarray, vols: np.ndarray, scalers: dict, n_train: int, quick: bool) -> pd.DataFrame:
    """Error and single-surface prediction time of the network and of each fitted baseline on the given rows."""
    x_row, network_vols = x[:1], unscale_vols(forward(weights, x), scalers)
    rows = [{"method": "Neural network (ours, seed 0)", "setting": "4 x 30 ELU, NumPy forward pass", "fit_seconds": np.nan,
             **score(lambda a: forward(weights, a), x, vols, scalers, network_vols),
             **time_one_surface(lambda a: forward(weights, a), x_row, 200 if quick else 2_000)}]
    for method, (description, model, seconds) in models.items():
        rows.append({"method": method, "setting": description, "fit_seconds": seconds,
                     **score(model.predict, x, vols, scalers, network_vols),
                     **time_one_surface(model.predict, x_row, 20 if quick else 200)})
    table = pd.DataFrame(rows)
    table.insert(2, "n_train", n_train)
    return table


def draw(table: pd.DataFrame, model: str, quick: bool, split: str) -> None:
    """Error against prediction time, both on log axes: good methods sit in the lower left."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ms = table["predict_us_median"] / 1000
    for k, row in table.reset_index(drop=True).iterrows():
        ax.plot([row["predict_us_median"] / 1000], [row["avg_rel_error_pct"]], "o", color=SERIES[k], markersize=10,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label=f"{row['method']}: {row['setting']}")
        ax.annotate(f"{row['avg_rel_error_pct']:.2f}%, {us_to_ms(row['predict_us_median'])} ms",
                    (row["predict_us_median"] / 1000, row["avg_rel_error_pct"]), xytext=(9, 6), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ms.min() / 3, ms.max() * 12)
    ax.set_ylim(table["avg_rel_error_pct"].min() / 1.8, table["avg_rel_error_pct"].max() * 1.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.10g}"))
    ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Time to predict one surface, milliseconds (log scale)")
    ax.set_ylabel("Average relative error (log scale)")
    ax.set_title(f"{DATASETS[model]['title']}: network against simple regressors on {SPLIT_NAMES[split]} rows", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=9.5)
    save(fig, model, f"baselines_{split}", quick)


def main(model: str, quick: bool = False) -> pd.DataFrame:
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    weights = load_weights(folder / "weights_seed0.npz")
    models = fit_baselines(data, quick)
    table = table_for(models, weights, data["x_val"], data["vols_val"], data["scalers"], len(data["x_train"]), quick)
    table.to_csv(folder / "baselines_val.csv", index=False)
    draw(table, model, quick, "val")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(main(args.model, args.quick).round(3).to_string(index=False))
