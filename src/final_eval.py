"""Final evaluation on the locked test set. THE ONLY SCRIPT THAT READS TEST ROWS.

It is run once, at milestone M7, after code, configuration and all networks are frozen. Nothing it
produces may feed back into a modelling decision. The 12,000 test rows are the authors' own test
rows, in their order, so "the first 5,000" are the surfaces their notebook calibrates.

  --dry-run         uses validation rows in place of test rows, to smoke-test this script before
                    the freeze without touching the test set. Outputs are labelled "dryrun".
  --paper-protocol  a separate comparison run (not part of M7): trains as the authors did, on all
                    68,000 rows with early stopping watching the test set and last-epoch weights,
                    to show how much the flaw we fixed changes the reported number.
  --authors-calibration  another comparison run (not part of M7): sets the authors' published
                    calibrations of the same test surfaces beside ours, and recomputes the RMSE of
                    their Figure 9 as their notebook does. Needs the M7 outputs and data/get_data.py.

Usage:  python -m src.final_eval --model rbergomi [--quick] [--dry-run | --paper-protocol | --authors-calibration]
"""
import argparse

import numpy as np
import pandas as pd

from src.experiments import authors_calibration, baselines, edge_of_box, sample_efficiency, seeds, worst_errors
from src import plots
from src.calibrate import GRADIENT_BASED, GRADIENT_FREE, calibrate_surfaces, forward, summarise
from src.config import CALIBRATION, DATASETS, N_PARAMS, QUICK, RECIPE, SPLIT_FILE, UPGRADE_MODEL, results_dir
from src.data import fit_scalers, load_raw, load_train_val, scale_params, scale_vols, unscale_vols
from src.evaluate import grid_table, relative_errors, summary_row
from src.model import load_weights, numpy_weights
from src.train import fit

N_QUICK_ROWS = 300


def load_test_rows(model: str, quick: bool) -> tuple[np.ndarray, np.ndarray]:
    """Parameters and vols of the locked test rows, in the authors' order. The only place they are read."""
    with np.load(SPLIT_FILE) as saved:
        idx = saved["test_idx"]
    if quick:
        idx = idx[:N_QUICK_ROWS]
    raw = load_raw(model)
    return raw[idx, :N_PARAMS], raw[idx, N_PARAMS:]


def evaluate_accuracy(model: str, quick: bool, label: str, theta, vols, scalers) -> pd.DataFrame:
    """Figure 6 quantities for every trained seed on the final rows."""
    folder, x, rows = results_dir(model, quick), scale_params(theta, scalers), []
    for seed in sorted(int(p.stem.split("seed")[1]) for p in folder.glob("weights_seed*.npz")):
        errors = relative_errors(load_weights(folder / f"weights_seed{seed}.npz"), x, vols, scalers)
        grid_table(errors).to_csv(folder / f"accuracy_{label}_seed{seed}.csv", index=False)
        rows.append({"seed": seed, "split": label, **summary_row(errors)})
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / f"accuracy_summary_{label}.csv", index=False)
    return summary


def evaluate_calibration(model: str, quick: bool, label: str, theta, vols, scalers, weights) -> pd.DataFrame:
    """Figures 8 and 9 on the first 5,000 final rows (100 for the gradient-free optimizers)."""
    folder, names = results_dir(model, quick), DATASETS[model]["param_names"]
    n_based = QUICK["n_calibrations"] if quick else CALIBRATION["n_gradient_based"]
    n_free = QUICK["n_calibrations_gradient_free"] if quick else CALIBRATION["n_gradient_free"]
    y = scale_vols(vols, scalers)
    parts = [calibrate_surfaces(weights, optimizers, theta[:n], vols[:n], y[:n], scalers, names)
             for optimizers, n in ((GRADIENT_BASED, n_based), (GRADIENT_FREE, n_free))]
    results = pd.concat(parts, ignore_index=True)
    results.to_csv(folder / f"calibration_{label}.csv.gz", index=False, float_format="%.8g")
    summary, param_errors = summarise(results, names)
    summary.to_csv(folder / f"calibration_summary_{label}.csv", index=False)
    param_errors.to_csv(folder / f"calibration_param_errors_{label}.csv", index=False)
    return summary


def final_evaluation(model: str, quick: bool = False, dry_run: bool = False) -> None:
    """Everything the README and the notebook report on test data, written next to the validation versions."""
    label = "dryrun" if dry_run else "test"
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    scalers, weights = data["scalers"], load_weights(folder / "weights_seed0.npz")
    theta, vols = (data["theta_val"], data["vols_val"]) if dry_run else load_test_rows(model, quick)
    x = scale_params(theta, scalers)
    print(f"{model}: final evaluation on {len(theta):,} {plots.SPLIT_NAMES[label]} rows", flush=True)

    accuracy = evaluate_accuracy(model, quick, label, theta, vols, scalers)
    seed_table = seeds.seed_table(accuracy, label)
    seed_table.to_csv(folder / f"seeds_{label}.csv", index=False)
    seeds.draw(seed_table, model, quick, label)
    plots.plot_error_heatmaps(model, quick, split=label)

    evaluate_calibration(model, quick, label, theta, vols, scalers, weights)
    plots.plot_calibration_times(model, quick, split=label)
    plots.plot_calibration_errors(model, quick, split=label)
    if model != UPGRADE_MODEL:
        return

    edge = edge_of_box.compute(model, weights, x, theta, vols, scalers)
    edge.to_csv(folder / f"edge_of_box_{label}.csv", index=False)
    edge_of_box.draw(edge, model, quick, label)

    worst = worst_errors.compute(model, weights, x, theta, vols, scalers)
    worst.to_csv(folder / f"worst_errors_{label}.csv", index=False)
    worst_errors.draw(worst, model, quick, label)

    efficiency = sample_efficiency.score_saved_runs(model, quick, theta, vols)
    efficiency.to_csv(folder / f"sample_efficiency_{label}.csv", index=False)
    sample_efficiency.draw(efficiency, model, quick, label)

    fitted = baselines.fit_baselines(data, quick)
    table = baselines.table_for(fitted, weights, x, vols, scalers, len(data["x_train"]), quick)
    table.to_csv(folder / f"baselines_{label}.csv", index=False)
    baselines.draw(table, model, quick, label)


def paper_protocol(model: str, quick: bool = False, seed: int = 0) -> pd.DataFrame:
    """Train the way the authors' notebook does and score on the test set, next to our honest protocol.

    Their way: scalers and training on all 68,000 non-test rows, early stopping on the TEST set,
    last-epoch weights. The resulting test error is optimistic by construction; the size of the
    gap to our protocol is what this run measures. Requires the M7 outputs.
    """
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    theta_test, vols_test = load_test_rows(model, quick)
    theta = np.vstack([data["theta_train"], data["theta_val"]])
    vols = np.vstack([data["vols_train"], data["vols_val"]])
    scalers = fit_scalers(model, vols)
    max_epochs = QUICK["max_epochs"] if quick else RECIPE["max_epochs"]
    net, history, summary = fit(scale_params(theta, scalers), scale_vols(vols, scalers), scale_params(theta_test, scalers),
                                scale_vols(vols_test, scalers), seed, max_epochs, restore_best=False)
    history.to_csv(folder / "paper_protocol_history.csv", index=False)
    predicted = unscale_vols(forward(numpy_weights(net), scale_params(theta_test, scalers)), scalers)
    errors = np.abs(predicted - vols_test) / vols_test
    ours = pd.read_csv(folder / "accuracy_summary_test.csv").set_index("seed").loc[seed]
    table = pd.DataFrame([
        {"protocol": "paper: 68,000 rows, early stopping on the test set, last-epoch weights", "seed": seed,
         "n_train": len(theta), "epochs_run": summary["epochs_run"], **summary_row(errors)},
        {"protocol": "ours: 60,000 rows, early stopping on 8,000 validation rows, best weights, test set untouched",
         "seed": seed, "n_train": len(data["theta_train"]), "epochs_run": np.nan, **ours.drop("split").to_dict()}])
    table.to_csv(folder / "paper_protocol.csv", index=False)
    return table


def compare_with_authors_calibration(model: str, quick: bool = False) -> pd.DataFrame:
    """Set the authors' published calibrations of the same test surfaces beside ours (stretch, after M7).

    Their repository holds the parameters their notebook fitted for the first 5,000 test rows and those
    parameters re-priced by their pricer. This reads the same test rows, hands them with our fitted
    parameters to src/experiments/authors_calibration.py, and writes authors_calibration_* only.
    """
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    theta, vols = load_test_rows(model, quick)
    ours = pd.read_csv(folder / "calibration_test.csv.gz")
    n = int((ours["optimizer"] == "Levenberg-Marquardt").sum())
    per_param, rmse, per_surface = authors_calibration.compare(model, load_weights(folder / "weights_seed0.npz"), theta[:n], vols[:n],
                                                               data["scalers"], ours)
    per_param.to_csv(folder / "authors_calibration_params_test.csv", index=False)
    rmse.to_csv(folder / "authors_calibration_rmse_test.csv", index=False)
    per_surface.to_csv(folder / "authors_calibration_test.csv.gz", index=False, float_format="%.8g")
    authors_calibration.draw(per_param, per_surface, model, quick)
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper-protocol", action="store_true")
    parser.add_argument("--authors-calibration", action="store_true")
    args = parser.parse_args()
    if args.paper_protocol:
        print(paper_protocol(args.model, args.quick).round(4).to_string(index=False))
    elif args.authors_calibration:
        print(compare_with_authors_calibration(args.model, args.quick).drop(columns="description").round(4).to_string(index=False))
    else:
        final_evaluation(args.model, args.quick, args.dry_run)
