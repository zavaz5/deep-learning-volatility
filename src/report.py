"""Documents whose numbers must come from files, not from typing.

Our numbers are read from results/; numbers quoted from the paper or the authors' notebooks come
from src/paper_numbers.py. This module only arranges them into text.

Usage:  python -m src.report --discrepancies | --facts | --time-figures | --check
"""
import argparse
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time

import pandas as pd
import scipy.optimize

from src.config import DATASETS, N_PARAMS, N_VOLS, RECIPE, REPO_URL, ROOT, SEEDS, results_dir
from src.ledger import us_to_ms
from src.paper_numbers import AUTHORS_COMMIT, NOTEBOOK, NOTEBOOK_ONEFACTOR, NOTEBOOK_RBERGOMI, PAPER, notebook, paper


def data_facts(model: str) -> dict:
    """Facts about one data file, written by src/data.py."""
    with open(results_dir(model) / "data_facts.json") as f:
        return json.load(f)


def weight_counts() -> tuple[int, int]:
    """(correct count, count by the paper's formula) for an 11-30-30-30-30-88 network.

    Four hidden layers need one input map, three hidden-to-hidden maps and one output map.
    The paper's formula multiplies the hidden-to-hidden term by four instead of three.
    """
    width = RECIPE["width"]
    input_map, hidden_map, output_map = (N_PARAMS + 1) * width, (width + 1) * width, (width + 1) * N_VOLS
    return input_map + 3 * hidden_map + output_map, input_map + 4 * hidden_map + output_map


def discrepancies_markdown(with_header: bool = True) -> str:
    """The differences between the paper and the authors' code and data, as Markdown: paper page on one side, notebook
    cell or data statistic on the other. The notebook shows the tables without the header; `--discrepancies` prints it whole."""
    rb, of = data_facts("rbergomi"), data_facts("onefactor")
    rho_rb = rb["param_names"].index("rho")
    correct, by_formula = weight_counts()
    assert correct == notebook("rbergomi", "n_weights") and by_formula == paper("n_weights_stated")
    scipy_default = inspect.signature(scipy.optimize.least_squares).parameters["method"].default
    low_rho = rb["share_in_lowest_5pct"]["rho"]
    fig6 = paper("fig6_colour_limits_pct")
    of_order = ", ".join(of["param_names"][8:])
    of_ranges = "; ".join(f"{n} in [{lo:.4g}, {hi:.4g}]" for n, lo, hi in
                          zip(of["param_names"][8:], of["param_min"][8:], of["param_max"][8:]))
    kept, failed = notebook("rbergomi", "n_rmse_rows_kept"), notebook("rbergomi", "n_surface_rows_with_failed_vols")

    header = f"""# Where the paper and the authors' code disagree

Built by `src/report.py`: the data statistics are read from `results/*/data_facts.json` and the quoted numbers
from `src/paper_numbers.py`.

**Paper:** Horvath, Muguruza, Tomas, *Deep Learning Volatility*, arXiv:1901.09647v2. Page numbers refer to that PDF.
**Code and data:** [{REPO_URL.split('github.com/')[1]}]({REPO_URL}) at commit `{AUTHORS_COMMIT[:12]}`.
The paper links to this repository on pages 4 and 8. Notebooks: `{NOTEBOOK_RBERGOMI}` and
`{NOTEBOOK_ONEFACTOR}`. Cell numbers count from 0.

"""
    body = f"""We read the notebooks to document these differences. We copied no code from them and never loaded their weights.

## 1. Six differences we checked first, re-verified

| # | Item | Paper | Authors' code and data | What we do |
| --- | --- | --- | --- | --- |
| 1 | Number of weights | {paper('n_weights_stated'):,} ({PAPER['n_weights_stated'][1]}) | {notebook('rbergomi', 'n_weights'):,}: the model summary in cell 11 lists layers of {', '.join(str(v) for v in notebook('rbergomi', 'layer_weights'))} weights | Our network has {correct:,} weights, asserted in `src/tests/`. The paper's formula counts four hidden-to-hidden maps; four hidden layers have three, and 3 instead of 4 gives {correct:,} instead of {by_formula:,} |
| 2 | Range of rho, rough Bergomi | Uniform on [{paper('rho_range')[0]}, {paper('rho_range')[1]}] ({PAPER['rho_range'][1]}) | The data file spans {rb['param_min'][rho_rb]:.5f} to {rb['param_max'][rho_rb]:.5f}; cell 8 scales with bounds [-1, 0]. The paper itself uses [-1, 0] for SPX ({PAPER['rho_range_spx'][1]}) | Scale with [-1, 0], the box the data was drawn from |
| 3 | Epochs | {paper('epochs')} ({PAPER['epochs'][1]}) | Up to {notebook('rbergomi', 'max_epochs')} (cell 13); training stopped at epoch {notebook('rbergomi', 'stopped_epoch')} for rough Bergomi and {notebook('onefactor', 'stopped_epoch')} for 1-factor | Allow {RECIPE['max_epochs']}, as the code that produced the published weights did |
| 4 | Early stopping | Stops when the error on the test set has not improved for {paper('patience')} steps ({PAPER['patience'][1]}) | Cell 13 passes the test arrays as `validation_data`, so the test set steers training and no untouched test set remains | Stop on a separate validation set of 8,000 rows; the 12,000 test rows stay locked until the final evaluation |
| 5 | Data behind Figures 6 and 7 | The captions say all {paper('n_train'):,} training samples (pp. 20-21) | Cell 25 computes the errors on `X_test`, and the heading above it (cell 24) says "Test set" | Show training, validation and test heatmaps separately and say which is which |
| 6 | Network pricing time | {us_to_ms(paper('nn_price_us'))} milliseconds ({PAPER['nn_price_us'][1]}) | Cell 22 times a hand-written NumPy forward pass: {us_to_ms(notebook('rbergomi', 'numpy_forward_us'))} milliseconds. Cell 23 times Keras `predict`: {us_to_ms(notebook('rbergomi', 'keras_predict_us'))} milliseconds | Time both a NumPy forward pass and PyTorch, and say which number is which |

## 2. Further differences we found

| # | Item | Paper | Authors' code and data | What we do |
| --- | --- | --- | --- | --- |
| 7 | The optimizer called Levenberg-Marquardt | Section 4.2, p. 21, picks Levenberg-Marquardt as the most balanced optimizer; Figures 8-10 are labelled with it | The calibration cell calls `scipy.optimize.least_squares` without a `method` argument. SciPy's default is `'{scipy_default}'` (trust-region reflective), not `'lm'` (checked against the installed SciPy by this script) | Run both: `method='lm'` as the headline, and the authors' exact call as a second, labelled result |
| 8 | Definition of the RMSE in Figures 9 and 10 | Section 4.2, p. 22: the root of the *sum* of squared differences between the *network* surface at the calibrated parameters and the target | Cell 38 takes the root of the *mean*. The calibrated surfaces are not network output: they are loaded from `Data/surfacesFromNNRoughBergomiTermStructure.txt`, whose first 11 columns are the calibrated parameters and whose vols contain the value -1.79769e+308 in {failed} of {notebook('rbergomi', 'n_calibrated'):,} rows. That value marks a failed implied-vol inversion in an external pricer; a network cannot produce it. Dropping those rows leaves the {kept:,} the notebook reports | Network surface against target, as the paper's formula says; root-mean-square as headline, root-sum-square saved next to it. Re-pricing by Monte Carlo is not tested here; it needs a rough Bergomi simulator, which we did not write. Their re-priced surfaces are published, so we recompute their RMSE as coded from that file and set it beside ours (`results/rbergomi/authors_calibration_rmse_test.csv`): its tail comes from surfaces holding a zero implied vol, failed inversions the notebook keeps |
| 9 | Parameter order, 1-factor Bergomi | Listed as ({', '.join(paper('onefactor_param_order'))}) ({PAPER['onefactor_param_order'][1]}) | Columns 8-10 of the file are ({of_order}): {of_ranges}. Cell 8 of the 1-factor notebook uses the same order | Set in `src/config.py` |
| 10 | Largest error: text against figure | Up to {paper('max_rel_error_text_pct'):.0f}% ({PAPER['max_rel_error_text_pct'][1]}) | The colour bar of Figure 6's right panel ends at {fig6['max'][1]:.1f}%. matplotlib scales a heatmap to its data, so that is the authors' largest error | Report our maximum next to both numbers |
| 11 | How evenly the box is sampled | Uniform sampling ({PAPER['rho_range'][1]}) | Only {100 * low_rho:.2f}% of our {rb['n_train_rows_used']:,} rough Bergomi training rows lie in the lowest 5% of the rho range, where uniform sampling gives 5%. A likely cause is that parameter sets whose Monte Carlo vols could not be inverted were dropped | Used in the edge-of-box experiment: the network saw half as many examples near rho = -1 |

## 3. Choices the paper does not state but the notebooks fix

| Item | Authors' notebook | What we do |
| --- | --- | --- |
| Bounds in calibration | The calibration cell passes no bounds to any optimizer | Bounds [-1, 1] where the method supports them; we record how often a solution leaves the box |
| Tolerances | `tol={notebook_setting('optimizer_tol'):g}`, `maxiter={notebook_setting('optimizer_maxiter')}`, `gtol={notebook_setting('least_squares_gtol'):g}` | The same |
| Gradient-free optimizers | Figure 8 shows COBYLA, Nelder-Mead and differential evolution, but no published notebook runs them (we searched the piecewise and the flat forward-variance notebooks), so their settings are unknown | COBYLA and Nelder-Mead with the same `tol` and `maxiter` as the other `minimize` calls, differential evolution with SciPy's defaults and a fixed seed, all with bounds. SciPy's COBYLA has also been re-implemented since 2019, so its timing is the least comparable |
| Which weights are kept | Keras `EarlyStopping` without `restore_best_weights`: the saved network is the last epoch ({notebook('rbergomi', 'stopped_epoch')}, validation loss {notebook('rbergomi', 'final_val_loss')}), not the best one (epoch {notebook('rbergomi', 'best_val_epoch')}, {notebook('rbergomi', 'best_val_loss'):.4f}) | Restore the best validation weights |
| Initialisation and Adam epsilon | Keras defaults: Glorot-uniform weights, zero biases, epsilon 1e-7 | Set the same in PyTorch, whose own defaults differ (epsilon {RECIPE['adam_eps']:g} is in `src/config.py`) |
| Reported loss | Keras reports the mean of per-batch RMSE values with batch size {RECIPE['batch_size']} | Log that and the whole-set RMSE; stop early on the former |
| What the pricing time covers | Cell 22 times scaled input to scaled output, without converting back to vols | Time that, and the full path from raw parameters to vols |
| Statistic in Figure 8 | Cell 32 plots the mean time per calibration | Save mean, median and quantiles; draw means for a like-for-like chart |
"""
    return header + body if with_header else body


def notebook_setting(key: str):
    """Value of a setting shared by both notebooks."""
    return NOTEBOOK[key][0]


def write_facts() -> None:
    """results/model_facts.json and results/environment.json: the weight count of our network and the machine we ran on."""
    from src.model import build_network, count_weights
    net = build_network()
    layers = [sum(p.numel() for p in layer.parameters()) for layer in net if list(layer.parameters())]
    (ROOT / "results" / "model_facts.json").write_text(json.dumps({"n_weights": count_weights(net), "layer_weights": layers}, indent=2))
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        system = f"macOS {platform.mac_ver()[0]}"
    else:
        system = platform.platform()
    (ROOT / "results" / "environment.json").write_text(json.dumps(
        {"python": platform.python_version(), "os": system, "cpu": cpu, "logical_cores": os.cpu_count()}, indent=2))
    print("wrote results/model_facts.json and results/environment.json")


FIGURE_COMMANDS = [
    ("results/figures/rbergomi/surfaces.png", "Five training surfaces and their mean", "python -m src.plots --model rbergomi --figure surfaces"),
    ("results/figures/rbergomi/learning_curves.png", "Learning curves, five seeds", "python -m src.plots --model rbergomi --figure learning_curves"),
    ("results/figures/rbergomi/error_heatmaps_test.png", "Paper's Figure 6", "python -m src.plots --model rbergomi --figure error_heatmaps_test"),
    ("results/figures/calibration_times_test.png", "Paper's Figure 8 (both models)", "python -m src.plots --model rbergomi --figure calibration_times_test"),
    ("results/figures/rbergomi/calibration_errors_test.png", "Paper's Figure 9", "python -m src.plots --model rbergomi --figure calibration_errors_test"),
    ("results/figures/onefactor/error_heatmaps_test.png", "Paper's Figure 7", "python -m src.plots --model onefactor --figure error_heatmaps_test"),
    ("results/figures/onefactor/calibration_errors_test.png", "Paper's Figure 10", "python -m src.plots --model onefactor --figure calibration_errors_test"),
    ("results/figures/rbergomi/seeds_val.png", "Upgrade: five seeds", "python -m src.experiments.seeds --model rbergomi"),
    ("results/figures/rbergomi/edge_of_box_val.png", "Upgrade: edge of the box", "python -m src.experiments.edge_of_box --model rbergomi"),
    ("results/figures/rbergomi/worst_errors_val.png", "Critique: where the largest errors are", "python -m src.experiments.worst_errors --model rbergomi"),
    ("results/figures/rbergomi/offline_cost.png", "Critique: offline cost", "python -m src.experiments.offline_cost --model rbergomi"),
    ("results/figures/rbergomi/noise_robustness_val.png", "Stretch: noisy quotes (2,000 calibrations)", "python -m src.experiments.noise_robustness --model rbergomi"),
    ("results/figures/rbergomi/ablation_activation_val.png", "Stretch: ELU against ReLU (trains one network)", "python -m src.experiments.ablation_activation --model rbergomi"),
    ("results/figures/rbergomi/authors_calibration_test.png", "Stretch: the authors' calibrations against ours (reads the test set)", "python -m src.final_eval --model rbergomi --authors-calibration"),
]
SLOW_FIGURE_COMMANDS = [
    ("results/figures/rbergomi/sample_efficiency_val.png", "Upgrade: sample efficiency (trains 12 networks)", "python -m src.experiments.sample_efficiency --model rbergomi"),
    ("results/figures/rbergomi/baselines_val.png", "Upgrade: simple-regressor baselines (fits them)", "python -m src.experiments.baselines --model rbergomi"),
]


def time_figures() -> None:
    """Run every cheap figure command once and record its wall time; slow ones take their time from results/runtimes_full.csv."""
    rows = []
    for figure, shows, command in FIGURE_COMMANDS:
        start = time.perf_counter()
        subprocess.run([sys.executable] + command.split()[1:], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        rows.append({"figure": figure, "shows": shows, "command": command, "seconds": time.perf_counter() - start})
    full = pd.read_csv(ROOT / "results" / "runtimes_full.csv").set_index("command")["seconds"]
    for figure, shows, command in SLOW_FIGURE_COMMANDS:
        rows.append({"figure": figure, "shows": shows, "command": command, "seconds": full[command]})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "figure_runtimes.csv", index=False)
    print("wrote results/figure_runtimes.csv")


NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
REFERENCE = re.compile(r"(https?://\S+|arXiv:\S+|DOI\s*\S+|commit\s+\S+|\b(19|20)\d{2}\b"
                       r"|(Figures?|Tables?|Sections?|Theorems?|Remark|pp?\.|pages?|cells?|slides?|seeds?|steps?|Tier|degree|items?)\s*"
                       r"\d[\d.]*(\s*(,|-|–|and|to)\s*\d[\d.]*)*)", re.IGNORECASE)


def tokens(text: str) -> set[str]:
    """Numeric tokens of a text, without thousands separators, after removing references to figures, pages and the like."""
    return {t.replace(",", "").rstrip(".") for t in NUMBER.findall(REFERENCE.sub(" ", text))}


def recheck(manifest: dict) -> list[str]:
    """Re-read every ledger entry of a manifest from its file in results/: a recorded value must not have drifted."""
    from src.ledger import SOURCES, read_value
    problems = []
    for document, record in manifest.items():
        for key, entry in record["ours"].items():
            now = read_value(SOURCES[key])
            if abs(now - entry["value"]) > 1e-12 * max(1.0, abs(now)):
                problems.append(f"{document}: {key} recorded {entry['value']} but {entry['file']} now gives {now}")
    return problems


def allowed(record: dict, extra: set[str]) -> set[str]:
    """Every numeric token the ledger handed to one document, ours and quoted, plus `extra`."""
    texts = [t for e in record["ours"].values() for t in e["texts"]] + [t for e in record["quoted"].values() for t in e["texts"]]
    return set().union(*(tokens(t) for t in texts), extra)


def check() -> list[str]:
    """Every number of ours in the README results table must trace to a file in results/.

    1. Each ledger entry is re-read from its file and must equal the recorded value.
    2. Each numeric token in the "Ours" column of the README table must be a number the ledger handed to the README.
    """
    from src.ledger import MANIFEST
    manifest = json.loads(MANIFEST.read_text())
    problems = recheck(manifest)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = [line for line in readme.splitlines() if line.startswith("| ") and line.count("|") >= 5]
    ok = allowed(manifest["README.md"], {str(len(SEEDS))})
    for line in table[2:]:
        ours = line.split("|")[3]
        for token in tokens(ours) - ok:
            problems.append(f"README results table: '{token}' in \"{ours.strip()}\" does not come from the ledger")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--discrepancies", action="store_true", help="print the table of differences between the paper and the authors' code")
    parser.add_argument("--facts", action="store_true", help="write results/model_facts.json and results/environment.json")
    parser.add_argument("--time-figures", action="store_true", help="time every figure command into results/figure_runtimes.csv")
    parser.add_argument("--check", action="store_true", help="verify that every number in the README table traces to results/")
    args = parser.parse_args()
    if args.discrepancies:
        print(discrepancies_markdown())
    if args.facts:
        write_facts()
    if args.time_figures:
        time_figures()
    if args.check:
        found = check()
        print("\n".join(found) if found else "traceability check passed: every number traces to results/ or to src/paper_numbers.py")
        raise SystemExit(1 if found else 0)
