"""Generate README.md. Every number of ours is fetched through the ledger (src/ledger.py) from results/;
every number quoted from the paper comes from src/paper_numbers.py. Only the prose is written here.

Usage:  python -m src.readme
"""
import json

import pandas as pd

from src.config import CALIBRATION, DATASETS, REPO_URL, ROOT, SEEDS, SPLIT
from src.ledger import Ledger, sig3, us_to_ms

MARKS = {"repro": "Reproduction", "upgrade": "**Our upgrade**", "critique": "**Our critique**", "stretch": "**Our stretch**", "not": "Not reproduced", "": ""}
ORIGIN_OVERRIDES = {"Training rows": "upgrade", "Share of test cells": "critique", "Monte Carlo pricer": "not", "Fitted parameters inside the training box": "critique",
                    "Offline cost": "critique"}

OPTIMIZER_ROWS = [("L-BFGS-B", "lbfgsb"), ("SLSQP", "slsqp"), ("BFGS", "bfgs"), ("Levenberg-Marquardt", "lm"),
                  ("COBYLA", "cobyla"), ("Differential Evolution", "de"), ("Nelder-Mead", "nm")]


def results_rows(L: Ledger) -> list[dict]:
    """The results table as data: item, paper's number, our number, comment, and the two values when both are numeric."""
    rows = []

    current = {"kind": "repro"}

    def add(item, paper_text, ours_text, comment, paper_value=None, ours_value=None):
        kind = next((k for prefix, k in ORIGIN_OVERRIDES.items() if item.startswith(prefix)), current["kind"])
        rows.append({"item": item, "paper": paper_text, "ours": ours_text, "comment": comment,
                     "paper_value": paper_value, "ours_value": ours_value, "kind": kind})

    def section(title, kind="repro"):
        current["kind"] = ""
        add(f"**{title}**", "", "", "")
        current["kind"] = kind

    hardware = "Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine."
    section("Rough Bergomi: the network")
    add("Weights in the network", L.q("n_weights_stated", "{:,}"), L.n("rb.n_weights"),
        "The paper's formula counts four hidden-to-hidden maps; four hidden layers have three. The authors' own model summary prints "
        + L.q("n_weights", "{:,}", model="rbergomi") + ".", L.qv("n_weights_stated"), L.value("rb.n_weights"))
    add("Training rows", L.q("n_train", "{:,}"), L.n("rb.n_train"),
        f"We set {SPLIT['n_val']:,} of the paper's training rows aside for validation, so that the test set is never used for a decision.")
    add("Epochs run, seed 0", f"{L.q('epochs')} in the text; the notebook stopped at {L.q('stopped_epoch', model='rbergomi')}",
        f"{L.n('rb.epochs_run')} (best epoch {L.n('rb.best_epoch')})",
        "Same patience of 25. We stop on validation rows and keep the best epoch; the authors stop on the test set and keep the last epoch.",
        L.qv("stopped_epoch", model="rbergomi"), L.value("rb.epochs_run"))
    add("Final loss, training / held-out (scaled RMSE, Keras convention)",
        f"{L.q('final_train_loss', '{:.4f}', model='rbergomi')} / {L.q('final_val_loss', '{:.4f}', model='rbergomi')} (notebook)",
        f"{L.n('rb.train_loss')} / {L.n('rb.val_loss')}",
        "Held-out is the test set for the authors and the validation set for us. We train on fewer rows and stop earlier.",
        L.qv("final_val_loss", model="rbergomi"), L.value("rb.val_loss"))
    add("Average relative error on the test set", f"far below {L.q('avg_rel_error_bound_pct')}%",
        f"{L.n('rb.test.avg_err_mean')} ± {L.n('rb.test.avg_err_std')} (mean ± std of {len(SEEDS)} seeds)",
        "Matches the claim. The paper reports one run whose stopping rule saw the test set; ours never did.",
        L.qv("avg_rel_error_bound_pct"), L.value("rb.test.avg_err_mean"))
    add("Largest cell average (top of Figure 6, left panel)", L.q("fig6_colour_limits_pct", "{:.2f}%", pick=["mean", 1]),
        L.n("rb.test.cell_mean_max"), "Same cell: maturity 0.1, strike 1.5. Seed 0.",
        L.qv("fig6_colour_limits_pct", pick=["mean", 1]), L.value("rb.test.cell_mean_max"))
    add("Largest cell standard deviation (Figure 6, middle)", L.q("fig6_colour_limits_pct", "{:.2f}%", pick=["std", 1]),
        L.n("rb.test.cell_std_max"), "Seed 0.", L.qv("fig6_colour_limits_pct", pick=["std", 1]), L.value("rb.test.cell_std_max"))
    add("Maximum relative error", f"{L.q('max_rel_error_text_pct', '{:.0f}')}% in the text, "
        f"{L.q('fig6_colour_limits_pct', '{:.1f}', pick=['max', 1])}% on Figure 6's colour bar", L.n("rb.test.max_err"),
        "The paper's text and its own figure disagree; ours is close to the figure. The error sits at maturity 0.1 and the extreme strikes.",
        L.qv("fig6_colour_limits_pct", pick=["max", 1]), L.value("rb.test.max_err"))
    add("Share of test cells with error above 1% (the paper's bid-ask yardstick)", "not reported", L.n("rb.test.share_above_1pct"),
        "The average hides a tail that misses the paper's own tolerance.")

    section("Rough Bergomi: speed of one full surface, milliseconds")
    add("Network, NumPy forward pass", L.q("nn_price_us", us_to_ms), L.n("rb.speed.numpy", us_to_ms), hardware + " Scaled input to scaled output, as in the authors' cell 22.",
        L.qv("nn_price_us"), L.value("rb.speed.numpy"))
    add("Network Jacobian (88 x 11)", L.q("nn_gradient_us", us_to_ms), L.n("rb.speed.jacobian", us_to_ms), hardware + " Analytic chain rule in NumPy.",
        L.qv("nn_gradient_us"), L.value("rb.speed.jacobian"))
    add("Deep-learning framework, single forward call", f"{L.q('keras_predict_us', us_to_ms, model='rbergomi')} (Keras predict, notebook)",
        f"{L.n('rb.speed.torch', us_to_ms)} (PyTorch)", "Different frameworks; both are slower than plain NumPy for a single small vector.",
        L.qv("keras_predict_us", model="rbergomi"), L.value("rb.speed.torch"))
    add("Monte Carlo pricer", L.q("mc_rbergomi_us", us_to_ms), "not reproduced",
        "Needs a rough Bergomi simulator, which we did not write. We quote the paper's number and never present it as ours.")

    n_based, n_free = L.n("rb.cal.n"), L.n("rb.cal.n_free")
    section(f"Rough Bergomi: mean calibration time, milliseconds (first {n_based} test surfaces; {n_free} for gradient-free optimizers)")
    notes = {"lm": "The paper's bar was produced by SciPy's default `least_squares` method (TRF), not by `method='lm'`. That call takes "
                   + L.n("rb.cal.trf_ms") + " ms here. " + hardware,
             "cobyla": "SciPy's COBYLA is now a pure-Python re-implementation, and the authors' gradient-free settings are unpublished. "
                       "We assume their `tol` and `maxiter`; COBYLA then stops at the iteration cap.",
             "de": "SciPy defaults, fixed seed. " + hardware, "nm": "Assumed settings, see COBYLA. " + hardware}
    for name, short in OPTIMIZER_ROWS:
        add(name, L.q("fig8_mean_ms", sig3, pick=["rbergomi", name]) + " (read off Figure 8)", L.n(f"rb.cal.{short}_ms"),
            notes.get(short, hardware), L.qv("fig8_mean_ms", pick=["rbergomi", name]), L.value(f"rb.cal.{short}_ms"))

    section("Rough Bergomi: calibration accuracy, Levenberg-Marquardt")
    add("Surface RMSE, 99% quantile", f"below {L.q('rmse_q99_bound_pct')}%", L.n("rb.cal.rmse_q99"),
        "Ours compares the network surface at the fitted parameters with the target, as the paper's formula says. The authors' notebook "
        "instead loads surfaces re-priced by an external pricer, which we cannot do without a simulator.",
        L.qv("rmse_q99_bound_pct"), L.value("rb.cal.rmse_q99"))
    names = DATASETS["rbergomi"]["param_names"]
    for label, p in (("xi1", "xi1"), ("xi8", "xi8"), ("nu", "nu"), ("rho", "rho"), ("H", "H")):
        theirs = L.q("lm_mean_rel_error", "{:.2%}", model="rbergomi", pick=names.index(p))
        comment = ("Relative error explodes when the true rho is near zero; the mean is driven by a few such rows." if p == "rho" else
                   "Later forward variances affect few grid cells, so they are weakly identified." if p == "xi8" else "")
        add(f"Mean relative error of {label}", f"{theirs} (notebook, cell 34)", L.n(f"rb.cal.err.{p}"), comment,
            100 * L.qv("lm_mean_rel_error", model="rbergomi", pick=names.index(p)), L.value(f"rb.cal.err.{p}"))
    add("Fitted parameters inside the training box", "not reported; no bounds in the notebook", L.n("rb.cal.lm_in_box"),
        "Levenberg-Marquardt cannot take bounds. The rest of the fits land outside the region the network was trained on.")

    section("1-factor Bergomi (same code, `--model onefactor`)")
    add("Average relative error on the test set", f"far below {L.q('avg_rel_error_bound_pct')}%",
        f"{L.n('of.test.avg_err_mean')} ± {L.n('of.test.avg_err_std')}", "Five seeds.", L.qv("avg_rel_error_bound_pct"), L.value("of.test.avg_err_mean"))
    add("Largest cell average (top of Figure 7, left panel)", L.q("fig7_colour_limits_pct", "{:.2f}%", pick=["mean", 1]), L.n("of.test.cell_mean_max"),
        "Seed 0.", L.qv("fig7_colour_limits_pct", pick=["mean", 1]), L.value("of.test.cell_mean_max"))
    add("Maximum relative error (Figure 7, right panel)", L.q("fig7_colour_limits_pct", "{:.1f}%", pick=["max", 1]), L.n("of.test.max_err"),
        "Seed 0.", L.qv("fig7_colour_limits_pct", pick=["max", 1]), L.value("of.test.max_err"))
    add("Final loss, training / held-out", f"{L.q('final_train_loss', '{:.4f}', model='onefactor')} / {L.q('final_val_loss', '{:.4f}', model='onefactor')} (notebook)",
        f"{L.n('of.train_loss')} / {L.n('of.val_loss')}", "See the rough Bergomi row.", L.qv("final_val_loss", model="onefactor"), L.value("of.val_loss"))
    add("Levenberg-Marquardt, mean milliseconds", L.q("fig8_mean_ms", sig3, pick=["onefactor", "Levenberg-Marquardt"]) + " (read off Figure 8)",
        L.n("of.cal.lm_ms"), hardware, L.qv("fig8_mean_ms", pick=["onefactor", "Levenberg-Marquardt"]), L.value("of.cal.lm_ms"))
    add("Surface RMSE, 99% quantile", f"below {L.q('rmse_q99_bound_pct')}%", L.n("of.cal.rmse_q99"), "Network surface against target.",
        L.qv("rmse_q99_bound_pct"), L.value("of.cal.rmse_q99"))

    section("Our upgrades (rough Bergomi, test set)", kind="upgrade")
    add("Fixed flaw: locked test set, five seeds", "one run, no untouched test set", f"{L.n('rb.test.avg_err_mean')} ± {L.n('rb.test.avg_err_std')}",
        f"Validation gave {L.n('rb.val.avg_err_mean')}; training rows of seed 0 give {L.n('rb.train.avg_err')}. No sign of classic overfitting.")
    add("Edge of the box: rho in the lowest 5% of its range against the middle 90%", "not tested",
        f"{L.n('edge.test.rho_low')} against {L.n('edge.test.rho_inside')}",
        f"Only {L.n('rb.low_rho_share')} of training rows lie in that band (uniform sampling would give 5%), and the paper's SPX fits sit there (its Figure 11).")
    add("Missing baseline: ridge regression on polynomial features", "no baseline",
        f"{L.n('base.test.ridge_err')} at {L.n('base.test.ridge_us', us_to_ms)} ms per surface; the network: {L.n('base.test.nn_err')} at {L.n('base.test.nn_us', us_to_ms)} ms",
        f"The polynomial fits in {L.n('base.test.ridge_fit_s')} s. Gradient boosting reaches {L.n('base.test.gb_err')}, k-nearest neighbours {L.n('base.test.knn_err')}.")
    add("Sample efficiency: 5,000 / 20,000 / 40,000 / 60,000 training rows", "one size, 68,000",
        " / ".join(L.n(f"eff.test.n{n}") for n in (5_000, 20_000, 40_000, 60_000)),
        "Mean of three seeds. The error stops improving after 40,000 rows; the polynomial baseline shows this is the small network's limit, not noise in the data.")
    add("Offline cost left out of the paper's speed-up", f"speed-up of {L.q('speedup_range', '{:,}', pick=0)} to {L.q('speedup_range', '{:,}', pick=1)}",
        f"{L.n('cost.offline_mc_cpu_hours')} CPU-hours; break-even after about {L.n('cost.break_even_calibrations')} calibrations",
        "Monte Carlo times are the paper's. Assumes 12 surfaces per Levenberg-Marquardt evaluation when the Jacobian comes from finite differences.")
    section("Stretch experiments (rough Bergomi)", kind="stretch")
    add("Training the authors' way: 68,000 rows, early stopping on the test set, last-epoch weights", "their protocol",
        f"{L.n('pp.paper_avg')} on the test set, against {L.n('pp.ours_avg')} with our protocol (seed 0)",
        "The gain mixes 8,000 more training rows with a stopping rule that saw the test set; one run cannot separate them. The flaw we fixed does not change the conclusion.")
    add("ELU against ReLU (validation rows)", "ELU chosen, no experiment",
        f"error {L.n('abl.elu_err')} against {L.n('abl.relu_err')}; mean parameter error after calibration {L.n('abl.elu_param')} against {L.n('abl.relu_param')}",
        f"Same data, seed and recipe; {L.n('abl.n')} surfaces, each network with its exact Jacobian. Supports the paper's Remarks 1 and 6.")
    add("Noisy quotes: 0.5% relative noise, median parameter error (validation rows)", "not tested",
        f"ν {L.n('noise.nu.clean')} to {L.n('noise.nu.noisy')}; ξ₈ {L.n('noise.xi8.clean')} to {L.n('noise.xi8.noisy')}",
        "Well-identified parameters barely move; late forward variances nearly double.")
    add(f"Authors' published calibrations of the same {L.n('auth.n')} test surfaces against ours", "not compared in the paper",
        f"median gap between the two fits {L.n('auth.par.all.gap')} of the true value (ν {L.n('auth.par.nu.gap')}, ρ {L.n('auth.par.rho.gap')}, H {L.n('auth.par.H.gap')}); "
        f"their mean error on ξ₈ recomputed from their file {L.n('auth.par.xi8.authors')}",
        f"Same surfaces, same SciPy call. Recomputing their errors from their published file gives their notebook's numbers (ξ₈: "
        f"{L.q('lm_mean_rel_error', '{:.2%}', model='rbergomi', pick=7)}), so the rows line up. The gap between the two fits is about as large as each fit's median error against the truth "
        f"({L.n('auth.par.all.authors_median')} and {L.n('auth.par.all.ours_median')}): two independently trained networks lead the same optimizer to different parameter vectors of the same quality, "
        "a sign of weak identification.")
    add("Paper's Figure 9 RMSE as coded, 99% quantile and largest value: their re-priced surfaces against the target, root of the mean", f"below {L.q('rmse_q99_bound_pct')}% (Figure 9)",
        f"{L.n('auth.rmse.paper_q99')} and {L.n('auth.rmse.paper_max')}; without the {L.n('auth.n_zero_rows')} surfaces holding a zero implied vol: "
        f"{L.n('auth.rmse.paper_clean_q99')} and {L.n('auth.rmse.paper_clean_max')}; ours: {L.n('auth.rmse.ours_q99')} and {L.n('auth.rmse.ours_max')}",
        f"Recomputed from the authors' published file, {L.n('auth.rmse.paper_n')} of the {L.n('auth.n')} rows kept as in their notebook ({L.n('auth.n_failed')} re-pricings failed). "
        f"The tail of their Figure 9 comes from surfaces with a failed inversion, a zero implied vol, that the notebook keeps; without them the two definitions agree. "
        f"Their parameters through our network give {L.n('auth.rmse.theirs_ournet_q99')}: each network's own fit is the best fit on that network.")
    return rows


def check_comments(rows: list[dict]) -> None:
    """Acceptance rule: a row whose number differs from the paper's by more than 20% must explain why."""
    for row in rows:
        p, o = row["paper_value"], row["ours_value"]
        if isinstance(p, (int, float)) and isinstance(o, (int, float)) and abs(o - p) > 0.2 * abs(p):
            assert row["comment"].strip(), f"missing comment: {row['item']}"


def markdown_table(rows: list[dict]) -> str:
    """The Origin column comes last so that "Ours" stays the third column, which src/report.py checks."""
    lines = ["| Item | Paper | Ours | Comment | Origin |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['item']} | {r['paper']} | {r['ours']} | {r['comment']} | {MARKS[r['kind']]} |" for r in rows]
    return "\n".join(lines)


def figure_table() -> str:
    """One command per figure, with the time it took to draw on this machine (results/figure_runtimes.csv). Shown in the notebook."""
    times = pd.read_csv(ROOT / "results" / "figure_runtimes.csv")
    lines = ["| Figure file | Shows | Command | Seconds here |", "| --- | --- | --- | --- |"]
    lines += [f"| `{r.figure}` | {r.shows} | `{r.command}` | {r.seconds:.0f} |" for r in times.itertuples()]
    return "\n".join(lines)


def step_table() -> str:
    """Runtime of every computation step, measured by a full run of src/run_all.py on this machine. Shown in the notebook."""
    steps = pd.read_csv(ROOT / "results" / "runtimes_full.csv")
    lines = ["| Step | Command | Minutes here |", "| --- | --- | --- |"]
    lines += [f"| {r.step} | `{r.command}` | {r.seconds / 60:.1f} |" for r in steps.itertuples()]
    lines.append(f"| **Total** | `python -m src.run_all --full` | **{steps['seconds'].sum() / 60:.0f}** |")
    return "\n".join(lines)


def write_readme() -> None:
    L = Ledger("README.md")
    rows = results_rows(L)
    check_comments(rows)
    with open(ROOT / "results" / "environment.json") as f:
        env = json.load(f)
    notebook_seconds = json.load(open(ROOT / "results" / "notebook_runtime.json"))["seconds"]
    repo_name = REPO_URL.split("github.com/")[1]
    total_minutes = pd.read_csv(ROOT / "results" / "runtimes_full.csv")["seconds"].sum() / 60
    text = f"""# Deep Learning Volatility: a reproduction in PyTorch

Code and results of a reproduction of

> B. Horvath, A. Muguruza, M. Tomas. *Deep learning volatility: a deep neural network perspective on pricing and
> calibration in (rough) volatility models.* Quantitative Finance 21(1), 2021, 11-27.
> DOI [10.1080/14697688.2020.1817974](https://doi.org/10.1080/14697688.2020.1817974). Preprint arXiv:1901.09647v2.

A small network maps 11 model parameters to an 8 x 11 grid of implied volatilities, and calibration runs an optimizer on the network.
We train it on the authors' published Monte Carlo data for rough Bergomi and 1-factor Bergomi, with their exact test split, which one
script reads once. Figure, table and page numbers refer to the arXiv v2 PDF.

## Files

| Path | Contents |
| --- | --- |
| `DeepLearningVolatility.ipynb` | The notebook: the data, every script checked on a subset, every table and figure rebuilt from `results/`, one command per figure and per step |
| `src/` | The code: data and split, network, training, evaluation, speed, calibration, figures, and the final evaluation, the only reader of the test set |
| `src/experiments/` | One script per experiment the paper does not contain |
| `src/run_all.py` | The whole pipeline, `--quick` or `--full` |
| `src/tests/` | The tests, including the rule that only `src/final_eval.py` reads test rows |
| `data/` | The authors' six data files (MIT licence, `data/LICENSE`) and `get_data.py`, which checks their hashes |
| `results/` | Every metric (CSV, JSON), the trained weights, the split, and every figure (`results/figures/`) |

## Reproduced, and not

Reproduced with our code on the authors' data, for both models: Figures 6 and 7 (error heatmaps), the network columns of Table 2
(speed), Figure 8 (calibration time of seven optimizers), Figures 9 and 10 (parameter errors and surface RMSE after calibration).

Not reproduced: Figures 4 and 5 and the Monte Carlo column of Table 2 (they need a rough Bergomi simulator, which is not published),
Figures 11 and 12 (the SPX data is not public), Figure 13 (no barrier-option data or pricer is published) and Figure 14 (the model
classifier, not attempted). The RMSE of Figures 9 and 10 exactly as the authors computed it needs their external pricer; we recompute
it from their published re-priced surfaces instead (last block of the table).

Beyond the paper, in `src/experiments/`: a locked test set with a separate validation set and five seeds, an edge-of-box test, simple
baselines, a sample-efficiency curve, the worst errors, the offline cost and four stretch experiments. Where the paper, the authors'
code and the data disagree is shown in section 5 of the notebook (`python -m src.report --discrepancies` prints it).

## Results

"Ours" is read by `src/readme.py` from `results/`, "Paper" from `src/paper_numbers.py`, where each number is typed once with its page;
`python -m src.report --check` verifies both. Ours are test-set numbers unless the row says otherwise.

{markdown_table(rows)}

## Run

Python {env['python']}, CPU only, versions pinned in `requirements.txt`. On Linux or Windows install the CPU build of PyTorch first:
`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.

```bash
pip install -r requirements.txt
python data/get_data.py         # checks the included data; downloads a file only if it is missing
python -m src.run_all --full    # every result and figure, about {total_minutes:.0f} minutes here
python -m pytest                # the tests
```

`data/` holds six files copied unchanged from [{repo_name}]({REPO_URL}), with the authors' MIT licence (`data/LICENSE`): the two training
sets and four small files of the authors' published calibrations. `--full` rewrites `results/`: accuracy numbers come out identical, timings become those of your machine
(`DLV_OUTPUT_ROOT=/some/folder` writes elsewhere). `--quick` checks every script on a subset in a few minutes and writes only to
`results/quick/`. The notebook (`jupyter notebook DeepLearningVolatility.ipynb`, then *Run all*, {notebook_seconds:.0f} s here) never reads test rows.

Split and seeds: the authors' `train_test_split(test_size={SPLIT['test_size']}, random_state={SPLIT['authors_random_state']})` on the row order of the file, so our 12,000
test rows are theirs; {SPLIT['n_val']:,} of their 68,000 training rows form a validation set (`numpy.random.default_rng({SPLIT['val_seed']})`), leaving 60,000 / 8,000 / 12,000
(`results/split_indices.npz`). Early stopping and every decision used validation rows. Only `src/final_eval.py` reads test rows; the main
evaluation ran once, after the code and the networks were frozen. Seeds {', '.join(str(s) for s in SEEDS)} (`torch.manual_seed`, NumPy and Python seeds, deterministic
algorithms, one thread, float64); differential evolution uses seed {CALIBRATION['de_seed']}. Reported numbers: {env['os']}, {env['cpu']}.

## References

- Horvath, B., Muguruza, A., Tomas, M. (2021). Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models. *Quantitative Finance* 21(1), 11-27. DOI 10.1080/14697688.2020.1817974. arXiv:1901.09647v2. Code and data: [{repo_name}]({REPO_URL}), MIT licence.
- Bayer, C., Friz, P., Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance* 16(6), 887-904.
- Hernandez, A. (2017). Model calibration with neural networks. *Risk*, June 2017.

Method and data are the authors'. No code here was copied from their notebooks, and their trained weights were never loaded.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    L.save()
    print("wrote README.md")


if __name__ == "__main__":
    write_readme()
