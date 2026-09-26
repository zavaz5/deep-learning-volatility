"""No hand-typed results: every number of ours in the README, and in any other document generated from results/,
is fetched through this ledger, which reads it from a file in results/ and records where it came from.

SOURCES maps a short key to (file, how to pick the number, default format). A document asks for
`ledger.n("rb.test.avg_err_mean")` and gets formatted text; the ledger remembers the key, the file,
the value and the text. `save()` writes that record to results/numbers_manifest.json (or to another manifest), and
`python -m src.report --check` later re-reads every file and confirms nothing has drifted.
Numbers quoted from the paper go through `q()` and come from src/paper_numbers.py instead.
"""
import json
from pathlib import Path

import pandas as pd

from src.config import DATASETS, RESULTS_DIR, ROOT
from src.paper_numbers import NOTEBOOK, PAPER

MANIFEST = RESULTS_DIR / "numbers_manifest.json"


def sig3(value: float) -> str:
    """Three significant digits without exponents: 0.395, 5.33, 110, 1,924."""
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"


def us_to_ms(value_us: float) -> str:
    """A time stored in microseconds, shown in milliseconds with three significant digits: 9.19 -> 0.00919, 500,000 -> 500.
    Times are stored as measured (results/ and src/paper_numbers.py keep microseconds); documents show milliseconds."""
    return sig3(value_us / 1000)


def _csv(path, column, fmt, agg=None, **where):
    return {"file": path, "column": column, "where": where, "agg": agg, "format": fmt}


def _json(path, key, fmt):
    return {"file": path, "json_key": key, "format": fmt}


def _model_sources(m: str, tag: str) -> dict:
    """The same set of keys for each model; `tag` is the short prefix used in documents (rb, of)."""
    r, lm = f"results/{m}", "Levenberg-Marquardt"
    s = {
        f"{tag}.n_weights": _json("results/model_facts.json", "n_weights", "{:,.0f}"),
        f"{tag}.low_rho_share": _json(f"{r}/data_facts.json", "share_in_lowest_5pct.rho", "{:.2%}"),
        f"{tag}.n_train": _json(f"{r}/data_facts.json", "n_train_rows_used", "{:,.0f}"),
        f"{tag}.epochs_run": _json(f"{r}/train_summary_seed0.json", "epochs_run", "{:.0f}"),
        f"{tag}.best_epoch": _json(f"{r}/train_summary_seed0.json", "best_epoch", "{:.0f}"),
        f"{tag}.val_loss": _json(f"{r}/train_summary_seed0.json", "best_val_loss_keras", "{:.4f}"),
        f"{tag}.train_loss": _json(f"{r}/train_summary_seed0.json", "train_loss_running_at_best", "{:.4f}"),
        f"{tag}.train_rmse": _json(f"{r}/train_summary_seed0.json", "train_rmse_at_best", "{:.4f}"),
        f"{tag}.val_rmse": _json(f"{r}/train_summary_seed0.json", "best_val_rmse", "{:.4f}"),
        f"{tag}.train_seconds": _json(f"{r}/train_summary_seed0.json", "seconds", "{:.0f}"),
        f"{tag}.train.avg_err": _csv(f"{r}/accuracy_summary.csv", "avg_rel_error_pct", "{:.3f}%", seed=0, split="train"),
        f"{tag}.val.avg_err": _csv(f"{r}/accuracy_summary.csv", "avg_rel_error_pct", "{:.3f}%", seed=0, split="val"),
        f"{tag}.val.avg_err_mean": _csv(f"{r}/seeds_val.csv", "avg_rel_error_pct", "{:.3f}%", seed="mean") if m == "rbergomi"
        else _csv(f"{r}/accuracy_summary.csv", "avg_rel_error_pct", "{:.3f}%", agg="mean", split="val"),
        f"{tag}.test.avg_err_mean": _csv(f"{r}/seeds_test.csv", "avg_rel_error_pct", "{:.3f}%", seed="mean"),
        f"{tag}.test.avg_err_std": _csv(f"{r}/seeds_test.csv", "avg_rel_error_pct", "{:.3f}%", seed="std"),
        f"{tag}.test.max_err_mean": _csv(f"{r}/seeds_test.csv", "max_rel_error_pct", "{:.1f}%", seed="mean"),
        f"{tag}.test.avg_err": _csv(f"{r}/accuracy_summary_test.csv", "avg_rel_error_pct", "{:.3f}%", seed=0),
        f"{tag}.test.cell_mean_max": _csv(f"{r}/accuracy_summary_test.csv", "largest_cell_mean_pct", "{:.2f}%", seed=0),
        f"{tag}.test.cell_mean_min": _csv(f"{r}/accuracy_summary_test.csv", "smallest_cell_mean_pct", "{:.2f}%", seed=0),
        f"{tag}.test.cell_std_max": _csv(f"{r}/accuracy_summary_test.csv", "largest_cell_std_pct", "{:.2f}%", seed=0),
        f"{tag}.test.max_err": _csv(f"{r}/accuracy_summary_test.csv", "max_rel_error_pct", "{:.1f}%", seed=0),
        f"{tag}.test.p99_err": _csv(f"{r}/accuracy_summary_test.csv", "p99_rel_error_pct", "{:.2f}%", seed=0),
        f"{tag}.test.share_above_1pct": _csv(f"{r}/accuracy_summary_test.csv", "share_of_cells_above_1pct", "{:.1%}", seed=0),
        f"{tag}.test.n_rows": _csv(f"{r}/accuracy_summary_test.csv", "n_rows", "{:,.0f}", seed=0),
        f"{tag}.speed.numpy": _csv(f"{r}/speed.csv", "median_us", "{:.1f}", what="numpy_forward_scaled"),
        f"{tag}.speed.numpy_full": _csv(f"{r}/speed.csv", "median_us", "{:.1f}", what="numpy_forward_full"),
        f"{tag}.speed.torch": _csv(f"{r}/speed.csv", "median_us", "{:.1f}", what="torch_forward"),
        f"{tag}.speed.jacobian": _csv(f"{r}/speed.csv", "median_us", "{:.1f}", what="numpy_jacobian"),
        f"{tag}.speed.autograd": _csv(f"{r}/speed.csv", "median_us", "{:,.0f}", what="torch_autograd_jacobian"),
        f"{tag}.cal.n": _csv(f"{r}/calibration_summary_test.csv", "n_surfaces", "{:,.0f}", optimizer=lm),
        f"{tag}.cal.n_free": _csv(f"{r}/calibration_summary_test.csv", "n_surfaces", "{:,.0f}", optimizer="Nelder-Mead"),
        f"{tag}.cal.lm_in_box": _csv(f"{r}/calibration_summary_test.csv", "share_in_box", "{:.1%}", optimizer=lm),
        f"{tag}.cal.lm_evals": _csv(f"{r}/calibration_summary_test.csv", "mean_evaluations", "{:.1f}", optimizer=lm),
        f"{tag}.cal.lbfgsb_evals": _csv(f"{r}/calibration_summary_test.csv", "mean_evaluations", "{:.0f}", optimizer="L-BFGS-B"),
        f"{tag}.cal.rmse_q99": _csv(f"{r}/calibration_summary_test.csv", "rmse_q99_pct", "{:.2f}%", optimizer=lm),
        f"{tag}.cal.rmse_median": _csv(f"{r}/calibration_summary_test.csv", "rmse_median_pct", "{:.3f}%", optimizer=lm),
        f"{tag}.cal.rmse_max": _csv(f"{r}/calibration_summary_test.csv", "rmse_max_pct", "{:.2f}%", optimizer=lm),
        f"{tag}.cal.bfgs_success": _csv(f"{r}/calibration_summary_test.csv", "share_success", "{:.0%}", optimizer="BFGS"),
    }
    for seed in range(5):
        s[f"{tag}.test.seed{seed}.avg"] = _csv(f"{r}/accuracy_summary_test.csv", "avg_rel_error_pct", "{:.3f}%", seed=seed)
        s[f"{tag}.test.seed{seed}.max"] = _csv(f"{r}/accuracy_summary_test.csv", "max_rel_error_pct", "{:.1f}%", seed=seed)
        s[f"{tag}.seed{seed}.best_epoch"] = _json(f"{r}/train_summary_seed{seed}.json", "best_epoch", "{:.0f}")
    for p in DATASETS[m]["param_names"]:
        s[f"{tag}.cal.err.{p}"] = _csv(f"{r}/calibration_param_errors_test.csv", "mean_pct", "{:.2f}%", optimizer=lm, parameter=p)
        s[f"{tag}.cal.err_q95.{p}"] = _csv(f"{r}/calibration_param_errors_test.csv", "q95_pct", "{:.1f}%", optimizer=lm, parameter=p)
    for short, name in (("lbfgsb", "L-BFGS-B"), ("slsqp", "SLSQP"), ("bfgs", "BFGS"), ("lm", lm), ("trf", "least_squares default (TRF)"),
                        ("cobyla", "COBYLA"), ("de", "Differential Evolution"), ("nm", "Nelder-Mead")):
        s[f"{tag}.cal.{short}_ms"] = _csv(f"{r}/calibration_summary_test.csv", "mean_ms", sig3, optimizer=name)
        s[f"{tag}.cal.{short}_median_ms"] = _csv(f"{r}/calibration_summary_test.csv", "median_ms", sig3, optimizer=name)
    return s


def _upgrade_sources() -> dict:
    """Numbers of the upgrade and critique experiments (rough Bergomi only)."""
    r, s = "results/rbergomi", {}
    for split in ("val", "test"):
        e = f"{r}/edge_of_box_{split}.csv"
        for name, region in (("rho_low", "lowest_5pct"), ("rho_inside", "inside"), ("rho_high", "highest_5pct")):
            s[f"edge.{split}.{name}"] = _csv(e, "mean_row_error_pct", "{:.2f}%", analysis="outer_band", parameter="rho", region=region)
        s[f"edge.{split}.rho_low_n"] = _csv(e, "n_rows", "{:,.0f}", analysis="outer_band", parameter="rho", region="lowest_5pct")
        s[f"edge.{split}.rho_focus"] = _csv(e, "mean_row_error_pct", "{:.2f}%", analysis="focus_region", parameter="rho", region="below_-0.9")
        s[f"edge.{split}.rho_rest"] = _csv(e, "mean_row_error_pct", "{:.2f}%", analysis="focus_region", parameter="rho", region="rest")
        s[f"edge.{split}.H_focus"] = _csv(e, "mean_row_error_pct", "{:.2f}%", analysis="focus_region", parameter="H", region="below_0.05")
        s[f"edge.{split}.H_rest"] = _csv(e, "mean_row_error_pct", "{:.2f}%", analysis="focus_region", parameter="H", region="rest")
        b = f"{r}/baselines_{split}.csv"
        for short, method in (("nn", "Neural network (ours, seed 0)"), ("ridge", "Ridge on polynomial features"),
                              ("knn", "k-nearest neighbours"), ("gb", "Gradient boosting")):
            s[f"base.{split}.{short}_err"] = _csv(b, "avg_rel_error_pct", "{:.3f}%", method=method)
            s[f"base.{split}.{short}_us"] = _csv(b, "predict_us_median", "{:,.0f}", method=method)
            s[f"base.{split}.{short}_max"] = _csv(b, "max_rel_error_pct", "{:.1f}%", method=method)
        s[f"base.{split}.ridge_fit_s"] = _csv(b, "fit_seconds", "{:.1f}", method="Ridge on polynomial features")
        s[f"base.{split}.ridge_gap"] = _csv(b, "avg_rel_gap_to_network_pct", "{:.2f}%", method="Ridge on polynomial features")
        for n in (5_000, 10_000, 20_000, 40_000, 60_000):
            s[f"eff.{split}.n{n}"] = _csv(f"{r}/sample_efficiency_{split}.csv", "avg_rel_error_pct", "{:.2f}%", agg="mean", n_train=n)
        w = f"{r}/worst_errors_{split}.csv"
        s[f"worst.{split}.n_rows"] = _csv(w, "n_rows", "{:,.0f}", analysis="worst_cell", maturity=0.1, strike=1.5)
        s[f"worst.{split}.at_k15"] = _csv(w, "n_worst_rows", "{:.0f}", analysis="worst_cell", maturity=0.1, strike=1.5)
        s[f"worst.{split}.at_k05"] = _csv(w, "n_worst_rows", "{:.0f}", analysis="worst_cell", maturity=0.1, strike=0.5)
        s[f"worst.{split}.threshold"] = _csv(w, "worst_row_threshold_pct", "{:.1f}%", analysis="worst_cell", maturity=0.1, strike=1.5)
    for quantity, fmt in (("offline_mc_cpu_hours", "{:.1f}"), ("break_even_calibrations", "{:,.0f}"), ("mc_seconds_per_calibration", "{:.0f}"),
                          ("mc_surfaces_per_calibration", "{:.0f}"), ("network_training_minutes", "{:.1f}")):
        s[f"cost.{quantity}"] = _csv(f"{r}/offline_cost.csv", "value", fmt, quantity=quantity)
    return s


def _stretch_sources() -> dict:
    """Numbers of the stretch experiments (rough Bergomi): paper-protocol run, ELU against ReLU, noisy quotes, the authors' calibrations."""
    r, s = "results/rbergomi", {}
    for short, protocol in (("paper", "paper: 68,000 rows, early stopping on the test set, last-epoch weights"),
                            ("ours", "ours: 60,000 rows, early stopping on 8,000 validation rows, best weights, test set untouched")):
        s[f"pp.{short}_avg"] = _csv(f"{r}/paper_protocol.csv", "avg_rel_error_pct", "{:.3f}%", protocol=protocol)
        s[f"pp.{short}_max"] = _csv(f"{r}/paper_protocol.csv", "max_rel_error_pct", "{:.1f}%", protocol=protocol)
    s["pp.paper_epochs"] = _csv(f"{r}/paper_protocol.csv", "epochs_run", "{:.0f}", protocol="paper: 68,000 rows, early stopping on the test set, last-epoch weights")
    for a in ("elu", "relu"):
        f = f"{r}/ablation_activation_val.csv"
        s[f"abl.{a}_err"] = _csv(f, "val_avg_rel_error_pct", "{:.3f}%", activation=a)
        s[f"abl.{a}_evals"] = _csv(f, "lm_mean_evaluations", "{:.1f}", activation=a)
        s[f"abl.{a}_rmse_q99"] = _csv(f, "rmse_q99_pct", "{:.2f}%", activation=a)
        s[f"abl.{a}_param"] = _csv(f, "mean_param_error_pct", "{:.1f}%", activation=a)
        s[f"abl.{a}_in_box"] = _csv(f, "share_in_box", "{:.0%}", activation=a)
    s["abl.n"] = _csv(f"{r}/ablation_activation_val.csv", "n_surfaces", "{:,.0f}", activation="elu")
    for p in ("xi1", "xi8", "nu", "rho", "H"):
        s[f"noise.{p}.clean"] = _csv(f"{r}/noise_robustness_val.csv", "clean_median_pct", "{:.2f}%", parameter=p)
        s[f"noise.{p}.noisy"] = _csv(f"{r}/noise_robustness_val.csv", "noisy_median_pct", "{:.2f}%", parameter=p)
    s["noise.n"] = _csv(f"{r}/noise_robustness_val.csv", "n_surfaces", "{:,.0f}", parameter="nu")
    f = f"{r}/authors_calibration_params_test.csv"
    for p in ("xi1", "xi8", "nu", "rho", "H", "all"):
        s[f"auth.par.{p}.authors"] = _csv(f, "authors_mean_pct", "{:.2f}%", parameter=p)
        s[f"auth.par.{p}.ours"] = _csv(f, "ours_lm_mean_pct", "{:.2f}%", parameter=p)
        s[f"auth.par.{p}.authors_median"] = _csv(f, "authors_median_pct", "{:.2f}%", parameter=p)
        s[f"auth.par.{p}.ours_median"] = _csv(f, "ours_lm_median_pct", "{:.2f}%", parameter=p)
        s[f"auth.par.{p}.gap"] = _csv(f, "gap_trf_median_pct", "{:.2f}%", parameter=p)
        s[f"auth.par.{p}.gap_mean"] = _csv(f, "gap_trf_mean_pct", "{:.2f}%", parameter=p)
    s["auth.n"] = _csv(f, "n_surfaces", "{:,.0f}", parameter="nu")
    g = f"{r}/authors_calibration_rmse_test.csv"
    for definition, short in (("paper_as_coded", "paper"), ("paper_as_coded_no_zero_vols", "paper_clean"), ("ours_lm", "ours"),
                              ("authors_params_our_network", "theirs_ournet")):
        for column, tag, fmt in (("q99_pct", "q99", "{:.2f}%"), ("median_pct", "median", "{:.3f}%"), ("max_pct", "max", "{:.2f}%"), ("n_surfaces", "n", "{:,.0f}")):
            s[f"auth.rmse.{short}_{tag}"] = _csv(g, column, fmt, definition=definition)
    s["auth.n_failed"] = _csv(g, "n_dropped", "{:.0f}", definition="paper_as_coded")
    s["auth.n_zero_rows"] = _csv(g, "n_rows_with_zero_vol", "{:.0f}", definition="paper_as_coded")
    h, as_run = "results/lbfgsb_memory_val.csv", "L-BFGS-B as run: bounds, memory 10"
    for model, tag in (("rbergomi", "rb"), ("onefactor", "of")):
        for short, setting in (("as_run", as_run), ("no_bounds", "L-BFGS-B without bounds, memory 10"),
                               ("memory20", "L-BFGS-B with bounds, memory 20"), ("bfgs", "BFGS: full curvature matrix")):
            s[f"lbm.{tag}.{short}"] = _csv(h, "mean_evaluations", "{:.0f}", model=model, setting=setting)
        s[f"lbm.{tag}.cond"] = _csv(h, "jacobian_condition_median", "{:.0f}", model=model, setting=as_run)
    s["lbm.n"] = _csv(h, "n_surfaces", "{:,.0f}", model="rbergomi", setting=as_run)
    return s


SOURCES = {**_model_sources("rbergomi", "rb"), **_model_sources("onefactor", "of"), **_upgrade_sources(), **_stretch_sources()}


def read_value(source: dict) -> float:
    """Fetch one number from its file as described by a SOURCES entry."""
    path = ROOT / source["file"]
    if "json_key" in source:
        with open(path) as f:
            node = json.load(f)
        for part in source["json_key"].split("."):
            node = node[part]
        return float(node)
    table = pd.read_csv(path)
    for column, wanted in source["where"].items():
        table = table[table[column].astype(str) == str(wanted)]
    values = table[source["column"]]
    if source["agg"] == "mean":
        return float(values.mean())
    assert len(values) == 1, (source, len(values))
    return float(values.iloc[0])


class Ledger:
    """Hands out formatted numbers to one document and keeps the record of what it handed out."""

    def __init__(self, document: str):
        self.document, self.ours, self.quoted = document, {}, {}

    def value(self, key: str) -> float:
        return read_value(SOURCES[key])

    def n(self, key: str, fmt: str | None = None) -> str:
        """One of OUR numbers as text, read from results/ right now."""
        source = SOURCES[key]
        value = read_value(source)
        fmt = fmt or source["format"]
        text = fmt(value) if callable(fmt) else fmt.format(value)
        entry = self.ours.setdefault(key, {"file": source["file"], "value": value, "texts": []})
        if text not in entry["texts"]:
            entry["texts"].append(text)
        return text

    def q(self, key: str, fmt: str = "{}", model: str | None = None, pick=None) -> str:
        """A number QUOTED from the paper (model=None) or from the authors' notebook; `pick` selects inside tuples or dicts."""
        value, source = (NOTEBOOK[model][key] if model else PAPER[key])
        if pick is not None:
            for step in (pick if isinstance(pick, (list, tuple)) else [pick]):
                value = value[step]
        text = fmt(value) if callable(fmt) else fmt.format(value)
        self.quoted.setdefault(f"{model or 'paper'}:{key}", {"source": source, "texts": []})["texts"].append(text)
        return text

    def qv(self, key: str, model: str | None = None, pick=None):
        """The raw quoted value (for comparisons), selected like q()."""
        value = (NOTEBOOK[model][key] if model else PAPER[key])[0]
        if pick is not None:
            for step in (pick if isinstance(pick, (list, tuple)) else [pick]):
                value = value[step]
        return value

    def save(self, path: Path = MANIFEST) -> None:
        """Merge this document's record into a manifest, by default results/numbers_manifest.json."""
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest[self.document] = {"ours": self.ours, "quoted": self.quoted}
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
