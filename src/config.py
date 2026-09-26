"""Project configuration: paths, the two datasets, the split and the training recipe.

Everything that differs between the rough Bergomi and the 1-factor Bergomi experiments
lives in DATASETS, so the second model is run by changing `--model`, not code.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_ROOT = Path(os.environ.get("DLV_OUTPUT_ROOT", ROOT))
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

REPO_URL = "https://github.com/amuguruza/NN-StochVol-Calibrations"
RAW_DATA_URL = "https://raw.githubusercontent.com/amuguruza/NN-StochVol-Calibrations/master/Data/"
OUR_REPO_URL = "https://github.com/zavaz5/deep-learning-volatility"

STRIKES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
MATURITIES = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
N_PARAMS = 11
N_VOLS = len(MATURITIES) * len(STRIKES)
N_ROWS = 80_000

_XI_NAMES = [f"xi{i}" for i in range(1, 9)]
_XI_LABELS = [rf"$\xi_{i}$" for i in range(1, 9)]

DATASETS = {
    "rbergomi": {
        "title": "Rough Bergomi",
        "file": "TrainrBergomiTermStructure.txt.gz",
        "sha256": "d19061aa9684dc93f50db89892a3a4445382938232768097bbe7cf8e5f987731",
        "authors_calibration": {"params": ("NNParametersrBergomiTermStructure.txt", "59d8816a57521583f88867472fed3bf930e8e38a6d927529c53aa0be640b2e00"),
                                "surfaces": ("surfacesFromNNRoughBergomiTermStructure.txt", "0d9d23f1c46f8a84c7a08c1f904c56fb9db37d00be29c845c938098c2b193a3e")},
        "param_names": _XI_NAMES + ["nu", "rho", "H"],
        "param_labels": _XI_LABELS + [r"$\nu$", r"$\rho$", r"$H$"],
        "lower": [0.01] * 8 + [0.5, -1.0, 0.025],
        "upper": [0.16] * 8 + [4.0, 0.0, 0.5],
        "vol_range_3dp": (0.034, 0.998),
        "row0_min_strike": 1.1,
        "paper_heatmap": "fig6_colour_limits_pct",
        "paper_error_cdf": "fig9_axis_top_pct",
        "edge_focus": [("rho", -0.9), ("H", 0.05)],
        "worst_error_axes": ("H", "nu"),
    },
    "onefactor": {
        "title": "1-factor Bergomi",
        "file": "Train1FactorTermStructure.txt.gz",
        "sha256": "46b63c41dd3ca9ced2f37cc5108cede5a9f703a4397e55dddadee0cc93c27485",
        "authors_calibration": {"params": ("NNParametersr1FactorTermStructure.txt", "b5dc2a6b190f7a9c421e48e62f164e98259fc3e353bb16e6b902a33d7bf644b1"),
                                "surfaces": ("SurfacesFromNN1FactorTermStructure.txt", "8da1a556ccd604b1d67d779618716c7f50d5b7648c05bf19040dc112768e3bdf")},
        "param_names": _XI_NAMES + ["nu", "beta", "rho"],
        "param_labels": _XI_LABELS + [r"$\nu$", r"$\beta$", r"$\rho$"],
        "lower": [0.01] * 8 + [0.5, 0.0, -0.95],
        "upper": [0.16] * 8 + [4.0, 10.0, -0.1],
        "vol_range_3dp": (0.041, 0.789),
        "row0_min_strike": 1.2,
        "paper_heatmap": "fig7_colour_limits_pct",
        "paper_error_cdf": "fig10_axis_top_pct",
        "edge_focus": [("rho", -0.9)],
        "worst_error_axes": ("beta", "nu"),
    },
}

SPLIT = {
    "test_size": 0.15,
    "authors_random_state": 42,
    "n_val": 8_000,
    "val_seed": 0,
}
SPLIT_FILE = RESULTS_DIR / "split_indices.npz"

RECIPE = {
    "n_hidden_layers": 4,
    "width": 30,
    "learning_rate": 1e-3,
    "adam_eps": 1e-7,
    "batch_size": 32,
    "max_epochs": 500,
    "patience": 25,
}
SEEDS = [0, 1, 2, 3, 4]

UPGRADE_MODEL = "rbergomi"

CALIBRATION = {
    "tol": 1e-10,
    "maxiter": 5_000,
    "gtol": 1e-10,
    "n_gradient_based": 5_000,
    "n_gradient_free": 100,
    "de_seed": 0,
}

QUICK = {
    "n_train": 1_600,
    "n_val": 400,
    "max_epochs": 5,
    "n_calibrations": 20,
    "n_calibrations_gradient_free": 3,
}


def results_dir(model: str, quick: bool = False) -> Path:
    """Folder for one model's result files. Quick runs are kept apart so they never overwrite real results."""
    base = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    path = base / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_root(quick: bool = False) -> Path:
    """Where figures go: results/figures, or results/quick/figures for a smoke test, so a quick run never overwrites a real figure."""
    return RESULTS_DIR / "quick" / "figures" if quick else FIGURES_DIR


def figures_dir(model: str, quick: bool = False) -> Path:
    """Folder for one model's figures, with the same quick/full separation as results_dir."""
    path = figures_root(quick) / model
    path.mkdir(parents=True, exist_ok=True)
    return path
