"""Tests for the split and the scalers (brief, section 8)."""
import numpy as np
import pytest

from src import data
from src.config import DATA_DIR, DATASETS, N_ROWS, SPLIT_FILE

AUTHORS_FIRST_TEST_ROWS = [47044, 44295, 74783, 70975, 46645, 8215, 65509, 62715]

needs_data = pytest.mark.skipif(
    not all((DATA_DIR / d["file"]).exists() for d in DATASETS.values()) or not SPLIT_FILE.exists(),
    reason="needs the data and a prepared split: python data/get_data.py && python -m src.data")


def test_split_has_the_right_sizes_and_no_overlap():
    split = data.build_split()
    assert [split[k].size for k in ("train_idx", "val_idx", "test_idx")] == [60_000, 8_000, 12_000]
    everything = np.concatenate(list(split.values()))
    assert np.array_equal(np.sort(everything), np.arange(N_ROWS))


def test_test_rows_are_the_authors_test_rows():
    assert data.build_split()["test_idx"][:8].tolist() == AUTHORS_FIRST_TEST_ROWS


@needs_data
def test_saved_split_equals_a_rebuild():
    rebuilt = data.build_split()
    with np.load(SPLIT_FILE) as saved:
        for name, idx in rebuilt.items():
            assert np.array_equal(saved[name], idx)


@pytest.mark.parametrize("model", list(DATASETS))
def test_parameter_scaling_maps_the_box_to_plus_minus_one(model):
    scalers = {"lower": np.array(DATASETS[model]["lower"]), "upper": np.array(DATASETS[model]["upper"])}
    assert np.allclose(data.scale_params(scalers["lower"], scalers), -1.0)
    assert np.allclose(data.scale_params(scalers["upper"], scalers), 1.0)
    theta = np.random.default_rng(0).uniform(scalers["lower"], scalers["upper"], size=(5, 11))
    assert np.allclose(data.unscale_params(data.scale_params(theta, scalers), scalers), theta)


@needs_data
@pytest.mark.parametrize("model", list(DATASETS))
def test_scalers_are_fitted_on_training_rows_only(model):
    rows = data.load_train_val(model)
    scalers = rows["scalers"]
    assert rows["vols_train"].shape == (60_000, 88)
    assert np.allclose(scalers["vol_mean"], rows["vols_train"].mean(0), rtol=0, atol=1e-14)
    assert np.allclose(scalers["vol_std"], rows["vols_train"].std(0), rtol=0, atol=1e-14)
    with_val = np.vstack([rows["vols_train"], rows["vols_val"]])
    assert np.abs(scalers["vol_mean"] - with_val.mean(0)).max() > 1e-8
    assert np.allclose(rows["y_train"].mean(0), 0.0, atol=1e-10)
    assert np.allclose(rows["y_train"].std(0), 1.0, atol=1e-10)
