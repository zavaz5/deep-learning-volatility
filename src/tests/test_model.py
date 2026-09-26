"""Tests for the network and the training loop. They use synthetic data only, so they need no download."""
import numpy as np
import torch

from src.model import (EXPECTED_WEIGHTS, build_network, count_weights, load_weights, network_from_weights,
                       numpy_weights, save_weights)
from src.train import fit, keras_style_rmse, rmse


def test_weight_count_is_5878():
    assert EXPECTED_WEIGHTS == 5_878
    assert count_weights(build_network()) == 5_878


def test_network_is_float64_with_keras_initialisation():
    torch.manual_seed(0)
    for W, b in numpy_weights(build_network()):
        assert W.dtype == np.float64
        assert not b.any()
        limit = np.sqrt(6.0 / (W.shape[0] + W.shape[1]))
        assert np.abs(W).max() <= limit and np.abs(W).max() > 0.8 * limit


def test_weights_survive_a_round_trip_to_disk(tmp_path):
    torch.manual_seed(1)
    net = build_network()
    save_weights(net, tmp_path / "w.npz")
    rebuilt = network_from_weights(load_weights(tmp_path / "w.npz"))
    x = torch.rand(7, 11, dtype=torch.float64)
    assert torch.equal(net(x), rebuilt(x))


def test_keras_style_loss_is_the_weighted_mean_of_batch_losses():
    torch.manual_seed(2)
    p, t = torch.randn(70, 88, dtype=torch.float64), torch.randn(70, 88, dtype=torch.float64)
    by_hand = (32 * rmse(p[:32], t[:32]) + 32 * rmse(p[32:64], t[32:64]) + 6 * rmse(p[64:], t[64:])) / 70
    assert abs(keras_style_rmse(p, t, 32) - by_hand.item()) < 1e-15
    assert keras_style_rmse(p, t, 32) <= rmse(p, t).item()


def _synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 11))
    return x, np.tanh(x @ rng.normal(size=(11, 88)))


def test_training_is_repeatable_and_restores_the_best_epoch():
    x, y = _synthetic(256, 0)
    xv, yv = _synthetic(64, 1)
    net_a, hist_a, sum_a = fit(x, y, xv, yv, seed=3, max_epochs=4, verbose=False)
    net_b, hist_b, sum_b = fit(x, y, xv, yv, seed=3, max_epochs=4, verbose=False)
    net_c, _, _ = fit(x, y, xv, yv, seed=4, max_epochs=4, verbose=False)
    for (Wa, _), (Wb, _), (Wc, _) in zip(numpy_weights(net_a), numpy_weights(net_b), numpy_weights(net_c)):
        assert np.array_equal(Wa, Wb)
        assert not np.array_equal(Wa, Wc)
    assert hist_a.equals(hist_b)
    assert sum_a["best_val_loss_keras"] == hist_a["val_loss_keras"].min()
    with torch.no_grad():
        restored = keras_style_rmse(net_a(torch.as_tensor(xv)), torch.as_tensor(yv), 32)
    assert abs(restored - sum_a["best_val_loss_keras"]) < 1e-15
