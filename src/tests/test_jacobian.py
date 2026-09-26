"""The hand-written NumPy network and its analytic Jacobian must agree with PyTorch and autograd."""
import numpy as np
import torch

from src.calibrate import forward, jacobian
from src.model import build_network, numpy_weights


def _random_network(seed: int):
    """A network with random weights and non-zero biases, so both ELU branches are exercised."""
    torch.manual_seed(seed)
    net = build_network()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return net, numpy_weights(net)


def test_numpy_forward_matches_pytorch():
    net, weights = _random_network(0)
    x = np.random.default_rng(0).uniform(-1, 1, size=(50, 11))
    with torch.no_grad():
        expected = net(torch.from_numpy(x)).numpy()
    assert np.abs(forward(weights, x) - expected).max() < 1e-12
    assert np.abs(forward(weights, x[0]) - expected[0]).max() < 1e-12


def test_analytic_jacobian_matches_autograd():
    net, weights = _random_network(1)
    worst = 0.0
    for x in np.random.default_rng(1).uniform(-1, 1, size=(25, 11)):
        by_autograd = torch.autograd.functional.jacobian(net, torch.from_numpy(x)).numpy()
        ours = jacobian(weights, x)
        assert ours.shape == (88, 11)
        worst = max(worst, np.abs(ours - by_autograd).max())
    assert worst < 1e-8, worst
