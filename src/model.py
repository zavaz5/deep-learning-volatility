"""The approximation network: 11 parameters in, 88 implied vols out (paper Section 3.2.1).

Four hidden layers of 30 units with ELU activation and a linear output layer, in float64 like
the authors' Keras model. Inputs and outputs are the scaled quantities from src/data.py.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import N_PARAMS, N_VOLS, RECIPE

EXPECTED_WEIGHTS = 5_878


def build_network(activation: str = "elu") -> nn.Sequential:
    """Create the network with the authors' initialisation.

    Keras initialises a Dense layer with Glorot-uniform weights and zero biases. PyTorch's own
    default is different, so we set the Keras scheme explicitly; then the framework and the random
    seed are the only differences between our training run and theirs.
    `activation` exists for the ELU-versus-ReLU ablation; the reproduction always uses "elu".
    """
    make_activation = {"elu": nn.ELU, "relu": nn.ReLU}[activation]
    sizes = [N_PARAMS] + [RECIPE["width"]] * RECIPE["n_hidden_layers"]
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out, dtype=torch.float64), make_activation()]
    layers.append(nn.Linear(sizes[-1], N_VOLS, dtype=torch.float64))
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    if activation == "elu":
        assert count_weights(net) == EXPECTED_WEIGHTS
    return net


def count_weights(net: nn.Module) -> int:
    """Number of trainable parameters (weights and biases)."""
    return sum(p.numel() for p in net.parameters())


def numpy_weights(net: nn.Sequential) -> list[tuple[np.ndarray, np.ndarray]]:
    """The network as a list of (W, b) per layer, W of shape (n_out, n_in), so that a layer is W @ x + b."""
    return [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
            for layer in net if isinstance(layer, nn.Linear)]


def save_weights(net: nn.Sequential, path: Path) -> None:
    """Store the weights as plain NumPy arrays: small, readable without PyTorch, and what calibration uses."""
    arrays = {}
    for k, (W, b) in enumerate(numpy_weights(net)):
        arrays[f"W{k}"], arrays[f"b{k}"] = W, b
    np.savez(path, **arrays)


def load_weights(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read weights saved by save_weights, in layer order."""
    with np.load(path) as saved:
        return [(saved[f"W{k}"], saved[f"b{k}"]) for k in range(len(saved.files) // 2)]


def network_from_weights(weights: list[tuple[np.ndarray, np.ndarray]], activation: str = "elu") -> nn.Sequential:
    """Rebuild the PyTorch network from saved arrays (used to check the NumPy code against autograd)."""
    net = build_network(activation)
    linear_layers = [layer for layer in net if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, (W, b) in zip(linear_layers, weights):
            layer.weight.copy_(torch.from_numpy(W))
            layer.bias.copy_(torch.from_numpy(b))
    return net
