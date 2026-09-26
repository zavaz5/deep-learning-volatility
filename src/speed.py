"""Speed of one full-surface evaluation and of its Jacobian (paper Table 2).

Run on an otherwise idle machine. Each measurement warms up first, then repeats a loop of calls
seven times with time.perf_counter, the scheme of IPython's %timeit that the authors used. We
report the median over the seven repeats with the fastest and slowest repeat as spread, plus
mean and standard deviation for a like-for-like reading against their "30.9 us +- 2.34 us".

The Monte Carlo column of Table 2 cannot be reproduced without a simulator; it is quoted, not measured.

Usage:  python -m src.speed --model rbergomi [--quick]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.calibrate import forward, jacobian
from src.config import DATASETS, results_dir
from src.data import load_train_val, scale_params, unscale_vols
from src.model import load_weights, network_from_weights

REPEATS = 7


def time_call(call, n_loops: int) -> dict:
    """Microseconds per call: median, fastest and slowest of REPEATS loops, plus mean and std."""
    for _ in range(max(200, n_loops // 5)):
        call()
    per_call = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(n_loops):
            call()
        per_call.append((time.perf_counter() - start) / n_loops * 1e6)
    return {"median_us": np.median(per_call), "fastest_us": min(per_call), "slowest_us": max(per_call),
            "mean_us": np.mean(per_call), "std_us": np.std(per_call), "repeats": REPEATS, "loops_per_repeat": n_loops}


def measure(model: str, quick: bool = False, seed: int = 0) -> pd.DataFrame:
    """Time the network for one parameter vector: the first validation row."""
    torch.set_num_threads(1)
    data, folder = load_train_val(model, quick), results_dir(model, quick)
    weights = load_weights(folder / f"weights_seed{seed}.npz")
    net, scalers = network_from_weights(weights), data["scalers"]
    theta, x = data["theta_val"][0], data["x_val"][0]
    x_torch = torch.from_numpy(x).reshape(1, -1)
    n = 1_000 if quick else 10_000

    def torch_forward():
        with torch.no_grad():
            return net(x_torch)

    cases = [
        ("numpy_forward_scaled", "NumPy forward pass, scaled 11-vector in, scaled 88-vector out (what the authors' 30.9 us measures)",
         lambda: forward(weights, x), n),
        ("numpy_forward_full", "NumPy: raw parameters -> scaling -> forward pass -> implied vols",
         lambda: unscale_vols(forward(weights, scale_params(theta, scalers)), scalers), n),
        ("torch_forward", "PyTorch forward pass under no_grad, one thread, input already a 1 x 11 tensor (counterpart of Keras predict)",
         torch_forward, n),
        ("numpy_jacobian", "NumPy analytic 88 x 11 Jacobian, scaled space (counterpart of Table 2's NN gradient)",
         lambda: jacobian(weights, x), n),
        ("torch_autograd_jacobian", "torch.autograd.functional.jacobian of the same network, one thread",
         lambda: torch.autograd.functional.jacobian(net, x_torch[0]), n // 20),
    ]
    rows = [{"what": name, **time_call(call, loops), "timed": description} for name, description, call, loops in cases]
    table = pd.DataFrame(rows)
    table.to_csv(folder / "speed.csv", index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(measure(args.model, args.quick).drop(columns="timed").round(2).to_string(index=False))
