"""Training with early stopping on the validation set.

Recipe, as in the authors' notebook: Adam (learning rate 1e-3, epsilon 1e-7), batch size 32,
root-mean-square loss on scaled outputs, patience 25, at most 500 epochs. Two deliberate
differences from their notebook: early stopping watches our validation rows, not the test
set, and we keep the best-validation weights instead of the last epoch's.

Usage:  python -m src.train --model rbergomi --seed 0 [--quick]
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from src.config import DATASETS, QUICK, RECIPE, results_dir
from src.data import load_train_val
from src.model import build_network, save_weights


def set_seeds(seed: int) -> None:
    """Make a run repeatable: same seed, same weights, on the same machine and library versions.

    One thread keeps floating-point sums in a fixed order; with this small network a single
    thread is also the fastest option, so nothing is lost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error over all rows and all 88 outputs: the training loss of one batch."""
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def keras_style_rmse(prediction: torch.Tensor, target: torch.Tensor, batch_size: int) -> float:
    """Mean of per-batch RMSE values, weighted by batch size: what Keras prints as `val_loss`.

    It is a little smaller than the whole-set RMSE (the mean of roots is below the root of the
    mean). We stop early on this quantity and log it so that our curves compare directly with
    the losses in the authors' training log.
    """
    total = 0.0
    for p, t in zip(prediction.split(batch_size), target.split(batch_size)):
        total += rmse(p, t).item() * len(p)
    return total / len(prediction)


def fit(x_train, y_train, x_val, y_val, seed: int, max_epochs: int, activation: str = "elu", verbose: bool = True,
        restore_best: bool = True):
    """Train one network and return (network with best-validation weights, per-epoch history, summary).

    The arrays are already scaled. Keeping this a plain function of arrays lets the experiments
    (sample efficiency, ablations) reuse exactly the same loop on other row subsets.
    restore_best=False keeps the last epoch's weights, which is what the authors' Keras run did;
    it is used only for the paper-protocol comparison.
    """
    set_seeds(seed)
    x_train, y_train, x_val, y_val = (torch.as_tensor(a, dtype=torch.float64) for a in (x_train, y_train, x_val, y_val))
    net = build_network(activation)
    optimizer = torch.optim.Adam(net.parameters(), lr=RECIPE["learning_rate"], eps=RECIPE["adam_eps"])
    batch_size, patience = RECIPE["batch_size"], RECIPE["patience"]
    shuffle = torch.Generator().manual_seed(seed)

    history, best_loss, best_state, best_epoch, waited = [], float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(len(x_train), generator=shuffle)
        x_shuffled, y_shuffled = x_train[order], y_train[order]
        running = 0.0
        for first in range(0, len(x_train), batch_size):
            x_batch, y_batch = x_shuffled[first:first + batch_size], y_shuffled[first:first + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = rmse(net(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x_batch)

        with torch.no_grad():
            val_prediction = net(x_val)
            row = {"epoch": epoch,
                   "train_loss_running": running / len(x_train),
                   "train_rmse": rmse(net(x_train), y_train).item(),
                   "val_loss_keras": keras_style_rmse(val_prediction, y_val, batch_size),
                   "val_rmse": rmse(val_prediction, y_val).item()}
        history.append(row)

        if row["val_loss_keras"] < best_loss:
            best_loss, best_epoch, waited = row["val_loss_keras"], epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            waited += 1
        if verbose:
            print(f"epoch {epoch:3d}  train {row['train_loss_running']:.4f}  val {row['val_loss_keras']:.4f}  "
                  f"best {best_loss:.4f} @ {best_epoch}", flush=True)
        if waited >= patience:
            break

    if restore_best:
        net.load_state_dict(best_state)
    summary = {"seed": seed, "activation": activation, "n_train": len(x_train), "n_val": len(x_val),
               "epochs_run": epoch, "best_epoch": best_epoch, "stopped_early": waited >= patience,
               "weights_kept": "best epoch" if restore_best else "last epoch",
               "best_val_loss_keras": best_loss, "best_val_rmse": history[best_epoch - 1]["val_rmse"],
               "train_loss_running_at_best": history[best_epoch - 1]["train_loss_running"],
               "train_rmse_at_best": history[best_epoch - 1]["train_rmse"],
               "seconds": time.perf_counter() - start}
    return net, pd.DataFrame(history), summary


def train_and_save(model: str, seed: int, quick: bool = False) -> dict:
    """Train on the prepared split and write weights, history and summary to results/<model>/."""
    data = load_train_val(model, quick)
    max_epochs = QUICK["max_epochs"] if quick else RECIPE["max_epochs"]
    net, history, summary = fit(data["x_train"], data["y_train"], data["x_val"], data["y_val"], seed, max_epochs)
    out = results_dir(model, quick)
    save_weights(net, out / f"weights_seed{seed}.npz")
    history.to_csv(out / f"history_seed{seed}.csv", index=False)
    with open(out / f"train_summary_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(DATASETS), default="rbergomi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = train_and_save(args.model, args.seed, args.quick)
    print(json.dumps(result, indent=2))
