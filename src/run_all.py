"""Run the whole project, step by step, exactly as the README documents each step.

  python -m src.run_all --quick   smoke test: 2,000 rows, 5 epochs, 20 calibrations. Finishes in a few
                                  minutes and regenerates every figure at low fidelity under
                                  results/quick/. Real results are never overwritten.
  python -m src.run_all --full    everything at full size (about 40 minutes on our machine). Training is
                                  deterministic, so the numbers reproduce; timings depend on the machine.

Each step is a separate process (`python -m <module> ...`). Seeds train in parallel, one thread each;
everything that measures time runs afterwards, one step at a time, on an otherwise idle machine.
The last step, src.final_eval, is the only one that reads the locked test rows.
"""
import argparse
import subprocess
import sys
import time

import pandas as pd

from src.config import DATASETS, RESULTS_DIR, SEEDS, UPGRADE_MODEL

EXPERIMENTS = ["seeds", "edge_of_box", "worst_errors", "offline_cost", "sample_efficiency", "baselines",
               "ablation_activation", "noise_robustness"]


RUNTIMES = []


def run(module: str, *args: str) -> None:
    """Run one step, record how long it took, and stop the whole pipeline if it fails."""
    command = [sys.executable, "-m", module, *args]
    print(f"\n=== {' '.join(command[1:])}", flush=True)
    start = time.perf_counter()
    subprocess.run(command, check=True)
    seconds = time.perf_counter() - start
    RUNTIMES.append({"step": module.split(".")[-1] + " " + " ".join(a for a in args if not a.startswith("--")),
                     "command": "python " + " ".join(command[1:]), "seconds": seconds})
    print(f"=== done in {seconds:.0f} s", flush=True)


def train_seeds(model: str, seeds: list[int], flags: list[str]) -> None:
    """Train all seeds at once, each in its own single-threaded process."""
    print(f"\n=== training {model}, seeds {seeds}, in parallel", flush=True)
    start = time.perf_counter()
    processes = [subprocess.Popen([sys.executable, "-m", "src.train", "--model", model, "--seed", str(s), *flags],
                                  stdout=subprocess.DEVNULL) for s in seeds]
    if any(p.wait() != 0 for p in processes):
        raise SystemExit(f"training failed for {model}")
    RUNTIMES.append({"step": f"train {model}, seeds {seeds[0]}-{seeds[-1]} in parallel",
                     "command": f"python -m src.train --model {model} --seed <s>" + (" --quick" if flags else ""),
                     "seconds": time.perf_counter() - start})


def main(quick: bool) -> None:
    flags = ["--quick"] if quick else []
    start = time.perf_counter()
    for model in DATASETS:
        run("src.data", "--model", model, *flags)
        train_seeds(model, [0] if quick else SEEDS, flags)
        for module in ("src.evaluate", "src.speed", "src.calibrate"):
            run(module, "--model", model, *flags)
        run("src.plots", "--model", model, "--figure", "all", *flags)
    for name in EXPERIMENTS:
        run(f"src.experiments.{name}", "--model", UPGRADE_MODEL, *flags)
    run("src.experiments.lbfgsb_memory", *flags)
    for model in DATASETS:
        run("src.final_eval", "--model", model, *flags)
    run("src.final_eval", "--model", UPGRADE_MODEL, "--paper-protocol", *flags)
    for model in DATASETS:
        run("src.final_eval", "--model", model, "--authors-calibration", *flags)
    out = RESULTS_DIR / "quick" / "runtimes.csv" if quick else RESULTS_DIR / "runtimes_full.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RUNTIMES).to_csv(out, index=False)
    print(f"\nAll steps finished in {(time.perf_counter() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="smoke test on 2,000 rows")
    mode.add_argument("--full", action="store_true", help="full-size run")
    args = parser.parse_args()
    main(quick=args.quick)
