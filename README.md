# Deep Learning Volatility: a reproduction in PyTorch

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

| Item | Paper | Ours | Comment | Origin |
| --- | --- | --- | --- | --- |
| **Rough Bergomi: the network** |  |  |  |  |
| Weights in the network | 6,808 | 5,878 | The paper's formula counts four hidden-to-hidden maps; four hidden layers have three. The authors' own model summary prints 5,878. | Reproduction |
| Training rows | 68,000 | 60,000 | We set 8,000 of the paper's training rows aside for validation, so that the test set is never used for a decision. | **Our upgrade** |
| Epochs run, seed 0 | 200 in the text; the notebook stopped at 261 | 204 (best epoch 179) | Same patience of 25. We stop on validation rows and keep the best epoch; the authors stop on the test set and keep the last epoch. | Reproduction |
| Final loss, training / held-out (scaled RMSE, Keras convention) | 0.0257 / 0.0247 (notebook) | 0.0283 / 0.0265 | Held-out is the test set for the authors and the validation set for us. We train on fewer rows and stop earlier. | Reproduction |
| Average relative error on the test set | far below 0.5% | 0.459% ± 0.012% (mean ± std of 5 seeds) | Matches the claim. The paper reports one run whose stopping rule saw the test set; ours never did. | Reproduction |
| Largest cell average (top of Figure 6, left panel) | 1.02% | 1.14% | Same cell: maturity 0.1, strike 1.5. Seed 0. | Reproduction |
| Largest cell standard deviation (Figure 6, middle) | 2.04% | 2.14% | Seed 0. | Reproduction |
| Maximum relative error | 25% in the text, 52.3% on Figure 6's colour bar | 58.0% | The paper's text and its own figure disagree; ours is close to the figure. The error sits at maturity 0.1 and the extreme strikes. | Reproduction |
| Share of test cells with error above 1% (the paper's bid-ask yardstick) | not reported | 9.8% | The average hides a tail that misses the paper's own tolerance. | **Our critique** |
| **Rough Bergomi: speed of one full surface, milliseconds** |  |  |  |  |
| Network, NumPy forward pass | 0.0309 | 0.00919 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. Scaled input to scaled output, as in the authors' cell 22. | Reproduction |
| Network Jacobian (88 x 11) | 0.113 | 0.0246 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. Analytic chain rule in NumPy. | Reproduction |
| Deep-learning framework, single forward call | 0.469 (Keras predict, notebook) | 0.0146 (PyTorch) | Different frameworks; both are slower than plain NumPy for a single small vector. | Reproduction |
| Monte Carlo pricer | 500 | not reproduced | Needs a rough Bergomi simulator, which we did not write. We quote the paper's number and never present it as ours. | Not reproduced |
| **Rough Bergomi: mean calibration time, milliseconds (first 5,000 test surfaces; 100 for gradient-free optimizers)** |  |  |  |  |
| L-BFGS-B | 24.8 (read off Figure 8) | 5.33 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| SLSQP | 20.2 (read off Figure 8) | 1.7 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| BFGS | 35.4 (read off Figure 8) | 4.72 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| Levenberg-Marquardt | 6.81 (read off Figure 8) | 0.395 | The paper's bar was produced by SciPy's default `least_squares` method (TRF), not by `method='lm'`. That call takes 0.627 ms here. Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| COBYLA | 257 (read off Figure 8) | 1,924 | SciPy's COBYLA is now a pure-Python re-implementation, and the authors' gradient-free settings are unpublished. We assume their `tol` and `maxiter`; COBYLA then stops at the iteration cap. | Reproduction |
| Differential Evolution | 8,820 (read off Figure 8) | 400 | SciPy defaults, fixed seed. Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| Nelder-Mead | 480 (read off Figure 8) | 110 | Assumed settings, see COBYLA. Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| **Rough Bergomi: calibration accuracy, Levenberg-Marquardt** |  |  |  |  |
| Surface RMSE, 99% quantile | below 1.0% | 0.34% | Ours compares the network surface at the fitted parameters with the target, as the paper's formula says. The authors' notebook instead loads surfaces re-priced by an external pricer, which we cannot do without a simulator. | Reproduction |
| Mean relative error of xi1 | 0.99% (notebook, cell 34) | 1.16% |  | Reproduction |
| Mean relative error of xi8 | 15.48% (notebook, cell 34) | 15.77% | Later forward variances affect few grid cells, so they are weakly identified. | Reproduction |
| Mean relative error of nu | 0.76% (notebook, cell 34) | 0.82% |  | Reproduction |
| Mean relative error of rho | 5.75% (notebook, cell 34) | 5.28% | Relative error explodes when the true rho is near zero; the mean is driven by a few such rows. | Reproduction |
| Mean relative error of H | 2.45% (notebook, cell 34) | 2.48% |  | Reproduction |
| Fitted parameters inside the training box | not reported; no bounds in the notebook | 79.3% | Levenberg-Marquardt cannot take bounds. The rest of the fits land outside the region the network was trained on. | **Our critique** |
| **1-factor Bergomi (same code, `--model onefactor`)** |  |  |  |  |
| Average relative error on the test set | far below 0.5% | 0.290% ± 0.009% | Five seeds. | Reproduction |
| Largest cell average (top of Figure 7, left panel) | 0.65% | 0.68% | Seed 0. | Reproduction |
| Maximum relative error (Figure 7, right panel) | 17.3% | 15.4% | Seed 0. | Reproduction |
| Final loss, training / held-out | 0.0227 / 0.0233 (notebook) | 0.0235 / 0.0212 | See the rough Bergomi row. | Reproduction |
| Levenberg-Marquardt, mean milliseconds | 4.86 (read off Figure 8) | 0.391 | Same kind of measurement on different hardware: our 2026 laptop CPU against the authors' 2019 machine. | Reproduction |
| Surface RMSE, 99% quantile | below 1.0% | 0.22% | Network surface against target. | Reproduction |
| **Our upgrades (rough Bergomi, test set)** |  |  |  |  |
| Fixed flaw: locked test set, five seeds | one run, no untouched test set | 0.459% ± 0.012% | Validation gave 0.457%; training rows of seed 0 give 0.458%. No sign of classic overfitting. | **Our upgrade** |
| Edge of the box: rho in the lowest 5% of its range against the middle 90% | not tested | 1.25% against 0.44% | Only 2.45% of training rows lie in that band (uniform sampling would give 5%), and the paper's SPX fits sit there (its Figure 11). | **Our upgrade** |
| Missing baseline: ridge regression on polynomial features | no baseline | 0.277% at 0.109 ms per surface; the network: 0.465% at 0.00996 ms | The polynomial fits in 3.5 s. Gradient boosting reaches 0.983%, k-nearest neighbours 4.788%. | **Our upgrade** |
| Sample efficiency: 5,000 / 20,000 / 40,000 / 60,000 training rows | one size, 68,000 | 0.71% / 0.53% / 0.46% / 0.46% | Mean of three seeds. The error stops improving after 40,000 rows; the polynomial baseline shows this is the small network's limit, not noise in the data. | **Our upgrade** |
| Offline cost left out of the paper's speed-up | speed-up of 9,000 to 16,000 | 11.1 CPU-hours; break-even after about 888 calibrations | Monte Carlo times are the paper's. Assumes 12 surfaces per Levenberg-Marquardt evaluation when the Jacobian comes from finite differences. | **Our critique** |
| **Stretch experiments (rough Bergomi)** |  |  |  |  |
| Training the authors' way: 68,000 rows, early stopping on the test set, last-epoch weights | their protocol | 0.437% on the test set, against 0.465% with our protocol (seed 0) | The gain mixes 8,000 more training rows with a stopping rule that saw the test set; one run cannot separate them. The flaw we fixed does not change the conclusion. | **Our stretch** |
| ELU against ReLU (validation rows) | ELU chosen, no experiment | error 0.461% against 0.758%; mean parameter error after calibration 4.9% against 16.4% | Same data, seed and recipe; 1,000 surfaces, each network with its exact Jacobian. Supports the paper's Remarks 1 and 6. | **Our stretch** |
| Noisy quotes: 0.5% relative noise, median parameter error (validation rows) | not tested | ν 0.54% to 0.65%; ξ₈ 5.67% to 10.43% | Well-identified parameters barely move; late forward variances nearly double. | **Our stretch** |
| Authors' published calibrations of the same 5,000 test surfaces against ours | not compared in the paper | median gap between the two fits 1.80% of the true value (ν 0.60%, ρ 1.32%, H 1.87%); their mean error on ξ₈ recomputed from their file 15.48% | Same surfaces, same SciPy call. Recomputing their errors from their published file gives their notebook's numbers (ξ₈: 15.48%), so the rows line up. The gap between the two fits is about as large as each fit's median error against the truth (1.30% and 1.46%): two independently trained networks lead the same optimizer to different parameter vectors of the same quality, a sign of weak identification. | **Our stretch** |
| Paper's Figure 9 RMSE as coded, 99% quantile and largest value: their re-priced surfaces against the target, root of the mean | below 1.0% (Figure 9) | 0.35% and 2.71%; without the 13 surfaces holding a zero implied vol: 0.33% and 0.77%; ours: 0.34% and 1.00% | Recomputed from the authors' published file, 4,941 of the 5,000 rows kept as in their notebook (59 re-pricings failed). The tail of their Figure 9 comes from surfaces with a failed inversion, a zero implied vol, that the notebook keeps; without them the two definitions agree. Their parameters through our network give 0.43%: each network's own fit is the best fit on that network. | **Our stretch** |

## Run

Python 3.13.15, CPU only, versions pinned in `requirements.txt`. On Linux or Windows install the CPU build of PyTorch first:
`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.

```bash
pip install -r requirements.txt
python data/get_data.py         # checks the included data; downloads a file only if it is missing
python -m src.run_all --full    # every result and figure, about 38 minutes here
python -m pytest                # the tests
```

`data/` holds six files copied unchanged from [amuguruza/NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations), with the authors' MIT licence (`data/LICENSE`): the two training
sets and four small files of the authors' published calibrations. `--full` rewrites `results/`: accuracy numbers come out identical, timings become those of your machine
(`DLV_OUTPUT_ROOT=/some/folder` writes elsewhere). `--quick` checks every script on a subset in a few minutes and writes only to
`results/quick/`. The notebook (`jupyter notebook DeepLearningVolatility.ipynb`, then *Run all*, 57 s here) never reads test rows.

Split and seeds: the authors' `train_test_split(test_size=0.15, random_state=42)` on the row order of the file, so our 12,000
test rows are theirs; 8,000 of their 68,000 training rows form a validation set (`numpy.random.default_rng(0)`), leaving 60,000 / 8,000 / 12,000
(`results/split_indices.npz`). Early stopping and every decision used validation rows. Only `src/final_eval.py` reads test rows; the main
evaluation ran once, after the code and the networks were frozen. Seeds 0, 1, 2, 3, 4 (`torch.manual_seed`, NumPy and Python seeds, deterministic
algorithms, one thread, float64); differential evolution uses seed 0. Reported numbers: macOS 26.6.2, Apple M5 Pro.

## References

- Horvath, B., Muguruza, A., Tomas, M. (2021). Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models. *Quantitative Finance* 21(1), 11-27. DOI 10.1080/14697688.2020.1817974. arXiv:1901.09647v2. Code and data: [amuguruza/NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations), MIT licence.
- Bayer, C., Friz, P., Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance* 16(6), 887-904.
- Hernandez, A. (2017). Model calibration with neural networks. *Risk*, June 2017.

Method and data are the authors'. No code here was copied from their notebooks, and their trained weights were never loaded.
