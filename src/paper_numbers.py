"""Every number we quote from the paper or from the authors' notebooks, typed once, with its source.

Paper: Horvath, Muguruza, Tomas, "Deep Learning Volatility", arXiv:1901.09647v2 (22 August 2019).
Page, figure, table and section numbers refer to that PDF.
Notebooks: https://github.com/amuguruza/NN-StochVol-Calibrations at commit
832690651e628481978b034f5ca08e090267bc4a (10 May 2020). Cell numbers count from 0.

Each entry is (value, source). Our own results never appear here; they live in results/.
"""

AUTHORS_COMMIT = "832690651e628481978b034f5ca08e090267bc4a"
NOTEBOOK_RBERGOMI = "RoughBergomi/Piecewise Forward Variance/NNPiecewisexi0rBergomi.ipynb"
NOTEBOOK_ONEFACTOR = "1Factor/Piecewise Forward Variance/NNPiecewisexi01Factor.ipynb"

PAPER = {
    "n_samples": (80_000, "Section 2.4.1, p. 10"),
    "n_train": (68_000, "Section 4.1.1, p. 20"),
    "n_test": (12_000, "Section 4.1.1, p. 20"),
    "mc_paths": (60_000, "Section 4.1, p. 18, and Section 4.1.1, p. 20"),
    "n_weights_stated": (6_808, "Section 4.1.1, p. 20; the formula 30n + 6478 is in Section 3.2.1, p. 15, item 4"),
    "batch_size": (32, "Section 3.2.2, p. 16"),
    "epochs": (200, "Section 3.2.2, p. 16, item 1"),
    "patience": (25, "Section 3.2.2, p. 16, item 1"),
    "rho_range": ((-0.95, -0.1), "Section 4.1.1, p. 20, sampling box of both models"),
    "rho_range_spx": ((-1.0, 0.0), "Section 4.2.2, p. 24, box used for the SPX calibration"),
    "onefactor_param_order": (("nu", "rho", "beta"), "Section 4.1.1, p. 20, order in which the sampling box is listed"),
    "sampling_box_rbergomi": ({"xi": (0.01, 0.16), "nu": (0.5, 4.0), "rho": (-0.95, -0.1), "H": (0.025, 0.5)},
                              "Section 4.1.1, p. 20, uniform sampling box of rough Bergomi (each of the 8 forward variances in the xi range)"),

    "avg_rel_error_bound_pct": (0.5, "Section 4.1.1, p. 21: average relative error far below 0.5%"),
    "std_rel_error_bound_pct": (1.0, "Section 4.1.1, p. 21: standard deviation below 1%"),
    "max_rel_error_text_pct": (25.0, "Section 4.1.1, p. 21: maximum relative error up to 25%"),
    "bid_ask_rel_pct": (1.0, "Section 4.1, p. 18: a 0.2% vol spread is 1% relative error at 20% vol"),
    "bid_ask_vol_pct": (0.2, "Section 4.1, p. 18: spread of the most liquid options below a year, in implied-vol terms"),
    "bid_ask_reference_vol_pct": (20.0, "Section 4.1, p. 18: the implied vol at which the 0.2% spread is 1% relative error"),
    "fig6_colour_limits_pct": ({"mean": (0.183, 1.020), "std": (0.166, 2.036), "max": (1.78, 52.27)},
                               "Figure 6, p. 20 (rough Bergomi)"),
    "fig7_colour_limits_pct": ({"mean": (0.138, 0.652), "std": (0.150, 0.852), "max": (2.27, 17.27)},
                               "Figure 7, p. 21 (1-factor Bergomi)"),

    "nn_price_us": (30.9, "Table 2, p. 18"),
    "nn_gradient_us": (113.0, "Table 2, p. 18"),
    "mc_rbergomi_us": (500_000.0, "Table 2, p. 18"),
    "mc_onefactor_us": (300_000.0, "Table 2, p. 18"),
    "speedup_range": ((9_000, 16_000), "Table 2, p. 18"),

    "rmse_q99_bound_pct": (1.0, "Section 4.2.1, p. 22: the 99% quantile of the RMSE is below 1%"),
    "fig8_mean_ms": ({"rbergomi": {"L-BFGS-B": 24.8, "SLSQP": 20.2, "BFGS": 35.4, "Levenberg-Marquardt": 6.81,
                                   "COBYLA": 257.0, "Differential Evolution": 8820.0, "Nelder-Mead": 480.0},
                      "onefactor": {"L-BFGS-B": 29.7, "SLSQP": 17.4, "BFGS": 32.5, "Levenberg-Marquardt": 4.86,
                                    "COBYLA": 240.0, "Differential Evolution": 9440.0, "Nelder-Mead": 385.0}},
                     "Figure 8, p. 22, piecewise forward variance bars"),
    "fig9_axis_top_pct": ({"param_error": 180.0, "rmse": 2.8}, "Figure 9, p. 23 (rough Bergomi)"),
    "fig10_axis_top_pct": ({"param_error": 45.0, "rmse": 4.2}, "Figure 10, p. 23 (1-factor Bergomi)"),

    "spx_grid": ((5, 9), "Section 4.2.2, p. 24: 5 maturities (1 to 12 months) by 9 strikes (0.85 to 1.25)"),
    "spx_H_range": ((0.1, 0.15), "Section 4.2.2, p. 24: calibrated H under Q, in line with Gatheral, Jaisson and Rosenbaum under P"),
    "spx_nn_vs_mc_gap_pct": (0.2, "Section 4.2.2, p. 24: network fit against brute-force Monte Carlo fit, below this most of the time"),
    "barrier_abs_error_bps": (10, "Section 4.3, p. 26: average absolute error below 10 bps, standard deviation 10 bps"),
}

NOTEBOOK = {
    "rbergomi": {
        "layer_weights": ((360, 930, 930, 930, 2728), "cell 11, model summary"),
        "n_weights": (5_878, "cell 11, model summary"),
        "max_epochs": (500, "cell 13"),
        "patience": (25, "cell 13"),
        "stopped_epoch": (261, "cell 13, training log"),
        "final_train_loss": (0.0257, "cell 13, epoch 261"),
        "final_val_loss": (0.0247, "cell 13, epoch 261; the validation data is the test set"),
        "best_val_loss": (0.0240, "cell 13, epoch 236"),
        "best_val_epoch": (236, "cell 13, training log"),
        "seconds_per_epoch": ((3, 5), "cell 13, training log"),
        "numpy_forward_us": (30.9, "cell 22, %timeit of the hand-written NumPy forward pass"),
        "keras_predict_us": (469.0, "cell 23, %timeit of Keras predict"),
        "n_calibrated": (5_000, "cell 30"),
        "n_rmse_rows_kept": (4_941, "cell 38, after dropping rows with infinite RMSE"),
        "n_surface_rows_with_failed_vols": (59, "Data/surfacesFromNNRoughBergomiTermStructure.txt, rows containing -1.79769e+308"),
        "lm_mean_rel_error": ((0.009909834958195486, 0.01384682637015444, 0.016822720581008067, 0.02616733990257942,
                               0.04112297458963902, 0.06004062805731214, 0.08570431790234273, 0.15477554127991433,
                               0.007638141111342179, 0.057501922204466156, 0.024513397879373542),
                              "cell 34, printed output, parameter order xi1..xi8, nu, rho, H"),
    },
    "onefactor": {
        "n_weights": (5_878, "cell 11, model summary"),
        "stopped_epoch": (189, "cell 13, training log"),
        "final_train_loss": (0.0227, "cell 13, epoch 189"),
        "final_val_loss": (0.0233, "cell 13, epoch 189; the validation data is the test set"),
        "best_val_loss": (0.0210, "cell 13, epoch 164"),
        "best_val_epoch": (164, "cell 13, training log"),
        "numpy_forward_us": (29.7, "cell 21"),
        "keras_predict_us": (488.0, "cell 22"),
        "lm_mean_rel_error": ((0.005084297152863123, 0.009724202081734015, 0.015359399581538173, 0.016260951479616276,
                               0.02481163464319792, 0.027892864086716724, 0.03536495984622075, 0.03947751691516167,
                               0.0176508692782693, 0.08928283646601017, 0.01985215998781099),
                              "cell 33, printed output, parameter order xi1..xi8, nu, beta, rho"),
    },
    "optimizer_tol": (1e-10, "calibration cell, tol for scipy.optimize.minimize"),
    "optimizer_maxiter": (5_000, "calibration cell, maxiter for scipy.optimize.minimize"),
    "least_squares_gtol": (1e-10, "calibration cell, gtol for scipy.optimize.least_squares"),
}


def paper(key: str):
    """Value of a number quoted from the paper."""
    return PAPER[key][0]


def notebook(model: str, key: str):
    """Value of a number quoted from one of the authors' notebooks."""
    return NOTEBOOK[model][key][0]
