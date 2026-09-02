# Phase-1 Volatility Prediction Comparison

The Phase-1 framework, Raw LSTM, and adapted GARCH–LSTM stack form a strict comparison: they use the same processed split, saved realized-volatility label bundle, seed, 15/50/100 budgets, and exactly identical 27,450 test targets. The stored Raw-OHLCV MLP is context only because it predates the shared label contract and evaluates 27,499 differently constructed targets.

## Results

| Epoch | Model | MAE | RMSE | MSE | Correlation |
|---:|---|---:|---:|---:|---:|
| 15 | Phase-1 framework | 0.040595 | 0.089076 | 0.007935 | 0.767419 |
| 15 | Raw LSTM | 0.051506 | 0.108625 | 0.011799 | 0.689854 |
| 15 | GARCH–LSTM stack | 0.032956 | 0.089026 | 0.007926 | 0.805865 |
| 50 | Phase-1 framework | 0.036816 | 0.086553 | 0.007491 | 0.769867 |
| 50 | Raw LSTM | 0.047948 | 0.098107 | 0.009625 | 0.676967 |
| 50 | GARCH–LSTM stack | 0.032673 | 0.086506 | 0.007483 | 0.810017 |
| 100 | Phase-1 framework | 0.035934 | 0.087684 | 0.007688 | 0.763630 |
| 100 | Raw LSTM | 0.047043 | 0.103105 | 0.010631 | 0.639976 |
| 100 | GARCH–LSTM stack | 0.030416 | 0.083372 | 0.006951 | 0.821897 |

## Relative Phase-1 performance

Positive error change means Phase-1 has lower error than the named baseline; negative means higher error.

| Epoch | MAE vs Raw LSTM | RMSE vs Raw LSTM | MSE vs Raw LSTM | MAE vs stack | RMSE vs stack | MSE vs stack | Corr difference vs Raw LSTM | Corr difference vs stack |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | +21.2% | +18.0% | +32.8% | -23.2% | -0.1% | -0.1% | +0.0776 | -0.0384 |
| 50 | +23.2% | +11.8% | +22.2% | -12.7% | -0.1% | -0.1% | +0.0929 | -0.0401 |
| 100 | +23.6% | +15.0% | +27.7% | -18.1% | -5.2% | -10.6% | +0.1237 | -0.0583 |

## Recorded target and prediction behavior

The shared target distribution shifts materially from training to test: mean realized volatility is `0.030779` on training rows and `0.080484` on test rows. The test target standard deviation is approximately `0.132715`, with values from `0.0` to `0.885255`. This temporal shift makes generalization harder and helps explain why test error remains much larger than final training loss.

| Epoch | Phase-1 prediction mean | Prediction std | Prediction min | Prediction median | Prediction max | Negative fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.066128 | 0.079756 | -0.105829 | 0.031917 | 0.578150 | 8.4% |
| 50 | 0.069547 | 0.088107 | -0.092067 | 0.027610 | 0.598833 | 5.8% |
| 100 | 0.067408 | 0.088118 | -0.149035 | 0.025532 | 0.576886 | 7.0% |

## Analysis

Phase-1 consistently outperforms the direct Raw LSTM benchmark on all three error metrics and correlation at every matched budget. Its MAE reduction is approximately 21–24%, while RMSE falls by approximately 12–18%. The result supports the claim that the frozen multi-branch representation provides useful volatility information beyond the end-to-end Raw LSTM under this shared task contract.

The adapted GARCH–LSTM stack remains stronger overall. At 15 and 50 epochs, Phase-1 and the stack have almost identical RMSE and MSE, but the stack has materially lower MAE and higher correlation. This means the aggregate squared-error magnitude is close while Phase-1 makes more typical absolute errors and tracks cross-sample variation less reliably. At 100 epochs, the stack leads clearly on every metric.

Within the Phase-1 sweep, MAE improves monotonically from 0.040595 to 0.035934. RMSE and MSE improve through 50 epochs but worsen at 100 epochs, and correlation peaks at 50 epochs before declining. Meanwhile, training loss falls continuously. This is consistent with late-stage overfitting or sensitivity to the train–test volatility-regime shift. The complete sweep is characterization evidence; the 50- or 100-epoch checkpoint must not be selected retrospectively from these test results.

The current framework `VolatilityRegressor` has an unconstrained linear output, and 5.8–8.4% of its predictions are negative. Raw LSTM uses a Softplus output, while the stack applies its predeclared nonnegative clipping rule. Therefore, the primary framework table preserves the raw saved predictions, and zero-clipped framework results are recorded below only as a post-run diagnostic—not as replacement headline metrics.

| Epoch | Raw Phase-1 MAE | Clipped diagnostic MAE | Raw Phase-1 RMSE | Clipped diagnostic RMSE | Raw Phase-1 MSE | Clipped diagnostic MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.040595 | 0.040273 | 0.089076 | 0.089044 | 0.007935 | 0.007929 |
| 50 | 0.036816 | 0.036571 | 0.086553 | 0.086479 | 0.007491 | 0.007479 |
| 100 | 0.035934 | 0.035611 | 0.087684 | 0.087594 | 0.007688 | 0.007673 |

Clipping gives only a small numerical improvement and does not change the main conclusion. The strict target equality check passed for every framework/Raw-LSTM/stack epoch pair, so the reported differences are not caused by sample-count or target-alignment mismatches.

## Legacy Raw-OHLCV MLP context

| Epoch | Legacy MLP MAE | Legacy MLP RMSE | Legacy MLP MSE | Legacy MLP correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.040360 | 0.093050 | 0.008658 | 0.727931 |
| 50 | 0.042581 | 0.094750 | 0.008978 | 0.705630 |
| 100 | 0.047047 | 0.100114 | 0.010023 | 0.660240 |

These MLP values are not used for strict improvement claims. The legacy runner constructs volatility targets from merged sequence arrays and retains 27,499 test rows, whereas the shared contract has contract-aware next-window targets and 27,450 rows. The Raw-OHLCV MLP must be rerun against the saved label bundle before it becomes the direct strict internal baseline.

## Conclusion and follow-up

The five-branch Phase-1 framework is stronger than Raw LSTM and competitive with the GARCH–LSTM stack on RMSE/MSE at the shorter two budgets, but it does not surpass the stack overall. This is a positive representation result, not a universal superiority claim.

- Rerun Raw-OHLCV MLP using the exact shared volatility label rows.
- Repeat the strict framework/baseline matrix over multiple seeds and report mean plus standard deviation.
- Add per-contract metrics and paired bootstrap intervals before making a strong comparative claim.
- Run branch ablations, particularly all-branches-minus-BYOL and volatility-relevant statistical-only features, to identify the source of the gain over Raw LSTM.
- Predeclare and rerun a nonnegative framework decoder (for example Softplus) so decoder constraints match the volatility task before final claims.
- Treat alternative fusion, loss weighting, or training budgets as newly predeclared experiments rather than selecting them from this locked-test sweep.
