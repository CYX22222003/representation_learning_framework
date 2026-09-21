# Phase 5 Exploratory Regression Sensitivities (Seed 0)

These tasks were frozen after the primary two-hour result and are interpreted as secondary sensitivities, not replacement targets selected from evaluation performance.

| Task | Walk | MAE | RMSE | Pearson | Spearman | Sign agreement | Reconstructed probability MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_delta_h8 | 1 | 0.00416801 | 0.01274698 | 0.059602 | 0.007261 | 0.384628 | 0.00416801 |
| raw_delta_h8 | 2 | 0.00542461 | 0.02126556 | -0.047109 | -0.033975 | 0.358655 | 0.00542461 |
| log_return_h2 | 1 | 0.05899403 | 0.10815900 | -0.005980 | 0.005214 | 0.366678 | 0.00252935 |
| log_return_h2 | 2 | 0.06260438 | 0.16412535 | -0.015256 | -0.003790 | 0.334125 | 0.00310111 |

## Pooled out-of-future observations

- Eight-hour raw change: MAE `0.00455960`, RMSE `0.01589894`, Pearson `0.014209`, Spearman `-0.005372`, sign agreement `0.376534`.
- Eight-hour zero-change reference: MAE `0.00342947`, RMSE `0.01505473`.
- Two-hour log return: MAE `0.06012765`, RMSE `0.12838709`, Pearson `-0.010259`, Spearman `0.003947`, sign agreement `0.356456`.
- Log-return zero-change reference: MAE `0.05568154`, RMSE `0.12697035`.
- Log-return reconstructed probability: MAE `0.00270888`, RMSE `0.01142996`.

## Judgement

The primary two-hour raw-change Pearson/Spearman were `0.080745/-0.000518`. Extending the horizon to eight hours reduced these to `0.014209/-0.005372`, with the walk-specific Pearson signs disagreeing. The longer horizon therefore did not expose a stable signed relationship.

Ordinary log return also has approximately zero Pearson/Spearman (`-0.010259/0.003947`). Its reconstructed probability MAE `0.00270888` is effectively unchanged from the primary model's `0.00271136`, while reconstructed RMSE is higher. The proportional target changes weighting toward low starting prices but does not recover direction or ordered magnitude.

Together, the sensitivities do not support horizon length or additive target units as the main explanation for the regression limitation. A learned raw temporal comparator is still needed to separate representation loss from intrinsic short-history unpredictability.

The existing two-hour raw-change result remains the primary Phase 5 regression experiment.
