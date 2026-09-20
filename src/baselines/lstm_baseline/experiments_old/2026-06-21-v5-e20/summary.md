# LSTM Baseline — v5 sweep run, 20 epochs

**Date:** 2026-06-21
**Run name:** `2026-06-21-v5-e20`
**Part of:** v5 epoch-budget characterization sweep (see `experiments/2026-06-21-v5-summary.md`)

## Configuration

| Item | Value |
|---|---|
| Loss | RMSE |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 64 |
| Epochs | 20 (fixed; no early stopping) |
| Seed | 0 |
| Input | univariate close, `[batch, 64, 1]` |
| Splits | train/test only (per-contract chronological 80/20) |
| Train / Test samples | 109,841 / 27,500 |
| Device | NVIDIA RTX 4060 Laptop GPU |

## Training trajectory (train RMSE)

| Epoch | Train RMSE |
|---|---|
| 0 | 0.0444 |
| 5 | 0.0233 |
| 10 | 0.0225 |
| 15 | 0.0223 |
| 19 | 0.0222 |

## Test results

| Metric | Value |
|---|---|
| **Test MAE** | **0.0072** |
| **Test RMSE** | **0.0157** |

## Plots

- `images/training_curve.png` — train RMSE per epoch (univariate close, RMSE loss, seed=0)
- `images/pred_vs_actual.png` — test scatter with y=x reference (N=27,500)

## Notes

This run is one point in the 5-run epoch sweep. Cross-run analysis is in `experiments/2026-06-21-v5-summary.md`. Do not interpret a single point as "the LSTM result" — the sweep is for characterizing behaviour across training durations, not for selecting a best run.
