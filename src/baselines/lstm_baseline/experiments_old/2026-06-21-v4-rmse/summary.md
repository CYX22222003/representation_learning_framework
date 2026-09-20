# LSTM Baseline — Price Prediction Experiment (v4)

**Date:** 2026-06-21
**Run name:** `2026-06-21-v4-rmse`
**Script:** `src/baselines/lstm_baseline/lstm_model.py --run-name 2026-06-21-v4-rmse`
**Plotting:** `src/baselines/lstm_baseline/plot_experiment.py experiments/2026-06-21-v4-rmse`

---

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4060 Laptop GPU (CUDA 12.6, torch 2.12.1+cu126) |
| Data source | Raw `.feather` via `four_hour_file_list` (close column only) |
| Timeframe / seq_len | 4-hour candles, 64 |
| Input shape | `[batch, 64, 1]` |
| Target | `close[t + 64]` |
| Train / Test split | 80/20 chronological per contract |
| Train / Val split | 80/20 random split of training set |
| Train samples | 109,841 |
| Test samples | 27,500 |

## Model

```
LSTM(1→50)  → Dropout(0.2)
LSTM(50→30) → Dropout(0.1)
LSTM(30→20) → take last timestep → Dropout(0.05) → Linear(20→1)
```

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Training loss | RMSE (`sqrt(mean((y_pred − y_true)²) + ε)`) |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Stopped at | Epoch 20 |
| **Best val epoch** | **10** |
| **Best val RMSE** | **0.01193** |

### Loss curve (selected epochs)

| Epoch | Train | Val |
|---|---|---|
| 0 | 0.0552 | 0.0208 |
| 5 | 0.0243 | 0.0139 |
| **10** | 0.0229 | **0.0119** ← best |
| 15 | 0.0226 | 0.0128 |
| 20 | 0.0225 | 0.0121 |

Full curve in `images/training_curve.png`.

## Test Results

| Metric | Value | Interpretation (prices in [0, 1]) |
|---|---|---|
| **Test MAE** | **0.01100** | average prediction off by ~1.10 percentage points |
| **Test RMSE** | **0.01758** | RMSE ~1.76 percentage points |

## Plots

- `images/training_curve.png` — train vs val RMSE per epoch; vertical red dashed line marks the best-val epoch (10).
- `images/pred_vs_actual.png` — scatter of predicted vs actual next-close on the test set (N=27,500) with the `y = x` reference. Confirms regression-to-mean behaviour at the extremes near 0 and 1.

## Cross-run comparison

| Run | Best val RMSE | Test MAE | Test RMSE | Epochs run |
|---|---|---|---|---|
| v2-rmse | — | 0.0079 | 0.0165 | 29 |
| v3-rmse | 0.0118 | 0.0118 | 0.0179 | 34 |
| **v4-rmse** | **0.0119** | **0.0110** | **0.0176** | **21** |

Test metrics across v2/v3/v4 sit in a tight band (MAE ~0.008–0.012, RMSE ~0.0165–0.0179). The spread is consistent with random weight init + random val split variance — none of these runs is materially better than the others. The v4 run stopped earliest (epoch 20) because the val loss bottomed out very fast (epoch 10) and then drifted without improving.

## Notes

This run is the first to be executed entirely through the new CLI layout (`--run-name` flag) with all artifacts written inside the run directory automatically.
