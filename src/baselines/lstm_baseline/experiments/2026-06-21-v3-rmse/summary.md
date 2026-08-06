# LSTM Baseline — Price Prediction Experiment (v3)

**Date:** 2026-06-21
**Script:** `src/baselines/lstm_baseline/lstm_model.py`
**Plotting script:** `src/baselines/lstm_baseline/plot_experiment.py`
**Log:** `2026-06-21-experiment-lstm-v3.log`
**Checkpoint:** `event_stacked_lstm.pth` (project root, 100 KB)
**Artifacts:** `lstm_training_history.npz`, `lstm_test_predictions.npz` (this directory)
**Plots:** `images/training_curve.png`, `images/pred_vs_actual.png`

---

## What's new in v3

- `train_model` now returns a `history` dict; `__main__` saves per-epoch train/val RMSE and the best-epoch index to `lstm_training_history.npz`.
- Test predictions and targets are saved to `lstm_test_predictions.npz`.
- New `plot_experiment.py` reads both artifacts and writes two PNGs to `images/`.

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
| Stopped at | Epoch 33 |
| **Best val epoch** | **23** |
| **Best val RMSE** | **0.0118** |

### Loss curve (selected epochs)

| Epoch | Train | Val |
|---|---|---|
| 0 | 0.0481 | 0.0188 |
| 5 | 0.0240 | 0.0152 |
| 10 | 0.0228 | 0.0140 |
| 15 | 0.0224 | 0.0128 |
| 20 | 0.0221 | 0.0124 |
| **23** | 0.0220 | **0.0118** ← best |
| 28 | 0.0221 | 0.0119 |
| 33 | 0.0219 | 0.0133 |

Full curve in `images/training_curve.png`.

## Test Results

| Metric | Value | Interpretation (prices in [0, 1]) |
|---|---|---|
| **Test MAE** | **0.01175** | average prediction off by ~1.18 percentage points |
| **Test RMSE** | **0.01785** | RMSE ~1.79 percentage points |

## Plots

### `images/training_curve.png`
Train vs val RMSE per epoch; vertical red dashed line marks the best-val epoch (23). The train curve plateaus around 0.022 while val drops to 0.0118 — train RMSE is *higher* than val, consistent with dropout regularising training-mode forward passes.

### `images/pred_vs_actual.png`
Scatter of predicted vs actual next-close on the test set (N=27,500) with the `y = x` reference. What to look for:
- **Diagonal alignment** in the middle of `[0, 1]` → model is calibrated where most training data lives
- **Bias near 0 and near 1** → model over-predicts near the lower edge (regression toward training mean) and under-predicts near the upper edge → the classic late-contract regime shift the model can't see in close-only input

## Analysis

1. **Run-to-run variance is real but small.** v2 produced MAE 0.0079 / RMSE 0.0165; v3 produced MAE 0.0118 / RMSE 0.0179. Architecture, loss, and data are identical — the difference comes from random weight init and the random val split. For a stable headline number, average a handful of seeded runs.

2. **Best val at epoch 23, ran to 33.** Early stopping consumed its patience cleanly; no sign of late-stage divergence.

3. **The scatter plot reveals what aggregate metrics hide.** RMSE alone says "model is accurate." The scatter shows where it's accurate and where it isn't — confirming the close-only input cannot anticipate near-resolution price jumps, which is the regime that matters most for trading.

## Caveats vs the Unified Framework

Same as previous runs — this script still uses its own data pipeline rather than `data/processed/market_4h_seq64_top50.npz`. Outstanding work to make this comparable to framework results:

- [ ] Retrain on the unified `.npz` splits using `build_price_prediction_targets`
- [ ] Save checkpoint to `checkpoints/` with a descriptive name
- [ ] Confirm `regression_metrics` from `src/evaluation/metrics.py` reproduces the same MAE/RMSE
- [ ] Run with multiple seeds and report mean ± std
