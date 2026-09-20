# LSTM Baseline — Price Prediction Experiment

**Date:** 2026-06-21
**Script:** `src/baselines/lstm_baseline/lstm_model.py` (`__main__` block)
**Log:** `2026-06-21-experiment-log-v2.log` (in this directory)
**Checkpoint:** `event_stacked_lstm.pth` (project root, 100 KB)

---

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4060 Laptop GPU (CUDA 12.6, torch 2.12.1+cu126) |
| Data source | Raw `.feather` files via `four_hour_file_list` (close column only) |
| Timeframe | 4-hour candles |
| Sequence length | 64 |
| Input shape | `[batch, 64, 1]` (univariate close prices) |
| Target | `close[t + 64]` (next close after window) |
| Train / Test split | 80/20 chronological per contract |
| Train / Val split | 80/20 random split of the training set |
| Train samples | 109,841 |
| Test samples | 27,500 |

## Model

3-layer stacked LSTM with decreasing hidden width and dropout:

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
| Stopped at | Epoch 28 |

### Loss curve (train / val RMSE)

| Epoch | Train | Val |
|---|---|---|
| 0 | 0.0489 | 0.0199 |
| 5 | 0.0243 | 0.0150 |
| 10 | 0.0229 | 0.0132 |
| 15 | 0.0228 | 0.0124 |
| **18** | 0.0225 | **0.0117** ← best val |
| 20 | 0.0223 | 0.0124 |
| 25 | 0.0222 | 0.0125 |
| 28 | 0.0222 | 0.0135 |

- Best val RMSE: **0.0117** at epoch 18
- Train RMSE plateaus around 0.022; val sits ~0.012–0.013

## Test Results

| Metric | Value | Interpretation (prices in [0, 1]) |
|---|---|---|
| **Test MAE** | **0.00792** | average prediction off by ~0.8 percentage points |
| **Test RMSE** | **0.01650** | RMSE ~1.65 percentage points |

## Analysis

1. **Generalization is healthy under MAE/RMSE.** The model predicts within ~1 percentage point of the true close on average. Train RMSE 0.0222 vs test RMSE 0.0165 — test is in fact slightly better than train, consistent with many test samples falling in late-contract regimes where prices are pinned near 0 or 1 and locally easy to predict.

2. **The univariate close-only input is the main remaining limitation.** The model cannot anticipate resolution-driven price jumps; it can only smooth through them. Adding volume / OHLC would likely tighten RMSE further.

3. **RMSE training loss is well-behaved.** Loss decreased monotonically on train; no NaN gradients; early stopping triggered cleanly at epoch 28.

## Caveats vs the Unified Framework

This run still uses the script's own data pipeline (raw `.feather` → ad-hoc sliding window) rather than the unified `data/processed/market_4h_seq64_top50.npz` splits used by the framework. Before this number sits in the comparison table:

- [ ] Retrain on `data/processed/market_4h_seq64_top50.npz` using `build_price_prediction_targets`
- [ ] Save checkpoint to `checkpoints/` with a descriptive name
- [ ] Confirm `regression_metrics` from `src/evaluation/metrics.py` produces the same MAE/RMSE
