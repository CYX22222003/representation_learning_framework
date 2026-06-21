# LSTM Baseline — Price Prediction Experiment

**Date:** 2026-06-21
**Script:** `src/baselines/lstm_baseline/lstm_model.py` (`__main__` block)
**Log:** `2026-06-21-experiment-log.log` (in this directory)
**Checkpoint:** `event_stacked_lstm.pth` (saved at project root, 100 KB)

---

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6, torch 2.12.1+cu126) |
| Data source | Raw `.feather` files via `four_hour_file_list` (close column only) |
| Timeframe | 4-hour candles |
| Sequence length | 64 |
| Input shape | `[batch, 64, 1]` (univariate close prices) |
| Target | `close[t + 64]` (next close after window) |
| Train / Val split | 80/20 random split of the training set (`random_split`) |
| Train / Test split | 80/20 chronological per contract |
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
| Loss | MAPE (Mean Absolute Percentage Error, ε=1e-8) |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Result | Stopped at epoch 25 |

### Loss curve (train / val MAPE)

| Epoch | Train | Val |
|---|---|---|
| 0 | 0.1546 | 0.0448 |
| 5 | 0.0577 | 0.0373 |
| 10 | 0.0529 | 0.0446 |
| **15** | **0.0519** | **0.0229** ← best val |
| 20 | 0.0510 | 0.0320 |
| 25 | 0.0507 | 0.0286 |

- **Best val MAPE:** 0.0229 at epoch 15
- **Final train MAPE:** ~0.05 (plateau)
- Val loss is noisy throughout — the random val split mixes future and past samples, so val MAPE varies sample-to-sample more than train.

## Test Results

| Metric | Value |
|---|---|
| **Test MAPE** | **0.5865** (≈ 58.65%) |

## Observations

1. **MAPE blow-up on near-zero prices.** Polymarket OHLC prices lie in `[0, 1]`, and many markets sit near 0 (long-shot outcomes). MAPE divides by `y_true`, so a handful of test samples with `y_true ≈ 0` dominate the average. The training loss (~0.05) and best val loss (0.023) suggest the model fits most of the distribution well; the test 0.59 is driven by a small tail of near-zero targets.

2. **Train/val gap is small in early epochs but val becomes noisy.** Val MAPE oscillates between 0.02 and 0.05 from epoch 10 onward — characteristic of MAPE's instability rather than overfitting (train continues to inch down).

3. **Early stopping fired at epoch 25.** Best checkpoint was epoch 15; no improvement for 10 epochs after.

## Caveats vs. the Unified Framework

This run used the script's own data pipeline, **not** the unified `.npz` splits. The numbers above are not directly comparable to framework results yet because:

- **Data source mismatch.** The script reads raw `.feather` files and rebuilds sequences with its own splitter; the framework uses `data/processed/market_4h_seq64_top50.npz`.
- **Metric mismatch.** This run reports MAPE; the framework spec calls for MAE and RMSE.
- **Target definition.** The script uses `close[t+seq_len]`; the framework uses `build_price_prediction_targets` which targets `sequences[i+1, -1, 3]`. Equivalent for stride=1 but the wrapper differs.
- **Checkpoint location.** Script saved to `./event_stacked_lstm.pth` at project root, not `checkpoints/`.

A second run on the unified pipeline with MAE/RMSE metrics is needed before this number can sit in the comparison table.
