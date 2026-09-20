# TA-MLP Baseline — Triclass Run (v1, epochs=25)

**Date:** 2026-06-22
**Run name:** `2026-06-22-v1/e25`
**Script:** `src/baselines/ta_mlp_baseline/ta_mlp_model.py --run-name 2026-06-22-v1/e25 --epochs 25 --seed 0 --label-mode triclass`
**Plotting:** `src/baselines/ta_mlp_baseline/plot_experiment.py experiments/2026-06-22-v1/e25`

---

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4060 Laptop GPU (CUDA 12.6, torch 2.12.1+cu126) |
| Data source | Raw `.feather` via `four_hour_file_list` (all 5 OHLCV columns + date for calendar features) |
| Timeframe | 4-hour candles |
| Features | 36 TA-Lib indicators (see `ta_features.py::FEATURE_NAMES`) |
| Per-contract scaling | z-score, mean/std fit on train rows only |
| Label mode | triclass — BUY (0) / HOLD (1) / SELL (2) |
| Labeling params | `b_window=5`, `f_window=2`, `hold_q=0.85`, `buy_sell_q=0.997` |
| Threshold fit | per-contract, train rows only |
| Train / Test split | 80/20 chronological per contract |

## Model

```
TAMLPClassifier(in_dim=36, n_classes=3):
  Linear(36→128) → LeakyReLU(0.01)
  Linear(128→64) → LeakyReLU(0.01)
  Linear(64→32)  → LeakyReLU(0.01)
  Linear(32→3)
```

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Training loss | CrossEntropyLoss |
| Epochs | 25 (fixed; no early stopping) |
| Seed | 0 |
| Final train CE | 0.5200 |

Selected epochs (CE loss):

| Epoch | Train CE |
|---|---|
| 0 | 0.6200 |
| 5 | 0.5750 |
| 10 | 0.5556 |
| 12 | 0.5498 |
| 24 | 0.5200 |

Full curve in `images/training_curve.png`.

## Test Results

| Metric | Value |
|---|---|
| **Accuracy** | **0.7115** |
| **Macro-F1** | **0.4651** |

### Per-class metrics

| Class | Precision | Recall | F1 |
|---|---|---|---|
| BUY  | 0.2288 | 0.4098 | 0.2936 |
| HOLD | 0.8560 | 0.8615 | 0.8587 |
| SELL | 0.4048 | 0.1735 | 0.2429 |

### Confusion matrix (rows = actual, cols = predicted)

|        | BUY | HOLD | SELL |
|---|---|---|---|
| **BUY**  | 684 | 792 | 193 |
| **HOLD** | 1249 | 10458 | 432 |
| **SELL** | 1057 | 968 | 425 |

## Plots

- `images/training_curve.png` — train CE per epoch.
- `images/confusion_matrix.png` — 3×3 heatmap with counts and row-normalized %.

## Notes

This run is part of the v1 sweep at epochs [15, 20, 25, 50, 100] (seed=0). See
`experiments/2026-06-22-v1/summary.md` for cross-run comparison and the
characterization framing — this is one of five runs used to describe the
baseline's behavior across training budgets, not a "best" run selected by
test metrics.
