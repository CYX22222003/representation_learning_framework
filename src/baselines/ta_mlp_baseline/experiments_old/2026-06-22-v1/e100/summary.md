# TA-MLP Baseline — Triclass Run (v1, epochs=100)

**Date:** 2026-06-22
**Run name:** `2026-06-22-v1/e100`
**Script:** `src/baselines/ta_mlp_baseline/ta_mlp_model.py --run-name 2026-06-22-v1/e100 --epochs 100 --seed 0 --label-mode triclass`
**Plotting:** `src/baselines/ta_mlp_baseline/plot_experiment.py experiments/2026-06-22-v1/e100`

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
| Epochs | 100 (fixed; no early stopping) |
| Seed | 0 |
| Final train CE | 0.4099 |

Selected epochs (CE loss):

| Epoch | Train CE |
|---|---|
| 0 | 0.6200 |
| 5 | 0.5750 |
| 10 | 0.5556 |
| 50 | 0.4673 |
| 99 | 0.4099 |

Full curve in `images/training_curve.png`.

## Test Results

| Metric | Value |
|---|---|
| **Accuracy** | **0.7065** |
| **Macro-F1** | **0.4595** |

### Per-class metrics

| Class | Precision | Recall | F1 |
|---|---|---|---|
| BUY  | 0.2326 | 0.3146 | 0.2674 |
| HOLD | 0.8314 | 0.8604 | 0.8457 |
| SELL | 0.3588 | 0.2106 | 0.2654 |

### Confusion matrix (rows = actual, cols = predicted)

|        | BUY | HOLD | SELL |
|---|---|---|---|
| **BUY**  | 525 | 889 | 255 |
| **HOLD** | 1027 | 10445 | 667 |
| **SELL** | 705 | 1229 | 516 |

## Plots

- `images/training_curve.png` — train CE per epoch.
- `images/confusion_matrix.png` — 3×3 heatmap with counts and row-normalized %.

## Notes

This run is part of the v1 sweep at epochs [15, 20, 25, 50, 100] (seed=0). See
`experiments/2026-06-22-v1/summary.md` for cross-run comparison and the
characterization framing — this is one of five runs used to describe the
baseline's behavior across training budgets, not a "best" run selected by
test metrics.
