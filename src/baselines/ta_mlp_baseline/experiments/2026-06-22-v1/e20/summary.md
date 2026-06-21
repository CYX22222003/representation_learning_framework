# TA-MLP Baseline — Triclass Run (v1, epochs=20)

**Date:** 2026-06-22
**Run name:** `2026-06-22-v1/e20`
**Script:** `src/baselines/ta_mlp_baseline/ta_mlp_model.py --run-name 2026-06-22-v1/e20 --epochs 20 --seed 0 --label-mode triclass`
**Plotting:** `src/baselines/ta_mlp_baseline/plot_experiment.py experiments/2026-06-22-v1/e20`

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
| Epochs | 20 (fixed; no early stopping) |
| Seed | 0 |
| Final train CE | 0.5317 |

Selected epochs (CE loss):

| Epoch | Train CE |
|---|---|
| 0 | 0.6200 |
| 5 | 0.5750 |
| 10 | 0.5556 |
| 19 | 0.5317 |

Full curve in `images/training_curve.png`.

## Test Results

| Metric | Value |
|---|---|
| **Accuracy** | **0.7256** |
| **Macro-F1** | **0.4524** |

### Per-class metrics

| Class | Precision | Recall | F1 |
|---|---|---|---|
| BUY  | 0.2493 | 0.3859 | 0.3029 |
| HOLD | 0.8372 | 0.8940 | 0.8647 |
| SELL | 0.4208 | 0.1224 | 0.1897 |

### Confusion matrix (rows = actual, cols = predicted)

|        | BUY | HOLD | SELL |
|---|---|---|---|
| **BUY**  | 644 | 892 | 133 |
| **HOLD** | 1007 | 10852 | 280 |
| **SELL** | 932 | 1218 | 300 |

## Plots

- `images/training_curve.png` — train CE per epoch.
- `images/confusion_matrix.png` — 3×3 heatmap with counts and row-normalized %.

## Notes

This run is part of the v1 sweep at epochs [15, 20, 25, 50, 100] (seed=0). See
`experiments/2026-06-22-v1/summary.md` for cross-run comparison and the
characterization framing — this is one of five runs used to describe the
baseline's behavior across training budgets, not a "best" run selected by
test metrics.
