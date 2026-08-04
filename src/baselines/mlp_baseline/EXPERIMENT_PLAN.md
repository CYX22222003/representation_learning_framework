# Raw-OHLCV MLP Baseline Experiment Plan

This document defines the setup for carrying out the internal Raw-OHLCV MLP baseline experiment.

The baseline is intentionally simple: it trains an MLP directly on processed OHLCV sequence tensors and attaches the same default task heads used by the framework. Its role is to test whether the multi-branch representation framework improves over a direct raw-sequence MLP.

## Scope

Run the Raw-OHLCV MLP baseline for the downstream tasks supported by `src/tasks/`:

| Task | Target builder | Task head | Primary metrics |
|---|---|---|---|
| Price prediction | `build_price_prediction_targets` | `PriceRegressor` | MAE, RMSE, MSE, correlation |
| Volatility prediction | `build_volatility_targets` | `VolatilityRegressor` | MAE, RMSE, MSE, correlation |
| Trend classification | `build_trend_labels` | `TrendClassifier` | Accuracy, F1 |

The baseline uses `RawOHLCVMLP` from `mlp_model.py` as the encoder:

```text
raw OHLCV sequence [seq_len, 5]
  -> flatten
  -> RawOHLCVMLP encoder
  -> task head
  -> task prediction
```

The encoder and task head are trained end-to-end. There is no unsupervised pretraining step.

## Data

Use the processed sequence files produced by the standard data pipeline:

```text
data/processed/market_1h_seq64_top50.npz
data/processed/market_4h_seq64_top50.npz
data/processed/market_1d_seq64_top50.npz
```

Each file must contain:

```text
train: float32 [N_train, seq_len, 5]
test:  float32 [N_test,  seq_len, 5]
```

The global split is already applied during preprocessing:

```text
per contract chronological split:
  first 80% -> train
  last 20%  -> test
```

The test split is locked. No model parameter, label statistic, scaler, stopping rule, or model-selection decision may be influenced by test sequences.

## Split Policy

Follow `docs/training_test_data_selection.md` as the authority:

- Use train and test only.
- Do not create a validation split.
- Do not use early stopping.
- Do not tune hyperparameters by inspecting test metrics.
- Build labels separately inside the train split and the test split.
- Report all epoch-budget results instead of selecting a best checkpoint from test performance.

Note: `docs/design.md` contains a benchmark-retraining sentence that mentions `train/val/test` splits. For this experiment, that wording is treated as stale relative to `docs/training_test_data_selection.md`, which explicitly says this project uses train/test only.

## Canonical Configuration

Default dataset for the first run:

```text
data/processed/market_4h_seq64_top50.npz
```

Default model configuration:

```text
hidden_dims = [512, 512, 256, 256, 128]
encoder_output_dim = 128
dropout = 0.1
head_hidden_dim = 128
```

Default training configuration:

```text
optimizer = Adam
learning_rate = 1e-4
weight_decay = 0.0
batch_size = 128
seed = 0
horizon = 1
price_index = 3  # close column
trend_threshold = 0.0
```

Canonical epoch-budget sweep:

```text
15, 20, 25, 50, 100
```

These are characterization runs, not a model-selection sweep. The final report should include all epoch budgets.

## Execution Order

1. Confirm the processed `.npz` exists and contains `train` and `test`.
2. Select one task: `price`, `volatility`, or `trend`.
3. Build supervised pairs from the train split only.
4. Build supervised pairs from the test split separately.
5. Initialize `RawOHLCVMLP` and the matching task head.
6. Train on the full train split for the largest epoch budget.
7. Save snapshots at every requested epoch budget.
8. Evaluate each saved snapshot on the test split once.
9. Write metrics, predictions, history, config, and summaries.
10. Repeat for the remaining tasks and timeframes if needed.

## Commands

Price prediction:

```bash
.venv/bin/python3 src/baselines/mlp_baseline/run_experiment.py \
  --task price \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name 2026-08-04-price-v1 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0
```

Volatility prediction:

```bash
.venv/bin/python3 src/baselines/mlp_baseline/run_experiment.py \
  --task volatility \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name 2026-08-04-volatility-v1 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0
```

Trend classification:

```bash
.venv/bin/python3 src/baselines/mlp_baseline/run_experiment.py \
  --task trend \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name 2026-08-04-trend-v1 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0
```

Generate diagrams after a run:

```bash
.venv/bin/python3 src/baselines/mlp_baseline/plot_experiment.py \
  src/baselines/mlp_baseline/experiments/2026-08-04-price-v1
```

## Expected Artifacts

Each experiment is written under:

```text
src/baselines/mlp_baseline/experiments/<run-name>/
```

Expected layout:

```text
config.json
dataset_manifest.json
summary.md
sweep_metrics.json
e15/
  checkpoint.pth
  history.npz
  predictions.npz
  metrics.json
  summary.md
  images/
    training_curve.png
    pred_vs_actual.png              # regression tasks only
    error_distribution.png          # regression tasks only
    confusion_matrix.png            # trend task only
    probability_histogram.png       # trend task only
e20/
  ...
e25/
  ...
e50/
  ...
e100/
  ...
images/
  epoch_sweep.png
```

The root `summary.md` should show the full epoch-budget table. Per-budget `summary.md` files should describe that checkpoint's metrics and artifact paths.

## Comparison Use

Use this baseline as an internal comparison against:

- Full framework embedding plus default task head.
- Single-branch ablations.
- External benchmarks where task alignment is valid.

The comparison is fair only when all models use the same processed train/test partitions and fixed training budgets.

## Leakage Checklist

Before treating a run as reportable, confirm:

- The input file is a standard `data/processed/market_*_seq64_top50.npz` file.
- Training reads only the `train` array.
- Evaluation reads only the `test` array.
- No validation split is created.
- No early stopping is used.
- No test metric is used to choose an epoch, seed, model width, learning rate, or task threshold.
- Every epoch budget in the sweep is reported.
- The run directory contains config, manifest, checkpoint, history, predictions, metrics, and summary files.
