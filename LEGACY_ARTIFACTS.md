# Legacy artifacts from the pre-split data pipeline

Date archived: 2026-09-20

All generated artifacts that depend on the preprocessing-before-split pipeline
have been moved to `_old` roots. They are retained for historical
characterisation and reproducibility, but must not be consumed by new training,
feature extraction, task evaluation, or strict baseline comparisons.

| Legacy artifact class | Archived location | New-output location |
|---|---|---|
| Shared framework and encoder experiments | `experiments_old/` | `experiments/` |
| Encoder and task checkpoints | `checkpoints_old/` | `checkpoints/` |
| Processed sequence bundles | `data/processed_old/` | `data/processed/` |
| Deterministic and neural feature bundles | `data/features_old/` | `data/features/` |
| Price, volatility, and classification labels | `data/task_labels_old/` | `data/task_labels/` |
| LSTM baseline experiments | `src/baselines/lstm_baseline/experiments_old/` | `src/baselines/lstm_baseline/experiments/` |
| Raw-OHLCV MLP experiments | `src/baselines/mlp_baseline/experiments_old/` | `src/baselines/mlp_baseline/experiments/` |
| TA-MLP experiments | `src/baselines/ta_mlp_baseline/experiments_old/` | `src/baselines/ta_mlp_baseline/experiments/` |
| GINN experiments | `src/baselines/ginn_baseline/experiments_old/` | `src/baselines/ginn_baseline/experiments/` |
| Raw LSTM volatility experiments | `src/baselines/raw_lstm_volatility/experiments_old/` | `src/baselines/raw_lstm_volatility/experiments/` |
| GARCH--LSTM stacking experiments | `src/baselines/garch_lstm_stacking/experiments_old/` | `src/baselines/garch_lstm_stacking/experiments/` |

Paths embedded inside archived manifests describe the paths used when those
runs were produced. They have intentionally not been rewritten, so provenance
records remain faithful to the original execution environment.

The historical root-level `event_stacked_lstm.pth` checkpoint is also archived
under `checkpoints_old/`.
