# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Running Scripts

All scripts are run from the **project root**. Each script in `scripts/` self-bootstraps its Python path by inserting `src/` into `sys.path` — no package install is needed beyond `requirements.txt`.

```bash
# Step 1 — build sequence tensors from raw feather files
python scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
# → data/processed/market_4h_seq64_top50.npz

# Step 2 — extract statistical + transformation features
python scripts/prepare_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/features/features_4h_seq64_top50.npz

# Step 2b — build the framework MVP feature bundle with frozen SSL branches
python scripts/prepare_framework_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/features/features_4h_seq64_top50.npz \
  --vae-checkpoint checkpoints/vae_4h_seq64_top50.pth \
  --contrastive-checkpoint checkpoints/contrastive_4h_seq64_top50.pth \
  --device cuda \
  --batch-size 1024 \
  --overwrite

# Validate feature dimensions, split metadata, and finite values
python scripts/validate_feature_store.py \
  --features-npz data/features/features_4h_seq64_top50.npz \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --expected-dim statistical=70 \
  --expected-dim transformed=55 \
  --expected-dim vae=64 \
  --expected-dim contrastive=128

# Or run both steps together across all timeframes
python scripts/prepare_data_pipeline.py --timeframes 1h,4h,1d --seq-len 64 --top-k 50

# Pretrain the contrastive encoder on the locked train split
python scripts/train_contrastive_encoder.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name contrastive-4h-seq64-top50 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0 \
  --device cuda \
  --canonical-checkpoint checkpoints/contrastive_4h_seq64_top50.pth

# Generate contrastive training plots and a markdown report
python scripts/plot_contrastive_experiment.py experiments/contrastive_encoder/contrastive-4h-seq64-top50

# Pretrain the VAE encoder on the locked train split
python scripts/train_vae_encoder.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name vae-4h-seq64-top50 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0 \
  --device cuda \
  --canonical-checkpoint checkpoints/vae_4h_seq64_top50.pth

# Generate VAE training plots and a markdown report
python scripts/plot_vae_experiment.py experiments/vae_encoder/vae-4h-seq64-top50

# Pretrain the BYOL encoder on the locked train split
python scripts/train_byol_encoder.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name byol-4h-seq64-top50 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0 \
  --device cuda \
  --canonical-checkpoint checkpoints/byol_4h_seq64_top50.pth

# Generate BYOL training plots and a markdown report
python scripts/plot_byol_experiment.py experiments/byol_encoder/byol-4h-seq64-top50

# Train the framework MVP on the price prediction task
python scripts/train_framework.py \
  --task price_prediction \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --features-npz data/features/features_4h_seq64_top50.npz \
  --run-name 4h_stat_transform_vae_contrastive_concat \
  --mode concat \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 512 \
  --device cuda \
  --overwrite

# Prepare TA-MLP-style tri-class labels for trend classification
python scripts/prepare_trend_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/trend_classification/triclass_4h_seq64_top50.npz \
  --timeframe 4h \
  --seq-len 64 \
  --top-k 50 \
  --overwrite

# Train the framework MVP on the trend classification task
python scripts/train_framework.py \
  --task trend_classification \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --features-npz data/features/features_4h_seq64_top50.npz \
  --labels-npz data/task_labels/trend_classification/triclass_4h_seq64_top50.npz \
  --run-name 4h_triclass_stat_transform_vae_contrastive_concat \
  --mode concat \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 512 \
  --device cuda \
  --overwrite

# Build the shared realised-volatility labels used by both volatility benchmarks
.venv/bin/python3 scripts/prepare_volatility_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --timeframe 4h \
  --seq-len 64 \
  --top-k 50 \
  --overwrite

# Train the direct Raw LSTM volatility benchmark (15/50/100 epoch snapshots)
PYTHONPATH=src .venv/bin/python3 src/baselines/raw_lstm_volatility/run_experiment.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --labels-npz data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --run-name 4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --device cuda

# Run the adapted GARCH--LSTM hybrid using the completed Raw LSTM artifacts
PYTHONPATH=src .venv/bin/python3 src/baselines/garch_lstm_stacking/run_experiment.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --labels-npz data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --raw-lstm-run src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0 \
  --run-name 4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --crossfit-folds 5 \
  --seed 0 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --elasticnet-alpha 1e-4 \
  --elasticnet-l1-ratio 0.5 \
  --device cuda
```

Trained model checkpoints are saved to and loaded from `checkpoints/`.

## Architecture

### Data flow

```
data/*.feather   (raw Polymarket OHLCV, one file per contract per timeframe)
      │
      ▼  scripts/prepare_sequences.py
data/processed/*.npz   ([N, seq_len, 5] float32 tensors, keyed "train"/"test")
      │
      ▼  scripts/prepare_features.py
data/features/*.npz    (FeatureBundle: deterministic arrays + named neural branches)
      │
      ▼  src/aggregation/aggregator.py  (RepresentationAggregator)
      unified embedding h_i
      │
      ▼  src/tasks/
      task-specific predictions
```

### Multi-branch representation

Feature extraction operates per-sequence, per-OHLCV-column (5 columns: open, high, low, close, volume):

| Branch | Module | Default output dim |
|---|---|---|
| `statistical` | `src/features/statistical.py` | 70 = 5 cols × (AR-5 coeffs + 2 residual stats + 7 GARCH features) |
| `transformed` | `src/features/transform.py` | 55 = 5 cols × (FFT top-8 + 3 Haar wavelet energies) |
| `vae` | `src/models/vae.py` | 64 (latent dim) |
| `contrastive` | `src/models/contrastive.py` | 128 (frozen backbone embedding; projector is also 128 by default) |
| `byol` | `src/models/byol.py` | 128 (online backbone hidden dim) |

The `statistical` and `transformed` branches are **deterministic** — no training required. The `vae`, `contrastive`, and `byol` neural encoders must be pretrained unsupervised (via `src/training/`) before the aggregator is trained.

`RepresentationAggregator` (`src/aggregation/aggregator.py`) accepts an arbitrary `branch_dims: dict[str, int]` and supports two fusion modes:

- **`mode="concat"` (default)** — branches are concatenated; no learnable parameters; `output_dim = sum of branch dims`.
- **`mode="gated"`** — each branch is projected to `out_dim`, then a gating network produces softmax weights; `output_dim = out_dim`.

Use `agg.output_dim` to size the task head regardless of mode. Use the dimension utilities to avoid hardcoding branch sizes:

```python
from features.statistical import statistical_feature_dim
from features.transform import transform_feature_dim
from aggregation.aggregator import RepresentationAggregator

branch_dims = {
    "statistical":  statistical_feature_dim(n_cols=5, ar_order=5),   # 70
    "transformed":  transform_feature_dim(n_cols=5),                  # 55
    "vae":          64,
    "contrastive":  128,
    "byol":         128,
}

# concat mode (default) — output_dim = 445, no learnable params in aggregator
agg = RepresentationAggregator(branch_dims)

# gated mode — output_dim = 128, adds projection + gate network
agg = RepresentationAggregator(branch_dims, out_dim=128, mode="gated")

embedding, weights = agg({          # weights is None in concat mode
    "statistical": stat_tensor,
    "transformed": trans_tensor,
    "vae":         vae_tensor,
    "contrastive": con_tensor,
    "byol":        byol_tensor,
})
task_head = nn.Linear(agg.output_dim, n_outputs)  # works for both modes
```

### Adding new features

**New statistical feature** (e.g. rolling skewness): add a helper in `src/features/statistical.py`, concatenate its output inside `compute_statistical_features`, and update `statistical_feature_dim()` to match. The aggregator will pick up the new dimension automatically through the utility function.

**New transformation feature** (e.g. STFT): same pattern in `src/features/transform.py` and `transform_feature_dim()`.

**New neural encoder** (e.g. Transformer): add a model file in `src/models/`, a training loop in `src/training/`, then register a new key in `branch_dims` when constructing the aggregator and save its frozen embeddings under that branch name. No changes needed to the aggregator class itself.

### Module responsibilities

| Directory | Responsibility |
|---|---|
| `src/data_processing/` | Preprocessing (ffill, volume z-score, sliding windows, 80/20 splits), `SequenceDataset`, `.npz` I/O |
| `src/features/` | Deterministic feature extractors; branch-aware `FeatureBundle` dataclass; `NpzFeatureStore` save/load |
| `src/aggregation/` | `RepresentationAggregator` nn.Module — concat or gated fusion of N branches |
| `src/models/` | Model architecture definitions and loss functions only (VAE, contrastive CNN, BYOL) |
| `src/training/` | Training loop functions (`train_vae_epoch`, `train_contrastive_epoch`, `train_byol_epoch`) |
| `src/tasks/` | Default decoder: task heads (`PriceRegressor`, `VolatilityRegressor`, `TrendClassifier`) + task-label builders. `volatility_labels.py` is the contract-aware shared volatility label contract for strict comparisons. |
| `src/evaluation/` | Unified metrics (`regression_metrics`, `mse_and_corr`, `classification_metrics`) |
| `src/baselines/` | Comparison models — `lstm_baseline/` (external price benchmark), `raw_lstm_volatility/` and `garch_lstm_stacking/` (external volatility benchmarks), `mlp_baseline/` (internal), `ta_mlp_baseline/` (external trend benchmark), `ginn_baseline/` (volatility limitation evidence) |
| `scripts/` | Runnable entry points; each inserts `src/` into `sys.path` |

### Key data contracts

- Raw feather files must have columns: `open`, `high`, `low`, `close`, `volume`
- Processed `.npz`: keys `train` and `test`, both `float32` of shape `[N, seq_len, 5]`
- Feature `.npz` (via `NpzFeatureStore`): keys `statistical`, `transformed`, plus one key per frozen neural branch such as `vae` or `contrastive`; a companion `.index.npz` stores `train_size`/`test_size` to recover the split after train+test concatenation. Legacy files with an empty or packed `neural` key remain loadable, but new neural features should be stored by branch name.
- Trend task label `.npz`: saved under `data/task_labels/trend_classification/`; keys include `train_labels`, `test_labels`, aligned train/test row indices, class names, and train-fitted threshold metadata. Horizon rows are dropped inside each split so labels never cross the train/test boundary.
- Volatility task label `.npz`: saved under `data/task_labels/volatility_prediction/`; contains realised-volatility targets, aligned train/test row indices, contract IDs, and window starts. The Raw LSTM, GARCH--LSTM stack, future framework volatility run, and Raw-OHLCV MLP volatility rerun must use this bundle for strict comparison; older MLP volatility artifacts are characterization-only.
- GARCH feature vector per column: `[omega, alpha, beta, persistence, uncond_var, mean_cond_var, std_cond_var]`
