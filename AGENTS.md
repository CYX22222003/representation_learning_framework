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

# Or run both steps together across all timeframes
python scripts/prepare_data_pipeline.py --timeframes 1h,4h,1d --seq-len 64 --top-k 50
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
data/features/*.npz    (FeatureBundle: statistical + transformed arrays)
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
| `contrastive` | `src/models/contrastive.py` | 128 (projector dim) |

The `statistical` and `transformed` branches are **deterministic** — no training required. The `vae` and `contrastive` neural encoders must be pretrained unsupervised (via `src/training/`) before the aggregator is trained.

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
}

# concat mode (default) — output_dim = 317, no learnable params in aggregator
agg = RepresentationAggregator(branch_dims)

# gated mode — output_dim = 128, adds projection + gate network
agg = RepresentationAggregator(branch_dims, out_dim=128, mode="gated")

embedding, weights = agg({          # weights is None in concat mode
    "statistical": stat_tensor,
    "transformed": trans_tensor,
    "vae":         vae_tensor,
    "contrastive": con_tensor,
})
task_head = nn.Linear(agg.output_dim, n_outputs)  # works for both modes
```

### Adding new features

**New statistical feature** (e.g. rolling skewness): add a helper in `src/features/statistical.py`, concatenate its output inside `compute_statistical_features`, and update `statistical_feature_dim()` to match. The aggregator will pick up the new dimension automatically through the utility function.

**New transformation feature** (e.g. STFT): same pattern in `src/features/transform.py` and `transform_feature_dim()`.

**New neural encoder** (e.g. Transformer): add a model file in `src/models/`, a training loop in `src/training/`, then register a new key in `branch_dims` when constructing the aggregator. No changes needed to the aggregator class itself.

### Module responsibilities

| Directory | Responsibility |
|---|---|
| `src/data_processing/` | Preprocessing (ffill, volume z-score, sliding windows, 80/20 splits), `SequenceDataset`, `.npz` I/O |
| `src/features/` | Deterministic feature extractors; `FeatureBundle` dataclass; `NpzFeatureStore` save/load |
| `src/aggregation/` | `RepresentationAggregator` nn.Module — concat or gated fusion of N branches |
| `src/models/` | Model architecture definitions and loss functions only (VAE, contrastive CNN) |
| `src/training/` | Training loop functions (`train_vae_epoch`, `train_contrastive_epoch`) |
| `src/tasks/` | Default decoder: task heads (`PriceRegressor`, `VolatilityRegressor`, `TrendClassifier`) + label builders. Shared by the framework and all internal baselines. |
| `src/evaluation/` | Unified metrics (`regression_metrics`, `mse_and_corr`, `classification_metrics`) |
| `src/baselines/` | Comparison models — `lstm_baseline/` (external benchmark), `mlp_baseline/` (internal), `ta_mlp_baseline/` (external benchmark), `ginn_baseline/` (external benchmark) |
| `scripts/` | Runnable entry points; each inserts `src/` into `sys.path` |

### Key data contracts

- Raw feather files must have columns: `open`, `high`, `low`, `close`, `volume`
- Processed `.npz`: keys `train` and `test`, both `float32` of shape `[N, seq_len, 5]`
- Feature `.npz` (via `NpzFeatureStore`): keys `statistical`, `transformed`, `neural`; a companion `.index.npz` stores `train_size`/`test_size` to recover the split after concatenation
- GARCH feature vector per column: `[omega, alpha, beta, persistence, uncond_var, mean_cond_var, std_cond_var]`
