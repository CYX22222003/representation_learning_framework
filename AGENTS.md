# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Running Scripts

> **Phase-4 execution gate (2026-09-20):** Phase 3 is concluded. Do not launch
> new downstream, baseline, or encoder training until the frozen Phase 4
> data-selection and walk-forward contract is implemented and validated
> under `scripts_v2/`. Per-contract lifecycle fractions alone are not a valid
> primary split for the pooled encoder because they can permit cross-contract
> calendar lookahead. The single-tail Phase 3
> absolute next-close results remain valid negative characterisation evidence;
> they must not be reused as Phase 4 results. See
> `docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`.
> The supporting lifecycle and representation-drift diagnostic is recorded in
> `docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md`.
> It motivates comparing a fixed first-walk encoder with fold-specific
> retraining, but does not establish that lifecycle stages require different
> encoder architectures.
> The top-80 target and walk-schedule feasibility evidence is recorded in
> `docs/data_analysis/2026-09-20-phase4-top80-contract-relative-walk-analysis.md`.
> Contract-relative anchored and fixed-length walks are valid retrospective
> lifecycle analyses, and both two and three walks have adequate aggregate
> samples. They are not the primary pooled-model split: relative cutoffs map to
> different calendar dates, and observed final contract length is future
> information unless the termination boundary was known at decision time.
> The one-hour timestamp and capacity audit is recorded in
> `docs/data_analysis/2026-09-20-phase4-top80-1h-timestamp-capacity.md`.
> Raw one-hour files have clean hourly timestamps and sufficient aggregate
> capacity even at a duration-matched 256-step context. For the current cohort,
> two global-calendar walks have materially better evaluation-contract coverage
> than three equally spaced walks. These are feasibility results only; the
> retrospective top-80 universe is not yet a deployment-safe selection rule.
> The one-hour activity-suitability audit is recorded in
> `docs/data_analysis/2026-09-20-phase4-top80-1h-activity-suitability.md`.
> It finds adequate non-trivial eight-hour movement overall, but also material
> inactive terminal tails: 19/80 contracts spend at least one quarter of the
> file at a trailing constant price. The causal recent-activity eligibility rule
> is now frozen and must be implemented identically for framework
> and baselines; retrospective last-change truncation is diagnostic only.
> The duration-matched frequency/universe comparison is recorded in
> `docs/data_analysis/2026-09-20-phase4-top50-top80-1h-4h-comparison.md`.
> One-hour and four-hour eight-hour targets have nearly identical staleness,
> lifecycle drift, and active-row distributions; aligned target correlations
> are 0.969--0.972. Four-hour top-80 with causal prior-24h activity is the
> frozen primary configuration. One-hour top-80 and four-hour top-50 remain
> resolution and universe-size sensitivities, respectively.
> The canonical frozen design is
> `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`:
> four-hour top-80, two fixed-duration rolling calendar walks, cutoff-local
> trailing-256-bar ranking, a causal prior-24h price-change mask, fold-specific
> models, and an eight-hour signed probability-movement target. Builder
> implementation and replay validation are the current execution gate.

> **Phase-2 execution pause (2026-09-20):** Do not launch training, feature
> extraction, or downstream evaluation against the current processed bundles.
> The legacy pipeline fits volume preprocessing and creates windows before the
> stored train/test split. Preserve all artifacts and first implement the
> raw-time-first replacement specified in
> `docs/data_processing_split_contract.md`.
> Legacy processed data, features, labels, checkpoints, framework experiments,
> and baseline experiments are archived under the `_old` roots listed in
> `LEGACY_ARTIFACTS.md`. Canonical output directories are reserved for rebuilt
> leakage-safe artifacts.

All scripts are run from the **project root**. Each script in `scripts/` self-bootstraps its Python path by inserting `src/` into `sys.path` — no package install is needed beyond `requirements.txt`.

```bash
# Step 1 — build sequence tensors from raw feather files
.venv/bin/python3 scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
# → data/processed/market_4h_seq64_top50_split_safe.npz
# plus a provenance manifest; refuses to overwrite by default

# Step 2 — extract statistical + transformation features
python scripts/prepare_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/features/features_4h_seq64_top50.npz

# Step 2b — build the five-branch Phase-1 Product feature bundle with frozen SSL branches
.venv/bin/python3 scripts/prepare_framework_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/features/features_4h_seq64_top50_phase1.npz \
  --vae-checkpoint checkpoints/vae_4h_seq64_top50.pth \
  --contrastive-checkpoint checkpoints/contrastive_4h_seq64_top50.pth \
  --byol-checkpoint checkpoints/byol_4h_seq64_top50.pth \
  --device cuda \
  --batch-size 1024 \
  --overwrite

# Validate feature dimensions, split metadata, and finite values
.venv/bin/python3 scripts/validate_feature_store.py \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --processed-npz data/processed/market_4h_seq64_top50.npz

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

# Pretrain the currently implemented Phase-2 contrastive backbone candidates.
# The selected-branch Phase-2 plan also requires matched BYOL LSTM/Transformer
# candidates; their BYOL-specific trainer is not implemented yet, so do not
# repurpose these NT-Xent commands for BYOL.
.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_lstm \
  --run-name contrastive_lstm-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --device cuda

.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_transformer \
  --run-name contrastive_transformer-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --device cuda

# Extract one frozen Phase-2 branch with checkpoint/data provenance
.venv/bin/python3 scripts/extract_phase2_encoder_features.py \
  --checkpoint checkpoints/phase2/contrastive_lstm-4h-seq64-top50-seed0.pth \
  --out-path data/features/phase2/contrastive_lstm_4h_seq64_top50_seed0.npz \
  --device cuda

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

# Build the isolated Phase-2 probability-movement labels (DOWN/STABLE/UP)
.venv/bin/python3 scripts/prepare_probability_movement_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz \
  --horizon 2 \
  --threshold 0.005 \
  --overwrite

# Build TA-MLP features and the common TA-eligible row intersection
.venv/bin/python3 scripts/prepare_phase2_ta_features.py \
  --labels-npz data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz \
  --out-path data/features/phase2_ta_probability_movement_4h_h2_tau005.npz \
  --overwrite

# Bootstrap the strict C1/C2/C5 Phase-2 classification matrix without running it
.venv/bin/python3 scripts/bootstrap_phase2_classification.py

# Resume/execute the frozen matrix (complete runs are skipped)
.venv/bin/python3 scripts/bootstrap_phase2_classification.py --execute

# Aggregate completed Phase-2 probabilistic-classification runs
.venv/bin/python3 scripts/report_phase2_classification.py \
  experiments/framework/phase2/classification_relabelling/4h_h2_tau005

# Build and verify the contract-local K=8 representation row map
.venv/bin/python3 scripts/prepare_phase2_temporal_index.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --context-length 8

# Build split-safe, contract-aware price labels for decoder refinement
.venv/bin/python3 scripts/prepare_price_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz \
  --horizon 1

# Freeze the 30-run D0-D4 price/volatility decoder matrix, then execute it
.venv/bin/python3 scripts/bootstrap_phase2_decoders.py
.venv/bin/python3 scripts/bootstrap_phase2_decoders.py --execute

# Aggregate replay-verified multi-seed decoder results and paired intervals
.venv/bin/python3 scripts/report_phase2_decoders.py \
  experiments/framework/phase2/decoder_refinement_1/4h_k8

# Canonical phase specifications:
# docs/phase_plan/2026-09-01-phase-1-product-readiness.md
# docs/phase_plan/phase1_experiment_observation_and_judgement.md
# docs/phase_plan/2026-09-08-phase-2-experiment-plan.md
# docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md
# docs/phase_plan/2026-09-08-phase-2-probabilistic-classification.md

# Execute the predeclared five-branch Phase-1 Product matrix and generate reports/plots
.venv/bin/python3 scripts/run_phase1_product.py \
  --stages features,train,plot \
  --run-name 4h_phase1_all5_concat \
  --device cuda

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

# Compare the completed Phase-1 volatility run against strict shared-label baselines
.venv/bin/python3 scripts/compare_volatility_phase1.py \
  --framework-root experiments/framework/phase1/volatility_prediction/4h_phase1_all5_concat

# Run the train-only Alpha101-style raw-OHLCV factor dry run. This leaves the
# global 20% task-test partition untouched and writes discovery/confirmation
# diagnostics only; it is not a trading backtest.
.venv/bin/python3 scripts/run_raw_alpha_dry_run.py \
  --timeframe 4h \
  --top-k 50 \
  --min-assets 10

# Run the constrained raw-OHLCV genetic-programming dry run. GP selection is
# discovery-only; the chronological confirmation segment is fixed-formula.
.venv/bin/python3 scripts/run_raw_gp_dry_run.py \
  --timeframe 4h \
  --top-k 50 \
  --min-assets 10 \
  --population-size 24 \
  --generations 3 \
  --max-depth 3 \
  --seed 7

# Exploratory direct-representation GP check. This tests fixed coordinates from
# the saved Phase-1 bundle and is intentionally distinct from model-output OOF
# alpha research; the global test partition remains excluded.
.venv/bin/python3 scripts/run_representation_gp_dry_run.py \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --top-k 50 \
  --min-assets 10

# Exhaustive exploratory screen: all 445 saved representation coordinates plus
# five causal OHLCV terminals; every terminal is seeded into the GP population.
.venv/bin/python3 scripts/run_representation_gp_dry_run.py \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --top-k 50 \
  --min-assets 10 \
  --include-raw-ohlcv \
  --all-representation-features \
  --population-size 512 \
  --generations 2 \
  --max-depth 4 \
  --seed 101 \
  --out-dir experiments/alpha/representation_ohlcv_gp_4h_top50_all_features
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
| `contrastive_lstm` | `src/models/encoder_variants.py` | 128 (Phase-2 experimental substitution; not yet trained) |
| `contrastive_transformer` | `src/models/encoder_variants.py` | 128 (Phase-2 experimental substitution; not yet trained) |

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
| `src/models/` | Model architecture definitions and loss functions only (VAE, contrastive CNN, BYOL, Phase-2 temporal backbone variants) |
| `src/training/` | Training loop functions (`train_vae_epoch`, `train_contrastive_epoch`, `train_byol_epoch`, Phase-2 temporal contrastive diagnostics) |
| `src/tasks/` | Task-owned heads, label builders, and experiment support. This includes the default `PriceRegressor`, `VolatilityRegressor`, and `TrendClassifier`; the shared volatility-label contract; `phase2_classification/` for isolated probability-movement labels, imbalance protocols, aligned loaders, probabilistic metrics, models, and artifact-producing training; and `phase2_decoders/` for contract-local row maps, D0–D4 models, fixed-budget training, replay, and reporting. |
| `src/alpha/` | Training-only alpha research: downstream-prediction primitives, chronological OOF utilities, shallow protected formulae/selection, plus causal raw-OHLCV Alpha101-style diagnostics and a bounded genetic-programming dry run. |
| `src/evaluation/` | Unified metrics (`regression_metrics`, `mse_and_corr`, `classification_metrics`) |
| `src/baselines/` | Comparison models — `lstm_baseline/` (external price benchmark), `raw_lstm_volatility/` and `garch_lstm_stacking/` (external volatility benchmarks), `mlp_baseline/` (internal), `ta_mlp_baseline/` (external trend benchmark), `ginn_baseline/` (volatility limitation evidence) |
| `scripts/` | Runnable entry points; each inserts `src/` into `sys.path` |

### Key data contracts

- Raw feather files must have columns: `open`, `high`, `low`, `close`, `volume`
- Processed `.npz`: keys `train` and `test`, both `float32` of shape `[N, seq_len, 5]`
- Feature `.npz` (via `NpzFeatureStore`): keys `statistical`, `transformed`, plus one key per frozen neural branch such as `vae`, `contrastive`, or `byol`; a companion `.index.npz` stores `train_size`/`test_size` to recover the split after train+test concatenation. Legacy files with an empty or packed `neural` key remain loadable, but new neural features should be stored by branch name.
- Trend task label `.npz`: saved under `data/task_labels/trend_classification/`; keys include `train_labels`, `test_labels`, aligned train/test row indices, class names, and train-fitted threshold metadata. Horizon rows are dropped inside each split so labels never cross the train/test boundary.
- Phase-2 movement label `.npz`: hard `DOWN/STABLE/UP` targets from absolute future probability movement plus aligned row indices, contract IDs, window starts, timestamps, current/future close, and realised delta. The TA feature bundle stores the frozen common eligible-row intersection used by strict C1/C2/C5 comparisons.
- Phase-4 probability-movement regression: planned continuous
  `close[t+h] - close[t]` labels constructed independently inside each
  contract and global calendar walk. Each primary walk requires separately
  trained encoder/head weights from information available before its cutoff;
  lifecycle position is reported within walks. Exact zero movement is the
  mandatory primary reference. Arithmetic-return regression is secondary and
  must report starting-price sensitivity. The builder and canonical artifact
  path are not implemented yet.
- Volatility task label `.npz`: saved under `data/task_labels/volatility_prediction/`; contains realised-volatility targets, aligned train/test row indices, contract IDs, and window starts. The Raw LSTM, GARCH--LSTM stack, framework volatility run, and Raw-OHLCV MLP volatility rerun must use this bundle for strict comparison; older MLP volatility artifacts are characterization-only.
- Phase-2 decoder temporal index: `data/features/phase2/temporal_index_4h_seq64_top50_k8.npz`; stores split-local `[N, 8]` feature-row contexts plus final row, contract, window-start, timestamp, hashes, and source provenance. Static D0--D2 and temporal D3--D4 use identical eligible final rows.
- Price label `.npz`: `data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz`; stores horizon-1 close targets and contract-safe identities built independently inside each stored split. It is required for new price experiments, not only decoder refinement. The final row of every contract is excluded, giving 109,791 train / 27,450 test eligible rows before any decoder-specific context restriction. Legacy Phase-1 and early Phase-2 runs with `labels_npz: null` used 109,840/27,499 merged-array rows and retained 49 invalid cross-contract transitions per split; see `docs/price_prediction_label_contract.md`.
- GARCH feature vector per column: `[omega, alpha, beta, persistence, uncond_var, mean_cond_var, std_cond_var]`
