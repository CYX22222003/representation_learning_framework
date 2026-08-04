# GINN Baseline

This baseline is a framework-aligned implementation of the external GINN volatility benchmark. It keeps the GINN identity: AR residual preprocessing, GARCH(1,1) volatility supervision, and a two-layer LSTM variance predictor. It intentionally follows this project's train/test-only experiment rules instead of the reference repository's validation-based checkpoint selection.

See `LIMITATIONS.md` for the observed GARCH-supervision failure mode on sparse
or near-static Polymarket contracts.

## Protocol

- Data scope: top-50 Polymarket contracts for the selected timeframe, initially `4h`.
- Split: chronological 80/20 per contract, with no validation split and no early stopping.
- Sequence length: `64`.
- Input: five AR-residual OHLCV channels.
- Statistical fitting: AR(5), close-residual GARCH parameters, residual mean, and initial variance are fit only on training-visible rows.
- Target: next-shifted-window realised close volatility.
- GARCH supervision: shifted-window GARCH volatility, not variance.
- Output transform: `linear` preserves the upstream architecture; `softplus`
  is available as a Polymarket adaptation to prevent invalid negative
  volatility predictions.
- Metrics: MSE and Pearson correlation against realised volatility. MAE, RMSE, GARCH-only metrics, negative predictions, and per-contract rows are diagnostics.
- Sweep: one seed-0 trajectory through epoch 100, with checkpoints reported at `15,20,25,50,100`. No checkpoint is selected from test performance.

## Commands

Run from the project root in WSL:

```bash
PYTHONPATH=src .venv/bin/python3 src/baselines/ginn_baseline/prepare_data.py \
  --timeframe 4h --seq-len 64 --top-k 50 --ar-order 5
```

```bash
PYTHONPATH=src .venv/bin/python3 src/baselines/ginn_baseline/run_experiment.py \
  --run-name 2026-08-04-smoke --epoch-budgets 1 --seed 0 \
  --batch-size 64 --device cpu
```

```bash
PYTHONPATH=src .venv/bin/python3 src/baselines/ginn_baseline/run_experiment.py \
  --run-name 2026-08-04-v1 --epoch-budgets 15,20,25,50,100 \
  --seed 0 --batch-size 64 --learning-rate 1e-4 \
  --lambda-garch 0.3 --device auto
```

Diagnostic non-negative output run:

```bash
timeout 45m env PYTHONPATH=src .venv/bin/python3 src/baselines/ginn_baseline/run_experiment.py \
  --run-name 2026-08-04-e15-softplus \
  --epoch-budgets 15 \
  --seed 0 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --lambda-garch 0.3 \
  --output-transform softplus \
  --device cuda
```

```bash
PYTHONPATH=src .venv/bin/python3 src/baselines/ginn_baseline/plot_experiment.py \
  src/baselines/ginn_baseline/experiments/2026-08-04-v1
```

## Cache

Default cache files:

```text
data/processed/ginn_4h_seq64_top50.npz
data/processed/ginn_4h_seq64_top50.manifest.json
```

The NPZ contains `X_train`, `y_gt_train`, `y_garch_train`, `contract_id_train`, `X_test`, `y_gt_test`, `y_garch_test`, and `contract_id_test`.

The manifest records configuration, ordered source files and digests, dataset ID, included/skipped contracts, GARCH fallback status, array shapes and dtypes, and quality counts. Training rejects a cache whose manifest disagrees with the expected preprocessing contract.

## Artifacts

The canonical run writes:

```text
src/baselines/ginn_baseline/experiments/2026-08-04-v1/
  config.json
  dataset_manifest.json
  sweep_metrics.json
  summary.md
  images/epoch_sweep.png
  e15|e20|e25|e50|e100/
    checkpoint.pth
    history.npz
    predictions.npz
    metrics.json
    summary.md
    images/
      training_curve.png
      pred_vs_actual.png
      error_distribution.png
```

Checkpoints and logs are ignored. JSON, Markdown, NPZ histories/predictions, and PNG result visuals are intended to be versioned for completed canonical runs.

## 4h Result Index

Canonical sweep pending. The row below is a first seed-0 CUDA characterization
entry at the 15-epoch budget; do not treat it as checkpoint selection.

| output_transform | epoch | mse | pearson_corr | mae | rmse | negative_prediction_fraction | final_fused_loss | artifact_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `linear` | 15 | 1887.2247314453 | 0.0487895527 | 12.7260322571 | 43.4421997070 | 0.6284000000 | 4792.0668945312 | `src/baselines/ginn_baseline/experiments/2026-08-04-e15-codex-gpu/e15` |
| `softplus` | 15 | 1313.9035644531 | -0.0702624972 | 14.3032207489 | 36.2478065491 | 0.0000000000 | 4820.1425781250 | `src/baselines/ginn_baseline/experiments/2026-08-04-e15-softplus-codex-gpu/e15` |
