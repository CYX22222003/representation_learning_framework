# Phase 5 entry points

Phase 5 executable entry points live in this directory. Reusable data,
training, and evaluation logic belongs under `src/`; `scripts_v3/` contains
only argument parsing, orchestration, and artifact writing.

The data-preparation gate is run before any encoder or downstream training:

```bash
.venv/bin/python3 scripts_v3/prepare_phase5_data.py
.venv/bin/python3 scripts_v3/validate_phase5_data.py
```

The commands consume the approved walk-specific
`candles_1h_clean_ffill1.parquet` files without modifying them and write one
replayable bundle per walk under `experiments/phase5/data_preparation/`.
Existing outputs are never overwritten unless `--overwrite` is supplied.

## Canonical encoder pretraining

The six-run matrix trains VAE, contrastive CNN, and BYOL CNN independently for
Walk 1 and Walk 2. Every run is one seed-0 50-epoch trajectory with snapshots
at 5, 15, and 50; epoch 50 is fixed for downstream feature extraction.

```bash
.venv/bin/python3 scripts_v3/launch_phase5_encoder_pretraining.py --device cuda
.venv/bin/python3 scripts_v3/launch_phase5_encoder_pretraining.py --execute
.venv/bin/python3 scripts_v3/validate_phase5_encoder_pretraining.py
.venv/bin/python3 scripts_v3/report_phase5_encoder_pretraining.py
```

The matrix is complete under `experiments/phase5/encoder_pretraining/`.

## Frozen features and framework downstream probes

The completed seed-0 framework stage extracts the canonical 445-dimensional
features, fits one supervised-train-only scaler per walk, trains both heads for
50 epochs, replays every snapshot, and writes the pooled report:

```bash
.venv/bin/python3 scripts_v3/launch_phase5_feature_extraction.py --device cuda --workers 6
.venv/bin/python3 scripts_v3/validate_phase5_features.py
.venv/bin/python3 scripts_v3/launch_phase5_downstream.py --device cuda
.venv/bin/python3 scripts_v3/validate_phase5_downstream.py
.venv/bin/python3 scripts_v3/report_phase5_downstream.py
```

Canonical artifacts live under `experiments/phase5/features/`,
`experiments/phase5/downstream/`, and
`experiments/phase5/reports/framework_downstream_seed0/`.

## Additional regression tasks

The completed post-primary tasks test eight-hour raw probability
change and two-hour ordinary log return without changing the 64-hour encoders:

```bash
.venv/bin/python3 scripts_v3/prepare_phase5_downstream_addon_data.py
.venv/bin/python3 scripts_v3/prepare_phase5_downstream_addon_features.py --device cuda --workers 6
.venv/bin/python3 scripts_v3/launch_phase5_regression_addons.py --device cuda
.venv/bin/python3 scripts_v3/validate_phase5_regression_addons.py
.venv/bin/python3 scripts_v3/report_phase5_regression_addons.py
.venv/bin/python3 scripts_v3/report_phase5_regression_rank_ic.py
```

The completed eight-hour absolute future-price probe reuses those same rows
and features:

```bash
.venv/bin/python3 scripts_v3/launch_phase5_absolute_price_h8.py --device cuda
.venv/bin/python3 scripts_v3/validate_phase5_absolute_price_h8.py
.venv/bin/python3 scripts_v3/report_phase5_absolute_price_h8.py
```
