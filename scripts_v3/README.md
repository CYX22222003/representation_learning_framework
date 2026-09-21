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
