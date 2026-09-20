# Phase 3 entry points

Only files in this directory may serve as Phase 3 executable entry points.
They may import reviewed implementation modules from `src/`, but they do not
invoke files under the legacy `scripts/` directory.

The encoder-pretraining workflow is intentionally gated:

```bash
# 1. Build and immediately validate the raw-time-first 4h encoder bundle.
.venv/bin/python3 scripts_v2/prepare_phase3_encoder_data.py

# 2. Independently repeat the provenance and leakage checks.
.venv/bin/python3 scripts_v2/validate_phase3_encoder_data.py

# 3. Freeze the seven-run matrix without launching training.
.venv/bin/python3 scripts_v2/launch_phase3_encoder_pretraining.py --device cuda

# 4. Review the frozen manifest, then explicitly execute it.
.venv/bin/python3 scripts_v2/launch_phase3_encoder_pretraining.py --execute

# 5. Validate completed artifacts and generate numerical/visual reports.
.venv/bin/python3 scripts_v2/report_phase3_encoder_pretraining.py
```

The initial contract is fixed to 4-hour data, sequence length 64, top 50,
seed 0, and one uninterrupted trajectory with checkpoints at epochs 5, 15,
and 50. Epoch 50 is predeclared for downstream feature extraction. Existing
outputs are never overwritten by default.

Canonical locations:

```text
data/phase3/processed/market_4h_seq64_top50.npz
experiments/phase3/manifests/encoder_pretraining_seed0.json
experiments/phase3/encoder_pretraining/<family>/<backbone>/seed0/
experiments/phase3/reports/encoder_pretraining_seed0/
```

The report command requires all seven runs and all three snapshots to be
complete. It rejects checkpoint, dataset-provenance, trajectory, or budget
mismatches before writing CSV, JSON, Markdown, or PNG outputs.

## Price-prediction workflow

After encoder pretraining is complete:

```bash
# Build contract-local horizon-1 targets; terminal rows are dropped per contract.
.venv/bin/python3 scripts_v2/prepare_phase3_price_labels.py

# Extract window-local statistical/transformation features and every frozen
# epoch-50 neural branch into one provenance-checked master store.
.venv/bin/python3 scripts_v2/extract_phase3_features.py --device cuda

# Freeze the complete H0/HC/HB framework matrix without executing it.
.venv/bin/python3 scripts_v2/launch_phase3_price.py --device cuda

# Review the matrix manifest before explicitly launching the 15 trajectories.
.venv/bin/python3 scripts_v2/launch_phase3_price.py --execute

# Replay all predictions and create consolidated numerical/visual results.
.venv/bin/python3 scripts_v2/report_phase3_price.py
```

The statistical branch fits AR(5) and GARCH(1,1) only inside each stored input
window. The transformation branch computes FFT magnitudes and Haar energies
only from that same window. Neither branch fits state across samples or uses a
future target. Downstream standardization is fitted after contract-safe label
eligibility is applied and uses selected training rows only.
