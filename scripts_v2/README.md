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

## Phase 4 walk-forward data analysis

The canonical post-analysis decision is frozen in
`docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`.
The primary configuration is four-hour top-80 under two fixed-duration rolling
calendar walks with cutoff-local universe ranking and a causal prior-24h
price-change mask. The commands below are supporting diagnostics, not the final
training builder.

These commands perform exploratory top-80 data analysis only. They do not
prepare training bundles or launch models. Both schemes use identical
contract-relative evaluation fractions. One keeps the relative start anchored
at zero; the other advances a fixed-width relative training window:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py --scheme relative_anchored
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py --scheme relative_window
.venv/bin/python3 scripts_v2/report_phase4_walk_data.py
```

The analysis covers signed horizon-2 probability movement, arithmetic return,
and fixed-threshold `DOWN/STABLE/UP` labels. The fixed top-80 cohort comes from
the Phase 3 per-contract prefix ranking. Relative position uses each contract's
final observed row count, so this is retrospective lifecycle evidence rather
than a causal pooled-model backtest. Relative cutoffs from different contracts
must not be treated as simultaneous calendar cutoffs.

The default command diagnoses three walks. A two-walk coverage sensitivity is
stored separately so it cannot overwrite the default artifacts:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py \
  --scheme relative_anchored --walk-count 2 \
  --output-root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py \
  --scheme relative_window --walk-count 2 \
  --output-root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
.venv/bin/python3 scripts_v2/report_phase4_walk_data.py \
  --root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
```

All 80 contracts contribute to every corrected relative evaluation interval.
Both two and three walks are feasible by aggregate sample count; three walks
give finer lifecycle resolution, while two reduce compute. Neither relative
scheme replaces the required global-calendar evaluation for deployment claims.

## One-hour timestamp and capacity audit

The following diagnostic verifies timestamp quality for the same top-80
contract identities and compares two- and three-walk sample capacity at
sequence lengths 64 and 256. The latter preserves the 256-hour context duration
of the four-hour, length-64 encoder inputs.

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_timestamp_capacity.py
```

This command performs no model training and writes only data-analysis artifacts
under `experiments/phase4/data_analysis/top80_1h_timestamp_capacity/`.

To examine whether those rows contain enough trading activity and target
movement to support meaningful modelling, run:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_activity_suitability.py
```

This audit measures per-contract volume coverage, unchanged-price fractions,
longest stale runs, eight-hour probability-movement distributions, lifecycle
drift, and the two-walk global-calendar evaluation intervals. It does not train
or select a model.

For a duration-matched top-50/top-80 comparison between one-hour and four-hour
data, run:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_4h_activity_comparison.py
```

The comparison fixes the context at 256 hours, target horizon at eight hours,
and causal activity lookback at 24 hours. It uses the same early-prefix-ranked
contract identities at both frequencies and launches no training.
