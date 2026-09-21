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

This section records the pre-December-2025 Phase 4 feasibility analysis. Its
four-hour top-80 walk contract was not implemented or trained. Phase 4 later
concluded by selecting recent clean native one-hour FinData with causal
isolated-one-bar filling for the exploratory Phase 5 loop. See
`docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`.
The commands below remain supporting diagnostics, not a current training
builder.

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

## Recent FinData acquisition audit

The independent acquisition module under `src/data_acquisition/` downloads
recent Polymarket data without creating model splits or training artifacts.
The default interval is `[2025-12-01, 2026-09-01)` UTC, so it includes all of
31 August 2026. Outputs go only under the Git-ignored `data_new/` root.

```bash
.venv/bin/python3 scripts_v2/collect_findata_prediction_markets.py \
  --top-k 3 \
  --candidate-limit 30 \
  --workers 6

.venv/bin/python3 scripts_v2/analyze_findata_prediction_markets.py
```

The collector stores native sparse 15-minute and one-hour Parquet files. The
four-hour file is an explicit UTC-aligned derivative of the native one-hour
rows because the documented native `4h` API query was not usable in the live
probe. Missing source hours are not filled during acquisition; every derived
bin records its observed-hour count and completeness. The current-snapshot MVP
selection is exploratory and must not be treated as the cutoff-local Phase 5
universe. See
`docs/data_analysis/2026-09-20-findata-recent-polymarket-mvp.md`.

For the expanded retrospective cohort and token-safe fallback:

```bash
.venv/bin/python3 scripts_v2/collect_findata_historical_cohort.py \
  --top-k 50 --candidate-limit 250 --max-per-family 2 --workers 8 \
  --defer-quarantine \
  --output-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31

.venv/bin/python3 scripts_v2/analyze_findata_prediction_markets.py \
  --input-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31

.venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py \
  --input-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31 \
  --audit-only

.venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py \
  --input-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31

.venv/bin/python3 scripts_v2/analyze_findata_native_dynamics.py \
  --maximum-fill-bars 1 --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_native_gaps.py --overwrite

.venv/bin/python3 scripts_v2/plot_findata_native_ohlcv.py --overwrite

.venv/bin/python3 scripts_v2/collect_findata_yes_trade_ohlcv.py --workers 8
.venv/bin/python3 scripts_v2/analyze_findata_yes_trade_ohlcv.py
```

The condition-level candle endpoint can mix complementary YES/NO observations.
Raw native files remain immutable source-audit artifacts. With
`--defer-quarantine`, collection finishes before a non-destructive
`--audit-only` preview is reviewed. Rule
`condition-orientation-v2-forward-confirmed` uses up to four later bars to
retain a persistent crash/repricing and quarantine a transient complementary
reversion or unsupported wide candle. Every decision records
`quarantine_available_at`; a walk-forward consumer may use it only after that
timestamp. Application writes versioned `*_clean.parquet` files,
`candle_quarantine.parquet`, and a hash-bearing manifest, preserves timestamp
gaps, and fails if more than 1% of either native resolution would be removed.
Retention above 99% is an operational budget, not proof of YES-token identity.

The native-frequency dynamics command reads only the new FinData 15-minute and
one-hour raw/clean artifacts. Its default coverage sensitivity fills at most
one missing native bar with the previous close, never overwrites source files,
and separately reports every target touching a synthetic row. Longer gaps stay
missing and no random augmentation is applied. Observed-only movement is the
primary descriptive result. See
`docs/data_analysis/2026-09-21-findata-native-15m-1h-dynamics.md`.

The OHLCV plotting command reads the clean native files, dynamically restricts
each x-axis to that contract-resolution's actual first and last candle, applies
the same isolated-one-bar flat-OHLC/zero-volume fill, and writes one five-panel
figure per contract and resolution. Imputed rows are marked, while longer gaps
remain visible line breaks. A CSV/JSON manifest records source hashes, actual
plot bounds, metadata overlap bounds, fill counts, and plot hashes.

The native-gap audit measures every internal missing slot, its run length,
per-contract coverage, the small marginal effect of pruning, and disagreement
between the independent 15-minute and one-hour endpoints. It does not fill or
modify either source file. The saved raw and clean files remain irregular;
forward filling exists only in the separate dynamics sensitivity.

The trade collector maps the declared `Yes` outcome to its token ID and
aggregates only those trades, without using settlement information. That valid
fallback is currently too sparse for a diverse seq64 encoder cohort. See
`docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md`.

Phase 4 concludes this audit by selecting the clean native one-hour condition
candles for an explicitly exploratory Phase 5 walk-forward loop. Only complete
isolated one-hour gaps may be filled, synthetic rows are context-only, longer
gaps split sequences, and native 15-minute data remains a sensitivity. The
condition-candle token-identity limitation and affected-contract exclusion
sensitivity remain mandatory.

## Recent one-hour walk-capacity audit

The following diagnostic applies isolated-one-bar filling, longer-gap sequence
breaks, observed decision/target endpoints, target maturity, causal prior-24h
activity, and global calendar cutoffs before counting sequences:

```bash
.venv/bin/python3 scripts_v2/analyze_findata_walk_capacity.py --overwrite
```

The balanced two-walk `seq64` candidate retains 31,828 and 37,173 active
training rows, with 21,401 and 10,086 supported evaluation rows. The mandatory
affected-contract exclusion retains 19,903/26,927 training and 15,622/7,445
evaluation rows. `seq256` and five monthly walks are weaker or concentrated.
These results establish feasibility only: the current 50 conditions were
selected retrospectively, so cutoff-local universe construction and causal
quarantine replay remain required before training. See
`docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md`.
