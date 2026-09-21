# FinData Polymarket Data-Exploration SOP

## Purpose

Use this procedure to expand the recent Polymarket condition-candle cohort
while keeping source anomalies, missingness, deterministic filling, and
staleness visible. This SOP standardizes exploration; it does not decide
whether a cohort is fit for training.

## 1. Freeze the Request

Before network acquisition, record:

- fixed UTC interval: `[2025-12-01T00:00:00Z, 2026-09-01T00:00:00Z)`;
- user-provided contract count `K`;
- both native resolutions: 15 minutes and one hour;
- optional inclusion/exclusion keywords or event families;
- candidate limit, ranking fields, diversity logic, event-family cap, and
  deterministic tie-breaker selected for this collection; and
- output root and whether it is new or an intentional overwrite.

The current collector ranks a historical catalog using volume, heuristic topic
diversity, and an event-family cap. It does not expose general keyword or family
filters. If the user supplies filters, inspect the discovered catalog and
extend or wrap the selection logic before fetching; never claim a filter was
used when it was not. Record requested filters, matched candidates, exclusions,
and the final selection order.

The existing collection entry point fetches both supported native resolutions,
which matches the current scope:

```bash
.venv/bin/python3 scripts_v2/collect_findata_historical_cohort.py \
  --top-k <K> \
  --candidate-limit <DECLARED_LIMIT> \
  --max-per-family <DECLARED_CAP> \
  --workers 8 \
  --defer-quarantine \
  --output-dir data_new/findata/polymarket/historical_diverse_top<K>_2025-12-01_2026-08-31
```

Use `--defer-quarantine`: acquisition must finish with immutable raw candle
files before any pruning. Authentication remains runtime-only and must never
enter logs, manifests, or reports.

Verify after collection:

- every timestamp is inside the global requested interval;
- unique `condition_id,date` identities and monotonic per-contract time;
- finite OHLCV, nonnegative volume, and valid OHLC ordering;
- 15-minute and one-hour contract coverage;
- source and metadata hashes; and
- the selected catalog/ranking is reproducible.

## 2. Audit Possible YES/NO Mixture

Run the raw source analysis, then the non-destructive quarantine preview:

```bash
.venv/bin/python3 scripts_v2/analyze_findata_prediction_markets.py \
  --input-dir <COHORT_ROOT>

.venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py \
  --input-dir <COHORT_ROOT> \
  --audit-only
```

Use the repository's versioned forward-confirmed rule. It may use up to four
later observed bars to distinguish a transient complementary reversion or
unsupported wide candle from a persistent repricing. Prices are never inverted
or repaired. Every candidate must retain its reason, confirmation window, and
`quarantine_available_at`.

Summarize before any pruning:

- raw rows, candidates, candidate rate, retained rate, and affected contracts
  per resolution;
- reason counts and per-contract candidate fractions;
- the most affected contracts and whether any exceeds 1%;
- persistent large changes retained by the rule;
- unconfirmed large changes deliberately retained; and
- the limitation that high retention does not establish YES-token identity.

The batch-level fail-closed limit is 1% for each resolution. A rate below 1%
permits review; it does not authorize pruning automatically. If either rate is
above 1%, do not apply the filter—report and investigate the source/cohort.

### Mandatory stop

After presenting the preview, ask the user to approve applying those exact
quarantine decisions. End the turn. Do not create or overwrite clean artifacts
until explicit approval is received.

## 3. Apply the Approved Quarantine

After approval only:

```bash
.venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py \
  --input-dir <COHORT_ROOT>
```

Verify that:

- raw hashes are unchanged;
- clean/raw row-count differences equal the audit counts;
- no quarantined identity remains in either clean native file;
- removed timestamps remain gaps; and
- the quarantine manifest hashes the raw inputs, clean outputs, parameters,
  rule version, counts, and decision-availability timestamps.

The resulting `candles_15min_clean.parquet` and
`candles_1h_clean.parquet` are the **observed-only clean** layer: anomalies are
pruned, but no timestamp has been forward-filled.

## 4. Investigate Native Gaps

Audit gaps after quarantine and before filling:

```bash
.venv/bin/python3 scripts_v2/analyze_findata_native_gaps.py \
  --input-dir <COHORT_ROOT> \
  --output-dir <COHORT_ROOT>/analysis/native_gap_audit \
  --overwrite
```

Report separately for 15-minute and one-hour data:

- observed rows, internal missing slots, and pooled/contract-median coverage;
- contracts below useful coverage thresholds such as 75% and 50%;
- gap-event counts and missing-row counts by run length;
- one-bar gaps, two-to-four-bar gaps, longer gaps, and the maximum run;
- concentration of missing rows in the top 10 and top 20 contracts;
- how many missing slots were introduced by quarantine; and
- cross-resolution evidence where one endpoint observed activity while the
  other did not.

Do not interpret every absent candle as verified inactivity. A few missing
points are handled by the bounded rule below; gap statistics must remain visible
instead of being erased by unlimited filling.

## 5. Build the Separate Bounded-Fill Layer

Create derived filled files separately from both raw and clean files. Never
overwrite either source layer. The standard rule is:

1. Work independently inside each contract and native resolution.
2. Fill only a **complete isolated one-bar internal gap** bounded by observed
   candles on both sides.
3. Set synthetic `open=high=low=close` to the previous observed close.
4. Set synthetic `volume=0`.
5. Do not fill leading gaps, trailing gaps, or runs longer than one native bar.
6. Preserve longer gaps as absent timestamps so later windows break there.
7. Add at least `is_observed`, `is_imputed`, `original_gap_length_bars`, and
   `time_since_last_observation` metadata.
8. Preserve condition ID, native resolution, timestamp, source hashes, and
   deterministic row ordering.

Use distinct names such as:

```text
candles_15min_clean_ffill1.parquet
candles_1h_clean_ffill1.parquet
fill_manifest.json
```

At present, `scripts_v2/analyze_findata_native_dynamics.py` simulates bounded
filling in memory; it does not materialize this derived layer and its input
root is hardcoded. Before claiming that reusable processed files exist, reuse
or implement a dedicated builder under `scripts_v2/`, add explicit
`--input-dir`/`--output-dir` arguments, and validate it with focused tests.
Do not silently run that script against its default top-50 path for a new
cohort.

The fill manifest must prove:

- observed OHLCV values and identities exactly match the clean source;
- every inserted row belongs to a one-bar internal gap;
- inserted OHLC is flat at the prior close and inserted volume is zero;
- no long gap, leading interval, or trailing interval was filled;
- output row counts equal clean rows plus inserted rows; and
- deterministic replay reproduces identities and hashes.

## 6. Analyze Filling Effects and Staleness

Use the filled series for the requested staleness analysis, but always place it
beside the observed-only clean result so the effect of synthetic rows is
quantified rather than hidden.

Standard horizons:

- native 15-minute: 15, 30, and 60 minutes;
- native one-hour: 1 and 2 hours.

For each resolution/horizon, require exact native-cadence paths and report:

- target count;
- exact-zero and non-zero movement rates;
- shares with `abs(delta) > 0.005` and `abs(delta) > 0.01`;
- mean absolute movement and, when useful, RMS movement;
- number/share of targets touching an imputed row; and
- the same metrics on the fill-touched and untouched subsets.

Use these modes consistently:

- `observed_only_clean`: post-quarantine clean rows with no fill in the path;
- `filled_all`: all exact-cadence targets on the bounded-fill series;
- `filled_untouched`: filled-series targets whose paths contain no imputation;
- `filled_touched`: filled-series targets whose paths contain at least one
  imputed row.

`observed_only_clean` should replay `filled_untouched` for the same exact paths.
Treat a mismatch as a pipeline error. State clearly that filling repairs
context continuity and mechanically increases zero movement; it does not create
new observed market information.

The current dynamics script can produce the comparison for the existing top-50
cohort. Generalize its input/output arguments before using it for another
cohort, then run with `--maximum-fill-bars 1`.

## 7. Plot Every Contract

After the clean/fill audit, produce one five-panel Open/High/Low/Close/Volume
figure per contract and native resolution. Use each contract-resolution's
actual first and last clean candle as its x-axis range. Mark imputed rows and
leave longer gaps as visible line breaks.

For the current top-50 cohort:

```bash
.venv/bin/python3 scripts_v2/plot_findata_native_ohlcv.py --overwrite
```

For another cohort, pass explicit input/output roots. The plot manifest must
record titles, condition IDs, observed and inserted row counts, actual plot
bounds, source hashes, and plot hashes.

## 8. Final Exploration Handoff

Return an evidence-first summary containing:

1. request and selection configuration;
2. artifact lineage and hashes;
3. quarantine preview, approval, and applied results;
4. gap structure and concentration;
5. bounded-fill counts and exposure;
6. observed-only versus filled staleness;
7. links to plots and machine-readable manifests; and
8. limitations, including outcome-token ambiguity and any code path that had
   to be generalized.

Do not declare the cohort training-ready. Report the evidence and let the user
make that separate research judgement.
