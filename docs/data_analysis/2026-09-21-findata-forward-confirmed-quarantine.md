# FinData Forward-Confirmed Condition-Candle Quarantine

## Decision

The FinData condition-candle issue is rare in aggregate but not uniformly rare
by contract. The local mitigation is therefore an audited exploratory filter,
not a conversion into canonical YES prices. Raw files remain immutable and the
token-identified trade path remains the only collected source with direct
outcome provenance.

Rule `condition-orientation-v2-forward-confirmed` uses up to the next four
observed bars within four expected intervals. At least two confirmation bars
must exist, and at least two thirds must support the same regime. A large move
is retained when future closes stay closer to the new level; this preserves a
genuine crash or other persistent repricing. A complementary transition is
quarantined when future closes revert toward the prior level. An unsupported
wide-range candle is also quarantined. A wide candle immediately following an
initially flagged candle is quarantined when its OHLC is contaminated by that
transition. Prices are never inverted or repaired.

Every audit row stores `confirmation_end_date` and
`quarantine_available_at`. This makes the look-ahead explicit: a walk-forward
consumer may use the quarantine decision only when `quarantine_available_at`
is strictly before its cutoff. Using the final clean file at an earlier cutoff
would leak future information.

## Raw-first expanded collection

The collector first wrote raw data only for 50 retrospectively selected
conditions over `[2025-12-01, 2026-09-01)` UTC:

| artifact | rows |
|---|---:|
| native 15-minute candles | 325,730 |
| native one-hour candles | 107,635 |
| raw one-hour-derived four-hour bins | 30,001 |

Selection probed 250 diverse candidates from 605 catalog rows meeting the
overlap-duration criterion. The final cohort covers eight heuristic categories.
This full-lifetime selection is appropriate for source diagnosis only and is
not a cutoff-local Phase 4 universe.

Before pruning, the raw hourly diagnostic found 268 approximately
complementary close transitions and 19 bars with `high-low > 0.5`, affecting
14 markets. These raw heuristics deliberately overcount because they cannot
distinguish a persistent repricing from a transient outcome switch.

## Forward-confirmed preview

The non-destructive preview investigated all 50 contracts and found quarantine
candidates in 13. It produced the following decision:

| resolution | raw rows | candidates | candidate share | retained | affected contracts |
|---|---:|---:|---:|---:|---:|
| native 15-minute | 325,730 | 116 | 0.0356% | 99.9644% | 10 |
| native one-hour | 107,635 | 147 | 0.1366% | 99.8634% | 12 |

Both rates are below the predeclared 1% global fail-closed budget. The most
affected individual contract has 0.5760% of its 15-minute rows flagged. At one
hour, three short contracts have 4.08%--4.40% flagged and another has 1.98%.
This concentration must be reported; an affected-contract exclusion
sensitivity remains mandatory for any exploratory analysis using these files.

The filter retained persistent large movements rather than classifying every
drastic change as an error:

| resolution | raw large jumps | future-persistent retained regime changes | future-reverting large jumps | unconfirmed large jumps |
|---|---:|---:|---:|---:|
| 15-minute | 267 | 112 | 103 | 52 |
| one-hour | 281 | 137 | 134 | 10 |

The exact quarantine composition is 97 complementary reversions plus 19 wide
candles at 15 minutes, and 129 complementary reversions plus 18 wide candles
at one hour. A single retained hourly wide candle is supported as a persistent
new regime. The rule deliberately leaves unconfirmed transitions untouched.

## Applied artifacts and verification

After the preview was reviewed, the same rule produced:

- `candles_15min_clean.parquet` with 325,614 rows;
- `candles_1h_clean.parquet` with 107,488 rows;
- `candles_4h_derived_clean.parquet` with 30,001 bins and explicit hourly
  coverage fields;
- `candle_quarantine.parquet` with 263 resolution-specific audit rows;
- `quarantine_manifest.json` with raw hashes, parameters, counts, and clean
  artifact hashes.

Verification confirmed that raw hashes did not change, the clean/raw row-count
differences exactly equal the audit counts, and no quarantined
`condition_id,date` key remains in either clean native file. Removed timestamps
remain gaps; downstream contexts and targets must require exact consecutive
timestamps and complete derived-hour coverage.

The raw four-hour derivative has 22,840 complete bins. Clean derivation has
22,705 complete bins: 135 bins became incomplete after hourly quarantine, and
six bins lost two observed hours. No missing hour was imputed. Every one of the
263 audit rows has a non-null availability timestamp; the decision delay is
45--75 minutes for 15-minute bars and four--five hours for hourly bars because
availability is measured after the final confirmation candle closes.

The machine-readable preview and its per-contract report are stored in the
Git-ignored cohort directory as `quarantine_preview.json`,
`candle_quarantine_preview.parquet`, and `quarantine_preview_report.md`.

## Interpretation

The aggregate anomaly rate is operationally small enough for the local filter,
but 99% retention is not evidence that remaining rows share one outcome
orientation. Complement changes near 0.5 may evade this rule, persistent
outcome switches are intentionally retained to avoid deleting genuine crashes,
and the candle schema still lacks token identity. These artifacts can support
a transparent exploratory sensitivity only. They do not authorize Phase 4
training or replace a token-specific price history.

## Reproduction

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
```
