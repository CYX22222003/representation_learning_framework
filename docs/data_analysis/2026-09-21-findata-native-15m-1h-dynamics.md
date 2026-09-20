# FinData Native 15-Minute and One-Hour Dynamics

> **Phase handoff:** The consolidated quantitative result and Phase 5 source
> decision are recorded in
> `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`.
> This document remains the detailed native-frequency evidence.

## Scope

This audit uses only the newly collected 50-condition FinData cohort. It does
not read the old feather archive and does not use derived four-hour data. Raw
and forward-confirmed clean files are compared at their native frequencies.
No random augmentation is applied.

The clean inputs contain 325,614 native 15-minute rows and 107,488 native
one-hour rows. Probability movement is measured only across exact native
cadence for the observed-only result. The filling sensitivity inserts the
previous close for one missing bar, marks the row synthetic, and leaves longer
gaps missing.

## Observed-data dynamics after pruning

| Native source | Horizon | Targets | Exact zero | Non-zero | `abs(delta) > 0.005` | `abs(delta) > 0.01` | Mean absolute move |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15-minute | 15 minutes | 265,672 | 31.53% | 68.47% | 17.32% | 9.63% | 0.002765 |
| 15-minute | 30 minutes | 231,649 | 28.46% | 71.54% | 19.42% | 11.44% | 0.003337 |
| 15-minute | 60 minutes | 192,089 | 25.15% | 74.85% | 22.24% | 13.99% | 0.004316 |
| one-hour | 60 minutes | 99,844 | 29.66% | 70.34% | 19.48% | 11.82% | 0.003583 |
| one-hour | 120 minutes | 94,743 | 26.87% | 73.13% | 21.37% | 13.80% | 0.004432 |

The recent cohort is not predominantly frozen at these short horizons:
approximately 68%--75% of valid targets change at all, while 17%--22% move by
more than 0.5 percentage point. This supports describing the collected recent
cohort as containing substantial short-horizon activity. It does not by itself
establish that 2025--2026 is more dynamic than an earlier cohort because this
audit intentionally excludes the old data.

## Effect of pruning

Pruning has little effect on the frequency of ordinary movements but reduces
the extreme-tail contribution:

- native one-hour, one-hour horizon: `abs(delta) > 0.005` changes from 19.68%
  raw to 19.48% clean, while mean absolute movement falls from 0.005750 to
  0.003583;
- native 15-minute, 30-minute horizon: `abs(delta) > 0.005` changes from
  19.46% raw to 19.42% clean, while mean absolute movement falls from 0.003802
  to 0.003337.

This is the desired behaviour: the filter removes rare extreme source
orientation artifacts without erasing the common movement distribution.

## Gap structure and bounded forward fill

The stored raw and clean Parquet files are irregular observations and have
**not** been forward-filled. Forward filling below is an in-memory sensitivity,
not a persisted property of the collected dataset.

| Source | Missing native-grid rows | One-bar fills | Fill rows / observed rows | Longer-gap rows left missing |
|---|---:|---:|---:|---:|
| clean 15-minute | 186,007 | 31,881 | 9.79% | 154,126 |
| clean one-hour | 20,455 | 4,534 | 4.22% | 15,921 |

The clean 15-minute data covers 63.64% of the internal native grid across the
50 contracts; the clean one-hour data covers 84.01%. Median contract-level
coverage is 68.33% and 91.78%, respectively. Coverage is heterogeneous:
26/50 15-minute contracts and 12/50 one-hour contracts fall below 75%, while
11/50 and 6/50 fall below 50%.

| Source | 1 missing bar | 2--4 missing bars | More than 4 missing bars | Maximum missing run |
|---|---:|---:|---:|---:|
| clean 15-minute | 31,881 events / 31,881 rows | 19,989 events / 51,655 rows | 8,022 events / 102,471 rows | 1,106 bars (276.5 hours) |
| clean one-hour | 4,534 events / 4,534 rows | 2,288 events / 5,831 rows | 772 events / 10,090 rows | 275 bars (275 hours) |

Pruning did not create the general gap problem. The raw files already contain
185,891 missing internal 15-minute slots and 20,308 missing internal one-hour
slots. Quarantining adds only 116 and 147 missing slots, reducing pooled
coverage by 0.023 and 0.115 percentage points.

The resolutions also provide inconsistent sparse views. Of 20,455 missing
native one-hour slots, 6,232 (30.47%) contain at least one native 15-minute
candle and 107 contain all four. Conversely, 113,390/186,007 (60.96%) missing
15-minute slots fall within an hour returned by the native one-hour endpoint.
Therefore, a missing candle is not reliable evidence that no trading occurred.
This supports retaining observed/imputed provenance and rejects silently
forward-filling every long gap as if it were verified market inactivity.

Although only one missing bar is filled, its influence covers multiple target
windows. Synthetic rows touch 25.49% of filled-grid 30-minute targets and
8.33% of filled-grid one-hour targets. Four-bar filling is substantially more
intrusive: it touches 41.38% and 14.69% of those targets, respectively.

The frozen exploratory rule is therefore:

1. fill at most one missing native bar;
2. set synthetic `open=high=low=close` to the previous observed close;
3. set `volume=0` and retain an `is_imputed` flag plus time since last observed
   trade/candle;
4. leave gaps longer than one bar missing;
5. do not use a forward-filled endpoint as a primary target, and report or
   exclude any sample whose context/target touches an imputed row;
6. do not add volatility noise or write simulated movements into canonical
   OHLCV.

Forward filling increases exact-zero rates because that is its mathematical
meaning. For the clean 30-minute series it raises the pooled zero fraction from
28.46% to 29.76%; for clean native one-hour changes it raises 29.66% to 32.78%.
Consequently, movement statistics must retain an observed-only primary result
and treat filled-grid results as a coverage sensitivity.

## Artifacts and reproduction

The default one-bar analysis writes `cohort_gap_summary.csv`,
`movement_summary.csv`, `contract_metrics.parquet`, `manifest.json`, and
`report.md` under the ignored FinData cohort analysis directory. A four-bar
sensitivity is stored separately.

The dedicated gap audit additionally writes every gap event, every missing
native-grid slot, per-contract coverage, the gap-length distribution, a
manifest, and a Markdown report under `analysis/native_gap_audit/`.

```bash
.venv/bin/python3 scripts_v2/analyze_findata_native_dynamics.py \
  --maximum-fill-bars 1 --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_native_dynamics.py \
  --maximum-fill-bars 4 \
  --output-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31/analysis/native_15m_1h_dynamics_cap4 \
  --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_native_gaps.py --overwrite
```
