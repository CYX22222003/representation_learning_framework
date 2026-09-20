# FinData Recent Polymarket Acquisition MVP

**Date:** 2026-09-20
**Status:** acquisition infrastructure and three-contract dry run complete; not a Phase 4 training bundle

> **Historical note:** This MVP predates the expanded 50-market audit. Its
> preliminary selection judgement is superseded by the
> [Phase 4 conclusion](../phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md),
> which selects the recent clean native one-hour cohort for an explicitly
> exploratory Phase 5 loop subject to the documented execution gate.

> **Expanded follow-up:** Subsequent 24- and 50-market audits found that the
> condition-level candle endpoint can mix complementary YES/NO token prices.
> Its candle movement statistics must therefore not be treated as a canonical
> probability series. The collectors now preserve the raw candles and also
> emit versioned, gap-preserving quarantine artifacts under a 1% fail-closed
> budget. The current rule retains forward-confirmed persistent crashes and
> records when each decision becomes available. The token-consistent trade
> fallback and current mitigation are documented in
> [the expanded audit](2026-09-20-findata-expanded-recent-cohort.md) and
> [the forward-confirmed 50-market audit](2026-09-21-findata-forward-confirmed-quarantine.md).

## Purpose

The repository's original raw files are concentrated in 2024--2025. This dry
run tests whether the lab FinData API can provide a reproducible, more recent
Polymarket source over the half-open UTC interval:

```text
[2025-12-01 00:00, 2026-09-01 00:00)
```

The exclusive endpoint includes all of 31 August 2026. The acquisition module
is independent of experiment preprocessing: it downloads source rows and
records provenance but does not create train/test splits, fitted transforms,
model windows, or labels.

## Implemented path

The read-only client is implemented in `src/data_acquisition/findata.py` and
the runnable entry point is `scripts_v2/collect_findata_prediction_markets.py`.
Authentication is read from `LUMID_API_KEYS` in the environment or root
`.env`; the token and Authorization header are never persisted.

The collector:

1. reads the monitored Polymarket universe;
2. deduplicates the two outcome-token rows by condition ID;
3. ranks the current snapshot deterministically;
4. probes candidates for sufficient hourly history;
5. downloads native 15-minute data in seven-day chunks and native one-hour
   data in 30-day chunks, always sending the explicit API limit;
6. validates timestamp range, OHLC consistency, finite values, volume, and
   duplicates;
7. preserves the raw files, quarantines sandwiched complementary and
   `high-low > 0.5` candles without repairing prices, and fails if more than
   1% of a native resolution is flagged;
8. derives separate UTC-aligned four-hour OHLCV artifacts from raw and clean
   hourly rows without filling missing hours; and
9. saves market metadata, discovery audit, universe snapshot, quarantine
   audit, hashes, and an immutable manifest.

Outputs live under `data_new/`, which is ignored by Git. The default MVP root
is:

```text
data_new/findata/polymarket/mvp_2025-12-01_2026-08-31/
```

## API findings

FinData's prediction-market candle response is a bare JSON array with fields
`bucket_ts`, `open`, `high`, `low`, `close`, `volume`, and `trades`. Date
filtering works with RFC3339 `from`/`to` parameters. The observed endpoint is
sparse: intervals with no emitted trade/midprice candle are absent rather than
stored as zero-volume rows.

Native `interval=15` and `interval=60` worked in bounded live calls. The usage
guide documents a `4h` alias, but the live endpoint returned HTTP 400 for that
value on 2026-09-20; `4hour` was also rejected. `interval=240` returned an
empty array for a condition that had native hourly rows. The MVP therefore does
not pretend that a native four-hour source was obtained. It derives four-hour
bins from hourly rows and stores `observed_1h_bars` plus
`complete_1h_coverage` on every bin.

## Three-contract dry-run result

The liquidity-ranked current-snapshot selection retained three long-running
2028 US nomination conditions: Tim Walz, Elise Stefanik, and Roy Cooper. This
is useful for infrastructure validation but is a narrow, correlated political
cohort rather than a representative prediction-market sample.

| artifact | rows | observed range |
|---|---:|---|
| Native 15-minute | 42,101 | 2025-12-05 17:30 to 2026-08-31 22:15 UTC |
| Native one-hour | 13,481 | 2025-12-05 19:00 to 2026-08-31 23:00 UTC |
| Derived four-hour | 3,913 | 2025-12-05 16:00 to 2026-08-31 20:00 UTC |

Hourly grid coverage by contract is `68.17%--71.00%`. The maximum hourly gap
is `188--210` hours. Only 2,515 of 3,913 derived four-hour bins (`64.27%`)
contain all four hourly source candles. Raw API output therefore cannot be fed
directly into the existing assumption of regular, gap-free bars.

On 3,486 hours for which all four 15-minute rows and a native hourly row were
present, the 15-minute-derived close exactly matched the native hourly close in
`42.14%` of cases. Close MAE was small (`0.000463`) but non-zero. Different API
resolutions must be treated as separately constructed source series, not
assumed to be algebraically identical.

## Movement and sequence feasibility

Restricting the derived data to fully observed, consecutive four-hour bars
produced 1,562 valid eight-hour targets:

- exact-zero movement: `34.38%`;
- stable at `|delta| <= 0.005`: `99.87%`;
- zero-movement baseline MAE: `0.000528`; and
- complete consecutive 64-bar context plus two-bar target rows without any
  imputation: only `27`.

The extremely stable target distribution is primarily a cohort-selection
problem: these are low-probability, long-horizon nomination contracts. High
snapshot liquidity alone did not create a varied regression cohort. Scaling
the collection must add event/category diversity and event-family controls,
not merely collect more rows from the same type of market.

## Research-use decision

The API is a viable source for recent raw Polymarket observations, and the MVP
proves authenticated, chunked, replayable acquisition at 15-minute and hourly
resolution. It does not yet justify replacing the frozen Phase 4 primary data.

Before recent FinData data becomes an experiment input, freeze and validate:

1. a broader market-discovery rule with event/category diversity and paired
   outcome/event-family handling;
2. a causal rule for gaps and inactive periods, fitted or applied independently
   inside each global calendar walk;
3. whether the experiment uses native hourly, native 15-minute, or a declared
   resampling source, because they are not exactly identical;
4. whether the predeclared source rule uses token-specific history or the
   versioned quarantine plus an affected-contract exclusion sensitivity;
5. cutoff-local universe selection rather than the current snapshot ranking
   used only for this dry run; and
6. a new walk schedule covering the recent period while retaining a final
   temporally untouched confirmation interval.

## Reproduction

```bash
.venv/bin/python3 scripts_v2/collect_findata_prediction_markets.py \
  --top-k 3 \
  --candidate-limit 30 \
  --workers 6

.venv/bin/python3 scripts_v2/analyze_findata_prediction_markets.py
```

The analysis report and replayable tables are written inside the ignored MVP
directory under `analysis/`.
