# Expanded Recent FinData Polymarket Cohort

**Date:** 2026-09-20
**Status:** exploratory acquisition and source-semantics audit complete; not a Phase 4 training bundle

## Research question

The original repository data is concentrated in 2024--2025. This analysis asks
whether the more active 2025--2026 prediction-market regime warrants a recent
data extension and whether the lab FinData API can supply valid high-frequency
probability histories for it.

The external market-landscape evidence supports the first motivation. FalconX
reports that prediction-market volume grew nearly fourfold to about USD 64
billion in 2025, with January 2026 volume near USD 27 billion, and describes a
broader mix of political, sports, crypto, and macroeconomic markets. This is
evidence of a changed activity and participation regime, not proof that every
individual contract has greater probability variation. See the
[FalconX article](https://www.falconx.io/newsroom/from-opinions-to-odds-emerging-trends-in-the-prediction-market-landscape)
and its [LinkedIn version](https://www.linkedin.com/pulse/from-opinions-odds-emerging-trends-prediction-market-landscape-n6lce/).

## Cohort and acquisition

The expanded retrospective cohort covers the half-open UTC interval
`[2025-12-01, 2026-09-01)` and contains 24 Polymarket conditions selected from
1,276 discovered open/closed market records. Of 605 markets with at least 14
days of overlap, candidates were ordered using loose topic diversity,
full-lifetime volume, and a maximum of two contracts per heuristic event
family. Topic labels are sampling aids only; occasional label errors do not
affect contract-level movement calculations.

The condition-level candle acquisition produced:

| source | rows |
|---|---:|
| Native FinData 15-minute candles | 124,904 |
| Native FinData one-hour candles | 36,040 |
| Four-hour bins derived from native hourly candles | 10,350 |
| Fully observed derived four-hour bins | 7,560 (73.04%) |

The API sometimes returns observations just outside the requested interval.
The collector now prunes each response to the market-specific half-open range
before validation and storage. Raw gaps are not forward-filled.

The ignored acquisition roots are:

```text
data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31/
data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31_yes_trades/
```

## Critical source-semantics finding

FinData documents Polymarket candles by `condition_id`, while a binary
Polymarket condition has separate YES and NO token IDs. The condition-level
candle records do not carry token identity. At least one downloaded hourly
series repeatedly moved between approximately `0.003` and `0.997`; the raw
trade endpoint confirms that these are complementary token prices rather than
belief changes. Passing an outcome token ID to the candle endpoint returned no
data in live probes.

Therefore the condition-level candle files are an API/source audit, not a
valid single-outcome probability series. A continuity heuristic or
settlement-informed orientation would be unacceptable because it could use
future resolution information and would be ambiguous near probability `0.5`.

## Local quarantine decision (2026-09-21)

> Historical note: this section records the original 24-market v1 replay. The
> completed 50-market raw-first recollection and forward-confirmed v2 decision
> supersede it for current infrastructure. See
> `2026-09-21-findata-forward-confirmed-quarantine.md`.

The source issue has been reported upstream, and a conservative local
quarantine is now implemented for exploratory reuse. Raw native files remain
immutable. Rule `condition-orientation-v1` quarantines a candle when it is
sandwiched between approximately complementary closes with at least one
exact-cadence neighbour, or when its own `high-low` range exceeds `0.5`.
Prices are never inverted or otherwise repaired; removed timestamps remain
gaps, so aggregation, contexts, and targets must require exact consecutive
timestamps.

Applied to this 24-market snapshot, the rule quarantines:

| resolution | raw rows | quarantined | retained | affected contracts |
|---|---:|---:|---:|---:|
| native 15-minute | 124,904 | 12 | 99.9904% | 4 |
| native one-hour | 36,040 | 42 | 99.8835% | 5 |

The hourly total contains 32 isolated complementary candles from the Xi
Jinping contract and ten wide bars. The 15-minute total contains two
sandwiched complementary candles and ten wide bars. The separate one-way
`0.238 -> 0.756` Iran repricing is deliberately retained: a large transition
alone is insufficient evidence of source mixing.

The implementation writes `candles_15min_clean.parquet`,
`candles_1h_clean.parquet`, `candles_4h_derived_clean.parquet`,
`candle_quarantine.parquet`, and a hash-bearing `quarantine_manifest.json`.
It fails closed if more than 1% of either native resolution would be removed.
That budget expresses the project's operational tolerance; it does not prove
that the remaining condition-level rows have YES-token identity, especially
near probability `0.5` where complement switches may not be detectable.

The original snapshot also predates the explicit candle `limit=5000` fix, so
its anomaly counts are lower bounds and its clean files remain audit outputs.
A usable exploratory rebuild must first recollect with the fixed collector.

The safe fallback uses FinData's token-identified trade endpoint. The market
metadata maps the declared `Yes` outcome to its token ID before any price rows
are processed. Only that token is aggregated into UTC-aligned trade-price
OHLCV; absent bins stay absent. No settlement value is used to orient bars.

## Token-consistent result

Only 12 of the 24 selected conditions returned token-identified trades during
the requested interval. The collection contains 47,711 all-outcome trades and
28,359 canonical YES trades.

Strict eight-hour targets require uninterrupted bars at the stated
resolution:

| resolution | YES bars | target markets | strict 8h targets | exact zero | stable within 0.005 | mean absolute move | complete seq64+h2 rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15m | 7,157 | 0 | 0 | n/a | n/a | n/a | 0 |
| 1h | 4,720 | 7 | 1,836 | 7.73% | 64.32% | 0.020835 | 0 |
| 4h | 1,864 | 11 | 1,327 | 9.27% | 58.78% | 0.015843 | 448 |

All 448 complete four-hour `seq64+h2` rows come from two closely related 2026
World Cup winner contracts. The resulting encoder sample would therefore be
far too narrow despite containing meaningful short-horizon movement.

## Comparison with the original data

The previous top-80 four-hour audit reported 43,046 all-row eight-hour targets,
46.18% exact zero, 69.44% stable within `0.005`, and mean absolute movement
`0.006874`. Its causal active subset reported 32,016 targets, 28.56% exact
zero, 59.11% stable, and mean absolute movement `0.009210`.

The token-consistent recent four-hour subset has fewer exact zeros and larger
mean movement than either older construction. This is consistent with the
hypothesis that selected recent active markets are more reactive. It is not a
clean causal old-versus-new comparison because the provider, market-selection
rule, missing-bar mechanism, and market mix differ. The recent result is also
based on only 11 target-bearing contracts and is dominated in sequence capacity
by one event family.

## Judgment

The decision should be split into two claims:

1. **Use a recent extension for representativeness: supported.** The market's
   aggregate scale and composition changed substantially in 2025--2026, and
   the token-consistent sample contains materially larger short-horizon moves.
2. **Replace the frozen Phase 4 data immediately: not supported.** The current
   FinData condition candles can mix outcome orientations, while the safe
   trade-only fallback is too sparse for a diverse seq64 encoder cohort. The
   local quarantine makes a future recollected condition-candle sensitivity
   feasible, but does not retroactively make this snapshot a Phase 4 bundle.

A confirmatory recent-data experiment should still prefer one of:

- provider support for token-specific historical candles or midpoint history;
- an independently collected token-specific order-book/midpoint source; or
- a redesigned trade-event representation whose sequence contract does not
  assume dense fixed-interval bars.

An exploratory condition-candle sensitivity may instead use the versioned
clean artifacts if it also reports the quarantine rate, excludes every window
crossing a quarantined or absent timestamp, and compares row-pruning results
against an affected-contract exclusion sensitivity.

Any later recent-data universe must also be selected separately at each global
calendar cutoff. Full-lifetime volume and complete market dates used in this
exploration remain retrospective and are not Phase 4 selection inputs.

## Reproduction

```bash
.venv/bin/python3 scripts_v2/collect_findata_historical_cohort.py \
  --top-k 24 --candidate-limit 120 --max-per-family 2 --workers 8

.venv/bin/python3 scripts_v2/analyze_findata_prediction_markets.py \
  --input-dir data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31

.venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py

.venv/bin/python3 scripts_v2/collect_findata_yes_trade_ohlcv.py --workers 8
.venv/bin/python3 scripts_v2/analyze_findata_yes_trade_ohlcv.py
```

The collectors never persist the API token or Authorization header. All raw
and derived acquisition artifacts remain under Git-ignored `data_new/`.
