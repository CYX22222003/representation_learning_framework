# Phase 4 Top-50/Top-80 One-Hour versus Four-Hour Comparison

**Date:** 2026-09-20
**Status:** exploratory data analysis; no model training or model selection

## Comparison contract

The comparison fixes the temporal meaning rather than merely matching sequence
length:

| setting | one-hour | four-hour |
|---|---:|---:|
| Input context | 256 bars = 256 hours | 64 bars = 256 hours |
| Target horizon | 8 bars = 8 hours | 2 bars = 8 hours |
| Recent-activity lookback | 24 bars = 24 hours | 6 bars = 24 hours |
| Probability-movement threshold | 0.005 | 0.005 |

The same four-hour early-prefix rank supplies the top-50 and top-80 contract
identities at both frequencies. This makes the timeframe comparison paired,
but the cohort remains retrospective until Phase 4 implements cutoff-local
universe selection.

Raw candle timestamps behave as bar-start labels. The analysis therefore uses
`timestamp + 1h` as the one-hour close availability time and `timestamp + 4h`
as the four-hour close availability time when applying calendar cutoffs.

## All-row target comparison

| timeframe | cohort | eligible rows | exact-zero 8h targets | stable 8h targets | mean absolute move | median contract meaningful share | contracts below 10% meaningful | trailing-flat >=25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1h | top-50 | 110,962 | 44.56% | 66.60% | 0.007408 | 40.36% | 8 | 12 |
| 4h | top-50 | 27,831 | 44.67% | 66.58% | 0.007479 | 40.05% | 8 | 12 |
| 1h | top-80 | 171,618 | 46.23% | 69.57% | 0.006795 | 39.36% | 20 | 19 |
| 4h | top-80 | 43,046 | 46.18% | 69.44% | 0.006874 | 38.79% | 20 | 19 |

The one-hour and four-hour results are nearly identical after matching context
and horizon duration. Increasing the cohort from 50 to 80 adds contract
coverage but also adds a disproportionate number of low-movement contracts:
the count below 10% meaningful movement increases from 8 to 20.

## Direct target alignment

The bar-end correction permits direct comparison of targets sharing the same
contract, decision availability time, and target availability time.

| cohort | aligned targets | share of 4h targets aligned | exact delta agreement | mean absolute delta difference | delta correlation | 24h activity-rule agreement |
|---|---:|---:|---:|---:|---:|---:|
| top-50 | 27,746 | 99.69% | 89.35% | 0.000946 | 0.9720 | 98.58% |
| top-80 | 42,917 | 99.70% | 88.12% | 0.000972 | 0.9690 | 98.09% |

Thus, the one-hour target series is not substantially less stale. Its main
sample-count advantage comes from decisions at four hourly offsets rather than
from a different eight-hour outcome process. It may still supply useful
within-four-hour input dynamics, but this audit does not establish that those
dynamics improve prediction.

## Causal active-row comparison

Requiring at least one price change during the preceding 24 hours gives:

| timeframe | cohort | active rows | retained | exact zero | stable | zero-baseline MAE |
|---|---:|---:|---:|---:|---:|---:|
| 1h | top-50 | 83,784 | 75.51% | 26.63% | 55.77% | 0.009810 |
| 4h | top-50 | 20,620 | 74.09% | 26.04% | 55.10% | 0.010066 |
| 1h | top-80 | 130,878 | 76.26% | 29.61% | 60.12% | 0.008907 |
| 4h | top-80 | 32,016 | 74.38% | 28.56% | 59.11% | 0.009210 |

Again, resolution has little effect on the target distribution. The 1-hour
data supplies approximately four times as many highly overlapping decision
rows, while retaining essentially the same contracts and target difficulty.

## Two-walk calendar results

### Top-50 active rows

| timeframe | walk | training rows / contracts | evaluation rows / contracts | stable | zero-baseline MAE |
|---|---:|---:|---:|---:|---:|
| 1h | 1 | 65,036 / 35 | 11,727 / 13 | 66.34% | 0.005171 |
| 4h | 1 | 15,962 / 35 | 2,914 / 13 | 65.96% | 0.005129 |
| 1h | 2 | 58,267 / 39 | 6,942 / 7 | 40.54% | 0.014357 |
| 4h | 2 | 14,399 / 39 | 1,722 / 7 | 39.14% | 0.015084 |

### Top-80 active rows

| timeframe | walk | training rows / contracts | evaluation rows / contracts | stable | zero-baseline MAE |
|---|---:|---:|---:|---:|---:|
| 1h | 1 | 104,629 / 59 | 12,876 / 14 | 62.95% | 0.005786 |
| 4h | 1 | 25,476 / 59 | 3,202 / 14 | 62.68% | 0.005745 |
| 1h | 2 | 90,504 / 59 | 13,278 / 13 | 39.86% | 0.016714 |
| 4h | 2 | 22,163 / 59 | 3,311 / 13 | 39.20% | 0.017232 |

Top-50 produces a somewhat more active population but leaves only seven
evaluation contracts in the second walk. Top-80 nearly doubles that coverage
to 13 while retaining 22,163--25,476 four-hour training rows. Those training
counts are comparable to or larger than the completed Phase 3 four-hour
encoder workload of 21,696 rows.

## Lifecycle comparison

The frequency comparison also reproduces nearly identical lifecycle drift. For
top-80 four-hour data, exact-zero eight-hour movement rises from 33.13% early
to 59.49% late, stable movement rises from 59.55% to 80.48%, and mean absolute
movement falls from 0.009418 to 0.004293. The matching one-hour values are
33.15% to 59.45%, 60.11% to 80.50%, and 0.009366 to 0.004269.

This confirms that terminal persistence is a market/lifecycle property rather
than an artefact caused by using four-hour candles.

## Judgement

The original choice to focus on active top-ranked four-hour contracts is
supported, with one refinement:

1. Four-hour data is not materially more stale than one-hour data for the
   duration-matched eight-hour target.
2. One-hour data provides roughly four times more overlapping windows, but not
   four times more independent information.
3. Four-hour top-80 with the causal prior-24h activity rule offers the better
   primary Phase 4 balance: adequate training size, 13--14 evaluation
   contracts, lower sequence length, and substantially lower compute.
4. Four-hour top-50 is useful as a higher-activity sensitivity cohort, but its
   second calendar test interval contains only seven contracts.
5. One-hour top-80 is worth retaining as a resolution sensitivity or later
   ablation to test whether within-four-hour paths add value. The current data
   analysis does not justify paying its much larger training cost as the
   primary experiment.
6. Neither frequency removes the need for active-market eligibility,
   zero-movement baselines, per-walk contract coverage, and lifecycle-stratified
   reporting.

## Reproduction

Run:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_4h_activity_comparison.py
```

Artifacts are stored under
`experiments/phase4/data_analysis/top50_top80_1h_4h_activity_comparison/`.
