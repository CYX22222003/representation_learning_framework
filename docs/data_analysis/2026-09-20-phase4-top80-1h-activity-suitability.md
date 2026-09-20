# Phase 4 Top-80 One-Hour Activity Suitability Audit

**Date:** 2026-09-20
**Status:** exploratory data analysis; no model training, feature extraction, or model selection

> **Historical note:** This pre-December-2025 top-80 audit remains qualitative
> feasibility evidence. Its prospective Phase 4 recommendations were not
> trained and are superseded for the next loop by the
> [Phase 4 conclusion](../phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md).

## Purpose and scope

This audit asks whether the one-hour data contains enough non-stale price
movement to support meaningful probability-movement regression and
classification. It complements the timestamp and sample-count audit: a large
number of overlapping windows is not useful if most targets are unchanged.

The cohort contains the same 80 contract identities used in the four-hour
analysis. Those identities were ranked from an early four-hour prefix and then
mapped to their one-hour files. This is a controlled timeframe comparison, not
an independently selected one-hour top-80 universe and not yet a deployable
cutoff-local universe.

All target results below use:

- a 256-hour input-eligibility requirement;
- an eight-hour future signed probability change, `p[t+8] - p[t]`;
- `|delta| <= 0.005` as the descriptive stable band; and
- only contract-local targets.

## Main findings

The 80 files contain 192,658 raw rows and 171,618 rows eligible after the
256-hour context and eight-hour target requirements.
Their raw lengths range from 1,286 to 8,344 hours, with a median of 1,649
hours. Thus, none of these selected files is too short to form a 256-hour
input; the more important problem is that observed file length can include a
long inactive tail after meaningful trading has stopped.

| diagnostic | result |
|---|---:|
| Exact-zero eight-hour targets | 46.23% |
| Stable eight-hour targets (`|delta| <= 0.005`) | 69.57% |
| Mean / median / p90 absolute eight-hour move | 0.006795 / 0.001000 / 0.020000 |
| Median contract close standard deviation | 0.072893 |
| Median contract fraction with `|8h delta| > 0.005` | 39.36% |
| Contracts with fewer than 10% meaningful eight-hour moves | 20 / 80 |
| Contracts with close standard deviation below 0.01 | 17 / 80 |
| Contracts at least 80% unchanged at one hour | 6 / 80 |
| Contracts with a constant-price run of at least 24h / 72h | 51 / 33 |
| Contracts with a trailing constant tail covering at least 25% of the file | 19 / 80 |

Therefore, the concern is real but concentrated. The one-hour cohort is not
uniformly stale: most contracts retain meaningful movement, while a material
minority contributes long inactive or resolution-like tails.

Examples include paired UEFA winner contracts with 4,095-hour trailing
constant runs and Romanian-election contracts with approximately 5,850-hour
trailing constant runs. In these tails, the close is typically `0.001` or
`0.999` and positive-volume hours are rare. These rows can inflate the apparent
training set while mostly teaching a trivial no-change pattern.

## Volume alone is insufficient

The per-contract rank correlation between the positive-volume-hour fraction
and the fraction of meaningful eight-hour movements is only `0.4451`.
Activity and price movement are related, but an early volume ranking does not
guarantee an informative target distribution throughout a contract's life.
Universe selection and row eligibility must therefore be treated separately.

## Causal recent-activity sensitivity

Trailing-flat length is known only after observing the future and must not be
used to select historical rows. The audit instead tested eligibility rules
that use only information at or before each decision timestamp.

| eligibility rule | rows | retained | contracts | exact zero | stable | mean absolute move |
|---|---:|---:|---:|---:|---:|---:|
| All eligible rows | 171,618 | 100.00% | 80 | 46.23% | 69.57% | 0.006795 |
| Positive volume in prior 24h | 136,885 | 79.76% | 80 | 32.59% | 61.85% | 0.008520 |
| Price change in prior 72h | 133,161 | 77.59% | 80 | 30.71% | 60.79% | 0.008758 |
| Price change in prior 24h | 130,878 | 76.26% | 80 | 29.61% | 60.12% | 0.008907 |
| Price change and positive volume in prior 24h | 130,876 | 76.26% | 80 | 29.61% | 60.12% | 0.008907 |

Requiring a recent price change removes much of the inactive tail without
using future information. The volume condition adds almost nothing once a
recent price change is required in this cohort.

This filter changes the research estimand: the model predicts movement for
recently active markets rather than for every listed market. It must therefore
be predeclared, applied identically to framework and baselines, and accompanied
by an all-row sensitivity result.

## Lifecycle effect

| lifecycle stage | rows | exact zero | stable | mean absolute move | near-boundary price |
|---|---:|---:|---:|---:|---:|
| Early | 43,825 | 33.15% | 60.11% | 0.009366 | 47.69% |
| Middle | 64,179 | 42.05% | 65.21% | 0.007544 | 56.72% |
| Late | 63,614 | 59.45% | 80.50% | 0.004269 | 79.36% |

The one-hour data reproduces the four-hour lifecycle finding: movement falls
and boundary concentration rises late in contract life. This is a substantive
property of prediction markets, amplified by inactive terminal tails.

## Two-walk calendar evaluation

Under the two global-calendar intervals proposed by the capacity audit:

| walk | eligibility | train rows | train contracts | evaluation rows | evaluation contracts | exact zero | stable | mean absolute move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | All rows | 118,708 | 59 | 28,592 | 20 | 64.21% | 83.32% | 0.002606 |
| 1 | Price change in prior 24h | 104,629 | 59 | 12,876 | 14 | 20.64% | 62.95% | 0.005786 |
| 2 | All rows | 119,770 | 59 | 24,134 | 15 | 57.06% | 66.91% | 0.009196 |
| 2 | Price change in prior 24h | 90,504 | 59 | 13,278 | 13 | 21.95% | 39.86% | 0.016714 |

Both active-row training folds retain more than 90,000 rows across 59
contracts. Their evaluation intervals retain more than 12,000 rows and at
least 13 contracts. They contain substantially more movement than the
unfiltered intervals, although the difference between walks shows that regime
and cohort composition remain important.

## Judgement

The one-hour data is worth using for Phase 4, subject to a stricter activity
contract. It has adequate duration-matched training capacity and enough
non-trivial eight-hour movement after a causal recent-activity rule. It is not
sound to train and evaluate only on all emitted rows without distinguishing
inactive tails, because a material minority of contracts would contribute
large numbers of nearly deterministic observations.

Before training, Phase 4 should freeze the following:

1. a cutoff-local one-hour universe rule, rather than silently treating the
   current four-hour-ranked cohort as the final one-hour universe;
2. entering eligibility after a complete 256-hour history;
3. leaving/inactive eligibility based only on trailing information available
   at the decision time, with the 24-hour recent-price-change rule as the
   current leading candidate;
4. the same eligibility mask for every framework and baseline model;
5. primary active-market metrics plus an all-listed-row sensitivity report;
6. exact-zero movement as a required regression baseline; and
7. per-walk contract coverage, stable share, lifecycle composition, and
   boundary-price composition alongside predictive metrics.

This audit establishes target variation and sample feasibility, not
predictability. Only leakage-safe walk-forward model comparisons against the
zero-movement and other causal baselines can establish useful predictive
performance.

## Reproduction

Run:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_activity_suitability.py
```

Artifacts are stored under
`experiments/phase4/data_analysis/top80_1h_activity_suitability/`.
