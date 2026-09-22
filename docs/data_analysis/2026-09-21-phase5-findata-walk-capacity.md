# Phase 5 Recent FinData Walk-Forward Capacity Audit

**Date:** 2026-09-21
**Status:** training-capacity feasibility complete; selection and builder gate
remain open; no model training launched

> **Phase 5 interpretation update:** The capacity measurements remain valid,
> but retrospective selection and final-clean pruning are now accepted research
> assumptions rather than blockers. The active implementation gate is defined
> in `docs/phase_plan/2026-09-21-phase-5-experiment-plan.md`.

## 1. Question and validity boundary

This audit asks whether the selected recent native one-hour FinData source can
provide enough causally eligible sequences inside fixed-duration global
calendar training windows. It counts usable model rows after the actual gap,
endpoint, activity, and target-maturity rules instead of treating all raw
candles or stride-one windows as independent samples.

The result has an important boundary. The 50-condition cohort was selected
retrospectively using full-period information for source diagnosis. It can
establish sequence-capacity feasibility, but it cannot validate the Phase 5
cutoff-local universe-selection policy. Training remains blocked until a
broader candidate pool is retained or recollected and each walk freezes its
universe using only information available at that cutoff.

## 2. Audited split mechanics

The main capacity candidate uses two non-overlapping evaluation intervals:

| Walk | Rolling training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

The intervals were derived from calendar duration and coverage rather than
downstream model performance. Walk 2 may train on historical rows that were
evaluated in walk 1, which is standard walk-forward refitting; it may never
update the walk-1 model or predictions.

Each count enforces:

- one-hour candle-start availability at `date + 1h`;
- a fixed rolling training start and one shared cutoff across contracts;
- target availability strictly before the cutoff for training and before the
  evaluation end for testing;
- complete isolated one-hour gaps filled with the previous close;
- sequence breaks at gaps longer than one hour;
- observed decision and target endpoints;
- a causal price change in the preceding 24 hours for the primary active row
  population;
- evaluation only for contracts with active training rows before the cutoff;
  and
- separate affected-contract exclusion results.

Two input contexts (`64h` and duration-matched `256h`) and one-/two-hour
probability-movement targets were audited. The one-/two-hour horizon choice
barely changes capacity, so it can be frozen from task meaning rather than
sample count.

## 3. Balanced two-walk result: 64-hour context, one-hour target

| Walk | Sensitivity | Active train rows | Disjoint input+target intervals | Train contracts | Contracts with at least 256 train rows | Train rows / Phase 3 reference | Active supported evaluation rows | Evaluation contracts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Full clean cohort | 31,828 | 591 | 36 | 31 | 1.467x | 21,401 | 14 |
| 2 | Full clean cohort | 37,173 | 633 | 22 | 22 | 1.713x | 10,086 | 16 |
| 1 | Exclude all affected contracts | 19,903 | 378 | 25 | 21 | 0.917x | 15,622 | 10 |
| 2 | Exclude all affected contracts | 26,927 | 458 | 16 | 16 | 1.241x | 7,445 | 11 |

The Phase 3 reference is 21,696 overlapping four-hour training windows. It is
a workload comparison, not a statistical sufficiency threshold. The disjoint
counts greedily select non-overlapping complete input-plus-target intervals
within each contract and show that the nominal stride-one totals are highly
correlated.

The full-cohort training populations cover 6--7 heuristic categories and
17--30 event families. Evaluation covers 5--6 categories and 11--13 event
families. After affected-contract exclusion, evaluation still covers five
categories and 9--10 families, but only 10--11 contracts. Row concentration
also rises: the top five contracts supply 56.30% of walk-1 and 68.96% of
walk-2 exclusion-sensitivity evaluation rows. The sensitivity is therefore
usable but not strong enough to dismiss source dependence.

## 4. Imputation exposure

An isolated fill is a small fraction of the completed grid but affects many
overlapping contexts:

| Walk | Sensitivity | Train rows with no imputed context hour | Train contexts touching a fill | Evaluation rows with no imputed context hour | Evaluation contexts touching a fill |
|---:|---|---:|---:|---:|---:|
| 1 | Full clean cohort | 18,483 | 41.93% | 14,615 | 31.71% |
| 2 | Full clean cohort | 25,816 | 30.55% | 6,803 | 32.55% |
| 1 | Exclude affected contracts | 11,752 | 40.95% | 10,942 | 29.96% |
| 2 | Exclude affected contracts | 18,683 | 30.62% | 5,119 | 31.24% |

This does not invalidate filling: every synthetic row is context-only and has
an explicit mask. It does mean imputation exposure must be a primary reporting
stratum, with an untouched-context sensitivity, rather than being hidden in a
single pooled score.

## 5. Why 256 hours is not the preferred primary context

With a 256-hour input, the full-cohort folds retain 18,399 and 26,772 active
training rows, but only 95 and 128 disjoint intervals. Under affected-contract
exclusion these fall to 10,457/19,454 overlapping rows and 56/93 disjoint
intervals. Walk-1 evaluation drops to eight contracts after exclusion, while
the top five contracts provide 76.13% of its rows.

The duration-matched 256-hour setting is therefore useful as a context-length
sensitivity, but it is unnecessarily restrictive for the primary recent-data
loop. A 64-hour context is the better capacity candidate. This is a feasibility
recommendation, not authorization to train before the Phase 5 builder and
universe rules are frozen.

## 6. Walk-count sensitivity

A five-walk monthly schedule keeps 31,498--37,898 active `seq64` training rows
per fold, but its August evaluation interval deteriorates to 2,654 rows over
eight contracts. The top five contracts contribute 94.39% of those rows. The
corresponding 256-hour evaluation has only two contracts.

Monthly refitting therefore creates a weak final fold in this cohort. The
balanced two-walk candidate is more defensible for the main exploratory loop;
monthly walks should not be chosen merely because they produce more model
retraining points.

## 7. Quarantine availability

All 136/11 hourly quarantine decisions falling in the two training windows
were available before their respective cutoffs. The first evaluation interval
contains 11 quarantined candles with four-hour decision delays, creating an
upper bound of 44 contract-hour decision slots that a retrospective final-clean
view could suppress before the quarantine decision was knowable. The second
evaluation interval contains none.

Consequently, the Phase 5 builder cannot simply load the final clean file for
every historical decision. It must replay quarantine decisions according to
`quarantine_available_at` or explicitly mark/exclude only predictions whose
eligibility was knowable at decision time. The affected-contract exclusion is
still required as a separate source sensitivity.

## 8. Judgement

The **split policy is sound in structure**:

- global calendar rather than per-contract-relative cutoffs;
- fixed rolling training histories and non-overlapping evaluation intervals;
- fold-specific preprocessing, encoders, heads, and baselines;
- mature contract-local targets;
- observed decision/target endpoints;
- identical framework/baseline rows; and
- no later-walk information revising an earlier prediction.

The **current data-selection implementation is not yet fully sound**. The
retrospective 50-contract cohort cannot prove cutoff-local universe selection,
and final-clean evaluation must not use quarantine decisions before they become
available. Therefore:

1. retain `seq64` one-hour inputs as the primary capacity candidate;
2. retain `seq256` only as a context sensitivity;
3. use the balanced two-walk schedule as the implementation candidate;
4. keep imputation-exposure and affected-contract exclusion sensitivities;
5. implement cutoff-local candidate discovery/universe freezing and causal
   quarantine replay; and
6. rerun this audit on the exact replayable Phase 5 bundle before training.

The present cohort has adequate **exploratory** capacity for `seq64`, but the
analysis does not clear the Phase 5 execution gate.

## 9. Reproduction

```bash
.venv/bin/python3 scripts_v2/analyze_findata_walk_capacity.py --overwrite
```

Machine-readable outputs are stored under the Git-ignored directory:

```text
data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31/
  analysis/phase5_walk_capacity/
```

They include `walk_capacity.csv`, `contract_capacity.parquet`,
`quarantine_cutoff_audit.csv`, `manifest.json`, and `report.md`.
