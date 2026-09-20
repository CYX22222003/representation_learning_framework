# Phase 4 Data Selection and Walk-Forward Contract

**Date:** 2026-09-20
**Status:** concluded as historical feasibility design; not executed
**Supersedes:** provisional expanding-walk language in the Phase 3 conclusion

> **Phase 4 conclusion (2026-09-21):** This four-hour top-80 contract was not
> implemented or trained. Phase 4 closed after the data-selection and
> evaluation-design investigation. The recent FinData audit subsequently
> selected bounded-forward-filled native one-hour data as the exploratory
> source for the next complete walk-forward training loop. That implementation,
> encoder retraining, downstream-task redefinition, and matched baseline matrix
> move to Phase 5. See
> [`2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`](2026-09-21-phase-4-data-exploration-observation-and-conclusion.md).
> The rules below remain valuable pre-December-2025 feasibility evidence and
> leakage constraints; they are not an active launch specification.

## 1. Decision

The primary Phase 4 experiment uses four-hour, top-80 prediction-market data
under two fixed-duration rolling global-calendar walks. Contracts are selected
at each training cutoff using training-only information, and rows are admitted
through a causal recent-activity rule. Every walk receives independently fitted
preprocessing, encoder weights, feature bundles, task heads, and baselines.

The primary regression target is eight-hour signed probability movement:

```text
delta[t] = close[t + 2 four-hour bars] - close[t]
```

Absolute next-close prediction remains Phase 3 negative characterisation
evidence. Arithmetic return is secondary because bounded prices near zero make
percentage changes unstable.

## 2. Primary and sensitivity configurations

| role | timeframe | universe | sequence | context | horizon | purpose |
|---|---|---:|---:|---:|---:|---|
| Primary | 4h | top-80 per walk | 64 bars | 256h | 2 bars = 8h | Main Phase 4 result |
| Universe sensitivity | 4h | nested top-50 | 64 bars | 256h | 2 bars = 8h | Higher-activity, lower-coverage comparison |
| Resolution sensitivity | 1h | top-80 per walk | 256 bars | 256h | 8 bars = 8h | Test whether within-4h paths add value |
| Row-population sensitivity | 4h | top-80 per walk | 64 bars | 256h | 2 bars = 8h | All eligible rows without the activity mask |

The one-hour configuration is not primary. Duration-matched analysis found
nearly identical target distributions at one and four hours, aligned-target
correlations of `0.969--0.972`, and approximately four times as many but more
heavily overlapping one-hour decision rows. Four-hour top-80 provides the best
current balance of training size, contract coverage, and compute.

## 3. Candle-time semantics

Raw `date` values label candle starts. Information availability is therefore:

```text
1h bar stamped t -> available at t + 1h
4h bar stamped t -> available at t + 4h
```

Every split, universe calculation, eligibility mask, input identity, and target
maturity check must use availability time rather than the unadjusted candle
label. Stored artifacts must retain both the raw candle-start timestamp and the
derived availability timestamp. The raw timestamps are timezone-naive; the
dataset clock convention must be recorded in the Phase 4 builder manifest
before validation passes.

## 4. Two rolling global-calendar walks

The primary evaluation uses the following half-open intervals:

| walk | rolling training interval | next evaluation interval |
|---:|---|---|
| 1 | `[2024-06-28 20:00, 2025-03-31 09:00)` | `[2025-03-31 09:00, 2025-08-16 04:00)` |
| 2 | `[2024-11-13 15:00, 2025-08-16 04:00)` | `[2025-08-16 04:00, 2025-12-31 23:00)` |

These are global timestamps shared by every contract. Three equally spaced
walks are rejected because the middle evaluation interval contained only two
contracts in the feasibility cohort.

Walk 2 may train on observations that were evaluated by walk 1 because those
outcomes are historical by the second cutoff. This is standard walk-forward
refitting. Walk 2 state may never update, select, or reinterpret the walk 1
model or its predictions. Predictions are pooled only after every row has been
generated out of future information.

## 5. Cutoff-local universe selection

The existing retrospective top-80 list is feasibility evidence, not the final
Phase 4 universe. Universe construction is repeated independently at each
training cutoff.

For cutoff `T_k`:

1. Consider only contracts whose required history is available before `T_k`.
2. Require 256 four-hour bars inside the permitted rolling training interval
   for the liquidity-ranking lookback.
3. On the trailing 256 permitted bars, calculate:
   - positive-volume-bar count; and
   - sum of `log1p(max(volume, 0))` over positive finite volume.
4. Rank lexicographically by descending positive-volume count, descending log
   volume score, then ascending filename as the deterministic tie-breaker.
5. Freeze at most the first 80 contracts for the following evaluation interval.
6. Define the top-50 sensitivity as the first 50 contracts from the identical
   ranking. Do not run a separately selected top-50 universe.

If fewer than 80 contracts qualify, use all qualifying contracts and record the
shortfall. Never fill a walk using a contract whose ranking history matures
after the cutoff. Universe manifests must store candidates, exclusion reasons,
scores, ranks, source hashes, and the exact lookback rows.

The evaluation universe remains fixed until the next cutoff. Newly listed
contracts can enter the next walk after satisfying its history and ranking
requirements; they do not enter midway through an evaluation interval.

## 6. Causal row activity

Within the frozen universe, a row is primary-eligible only if at least one
non-zero close-to-close change occurred in the six most recent four-hour
intervals ending at the decision bar. This is the causal prior-24h price-change
rule.

The rule:

- uses only observations available at the decision time;
- applies identically to encoder inputs, framework task rows, and baselines;
- permits a contract to become inactive and later re-enter if movement resumes;
- must be computed independently inside each contract; and
- must not be replaced by retrospective truncation at the last observed price
  change.

Primary metrics use the active rows. A required sensitivity evaluates every
otherwise eligible row in the same frozen universe. This makes explicit that
the primary estimand is movement prediction for recently active contracts, not
for every emitted post-resolution candle.

## 7. Exact row allocation

### 7.1 Encoder pretraining rows

A sequence may update an encoder for walk `k` only when:

- all 64 bars belong to one contract;
- its raw input window begins at or after the rolling training start;
- its final bar is available strictly before `T_k`; and
- its decision row passes the causal activity rule.

No evaluation sequence, target, activity observation, or universe statistic may
affect the walk's encoder.

### 7.2 Supervised training rows

A downstream or baseline training row is eligible only when:

- its complete input window satisfies the encoder-window conditions;
- its target belongs to the same contract;
- its target availability time is strictly before `T_k`; and
- it passes the identical activity rule.

Every fitted imputer, scaler, target transformation, aggregator, task head, and
baseline parameter uses only these permitted training identities.

### 7.3 Evaluation rows

An evaluation row is eligible only when:

- its decision availability is at or after `T_k`;
- its target availability is strictly before the evaluation end;
- input and target belong to the frozen contract universe and the same
  contract; and
- it passes the identical causal activity rule.

An evaluation input may use historical context before `T_k`; this is valid
decision-time information. No evaluation-period observation may enter a
training input, target, fitted state, gradient update, or selection rule.

## 8. Fold-specific model lifecycle

Every walk has independent:

- preprocessing/imputation/scaling state;
- VAE checkpoint;
- contrastive CNN/LSTM/Transformer checkpoints;
- BYOL CNN/LSTM/Transformer checkpoints;
- deterministic and neural feature bundles;
- representation aggregator and downstream task head; and
- baseline model weights.

The primary adaptive result retrains the same declared architecture in each
walk. Reusing walk 1 encoder weights in walk 2 is a separate temporal-transfer
ablation, not the primary result. Phase 3 weights are not Phase 4 inputs.

There is no validation split, early stopping, or test-driven checkpoint
selection. Each deep model follows one uninterrupted 50-epoch trajectory with
snapshots at epochs `5`, `15`, and `50`. All snapshots are reported; no epoch
is chosen from evaluation performance.

## 9. Required reporting and baselines

Probability-movement regression must report:

- MAE, RMSE/MSE, Pearson and Spearman correlation, and sign agreement;
- exact-zero movement as the mandatory primary baseline;
- pooled row-weighted and contract-macro metrics;
- metrics for each walk separately and pooled only after causal prediction;
- contract count, row count, exact-zero/stable share, boundary-price share,
  and lifecycle composition for every reported partition;
- early/middle/late lifecycle strata inside each calendar walk; and
- non-zero and threshold-exceeding movement diagnostics.

Framework configurations and all internal/external baselines must consume the
same stored row identities and activity mask. The top-50, one-hour, all-row,
fixed-first-walk-encoder, and any lifecycle-conditioned result must be labelled
as sensitivities or ablations rather than silently mixed into the primary
matrix.

## 10. Artifact and validation requirements

Phase 4 artifacts live under `experiments/phase4/` and are produced only by
new `scripts_v2/` entry points. Each walk manifest must preserve:

- interval boundaries and their inclusivity;
- candle-start and availability timestamps;
- universe candidate/ranking provenance;
- active/all-row eligibility identities;
- contract, window-start, decision, and target identities;
- fitted preprocessing provenance;
- source, checkpoint, feature, label, prediction, and configuration hashes;
- fixed architecture, seed, and epoch-budget declarations; and
- replayed metrics from stored predictions.

Validation must reject cross-contract rows, immature training targets, windows
outside the rolling training interval, evaluation targets crossing the interval
end, post-cutoff fitted state, universe selection using later history, row-mask
differences between strict comparators, or non-finite arrays.

## 11. Historical execution gate

This document freezes the design but does not authorize immediate training.
Before the first Phase 4 encoder run:

1. implement the walk/universe/activity builder in `scripts_v2/`;
2. generate manifests without training;
3. validate raw-time identities and all cutoff invariants;
4. confirm final per-walk universe, train, and evaluation counts; and
5. freeze the experiment matrix and launch order.

No Phase 4 training was launched. The unresolved implementation work is
superseded by the Phase 5 handoff in the Phase 4 conclusion rather than being
silently counted as completed.

## 12. Recent FinData source extension

An independent FinData acquisition module now covers `[2025-12-01,
2026-09-01)`. The initial three-contract dry run was expanded first to 24 and
then to a raw-first retrospectively diverse 50-market cohort. A source-semantics audit found that
condition-level candles can mix complementary YES/NO token prices, so those
candles are not canonical probability histories. The forward-confirmed local
quarantine removes 0.0356% of 15-minute and 0.1366% of hourly rows while
retaining persistent large moves, but remains exploratory. The safe token-identified
trade fallback yielded 28,359 YES trades from 12 conditions, but only 448
complete four-hour `seq64+h2` rows, all from two related World Cup contracts.
This remains recent-source feasibility evidence, not an amendment to the
primary Phase 4 universe or walk schedule.

No FinData artifact may enter Phase 4 training until an amendment freezes a
cutoff-local diverse universe, event-family handling, causal gap policy,
token-specific probability source, and revised recent-period
walk/confirmation boundaries. The expanded audit and reproduction commands
are recorded in
`docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md`.
