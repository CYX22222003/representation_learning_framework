# Phase 5 Intermediate Experiment Observation

**Date:** 2026-09-22
**Status:** Intermediate evidence and reporting decision; Phase 5 remains open
**Authority:** Supplements `2026-09-21-phase-5-experiment-plan.md` without
retrospectively changing the predeclared execution contracts

## 1. Scope

Phase 5 has completed the recent-data walk builder, six walk-specific neural
encoder trajectories, canonical five-branch feature extraction, two-hour
movement regression/classification heads, additional raw-change/log-return
tasks, and an eight-hour absolute future-price task. This document records the
current interpretation before learned-baseline execution, additional seeds,
and final Phase 5 conclusions. The baseline pipeline was subsequently
implemented under `2026-09-22-phase-5-baseline-amendment.md` but remains
unexecuted.

The evidence is generated from two global-calendar walks with separately
trained encoder and downstream weights. All fitted preprocessing uses permitted
training history only; evaluation rows, targets, and predictions replay from
saved identities and checkpoints.

## 2. Current downstream evidence

### 2.1 Movement classification

The two-hour three-class framework obtains pooled macro-F1 `0.4541`, balanced
accuracy `0.4730`, and macro ROC-AUC `0.7412`. It improves minority-sensitive
metrics over always-`STABLE`, although further diagnostics show that much of
the learned signal distinguishes movement from stability rather than `UP`
from `DOWN`.

### 2.2 Direct movement and return regression

Direct two-hour raw-change, eight-hour raw-change, and two-hour ordinary
log-return heads do not recover stable signed ordering. Their pooled mean
cross-sectional Rank IC values are `0.0038`, `-0.0028`, and `0.0040`,
respectively, with approximately half of timestamps positive. These tasks are
retained as useful negative characterization evidence about target formulation.

### 2.3 Eight-hour future-price regression

Direct sigmoid-bounded prediction of `close[t+8h]` is a clearer regression
transfer task. It obtains:

| Metric | Walk 1 | Walk 2 | Pooled |
|---|---:|---:|---:|
| Price Pearson | 0.9962 | 0.9875 | 0.9938 |
| Price Spearman | 0.9511 | 0.9823 | 0.9644 |
| Implied-change Pearson | 0.2094 | -0.0035 | 0.1025 |
| Implied-change Spearman | 0.1474 | 0.1000 | 0.1287 |
| Mean cross-sectional implied-change Rank IC | 0.1440 | 0.0947 | 0.1264 |

The pooled Rank IC median is `0.1364`, 65.83% of eligible timestamps have
positive IC, and non-zero movement sign accuracy is approximately 55.7%.
Unlike the direct movement heads, the price-supervised decoder therefore
extracts a consistent cross-contract ranking of subsequent eight-hour
probability movements from the frozen representation.

## 3. Updated transferability interpretation

For the final framework narrative, eight-hour future-price prediction is the
clearest regression audience for the transferable representation:

1. the target is directly interpretable as a future market probability;
2. predictions are naturally bounded to `[0,1]`;
3. the same frozen five-branch representation supports both probability-level
   prediction and movement classification; and
4. subtracting the decision-time price produces an economically relevant
   implied-movement score with positive Rank IC in both walks.

This does not erase the original movement-regression experiment. The latter
remains the predeclared Phase 5 task and documents that direct optimization of
small signed changes did not transfer well. The price task is the stronger
empirical demonstration of regression transfer discovered during the Phase 5
iteration.

Point-error metrics and financial ranking metrics measure different
capabilities. Detailed constant-price comparisons remain in the experiment
artifacts and supporting/appendix material. The main research discussion will
focus on future-price prediction and implied-movement Rank IC, while noting
briefly that conservative level forecasts can minimize error without ranking
subsequent returns. A constant implied-change score has undefined Rank IC and
cannot rank contracts.

## 4. Candidate short-horizon reversal factor

The causal score

```text
last_hour_reversal[t] = -(close[t] - close[t-1])
```

has mean cross-sectional Rank IC `0.3012` in Walk 1, `0.2400` in Walk 2, and
`0.2794` pooled. Its pooled median IC is `0.2990`, 83.30% of timestamps are
positive, and its unannualized ICIR is `0.9148`.

This is a strong candidate short-horizon reversal factor in the accepted
recent Polymarket condition-candle data. It is an empirical domain finding,
not a claim of a new alpha-mining algorithm or established trading
profitability. It was identified after examining the completed evaluation and
therefore requires confirmation on a later walk or fresh holdout, together
with transaction costs, liquidity, signal-spacing, and source-orientation
checks before a profitable-alpha claim.

The framework implied score is not merely the reversal score: their rank
correlation is only approximately `0.18--0.19`. This leaves open the
possibility that the representation supplies complementary probability-level,
lifecycle, activity, or volatility information. Incremental value has not yet
been established.

## 5. Leakage and validity judgement

No future-information path has been found in the executed price task or
reversal diagnostic:

- encoder eligibility is target-free;
- each walk's encoder and head use only information available before its
  cutoff;
- eight-hour targets are observed, same-segment, and mature within their
  train/evaluation interval;
- feature scaling is fitted on supervised training features only;
- current and preceding prices used by reversal are decision-time information;
- evaluation rows are disjoint from encoder and supervised training rows; and
- epoch 50 was fixed before evaluation and all predictions replay from saved
  checkpoints.

This judgement remains conditional on the accepted Phase 5 assumptions of
retrospective cohort selection and approved offline condition-candle pruning.

## 6. Claims currently supported

The intermediate evidence supports the following statements:

1. The canonical five-branch representation transfers to recent-data
   probability-level regression and movement classification using lightweight
   task heads.
2. Direct future-price supervision extracts substantially stronger subsequent-
   movement ranking information than direct raw-change or log-return
   supervision.
3. The price model's implied movement has positive cross-sectional Rank IC in
   both global-calendar walks.
4. Recent Polymarket condition candles exhibit a strong candidate one-hour
   reversal relationship with subsequent eight-hour movement.

The evidence does not yet establish:

- superiority over matched raw MLP or sequential models;
- incremental framework value beyond the reversal factor;
- statistically independent IC observations, because stride-one horizons
  overlap;
- profitable execution after spreads, fees, and liquidity; or
- generalization beyond the two accepted Phase 5 cohorts.

## 7. Immediate next experiments

Before the final Phase 5 conclusion:

1. train matched Raw-OHLCV MLP and raw temporal LSTM/TCN price models on the
   exact eight-hour rows;
2. include last-hour reversal as a fixed causal reference;
3. test whether framework implied movement retains IC after training-only
   neutralization against reversal;
4. evaluate a training-only combination of framework and reversal scores;
5. confirm the reversal finding on a later walk or fresh holdout; and
6. complete the learned classification comparisons on identical two-hour
   rows.

All additional-task artifacts are organized under
`experiments/phase5/downstream_addons/`.
