# Phase 3 Experiment Observation and Conclusion

Date: 2026-09-20
Status: Phase 3 concluded; Phase 4 redesign required before further downstream execution

Tracking issue: [#19 — Phase 4: global calendar walk-forward, fold-specific encoders, and movement regression](https://github.com/CYX22222003/representation_learning_framework/issues/19)

## 1. Scope actually completed

Phase 3 rebuilt the 4-hour experiment path after the Phase 1/2 leakage audit.
It completed:

- a raw-time-first, training-prefix-selected top-50 processed bundle;
- seed-0 VAE, contrastive CNN/LSTM/Transformer, and BYOL
  CNN/LSTM/Transformer pretraining trajectories;
- frozen statistical, transformation, and seven neural feature branches;
- contract-local horizon-1 price labels;
- the 15-configuration H0/HC/HB framework price matrix at epochs 5, 15, and
  50; and
- checkpoint-provenance, row-identity, prediction, and full-metric replay.

The planned Phase 3 movement-classification and matched price/classification
baseline matrices were not executed. They are not Phase 3 completion work any
longer: Phase 3 is deliberately closed because the downstream evaluation
contract must change first.

## 2. What Phase 3 corrected

Phase 3 removed the future-information defects identified in the legacy
pipeline:

1. Each contract's raw chronological boundary was established before fitted
   preprocessing or window generation.
2. Universe ranking used only a fixed 256-row training prefix rather than
   full-file activity or availability.
3. Imputation and volume statistics were fitted from permitted training
   history only and applied causally.
4. Training and test windows were constructed independently and share no raw
   observations.
5. Encoders trained only on training windows and were frozen before test
   inference.
6. Downstream feature standardisation used eligible training rows only.
7. Horizon targets were built independently inside every contract and split;
   terminal rows were removed, preventing cross-contract and cross-split
   targets.
8. The downstream matrix was predeclared and all fixed snapshots were
   reported without early stopping or test-driven checkpoint selection.

The resulting Phase 3 measurements are internally leakage-safe under their
declared contract. They remain characterisation evidence because the broader
research design was informed by earlier observations on the same historical
period.

## 3. Encoder observations

All seven seed-0 encoders completed one uninterrupted 50-epoch trajectory with
snapshots at epochs 5, 15, and 50. The consolidated health report found no
representation-collapse warning. Epoch 50 was frozen in advance as the feature
source for the downstream matrix; no task metric selected an encoder snapshot.

The downstream price matrix showed that the three-backbone BYOL addition
(`HB-ALT`) had the lowest framework error at epoch 50 (MAE `0.009234`, RMSE
`0.022782`, correlation `0.998993`). This ranking describes the framework
matrix only. It does not establish useful forecasting performance because the
correct no-change reference is much stronger.

## 4. Downstream discovery 1: the single tail split is lifecycle-biased

The original raw 80/20 split is chronologically honest, but the last 20% of a
prediction-market contract is not exchangeable with its earlier history.
Contracts frequently approach settlement during the tail: prices cluster near
zero or one and update less often.

The 4-hour horizon-1 label audit found:

| diagnostic | training | test |
|---|---:|---:|
| eligible rows | 21,646 | 3,035 |
| exactly unchanged next close | 43.74% | 75.45% |
| no-change MAE | 0.006703 | 0.001063 |
| no-change RMSE | 0.015455 | 0.003789 |
| current price at or near 0/1 | 36.10% | 77.83% |
| 95th percentile absolute move | 0.0300 | 0.0060 |

Thus persistence exists in both partitions, but it is far stronger in the
chronological test tail. The test price-level standard deviation is high
because the pooled distribution is bimodal near zero and one; high
cross-contract level variance does not imply high within-contract next-step
movement.

This is not future leakage. It is a mismatch between the single-tail
evaluation and the intended question of representation usefulness throughout
the contract lifecycle.

## 5. Downstream discovery 2: next-close prediction is not informative

The horizon-1 absolute-close target is dominated by the identity

```text
future close approximately equals current close.
```

On the exact 3,035 Phase 3 test identities, the causal no-change persistence
reference achieved MAE `0.001063`, RMSE `0.003789`, and correlation `0.999967`.
The best framework row, `HB-ALT` at epoch 50, achieved MAE `0.009234` and RMSE
`0.022782`: approximately 8.7 times the persistence MAE and 6.0 times its RMSE.
Persistence also remained better on non-zero moves and on absolute movements
above `0.001`, `0.005`, and `0.01`.

The framework therefore learned approximate price level but did not add value
over copying the latest observable close. High absolute-price correlation is
not evidence of incremental forecasting information in this setting.

## 6. Phase 4 solution 1: global calendar walk-forward evaluation

A further design audit found that per-contract lifecycle fractions cannot be
the primary split for a model pooled across many prediction-market contracts.
Contracts begin, trade, and resolve on different calendar dates. A shared
encoder could otherwise train on a late timestamp from one contract and be
scored on an earlier timestamp from another contract. That is impossible in a
real deployment even if every individual contract remains internally ordered.
Defining fractions from a contract's final observed length can also use future
lifetime information unless the resolution schedule was known at decision
time.

A targeted follow-up on 27,831 eligible 4-hour, horizon-2 rows confirms that
the lifecycle effect is systematic across contracts. From the early to late
third, the exact-zero movement rate increased from `30.00%` to `58.06%`, the
near-boundary price rate increased from `29.62%` to `55.97%`, and the causal
zero-movement baseline MAE fell from `0.010659` to `0.004268`. Contract-paired
bootstrap intervals for all three late-minus-early effects excluded zero.
Every saved Phase 3 feature branch also made lifecycle stage descriptively
separable above nominal chance in a contract-grouped linear probe. This
supports testing temporal representation adaptation, but it does not establish
that different encoder architectures are necessary. See the
[calendar/lifecycle exploration](../data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md).

The subsequent Phase 4 feasibility study froze two fixed-duration rolling
**global calendar-time** walks. The canonical intervals, top-80 cutoff-local
universe, and causal activity rule are specified in
[`2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`](2026-09-20-phase-4-data-selection-and-walk-forward-contract.md).
Conceptually:

| walk | permitted training information | next evaluation interval |
|---|---|---|
| 1 | eligible observations inside rolling window 1 and available before `T1` | decision times in `[T1, T2)` |
| 2 | eligible observations inside rolling window 2 and available before `T2` | decision times in `[T2, T3)` |

The exact timestamps, two-walk count, minimum-history rules, contract-universe
rule, target-maturity rule, and activity mask are now frozen in the Phase 4
contract; implementation and manifest validation remain. Per-contract
early/middle/late lifecycle buckets remain useful reporting strata inside each
global evaluation interval; they are not allowed to define the primary
training boundary.

For every walk, no information whose availability timestamp is at or after the
training cutoff may affect:

- universe membership or eligibility;
- imputation or scaling;
- input windows or labels;
- encoder, downstream head, or baseline parameters; or
- configuration and checkpoint selection.

For the primary adaptive-system evaluation, each global walk receives its own
preprocessing state, neural encoder weights, feature bundles, downstream head,
and baseline weights. The encoder is trained only on that walk's permitted
history and frozen before inference in the next interval. It is valid for the
encoder and downstream head to learn from the same permitted training prefix;
unsupervised pretraining does not exempt the encoder from the calendar cutoff.
Phase 3 weights cannot be reused because their 80% training prefix overlaps
the proposed Phase 4 evaluation history.

Training information admitted by a later calendar walk must never update or
select a model evaluated in an earlier walk. An optional transferability
ablation may train the encoder at the first cutoff, keep it fixed in later
walks, and retrain only the downstream head. That ablation measures temporal
encoder reuse; it does not replace the fold-specific adaptive evaluation.
If decision-time lifecycle metadata and within-stage sample sizes are adequate,
an additional predeclared ablation may compare a lifecycle-conditioned shared
model or stage-specific experts. This is separate from the fixed-versus-adaptive
encoder comparison and remains a hypothesis until paired downstream results
show an advantage on identical walk rows.
Predictions may be aggregated only after every row has been generated strictly
out of future data. Lifecycle-stage results are then reported within each
calendar walk, including the final tail rather than using it as the only test
regime.

## 7. Phase 4 solution 2: probability-movement regression

Phase 4 replaces absolute next-close regression as the primary regression
probe with continuous probability movement:

```text
delta[t,h] = close[t+h] - close[t]
```

The frozen Phase 4 contract aligns with the existing classification concept by
using 4-hour data and the predeclared `h=2` horizon (eight hours). Signed probability change, optionally reported as
probability points (`100 * delta`), remains the primary target. Conventional
arithmetic return `delta / close[t]` is a secondary exploratory target because
it is unstable and strongly reweights observations when prediction-market
prices approach zero. Any return experiment must predeclare zero-price
eligibility, use train-only target scaling, report results by starting-price
band, and reconstruct future probability for a common-scale comparison.

Movement classification and movement regression are related but distinct:

- classification maps the continuous delta through a fixed threshold into
  `DOWN`, `STABLE`, and `UP` and evaluates directional discrimination; and
- regression retains the sign and magnitude of the continuous delta.

The exact zero-movement predictor is the required primary regression baseline.
Raw-OHLCV MLP and sequential baselines must consume identical walk identities.
Report MAE, RMSE/MSE, Pearson and Spearman correlation, sign agreement,
per-contract and per-lifecycle-stage results, and separate diagnostics for
non-zero and threshold-exceeding movements. Current close is decision-time
information and, if supplied explicitly, must be supplied consistently across
all applicable framework configurations and baselines.

## 8. Interpretation and transition

Phase 3 succeeded at its most important methodological goal: it demonstrated
an end-to-end path without future leakage and exposed limitations that legacy
experiments could not cleanly diagnose. Its absolute-close results are retained
as valid negative characterisation evidence, not discarded or relabelled.

Phase 4 begins at the data/evaluation-contract layer. The canonical Phase 4
document now specifies the rolling walks, fold-specific encoder policy,
lifecycle reporting rule, provenance rules, movement target, zero-movement
baseline, and sensitivities. No Phase 4 encoder, downstream, or baseline
experiment should launch until its builder, manifests, row identities, replay
checks, and complete matrix are implemented and validated. Existing Phase 3
artifacts remain immutable and are not Phase 4 inputs unless the Phase 4 plan
explicitly proves that reuse is causal for every fold.
