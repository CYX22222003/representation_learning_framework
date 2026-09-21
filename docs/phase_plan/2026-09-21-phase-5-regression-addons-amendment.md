# Phase 5 Additional Regression Tasks Amendment

**Date:** 2026-09-21
**Status:** Implemented, executed at seed 0, and replay-validated
**Parent authority:** `2026-09-21-phase-5-experiment-plan.md`

## 1. Motivation and status

The primary two-hour raw-probability-change regression is already complete.
Its absolute errors are small, but its Pearson correlation is weak, Spearman
correlation is approximately zero, and non-zero-target sign agreement is near
chance. Two additional framework-only regressions are therefore approved to
diagnose whether the result is specific to the short horizon or additive
probability-change target.

These experiments were specified after observing the primary result. They are
additional downstream regression tasks, not replacement targets selected by
test performance. The existing two-hour raw-change result remains primary.

## 2. Frozen tasks

| Identifier | Scientific target | Isolated question |
|---|---|---|
| `raw_delta_h8` | `close[t+8h] - close[t]` | Is two hours too noisy for the 64-hour representation? |
| `log_return_h2` | `log(close[t+2h] / close[t])` | Is movement more predictable in proportional than additive units? |

Both retain the 64-hour five-channel context, prior-24-hour causal activity
rule, two calendar walks, walk-specific epoch-50 encoders, canonical
five-branch 445-dimensional concat representation, seed 0, batch size 512,
Adam learning rate `1e-4`, 50 epochs, and snapshots at 5/15/50. Epoch 50 is
predeclared as the principal result. There is no validation split, early
stopping, or evaluation-driven checkpoint selection.

## 3. Eight-hour row construction

The eight-hour task receives a separate replayable data bundle. Encoder rows
must remain byte-identical to the primary two-hour bundle because encoder
eligibility is target-free. Supervised rows are independently rebuilt so that:

- the exact `t+8h` target exists in the same contract and contiguous gap
  segment;
- the target endpoint is observed;
- a training target becomes available strictly before the walk cutoff;
- an evaluation target becomes available strictly before the evaluation end;
  and
- evaluation contracts have active eight-hour supervised training support.

The same walk-local volume scaler is recovered from the same permitted candle
history. The existing encoder checkpoints are valid only after their target-
free encoder population is proven byte-identical across the two-hour and
eight-hour bundles.

## 4. Target optimization and reporting

`raw_delta_h8` uses the fixed probability-point optimization unit
`100 * delta`, inverted before reporting. `log_return_h2` requires strictly
positive current and future prices. It is standardized using its training
mean and population standard deviation only; the transform is frozen and
inverted before scientific metrics.

Both tasks report MAE, RMSE/MSE, Pearson, Spearman, sign agreement,
contract-macro, lifecycle, imputation-exposure, and starting-price-band
breakdowns. Log return additionally reconstructs:

```text
predicted_future_probability = current_probability * exp(predicted_log_return)
```

and reports reconstructed probability MAE/RMSE plus the fraction outside
`[0,1]`. Starting-price bands are `p<=0.01`, `(0.01,0.10]`, `(0.10,0.90)`,
and `p>=0.90`.

Zero change is retained as a descriptive sanity reference, not an exploratory
pass/fail gate. Learned Raw-OHLCV and temporal baselines remain a later final-
comparison stage.

## 5. Artifact contract

All generated artifacts remain under `experiments/phase5/downstream_addons/`:

```text
shared/h8/data/
shared/h8/features/
shared/h8/feature_scalers/
tasks/{raw_delta_h8,log_return_h2}/
reports/regression_tasks_seed0/
reports/regression_rank_ic_seed0/
manifests/regression_tasks_seed0.json
```

Every run must retain configuration, data/feature/scaler hashes, target
transform, checkpoints, cumulative histories, predictions, detailed metrics,
completion marker, and CPU prediction replay.

## 6. Completed results

The eight-hour bundles contain 36,773/29,834 supervised train/evaluation rows
in Walk 1 and 56,652/13,506 in Walk 2. Their encoder populations are
byte-identical to the primary two-hour bundles. More than 97% of the required
feature rows were reused only after exact identity and context matching; the
remaining 1,777 contexts were freshly extracted. Both add-on feature
stores contain 445 finite coordinates and pass hash and sample replay.

All four heads completed one seed-0 50-epoch trajectory with 5/15/50
snapshots. Every checkpoint, target transform, prediction identity, and CPU
prediction replay passed.

| Task | Walk | MAE | RMSE | Pearson | Spearman | Sign agreement | Reconstructed probability MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eight-hour raw change | 1 | 0.00416801 | 0.01274698 | 0.059602 | 0.007261 | 0.384628 | 0.00416801 |
| Eight-hour raw change | 2 | 0.00542461 | 0.02126556 | -0.047109 | -0.033975 | 0.358655 | 0.00542461 |
| Two-hour log return | 1 | 0.05899403 | 0.10815900 | -0.005980 | 0.005214 | 0.366678 | 0.00252935 |
| Two-hour log return | 2 | 0.06260438 | 0.16412535 | -0.015256 | -0.003790 | 0.334125 | 0.00310111 |

Pooled eight-hour raw-change Pearson/Spearman are `0.014209/-0.005372`;
non-zero-target sign agreement remains approximately chance (`0.5040` in
Walk 1 and `0.4840` in Walk 2). Moving from two to eight hours therefore does
not recover stable signed predictability. Its pooled MAE/RMSE are
`0.00455960/0.01589894`, while the descriptive no-change reference records
`0.00342947/0.01505473`.

Pooled two-hour log-return Pearson/Spearman are `-0.010259/0.003947`.
Reconstructed probability MAE is `0.00270888`, essentially the same as the
primary raw-change model's `0.00271136`, while reconstructed RMSE is worse
(`0.01142996` versus `0.01083505`). The log transform changes target weighting
but does not reveal additional out-of-future relationship.

The additional-task evidence therefore strengthens the interpretation that the
current representation/head can detect movement activity but does not recover
the future signed direction or ordered magnitude. Neither a longer horizon nor
ordinary log returns resolves the primary regression limitation. These are
exploratory framework-only findings; learned raw and temporal baselines remain
necessary to distinguish representation loss from intrinsic target difficulty.

## 7. Rank IC diagnostic

The saved Spearman metric is the global rank correlation across all evaluation
rows. A finance-style Rank IC diagnostic additionally computes Spearman
correlation cross-sectionally at every decision timestamp with at least five
active contracts and summarizes the resulting hourly IC series.

| Target | Pooled mean cross-sectional Rank IC | Median | Unannualized ICIR | Positive fraction |
|---|---:|---:|---:|---:|
| Two-hour raw change | 0.003805 | 0.000000 | 0.012158 | 0.4975 |
| Eight-hour raw change | -0.002830 | -0.000759 | -0.008836 | 0.4956 |
| Two-hour log return | 0.004015 | -0.001470 | 0.013715 | 0.4951 |

All three means and ICIRs are approximately zero, and positive IC fractions
are approximately 50%. The predictions therefore show no stable
cross-sectional ranking power. Hourly stride-one decisions and targets overlap,
so these ICIRs are descriptive and are not annualized or treated as
independent-observation significance statistics. The complete series and
summary are stored under `experiments/phase5/downstream_addons/reports/regression_rank_ic_seed0/`.
