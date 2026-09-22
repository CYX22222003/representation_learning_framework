# Phase 5 Eight-Hour Absolute-Price Add-on Amendment

**Date:** 2026-09-21
**Status:** Implemented, executed at seed 0, and replay-validated
**Parent authority:** `2026-09-21-phase-5-experiment-plan.md`

## 1. Purpose

This framework-only exploratory task tests whether direct future-probability
prediction is easier for the frozen representation than signed movement
regression on the new Phase 5 one-hour calendar walks. It does not reuse the
older Phase 1--3 four-hour data or per-contract final-tail evaluation.

The scientific target is:

```text
absolute_price_h8[t] = close[t + 8 hourly bars]
```

The already validated `raw_delta_h8` bundles provide the exact observed,
same-segment, mature target rows. Their target-free encoder populations and
walk-specific epoch-50 encoders remain unchanged.

## 2. Frozen model and training contract

- Input: canonical 445-dimensional concat representation.
- Head: `445 -> 128 -> 64 -> 1` with the existing regression dropout pattern.
- Output transform: sigmoid, enforcing a future probability in `[0,1]`.
- Loss: MSE directly in raw probability units.
- Optimizer: Adam, learning rate `1e-4`, no weight decay.
- Batch size: 512.
- Seed: 0.
- One uninterrupted 50-epoch trajectory per walk.
- Snapshots: epochs 5, 15, and 50.
- Principal result: epoch 50, fixed before evaluation.
- No validation split, early stopping, or evaluation-driven selection.

The current close is not supplied as an explicit skip connection. Adding
`current_close + predicted_delta` would merely reproduce the already executed
raw-change task. The head must infer future level from the saved representation.

## 3. Evaluation contract

### Price-level view

Report MAE, RMSE/MSE, Pearson, Spearman, prediction range, and contract-macro,
lifecycle, imputation-exposure, and starting-price-band breakdowns.

The descriptive persistence reference is:

```text
predicted_future_price = current_close
```

Report MAE/RMSE/MSE skill relative to persistence. This reference is
diagnostic rather than an exploratory pass/fail gate.

### Implied-movement view

Convert the direct price output to:

```text
predicted_delta_h8 = predicted_future_price - current_close
```

Compare it with the realised eight-hour raw change using Pearson, Spearman,
non-zero sign agreement, global Rank IC, and timestamp-level cross-sectional
Rank IC with at least five active contracts. This distinguishes probability-
level reconstruction from incremental movement prediction.

## 4. Artifact contract

All outputs remain under:

```text
experiments/phase5/downstream_addons/tasks/absolute_price_h8/walk{1,2}/seed0/
experiments/phase5/downstream_addons/manifests/absolute_price_h8_seed0.json
experiments/phase5/downstream_addons/reports/absolute_price_h8_seed0/
```

Every run retains configuration and source hashes, checkpoints, cumulative
histories, predictions, full metrics, completion marker, and CPU prediction
replay.

## 5. Completed results

Both walk-specific heads completed 50 epochs with 5/15/50 snapshots, bounded
predictions, cumulative histories, and CPU prediction replay.

| Walk | Price MAE | Price RMSE | Price Pearson | Price Spearman | Persistence MAE | Implied-delta Pearson | Implied-delta Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00798540 | 0.01890583 | 0.996198 | 0.951078 | 0.00297826 | 0.209421 | 0.147363 |
| 2 | 0.01005353 | 0.03007260 | 0.987541 | 0.982321 | 0.00442618 | -0.003541 | 0.099988 |

Pooled price-level MAE/RMSE are `0.00862989/0.02297542`, with Pearson
`0.993783`. Persistence remains materially more accurate at
`0.00342947/0.01505473`; the direct-price model's pooled MSE skill relative to
persistence is `-1.3291`. High price-level correlation therefore primarily
demonstrates market-state reconstruction.

The implied eight-hour movement is more informative than the prior movement
heads: pooled Pearson/Spearman are `0.102503/0.128726`, and non-zero sign
agreement is `0.5585` in Walk 1 and `0.5549` in Walk 2. Mean timestamp-level
cross-sectional Rank IC is `0.143956` in Walk 1 and `0.094730` in Walk 2;
pooled mean/median IC are `0.126384/0.136364`, with 65.83% positive IC
timestamps and an unannualized ICIR of `0.3815`.

However, a simple last-hour reversal score,
`-(close[t] - close[t-1])`, is substantially stronger: mean cross-sectional
Rank IC is `0.301238` in Walk 1, `0.240011` in Walk 2, and `0.279444` pooled.
The direct-price objective therefore uncovers a consistent reversal-related
ranking signal, but it does not yet establish representation value over simple
raw temporal information. A matched raw temporal comparator is the next
decisive experiment.
