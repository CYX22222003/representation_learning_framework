# Phase 6.9 xLSTM-Mixer Endpoint-Only Amendment

**Date:** 2026-10-09
**Authority:** Owner-approved replacement of the active price contract in
`2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`.

## Decision

The primary xLSTM-Mixer comparator is now **XM-C8**, a direct endpoint-only
adaptation. The completed **XM-MV8** full-path experiment is immutable
historical/contextual evidence, not the headline endpoint-matched comparator.
This amendment follows review of XM-MV8 results and is explicitly a
post-result protocol correction, not a previously precommitted experiment.
No endpoint result has been used to select this recipe. Strong confirmatory
generalization claims require fresh data or separately approved replication.

## Exact target and data

- Input: the existing float32 historical `[B,64,5]` OHLCV contexts.
- Output and sole label: `[B,1]`, raw `close[t+8 hourly bars]`.
- One forecast endpoint eight hours ahead; not `close[t+1]` and not a
  supervised eight-step path followed by endpoint extraction.
- Reuse the original immutable Phase 5 h8 bundles under
  `experiments/phase5/downstream_addons/shared/h8/data/walk{1,2}/`.
- Preserve every original train/test identity, order, context, target,
  segment/maturity rule, observed endpoint rule, and metadata field.
- Do not require intermediate future candles to be observed. Do not fetch
  future OHLCV, fit a new scaler, delete rows, or create an intersection.

| Walk | Training rows | Evaluation rows |
|---:|---:|---:|
| 1 | 36,773 | 29,834 |
| 2 | 56,652 | 13,506 |

The original walk calendars and accepted retrospective selection/offline
cleaning assumptions remain unchanged. Train-only Phase 5 volume scaling is
reused. Train/test only: no validation split or early stopping.

## Architecture and training freeze

Keep the audited non-affine RevIN, last-value-centred shared NLinear,
128-wide one-block/eight-head sLSTM, kernel 0, dropout 0.1, one learned initial
token, packing 1, released latent-feature reversal, and reverse-first shared
stack call order. Change the shared time projection to `64 -> 1`, up
projection to `1 -> 128`, and output projection to `256 -> 1`. Decode only
the mixed close token and inverse-normalize with its historical close RevIN
statistics. The preliminary five scalars are internal features, not future
labels or auxiliary predictions.

The retained inverse-RevIN output is **unclipped**; no sigmoid is applied
after inverse normalization. Report out-of-range forecasts without repair.
This is a disclosed architectural difference from the original sigmoid price
heads, not a difference in target, loss, rows, or supervision. It is an
endpoint adaptation, not a reproduction of the paper's full-path objective.

- MSE averaged over the sole raw-probability endpoint, matching price controls.
- Float32, Adam `1e-4`, weight decay 0, batch 512, seed 0, no AMP/scheduler.
- Retain clip norm 1.0 from the prior xLSTM runtime recipe; disclose this
  stabilization difference from controls that did not use clipping.
- Fresh independent weights per walk; no reuse of XM-MV8 learned weights.
- Exactly 50 epochs, snapshots 5/15/50, epoch 50 predeclared principal.
- Evaluate only after the trajectory; never select a checkpoint or recipe
  using evaluation results.

## Controls, artifacts, and gates

Original H0-D0 and Raw LSTM results may be reused only after checking original
dataset hashes, both train/test identities and targets, seed/budget/optimizer
settings, prediction hashes, and their existing standalone replay evidence.
The former mandatory intersection reruns are **superseded**, not completed.
Matching rows and endpoint supervision still does not isolate architecture
causality or turn XM-C8 into a frozen-representation comparator.

New artifacts belong exclusively under
`experiments/phase6_9/xlstm_mixer_endpoint/`. The original
`experiments/phase6_9/xlstm_mixer/` artifacts and fingerprinted implementation
remain unchanged and replayable. XM-MV8 checkpoints cannot resume XM-C8.

Use a new method-specific CPU smoke, exact-source data audit, immutable
two-walk matrix, and selected-backend CUDA admission. Training remains
explicit `--execute` only. Epoch-resume must guard model/source/code/data/
dependency/runtime identities and retain optimizer and RNG state. Checkpoint,
prediction, identity, target, history, and metric replay is required before
completion. Compare price MAE/RMSE, persistence skill, implied movement and
timestamp Rank IC (at least five contracts), subgroups, and resources under
the existing price evaluation definitions. Report the full two-walk outcome,
not only an observed winner.

Monotone-VI, volatility, optional Phase 6.8, Phase 7A, and the deferred
decoder/fusion studies are unchanged by this price-only amendment.
