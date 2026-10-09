# xLSTM-Mixer Phase 6.9 integration proposal

> **Superseded price contract (2026-10-09):** XM-C8 endpoint-only adaptation
> is now primary. Read
> `../../phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md`.
> Use every original h8 row, one close target, MSE, and verified original
> H0/Raw-LSTM results. This proposal's XM-MV8 path/intersection requirements
> remain historical context, not an active launch specification.

## 1. Research role

`XM-MV8` answers:

> How competitive is the reusable H0 representation on eight-hour future
> price when compared with a recent specialized multivariate forecasting
> system trained end-to-end on its native full-path objective?

This is a complete-system task comparison. It is not a frozen-representation
comparison, decoder ablation, or parameter-matched causal test.

## 2. Phase ownership

- Phase 6.9 is the sole technical, implementation, artifact, and reporting
  authority for xLSTM-Mixer.
- Only one xLSTM-Mixer trajectory per walk is permitted.
- Canonical artifacts live under `experiments/phase6_9/xlstm_mixer/`.
- The old Phase 6.6B listing is a superseded historical reference and cannot
  launch or own a second run. Phase 6.6A raw fusion and Phase 6.6C decoder
  capacity remain deferred.

## 3. Shared inputs and endpoint

For each walk, reuse:

- the accepted `[N,64,5]` historical OHLCV contexts;
- the exact `absolute_price_h8` identities;
- contract, segment, decision-time, and target-time provenance;
- the existing H0-D0, Raw LSTM, persistence, and reversal definitions; and
- existing price-level and implied-movement metrics.

`XM-MV8` additionally requires observed `[N,8,5]` future paths. A deterministic
metadata join freezes the common intersection before training. It may inspect
identities, timestamps, segment continuity, observed/imputed flags, and
finiteness, but not target values or metrics. The read-only check already
shows that matched intersection reruns are required.

## 4. Candidate and strict controls

The primary candidate is:

| ID | Training target | Evaluation endpoint | Role |
|---|---|---|---|
| `XM-MV8` | all five channels at `t+1,...,t+8` | `close[t+8]` | recent specialized complete-system baseline |

Create strict matched controls on the frozen fully observed intersection:

| ID | Model | Required action |
|---|---|---|
| `XM-H0-D0` | canonical frozen H0 plus simple price probe | retrain on the intersected train rows and evaluate on intersected evaluation rows |
| `XM-RL` | established Raw LSTM architecture | retrain on those same identities |

Broader-row historical results may remain contextual but may not be presented
as strict head-to-head comparisons.

## 5. Frozen adaptation boundary

The architecture and optimizer choices must be approved in
`upstream_clarification_request.md`. The immutable scientific elements are:

```text
input context: [64,5]
channel order: open, high, low, close, volume
future target: [8,5]
headline output: horizon index 7, close index 3
loss family: source-aligned full-path L1 in accepted project units
walk-specific fits: yes
seed: 0
retained epochs: 5, 15, 50
primary epoch: 50
validation/early stopping: none
```

Timestamp embeddings, output clipping, OHLC repair, ordinary-LSTM
substitution, target-path imputation, and evaluation-selected configurations
are excluded.

## 6. Reporting matrix

Report each walk separately and pool only metrics whose existing evaluator
already defines a valid pooled aggregation. For `XM-MV8`, H0, Raw LSTM, and
references, report:

- price MAE, RMSE/MSE, Pearson, and Spearman;
- persistence-relative skill;
- implied-movement Pearson/Spearman and sign agreement;
- global and timestamp-level cross-sectional Rank IC;
- contract-macro, lifecycle/activity, staleness/imputation-exposure, price-
  level, and target-movement subgroups where already defined;
- prediction validity rates for `[0,1]`, OHLC ordering, and nonnegative volume;
- trainable/total parameters, epoch time, total time, peak memory, and
  prediction latency; and
- exact row counts, contracts, source/config/upstream-volume-transform/
  checkpoint hashes.

No second xLSTM-specific channel scaler is fitted: OHLC remains in `[0,1]`
and the existing walk-training-only volume transform is reused. The complete
`[N,8,5]` predictions remain replay artifacts. Auxiliary-channel
errors may be reported as diagnostics but may not be averaged into the
headline Phase 6.9 comparison.

## 7. Claim rules

Allowed claims are bounded to the executed walks and endpoint:

- H0 is or is not competitive with `XM-MV8` on named price metrics;
- the specialized model's additional supervision/runtime/parameter cost;
- consistency or inconsistency across the two walks; and
- whether one unified representation pays an observable task-specialization
  penalty on this task.

Do not claim:

- that xLSTM-Mixer is universally better or worse;
- that sLSTM causes a result relative to Raw LSTM;
- that `XM-MV8` is a direct representation-quality baseline;
- that extra full-path supervision is controlled away;
- profitable trading or fresh-holdout alpha; or
- independent confirmation from “Phase 6.6” and “Phase 6.9,” because Phase
  6.6B no longer owns an active experiment.

## 8. Admission and failure policy

The method is admitted to implementation with exactly one learned initial
token and the approved licence boundary. It is admitted to training only after
the common-path artifact and selected-runtime CUDA-device smoke. The
2026-10-09 owner-directed amendment permits local WSL vanilla sLSTM without
requiring Lumid or nvcc. Same-runtime
checkpoint/prediction replay is required; cross-backend CPU/CUDA numerical
parity is not.

If the pinned sLSTM cannot be built, its licences cannot be accommodated, or
its CUDA path cannot pass same-runtime integrity checks, record xLSTM-Mixer as infeasible under
the available environment. Ordinary LSTM, mLSTM, or a changed forecast target
cannot inherit the `XM-MV8` label without a dated amendment.
