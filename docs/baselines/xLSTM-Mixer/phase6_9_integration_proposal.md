# xLSTM-Mixer Phase 6.6B/6.9 integration proposal

## 1. Research role

`XM-MV8` answers:

> How competitive is the reusable H0 representation on eight-hour future
> price when compared with a recent specialized multivariate forecasting
> system trained end-to-end on its native full-path objective?

This is a complete-system task comparison. It is not a frozen-representation
comparison, decoder ablation, or parameter-matched causal test.

## 2. Phase ownership

- Phase 6.6B remains the technical, implementation, and artifact authority.
- Phase 6.9 activates that one experiment as the price row in a task-separated
  competitiveness report.
- Only one xLSTM-Mixer trajectory per walk is permitted.
- Phase 6.9 references the Phase 6.6B checkpoint/prediction hashes and does not
  copy or regenerate them.
- Phase 6.6A raw fusion and Phase 6.6C decoder capacity remain deferred.

## 3. Shared inputs and endpoint

For each walk, reuse:

- the accepted `[N,64,5]` historical OHLCV contexts;
- the exact `absolute_price_h8` identities;
- contract, segment, decision-time, and target-time provenance;
- the existing H0-D0, Raw LSTM, persistence, and reversal definitions; and
- existing price-level and implied-movement metrics.

`XM-MV8` additionally requires observed `[N,8,5]` future paths. A metadata-only
audit freezes either all existing rows or a common intersection before any
training. It may inspect identities, timestamps, segment continuity,
observed/imputed flags, and finiteness, but not target values or metrics.

## 4. Candidate and strict controls

The primary candidate is:

| ID | Training target | Evaluation endpoint | Role |
|---|---|---|---|
| `XM-MV8` | all five channels at `t+1,...,t+8` | `close[t+8]` | recent specialized complete-system baseline |

If every existing price row has a complete target path, compare with immutable
existing H0-D0 and Raw LSTM results on the same identities. If the intersection
shrinks, create strict matched controls:

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
loss family: source-aligned full-path L1 in a training-standardized domain
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
- exact row counts, contracts, source/config/scaler/checkpoint hashes.

The complete `[N,8,5]` predictions remain replay artifacts. Auxiliary-channel
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
- independent confirmation from “Phase 6.6” and “Phase 6.9,” because they are
  two labels for one artifact set.

## 8. Admission and failure policy

The method is admitted to implementation only after owner decisions, licence
handling, and focused CPU tests. It is admitted to training only after the
full-path audit and persistent-runtime CUDA smoke.

If the pinned sLSTM cannot be built, its licences cannot be accommodated, or
its CPU/CUDA paths cannot be validated, record xLSTM-Mixer as infeasible under
the available environment. Ordinary LSTM, mLSTM, or a changed forecast target
cannot inherit the `XM-MV8` label without a dated amendment.
