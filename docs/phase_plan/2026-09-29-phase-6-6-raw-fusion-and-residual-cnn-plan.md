# Phase 6.6 Price-Focused Fusion and Decoder-Capacity Plan

**Date:** 2026-09-29
**Status:** Approved planning contract, amended 2026-09-29; implementation and
execution have not started. The former Phase 6.6B residual-CNN study now
belongs to price-focused Phase 6.5D. Phase 6.6A/C are restricted to eight-hour
future-price prediction in this first round. The existing filename is retained
so repository links remain stable.
**Predecessors:** `2026-09-26-phase-6-experiment-observation-and-outcomes.md`
and `2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`

## 1. Purpose and boundary

Phase 6.6 contains two related downstream-system studies motivated by the
possibility that the frozen representation discards useful ordering
information or that the simple downstream probe cannot expose useful
interactions:

- **Phase 6.6A -- raw-representation residual fusion:** combine the canonical
  frozen `H0` vector with a supervised LSTM or bidirectional-LSTM encoding of
  the exact raw 64-hour OHLCV context; and
- **Phase 6.6C -- decoder capacity:** test two richer static decoders on the
  immutable canonical `H0` branches while keeping future-price rows and
  targets fixed.

The studies answer different questions. Phase 6.6A asks whether frozen
representations and exact raw temporal ordering are complementary in one
supervised price model. Phase 6.6C asks whether useful price information is
already present in `H0` but underused by the simple probe. Phase 6.6C changes
only the supervised mapping and, in one named row, the fusion rule over
immutable canonical branches. Their results must not be combined into one
candidate during this phase.

Post-hoc SHAP-style attribution is deliberately placed after model execution
and comparison. It is not a training input, model-selection rule, or Phase
6.6 model-matrix gate. A later analysis amendment must freeze its background
population, evaluation subset, feature groups, masking rule, and estimator
before attribution values are computed.

Phase 6.6 does not alter completed Phase 6.5A--C. The residual-CNN SSL study
formerly specified here is now Phase 6.5D and remains a separate encoder
experiment with the simple probe. The Phase 6.6C architecture remains the
decoder design previously frozen here, but its active matrix is narrowed to
future price. Phase 6.6 also does not change the Phase 7A canonical branch-
ablation matrix.

## 2. Shared data and evaluation contract

Reuse the accepted Phase 5/6 one-hour global-calendar walks:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every fitted parameter remains walk-specific. Walk 2 history may not update,
select, or reinterpret a Walk 1 model. Both studies reuse:

- the exact saved 64-by-5 walk-scaled OHLCV contexts;
- the exact canonical epoch-50 `H0` branch arrays;
- the exact eight-hour future-price training and evaluation identities; and
- the existing price-level and implied-movement diagnostics and references.

No cohort, window, activity filter, label, target, reference, or row set may be
rebuilt for this phase. Feature and model fitting uses training rows only. The
test-only policy remains: no validation split, early stopping, restart
selection, or evaluation-driven architecture choice. Seed `0`, one 50-epoch
trajectory, 5/15/50 snapshots, and predeclared epoch 50 remain the initial
characterisation contract. Classification and realised-variance extensions
are deferred and may not be added after reading the price results without a
separate amendment.

## 3. Phase 6.6A -- raw-representation residual fusion

### 3.1 Research question

> Does the exact raw historical OHLCV ordering provide useful eight-hour
> future-price signal beyond the canonical frozen representation when
> supervised capacity and price rows are held fixed?

The input sources describe the same decision-time window in different forms:

- `H0` is a 445-dimensional concatenation of statistical, transformed, VAE,
  contrastive-CNN, and BYOL-CNN branches; and
- raw OHLCV is the saved 64-by-5 ordered sequence before representation
  compression.

The raw sequence may preserve recency, reversal, acceleration, and inactive-
period ordering that global pooling or window-level summaries discard.

### 3.2 Frozen towers

The representation tower is shared by every configuration that consumes
`H0`:

```text
H0 [445]
-> LayerNorm(445)
-> Linear(445, 128)
-> GELU
-> z [128]
```

The raw unidirectional tower is:

```text
OHLCV [64, 5]
-> two-layer LSTM(input=5, hidden=128, dropout=0.1)
-> final hidden state of layer 2
-> r [128]
```

The raw bidirectional tower is:

```text
OHLCV [64, 5]
-> two-layer bidirectional LSTM(input=5, hidden=64 per direction,
                               inter-layer dropout=0.1)
-> concatenate final forward/backward states
-> r [128]
```

Bidirectionality is permitted only inside the already observed 64-hour
context. No target-interval candle or later timestamp enters either direction.

### 3.3 Residual fusion and task head

For a fused model:

```text
h = LayerNorm(z + alpha * r)
```

`alpha` is one learned scalar initialized to `0.1`. It is stored and reported
at every snapshot. The nonzero initialization lets gradients reach the raw
tower immediately while making the representation path the initial dominant
path. Sample-dependent or coordinate-dependent gates are excluded from this
first bounded study.

Every configuration ends with the same task mapping:

```text
h [128] -> Linear(128, 64) -> GELU -> task output
```

The output is a sigmoid-bounded future probability trained with mean squared
error, matching the completed eight-hour future-price task.

### 3.4 Frozen comparison matrix

| ID | Inputs and model | Role |
|---|---|---|
| `F-H0` | representation tower only | matched supervised-capacity control |
| `F-RL` | raw unidirectional LSTM tower only | raw temporal reference |
| `F-H0-RL` | `H0` plus residual raw LSTM | primary unidirectional fusion candidate |
| `F-H0-RBL` | `H0` plus residual raw BiLSTM | primary bidirectional fusion candidate |

Run all four configurations on future price in both walks:

```text
4 configurations x 1 task x 2 walks = 8 downstream trajectories
```

The primary comparisons are:

```text
F-H0-RL  - F-H0
F-H0-RBL - F-H0
```

`F-RL` and the completed raw-LSTM baseline provide contextual raw-only
comparisons. `H0-D0` remains the primary simple representation probe. `F-H0`
is required because comparing a fused model only with `H0-D0` would confound
raw information with the new supervised projection.

To support a complementarity statement rather than only “raw information
helps H0,” also report each fused model against `F-RL`. A fused win over both
unimodal controls is complete-system evidence of complementary usefulness,
not parameter-matched causal attribution.

Fit the H0 coordinate scaler only on eligible price-training rows.
The raw tower consumes the exact saved walk-scaled sequences used by the
matched task rows and does not fit a second full-frame normalization.

## 4. Phase 6.6C -- canonical decoder-capacity sensitivity

### 4.1 Research question and interpretation

> With the canonical five frozen branches and all future-price data held
> fixed, does a predeclared richer static decoder improve price prediction
> relative to the simple Phase 6 probe?

The existing shallow task head remains the primary representation probe: its
low capacity makes it easier to attribute performance to the frozen features.
Phase 6.6C is a sensitivity asking whether those features contain useful
nonlinear or cross-branch interactions that the probe cannot expose. An
improvement supports a representation--decoder interaction or a stronger
complete system; it does not retroactively prove that the representation alone
improved.

### 4.2 Frozen decoder configurations

Use only canonical `H0` branch arrays and the completed task-specific
train-only coordinate scalers. The immutable reference is:

| ID | Decoder | Role |
|---|---|---|
| `D0` | completed Phase 6 shallow task head | Primary representation probe; no retraining required |

Two new static candidates are frozen:

**`D1-RP` -- residual projection head**

```text
concat H0 (445)
-> LayerNorm
-> Linear(445, 256) -> GELU -> Dropout(0.1)
-> two pre-norm residual MLP blocks at width 256
   [LayerNorm -> Linear(256, 256) -> GELU -> Dropout(0.1)
    -> Linear(256, 256) -> residual add]
-> Linear(256, 128) -> GELU
-> task output layer
```

This is a supervised projection decoder. It is separate from, and does not
reuse, either SSL method's pretraining projector.

**`D2-BG` -- branch-aware gated projection head**

```text
each canonical branch -> branch-specific Linear(input_dim, 128)
                      -> GELU -> LayerNorm
concatenated projected branches -> Linear(5*128, 128) -> GELU
                                -> Linear(128, 5) -> softmax gates
weighted sum of five projected branches
-> residual MLP [LayerNorm -> Linear(128, 256) -> GELU
                 -> Dropout(0.1) -> Linear(256, 128) -> residual add]
-> task output layer
```

`D2-BG` changes both supervised capacity and fusion, so it is a named
complete-system sensitivity rather than a decoder-only causal contrast.
Neither candidate consumes temporal sequences of embeddings. Historical
recurrent/attention decoder ideas remain outside Phase 6.6C because they would
change context construction and row eligibility.

### 4.3 Price training and matrix

Run both candidates on the established future-price task and both walks:

```text
2 new decoders x 1 task x 2 walks = 4 new downstream trajectories
```

Reuse the exact Phase 6 `absolute_price_h8` rows, MSE loss, sigmoid output,
current-price persistence and reversal references, batch size `512`, Adam
learning rate `1e-4`, seed `0`, and one uninterrupted 50-epoch trajectory with
snapshots at 5, 15, and 50. Retain price-level and implied-movement metrics.

Epoch 50 remains the principal result. No dropout, width, depth, optimizer,
loss, or output transform may be selected from evaluation performance.

### 4.4 Reporting and claim boundary

Compare `D1-RP - D0` as the primary decoder-capacity contrast for each walk.
Compare `D2-BG` separately with both `D0` and `D1-RP`. Report total
and trainable parameters, training and inference time, peak device memory,
checkpoint histories, prediction replay, and the unchanged price metrics and
breakdowns. The completed simple-head results remain the headline
representation-quality evidence even if a richer complete system performs
better.

## 5. Training, reporting, and claim boundary

Phase 6.6A trains its towers and task mapping end to end on supervised
price-training rows. Phase 6.6C trains only its static decoder on the unchanged
frozen H0 features. Both studies retain price-training-only scaling, batch
size `512`, Adam learning rate `1e-4`, weight decay `0`, and the completed
price metrics and references.

Report every configuration, walk, and snapshot. Primary conclusions use epoch
50 and remain separated by walk. Record parameter counts,
training time, inference time, peak device memory, loss histories, prediction
hashes, and replay results.

The studies may support only narrow conclusions:

- residual fusion improves or does not improve future price in the named walk relative
  to its matched `F-H0` control;
- a fused model does or does not improve on the raw-only `F-RL` system;
- bidirectional historical processing changes performance relative to the
  unidirectional fused model;
- a richer static decoder exposes or does not expose additional price signal
  relative to the immutable simple `D0` probe; and
- any effect is seed-0, two-walk characterisation.

They cannot establish causal feature importance, optimal fusion, an optimal
decoder, universal bidirectional superiority, cross-task superiority, robust
multi-seed superiority, or a profitable strategy.

## 6. Later attribution-analysis boundary

Attribution begins only after all Phase 6.6 predictions and comparisons are
frozen. The later amendment should prefer hierarchical groups over hundreds
of correlated coordinates:

- representation groups: statistical, transformed, VAE, contrastive, BYOL;
- raw-channel groups: open, high, low, close, volume; and
- raw-lag groups: 1--4, 5--16, 17--32, and 33--64 hours before the decision.

Grouped SHAP may describe the trained fusion system globally. GradientSHAP or
Integrated Gradients may provide detailed raw sequence attribution before
aggregation by channel and lag. Branch ablations and matched fusion controls,
not SHAP values, remain the performance-contribution evidence. Correlated
inputs can divide attribution credit arbitrarily, and attribution must not be
used to add, remove, or select a model after evaluation.

This later attribution analysis is not a Phase 6.6 completion condition until
its own estimator and sampling amendment is approved.

## 7. Artifacts and execution gates

Generated artifacts belong under a new immutable root:

```text
experiments/phase6_6/
  manifests/
  raw_representation_fusion/
    downstream/absolute_price_h8/
      walk{1,2}/{f_h0,f_rl,f_h0_rl,f_h0_rbl}/seed0/
    diagnostics/resources/
    reports/
  decoder_capacity/
    downstream/absolute_price_h8/walk{1,2}/{d1_rp,d2_bg}/seed0/
    diagnostics/resources/
    reports/
```

Reusable implementation belongs under `src/`; thin orchestration belongs in
the next script generation. Bootstrap entry points are manifest-only by
default and require an explicit execution flag.

Ordered gates are:

1. Replay the exact Phase 5/6 source datasets, H0 features, future-price rows,
   and named reference predictions.
2. Implement all four fusion configurations and CPU-test forward, loss,
   backward, directionality, output transforms, occupied paths, and train-only
   scaling.
3. Freeze the eight-entry fusion manifest before any fused-model evaluation.
4. Train, replay, and report the complete fusion matrix without selecting a
   walk winner.
5. Implement `D1-RP` and `D2-BG` without altering the completed `D0` model or
   any frozen Phase 6 feature store; add forward/backward, output-transform,
   occupied-path, and parameter-count tests.
6. Freeze the four-entry decoder-capacity manifest after CPU smoke tests verify
    exact price rows, targets, train-only scalers, loss, and references.
7. Train, replay, and report the complete Phase 6.6C matrix with resource
    tables and the predeclared `D1-RP - D0` and `D2-BG` comparisons.
8. Only afterward, write and approve the separate grouped-attribution
    amendment before computing SHAP-style results.

## 8. Completion conditions

Phase 6.6 is complete only when:

- all eight fusion trajectories pass checkpoint, prediction, target, identity,
  scaler, and metric replay;
- `F-H0` is used as the primary matched-capacity fusion control;
- fusion is also reported against `F-RL` so complementarity is not inferred
  only from improvement over representation-only input;
- all four Phase 6.6C trajectories pass replay on unchanged future-price rows,
  and decoder-capacity, fusion, and resource comparisons are reported without
  selecting an evaluation winner; and
- limitations are stated as single-seed, two-walk characterisation.

Grouped SHAP or other attribution is a later analysis deliverable and is not
required to declare the frozen Phase 6.6 model matrix complete.
