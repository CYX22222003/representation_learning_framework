# Phase 6.6 Raw-Representation Fusion and Residual-CNN Plan

**Date:** 2026-09-29
**Status:** Approved planning contract; implementation and execution have not
started
**Predecessors:** `2026-09-26-phase-6-experiment-observation-and-outcomes.md`
and `2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`

## 1. Purpose and boundary

Phase 6.6 adds two independent model studies motivated by the possibility that
the frozen representation discards useful ordering information and that the
canonical two-layer CNN is too shallow to benefit from residual learning:

- **Phase 6.6A -- raw-representation residual fusion:** combine the canonical
  frozen `H0` vector with a supervised LSTM or bidirectional-LSTM encoding of
  the exact raw 64-hour OHLCV context; and
- **Phase 6.6B -- residual CNN encoder:** replace one canonical SSL CNN at a
  time with a deeper one-dimensional residual CNN while holding the SSL
  objective and 128-dimensional output fixed.

The studies answer different questions. Phase 6.6A changes the supervised
downstream system while preserving all frozen `H0` encoders. Phase 6.6B
changes one unsupervised encoder backbone while retaining the simple
downstream probe. Their results must not be combined into one candidate during
this phase.

Post-hoc SHAP-style attribution is deliberately placed after model execution
and comparison. It is not a training input, model-selection rule, or Phase
6.6 model-matrix gate. A later analysis amendment must freeze its background
population, evaluation subset, feature groups, masking rule, and estimator
before attribution values are computed.

Phase 6.6 does not alter or reopen Phase 6.5A. In particular, the two-layer
unidirectional SSL LSTM remains the exact predeclared Phase 6.5A candidate.
Phase 6.6 also does not change the Phase 7A canonical branch-ablation matrix.

## 2. Shared data and evaluation contract

Reuse the accepted Phase 5/6 one-hour global-calendar walks:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every fitted parameter remains walk-specific. Walk 2 history may not update,
select, or reinterpret a Walk 1 model. Both studies reuse:

- the exact saved 64-by-5 walk-scaled OHLCV contexts;
- the exact canonical epoch-50 `H0` branch arrays where applicable;
- the exact task-specific training and evaluation identities;
- two-hour `DOWN/STABLE/UP` classification at `tau=0.001`;
- eight-hour future-price prediction and implied-movement diagnostics; and
- strict eight-hour future realised variance from observed raw probability
  changes.

No cohort, window, activity filter, label, target, reference, or row set may be
rebuilt for this phase. Feature and model fitting uses training rows only. The
test-only policy remains: no validation split, early stopping, restart
selection, or evaluation-driven architecture choice. Seed `0`, one 50-epoch
trajectory, 5/15/50 snapshots, and predeclared epoch 50 remain the initial
characterisation contract.

## 3. Phase 6.6A -- raw-representation residual fusion

### 3.1 Research question

> Does the exact raw historical OHLCV ordering provide useful task signal
> beyond the canonical frozen representation when supervised capacity and
> task rows are held fixed?

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

The task outputs and losses remain unchanged:

- classification: three logits with train-prior logit-adjusted cross-entropy;
- future price: sigmoid probability with mean squared error; and
- realised variance: Softplus output with Smooth L1 loss on `10000 * RV`.

### 3.4 Frozen comparison matrix

| ID | Inputs and model | Role |
|---|---|---|
| `F-H0` | representation tower only | matched supervised-capacity control |
| `F-RL` | raw unidirectional LSTM tower only | raw temporal reference |
| `F-H0-RL` | `H0` plus residual raw LSTM | primary unidirectional fusion candidate |
| `F-H0-RBL` | `H0` plus residual raw BiLSTM | primary bidirectional fusion candidate |

Run all four configurations on all three tasks and both walks:

```text
4 configurations x 3 tasks x 2 walks = 24 downstream trajectories
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

For each task, fit the H0 coordinate scaler only on eligible training rows.
The raw tower consumes the exact saved walk-scaled sequences used by the
matched task rows and does not fit a second full-frame normalization.

## 4. Phase 6.6B -- residual CNN SSL encoder

### 4.1 Research question

> When the SSL family, training identities, augmentations, output width, and
> simple downstream head are fixed, does a deeper residual temporal CNN
> produce more useful representations than the canonical two-convolution CNN?

The canonical CNN is only two convolutional layers deep. The residual
candidate therefore tests a deeper practical architecture; it does not claim
that the shallow reference currently suffers proven vanishing gradients.

### 4.2 Frozen residual backbone

The candidate consumes `[N,64,5]` and returns one 128-dimensional vector:

```text
transpose to [N,5,64]
-> Conv1d(5,128,kernel=5,padding=2)
-> three pre-activation residual blocks at width 128
   [GroupNorm(8,128) -> GELU -> Conv1d(128,128,kernel=3,padding=1)
    -> GroupNorm(8,128) -> GELU -> Dropout(0.1)
    -> Conv1d(128,128,kernel=3,padding=1)
    -> residual add]
-> AdaptiveAvgPool1d(1)
-> width-128 representation
```

There is no temporal downsampling, dilation, attention, channel-width search,
or stochastic-depth search. The skip is identity because every block retains
width and sequence length.

Train the backbone independently under both existing SSL families:

| SSL family | Branch name | Walk-specific trajectories |
|---|---|---:|
| Contrastive / NT-Xent | `contrastive_rescnn` | 2 |
| BYOL | `byol_rescnn` | 2 |

This produces four new encoder trajectories. Both candidates retain the Phase
6 SSL views, projectors/predictor, temperature `0.2`, target decay `0.99`,
AdamW learning rate `1e-3`, weight decay `1e-4`, batch size `256`, seed `0`,
and 5/15/50 checkpoints. Epoch 50 is the sole downstream feature source.

### 4.3 Feature configurations

| ID | Change from canonical `H0` | Width | Role |
|---|---|---:|---|
| `HC-SR` | contrastive CNN -> contrastive ResCNN | 445 | same-width substitution |
| `HB-SR` | BYOL CNN -> BYOL ResCNN | 445 | same-width substitution |
| `HC-AR` | add contrastive ResCNN to `H0` | 573 | heterogeneous addition |
| `HB-AR` | add BYOL ResCNN to `H0` | 573 | heterogeneous addition |

All four configurations run on all three tasks and both walks:

```text
4 configurations x 3 tasks x 2 walks = 24 downstream trajectories
```

Substitutions are compared with `H0`. Additions are compared with both `H0`
and the completed same-family duplicate-CNN controls `HC-DC` or `HB-DC`.
Consequently, an addition is not interpreted as new information merely
because it widens the task head.

Centered linear CKA compares each residual CNN with its same-family canonical
CNN on the predeclared first 4,096 encoder-training identities. CKA is
descriptive and cannot select a candidate or establish predictive usefulness.

## 5. Training, reporting, and claim boundary

Phase 6.6A trains its towers and task mapping end to end on supervised
training rows. Phase 6.6B freezes each epoch-50 SSL backbone before fitting
the unchanged simple task head. Both studies retain task-specific training-
only scaling, batch size `512`, Adam learning rate `1e-4`, weight decay `0`,
and the completed task metrics and references.

Report every configuration, task, walk, and snapshot. Primary conclusions use
epoch 50 and remain separated by task and walk. Record parameter counts,
training time, inference time, peak device memory, loss histories, prediction
hashes, and replay results.

The studies may support only narrow conclusions:

- residual fusion improves or does not improve the named task/walk relative
  to its matched `F-H0` control;
- bidirectional historical processing changes performance relative to the
  unidirectional fused model;
- the fixed residual CNN substitutes for or complements its same-family CNN
  under the named SSL objective; and
- any effect is seed-0, two-walk characterisation.

They cannot establish causal feature importance, optimal fusion, optimal CNN
depth, universal bidirectional superiority, robust multi-seed superiority, or
a profitable strategy.

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
    downstream/{classification_h2,absolute_price_h8,realised_variance}/
    diagnostics/resources/
    reports/
  residual_cnn/
    pretraining/walk{1,2}/{contrastive,byol}/rescnn/seed0/
    features/walk{1,2}/
    downstream/{classification_h2,absolute_price_h8,realised_variance}/
    diagnostics/{health,cka,resources}/
    reports/
```

Reusable implementation belongs under `src/`; thin orchestration belongs in
the next script generation. Bootstrap entry points are manifest-only by
default and require an explicit execution flag.

Ordered gates are:

1. Replay the exact Phase 5/6 source datasets, H0 features, task rows, and
   named reference predictions.
2. Implement all four fusion configurations and CPU-test forward, loss,
   backward, directionality, output transforms, occupied paths, and train-only
   scaling.
3. Freeze the 24-entry fusion manifest before any fused-model evaluation.
4. Train, replay, and report the complete fusion matrix without selecting a
   task or walk winner.
5. Implement the residual CNN once and wrap it independently in the existing
   Contrastive and BYOL semantics.
6. CPU-test residual identity shapes, gradients, BYOL EMA, collapse checks,
   parameter counts, provenance failure, and occupied paths; then freeze the
   four-encoder manifest.
7. Train and replay all four residual-CNN encoders.
8. Extract and validate all task/walk feature stores; freeze the 24-entry
   residual-CNN downstream manifest.
9. Train, replay, and report the complete substitution/addition matrix with
   CKA, resources, H0 references, and duplicate controls.
10. Only afterward, write and approve the separate grouped-attribution
    amendment before computing SHAP-style results.

## 8. Completion conditions

Phase 6.6 is complete only when:

- all 24 fusion trajectories pass checkpoint, prediction, target, identity,
  scaler, and metric replay;
- `F-H0` is used as the primary matched-capacity fusion control;
- all four residual-CNN SSL trajectories pass health and checkpoint replay;
- all six residual-CNN task/walk feature stores preserve exact H0 coordinates
  and task identities;
- all 24 residual-CNN downstream trajectories pass replay;
- substitution, addition, duplicate-control, CKA, and resource comparisons
  are reported without result-driven matrix changes; and
- limitations are stated as single-seed, two-walk characterisation.

Grouped SHAP or other attribution is a later analysis deliverable and is not
required to declare the frozen Phase 6.6 model matrix complete.
