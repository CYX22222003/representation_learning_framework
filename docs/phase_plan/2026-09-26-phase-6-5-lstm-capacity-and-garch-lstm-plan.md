# Phase 6.5 Capacity and Task-Benchmark Plan

**Date:** 2026-09-26
**Scope amended:** 2026-09-29 to add the classification-only TA-MLP benchmark
and the canonical decoder-capacity sensitivity
**Status:** Approved contract. Phase 6.5A is complete for its frozen seed-0,
two-walk scope: all four two-layer SSL encoder trajectories, six task/walk
feature stores, 24 downstream trajectories, 72 epoch snapshots, two linear-
CKA diagnostics, and the complete report are replay-validated or generated
after prerequisite replay. Phase 6.5B and Phase
6.5C model/data/training infrastructure is implemented and CPU-tested under
`src/baselines/` and `scripts_v5/`, but their canonical manifests, generated
TA stores, training trajectories, predictions, and reports do not yet exist.
Phase 6.5D has not started.
**Predecessor:** `2026-09-26-phase-6-experiment-observation-and-outcomes.md`

## 1. Purpose and scope

Phase 6.5 contains four separate follow-ups motivated by Phase 6 and the
subsequent baseline review:

- **Phase 6.5A -- LSTM encoder capacity:** test whether the deliberately
  compact one-layer temporal encoder limited the LSTM representation results;
  and
- **Phase 6.5B -- adapted GARCH--LSTM:** complete the deferred, task-specific
  hybrid benchmark on the strict eight-hour future-realised-variance task; and
- **Phase 6.5C -- adapted TA-MLP:** restore the paper-inspired handcrafted-
  indicator MLP as a strict classification-only benchmark on the current
  two-hour probability-movement task; and
- **Phase 6.5D -- decoder capacity:** test whether the intentionally simple
  downstream probe masks useful information in the frozen canonical
  representation.

The four studies answer different questions and must be reported separately.
Phase 6.5A changes an unsupervised representation backbone while holding the
self-supervised objective and 128-dimensional output fixed. Phase 6.5B changes
the volatility forecasting system by adding a causal econometric branch and a
train-only stacking model. Phase 6.5C compares learned representations and raw
sequences with one fixed handcrafted-feature classifier under a matched
classification contract. None may be interpreted as selecting a universal
model from the completed Phase 6 evaluation results. Phase 6.5D changes only
the supervised mapping and, in one named row, the fusion rule over immutable
canonical branches.

All four studies retain the two accepted Phase 5/6 one-hour global-calendar walks,
the 64-hour OHLCV context, walk-specific fitting, fixed epoch budgets, and the
train/test-only policy. There is no validation split, early stopping, restart
selection, or evaluation-driven hyperparameter choice.

Phase 6.5 does not include Transformer retuning, hidden-width search, temporal
decoder search, branch ablation, alpha research, or a multi-seed architecture
claim. The TA-MLP addition is classification-only. Decoder capacity is limited
to the two predeclared static candidates below rather than an open-ended head
search.

## 2. Shared data and evaluation boundary

The walk schedule remains immutable:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every fitted object is independent per walk. A Walk 1 checkpoint, feature
scaler, GARCH state, OOF prediction, or meta-learner may not use Walk 2 history.
Evaluation targets may be read only after the complete matrix and recipes have
been frozen and the implementation has passed its replay checks.

Phase 6.5A reuses the exact target-free `encoder_train_sequences` used by the
completed Phase 6 encoders. Phase 6.5B reuses the exact H=8 volatility rows and
target arrays frozen by
`2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`. Phase 6.5C starts
from the exact completed Phase 6/Phase 5 two-hour classification rows and
labels at `tau=0.001`, then applies only a feature-availability intersection
frozen before training. Phase 6.5D reuses the exact canonical `H0` features,
train-only feature scalers, task rows, targets, and references for all three
completed Phase 6 tasks. No study may rebuild a cohort, window, label,
activity filter, or outcome-dependent comparator row set.

## 3. Phase 6.5A -- deeper LSTM encoder

**Execution update (2026-09-29):** Gates 1--6 are complete. The four encoder
trajectories and all 5/15/50 checkpoints replay; all six aligned feature
stores validate; the frozen 24-entry downstream matrix and all 72 snapshots
pass checkpoint, prediction, target, identity, scaler, and metric replay; and
both walk-specific linear-CKA diagnostics validate. The generated report is
under `experiments/phase6_5/lstm_capacity/reports/complete_seed0/`. This is
single-seed capacity characterisation and does not identify an optimal LSTM
depth.

### 3.1 Research question

> When the SSL family, training data, augmentations, downstream output width,
> and task heads are held fixed, does the predeclared two-layer LSTM candidate
> improve representation or downstream usefulness over the one-layer model?

This is a targeted depth/capacity test. It is not a general LSTM architecture
search.

### 3.2 Frozen architecture change

The completed Phase 6 reference is:

```text
LSTM(input=5, hidden=128, layers=1, dropout=0.0)
-> final hidden state, width 128
```

The single Phase 6.5A candidate is:

```text
LSTM(input=5, hidden=128, layers=2, inter-layer dropout=0.1)
-> final hidden state of layer 2, width 128
```

The hidden and downstream widths remain 128. Increasing depth rather than
hidden width preserves compatibility with the completed feature and head
contracts and avoids confounding capacity with a wider representation. The
candidate remains unidirectional and uses no attention, pooling, residual
connection, bidirectionality, or projection layer.

The candidate changes both recurrent depth and the inter-layer dropout that
becomes available only in a stacked LSTM. It therefore tests this practical
higher-capacity recipe, not a pure causal effect of layer count in isolation.

The candidate is trained independently under both existing SSL families:

| SSL family | New branch name | Walk-specific trajectories |
|---|---|---:|
| Contrastive / NT-Xent | `contrastive_lstm2` | 2 |
| BYOL | `byol_lstm2` | 2 |

This creates four new encoder trajectories. The completed one-layer LSTMs and
CNNs are immutable references and are not retrained.

### 3.3 Frozen SSL recipe

Apart from `lstm_num_layers=2` and `lstm_dropout=0.1`, Phase 6.5A retains the
Phase 6 temporal-encoder recipe exactly:

- the same walk-specific target-free training identities and sequence bytes;
- the same scaling, jitter, and time-mask views;
- the same contrastive projector and NT-Xent temperature `0.2`;
- the same BYOL online/EMA-target structure, projector, predictor, loss, and
  target decay `0.99`;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `256`, shuffled rows, and dropped incomplete batch;
- one uninterrupted 50-epoch trajectory with snapshots at 5, 15, and 50;
- epoch 50 fixed in advance as the downstream feature source; and
- seed `0` only.

Additional seeds are not required in this round. Consequently, results are
capacity characterisation at seed 0 and cannot establish robust superiority
across initializations.

### 3.4 Frozen feature configurations

The new two-layer branches are tested in the same substitution and addition
roles as their completed one-layer references:

| ID | Configuration change from canonical `H0` | Width |
|---|---|---:|
| `HC-SL2` | contrastive CNN -> `contrastive_lstm2` | 445 |
| `HB-SL2` | BYOL CNN -> `byol_lstm2` | 445 |
| `HC-AL2` | add `contrastive_lstm2` to `H0` | 573 |
| `HB-AL2` | add `byol_lstm2` to `H0` | 573 |

The primary capacity comparisons are:

```text
HC-SL2 - HC-SL
HB-SL2 - HB-SL
HC-AL2 - HC-AL
HB-AL2 - HB-AL
```

The corresponding comparisons with `H0` remain necessary for complete-system
context. Addition configurations must also be shown beside the existing
same-family duplicate-CNN controls `HC-DC` and `HB-DC`. The deep and shallow
LSTM outputs have the same width, so their paired comparison does not change
downstream input width.

No observed Phase 6 winner is promoted into this matrix. Both SSL families and
both representation roles are precommitted.

### 3.5 Downstream matrix and training

All four new configurations run on all three established tasks and both walks:

```text
4 configurations x 3 tasks x 2 walks = 24 new downstream trajectories
```

The tasks remain:

1. two-hour `DOWN/STABLE/UP` movement classification at `tau=0.001`;
2. eight-hour future-price prediction with implied-movement diagnostics; and
3. eight-hour future realised-variance prediction.

Each task reuses its exact completed Phase 6 row identities, targets, head
family, loss, references, batch size, learning rate, 5/15/50 snapshots, and
epoch-50 principal result. Each configuration receives its own task-training-
only coordinate scaler, frozen before evaluation.

### 3.6 Diagnostics and interpretation

Report trainable and total parameters, training time, inference time, peak
device memory, SSL loss, and embedding-health diagnostics beside the one-layer
reference.

Centered linear CKA is retained as a descriptive diagnostic on the same fixed
first 4,096 walk-specific encoder-training identities. For each SSL family,
compare the two-layer LSTM with:

- its completed one-layer LSTM; and
- its immutable CNN reference.

CKA does not select a checkpoint or establish predictive value. The capacity
question is answered by the predeclared paired downstream comparisons.

A positive result supports only the statement that this specific two-layer,
dropout-regularized candidate was more useful than the one-layer candidate for the named
family/task/walk at seed 0. A negative result is evidence that adding this
depth did not resolve the Phase 6 limitation under the frozen recipe. Neither
outcome identifies an optimal LSTM depth.

## 4. Phase 6.5B -- strict adapted GARCH--LSTM benchmark

### 4.1 Research question

> Does a causal, task-specific combination of GARCH conditional-variance
> forecasts and Raw-LSTM realised-variance forecasts improve the strict H=8
> volatility comparison on the identical Phase 6 rows?

This is a complete-system hybrid benchmark. It is not an ablation of the
representation framework and it does not test standalone GARCH superiority.

### 4.2 Scientific target and GARCH branch

The target remains raw future realised variance:

```text
RV(t,8h) = sum(j=1..8) (p[t+j] - p[t+j-1])^2
```

The econometric branch models raw hourly probability changes, not percentage
or log returns. For a decision at `t`, it recursively forecasts eight
one-hour conditional variances and sums them into one H=8 forecast. All
location, scale, GARCH parameters, numerical caps, convergence rules, and
fallbacks are fitted from causally permitted training history only and are
stored in the artifact manifest.

Evaluation-period outcomes do not update the frozen GARCH state in this
experiment. This deliberately matches the frozen walk model rather than
introducing a different online-update policy for one comparator.

### 4.3 Chronological OOF stacking contract

The meta-learner must never train on in-sample base-model predictions.
Training meta-features are constructed with five fixed, contract-aware,
expanding chronological folds inside each walk's training interval:

- every OOF row is predicted by base models fitted only on earlier permitted
  training history;
- no fold may randomly shuffle time or use a later row to predict an earlier
  row;
- fold membership, discarded warm-up rows, and final OOF row identities are
  frozen before model fitting; and
- the GARCH and Raw-LSTM OOF predictions must share the exact same ordered
  identities and H=8 targets.

The Raw-LSTM OOF models retain the Phase 6 raw-volatility architecture and
optimization recipe. Seed `0`, fixed epochs, and no validation/early stopping
are used in every fold. The completed full-training Raw-LSTM checkpoints and
evaluation predictions may be reused only after their data, architecture,
checkpoint, identity, target, and prediction hashes replay exactly.

For each epoch snapshot `e in {5,15,50}`, construct:

```text
X_meta_e = [garch_forecast, raw_lstm_forecast_e,
            garch_forecast * raw_lstm_forecast_e]
```

Fit a train-OOF-only `RobustScaler`, followed by ElasticNet with:

- `alpha=1e-4`;
- `l1_ratio=0.5`;
- cyclic coordinate updates; and
- at most 10,000 iterations.

The final prediction is clipped at zero. Report the number and fraction of
clipped predictions. Epoch 50 remains the principal result fixed before
evaluation; epochs 5 and 15 are reported snapshots, not selection candidates.

### 4.4 Evaluation and comparisons

Run one independent stack per walk. Use the exact H=8 evaluation rows and raw
realised-variance metrics from Phase 6:

- MAE, RMSE, and MSE;
- Pearson and Spearman correlation;
- row-weighted and contract-macro results;
- zero/positive target, starting-price, lifecycle, activity, imputation, and
  tail-concentration breakdowns; and
- the existing zero, training-median, and historical-volatility-persistence
  references.

Place the stack beside the completed Raw LSTM, Raw-OHLCV MLP, canonical `H0`,
and temporal-configuration results. Report the saved ElasticNet coefficients
and performance of the complete stack, but do not interpret a coefficient as
proof that one base branch is independently superior.

Because the target intervals overlap, row-wise IID significance claims are
not valid. Any uncertainty analysis requires a separately frozen contract/
calendar-block procedure.

## 5. Phase 6.5C -- strict adapted TA-MLP classification benchmark

### 5.1 Research question

> On the current two-hour `DOWN/STABLE/UP` probability-movement task, does the
> fixed paper-inspired TA-MLP provide a competitive complete-system
> classification benchmark against the canonical representation framework and
> raw-sequence models on identical eligible rows?

This restores TA-MLP as an external handcrafted-feature benchmark. It does not
restore the historical stock-style `BUY/HOLD/SELL` target, and it is not a
source-faithful reproduction of the paper's random split or test-driven model
selection. The adaptation preserves the paper-inspired feature family and MLP
architecture while enforcing this project's walk-forward and locked-
evaluation rules.

### 5.2 Frozen target and causal TA-feature rows

Reuse the exact walk-specific Phase 6 classification target:

```text
horizon = 2 hours
tau = 0.001 probability points
classes = DOWN / STABLE / UP
```

The 36 existing TA features remain the candidate input: oscillators, moving-
average ratios, calendar features, and 23 candlestick-pattern indicators. They
must be evaluated at each saved decision timestamp from only the same contract
and causally available history. Rolling indicators reset at a longer-gap
segment boundary. Evaluation candles may never affect a training feature.

The legacy feature helper cannot be reused blindly because its `zsVol`
calculation standardizes volume over the complete supplied frame. For each
walk, volume location/scale and every later coordinate scaler are fitted on
permitted training history only and then frozen for evaluation. Rolling
warm-up, including the 100-hour moving average, may make early task rows
unavailable. Freeze one train/evaluation intersection from TA-feature
availability alone, save every included and excluded identity with a reason,
and apply that exact ordered intersection to every strict comparator. No row
may be removed because of its label or any model result.

### 5.3 Frozen model, protocols, and matched matrix

The adapted TA-MLP remains:

```text
36 -> 128 -> 64 -> 32 -> 3
LeakyReLU(0.01) after each hidden layer
```

Use Adam, learning rate `1e-3`, batch size `64`, seed `0`, and one uninterrupted
50-epoch trajectory with snapshots at 5, 15, and 50. Epoch 50 is the
predeclared principal result. Fit the 36-coordinate standardizer on the exact
walk-specific training intersection only.

Two TA-MLP protocols are frozen:

| ID | Training rule | Role |
|---|---|---|
| `TA-P2` | Natural training rows plus train-prior logit-adjusted cross-entropy, `lambda=1.0` | Primary protocol matched to the completed Phase 5/6 classifiers |
| `TA-P1U` | Retain every minority row and randomly undersample only the majority `STABLE` class on training rows; ordinary cross-entropy | Source-paper-derived imbalance sensitivity |

The evaluation distribution is never resampled. `TA-P1U` stores the original
counts, selected source identities, sampler seed, and draw counts. It is a
training-protocol sensitivity, not the primary architecture comparison.
Natural cross-entropy and balanced oversampling are outside this bounded
follow-up.

The strict `P2` matrix retrains four models on the identical TA-eligible rows
for both walks:

| ID | Model/input | Walk-specific trajectories |
|---|---|---:|
| `C-H0-P2` | canonical 445-dimensional five-branch framework | 2 |
| `C-RM-P2` | flattened Raw-OHLCV MLP | 2 |
| `C-RL-P2` | three-layer Raw-OHLCV LSTM | 2 |
| `C-TA-P2` | 36-feature TA-MLP | 2 |

The two `C-TA-P1U` sensitivity runs bring Phase 6.5C to ten new
classification trajectories. H0, Raw MLP, and Raw LSTM retain their existing
architectures and optimization recipes; only their training/evaluation row
set changes to the frozen common intersection. Completed full-row Phase 6 and
Phase 6.5A classification results remain contextual and are not silently
relabeled as members of this strict subset matrix.

### 5.4 Evaluation and claim boundary

Save logits, softmax scores, predictions, targets, contract IDs, timestamps,
and source-row identities. Report macro-F1 as the primary metric, followed by
balanced accuracy, per-class precision/recall/F1, accuracy, predicted-class
counts, confusion matrices, and macro/per-class one-vs-rest ROC-AUC and
average precision. Include the always-`STABLE` and repeated train-prior
references and retain lifecycle, activity, imputation, and per-contract
breakdowns used by the completed classification reports.

The primary claim is the `C-TA-P2` comparison with the three `P2` models on
identical rows. The `C-TA-P1U` comparison answers only whether the paper-
derived sampling choice changes this adapted TA-MLP. Legacy TA-MLP results on
four-hour data or `BUY/HOLD/SELL` labels are historical characterisation and
cannot establish performance on the current task.

## 6. Phase 6.5D -- canonical decoder-capacity sensitivity

### 6.1 Research question and interpretation

> With the canonical five frozen branches and all task data held fixed, does a
> predeclared richer static decoder improve downstream performance relative to
> the simple Phase 6 probe?

The existing shallow task head remains the primary representation probe: its
low capacity makes it easier to attribute performance to the frozen features.
Phase 6.5D is a sensitivity asking whether those features contain useful
nonlinear or cross-branch interactions that the probe cannot expose. An
improvement supports a representation--decoder interaction or a stronger
complete system; it does not retroactively prove that the representation alone
improved.

### 6.2 Frozen decoder configurations

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
recurrent/attention D3--D4 decoder ideas remain outside Phase 6.5 because they
would change context construction and row eligibility.

### 6.3 Tasks, training, and matrix

Run both candidates on all three established tasks and both walks:

```text
2 new decoders x 3 tasks x 2 walks = 12 new downstream trajectories
```

Reuse each task's exact Phase 6 loss, output transform, references, batch size
`512`, Adam learning rate `1e-4`, seed `0`, and one uninterrupted 50-epoch
trajectory with snapshots at 5, 15, and 50. In particular:

- classification emits three logits and retains train-prior logit-adjusted
  cross-entropy;
- future price retains its sigmoid output and implied-movement diagnostics;
  and
- realised variance retains the fixed `10000 * RV` unit, Smooth L1 loss, and
  Softplus output.

Epoch 50 remains the principal result. No dropout, width, depth, optimizer,
loss, or output transform may be selected from evaluation performance.

### 6.4 Reporting and claim boundary

Compare `D1-RP - D0` as the primary decoder-capacity contrast for every
task/walk. Compare `D2-BG` separately with both `D0` and `D1-RP`. Report total
and trainable parameters, training and inference time, peak device memory,
checkpoint histories, prediction replay, and the unchanged task metrics and
breakdowns. The completed simple-head results remain the headline
representation-quality evidence even if a richer complete system performs
better.

## 7. Artifact and implementation boundary

New generated artifacts belong under a separate root so Phase 6 evidence is
immutable:

```text
experiments/phase6_5/
  manifests/
  lstm_capacity/
    pretraining/walk{1,2}/{contrastive,byol}/lstm2/seed0/
    features/walk{1,2}/
    downstream/{classification_h2,absolute_price_h8,realised_variance}/
    diagnostics/{health,cka,resources}/
    reports/
  garch_lstm/
    walk{1,2}/
    oof/
    reports/
  ta_mlp/
    data_preparation/walk{1,2}/
    manifests/
    downstream/{h0,raw_mlp,raw_lstm,ta_mlp}/walk{1,2}/
    reports/
  decoder_capacity/
    manifests/
    downstream/{classification_h2,absolute_price_h8,realised_variance}/
      walk{1,2}/{d1_rp,d2_bg}/seed0/
    diagnostics/resources/
    reports/
```

Reusable code belongs under `src/`; thin new orchestration belongs under a new
script generation rather than altering completed Phase 6 artifacts in place.
Default bootstrap commands must be manifest-only and require an explicit
execution flag for training.

Every run must retain source/data/identity hashes, frozen configuration,
environment, checkpoints or fitted econometric state, complete histories,
predictions, metrics, completion markers, and independent CPU or deterministic
numerical replay as applicable.

## 8. Ordered gates

1. Replay the completed Phase 6 source bundles, encoders, task rows, and
   reference predictions.
2. Implement the two-layer LSTM configuration without changing the one-layer
   default used by Phase 6.
3. Add CPU forward/backward, BYOL EMA, collapse, provenance, and occupied-path
   tests; freeze the four-encoder manifest.
4. Train and replay all four deep-LSTM encoders.
5. Build and validate the four new feature configurations for every task and
   walk; then freeze the 24-entry downstream manifest.
6. Train, replay, and report the complete Phase 6.5A matrix and linear-CKA
   diagnostics.
7. Audit the existing GARCH--LSTM implementation against the strict raw-change
   H=8 and walk-local causality contract.
8. Freeze and replay the five expanding OOF folds before fitting any
   meta-learner.
9. Fit, replay, and report both walk-specific GARCH--LSTM stacks at all three
   declared snapshots.
10. Build, hash, and replay the causal 36-feature TA stores and feature-
    availability-only common row intersections for both walks.
11. Freeze the ten-entry Phase 6.5C matrix after CPU smoke tests verify the
    matched `P2` loss, training-only `P1U` sampling, identities, and metrics.
12. Train, replay, and report all strict TA-eligible classification runs at
    the three declared snapshots.
13. Implement `D1-RP` and `D2-BG` without altering the completed `D0` model or
    any frozen Phase 6 feature store; add forward/backward, output-transform,
    occupied-path, and parameter-count tests.
14. Freeze the 12-entry decoder-capacity manifest after CPU smoke tests verify
    exact task rows, targets, train-only scalers, losses, and references.
15. Train, replay, and report the full Phase 6.5D matrix with resource tables
    and the predeclared `D1-RP - D0` and `D2-BG` comparisons.

Phase 6.5A, 6.5B, 6.5C, and 6.5D are scientifically independent. The order
above is an engineering order, not a condition that one result determines
whether the other studies run.

## 9. Exit conditions and claim boundary

Phase 6.5 is complete only when:

- all four two-layer SSL encoder trajectories and all 24 downstream
  trajectories pass replay;
- deep-versus-shallow comparisons are reported for every family, role, task,
  and walk without selecting an observed winner;
- the strict GARCH and Raw-LSTM OOF rows are chronological, identical, and
  replayable;
- both GARCH--LSTM evaluation runs use the unchanged Phase 6 H=8 rows;
- all train-fitted scalers, econometric state, and meta-learner parameters are
  shown to exclude evaluation data;
- both TA stores and common classification intersections replay from causal
  feature availability without outcome-dependent filtering;
- all ten Phase 6.5C classification trajectories pass checkpoint, prediction,
  target, and identity replay, and the primary TA-MLP comparison uses the
  matched `P2` rows and loss;
- all 12 Phase 6.5D trajectories pass replay on unchanged Phase 6 task rows,
  and decoder-capacity, fusion, and resource comparisons are reported without
  selecting an evaluation winner; and
- limitations are stated as single-seed, two-walk characterisation.

This phase may support a narrow depth-capacity conclusion and a narrow
complete-system hybrid comparison, a narrow statement about the adapted
TA-MLP on the named movement-classification task, and a bounded conclusion
about whether the two richer static heads expose additional task signal. It
cannot establish an optimal recurrent architecture, universal LSTM
superiority, standalone GARCH
superiority, universal technical-indicator superiority, an optimal decoder,
reproduction of the TA-MLP paper's original claim, or a profitable forecasting
strategy.
