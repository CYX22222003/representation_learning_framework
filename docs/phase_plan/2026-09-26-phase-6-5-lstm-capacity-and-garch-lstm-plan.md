# Phase 6.5 Capacity and Task-Benchmark Plan

**Date:** 2026-09-26
**Scope amended:** 2026-09-29 to add the classification-only TA-MLP benchmark
and again to move the residual-CNN SSL encoder study from the former Phase
6.6B into a new price-focused Phase 6.5D. The canonical decoder-capacity
sensitivity remains in Phase 6.6C.
**Status:** Approved contract. Phase 6.5A is complete for its frozen seed-0,
two-walk scope: all four two-layer SSL encoder trajectories, six task/walk
feature stores, 24 downstream trajectories, 72 epoch snapshots, two linear-
CKA diagnostics, and the complete report are replay-validated or generated
after prerequisite replay. Phase 6.5B and Phase 6.5C execution is also
complete: both chronological GARCH--LSTM stacks, both causal TA stores, and
all ten strict classification trajectories pass standalone replay at epochs
5/15/50. The compact epoch-50 result summary is under
`experiments/phase6_5/reports/b_c_seed0/`; contract-macro and subgroup report
expansion remains the only Phase 6.5B/C reporting follow-up.
Phase 6.5D is complete for its frozen price-only seed-0 scope. All four
residual-CNN encoders, both feature stores, all eight price trajectories, all
5/15/50 snapshots, both CKA diagnostics, and the complete report pass replay.
The report is under
`experiments/phase6_5/residual_cnn/reports/complete_seed0/`.
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
- **Phase 6.5D -- residual CNN encoder capacity:** test a deeper residual
  one-dimensional CNN under both SSL families, using eight-hour future-price
  prediction as the sole downstream task in this first bounded study.

The four studies answer different questions and must be reported separately.
Phase 6.5A changes an unsupervised representation backbone while holding the
self-supervised objective and 128-dimensional output fixed. Phase 6.5B changes
the volatility forecasting system by adding a causal econometric branch and a
train-only stacking model. Phase 6.5C compares learned representations and raw
sequences with one fixed handcrafted-feature classifier under a matched
classification contract. Phase 6.5D changes an unsupervised CNN backbone while
holding the SSL family, 128-dimensional output, simple probe, and price rows
fixed. None may be interpreted as selecting a universal
model from the completed Phase 6 evaluation results. The separately frozen
decoder-capacity study now belongs to Phase 6.6C.

All four studies retain the two accepted Phase 5/6 one-hour global-calendar walks,
the 64-hour OHLCV context, walk-specific fitting, fixed epoch budgets, and the
train/test-only policy. There is no validation split, early stopping, restart
selection, or evaluation-driven hyperparameter choice.

Phase 6.5 does not include Transformer retuning, hidden-width search, temporal
decoder search, branch ablation, alpha research, or a multi-seed architecture
claim. The TA-MLP addition is classification-only. Phase 6.5D is price-only;
classification and realised-variance probes of the residual CNN are deferred
unless separately amended before their evaluation.

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
frozen before training. No study may rebuild a cohort, window, label,
activity filter, or outcome-dependent comparator row set.

Phase 6.5D also reuses the unchanged target-free encoder rows and the exact
completed `absolute_price_h8` training/evaluation identities. Its restricted
price scope is motivated by the existing evidence: classification provides
the clearest representation advantage, volatility remains metric- and walk-
dependent, and Raw LSTM plus persistence/reversal references still make the
incremental future-price edge unconvincing. This prioritisation is frozen
before any residual-CNN checkpoint or prediction is generated.

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

**Execution update (2026-09-29):** Both walk-specific stacks are trained and
replay-valid at epochs 5/15/50. The final chronological OOF populations contain
25,174 rows in Walk 1 and 44,874 in Walk 2. Contract rows lacking two causal
prices at a fold start are recorded as discarded warm-up rows. An evaluation
condition absent from training receives a deterministic median of guarded H=8
forecasts from contract-local training-only GARCH states; evaluation outcomes
never update that fallback. At epoch 50 the stack lowers MSE only marginally
against Raw LSTM, while worsening MAE and Spearman in both walks, so the result
does not support a broad hybrid win.

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

**Execution update (2026-09-29):** Both 36-feature stores and all ten
trajectories pass data, checkpoint, prediction, target, and identity replay at
epochs 5/15/50. TA-MLP/P2 leads the strict Walk 1 macro-F1 comparison
(`0.4810` versus H0 `0.4477`) but trails H0 in Walk 2 (`0.4564` versus
`0.4632`). P1U is marginally stronger than TA-P2 in Walk 1 and weaker in Walk
2. The evidence supports competitiveness on one walk, not universal
technical-indicator or sampling superiority.

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

## 6. Phase 6.5D -- price-focused residual CNN SSL encoder

### 6.1 Research question

> When the SSL family, training identities, augmentations, output width, and
> simple downstream price head are fixed, does a deeper residual temporal CNN
> produce more useful eight-hour future-price representations than the
> canonical two-convolution CNN?

**Execution update (2026-09-29):** All Phase 6.5D gates are complete. The
Contrastive substitution and addition reduce price MAE/RMSE relative to H0 in
both walks; the addition also beats the same-width `HC-DC` control in both
walks, so its error gain is not explained by width alone. The BYOL variants do
not improve error consistently. Raw LSTM and current-price persistence remain
stronger error references, while the residual variants do not improve
movement Rank IC over H0 overall. This is a narrow price-level reconstruction
result, not evidence of profitable trading or cross-task superiority.

The canonical CNN is only two convolutional layers deep. This bounded study
tests one deeper practical architecture; it does not claim that the canonical
reference suffers proven vanishing gradients or identify an optimal CNN.

### 6.2 Frozen residual backbone and SSL recipe

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
and one uninterrupted 50-epoch trajectory with 5/15/50 checkpoints. Epoch 50
is the sole downstream feature source.

### 6.3 Frozen feature configurations and price matrix

| ID | Change from canonical `H0` | Width | Role |
|---|---|---:|---|
| `HC-SR` | contrastive CNN -> contrastive ResCNN | 445 | same-width substitution |
| `HB-SR` | BYOL CNN -> BYOL ResCNN | 445 | same-width substitution |
| `HC-AR` | add contrastive ResCNN to `H0` | 573 | heterogeneous addition |
| `HB-AR` | add BYOL ResCNN to `H0` | 573 | heterogeneous addition |

All four configurations run only on the established eight-hour future-price
task in both walks:

```text
4 configurations x 1 task x 2 walks = 8 downstream trajectories
```

Substitutions are compared with immutable `H0`. Additions are compared with
both `H0` and the completed same-family duplicate-CNN controls `HC-DC` or
`HB-DC`, so a wider head is not mistaken for evidence of new information.
The completed Raw MLP and Raw LSTM are contextual learned baselines;
current-price persistence and last-hour reversal remain required references.

Reuse the exact `absolute_price_h8` rows, sigmoid output, MSE training loss,
train-only coordinate scaling, Adam learning rate `1e-4`, batch size `512`,
seed `0`, and 5/15/50 snapshots. Report price MAE/RMSE/MSE and Pearson, then
implied-movement Pearson/Spearman, sign agreement, timestamp-level
cross-sectional Rank IC/ICIR, contract/lifecycle breakdowns, and the unchanged
references. Epoch 50 remains the predeclared principal result.

### 6.4 Diagnostics and claim boundary

Centered linear CKA compares each residual CNN with its same-family canonical
CNN on the predeclared first 4,096 encoder-training identities. Report SSL
loss, embedding health, total/trainable parameters, training and inference
time, and peak device memory. CKA is descriptive and cannot select a model.

A positive result supports only a named family/role/walk statement that this
fixed residual backbone improved the future-price probe at seed 0. It does
not establish task-general representation superiority, profitable trading,
or an optimal residual depth. Classification and realised-variance evaluation
remain outside Phase 6.5D unless frozen by a later amendment.

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
  residual_cnn/
    pretraining/walk{1,2}/{contrastive,byol}/rescnn/seed0/
    features/walk{1,2}/absolute_price_h8/
    downstream/absolute_price_h8/walk{1,2}/
    diagnostics/{health,cka,resources}/
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
13. Implement the residual CNN once and wrap it independently in the existing
    Contrastive and BYOL semantics; add CPU shape, gradient, residual-path,
    BYOL EMA, collapse, provenance, and parameter-count tests.
14. Freeze the four-encoder Phase 6.5D manifest before training or evaluation.
15. Train and replay all four residual-CNN encoder trajectories, then extract
    and validate the two walk-specific future-price feature stores.
16. Freeze the eight-entry future-price downstream manifest after replaying
    the exact rows, targets, H0 references, duplicate controls, and train-only
    scalers.
17. Train, replay, and report all eight price trajectories with CKA and
    resource diagnostics.

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
- all four residual-CNN SSL trajectories pass health and checkpoint replay;
- both future-price residual-CNN feature stores preserve exact canonical
  coordinates and task identities, and all eight Phase 6.5D downstream
  trajectories pass replay;
- substitution, addition, duplicate-control, price-ranking, CKA, and resource
  comparisons are reported without adding classification or volatility after
  reading price results; and
- limitations are stated as single-seed, two-walk characterisation.

This phase may support a narrow depth-capacity conclusion, a narrow
complete-system hybrid comparison, and a narrow statement about the adapted
TA-MLP on the named movement-classification task. It cannot establish an
optimal recurrent architecture, universal LSTM
superiority, standalone GARCH
superiority, universal technical-indicator superiority,
reproduction of the TA-MLP paper's original claim, or a profitable forecasting
strategy. Phase 6.5D may additionally support only a narrow future-price
statement about the frozen residual-CNN candidate; it cannot establish
cross-task superiority.
