# Phase 6.5 LSTM Capacity and GARCH--LSTM Plan

**Date:** 2026-09-26
**Status:** Approved planning contract; implementation and execution have not
started
**Predecessor:** `2026-09-26-phase-6-experiment-observation-and-outcomes.md`

## 1. Purpose and scope

Phase 6.5 contains two separate follow-ups motivated by Phase 6:

- **Phase 6.5A -- LSTM encoder capacity:** test whether the deliberately
  compact one-layer temporal encoder limited the LSTM representation results;
  and
- **Phase 6.5B -- adapted GARCH--LSTM:** complete the deferred, task-specific
  hybrid benchmark on the strict eight-hour future-realised-variance task.

The two studies answer different questions and must be reported separately.
Phase 6.5A changes an unsupervised representation backbone while holding the
self-supervised objective and 128-dimensional output fixed. Phase 6.5B changes
the volatility forecasting system by adding a causal econometric branch and a
train-only stacking model. Neither study may be interpreted as selecting a
universal model from the completed Phase 6 evaluation results.

Both studies retain the two accepted Phase 5/6 one-hour global-calendar walks,
the 64-hour OHLCV context, walk-specific fitting, fixed epoch budgets, and the
train/test-only policy. There is no validation split, early stopping, restart
selection, or evaluation-driven hyperparameter choice.

Phase 6.5 does not include Transformer retuning, hidden-width search, gated
fusion, decoder search, branch ablation, alpha research, or a multi-seed
architecture claim.

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
`2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`. Neither task may
rebuild a cohort, window, label, activity filter, or comparator-specific row
set.

## 3. Phase 6.5A -- deeper LSTM encoder

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

## 5. Artifact and implementation boundary

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
```

Reusable code belongs under `src/`; thin new orchestration belongs under a new
script generation rather than altering completed Phase 6 artifacts in place.
Default bootstrap commands must be manifest-only and require an explicit
execution flag for training.

Every run must retain source/data/identity hashes, frozen configuration,
environment, checkpoints or fitted econometric state, complete histories,
predictions, metrics, completion markers, and independent CPU or deterministic
numerical replay as applicable.

## 6. Ordered gates

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

Phase 6.5A and 6.5B are scientifically independent. The order above is an
engineering order, not a condition that one result determines whether the
other runs.

## 7. Exit conditions and claim boundary

Phase 6.5 is complete only when:

- all four two-layer SSL encoder trajectories and all 24 downstream
  trajectories pass replay;
- deep-versus-shallow comparisons are reported for every family, role, task,
  and walk without selecting an observed winner;
- the strict GARCH and Raw-LSTM OOF rows are chronological, identical, and
  replayable;
- both GARCH--LSTM evaluation runs use the unchanged Phase 6 H=8 rows;
- all train-fitted scalers, econometric state, and meta-learner parameters are
  shown to exclude evaluation data; and
- limitations are stated as single-seed, two-walk characterisation.

This phase may support a narrow depth-capacity conclusion and a narrow
complete-system hybrid comparison. It cannot establish an optimal recurrent
architecture, universal LSTM superiority, standalone GARCH superiority, or a
profitable forecasting strategy.
