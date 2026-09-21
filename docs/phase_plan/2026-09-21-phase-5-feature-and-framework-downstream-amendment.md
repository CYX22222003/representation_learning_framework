# Phase 5 Feature Extraction and Framework Downstream Amendment

**Date:** 2026-09-21
**Status:** Implemented, executed at seed 0, and replay-validated
**Parent authority:** `2026-09-21-phase-5-experiment-plan.md`

## 1. Scope

This stage tests the downstream probing ability of the canonical five-branch
framework only. It extracts frozen epoch-50 features for both walks and trains
the framework regression and classification heads at seed 0. Learned Raw-
OHLCV MLP, raw LSTM, adapted TA-MLP, additional seeds, and strict cross-model
comparisons are deferred to the later baseline stage.

The mandatory non-trained references remain in scope because they are task
sanity checks rather than learned baselines:

- regression: predict exact zero probability movement; and
- classification: always `STABLE` and repeated training-prior probabilities.

## 2. Feature extraction

For each walk, feature extraction consumes that walk's saved supervised
`train_sequences` and `test_sequences` in their existing order. Every branch
must preserve the exact saved decision identities.

| Branch | Source | Dimension |
|---|---|---:|
| Statistical | window-local AR/GARCH | 70 |
| Transformed | window-local FFT/Haar | 55 |
| VAE | walk-specific frozen epoch-50 latent mean | 64 |
| Contrastive | walk-specific frozen epoch-50 CNN backbone | 128 |
| BYOL | walk-specific frozen epoch-50 online CNN backbone | 128 |

The canonical concat representation has dimension 445. Neural checkpoint
hashes, source-bundle hashes, split identity hashes, branch-array hashes,
dimensions, and branch order must be recorded. Feature extraction may run
frozen encoders on evaluation sequences; no evaluation gradient or fitted
state is permitted.

## 3. Downstream feature standardization

The five branches have materially different units and the independently
trained walk encoders have different raw embedding scales. Therefore every
concatenated coordinate is standardized before the downstream head:

```text
mean_j = mean(train_feature[:, j])
std_j  = std(train_feature[:, j])
std_j  = 1 when std_j < 1e-8

train_scaled = clip((train_feature - mean) / std, -10, 10)
test_scaled  = clip((test_feature  - mean) / std, -10, 10)
```

This fitted state is independent per walk and uses only supervised training
rows. Evaluation features may never affect the mean, standard deviation,
zero-variance rule, clipping rule, architecture, checkpoint, or training
budget. The scaler arrays and their hashes must be stored and replayed.

The standardization does not change row identities or targets. It is a
downstream optimization transform, not an additional representation branch.
Input normalization is well established in neural time-series forecasting;
DeepAR explicitly rescales related series with different magnitudes
([Salinas et al., 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001)),
while financial time-series research documents the importance and potential
impact of normalization choices
([Passalis et al., 2020](https://doi.org/10.1109/TNNLS.2019.2944933)).
Phase 5 uses the simpler fixed train-only z-score contract above rather than
learning an adaptive normalization layer.

## 4. Regression unit and inverse transform

The scientific target remains signed two-hour probability movement:

```text
delta = close[t + 2h] - close[t]
```

For numerical optimization only, every learned regression model in the
complete Phase 5 programme uses probability-point units:

```text
y_train_pp = 100 * delta_train
prediction_delta = prediction_pp / 100
```

Multiplication by 100 is a fixed, exactly invertible change of units. It is not
estimated from training or evaluation data and cannot leak future information.
The same transform must later be used by every learned regression comparator.
Saved predictions must include both probability-point and reconstructed raw-
delta values. Primary metrics are calculated on raw delta; probability-point
metrics may be reported as an interpretable rescaling.

This choice retains raw probability change instead of unstable conventional
percentage/log return near bounded prices, following the prediction-market
evidence of Restocchi, McGroarty, and Gerding
([2019](https://doi.org/10.1016/j.physa.2018.09.183)). Fixed target rescaling
is also consistent with the scale-handling practice used in modern neural
time-series forecasting, provided predictions are transformed back before
scientific interpretation.

## 5. Frozen framework recipes

Both tasks use the five-branch 445-dimensional concat representation and the
existing lightweight `128 -> 64` task-head pattern.

### 5.1 Probability-movement regression

- output: one probability-point prediction;
- loss: MSE in probability-point units;
- optimizer: Adam, learning rate `1e-4`;
- batch size: 512;
- seed: 0.

### 5.2 Probability-movement classification

- output: three raw logits ordered `DOWN`, `STABLE`, `UP`;
- loss: training-prior logit-adjusted cross-entropy, `lambda=1.0`;
- natural training rows; no over/undersampling;
- evaluation retains the natural class distribution;
- optimizer: Adam, learning rate `1e-4`;
- batch size: 512;
- seed: 0.

Each walk/task uses one uninterrupted 50-epoch trajectory with snapshots at
epochs 5, 15, and 50. Epoch 50 is the predeclared principal result. Every
snapshot is retained and reported; no validation split, early stopping, or
evaluation-driven checkpoint selection is permitted.

The stage contains four trajectories: two walks times two tasks. Walk-specific
feature scalers, heads, checkpoints, predictions, and metrics remain separate.

## 6. Evaluation and reporting

Regression reports MAE, RMSE/MSE, Pearson and Spearman correlation, and sign
agreement on raw delta, together with exact-zero, non-zero, threshold-
exceeding, contract-macro, lifecycle, and imputation-exposure breakdowns.

Classification reports macro-F1 and balanced accuracy as primary metrics,
plus accuracy, weighted-F1, per-class precision/recall/F1, confusion matrix,
predicted-class counts, and one-vs-rest ROC-AUC/PR-AUC. The test class
distribution is never resampled.

Predictions from Walk 1 and Walk 2 may be pooled only after each has been
generated by its own frozen, out-of-future model. Full learned-baseline and
multi-seed comparison claims are prohibited until the later baseline matrix is
implemented on identical rows and budgets.

## 7. Execution gate

Before feature extraction:

1. replay both Phase 5 data bundles;
2. replay all six epoch-50 checkpoints;
3. prove every supervised training identity maps to a byte-identical encoder-
   training context; and
4. prove evaluation identities are disjoint from encoder and supervised
   training identities.

Before downstream execution:

1. validate feature dimensions, finiteness, hashes, and row identities;
2. verify train-only feature-scaler perturbation invariance;
3. complete CPU smoke tests for both heads and both losses;
4. freeze the four-run seed-0 matrix; and
5. only then train and evaluate the framework.

## 8. Completed artifacts and seed-0 observations

The complete implementation is in `src/features/phase5_features.py` and
`src/training/phase5_downstream.py`, with thin entry points under
`scripts_v3/`. Canonical outputs are stored at:

```text
experiments/phase5/features/walk{1,2}/five_branch_epoch50.npz
experiments/phase5/downstream/walk{1,2}/{regression,classification}/seed0/
experiments/phase5/reports/framework_downstream_seed0/
```

Both feature stores preserve the saved supervised row order and contain 445
finite coordinates per row. Their SHA-256 hashes are
`27eadd3de0b964ccecb8f116bd81192a61773ce6e454ee646568054599a407d0`
for Walk 1 and
`b7148b3f058b650e7ab4132b19354346b33fa92404afce7980233b143755b86d`
for Walk 2. Every supervised training row maps to a byte-identical encoder
context, and evaluation identities are disjoint from both encoder and
supervised training identities. The two train-only feature scalers have no
zero-variance coordinates and pass the declared evaluation-perturbation test.

All four trajectories completed 50 epochs and retained 5/15/50 checkpoints,
histories, predictions, detailed metrics, non-trained references, and CPU
prediction replay. Epoch-50 principal results are:

| Walk | Regression MAE | Regression RMSE | Pearson | Spearman | Classification macro-F1 | Balanced accuracy | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00255587 | 0.00929351 | 0.112727 | 0.001235 | 0.453469 | 0.465787 | 0.647956 |
| 2 | 0.00305107 | 0.01360845 | 0.030296 | -0.005724 | 0.453202 | 0.489162 | 0.612659 |

After pooling only the already generated out-of-future predictions, regression
obtains MAE `0.00271136`, RMSE `0.01083505`, Pearson `0.080745`, and Spearman
`-0.000518`. The exact-zero reference is stronger on pooled MAE (`0.00224972`)
and RMSE (`0.01077716`), so the seed-0 framework does not establish useful
probability-movement regression accuracy despite its weak positive Pearson
correlation.

Pooled classification obtains macro-F1 `0.454116`, balanced accuracy
`0.472953`, accuracy `0.636873`, macro ROC-AUC `0.741175`, and macro average
precision `0.460798`. Always-`STABLE` and repeated walk-local training-prior
references obtain macro-F1 `0.283121`, balanced accuracy `0.333333`, and
accuracy `0.738169`. The framework therefore trades majority-class accuracy
for materially better minority-sensitive metrics, as intended by the frozen
logit-adjusted objective.

These results are framework-only seed-0 evidence. They do not support a claim
of superiority over learned alternatives. The next Phase 5 gate is a separately
frozen learned-baseline matrix on identical rows, target units, budgets, and
evaluation rules; additional framework seeds remain pending.
