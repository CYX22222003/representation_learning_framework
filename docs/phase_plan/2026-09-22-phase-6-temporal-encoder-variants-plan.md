# Phase 6 Temporal Encoder Variants and Heterogeneous Features Plan

**Date:** 2026-09-22

**Status (updated 2026-09-26):** Integration infrastructure and thin
`scripts_v4` entry points are implemented and CPU-tested for all encoder
families, feature widths, and task heads. The seed-0 encoder manifest is
frozen, all eight walk-specific temporal encoder trajectories are trained and
replay-validated at epochs 5/15/50, and all six task/walk epoch-50 master
feature stores pass identity, hash, width, finiteness, and exact-duplicate
replay for the 11 configurations. The sibling volatility gate is complete.
The 66-entry downstream manifest is frozen with six replayed H0 references and
60 required new trajectories; its manifest-only freeze did not train models.
Variant downstream trajectories, CKA, and the comparison report have not run;
execution gate 10 is next.

**Predecessors:** `2026-09-20-phase-3-experiment-plan.md` and
`2026-09-21-phase-5-experiment-plan.md`

**Sibling task:** `2026-09-22-phase-6-volatility-forecasting-plan.md`

## 1. Purpose and authority

This document specifies Phase 6 Task 2: test whether LSTM and Transformer
backbones improve or complement the canonical CNN representations when moved
from the superseded Phase 2/3 setting into the accepted recent one-hour,
global-calendar-walk setting.

For this task, this document supersedes the future-execution portions of:

- `2026-09-13-phase-2-encoder-variant-experiment-run-plan.md`;
- `2026-09-14-phase-2-selected-branch-encoder-amendment.md`; and
- the Phase 3 encoder addition matrix in
  `2026-09-20-phase-3-experiment-plan.md`.

Those documents and their artifacts remain design and characterisation
evidence. Phase 2 used the invalid preprocessing-before-split pipeline. Phase
3 corrected the raw-time boundary and successfully trained all temporal
variants, but evaluated them on an absolute next-close task dominated by
persistence and on a single lifecycle-biased tail. Neither generation can
answer the Phase 6 question without retraining on the Phase 5 walks.

This plan fixes the architecture questions, seed, feature configurations,
tasks, comparison logic, and execution order. It does not authorize training
until the Phase 6 implementation and manifest gates pass.

## 2. Research questions

Phase 6 separates two questions that must not be collapsed into one claim.

### 2.1 Backbone substitution

> When all other branches and the self-supervised objective are fixed, is an
> LSTM or Transformer a better practical backbone than the corresponding CNN?

Substitution keeps the complete representation width fixed at 445 dimensions
and replaces exactly one 128-dimensional CNN branch.

### 2.2 Heterogeneous feature complementarity

> Does an LSTM or Transformer preserve useful temporal information that is not
> already available from the corresponding CNN and the other representation
> branches?

Addition retains the CNN and appends one independently trained 128-dimensional
temporal branch. Because this widens the representation and downstream head,
every addition is compared with a same-width duplicated-CNN control.

The experiment does not ask whether one backbone is universally superior.
Results must be stated by SSL family, task, and walk.

## 3. Controlled data and model lifecycle

Phase 6 reuses the two accepted Phase 5 clean native one-hour walk cohorts and
their global calendar intervals:

| Walk | Permitted encoder/downstream training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

The input remains a 64-by-5 one-hour OHLCV sequence. The Phase 5 isolated-one-
hour fill, observed decision endpoint, causal 24-hour activity rule, gap
segments, walk cutoffs, retrospective cohort-selection assumption, and
approved offline-cleaning assumption remain unchanged.

Every neural encoder is walk-specific:

- a Walk 1 encoder trains only on Walk 1 `encoder_train_sequences`;
- a Walk 2 encoder trains only on Walk 2 `encoder_train_sequences`; and
- Walk 2 history must never update, select, or reinterpret a Walk 1 model or
  prediction.

Encoder eligibility remains target-free. Price, classification, and volatility
target existence or value must not change the encoder-training population.
Changing a downstream label bundle must leave encoder sequence bytes and
identity hashes unchanged.

The completed Phase 5 epoch-50 contrastive-CNN and BYOL-CNN checkpoints are the
immutable `H0` references after their data, recipe, checkpoint, and inference
hashes replay. Phase 3 checkpoints are not initialization inputs.

## 4. Temporal encoder contract

For a code-aligned description of tensor shapes, recurrent gates, attention,
positional encoding, SSL wrappers, and downstream readout, see
[`../phase6_temporal_encoder_architecture.md`](../phase6_temporal_encoder_architecture.md).

### 4.1 Candidate inventory

Train four candidates independently in each walk:

| SSL family | Candidate backbone | Branch name | Output width |
|---|---|---|---:|
| Contrastive / NT-Xent | one-layer LSTM | `contrastive_lstm` | 128 |
| Contrastive / NT-Xent | compact Transformer | `contrastive_transformer` | 128 |
| BYOL | one-layer LSTM | `byol_lstm` | 128 |
| BYOL | compact Transformer | `byol_transformer` | 128 |

This creates eight new encoder trajectories:

```text
2 walks x 2 SSL families x 2 temporal backbones = 8
```

VAE remains the canonical 64-dimensional MLP encoder/decoder. Statistical and
transformed branches remain deterministic. Phase 6 does not introduce a VAE
temporal-backbone matrix.

### 4.2 Architecture settings

The accepted Phase 3 temporal model definitions are the starting
implementation contract:

| Setting | LSTM | Transformer |
|---|---:|---:|
| hidden/downstream width | 128 | 128 |
| recurrent/encoder layers | 1 | 2 |
| attention heads | n/a | 4 |
| feed-forward width | n/a | 256 |
| dropout | 0.0 | 0.1 |
| position encoding | n/a | sinusoidal |
| readout | final hidden state | normalized final token |
| maximum supported sequence length | n/a | 512 |

Bidirectional attention inside the historical 64-hour context is permitted:
all context tokens are available at the decision time. The Transformer must
not receive any target-interval token.

The output width is controlled, but trainable parameter counts are not equal
across CNN, LSTM, and Transformer backbones. The study therefore compares
practical fixed-width backbone substitutions, not mathematically parameter-
matched architectures. Parameter count, training time, inference time, and
peak device memory are mandatory reporting fields.

### 4.3 SSL semantics and training recipe

Within each SSL family, hold fixed:

- the same walk-specific target-free training sequences;
- the existing scaling, jitter, and time-mask view generator;
- projector output width 128;
- NT-Xent temperature `0.2` for contrastive candidates;
- distinct BYOL online and EMA target backbones;
- BYOL projector, predictor, loss semantics, and target decay `0.99`;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `256`, shuffled training rows, and dropped incomplete batch;
- one uninterrupted 50-epoch trajectory;
- checkpoints at epochs 5, 15, and 50; and
- seed `0` only.

Epoch 50 is frozen as the sole downstream feature source. Epochs 5 and 15 are
training-health snapshots and cannot be selected from downstream results.

The shared view generator can perturb the implicit zero-volume missingness
signal in temporary SSL views. Keeping it identical across backbones preserves
the architecture comparison and matches the canonical Phase 5 reference.
Designing mask-aware augmentations would change the pretraining method and is
outside this matrix.

## 5. Frozen feature-configuration matrix

The canonical base is:

```text
H0 = statistical + transformed + VAE + contrastive CNN + BYOL CNN
```

Its width is:

```text
70 + 55 + 64 + 128 + 128 = 445
```

### 5.1 Substitution configurations

| ID | Branch change from `H0` | Width | Primary comparison |
|---|---|---:|---|
| `H0` | none | 445 | immutable canonical reference |
| `HC-SL` | contrastive CNN -> contrastive LSTM | 445 | `HC-SL - H0` |
| `HC-ST` | contrastive CNN -> contrastive Transformer | 445 | `HC-ST - H0` |
| `HB-SL` | BYOL CNN -> BYOL LSTM | 445 | `HB-SL - H0` |
| `HB-ST` | BYOL CNN -> BYOL Transformer | 445 | `HB-ST - H0` |

Every substitution contains exactly five branches. A primary substitution may
not contain both the CNN and its temporal alternative.

### 5.2 Heterogeneous single-addition configurations

| ID | Branch addition to `H0` | Width | Questions |
|---|---|---:|---|
| `HC-AL` | add contrastive LSTM | 573 | contrastive LSTM complementarity |
| `HC-AT` | add contrastive Transformer | 573 | contrastive Transformer complementarity |
| `HB-AL` | add BYOL LSTM | 573 | BYOL LSTM complementarity |
| `HB-AT` | add BYOL Transformer | 573 | BYOL Transformer complementarity |

Each addition retains the canonical CNN, appends one temporal branch, uses
concat fusion, and fits the unchanged task-head family. No coordinate
alignment between independent embeddings is required: every branch remains a
separately named block, and every coordinate is standardized from the task's
training rows only.

### 5.3 Matched-width duplicate controls

| ID | Addition to `H0` | Width | Control for |
|---|---|---:|---|
| `HC-DC` | exact duplicate of contrastive-CNN features | 573 | `HC-AL`, `HC-AT` |
| `HB-DC` | exact duplicate of BYOL-CNN features | 573 | `HB-AL`, `HB-AT` |

The duplicate array is stored under a distinct branch name but retains the
source feature hash. It adds downstream input width and parameters without
adding new information. Because duplicate coordinates are collinear, this is
an informative but imperfect capacity control; reports must state that
limitation.

For an addition to support complementarity, interpret both comparisons:

```text
addition versus H0                 complete-system improvement
addition versus family duplicate  evidence beyond extra width alone
```

### 5.4 Excluded configurations

The initial Phase 6 matrix excludes:

- CNN + LSTM + Transformer `ALT` bundles and double-duplicate controls;
- configurations replacing both contrastive and BYOL CNNs simultaneously;
- combinations selected after reading a task result;
- branch-only or leave-one-out ablations;
- gated or learned fusion;
- downstream decoder variants;
- fixed-first-walk encoder transfer;
- lifecycle-conditioned encoders or experts; and
- additional seeds.

These are different research questions. Multiple-seed confirmation is
explicitly deferred until after Phase 6 and has the lowest current priority.

## 6. Downstream task matrix

All 11 configurations are committed to all three tasks. A weak result on one
of the earlier available tasks does not remove a configuration from the later
volatility comparison.

### 6.1 Two-hour movement classification

Reuse the validated Phase 5 `h=2h`, `tau=0.001` rows and natural evaluation
distribution. Training uses the same train-prior logit-adjusted cross-entropy
with `lambda=1.0` as the canonical framework.

Primary metrics:

- macro-F1; and
- balanced accuracy.

Supporting outputs include accuracy, weighted-F1, per-class precision/recall/
F1, confusion matrix, predicted counts, ROC-AUC, PR-AUC, NLL, and Brier score.
Always-`STABLE` and repeated training-prior scores remain non-learned
references.

### 6.2 Eight-hour future-price prediction

Reuse the exact Phase 5 `absolute_price_h8` rows. The head predicts the future
probability through the same sigmoid-bounded architecture without an explicit
current-price skip.

Report:

- price-level MAE, RMSE/MSE, Pearson, and Spearman;
- current-price persistence comparison;
- implied-movement Pearson, Spearman, and sign diagnostics;
- timestamp-level cross-sectional Rank IC; and
- the fixed last-hour reversal reference.

Price-level reconstruction alone cannot support an encoder-quality claim.
Implied-movement ranking and comparison with persistence/reversal remain the
financial interpretation.

### 6.3 Future realised-variance prediction

Every configuration must also be evaluated on the Phase 6 future interval
realised-variance task defined in
`2026-09-22-phase-6-volatility-forecasting-plan.md`:

$$
RV^{PM}_{t,H,\delta}
=
\sum_{j=1}^{H/\delta}
\left(p_{t+j\delta}-p_{t+(j-1)\delta}\right)^2.
$$

This part of the downstream launcher remains blocked until the sibling task:

1. completes its training-period horizon audit;
2. freezes `H` in a dated amendment;
3. builds and validates both walk-specific label bundles; and
4. freezes the volatility head, loss, scaling, references, and metrics.

Temporal encoder pretraining is allowed before those items because it is
target-free. Comparative price or classification results must not be used to
change which configurations enter volatility: all 11 are precommitted.

The volatility task uses the sibling plan's exact rows, nonnegative output
contract, raw-unit evaluation, zero/training-location/historical-volatility
references, and MAE, RMSE/MSE, Pearson, and Spearman metrics.

### 6.4 Matrix size

The complete comparison table contains:

```text
11 configurations x 2 walks x 3 tasks = 66 configuration-walk-task entries
```

All six `H0` entries now exist: four Phase 5 classification/future-price
references and two independently completed sibling-task volatility references.
They may be reused only after complete artifact, feature, head, task-row, and
prediction replay. The remaining new downstream execution count is therefore:

```text
40 temporal/control entries for classification and future price
+ 20 temporal/control entries for future realised variance
= 60 new downstream trajectories
```

## 7. Downstream training and fairness contract

For each task, every configuration must use identical ordered training and
evaluation row identities and targets. Task-specific populations may differ
across tasks because their horizons and eligibility rules differ.

For every configuration and walk:

- extract all neural features through frozen epoch-50 encoders;
- fit coordinatewise mean and population standard deviation using eligible
  task-training features only;
- apply the frozen scaler to evaluation features and clip to `[-10, 10]`;
- train only the unchanged task-head family;
- use seed `0`, batch size `512`, Adam learning rate `1e-4`, and weight decay
  `0` unless the sibling volatility plan freezes a task-specific exception;
- run one uninterrupted 50-epoch downstream trajectory with snapshots at
  epochs 5, 15, and 50;
- treat epoch 50 as the predeclared principal result; and
- use no validation split, early stopping, restart selection, or evaluation-
  driven configuration choice.

All budgets are reported. No lowest-evaluation-error snapshot may be relabelled
as the selected model.

Feature and prediction manifests must reject missing, unexpected, duplicated,
or reordered rows. Different feature widths are expected only for the declared
445- and 573-dimensional configurations.

## 8. Diagnostics and interpretation

### 8.1 Encoder health

At every encoder snapshot, record:

- finite loss, gradients, weights, and embeddings;
- average coordinate standard deviation;
- embedding norm and absolute mean;
- collapse warning under the predeclared threshold;
- parameter count, training time, inference time, and peak memory; and
- full data/configuration/checkpoint provenance.

A failed or collapsed run may be repaired only for a documented implementation
fault. It does not authorize choosing another architecture or snapshot from
downstream evaluation.

### 8.2 Representation similarity

Compute centered linear CKA between each temporal branch and its family CNN on
one predeclared common sample of walk-specific encoder-training identities.
The sample identities and extraction checkpoints must be frozen before CKA is
computed.

CKA is descriptive:

- high CKA can indicate redundancy;
- low CKA indicates difference, not usefulness; and
- only paired downstream performance on identical rows tests predictive
  complementarity.

CKA must not select configurations, tasks, or checkpoints.

### 8.3 Required comparisons

For each task and walk, report paired metric differences for:

- `HC-SL - H0` and `HC-ST - H0`;
- `HB-SL - H0` and `HB-ST - H0`;
- `HC-AL - H0`, `HC-AT - H0`, and each addition minus `HC-DC`; and
- `HB-AL - H0`, `HB-AT - H0`, and each addition minus `HB-DC`.

Contrastive and BYOL families are interpreted separately. Do not rank all 11
configurations and call the best row a universally selected architecture.

Stride-one decision rows and targets are temporally dependent. If paired
uncertainty intervals are reported, use a predeclared contract/calendar-block
resampling procedure. Such intervals quantify evaluation-row uncertainty but
do not replace initialization-seed uncertainty.

## 9. Artifact and implementation contract

New Phase 6 generated artifacts belong under:

```text
experiments/phase6/encoder_variants/
  manifests/
  pretraining/walk{1,2}/{contrastive,byol}/{lstm,transformer}/seed0/
  features/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/
  diagnostics/{health,cka,resources}/
  reports/
```

Thin Phase 6 entry points should live under `scripts_v4/`; reusable training,
feature, validation, and reporting logic remains under `src/`. Phase 2/3
scripts and artifacts must not be overwritten or imported as executable
workflows. Reusable model definitions may be adapted only after their current
tests and architecture semantics are verified.

Every run retains configuration, environment, source commit, input and
identity hashes, model architecture, parameter count, checkpoints, cumulative
history, predictions, metrics, completion marker, and CPU prediction replay.
The bootstrap command must be manifest-only by default and require an explicit
execution flag.

## 10. Ordered execution gates

1. **Approve this plan.** Freeze the 11 configurations and three-task scope.
2. **Audit the reusable Phase 3 model definitions.** Verify LSTM/Transformer,
   contrastive, and BYOL semantics on `[N,64,5]` inputs.
3. **Implement Phase 6 integration and tests.** Add walk-aware trainers,
   feature extraction, configuration construction, duplicate controls,
   downstream loaders, replay, and reporting without changing Phase 5 code.
4. **Run disposable CPU smoke tests.** Cover every encoder family, feature
   width, task head, occupied-path refusal, and provenance failure.
5. **Freeze the encoder manifest without execution.** Record all eight new
   trajectories, exact recipes, paths, hashes, and expected artifacts.
6. **Train and validate all eight temporal encoders.** Extract only epoch-50
   branches after health and provenance checks pass.
7. **Build and validate all 11 feature configurations.** Prove branch lists,
   widths, identities, duplicate-source hashes, and finite values.
8. **Complete the sibling volatility gate.** Freeze `H`, labels, head, loss,
   references, and metrics before downstream comparative results are used.
9. **Freeze the complete 66-entry downstream manifest.** Include the six
   replayed `H0` task/walk references and all 60 new trajectories.
10. **Execute every downstream configuration.** Weak early results do not
    truncate the matrix.
11. **Replay checkpoints, predictions, metrics, and identities.** Reject
    partial or mismatched artifacts before comparison.
12. **Report the complete matrix.** Separate substitution from complementarity
    and representation evidence from width, optimization, and task effects.

## 11. Non-goals and claim boundaries

Phase 6 does not establish:

- robust architecture superiority across initialization seeds;
- the optimal encoder depth, hidden width, attention head count, or
  parameter-matched architecture;
- that lower SSL loss implies better downstream prediction;
- that representation difference in CKA implies useful information;
- that a wider heterogeneous bundle is better without its duplicate control;
- that one task's winner transfers universally to another task; or
- that improved forecast metrics imply a profitable trading strategy.

The task supports three narrower conclusions when evidence permits:

1. a fixed-width temporal backbone substitutes effectively for its CNN within
   a named SSL family;
2. a temporal branch adds predictive information beyond the canonical bundle
   and a same-width duplicate control; and
3. an effect repeats or changes across classification, future-price, and
   future-realised-variance tasks and across the two calendar walks.

All conclusions remain single-seed characterisation evidence. Multi-seed
confirmation is deferred until after Phase 6.

## 12. Completion conditions

The Phase 6 temporal-encoder task is complete only when:

- all eight walk-specific temporal encoder trajectories pass health,
  provenance, and checkpoint replay;
- epoch-50 features for every temporal branch pass identity and finite-value
  validation;
- all 11 feature configurations have exact declared branches and widths;
- the volatility horizon and label contract are frozen independently of
  encoder results;
- all 66 task/walk/configuration entries exist, including six replayed `H0`
  references and 60 new trajectories;
- every comparison uses identical rows and targets within its task and walk;
- every prediction and metric replays from its saved checkpoint;
- substitution and heterogeneous-addition results are reported separately
  with duplicate controls, resources, CKA, and task references; and
- limitations and negative results are retained without post-hoc combination
  or configuration selection.

Until then, the task must be described as planned, implemented, or partially
executed according to the available evidence, never as completed merely
because the earlier Phase 3 model classes exist.
