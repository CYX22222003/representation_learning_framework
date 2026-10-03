# Phase 6.7 Recent Frozen-Representation Baseline Plan

**Date:** 2026-10-04  
**Status:** Approved next-phase planning contract. Feasibility review,
implementation, training, feature extraction, downstream evaluation, and
reporting have not started.  
**Predecessors:**
`2026-09-26-phase-6-experiment-observation-and-outcomes.md`,
`2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`, and
`2026-10-03-mentor-advisor-feedback-and-next-direction.md`

## 1. Decision and priority

The next experimental phase will compare the canonical unified multi-branch
representation with recent self-supervised or unsupervised time-series
representation methods under one matched frozen-probing protocol.

This phase addresses the main remaining evidence gap: the project has matched
raw-input, handcrafted-feature, hybrid, and task-specific forecasting
comparators, but it has not directly compared its reusable representation with
recent reusable representations from the closest prior work.

This decision changes experiment order, not completed results:

1. Phase 6.7 is the immediate implementation and execution priority.
2. Phase 7A canonical single-branch and leave-one-out analysis follows Phase
   6.7 and remains unimplemented.
3. The whole Phase 6.6 price-focused fusion, xLSTM-Mixer, and decoder-capacity
   programme is deferred until the representation-baseline and branch-ablation
   evidence are available.
4. xLSTM-Mixer remains an approved later task-specific complete-system
   baseline. It is not evidence of reusable representation quality and is not
   a Phase 6.7 completion gate.
5. Phase 7B alpha research remains unspecified and unauthorized.

The canonical project method should be called the **unified multi-branch
time-series representation framework**, not simply an SSL framework. Its
statistical and transformed branches are deterministic, its VAE branch is
reconstruction-based, and only the contrastive and BYOL branches use the
current augmentation-based SSL objectives.

## 2. Research questions

Phase 6.7 asks:

1. Does the canonical frozen `H0` representation perform competitively with
   recent target-free time-series representations when the downstream data,
   head, loss, training budget, and evaluation metrics are fixed?
2. Is any advantage consistent across movement classification, future-price
   prediction, and future-realised-variance prediction, rather than confined
   to one task or one walk?
3. What compute, memory, representation-width, and implementation costs
   accompany each result?

Phase 6.7 does not ask whether one end-to-end forecasting architecture is the
best price predictor. It also does not establish causal branch importance,
profitable alpha, universal transfer, or state-of-the-art performance outside
the two accepted Polymarket walks.

## 3. Frozen candidate roster

The intended primary roster is:

| ID | Method | Source type | Project role |
|---|---|---|---|
| `TD-F` | TimeDART-Frozen | ICML 2025 | Autoregressive generative/self-supervised representation baseline |
| `LWA-F` | Learning Without Augmenting-Frozen | NeurIPS 2025 | Frozen multi-domain time/Fourier/time-frequency representation baseline |
| `SAURL-F` | SaURL-TS-Frozen | Pattern Recognition 2026 | Adaptive time/frequency bootstrap representation baseline |

This roster intentionally contains two recent top-conference methods and one
recent top-journal method. The journal entry is included because it is closer
to the project's reusable-representation contract than a finance-specific
method whose pretraining uses future labels or unavailable inputs.

`SISSEL-Frozen` is the only pre-approved reserve. It replaces `SAURL-F` only
if the pre-evaluation feasibility gate finds that SaURL-TS cannot be
implemented faithfully within the available source, licence, hardware, or
time budget. TimeDART or LWA may not be silently replaced; rejecting either
requires a dated amendment before downstream evaluation.

GCFin and MCSIP are not admitted to the primary matrix. Their future-label or
extra-input assumptions do not provide a clean target-free, OHLCV-only test of
reusable representation quality. They remain literature context only.

## 4. Stage A: source and feasibility freeze

No model training begins until one read-only feasibility manifest records for
every candidate:

- paper and official-code identity, licence, repository commit or version;
- supported input length, channel handling, patching or view requirements;
- exact pretraining objective and confirmation that it is target-free under
  this project's use;
- exact inference representation, pooling rule, output width, and components
  discarded after pretraining;
- source-derived architecture and optimizer settings;
- expected parameter count, GPU-memory demand, and a CPU/CUDA smoke-test
  result on synthetic and small real training-only batches;
- any deviation required by the 64-by-5 OHLCV input; and
- the final decision to admit, reserve-replace, or reject the candidate.

The intended extraction points, subject to official-code verification, are:

- `TD-F`: pooled causal encoder patch states with the diffusion decoder
  removed after pretraining;
- `LWA-F`: the source-defined frozen time representation and learned
  cross-domain mappings, with auxiliary training-only components removed as
  specified by the source evaluation path; and
- `SAURL-F`: the source-defined attention-combined time, frequency, and
  cross-domain representation, excluding pretraining projectors/predictors.

The feasibility audit may inspect training rows, shapes, metadata, source
code, runtime, and resource use. It may not inspect downstream evaluation
metrics or use evaluation targets to choose a model, extraction point, width,
or hyperparameter.

The complete admitted roster and all model/task/walk artifact paths must be
written to a manifest before the first downstream evaluation is read.

## 5. Shared data and leakage contract

Reuse the accepted Phase 5/6 one-hour global-calendar walks:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every external encoder is trained separately for each walk. Walk 2 history
may not update, select, or reinterpret a Walk 1 encoder or head.

All candidates consume the same saved 64-by-5 OHLCV contexts and the same
target-free encoder-training population used by the canonical walk-specific
neural encoders. Movement, price, and volatility targets are unavailable to
encoder pretraining. Any candidate-specific scaler, normalization, spectral
transform, mask, or view statistic is fitted on the permitted walk-training
population only and replayed unchanged for downstream rows.

Target maturity, observed future endpoints, segment continuity, and task-
specific eligibility are applied only when aligning frozen embeddings to the
existing supervised task bundles. No method may obtain more downstream rows
than another method within the same task/walk comparison because of a
model-specific filter. If a method cannot produce a finite embedding for an
established row, it fails replay; the task population is not silently reduced.

The repository's train/test-only rule remains unchanged: no validation split,
early stopping, evaluation-driven restart selection, or best-on-evaluation
checkpoint selection is allowed.

## 6. Stage B: matched encoder pretraining

For the admitted three-model roster:

```text
3 methods x 2 walks = 6 new encoder trajectories
```

The primary characterisation uses seed `0` and one uninterrupted 50-epoch
trajectory per method/walk, with checkpoints at epochs 5, 15, and 50. Epoch 50
is fixed as the representation-extraction checkpoint before downstream
evaluation. If a source implementation counts optimizer steps rather than
epochs, the feasibility manifest must convert the budget to full training-set
passes and freeze the equivalent 5/15/50 boundaries before training.

Source-derived architecture, objective, augmentation/view, and optimizer
semantics should be preserved where they do not violate the data contract.
The project does not tune source hyperparameters on a new validation or
evaluation split. Any unavoidable adaptation is named explicitly, so the
result is reported as a project `-Frozen` adaptation rather than an exact
paper reproduction.

Every trajectory stores configuration, source version, data and row hashes,
random seed, checkpoints, loss history, parameter count, elapsed time, peak
memory, and replay metadata. Training collapse or non-finite behaviour is a
reportable result; it is not permission to introduce an unplanned model.

## 7. Stage C: frozen features and downstream matrix

Freeze each epoch-50 encoder and extract one row embedding for every existing
downstream identity. This produces:

```text
3 methods x 2 walks = 6 new representation stores
```

Each store records the source checkpoint hash, ordered row identity, output
width, finiteness, and extraction hash. The encoder receives no downstream
gradient.

The three current tasks are:

| Task | Existing contract |
|---|---|
| Movement classification | two-hour `DOWN/STABLE/UP`, `tau=0.001` |
| Future price | absolute `close[t+8]`, with implied-movement diagnostics |
| Future realised variance | raw probability-change RV over `(t,t+8h]` |

Within each task and walk, reuse the exact canonical `H0` training and
evaluation identities, labels, train-only scaling rule, head family, loss,
output transform, batch size, optimizer, learning rate, seed, epoch budgets,
non-learned references, and metric implementation. Fit a separate coordinate
scaler to each method's task-training embeddings and freeze it for evaluation.
The different native representation widths are retained and
reported; no method-specific supervised bottleneck is added.

The new downstream matrix is:

```text
3 representations x 3 tasks x 2 walks = 18 new downstream trajectories
18 trajectories x 3 snapshots = 54 evaluated snapshots
```

Six immutable epoch-50 `H0` task/walk trajectories are the direct references,
giving a 24-trajectory representation-comparison table. Existing raw-input,
handcrafted, hybrid, and non-learned results remain contextual references and
are not counted as new Phase 6.7 runs.

Downstream trajectories use the established 5/15/50 snapshot contract and
epoch 50 principal result. The entire predeclared matrix is reported; a
method, task, walk, or checkpoint may not be removed because its result is
negative.

## 8. Fairness and capacity reporting

The primary comparison holds the supervised probe design fixed while
retaining each source method's native embedding width. Consequently, first-
layer parameter counts can differ. Report for every method:

- embedding width;
- encoder, probe, and total parameter counts;
- pretraining and embedding-extraction time;
- downstream training and inference time;
- peak device memory; and
- whether the source recipe required a project adaptation.

A shared-width projection may be studied later only as a separately amended
matched-capacity sensitivity applied to `H0` and every external method. It is
not part of the primary Phase 6.7 matrix.

## 9. Evaluation and claim boundary

Report results separately by walk and as a clearly labelled pooled summary.
For every task/walk, report absolute values and paired differences from `H0`
on identical evaluation rows.

- Classification: accuracy, macro-F1, balanced accuracy, per-class metrics,
  confusion matrix, and class-collapse diagnostics.
- Future price: MAE, RMSE/MSE, Pearson/Spearman, persistence skill, implied-
  movement correlation and sign agreement, global Rank IC, timestamp-level
  cross-sectional Rank IC, and existing subgroup diagnostics.
- Volatility: MAE, RMSE/MSE, Pearson/Spearman, historical-persistence skill,
  contract-macro results, and existing tail/subgroup diagnostics.

Because rows and horizons overlap, row-wise IID significance tests are not
valid. Any uncertainty analysis requires a separately frozen contract/calendar
block bootstrap. Linear or kernel CKA may be added later as descriptive
representation analysis, but it cannot select a baseline or substitute for
matched downstream results.

A result may support only a statement such as:

> Under the fixed frozen-probing protocol, representation A was stronger or
> weaker than canonical `H0` on task T and walk W.

One method winning on one task does not establish universal representation
superiority. The initial seed-0, two-walk matrix is characterisation evidence,
not a state-of-the-art or profitable-trading claim.

## 10. Artifacts and implementation boundary

New outputs belong under:

```text
experiments/phase6_7/
  feasibility/
  manifests/
  encoder_pretraining/{timedart_frozen,lwa_frozen,saurl_frozen_or_sissel_frozen}/
  features/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/
  reports/frozen_representation_seed0/
```

Reusable model, training, feature, replay, and reporting code belongs under
`src/`. Orchestration belongs under a new `scripts_v6/` generation so the
completed `scripts_v3/`--`scripts_v5/` contracts remain immutable. Bootstrap
commands are manifest-only by default; training requires an explicit
`--execute` flag.

The implementation must not alter canonical Phase 5/6 artifacts. Immutable
`H0` references are replayed, not retrained, unless a documented identity or
artifact failure is discovered.

## 11. Ordered execution gates

1. Replay the two accepted walk bundles, target-free encoder populations, six
   task/walk label populations, and six immutable `H0` references.
2. Complete the source/licence/API/hardware feasibility manifest without
   reading evaluation metrics.
3. Freeze the admitted roster, fallback decision, extraction points,
   architecture settings, budgets, paths, and expected matrix.
4. Implement source-faithful adapters and pass CPU/CUDA shape,
   forward/backward, determinism, provenance-failure, and small-batch tests.
5. Train and replay all six admitted encoder trajectories.
6. Extract and replay all six frozen representation stores.
7. Freeze the 24-trajectory comparison manifest containing 18 new runs and six
   immutable `H0` references.
8. Execute all 18 downstream trajectories and all 54 snapshots without
   evaluation-driven truncation.
9. Replay checkpoints, predictions, metrics, scalers, row identities, and
   representation hashes.
10. Generate the complete per-walk, pooled, resource, and adaptation report.

## 12. Exit conditions and handoff

Phase 6.7 is complete only when:

- the final roster contains TimeDART-Frozen, LWA-Frozen, and either
  SaURL-TS-Frozen or the pre-approved SISSEL-Frozen fallback;
- all six encoder trajectories and six representation stores pass replay;
- all 18 new downstream trajectories and six immutable `H0` references pass
  standalone prediction and metric replay;
- every task/walk comparison uses identical ordered rows and the established
  task contract;
- the full snapshot, resource, and adaptation tables are preserved, including
  failed or negative outcomes; and
- the report maintains the seed-0, two-walk, native-width, source-adaptation,
  and non-trading claim boundaries.

After this exit, Phase 7A should execute the canonical branch matrix and add
predeclared branch-role diagnostics. Only then should the project reassess
whether the deferred Phase 6.6 price-specific fusion, xLSTM-Mixer, and richer
decoder studies are necessary for the final report.
