# Phase 6.7 Recent Frozen-Representation Baseline Plan

**Date:** 2026-10-04  
**Status:** Required SaURL/LWA execution complete; integrated core reporting
pending, updated 2026-10-05.
The independently authored SaURL-TS model adapter, SaURL-specific Stage 0--4
infrastructure, and focused CPU unit tests are implemented. Its mask contract
uses the paper's deterministic sigmoid threshold, with no repository-derived
mask noise. The feasibility audit entry point, gated alternating pretrainer,
two-store extractor/replay path, and native-width common probes now live under
`src/` and `scripts_v6/`. The CUDA/resource gate and both 50-epoch SaURL
pretraining trajectories are complete; all 5/15/50 checkpoints pass CPU
replay. Same-device CUDA replay is bit-exact. The accepted cross-device
criterion is relative L2 at most `5e-4` and cosine at least `0.999999`, frozen
after diagnosing scale-amplified Conv1d/max-pooling drift without reading any
downstream metric. Near-zero learned masks and large embedding norms are
retained as reportable diagnostics. The LWA v2 paper and official commit have
now been audited in `docs/baselines/LWA/`, including the no-software-licence
boundary, 384-dimensional extraction path, length-64 adaptation, and static
resource estimate. All twelve LWA adaptation decisions were approved on
2026-10-04: PyWavelets with chunked float32 CWT caching is required and
physical batch 128 is authoritative for both walks. The Lumid container,
CUDA runtime, and pinned `PyWavelets==1.8.0` CWT path are verified. The
independent LWA model package and Stage 2--4 cache/pretraining, frozen-feature,
and native-width downstream launch/replay infrastructure are implemented. The
original 17 model tests pass in both local and admitted remote runtimes, and
all 25 local LWA tests pass including the synthetic atomic-cache and
pipeline contracts. Both full transform caches and both walk-specific
joint-50/mapper-50 trajectories now exist on Lumid. The original strict
elementwise CPU replay check stopped the launcher before Walk 2 on benign
Wavelet CPU/CUDA drift: maximum absolute difference `9.39e-5`, relative L2
`3.02e-5`, and cosine effectively one. LWA replay now records strict misses
inside the already accepted relative-L2 `5e-4`/cosine `0.999999` cross-device
bounds as warnings in `replay_validation.json`; missing/corrupt artifacts,
provenance/shape mismatch, non-finite output, or material drift remains fatal.
Both 384-dimensional LWA master stores and all six downstream trajectories
are complete and valid at epochs 5/15/50. LWA does not consistently
outperform immutable H0: it is materially weaker on Walk 1 classification and
both price MAEs, while isolated Walk 2 price/volatility RMSE and correlation
gains do not establish broad superiority. The full shared core manifest and
integrated H0/SaURL/LWA reporting remain pending.
On resume, a rerun of the CUDA smoke may regenerate nondeterministic loss and
timing fields. The renewed manifest is therefore revalidated by stable source,
dependency, cache, physical-batch, resource-limit, and admission semantics;
hash-only drift is recorded, while a changed gate condition remains fatal.
Both SaURL master stores and all
six staged downstream trajectories are complete and replay-valid at 5/15/50.
The epoch-50 result is generally weaker than immutable H0, with isolated
metric-specific improvements that do not support consistent superiority. The
interim staged result is recorded under
`experiments/phase6_7/reports/saurl_staged_seed0/`.
The project owner subsequently approved staged execution on 2026-10-04:
SaURL's six downstream trajectories may run from their frozen SaURL-only
manifest before LWA is implemented. LWA remains a required core method, and
SaURL metrics may not select or change its source, architecture, extraction
point, hyperparameters, rows, budget, or retry policy.
LWA-Frozen and SaURL-TS-Frozen are both mandatory core methods. Independently,
SISSEL-Frozen and TimeDART-Frozen are peer optional extensions. The owner has
commissioned TimeDART preparation after the required method execution; its
model implementation remains gated on the method-specific decision record.
SISSEL remains uncommissioned pending mentor review. Any admitted extension
is post-core exploratory evidence and cannot retroactively select, replace,
or change the required core methods or their conclusions. Its source,
adaptation, extraction, resource, and complete matrix contract must be frozen
before implementation or evaluation.

On 2026-10-05 the project owner commissioned the TimeDART preparation stage.
The ICML paper, official source at commit
`e658a648cf6b04612ca643d10a634b45e136194c`, licence boundary, executable
`[B,64,5]` behavior, candidate extraction boundary, comparison contract, and
staged implementation plan are now audited under `docs/baselines/TimeDART/`.
The owner approved all eleven substantive TimeDART decisions on 2026-10-05;
Question 7 was an informational clarification. The approved extraction has
160 channelwise pooled encoder coordinates plus the five means and five
standard deviations that source forecasting uses outside its encoder, giving
170 coordinates. The independently authored model and ten focused CPU tests
are complete under `src/baselines/timedart/` and `tests/baselines/timedart/`.
A small accepted Walk 1 encoder-row forward/backward smoke passes. The owner
instructed that work stop after model implementation for evaluation: no
TimeDART training launcher, trajectory, feature store, downstream integration,
or report exists. The next Lumid sandbox is nevertheless synchronized and
environment-ready: persistent GitHub SSH, all container dependencies, all six
input bundles, focused CPU tests, and a bounded real CUDA batch-16 smoke have
been verified. This remains a post-core exploratory extension and cannot alter
the completed H0/SaURL/LWA core.

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

The required core roster is:

| ID | Method | Source type | Project role |
|---|---|---|---|
| `LWA-F` | Learning Without Augmenting-Frozen | NeurIPS 2025 | Frozen multi-domain time/Fourier/time-frequency representation baseline |
| `SAURL-F` | SaURL-TS-Frozen | Pattern Recognition 2026 | Adaptive time/frequency bootstrap representation baseline |

This core intentionally combines one recent top-conference method and one
recent top-journal method. Both expose reusable target-free representations
and lightweight source evaluations, while testing materially different
learning assumptions: fixed frame projections with latent mappings versus
learned adaptive transformations with multi-domain bootstrap learning. The
existing raw, recurrent, handcrafted, and hybrid task-specific models remain
contextual comparators.

This is a targeted recent representation-learning comparison, not an
exhaustive SOTA survey. Two direct baselines are sufficient for the approved
FYP scope because they span different publication venues and materially
different SSL assumptions, and because the project already retains matched
task-specific and non-learned contextual baselines. The final report must use
that bounded wording rather than claim comprehensive coverage of recent
methods.

`SISSEL-Frozen` and `TimeDART-Frozen` are peer optional extensions, not Phase
6.7 exit conditions and not substitutes for either required method. The owner
commissioned TimeDART's post-core preparation after the required LWA/SaURL
execution; its implementation remains gated on the separate decision record.
SISSEL remains uncommissioned pending mentor review. Either optional study is
explicitly post-core exploratory evidence rather than part of the
confirmatory core. Before implementation or execution, its source, licence,
architecture, extraction point, hardware limit, training budget, and complete
two-walk/three-task inventory must be frozen; its complete matrix must be
preserved and reported. Optional results may extend the discussion but cannot
retroactively replace a core method, modify a core trajectory, or redefine the
primary Phase 6.7 conclusion.

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
- source-derived architecture and optimizer settings, or a frozen documented
  reconstruction when direct source reuse is prohibited;
- expected parameter count, GPU-memory demand, and a CPU/CUDA smoke-test
  result on synthetic and small real training-only batches;
- any deviation required by the 64-by-5 OHLCV input; and
- the final decision to admit, reserve-replace, or reject the candidate.

The intended extraction points, subject to the candidate-specific source or
reconstruction contract, are:

- `LWA-F`: the source-defined frozen time representation and learned
  time-to-Fourier and time-to-Gabor representation mappings, concatenated as
  the source-defined 384-dimensional inference representation, with auxiliary
  encoders and projectors removed as specified by the source evaluation path;
  and
- `SAURL-F`: the approved reconstruction's 128-dimensional RwAM-combined time,
  frequency, and cross-domain representation, excluding pretraining
  projectors/predictors.

If the optional extensions are admitted, their intended extraction boundaries
must be frozen by separate method-specific dossiers. `TD-F` was initially
expected to use pooled causal encoder patch states with the diffusion decoder
removed after pretraining. The audited source instead removes the causal mask
during downstream transfer; the owner-approved TimeDART dossier uses that
path with per-channel pooling and source-used instance statistics. This
method-specific extraction replaces the initial expectation.
`SISSEL-F` requires its own independent paper/source clarification
process before its common frozen representation is selected; the current
paper/code audit is feasibility evidence, not an implementation contract.

The feasibility audit may inspect training rows, shapes, metadata, source
code, runtime, and resource use. It may not inspect downstream evaluation
metrics or use evaluation targets to choose a model, extraction point, width,
or hyperparameter.

Each method's complete task/walk inventory must be frozen before that method's
first downstream execution. Staged SaURL execution is permitted from
`saurl_downstream_seed0.json`; the later full core manifest must incorporate
those immutable SaURL runs alongside LWA rather than modifying or rerunning
them in response to their metrics.

The LWA Lumid execution manifest is likewise limited to its six new
task/walk trajectories. LWA head training and replay do not require H0
feature/checkpoint artifacts to be copied into the remote sandbox. The six
already completed immutable H0 references remain part of the final core
comparison and are joined and replayed when assembling/reporting that core
matrix; they are not rerun as part of LWA execution.

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

For the required two-model core:

```text
2 methods x 2 walks = 4 new encoder trajectories
```

For SaURL-TS, one encoder trajectory includes its alternating SaDA and SaSSL
updates; SaDA is not counted as a separate downstream model. Each admitted
optional method adds two encoder trajectories; admitting both adds four.

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

For the paper-guided SaURL-TS adaptation, Questions 4--11 were approved by the
project owner on 2026-10-04. The frequency-view path is selected
before implementation and evaluation: apply `rfft` over the 64-step time
axis, transform magnitude only with frequency SaDA, preserve the original
phase, recombine magnitude and phase, and apply `irfft(..., n=64)` before the
real `[B,64,5]` view enters `E_F`. At frozen inference, pass the normalized
real input directly to `E_F`; an unchanged FFT/inverse-FFT round trip is
mathematically redundant. The selected adapter additionally uses global
maximum branch pooling; an eight-region RwAM with shared
`Conv1d(3,1,1) -> ReLU -> Conv1d(1,3,1)` average/max paths; temporal and
reconstructed-frequency views as the cross-domain BYOL pair; separate
temporal/frequency SaDA modules with a shared factorizer but view-specific
transform heads; a shared deterministic paper-threshold mask per domain with
a straight-through gradient estimator and no logistic/Gumbel mask noise; six
hidden-64 dilated residual blocks; 128-wide representations/projectors/
predictors; EMA `0.99`;
and parameter-isolated alternating updates in which SaDA updates first every
two minibatches and SaSSL updates every minibatch. This is an independently
authored reconstruction, not an official reproduction.
The normative input-domain five-kernel MMD, view bundles, gradient boundaries,
optimizer order, no-drop batching, checkpoint state, and extraction pseudocode
are frozen in
`docs/baselines/SaURL_TS/upstream_clarification_request.md`; implementation and
comparison details are frozen in the sibling implementation and integration
documents.

Every trajectory stores configuration, source version, data and row hashes,
random seed, checkpoints, loss history, parameter count, elapsed time, peak
memory, and replay metadata. Training collapse or non-finite behaviour is a
reportable result; it is not permission to introduce an unplanned model.

## 7. Stage C: frozen features and downstream matrix

Freeze each epoch-50 encoder and extract one row embedding for every existing
downstream identity. This produces:

```text
2 methods x 2 walks = 4 new representation stores
```

Each admitted optional method adds two representation stores; admitting both
adds four.

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
2 representations x 3 tasks x 2 walks = 12 new downstream trajectories
12 trajectories x 3 snapshots = 36 evaluated snapshots
```

Six immutable epoch-50 `H0` task/walk trajectories are the direct references,
giving an 18-trajectory core representation-comparison table. Each admitted
optional method adds six downstream trajectories and 18 snapshots; admitting
both produces a 30-trajectory expanded table. Existing raw-input,
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
  encoder_pretraining/{lwa_frozen,saurl_frozen,sissel_frozen_optional,timedart_frozen_optional}/
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
2. Complete the provenance/no-upstream-code-reuse/API/hardware feasibility
   manifest for the mandatory LWA/SaURL core without using downstream metrics
   to change either method.
3. Freeze the core roster and SaURL extraction point, architecture settings,
   budget, paths, and six-run staged matrix.
4. Implement the explicitly documented independent SaURL adapter and pass CPU/CUDA shape,
   forward/backward, determinism, provenance-failure, and small-batch tests.
5. Train and replay both SaURL encoder trajectories.
6. Extract and replay both SaURL frozen representation stores.
7. Freeze the 12-entry staged SaURL manifest containing six SaURL runs and six
   immutable `H0` references. SaURL execution may proceed at this point.
8. Implement, train, extract, and independently freeze LWA without using
   SaURL downstream metrics to change its contract.
9. Assemble the 18-trajectory core manifest by incorporating the immutable
   SaURL runs, six LWA runs, and six `H0` references. Execute every remaining
   LWA trajectory without evaluation-driven truncation. Preserve all 5/15/50
   SaURL snapshots already produced by the staged run.
10. Replay checkpoints, predictions, metrics, scalers, row identities, and
   representation hashes.
11. Generate the complete per-walk, pooled, resource, and adaptation report.
12. TimeDART's eleven substantive decisions and model-only implementation are
   complete. The owner will evaluate the model before authorizing any further
   experiment infrastructure. If that work later resumes, execute and replay
   the complete post-core two-walk/three-task extension. Mentor review still
   decides whether SISSEL should be commissioned separately. No intermediate
   optional result may truncate either admitted matrix.

## 12. Exit conditions and handoff

Phase 6.7 is complete only when:

- the core roster contains both LWA-Frozen and SaURL-TS-Frozen;
- all four core encoder trajectories and four core representation stores pass
  replay;
- all 12 new core downstream trajectories and six immutable `H0` references pass
  standalone prediction and metric replay;
- SISSEL-Frozen and TimeDART-Frozen remain optional post-core extensions and
  do not block core completion; if either is later admitted, its two encoders,
  two stores, and six downstream trajectories must pass replay and reporting
  before claims about that extension are made;
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
