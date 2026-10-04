# Proposal: integrate Learning Without Augmenting into Phase 6.7

**Method ID:** `lwa_frozen`
**Reporting label:** `LWA-Frozen (paper-guided independent implementation)`
**Current gate:** architecture and runtime admission passed. Both full caches
and the complete Walk 1 two-stage trajectory exist; deployment of the
severity-aware replay amendment and resume into Walk 2 are next.

## 1. Decision rule

Admit LWA as the remaining mandatory Phase 6.7 core method if:

- the implementation matches the approved paper/source resolutions without
  consulting downstream results;
- the independently authored adapter passes tensor, gradient, determinism,
  dependency, and checkpoint smoke tests;
- authoritative physical batch 128 completes a forward/backward smoke on the
  selected professor-provided container;
- both walks can build replayable training-only CWT/FFT caches; and
- the final frozen extractor returns one finite 384-dimensional vector for
  every required row.

If this gate fails, record an LWA feasibility failure. Do not substitute
SISSEL, TimeDART, or another method implicitly.

## 2. Comparison contract

LWA uses the same two global-calendar walks as H0 and SaURL:

| Walk | Training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Each walk has independent input-scaler, LWA encoder, transformed-domain
encoders, projectors, embedding mappers, representation mappers, optimizers,
and RNG state. Walk 2 cannot update or select Walk 1.

LWA pretraining consumes only the target-free `encoder_train` population:

| Walk | Rows | Shape |
|---:|---:|---|
| 1 | 39,070 | `[39070,64,5]` |
| 2 | 58,473 | `[58473,64,5]` |

Movement, price, and realised-variance targets are unavailable to both LWA
training stages.

## 3. Data preparation

### Walk-local input scaling

Fit one coordinatewise mean/std scaler over all `(row,time)` observations in
the target-free walk population. Store its values and the exact identity hash.
Apply it before every time/Fourier/CWT view operation.

### Training-only transformed views

- Fourier cache: complex orthonormal `rfft` on the 64-step axis, shape
  `[N,5,33]`.
- CWT cache: magnitude of `cmor1-1` coefficients at 48 log-spaced scales,
  shape `[N,5,48,64]`.

The cache manifest records input/scaler/source hashes, transform library and
version, wavelet/scales, dtype, shape, row order, finiteness, and content hash.
Cached views are deterministic preprocessing, not learned augmentations.

No transformed view is required for downstream extraction because the
efficient frozen path retains only the time encoder and representation
mappers.

## 4. Frozen adaptation

The project owner approved and froze:

| Item | Proposed value |
|---|---|
| Input | channel-standardized `[B,64,5]` |
| Time encoder | eight-block source-style ResNet1D, output 128 |
| Fourier encoder | magnitude/phase source topology with length-9 reductions, output 128 |
| Wavelet encoder | source multiscale 2D topology with `8 -> 1`, `6 -> 1` reductions, output 128 |
| Projectors | three `128 -> 128 -> 128` ReLU MLPs |
| NT-Xent | three symmetric domain pairs, temperature `0.15` |
| Embedding mappings | two convolutional `1 -> 64 -> 1` mappings |
| Joint loss | three NT-Xent terms plus paper-reduced two-way L1 mapping loss |
| Joint optimizer | Adam, lr `3e-3`, wd `1e-6`, cosine decay |
| Representation mappings | two newly initialized convolutional `1 -> 64 -> 1` mappings |
| Mapper optimizer | two Adam optimizers, lr `3e-3`, wd `1e-6`, constant lr |
| Budget | 50 joint epochs followed by 50 mapper epochs |
| Batch | physical batch 128 for both walks; `drop_last=True`; no sweep or automatic fallback |
| Extraction | final joint epoch 50 + mapper epoch 50 |
| Output | `[h_t, mapped_h_F, mapped_h_W]`, width 384 |

This is a project adaptation, not an exact paper reproduction. It changes the
reported 256-epoch and batch-1024 setting, generalizes length-specific layers,
uses the project's fixed final checkpoint, and follows explicitly approved
paper/source resolutions.

## 5. Checkpoint and replay contract

Joint checkpoints at epochs 5, 15, and 50 contain:

- all three encoders, projectors, and embedding mappings;
- input scaler and transform-cache provenance;
- optimizer/scheduler, epoch, sampler, and RNG states;
- source/adaptation configuration and data hashes;
- fixed probe inputs plus expected domain representations/projections; and
- loss/resource histories.

Mapper checkpoints at epochs 5, 15, and 50 additionally contain the two
representation mappings and their optimizer states, while referring to the
immutable joint epoch-50 checkpoint hash.

The extractor checkpoint is the pair `(joint_e50, mapper_e50)`. Replay must
reject missing or mismatched halves.

Checkpoint replay distinguishes numerical warnings from critical failures.
The preferred result remains the strict elementwise check at `rtol=1e-5` and
`atol=1e-6`. A CPU/CUDA probe that misses that strict check is retained as a
non-fatal warning when relative L2 is at most `5e-4` and cosine similarity is
at least `0.999999`. Every domain/stage/epoch diagnostic is written to
`replay_validation.json`, and execution continues. Missing or corrupt
artifacts, provenance/hash/configuration mismatches, shape/type mismatches,
non-finite values, or drift outside the scale-aware bounds remain critical and
stop the pipeline.

## 6. Representation stores

Build one LWA master store per walk. For each established task bundle, map its
exact train/evaluation sequences through the frozen input scaler, time encoder,
and two representation mappers. Save the 384-wide result and task-prefixed
identity fields.

The store must prove:

- joint and mapper checkpoint hashes match the frozen manifest;
- input scaler and slice order match;
- output is finite and exactly width 384;
- every established task identity is present once in the established order;
- repeated extraction reproduces the feature hash under the accepted device
  tolerance; and
- FFT/CWT and auxiliary encoders are not invoked during extraction.

## 7. Downstream matrix

LWA adds exactly six trajectories:

```text
2 walks x 3 tasks = 6 trajectories
6 trajectories x epochs {5,15,50} = 18 snapshots
```

Tasks:

- two-hour `DOWN/STABLE/UP` classification at `tau=0.001`;
- absolute `close[t+8]` prediction plus implied-movement diagnostics; and
- realised variance over `(t,t+8h]` from raw future probability changes.

For each task/walk, reuse the exact H0/SaURL row identities, label bundle,
training-only feature standardizer semantics, head family and hidden width,
loss, target transform, optimizer, learning rate, batch size, seed, snapshot
budgets, metrics, and non-learned references. Only the input width changes to
384. No supervised bottleneck is inserted.

## 8. Fairness and leakage safeguards

- No target enters either LWA training stage.
- CWT/FFT caches contain only the walk's target-free training population.
- No evaluation metric selects mapper width, loss reduction, batch size,
  checkpoint, or retry.
- Resource fallback uses OOM/peak-memory evidence only.
- The final joint and mapper epoch-50 states are fixed before feature
  extraction.
- Downstream coordinate scaling fits only the task's supervised training
  embeddings and is frozen for evaluation.
- LWA must return features for every common row; failures do not shrink the
  comparison population.
- SaURL results cannot alter LWA decisions, even though SaURL completed first.

## 9. Artifact plan

```text
docs/baselines/LWA/
src/baselines/lwa/
tests/baselines/lwa/
experiments/phase6_7/
  feasibility/lwa/
  manifests/
  encoder_pretraining/lwa_frozen/walk{1,2}/
    transforms/
    joint/
    mappers/
  features/walk{1,2}/lwa_frozen.npz
  downstream/{classification_h2,absolute_price_h8,realised_variance}/
  reports/frozen_representation_seed0/
scripts_v6/
```

Large CWT caches, checkpoints, and feature arrays remain local/Git-ignored.
Small manifests, configurations, summaries, and replay evidence are tracked.

## 10. Risks and planned responses

| Risk | Response |
|---|---|
| No source licence | independent implementation only |
| Paper/source mapper conflict | owner freezes one interpretation before code |
| Fixed batch 128 fails on selected container | stop the gate and request an explicit amendment; do not fall back automatically |
| CWT cache consumes about 5.6 GiB total | free-space check, atomic cache build, hashes, optional safe chunking |
| PyWavelets runtime drift | use the verified `PyWavelets==1.8.0` pin and record it in every runtime/cache manifest |
| Two-stage budget doubles passes | disclose 50+50 and time both stages separately |
| Mapping/contrastive scale instability | finite/gradient diagnostics; record failure rather than tune from evaluation |
| BatchNorm/dropout replay drift | evaluation-mode fixed probes; strict misses inside the frozen relative-L2/cosine bounds are recorded warnings, while material or structural mismatch remains fatal |
| Negative downstream outcome | retain and report all six trajectories |

## 11. Admission checklist

- [x] Paper identity and version verified.
- [x] Official repository and exact commit audited.
- [x] Software-licence absence recorded.
- [x] 384-dimensional extraction boundary identified.
- [x] Length-64 tensor adaptation derived.
- [x] Static storage/parameter feasibility estimated.
- [x] Owner decisions 1--12 approved.
- [x] Independent model implementation complete and owner-reviewed.
- [x] Cache/pretraining, frozen-feature, and downstream launch/replay code is
  implemented and approved for execution.
- [x] Both full transform caches build and replay.
- [x] CPU forward/backward/determinism tests pass (the original 17 model tests
  pass locally and remotely; all 24 local tests include pipeline and
  severity-aware replay contracts).
- [x] Fixed batch-128 correctness/resource smoke passes on the selected container.
- [ ] Both walk trajectories replay.
- [ ] Both master stores replay.
- [ ] Six downstream trajectories replay.
- [ ] Full H0/SaURL/LWA core report generated.
