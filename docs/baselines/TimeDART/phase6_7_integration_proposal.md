# Proposal: integrate TimeDART-Frozen into Phase 6.7

**Method ID:** `timedart_frozen`  
**Reporting label:** `TimeDART (frozen encoder; independent implementation)`  
**Current gate:** all method decisions, the independent model, and the full
experiment pipeline are implemented. The owner will verify the scripts before
experiment execution.

## 1. Role and claim boundary

TimeDART is an optional post-core exploratory extension. It adds a
causal-denoising reconstruction method to the completed comparison between
canonical H0, SaURL-TS-Frozen, and LWA-Frozen. Its results may extend the
discussion but cannot replace a required core method or alter the core result.

The admissible conclusion is local: under identical rows and common probes,
TimeDART-Frozen was stronger or weaker than H0 on a named task and walk. This
is not a reproduction of the ICML fine-tuning tables or a state-of-the-art
claim.

## 2. Comparison contract

| Walk | Permitted training interval | Evaluation interval | Encoder rows |
|---:|---|---|---:|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` | 39,070 |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` | 58,473 |

Train separate TimeDART weights for each walk using only
`encoder_train_sequences [N,64,5]`. Movement, price, and realised-variance
targets are unavailable to pretraining. Walk 2 cannot update or reinterpret
Walk 1.

## 3. Proposed frozen adaptation

| Item | Proposed value |
|---|---|
| Variant | original compact TimeDART |
| Input | accepted `[B,64,5]` with already scaled volume; row-local instance norm only |
| Channels | shared channel-independent encoder |
| Patch | length/stride 2, 32 patches |
| Encoder | 2 layers, width 32, 8 heads, FFN 64 |
| Decoder | 1 layer, self-only masks |
| Corruption | 1,000-step cosine, independent step per channel/patch |
| Timestep input | none, preserving paper/source behavior |
| Projector | paper-aligned patchwise `32 -> 2` |
| Optimizer | Adam `1e-3`, no weight decay |
| Scheduler | exponential gamma `0.95` |
| Batch | physical 16, drop last |
| Budget | 50 epochs, snapshots 5/15/50, seed 0 |
| Extraction | unmasked clean encoder; per-channel patch max; O/H/L/C/V concat; append five row means and five row stds |
| Output | owner-approved 170 dimensions |

Every difference from the paper/source is disclosed: project length 64,
fixed finance-oriented settings, seed 0, fixed-final checkpoint, no
validation, paper-over-source projector resolution, no extra input scaler,
and generic frozen pooling plus source-used instance statistics. The owner
approved the encoder path and extraction on 2026-10-05.

## 4. Frozen feature stores

Build one master store per walk:

```text
experiments/phase6_7/features/timedart_frozen/walk1/representations.npz
experiments/phase6_7/features/timedart_frozen/walk2/representations.npz
```

Each store contains exact task-prefixed train/evaluation representations and
identity fields for:

- `classification_h2`;
- `absolute_price_h8`; and
- `realised_variance`.

The manifest records checkpoint/config hashes, the accepted input bundle's
volume-scaler provenance, ordered row hashes, the approved 170-wide output,
coordinate slices, finiteness, device, elapsed time, and feature hashes. A
failure to represent any established row fails the store;
it cannot shrink the comparison population.

## 5. Downstream matrix

TimeDART adds exactly:

```text
2 walks x 3 tasks = 6 trajectories
6 trajectories x epochs {5,15,50} = 18 snapshots
```

Reuse the exact H0/SaURL/LWA task identities, train-only embedding scaler,
head family and hidden width, loss, target transform, optimizer, learning
rate, batch size, seed, metrics, and non-learned references. Only native input
width changes to 170.

Primary comparisons are against immutable H0. SaURL and LWA remain completed
core references. Existing raw and task-specific models remain contextual
complete-system comparators.

## 6. Fairness and leakage safeguards

- no target or evaluation row enters the encoder trajectory or the already
  frozen walk-volume scaler;
- no validation split, early stopping, or best-loss checkpoint is used;
- patch/width/pooling/projector choices are frozen before implementation;
- epoch 50 is fixed before downstream evaluation;
- task coordinate scaling fits only supervised training embeddings;
- all six runs and all snapshots remain reportable, including collapse or
  negative results; and
- resource failure stops the candidate rather than triggering silent tuning.

## 7. Artifact plan

```text
docs/baselines/TimeDART/
src/baselines/timedart/
tests/baselines/timedart/
src/training/phase6_7_timedart.py
src/features/phase6_7_timedart_features.py
scripts_v6/
experiments/phase6_7/
  feasibility/timedart/
  manifests/timedart_*.json
  encoder_pretraining/timedart_frozen/walk{1,2}/seed0/
  features/timedart_frozen/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/timedart_frozen/
  reports/timedart_staged_seed0/
```

Large checkpoints and arrays remain Git-ignored. Small manifests and replay
evidence remain versioned where existing project policy permits.

## 8. Admission checklist

- [x] Paper identity, venue, and local hash verified.
- [x] Official source and exact commit audited.
- [x] Missing software licence recorded.
- [x] `[B,64,5]` CPU forward/backward feasibility verified.
- [x] Owner-approved extraction is 170 coordinates after auditing source
  instance denormalization.
- [x] Complete two-walk/three-task inventory proposed.
- [x] All eleven owner decisions recorded.
- [x] Question 7 terminology clarified; no new mechanism introduced.
- [x] Questions 3 and 9 resolved before implementation.
- [x] Independent model and full experiment pipeline implementation complete;
  awaiting owner verification before execution.
- [x] Fifteen focused CPU tests and a real-row smoke pass.
- [ ] CUDA/resource gate passes on the selected runtime.
- [ ] Two encoder trajectories replay.
- [ ] Two feature stores replay.
- [ ] Six downstream trajectories and 18 snapshots replay.
- [ ] TimeDART extension report generated.
