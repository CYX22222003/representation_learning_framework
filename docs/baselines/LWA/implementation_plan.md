# Learning Without Augmenting Phase 6.7 implementation plan

**Status:** architecture and experiment decisions approved; Stage 1 model and
focused CPU tests implemented on 2026-10-04 and awaiting owner review
**Execution authority:** none; bootstrap commands must remain manifest-only by
default
**Method:** independently authored `LWA-Frozen`

The work is staged to support the same manual review process used for SaURL.
Do not proceed from one stage to the next until the project owner reviews the
current stage.

## Stage 0 — freeze provenance and decisions

1. Preserve the approved responses in
   `upstream_clarification_request.md`.
2. Keep every dossier document synchronized with those approved values.
3. Keep `source_manifest.json` at `admitted_for_implementation: true` while
   training remains separately gated.
4. Record the no-source-code-reuse boundary in model file headers and the
   feasibility manifest.
5. Do not read SaURL downstream metrics to revise LWA.

**Review output:** documentation only.

## Stage 1 — implement only the model and relevant utilities

**Implementation status:** complete pending the manual review gate. The package
exists at `src/baselines/lwa/`; 16 local CPU tests pass with one dependency
skip, and all 17 pass in the PyWavelets-equipped remote runtime. Stages 2--6
remain unimplemented and no experiment has run.

Create an independent package:

```text
src/baselines/lwa/
  __init__.py
  config.py
  transforms.py
  blocks.py
  encoders.py
  mappers.py
  losses.py
  model.py
```

Suggested responsibilities:

- `config.py` — immutable architecture/view/loss settings and validation.
- `transforms.py` — orthonormal channelwise rFFT and explicit `PyWavelets` CPU
  CWT wrapper; no task logic.
- `blocks.py` — same-padded 1D residual block and multiscale 2D block.
- `encoders.py` — time, Fourier, and wavelet encoders with explicit length-64
  shapes.
- `mappers.py` — shared convolutional mapping primitive.
- `losses.py` — dynamic-batch symmetric NT-Xent and approved L1 reduction.
- `model.py` — joint pretraining model and frozen inference extractor.

Stage 1 includes only source-independent unit tests for:

- exact time/Fourier/CWT/domain shapes;
- transform axis and `norm="ortho"` behavior;
- finite forward/backward pass;
- residual and mapper length preservation;
- positive-pair and negative-mask construction;
- mapping-loss reduction on hand-computed tensors;
- final output width/slice order;
- inference exclusion of auxiliary transforms/encoders; and
- deterministic evaluation-mode output on a fixed CPU input.

Do not implement data manifests, training orchestration, feature stores, or
downstream execution in Stage 1.

**Manual review gate:** architecture, losses, and tensor semantics.

## Stage 2 — implement view-cache preparation and two-stage pretraining

Extend the existing Phase 6.7 encoder infrastructure without changing
completed SaURL artifacts.

Required behavior:

1. Load and replay the accepted walk bundle.
2. Fit the LWA input scaler only on `encoder_train_sequences`.
3. Build chunked float32, atomic FFT/CWT disk-backed caches with
   row/scaler/source/library-version hashes.
4. Create a deterministic shuffled loader with fixed physical batch 128 and
   `drop_last=True`; do not implement automatic batch fallback or gradient
   accumulation.
5. Train the joint model for the frozen budget; checkpoint 5/15/50.
6. Freeze the final joint encoder state.
7. Initialize and train representation mappers for the frozen second-stage
   budget; checkpoint 5/15/50.
8. Save complete histories, timing, peak memory, gradient/finiteness
   diagnostics, parameter counts, and RNG/sampler states.
9. Resume only from complete state with matching configuration/data/cache
   hashes.

The two stage histories must be separate. A mapper epoch is not relabelled as
an encoder epoch.

Proposed shared/source paths:

```text
src/training/phase6_7_external_encoders.py  # generalize dispatch carefully
src/training/phase6_7_lwa.py                # LWA-specific cache/two-stage logic
scripts_v6/audit_phase6_7_lwa.py
scripts_v6/bootstrap_phase6_7_lwa.py
```

SaURL's existing completed manifest and checkpoint semantics remain immutable.

**Manual review gate:** data/cache provenance, optimizer order, checkpoint and
resume state. No experiment runs before owner approval.

## Stage 3 — implement two LWA master stores

For each walk:

1. load the joint epoch-50 plus mapper epoch-50 checkpoint pair;
2. load the saved LWA input scaler;
3. map exact task train/evaluation sequences through the efficient inference
   path only;
4. concatenate the approved three 128-wide slices;
5. save one task-prefixed 384-wide master store; and
6. replay identities, checkpoint/scaler hashes, width, finiteness, and feature
   hashes.

Suggested paths:

```text
src/features/phase6_7_lwa_features.py
scripts_v6/prepare_phase6_7_lwa_features.py
```

Do not duplicate rows merely to make task-specific standalone stores; preserve
the existing Phase 6.7 master-store pattern.

**Manual review gate:** extraction boundary and row alignment.

## Stage 4 — integrate common native-width probes

Reuse the already implemented generic Phase 6.7 downstream behavior, extending
method dispatch and manifests to `lwa_frozen` while preserving SaURL runs.

For each task/walk:

- assert exact H0 task identities;
- fit the generic coordinate standardizer on LWA task-training embeddings;
- keep hidden width, head family, loss, target transform, optimizer, learning
  rate, seed, batch size, and snapshot budgets unchanged;
- instantiate only the first layer from declared input width 384; and
- save checkpoint, predictions, metrics, scaler, configuration, and row hashes
  at epochs 5/15/50.

Suggested orchestration:

```text
scripts_v6/bootstrap_phase6_7_lwa_downstream.py
```

The script is manifest/smoke-only without `--execute`.

**Manual review gate:** six-run matrix and immutable SaURL/H0 references.

## Stage 5 — feasibility and execution sequence

Only after Stages 1--4 are reviewed:

1. run dependency, disk, CPU tensor, and small real training-only smoke tests;
2. verify one forward/backward step at the authoritative physical batch 128;
   this is a correctness/resource check, not a batch search;
3. freeze `lwa_pretraining_seed0.json`;
4. train and replay Walk 1;
5. train and replay Walk 2;
6. extract and replay both LWA master stores;
7. assemble the 18-trajectory core manifest with immutable H0 and SaURL runs;
8. execute all six LWA downstream trajectories; and
9. replay every 5/15/50 head snapshot.

No optional baseline decision occurs inside this sequence.

## Stage 6 — complete core reporting

Generate the full H0/SaURL/LWA report with:

- per-walk and pooled task metrics;
- paired differences from H0 on identical rows;
- classification collapse/per-class tables;
- future-price persistence, implied-movement, rank-IC, and subgroup tables;
- volatility persistence, contract-macro, tail, and subgroup tables;
- embedding widths and probe parameter counts;
- joint-stage, mapper-stage, extraction, and downstream time;
- encoder/retained/total parameter counts and peak memory;
- source deviations and failed diagnostics; and
- the seed-0/two-walk/non-trading claim boundary.

Negative LWA results remain in the main core table.

## Proposed command contract

Names are provisional until code review:

```bash
# Read-only/source/cache/CPU feasibility. Never trains a trajectory.
.venv/bin/python3 scripts_v6/audit_phase6_7_lwa.py --device cpu

# Manifest and CPU smoke only.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_lwa.py --device cpu

# Explicit training only after review and CUDA/resource admission.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_lwa.py --device cuda --execute

# Frozen efficient-path feature extraction.
.venv/bin/python3 scripts_v6/prepare_phase6_7_lwa_features.py --device cuda

# Manifest/smoke, then explicit six-run execution.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_lwa_downstream.py --device cpu
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_lwa_downstream.py --device cuda --execute
```

Dedicated validation/audit entry points are intentionally omitted from this
key-script summary; the implementation must still provide replay functions
consistent with existing Phase 6.7 artifacts.

## Definition of done for LWA-Frozen

- approved dossier contains no unresolved architecture/training decision;
- `PyWavelets` is installed and its exact version is recorded in the runtime
  manifest;
- independent source boundary is documented;
- length-64 five-channel model passes focused tests;
- FFT/CWT caches replay from training-only rows;
- both two-stage walk trajectories pass checkpoint replay;
- both 384-wide master stores pass identity and feature replay;
- six downstream trajectories and all 18 snapshots pass standalone replay;
- the full mandatory core report includes H0, SaURL, and LWA; and
- no implementation or run was selected from downstream evaluation metrics.
