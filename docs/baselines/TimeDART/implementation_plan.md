# TimeDART Phase 6.7 implementation plan

**Status:** Stages 0 and 1 complete; stopped for owner evaluation before Stage 2  
**Execution authority:** no trajectory training authorized by this document  
**Method:** independently authored TimeDART with frozen-encoder probe contract

The work follows the manual review gates used for SaURL and LWA.

## Stage 0 — freeze provenance and decisions

1. Resolve Questions 3 and 9 in `upstream_clarification_request.md`; all
   eleven owner decisions and the Question 7 terminology clarification are
   recorded.
2. Synchronize the approved values across this dossier and source manifest.
3. Preserve the no-upstream-code-reuse boundary.
4. Freeze the exact two-walk/six-run inventory before implementation.
5. Do not use existing downstream metrics to alter the TimeDART contract.

**Exit:** complete.

## Stage 1 — independently implement the model only

**Status:** complete. The independent model package is present under
`src/baselines/timedart/`. Ten focused CPU tests pass. A forward/backward
probe on two accepted Walk 1 encoder rows produces finite reconstruction,
gradients, and `[2,170]` frozen features. The implementation has 30,242
pretraining parameters and retains 17,248 encoder parameters after discarding
the decoder and projector. These counts are lower than the active source
because the paper-aligned patchwise projector replaces its large flattening
head and this implementation omits source layers unused in the forward path.

Create:

```text
src/baselines/timedart/
  __init__.py
  config.py
  patching.py
  diffusion.py
  attention.py
  model.py
```

Suggested responsibilities:

- `config.py` — immutable input/patch/model/noise settings and validation;
- `patching.py` — channel independence, non-overlapping patches, positions,
  and shifted SOS construction;
- `diffusion.py` — registered cosine schedule buffers and independent
  corruption;
- `attention.py` — independent Transformer encoder/decoder blocks and exact
  causal/self-only masks; and
- `model.py` — pretrainer plus decoder-free frozen extractor.

Focused tests under `tests/baselines/timedart/` must cover:

- exact shapes for `[B,64,5]`;
- patch length equals stride and complete coverage of 64 timestamps;
- shifted clean-target direct-attention isolation with hand-constructed
  already-normalized patches, plus a test demonstrating the limitation from
  whole-window instance normalization;
- causal and self-only mask truth tables;
- one independently sampled noise step per channel/patch;
- cosine-buffer state-dict/device behavior;
- shared clean/noisy patch embedding;
- the approved projector semantics;
- finite forward/backward loss and gradients;
- extractor width/slice ordering and decoder exclusion;
- unmasked versus causal extraction behavior as approved; and
- deterministic eval-mode save/load replay.

Do not implement training orchestration or feature stores in Stage 1.

**Manual review gate:** ready for owner evaluation. Work stops here per the
owner's 2026-10-05 instruction; Stages 2--6 remain unimplemented.

## Stage 2 — implement gated walk-specific pretraining

Create `src/training/phase6_7_timedart.py` and scripts:

```text
scripts_v6/audit_phase6_7_timedart.py
scripts_v6/bootstrap_phase6_7_timedart.py
scripts_v6/validate_phase6_7_timedart.py
```

Required behavior:

1. replay the accepted walk dataset and exact target-free population;
2. consume the accepted walk input directly, preserving its upstream
   training-interval volume-scaler provenance; use only the model's own
   per-instance/channel normalization;
3. run manifest-only CPU shape/backward checks by default;
4. require an admitted CUDA/resource manifest before `--execute`;
5. train one seeded shuffled fixed-budget trajectory per walk;
6. save complete epochs 5/15/50 atomically;
7. record dropped remainders, losses, time, memory, parameter counts, gradient
   health, RNG/sampler, configuration, data, and source hashes;
8. resume only from a complete matching state; and
9. replay every snapshot independently on CPU and the training device;
   log strict elementwise cross-device misses as warnings when relative L2 is
   at most `5e-4` and cosine is at least `0.999999`, while keeping corrupted
   artifacts, identity/provenance mismatches, non-finite output, and material
   drift fatal so the Lumid launcher stops on invalid state.

The bootstrap must never train without `--execute`. It must not create a
validation set or select a checkpoint from loss.

**Manual review gate:** data/scaler provenance, training order, checkpoint
state, resume behavior, and resource admission.

## Stage 3 — implement two master feature stores

Create:

```text
src/features/phase6_7_timedart_features.py
scripts_v6/prepare_phase6_7_timedart_features.py
scripts_v6/validate_phase6_7_timedart_features.py
```

For each walk:

1. load the final epoch-50 retained extractor and accepted task input bundle;
2. extract all exact task train/evaluation contexts in batches;
3. save task-prefixed representations at the owner-approved native width plus
   identity fields (170 coordinates under the revised Question 9 proposal);
4. assert O/H/L/C/V slice boundaries and no decoder/noise construction; and
5. replay checkpoint and input-bundle hashes, ordered identities, finiteness, width,
   feature content, batching invariance, and CPU/CUDA tolerance.

**Manual review gate:** extraction boundary and row alignment.

## Stage 4 — extend native-width common probes

Update `src/training/phase6_7_downstream.py` carefully:

- add `timedart_frozen: 170` to the native-width registry if the revised
  Question 9 proposal is approved;
- replace the current SaURL-versus-LWA binary loader branch with explicit
  method dispatch so completed methods remain immutable;
- reuse the generic training-only coordinate standardizer and task heads; and
- add focused loader/config/replay tests for all three methods.

Create:

```text
scripts_v6/bootstrap_phase6_7_timedart_downstream.py
scripts_v6/validate_phase6_7_timedart_downstream.py
```

Freeze a six-entry TimeDART-only execution manifest before the first head is
trained. The script remains manifest/smoke-only without `--execute`.

**Manual review gate:** six-run matrix, method dispatch, and unchanged prior
artifacts.

## Stage 5 — implement a durable launcher

Create `scripts_v6/run_phase6_7_timedart_experiment.sh` only after Stages 1--4
are reviewed. It should write persistent timestamped logs, stop at the first
failed gate, and safely validate/reuse completed artifacts.

Proposed command order:

```bash
.venv/bin/python3 scripts_v6/audit_phase6_7_timedart.py --device cpu
.venv/bin/python3 scripts_v6/audit_phase6_7_timedart.py \
  --device cuda --admit-training \
  --max-peak-memory-gib <GIB> --max-smoke-seconds <SECONDS>
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_timedart.py --device cpu
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_timedart.py --device cuda --execute
.venv/bin/python3 scripts_v6/validate_phase6_7_timedart.py
.venv/bin/python3 scripts_v6/prepare_phase6_7_timedart_features.py --device cuda
.venv/bin/python3 scripts_v6/validate_phase6_7_timedart_features.py --device cuda
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_timedart_downstream.py --device cpu
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_timedart_downstream.py --device cuda --execute
.venv/bin/python3 scripts_v6/validate_phase6_7_timedart_downstream.py
```

## Stage 6 — report the complete extension

Report:

- every per-walk and pooled task metric at 5/15/50;
- paired differences from immutable H0 on identical rows;
- classification collapse/per-class evidence;
- price persistence and implied-movement diagnostics;
- volatility references, contract-macro, tail, and subgroups;
- width and common-head parameter counts;
- pretraining, extraction, probe time, peak memory, and parameter counts;
- paper/source/project deviations; and
- seed-0, two-walk, post-core exploratory, non-trading boundaries.

Negative or collapsed results remain in the report.

## Definition of done

- all method decisions are resolved and synchronized;
- no upstream source code has been copied;
- independent model tests pass;
- CUDA/resource admission passes before training;
- both walk trajectories and all 5/15/50 checkpoints replay;
- both stores replay exact task identities and the approved native width;
- all six downstream trajectories and 18 snapshots replay; and
- the complete TimeDART extension report is generated without changing the
  completed core comparison.
