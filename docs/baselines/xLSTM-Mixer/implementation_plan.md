# xLSTM-Mixer implementation and execution plan

**Status:** paper/source audit complete; implementation not started

**Authority:** Phase 6.9 for model, artifacts, execution, and cross-task report

## Stage 0 — resolve the source contract

**Current status:** decisions 1--4 and 6--14 approved; initial-token decision
5 remains pending.

Complete all decisions in `upstream_clarification_request.md`, then update:

- `source_manifest.json`;
- Phase 6.9's architecture and training-freeze text;
- the relevant project skills/document pointers; and
- one immutable implementation configuration.

View semantics, RevIN, dependency/licence handling, project training settings,
and artifact ownership are resolved. No model code or runtime installation
should begin until the initial-token count is confirmed.

## Stage 1 — audit and build the full-path data contract

Create reusable logic under:

```text
src/data_processing/phase6_9_xlstm_mixer.py
```

and proposed scripts under a new phase-generation directory:

```text
scripts_v8/prepare_phase6_9_xlstm_mixer_data.py
scripts_v8/validate_phase6_9_xlstm_mixer_data.py
```

Required order:

1. inventory immutable existing price identities and accepted source hashes;
2. reuse the strict Phase 6 observed-path identity logic to inspect future
   timestamps, contract/segment identity, observed/imputed flags, and
   finiteness;
3. freeze the already demonstrated common full-path intersection;
4. build split-local `[8,5]` targets without crossing train/evaluation
   boundaries;
5. reuse the existing walk-training-only volume transform for target volume;
6. save targets, volume-transform provenance, ordered identities, and
   source/procedure hashes; and
7. independently replay every row against accepted source data.

Preparation may write data artifacts but may not train a model. It must not
print or aggregate target values. This is a bounded metadata join and target
construction step, not a new horizon-selection experiment.

**Manual review gate:** row/contract attrition, segment continuity, target
observation policy, upstream volume-transform provenance, and exact
H0/Raw-LSTM matching need.

## Stage 2 — implement the minimal model adapter

Create:

```text
src/baselines/xlstm_mixer/
tests/baselines/xlstm_mixer/
```

The adapter should include only:

- approved non-affine RevIN;
- shared NLinear and up/down projections;
- owner-confirmed learned token handling;
- the pinned sLSTM stack/backend boundary;
- the exact approved reverse-view operation; and
- configuration/state-dict serialization.

Do not copy data loaders, Lightning CLI, unrelated baseline models, W&B
plumbing, or source checkpoint selection.

Focused CPU tests must cover every item in
`architecture_and_dataflow.md` Section 8, plus state-dict round trips,
finite gradients, deterministic fixed probes, and rejection of wrong
configuration/source hashes.

**Manual review gate:** paper/source crosswalk, licence notices, reversal axis,
normalization math, initialization, and parameter sharing.

## Stage 3 — persistent-runtime dependency and resource admission

Create an audit entry point:

```text
scripts_v8/audit_phase6_9_xlstm_mixer_runtime.py
```

It must record:

- Python, PyTorch, CUDA toolkit/runtime, compiler, GPU, memory, and compute
  capability;
- exact `xlstm`, `einops`, and other relevant package versions/hashes;
- xLSTM dependency licence and selected backend;
- a bounded vanilla-backend CPU shape/backward test where available;
- CUDA extension build log and fixed-probe forward/backward output;
- same-CUDA-backend checkpoint save/load and fixed-probe replay;
- physical batch-512 peak allocation and step time, or an owner-reviewed
  smaller fixed batch proposed before training; and
- full environment and code hashes.

The audit is manifest-only unless an explicit `--admit-training` flag is
provided. Admission thresholds and replay tolerances must be arguments and
written to the manifest. Repeating an audit may update nondeterministic
timing/memory evidence but may not silently change code, dependencies, batch,
backend, or thresholds.

The experiment runtime is the persistent Lumid Sandbox. Project Docker and
devcontainer definitions are development/workflow references and do not
replace this admission record.

**Manual review gate:** licence disposition, successful kernel build,
same-backend replay, and memory headroom.

## Stage 4 — implement gated walk-specific training

Create:

```text
src/training/phase6_9_xlstm_mixer.py
scripts_v8/bootstrap_phase6_9_xlstm_mixer.py
scripts_v8/validate_phase6_9_xlstm_mixer.py
```

The bootstrap is manifest/smoke-only by default. It trains only with an
explicit `--execute` flag and a matching data/runtime admission manifest.

Required behavior:

1. replay data, upstream volume-transform, source, code, dependency, and
   configuration hashes;
2. train one independent seed-0 trajectory per walk;
3. use the complete standardized `[8,5]` L1 objective;
4. save complete epochs 5/15/50 atomically;
5. never allocate validation rows or select a checkpoint;
6. record losses, gradient health, time, memory, batch/remainder order,
   parameter counts, RNG/sampler state, and environment;
7. resume only from a complete matching state; and
8. generate full-path predictions only after checkpoint replay passes.

Canonical artifact root:

```text
experiments/phase6_9/xlstm_mixer/
```

**Manual review gate:** complete checkpoint state, fixed epoch semantics,
training-only transforms, and cross-walk isolation.

## Stage 5 — execute any required matched controls

Stage 1 necessarily reduces the price row set. Extend the established
downstream/Raw-LSTM launchers with explicit `XM-H0-D0` and `XM-RL` methods on
the frozen intersection.

Controls use their existing architectures, seed, epochs, losses, and task
endpoint. They do not receive full-path auxiliary targets. Save them under the
xLSTM comparison root without overwriting broader-row historical artifacts.

**Manual review gate:** identical train/evaluation identities across all strict
comparators.

## Stage 6 — validate predictions and metrics

Standalone validation must reconstruct, without training:

- checkpoint model and optimizer state;
- fixed-probe model outputs;
- complete evaluation `[N,8,5]` predictions;
- accepted-unit output reconstruction after RevIN;
- extracted `close[t+8]` values;
- identity order and target alignment;
- all price and implied-movement metrics; and
- invalid output diagnostics.

Structural, provenance, identity, non-finite, missing-artifact, or same-runtime
replay failures are fatal. Cross-backend CPU/CUDA numerical equivalence is not
a completion gate.

## Stage 7 — integrate the Phase 6.9 report

Create one task-separated Phase 6.9 report under the frozen Phase 6.9
reporting path. The price section references the canonical xLSTM-Mixer
artifact hashes and includes:

- both walks and pooled views where already valid;
- epoch 5/15/50 trajectories with epoch 50 primary;
- matched H0, Raw LSTM, persistence, and reversal results;
- paired prediction differences on exact rows;
- subgroup and invalid-path diagnostics;
- parameter/time/memory/latency costs;
- extra supervision and every paper/source/project deviation; and
- seed-0, two-walk, non-trading, non-universal claim boundaries.

Do not present the superseded Phase 6.6B listing as a replication.

## Proposed launcher order

Only after Stage 0 approval:

```bash
.venv/bin/python3 scripts_v8/prepare_phase6_9_xlstm_mixer_data.py
.venv/bin/python3 scripts_v8/validate_phase6_9_xlstm_mixer_data.py
.venv/bin/python3 scripts_v8/audit_phase6_9_xlstm_mixer_runtime.py --device cpu
.venv/bin/python3 scripts_v8/audit_phase6_9_xlstm_mixer_runtime.py \
  --device cuda --admit-training \
  --max-peak-memory-gib <GIB> --max-smoke-seconds <SECONDS>
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_xlstm_mixer.py --device cpu
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_xlstm_mixer.py \
  --device cuda --execute
.venv/bin/python3 scripts_v8/validate_phase6_9_xlstm_mixer.py
```

These commands are planned interfaces, not currently implemented entry
points.

## Definition of done

- all owner decisions are approved and synchronized;
- data/full-path/upstream-volume-transform replay passes for both walks;
- licence/dependency and CUDA admission pass;
- focused model tests pass;
- both walk trajectories and all 5/15/50 checkpoints replay;
- complete full-path predictions and extracted price metrics replay;
- any required matched H0-D0 and Raw-LSTM controls replay on identical rows;
- Phase 6.9 owns the one canonical artifact set; and
- the final report preserves extra-supervision and bounded-claim disclosures.
