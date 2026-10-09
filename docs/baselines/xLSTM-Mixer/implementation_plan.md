# xLSTM-Mixer implementation and execution plan

**Status:** paper/source audit complete; implementation not started

**Authority:** Phase 6.6B for model/artifacts; Phase 6.9 for cross-task report

## Stage 0 — resolve the source contract

**Current status:** pending owner decisions.

Complete all decisions in `upstream_clarification_request.md`, then update:

- `source_manifest.json`;
- Phase 6.6B's architecture and training-freeze text;
- the relevant project skills/document pointers; and
- one immutable implementation configuration.

No model code or runtime installation should begin while view semantics,
RevIN, initial-token count, dependency/licence handling, or optimizer settings
remain open.

## Stage 1 — audit and build the full-path data contract

Create reusable logic under:

```text
src/data_processing/phase6_6_xlstm_mixer.py
```

and proposed scripts under a new phase-generation directory:

```text
scripts_v8/audit_phase6_6_xlstm_mixer_data.py
scripts_v8/prepare_phase6_6_xlstm_mixer_data.py
scripts_v8/validate_phase6_6_xlstm_mixer_data.py
```

Required order:

1. inventory immutable existing price identities and accepted source hashes;
2. inspect only future timestamps, contract/segment identity,
   observed/imputed flags, and finiteness;
3. freeze all rows or a common full-path intersection;
4. build split-local `[8,5]` targets without crossing train/evaluation
   boundaries;
5. fit each five-channel scaler from permitted walk-training candles only;
6. save targets, scaler, ordered identities, and source/procedure hashes; and
7. independently replay every row against accepted source data.

Preparation may write data artifacts but may not train a model. The audit must
not print or aggregate target values.

**Manual review gate:** row/contract attrition, segment continuity, target
observation policy, scaler population, and exact H0/Raw-LSTM matching need.

## Stage 2 — implement the minimal model adapter

Create:

```text
src/baselines/xlstm_mixer/
tests/baselines/xlstm_mixer/
```

The adapter should include only:

- non-affine or owner-approved RevIN;
- shared NLinear and up/down projections;
- owner-approved learned token handling;
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
scripts_v8/audit_phase6_6_xlstm_mixer_runtime.py
```

It must record:

- Python, PyTorch, CUDA toolkit/runtime, compiler, GPU, memory, and compute
  capability;
- exact `xlstm`, `einops`, and other relevant package versions/hashes;
- xLSTM dependency licence and selected backend;
- CPU vanilla forward/backward shape and fixed-probe output;
- CUDA extension build log and fixed-probe forward/backward output;
- CPU/CUDA state-dict compatibility and numerical drift;
- physical batch-128 peak allocation and step time, or an owner-reviewed
  smaller fixed batch proposed before training; and
- full environment and code hashes.

The audit is manifest-only unless an explicit `--admit-training` flag is
provided. Admission thresholds and replay tolerances must be arguments and
written to the manifest. Repeating an audit may update nondeterministic
timing/memory evidence but may not silently change code, dependencies, batch,
backend, or thresholds.

**Manual review gate:** licence disposition, successful kernel build,
numerical compatibility, and memory headroom.

## Stage 4 — implement gated walk-specific training

Create:

```text
src/training/phase6_6_xlstm_mixer.py
scripts_v8/bootstrap_phase6_6_xlstm_mixer.py
scripts_v8/validate_phase6_6_xlstm_mixer.py
```

The bootstrap is manifest/smoke-only by default. It trains only with an
explicit `--execute` flag and a matching data/runtime admission manifest.

Required behavior:

1. replay data, scaler, source, code, dependency, and configuration hashes;
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
experiments/phase6_6/recent_forecasting_baseline/
```

**Manual review gate:** complete checkpoint state, fixed epoch semantics,
training-only transforms, and cross-walk isolation.

## Stage 5 — execute any required matched controls

If Stage 1 retained every price row, reuse immutable comparator artifacts after
identity/hash replay. If Stage 1 reduced the row set, extend the established
downstream/Raw-LSTM launchers with explicit `XM-H0-D0` and `XM-RL` methods on
the frozen intersection.

Controls use their existing architectures, seed, epochs, losses, and task
endpoint. They do not receive full-path auxiliary targets. Save them under the
xLSTM comparison root without overwriting broader-row historical artifacts.

**Manual review gate:** identical train/evaluation identities across all strict
comparators.

## Stage 6 — validate predictions and metrics

Standalone validation must reconstruct, without training:

- checkpoint model and optimizer/scheduler/scaler state;
- fixed-probe model outputs;
- complete evaluation `[N,8,5]` predictions;
- inverse channel scaling;
- extracted `close[t+8]` values;
- identity order and target alignment;
- all price and implied-movement metrics; and
- invalid output diagnostics.

Structural, provenance, identity, non-finite, missing-artifact, or material
numerical failures are fatal. Any allowed cross-device numerical warning must
use thresholds frozen in Stage 3 and remain visible in replay records.

## Stage 7 — integrate the Phase 6.9 report

Create one task-separated Phase 6.9 report under a later frozen reporting
path. The price section references Phase 6.6B artifact hashes and includes:

- both walks and pooled views where already valid;
- epoch 5/15/50 trajectories with epoch 50 primary;
- matched H0, Raw LSTM, persistence, and reversal results;
- paired prediction differences on exact rows;
- subgroup and invalid-path diagnostics;
- parameter/time/memory/latency costs;
- extra supervision and every paper/source/project deviation; and
- seed-0, two-walk, non-trading, non-universal claim boundaries.

Do not present the Phase 6.6B and Phase 6.9 labels as two replications.

## Proposed launcher order

Only after Stage 0 approval:

```bash
.venv/bin/python3 scripts_v8/audit_phase6_6_xlstm_mixer_data.py
.venv/bin/python3 scripts_v8/prepare_phase6_6_xlstm_mixer_data.py
.venv/bin/python3 scripts_v8/validate_phase6_6_xlstm_mixer_data.py
.venv/bin/python3 scripts_v8/audit_phase6_6_xlstm_mixer_runtime.py --device cpu
.venv/bin/python3 scripts_v8/audit_phase6_6_xlstm_mixer_runtime.py \
  --device cuda --admit-training \
  --max-peak-memory-gib <GIB> --max-smoke-seconds <SECONDS>
.venv/bin/python3 scripts_v8/bootstrap_phase6_6_xlstm_mixer.py --device cpu
.venv/bin/python3 scripts_v8/bootstrap_phase6_6_xlstm_mixer.py \
  --device cuda --execute
.venv/bin/python3 scripts_v8/validate_phase6_6_xlstm_mixer.py
```

These commands are planned interfaces, not currently implemented entry
points.

## Definition of done

- all owner decisions are approved and synchronized;
- data/full-path/scaler replay passes for both walks;
- licence/dependency and CUDA admission pass;
- focused model tests pass;
- both walk trajectories and all 5/15/50 checkpoints replay;
- complete full-path predictions and extracted price metrics replay;
- any required matched H0-D0 and Raw-LSTM controls replay on identical rows;
- Phase 6.9 references the one canonical artifact set; and
- the final report preserves extra-supervision and bounded-claim disclosures.
