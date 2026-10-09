# SGN-C Phase 6.9 implementation plan

**Status:** Stages 0--6 complete for the frozen seed-0 scope
**Execution authority:** implementation and bounded admission are complete;
only explicit `--execute` authorizes real training
**Method:** independently authored supervised SGN-C with native head

## Stage 0 — approve provenance and numerical decisions

1. Preserve all seven decisions approved on 2026-10-09.
2. Synchronize approved values across this dossier and the Phase 6.9 planning
   documents.
3. Preserve the independent-only no-upstream-code-reuse boundary.
4. Freeze the exact two-walk inventory before implementation.
5. Do not use H0, Raw LSTM, TA-MLP, or future SGN evaluation metrics to change
   the architecture.

**Exit:** all seven decisions marked approved and machine-readable in
`source_manifest.json`.

## Stage 1 — independently implement initialization and model

**Status:** complete. The independently authored package and 17 focused CPU
tests pass, including interrupted/resumed bit-exact fixture training.

Proposed package:

```text
src/baselines/sgn/
  __init__.py
  config.py
  grouping.py
  model.py
```

Responsibilities:

- `config.py` — immutable dimensions, hierarchy simulation, loss/temperature
  settings, and invalid-config rejection;
- `grouping.py` — bounded BDC, deterministic sampler/K-means, assignment
  logits, Gumbel-Softmax, temperature state, regularizer, and diagnostics;
- `model.py` — shared embedding, averaged odd-kernel banks, intra-/inter-group
  pointwise mixing, shifted/merged hierarchy, native classifier, and structured
  diagnostics.

Do not reproduce upstream naming/layout line for line. Implement from the
paper, approved equations, tensor contract, and disclosed behavior.

### Focused tests

Create `tests/baselines/sgn/` covering at least:

- config validation for `[64,5]`, `G`, `P`, depths, kernels, and safe merge
  schedule;
- exact BDC symmetry, unit diagonal, finiteness, and independence from any
  evaluation-row perturbation;
- deterministic contract-stratified sampling and K-means replay;
- nonempty groups and the expected synthetic OHLC/volume grouping fixture;
- soft train versus hard deterministic evaluation assignments;
- temperature step/save/load/resume behavior and device movement;
- assignment-logit and Gumbel RNG replay;
- weighted group fusion, embedding shapes, and group ordering;
- temporal kernel sizes and average aggregation;
- intra-group isolation and inter-group connectivity with hand-set weights;
- shifted-window boundary truth tables;
- merge schedules for above-four, four-or-fewer, odd-window, and `N=1`
  cases;
- native logits `[B,3]`, finite total/task/group losses, and gradients;
- BatchNorm/dropout train/eval behavior;
- checkpoint round-trip and deterministic fixed-probe inference; and
- all-original-row loader identity and label replay.

**Manual review gate:** independent authorship, shapes, causal initializer,
paper/source deviations, and no evaluation access.

## Stage 2 — implement replayable walk initialization

**Status:** complete. Both 2,000-row contract-stratified training-only
initializers reproduce byte-identical arrays and recover `[OHLC]` versus
`[volume]`; approved `P=16` is present in both FFT candidate sets.

Create reusable logic under `src/training/phase6_9_sgn.py` and a manifest-first
entry point under `scripts_v8/`, for example:

```text
scripts_v8/prepare_phase6_9_sgn_initialization.py
scripts_v8/validate_phase6_9_sgn_initialization.py
```

For each walk:

1. validate the accepted Phase 5 bundle and manifest;
2. load training contexts/identities only;
3. create the approved BDC sample and similarity matrix;
4. fit/replay K-means and assignment initialization;
5. compute/record the training-only period spectrum;
6. verify frozen `G/P` admission conditions; and
7. save a small initialization manifest with all identities, arrays, hashes,
   code settings, and diagnostics.

The validator must reconstruct every value from the source bundle and fail on
evaluation access, hash drift, changed membership, changed period evidence,
non-finite values, or empty groups.

**Manual review gate:** both walk initializers replay and no evaluation array
was opened by the fitting path.

## Stage 3 — implement the fixed-budget lifecycle

**Status:** complete and CPU-tested. The lifecycle is fixed-budget,
checkpoint-resumable, `weights_only=True` compatible, and evaluation-isolated.

Add model/training functionality under `src/training/phase6_9_sgn.py` and
entry points:

```text
scripts_v8/bootstrap_phase6_9_sgn.py
scripts_v8/validate_phase6_9_sgn.py
```

Required behavior:

1. replay the accepted bundle, initializer, labels, priors, and exact row
   identities;
2. run CPU shape/loss/backward/resume smoke tests by default;
3. require an approved implementation manifest and admitted CUDA resource
   record before `--execute`;
4. train one seed-0 50-epoch trajectory per walk with no validation loader;
5. save complete state atomically at 5/15/50 and a resumable latest state;
6. persist task/grouping loss histories, temperatures, assignments, entropy,
   group masses, gradients, time, memory, parameters, sampler/RNG state, and
   environment hashes;
7. resume only from a fully matching manifest and exact next epoch/batch; and
8. never evaluate the locked interval until the full trajectory completes.

Functional failures stop the stage. Replay discrepancies must be classified
as structural/provenance, non-finite, identity, checkpoint-state, numerical,
prediction, or metric failures rather than hidden as warnings.

**Manual review gate:** interrupted/resumed fixture equals uninterrupted
fixture, and a complete CPU synthetic trajectory replays.

## Stage 4 — resource admission and two-run manifest freeze

**Status:** complete. The two-entry seed-0 matrix and implementation hash are
frozen. The selected RTX 4060/PyTorch CUDA runtime admits real Walk 1 contexts
at physical batch 256 plus a one-row remainder: 275,853 parameters,
778,075,136 bytes peak allocation, and same-backend checkpoint replay.

The CUDA audit must run real accepted rows without training a real trajectory:

- one full physical batch forward/backward/update;
- one remainder batch forward/backward/update;
- checkpoint save/load and fixed-probe replay on the selected backend;
- peak allocated/reserved memory, wall time, throughput, and parameter count;
- finite losses/gradients/assignments; and
- an explicit resource ceiling and approved fallback batch.

Freeze one two-entry manifest (`walk1`, `walk2`) containing all source, data,
initializer, architecture, optimizer, runtime, and artifact paths. Both walks
must use the same numerical recipe and physical batch unless the pre-training
admission explicitly records a shared reduction.

The bootstrap remains non-training without `--execute`.

**Manual review gate:** selected CUDA environment, batch, resource limits, and
two-run manifest admitted.

## Stage 5 — execute only after explicit approval

**Status:** complete. Both walk-specific 50-epoch trajectories and all six
snapshots pass same-backend replay.

With explicit `--execute` authorization:

1. train Walk 1 to epoch 50 or resume an exact incomplete run;
2. train Walk 2 under the identical frozen recipe;
3. retain epochs 5/15/50 without evaluating between them;
4. after both trajectories are closed, predict all original evaluation rows
   at every retained snapshot; and
5. independently replay checkpoints, hard assignments, logits, predictions,
   identities, and metrics.

No failed/restarted trajectory may be selected by evaluation performance.
Any numerical instability requiring a recipe change stops both-walk execution
for a documented amendment.

## Stage 6 — matched reporting

**Status:** complete under
`experiments/phase6_9/sgn_classification/reports/seed0/`.

Create:

```text
scripts_v8/report_phase6_9_sgn.py
experiments/phase6_9/sgn_classification/reports/seed0/
```

The report must:

- verify H0-D0 and Raw-LSTM control reuse before joining results;
- show every 5/15/50 per-walk metric, with epoch 50 primary;
- report pooled out-of-future metrics only after per-walk tables;
- include per-class/collapse, contract-macro, lifecycle/category/imputation,
  assignment, grouping-loss, and resource diagnostics;
- disclose all paper/source/project differences and SGN's extra supervision;
- preserve negative or mixed evidence; and
- state the seed-0, two-walk, non-trading, complete-system claim boundary.

The final Phase 6.9 task-specific synthesis remains separate because price,
classification, and volatility use heterogeneous comparators.

## Proposed command lifecycle

Names are provisional until Stage 1 review:

```bash
.venv/bin/python3 scripts_v8/prepare_phase6_9_sgn_initialization.py
.venv/bin/python3 scripts_v8/validate_phase6_9_sgn_initialization.py
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_sgn.py --device cpu
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_sgn.py \
  --device cuda --admit-training \
  --max-peak-memory-gib <GIB> --max-smoke-seconds <SECONDS>
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_sgn.py --device cuda
# Real trajectories only after explicit approval:
.venv/bin/python3 scripts_v8/execute_phase6_9_sgn.py --device cuda
.venv/bin/python3 scripts_v8/validate_phase6_9_sgn_same_backend.py --device cuda
.venv/bin/python3 scripts_v8/report_phase6_9_sgn.py
.venv/bin/python3 scripts_v8/report_phase6_9_sgn_matched.py
```

## Definition of done

- all seven owner decisions are approved and synchronized;
- no upstream source code is copied;
- both walk-specific initialization artifacts replay from training rows only;
- focused CPU tests and selected-backend admission pass;
- the two-run manifest is frozen before evaluation;
- both 50-epoch trajectories and all six snapshots replay independently;
- every original evaluation row has finite three-class logits/predictions;
- H0-D0 and Raw-LSTM controls pass source/identity/prediction audit; and
- the matched classification report and task-separated Phase 6.9 synthesis
  preserve the approved claim boundary.
