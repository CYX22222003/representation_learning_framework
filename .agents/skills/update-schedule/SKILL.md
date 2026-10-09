---
name: update-schedule
description: Use when auditing repository progress against docs/schedule.md, correcting completion statuses, determining the current phase, or identifying the next action from git, code, checkpoint, and generated-data evidence.
---

# Update Schedule From Evidence

Audit the repository before changing `docs/schedule.md`. Schedule text is not proof of completion.

## 1. Gather Git Evidence

Run:

```bash
git log --oneline -20
git diff HEAD~5 -- src/ scripts/ checkpoints/
git status --short
```

Adjust the diff range only when history shows a different range is relevant. Summarize recent changes by area and preserve awareness of uncommitted user work.

List `checkpoints/`. A checkpoint is strong evidence that the corresponding model trained and saved, but inspect naming and context before attributing it.

## 2. Inspect Runtime Evidence

For schedule items marked Implemented but not run or Not started, inspect:

- `src/models/vae.py`
- `src/models/contrastive.py`
- `src/training/`
- `src/aggregation/aggregator.py`
- `src/tasks/`
- `src/evaluation/`
- `src/baselines/`
- `scripts/`
- `data/processed/`
- `data/features/`

Check that files contain substantive code. Generated NPZ files and checkpoints provide execution evidence; source files alone do not.

## 3. Classify Each Schedule Item

Use these states:

| State | Required evidence |
|---|---|
| Done | The deliverable exists; trained components also have checkpoint or equivalent run evidence. |
| Implemented, not run | Substantive implementation exists but training, generation, or evaluation evidence is absent. |
| Not started | No substantive file, commit, artifact, or other evidence exists. |

Compare every achievement-table row in `docs/schedule.md` with the evidence. Also verify the summary paragraph, current phase, and phase exit conditions.
For the current Phase 4-to-5 transition, read
`docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`
and `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`
and `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`.
For recent one-hour sample-capacity and walk-readiness claims, also inspect
`docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md` and its
Git-ignored manifest. Do not mark the Phase 5 data contract complete merely
because the retrospective 50-condition cohort has enough `seq64` rows.
Distinguish concluded exploratory scope from completed model execution: Phase
4 ran no model training. Phase 5 is now complete only to the extent supported
by its saved walk bundles, checkpoints, predictions, reports, and replay
validation; do not infer additional seeds or deferred attribution work.

For Phase 6 status, read
`docs/phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md` and
`docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md` and
`docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md` in full.
Distinguish a frozen planning contract from implementation and execution.
Verify any claimed progress through substantive Phase 6 source/entry points,
generated manifests, label bundles, checkpoints, predictions, and replay
reports. Prior Phase 3 temporal classes do not prove that the Phase 6 walk-
specific matrix exists, and a frozen H=8 decision does not prove that the
future-interval label bundles have been built.

For Phase 6.5, Phase 6.6, and Phase 7 status, also read
`docs/phase_plan/2026-09-26-phase-6-experiment-observation-and-outcomes.md`,
`docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`,
`docs/phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md`,
`docs/phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md`,
`docs/phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`,
`docs/phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`,
and `docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md` in
full. The Phase 6.5 two-layer LSTM capacity matrix, strict H=8 adapted GARCH--
LSTM, and current-task TA-MLP classification matrix now have their own
manifests, generated stores where applicable, checkpoints, predictions, and
replay evidence; treat them as executed. Phase 6.5D now also has replay-valid
checkpoints, feature stores, downstream runs/predictions, CKA, resources,
subgroups, and its complete report; treat its frozen price-only seed-0 scope
as executed. Phase 6.7 is closed for its frozen seed-0 scope. SaURL, LWA, and
the commissioned optional TimeDART extension each have two walk-specific
encoders, two native-width stores, and six downstream trajectories with replay
evidence. The integrated epoch-50 comparison is generated at
`experiments/phase6_7/reports/frozen_representation_seed0/summary.md`; a
separate TimeDART-only report is not required. SISSEL remains optional and
uncommissioned and does not reopen
the phase. Phase 6.8 is now optional. Its roster and comparison contract are
frozen, but Di-COT/Monotone-VI source dossiers, implementation, fits, stores,
downstream runs, and reports do not exist and must remain planned. Phase 6.9
is the current approved planning handoff. Read
`docs/phase_plan/2026-10-09-phase-6-9-sgn-classification-amendment.md`:
independently authored supervised SGN-C replaces the unimplemented Monotone-VI
classification leg, not optional Phase 6.8's representation roster. Its
source/settings/licence audit and implementation specification are pending;
SGN implementation, admission, training, and replay do not exist yet.
The xLSTM-Mixer price leg has a completed
paper/source audit, owner decision record, guarded model/training/replay
infrastructure, observed-path builder, and local vanilla-GPU runner under
`docs/baselines/xLSTM-Mixer/`. Both data bundles pass source replay and 26
focused tests pass. Read the local feasibility manifest for CUDA admission
evidence. Both real-data seed-0 50-epoch trajectories and all six 5/15/50
snapshots are complete and replay-valid; the XM-only diagnostic report is
under `experiments/phase6_9/xlstm_mixer/reports/seed0/` for historical XM-MV8.
Read `docs/phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md`:
primary XM-C8 uses one close[t+8h], endpoint MSE, all original price rows,
and audited original controls. Intersection reruns are superseded. Endpoint
model/lifecycle and 18 CPU tests pass. Both fresh 50-epoch endpoint walks,
all six independently replayed snapshots, and matched reporting are complete
under `experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/`. SGN
classification remains open; do not mark Phase 6.9 closed.
The owner-directed local amendment removes Lumid as a prerequisite. Its volatility leg
reuses completed Phase 6.5B artifacts. Phase 7A remains unimplemented.
Phase 6.6A/C's
deferred price-focused fusion/decoder scope and the
canonical branch-ablation matrix remain only planned. Source and CPU tests
alone support only an "implemented, not run" status. Phase 7B alpha
research is deferred and has no approved execution contract. Phase 6.6 is
planning evidence only until its fusion/decoder-capacity source, manifests,
checkpoints, predictions, and replay reports exist. Phase 6.9 solely owns the
xLSTM-Mixer contract; the former Phase 6.6B label is historical. Grouped
attribution is a later analysis and not model-completion evidence.

## 4. Update Narrowly

Apply only evidence-supported changes:

- Change status labels only where evidence is clear.
- Set the Last updated field to the current date.
- Rewrite the summary to state the actual blocker and next action.
- Record a met phase exit condition when its evidence is complete.

Do not invent progress. Treat empty or skeletal files as Not started or Implemented, not run according to their substance, and report uncertainty.

## 5. Report

Summarize changed statuses with their evidence, the current phase and remaining exit conditions, and the single most important next action.
