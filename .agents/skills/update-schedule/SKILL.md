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

## 4. Update Narrowly

Apply only evidence-supported changes:

- Change status labels only where evidence is clear.
- Set the Last updated field to the current date.
- Rewrite the summary to state the actual blocker and next action.
- Record a met phase exit condition when its evidence is complete.

Do not invent progress. Treat empty or skeletal files as Not started or Implemented, not run according to their substance, and report uncertainty.

## 5. Report

Summarize changed statuses with their evidence, the current phase and remaining exit conditions, and the single most important next action.
