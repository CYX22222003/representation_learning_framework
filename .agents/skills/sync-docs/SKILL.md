---
name: sync-docs
description: Use when a project design decision, architecture change, new model or baseline, revised data rule, evaluation change, or progress update may leave research documents and repository skills inconsistent.
---

# Sync Project Documentation

Propagate an established change across affected documents without rewriting sections that are already consistent. The project follows an ideation -> plan/design -> implement -> evaluate loop, so each document must describe the same current system.

## 1. Identify the Change

Derive a one-sentence change statement from the conversation. If the change is ambiguous, ask the user for that sentence before editing.

## 2. Audit Relevant Documents

Read the relevant sections before editing. Change only material affected by the decision.

### `docs/Research_Ideas_Writeup.md`

Check architecture diagrams, training overview, innovation claims, evaluation tasks, metrics, and the provisional baseline list.

### `docs/design.md`

Check Architecture Design, Representation Learning, Training Procedure, and Evaluation Process. Preserve leakage-prevention rules and the benchmark-versus-internal-baseline comparison.

### `docs/research_plan.md`

Check Stage 2 model components, Stage 3 baselines, Stage 4 tasks and metrics, and the stable roadmap. Do not use this document for routine progress snapshots or completion markers; those belong in `docs/schedule.md` and experiment reports.

### `docs/schedule.md`

Check achievement statuses, the summary paragraph, phase scope, and exit conditions. Update status only when supported by repository evidence.

### `AGENTS.md`

Check the branch table, source modules, dimensions, aggregator example, extension guidance, module responsibilities, script commands, and data contracts.

### `docs/training_test_data_selection.md`

Check the allocation table, operation order, and rules summary. New components must use the correct train/test boundaries and must not introduce validation splits, early stopping, or test-driven model selection unless the project methodology is deliberately changed.

## 3. Audit Repository Skills

Update a skill only when its triggers, document pointers, or workflow are affected:

- `.agents/skills/experiment-setup/SKILL.md` for data rules, training order, or evaluation fairness.
- `.agents/skills/project-design/SKILL.md` for architecture, branches, dimensions, aggregators, tasks, or metrics.
- `.agents/skills/project-objectives/SKILL.md` for motivation, positioning, tasks, or innovation claims.
- `.agents/skills/project-plan/SKILL.md` for phase structure, schedule, blockers, or priorities.
- `.agents/skills/progress-logging-report/SKILL.md` for supervisor-facing progress log sources, format, or separation from internal working docs.
- `.agents/skills/python-env/SKILL.md` for runtime, virtual environment, or command conventions.
- `.agents/skills/update-schedule/SKILL.md` for evidence sources, statuses, paths, or schedule structure.
- `.agents/skills/sync-docs/SKILL.md` when new authoritative documents or skills enter the consistency set.

After changing any skill, verify that its `agents/openai.yaml` still matches its purpose and run the official skill validator.

## 4. Confirm Consistency

Report which documents and skills changed and the specific decision propagated to each. If a previously open decision is resolved, replace its provisional marker with the supported decision. Report unresolved conflicts rather than masking them.
