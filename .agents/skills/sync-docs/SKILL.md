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

### `docs/data_processing_split_contract.md`

Check this document whenever raw split placement, fitted preprocessing,
window-context policy, processed-bundle provenance, or the Phase-2 execution
pause changes. Do not describe a stored train/test split as leakage-safe merely
because downstream row counts or hashes match.

### `docs/price_prediction_label_contract.md`

Check this document whenever price-target construction, eligible row counts,
contract-boundary handling, price result comparability, or migration of a
legacy price experiment changes. Keep its active bundle path and row counts
consistent with `AGENTS.md`, `docs/design.md`, and the relevant phase plans.

### `docs/phase_plan/`

When a change affects phase scope, identifiers, execution order, comparison
fairness, readiness, result judgement, or completion gates, check the
applicable canonical Phase 1 and/or Phase 2 documents in this directory.
For Phase 2 decoder work, include
`docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md`.
For Phase 3 scope, implementation, execution, or result judgement, include
`docs/phase_plan/2026-09-20-phase-3-experiment-plan.md`.
For Phase 3 conclusions or Phase 4 transition decisions, also include
`docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`.
For any Phase 4 data selection, rolling-walk, universe, activity-eligibility,
target-allocation, or execution-gate change, include
`docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`.
For the concluded Phase 4 evidence or any Phase 5 source, causal-fill,
walk-forward, training-gate, or experiment-scope change, also include
`docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`.
For every Phase 5 design, implementation, readiness, or reporting change, read
and update `docs/phase_plan/2026-09-21-phase-5-experiment-plan.md` as the
canonical authority. Initial research documents may retain historical context
but must point readers to this plan when their original design is superseded.
For Phase 5 canonical encoder work, include
`docs/phase_plan/2026-09-21-phase-5-encoder-pretraining-amendment.md`. For
feature extraction, train-only feature scaling, probability-point target
units, or seed-0 framework probing, include
`docs/phase_plan/2026-09-21-phase-5-feature-and-framework-downstream-amendment.md`.
For eight-hour raw-change or two-hour log-return add-on design,
implementation, or results, include
`docs/phase_plan/2026-09-21-phase-5-regression-addons-amendment.md`.
For eight-hour absolute future-price design, implementation, or results,
include `docs/phase_plan/2026-09-21-phase-5-absolute-price-h8-amendment.md`.
For the current Phase 5 interpretation, task-priority change, supported claims,
or candidate reversal factor, include
`docs/phase_plan/2026-09-22-phase-5-intermediate-observation.md`.
For Phase 5 learned-baseline architecture, matrix scope, implementation,
execution, or reporting, include
`docs/phase_plan/2026-09-22-phase-5-baseline-amendment.md`.
For Phase 5 sequence length, walk capacity, contract concentration,
imputation exposure, or selection-readiness changes, include
`docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md`.
For the Phase 6 volatility definition, horizon audit, label eligibility,
comparison matrix, implementation status, or result judgement, include
`docs/phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md`.
For the frozen Phase 6 volatility horizon, future interval, or transition to
label construction, also include
`docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`.
For Phase 6 temporal encoder substitution, heterogeneous additions,
duplicate-width controls, CKA, task scope, implementation status, or result
judgement, include
`docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`.
For Phase 6.5 LSTM depth/capacity or the strict adapted GARCH--LSTM follow-up,
include
`docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`.
For canonical single-branch or leave-one-branch-out attribution, or the
Phase 7B deferral boundary, include
`docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`.

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
