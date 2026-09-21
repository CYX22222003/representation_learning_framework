---
name: project-plan
description: Use when asking about this project's current progress, research schedule, priorities, blockers, next actions, phase exit conditions, or work that is completed versus only implemented.
---

# Project Plan

Use the maintained schedule to describe project state, and use the research plan only as the stable roadmap. Do not treat code existence as evidence that training or evaluation has run.

## Read First

Read these in order:

1. `docs/schedule.md` in full, focusing on the achievement tables, summary, and phases A through D. This is the source of truth for current progress, checkpoints, blockers, and next actions.
2. `docs/data_processing_split_contract.md` in full when reporting the current
   blocker or proposing experiment execution.
3. `docs/research_plan.md` in full, including all four stages and final documentation work. This is the stable project guideline; update it only when project direction, planned stages, comparison scope, task definitions, or evaluation methodology changes.
4. When the request concerns a phase plan, read the applicable canonical documents in `docs/phase_plan/` in full. For Phase 2, this includes `2026-09-08-phase-2-experiment-plan.md`; for Part 1 decoder scope or execution, also read `2026-09-14-phase-2-decoder-refinement.md`; for Part 3 classification scope or execution, also read `2026-09-08-phase-2-probabilistic-classification.md`. For Phase 1 readiness or result judgement, read `2026-09-01-phase-1-product-readiness.md` and `phase1_experiment_observation_and_judgement.md`.
   For Phase 3 planning, implementation, or execution, read
   `docs/phase_plan/2026-09-20-phase-3-experiment-plan.md` in full.
   For the Phase 3 conclusion and Phase 4 transition, also read
   `docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`
   and `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`
   and
   `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`
   in full. Then read
   `docs/phase_plan/2026-09-21-phase-5-experiment-plan.md` in full; it is the
   authoritative Phase 5 contract and supersedes conflicting handoff language.
   For Phase 5 training-capacity or readiness claims, also read
   `docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md` in full.

## Response Contract

Report:

- What is completed.
- What is implemented but has not been trained, executed, or evaluated.
- What has not started.
- The current numbered experiment phase and, when useful, its A--D workstream.
- The most immediate actions required by that phase's exit conditions.
- Phase 4 as concluded data-selection/exploration work, explicitly noting that
  it ran no model training. For Phase 5, report whether revised recent-period
  walks, the selected one-hour sequence/label builder, activity/target
  eligibility, walk-specific model lifecycle,
  identical baseline rows, and replay checks are implemented and validated.
  The data builder and common identities are currently implemented through
  `scripts_v3/` with artifacts under `experiments/phase5/data_preparation/`;
  do not conflate that completion with the still-pending model lifecycle and
  model-level replay gate.
  Retrospective selection and final pruning are accepted Phase 5 assumptions;
  cutoff-local catalog selection and quarantine-availability replay are not
  implementation blockers.
  Do not describe per-contract lifecycle fractions as deployment-valid folds.
- Scope that remains open or depends on the literature review.
- When relevant, the alpha-research capability's dependency on completed predictive heads, ablations, and leakage-safe chronological OOF predictions; treat it as deferred unless the user explicitly expands the current task-evaluation budget.

When recommending priorities, follow the phase structure and its exit conditions instead of inventing a rigid ordering. If `docs/schedule.md` and `docs/research_plan.md` appear to disagree, treat `docs/schedule.md` as the current progress source and `docs/research_plan.md` as the intended roadmap, then use repository evidence before asserting the current state.
