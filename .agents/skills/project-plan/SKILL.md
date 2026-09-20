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

## Response Contract

Report:

- What is completed.
- What is implemented but has not been trained, executed, or evaluated.
- What has not started.
- The current phase: A, B, C, or D.
- The most immediate actions required by that phase's exit conditions.
- Scope that remains open or depends on the literature review.
- When relevant, the alpha-research capability's dependency on completed predictive heads, ablations, and leakage-safe chronological OOF predictions; treat it as deferred unless the user explicitly expands the current task-evaluation budget.

When recommending priorities, follow the phase structure and its exit conditions instead of inventing a rigid ordering. If `docs/schedule.md` and `docs/research_plan.md` appear to disagree, treat `docs/schedule.md` as the current progress source and `docs/research_plan.md` as the intended roadmap, then use repository evidence before asserting the current state.
