---
name: project-design
description: Use when answering questions about this project's architecture, representation branches, embedding dimensions, aggregator modes, downstream tasks, model design, or planned extensions.
---

# Project Design

Ground design answers in the current project documents rather than assumptions.

## Read First

Read these in order:

1. `docs/Research_Ideas_Writeup.md`, especially sections 3.1 through 3.4 and 5.3 through 5.6 when downstream evaluation or alpha research is relevant.
2. The Architecture Design and Representation Learning sections of `docs/design.md`.
3. The Architecture section of `AGENTS.md`, including the branch table, aggregator modes, extension guide, and module responsibilities.
4. For Phase 2 decoder questions, read
   `docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md` in full.
5. For Phase 5 downstream-task or evaluation design, read
   `docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`
   and `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`
   and
   `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`
   and `docs/phase_plan/2026-09-21-phase-5-experiment-plan.md` in full. The
   Phase 5 plan supersedes conflicting initial design and handoff language.
6. For prediction-market lifecycle effects, representation drift, or temporal
   encoder adaptation, read
   `docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md` in
   full.

## Response Contract

Present the parts relevant to the request:

- Each current representation branch, source module, and output dimension.
- The concat and gated aggregator modes, their output dimensions, and when each is appropriate.
- Probability-movement regression, historical absolute next-close prediction,
  volatility prediction, and tri-class movement/trend classification,
  including their documented metrics and label/target contracts when relevant.
- For Phase 5, describe the canonical five-branch concat framework, separate
  weights per global walk, shared two-hour regression/classification horizon,
  and `tau=0.001` classification. Lifecycle is a reporting stratum.
- Treat encoder variants, fixed-first-walk transfer, gated fusion,
  lifecycle-conditioned models, and branch ablations as Phase 6 work.
- The additional alpha-research capability: downstream predictions rather than latent dimensions as primitives, shallow symbolic search, and chronological OOF-only formula selection.
- Components, methods, or scope explicitly marked as open, provisional, or dependent on later work.

Prefer dimension utilities and `RepresentationAggregator.output_dim` over hard-coded assumptions. If documents disagree with current source code, call out the discrepancy and inspect the implementation before recommending a change.
