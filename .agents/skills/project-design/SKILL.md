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
   For canonical Phase 5 encoder architecture, recipes, or completed weights,
   also read
   `docs/phase_plan/2026-09-21-phase-5-encoder-pretraining-amendment.md`.
   For Phase 5 feature extraction and seed-0 framework downstream probing,
   also read
   `docs/phase_plan/2026-09-21-phase-5-feature-and-framework-downstream-amendment.md`.
   For the executed eight-hour raw-change and two-hour log-return additional
   tasks, also read
   `docs/phase_plan/2026-09-21-phase-5-regression-addons-amendment.md`.
   For the executed eight-hour absolute future-price probe, also read
   `docs/phase_plan/2026-09-21-phase-5-absolute-price-h8-amendment.md`.
   For its intermediate interpretation and current regression-task priority,
   also read `docs/phase_plan/2026-09-22-phase-5-intermediate-observation.md`.
   For the matched Raw-OHLCV MLP/raw LSTM architecture and three-task matrix,
   also read `docs/phase_plan/2026-09-22-phase-5-baseline-amendment.md`.
6. For prediction-market lifecycle effects, representation drift, or temporal
   encoder adaptation, read
   `docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md` in
   full.
7. For Phase 6 volatility-task design, read
   `docs/phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md` in full.
   Preserve its distinction between future interval realised variance and the
   historical overlapping shifted-window proxy. Then read
   `docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`;
   the primary target is frozen to eight hourly increments over `(t,t+8h]`.
8. For Phase 6 temporal encoder substitution, heterogeneous feature addition,
   duplicate-width controls, CKA, or task-transfer design, read
   `docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md` in
   full.
9. For the approved two-layer LSTM capacity extension or strict adapted
   GARCH--LSTM design, read
   `docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`.
   For canonical single-branch and leave-one-out attribution, read
   `docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`.

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
  The canonical VAE, contrastive CNN, and BYOL CNN now have independently
  trained, replay-validated epoch-50 weights for both walks; statistical and
  transformed branches remain deterministic. The resulting 445-dimensional
  feature stores and seed-0 regression/classification probes are complete and
  replay-validated. The exploratory eight-hour raw-change and two-hour
  log-return probes are also complete and did not recover signed correlation;
  the later absolute-price probe recovers positive implied-movement Rank IC
  but remains worse than persistence on level error and weaker than a simple
  last-hour reversal score. The matched Raw-OHLCV MLP and three-layer raw
  OHLCV LSTM comparison is complete and replay-validated for h2 regression/
  classification and h8 future price across both walks.
  The intermediate reporting direction uses future-price prediction as the
  clearest regression transfer task and treats implied-movement Rank IC as the
  primary financial interpretation of that output.
- Treat temporal encoder substitutions and controlled heterogeneous additions
  as completed Phase 6 work. Phase 6.5 and Phase 7A are frozen follow-up plans
  but remain unimplemented: one two-layer LSTM candidate per SSL family, a
  separate strict H=8 GARCH--LSTM benchmark, and canonical single/leave-one-
  out ablations. Fixed-first-walk transfer, gated fusion, lifecycle-conditioned
  models, decoder variants, and additional seeds remain outside these plans.
- Treat Phase 7B alpha research as deferred pending literature review; no
  search protocol or profitable-alpha claim is currently approved.
- Components, methods, or scope explicitly marked as open, provisional, or dependent on later work.

Prefer dimension utilities and `RepresentationAggregator.output_dim` over hard-coded assumptions. If documents disagree with current source code, call out the discrepancy and inspect the implementation before recommending a change.
