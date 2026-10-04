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
9. For the approved two-layer LSTM capacity extension, strict adapted
   GARCH--LSTM design, current-task TA-MLP comparison, or price-focused
   residual-CNN SSL study, read
   `docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`.
   For canonical single-branch and leave-one-out attribution, read
   `docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`.
10. For price-focused raw/representation residual fusion, supervised raw
    LSTM/BiLSTM towers, the source-faithful xLSTM-Mixer candidate, canonical
    decoder-capacity sensitivity, or grouped post-hoc attribution, read
    `docs/phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md`.
11. For the immediate recent frozen-representation comparison, candidate
    roster, extraction boundary, or common-probe design, read
    `docs/phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md`.
    For LWA architecture, views, losses, mapping stages, or extraction, also
    read the complete `docs/baselines/LWA/` dossier.

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
  as completed Phase 6 work. Phase 6.5A is complete for encoders, features,
  downstream probes, CKA, and reporting. Phase 6.5B/C is executed and replay-
  valid for both walks: two stacks, two causal TA stores, ten classification
  trajectories, and all 5/15/50 snapshots. The principal report is generated;
  expanded contract-macro/subgroup reporting remains. Phase 6.5D's four
  residual-CNN SSL encoders, two feature stores, eight future-price probes,
  CKA, resources, subgroups, and report are complete and replay-valid. Its
  Contrastive variants improve price error over H0 in both walks but do not
  improve movement ranking overall. Phase 6.7 is the approved immediate next
  phase: the required SaURL-Frozen scope is complete; LWA-Frozen's paper/source
  audit, independent-adaptation dossier, and twelve owner decisions are
  complete, and the Lumid/CUDA plus pinned-PyWavelets runtime audit passes,
  while its Stage 1 model and focused CPU tests are implemented pending owner
  review; remote transform/cache replay, model-specific admission, and
  execution remain pending. Both core
  methods use per-walk pretraining, freezing, and all three current tasks.
  SISSEL-Frozen and TimeDART-Frozen are peer optional post-core
  extensions considered after mentor review and are not required for phase
  completion. Phase 7A follows
  and remains unimplemented. Phase 6.6 is deferred; it freezes price-only
  matched raw/`H0` residual fusion, inserts a Phase 6.6B source-faithful
  xLSTM-Mixer full-path candidate before Phase 6.6C, and retains Phase 6.6C's
  two richer static canonical decoders. xLSTM-Mixer extracts `close[t+8]` from
  an eight-step five-channel forecast and is a contextual complete-system
  baseline because it receives additional target supervision. The simple head remains the primary representation
  probe; the branch-gated decoder is a complete-system sensitivity. Grouped
  SHAP remains later
  descriptive analysis rather than model selection. Fixed-first-walk
  transfer, other temporal decoders, lifecycle-
  conditioned models, and additional seeds remain outside these plans.
- Treat Phase 7B alpha research as deferred pending literature review; no
  search protocol or profitable-alpha claim is currently approved.
- Components, methods, or scope explicitly marked as open, provisional, or dependent on later work.

Prefer dimension utilities and `RepresentationAggregator.output_dim` over hard-coded assumptions. If documents disagree with current source code, call out the discrepancy and inspect the implementation before recommending a change.
