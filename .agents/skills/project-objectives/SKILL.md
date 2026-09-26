---
name: project-objectives
description: Use when explaining or writing about this project's research motivation, problem statement, objectives, related-work gap, Polymarket domain choice, innovation claims, or downstream-task rationale.
---

# Project Objectives

Use the research write-up as the authoritative source for positioning. Avoid inventing novelty claims or overstating what the implementation demonstrates.

## Read First

Read these sections of `docs/Research_Ideas_Writeup.md`:

1. Section 1 for the topic and high-level goal.
2. Sections 2.1 through 2.3 for limitations, embedding approaches, and crypto-market motivation.
3. Section 3.1 for the problem addressed by the proposed model.
4. Section 5.4 for the alpha-research contribution boundary when relevant.
5. Section 6 for the stated inspiration and motivation.
6. For the current Phase 5 transferability and candidate-alpha interpretation,
   read `docs/phase_plan/2026-09-22-phase-5-intermediate-observation.md`.
7. For the Phase 6 volatility rationale and target boundary, read
   `docs/phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md`.
8. For the Phase 6 temporal-backbone and heterogeneous-complementarity
   questions, read
   `docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`.

## Response Contract

Present the relevant parts of:

- The research gap addressed by the project.
- Task-specific overfitting, limited feature diversity, and the transferability gap.
- Why Polymarket event contracts are a suitable and challenging domain.
- The documented innovation claims, clearly distinguishing implemented work from intended contributions.
- The three downstream evaluation tasks and why they test transferability,
  including the Phase 4 decision to replace absolute next-close prediction
  with continuous probability-movement regression and Phase 5 global calendar-
  time walks for pooled prediction-market contracts.
- The Phase 5 intermediate finding that eight-hour future-price prediction is
  the clearest regression transfer task, with implied-movement Rank IC as the
  finance-relevant diagnostic; direct raw/log movement heads are diagnostic
  negative evidence.
- Last-hour reversal as a candidate empirical factor discovered in the recent
  Polymarket cohorts, not a confirmed profitable strategy or novel mining
  algorithm.
- The Phase 6 volatility target as future interval realised variance from raw
  probability changes, with its horizon selected from training-period data
  diagnostics rather than model evaluation.
- The Phase 6 encoder distinction between fixed-width backbone substitution
  and heterogeneous feature complementarity, including duplicated-CNN width
  controls and evaluation on classification, future price, and volatility.
- The alpha-research capability as supportive downstream evidence, rather than a claim of a novel alpha-mining algorithm or profitable trading system.

For report-writing or related-work requests, keep claims proportional to the evidence in the source document and identify provisional language that still needs experimental support.
