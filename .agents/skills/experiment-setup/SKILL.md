---
name: experiment-setup
description: Use when planning training, validation, testing, baseline fairness, data allocation, or evaluation for this project, especially when checking for leakage or the correct experiment order.
---

# Experiment Setup

Use the project documents as the authority for experiment design. Do not infer allocation rules from implementation details alone.

## Read First

Read these in order:

1. `docs/training_test_data_selection.md` in full, including the global split, train/test-only policy, allocation table, task label bundles, test feature extraction, operation order, aggregator modes, and rules summary.
2. `docs/data_processing_split_contract.md` in full. Treat legacy processed and
   feature bundles as historical characterisation evidence until the required
   raw-time-first rebuild is complete.
3. The Experiment Design section of `docs/design.md`, including Data Preparation, Representation Learning, Training Procedure, and Evaluation Process.
4. For price prediction or cross-generation price comparisons,
   `docs/price_prediction_label_contract.md` in full.
5. For Phase 3 preparation, training, extraction, downstream evaluation, or
   baselines, `docs/phase_plan/2026-09-20-phase-3-experiment-plan.md` in full.
6. For Phase 4 design or any post-Phase-3 execution, read
   `docs/phase_plan/2026-09-20-phase-3-experiment-observation-and-conclusion.md`
   and `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`
   and
   `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`
   in full. Then read
   `docs/phase_plan/2026-09-21-phase-5-experiment-plan.md` in full. The Phase 5
   plan is authoritative: the earlier four-hour top-80 contract was not
   executed and the initial research documents may contain superseded design.
7. For Phase 4 walk-count, expanding-versus-rolling, target-distribution, or
   top-80 sample-feasibility questions, read
   `docs/data_analysis/2026-09-20-phase4-top80-contract-relative-walk-analysis.md`
   in full. Treat contract-relative anchored/fixed-length results as
   retrospective lifecycle evidence, not as a causal pooled-model split.
8. For one-hour timestamp availability, context-duration matching, or Phase 4
   sample-capacity questions, read
   `docs/data_analysis/2026-09-20-phase4-top80-1h-timestamp-capacity.md` in full.
   Do not infer evaluation quality from row count alone; check unique-contract
   coverage in every global-calendar interval.
9. For one-hour staleness, volume, inactive-tail, or target-variation questions,
   read `docs/data_analysis/2026-09-20-phase4-top80-1h-activity-suitability.md`
   in full. Retrospective trailing-flat duration is diagnostic only. Any active
   market eligibility rule used in an experiment must depend only on history
   available at the decision timestamp and be shared by all comparators.
10. For one-hour versus four-hour or top-50 versus top-80 design choices, read
    `docs/data_analysis/2026-09-20-phase4-top50-top80-1h-4h-comparison.md` in
    full. Match context, horizon, and activity duration in hours, not bars, and
    treat raw candle timestamps as bar starts when enforcing calendar cutoffs.
11. For the lab FinData API, recent-data collection, sparse 15m/1h candles, or
    the December-2025--August-2026 cohort, read both
    `docs/data_analysis/2026-09-20-findata-expanded-recent-cohort.md` and
    `docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md` and
    `docs/data_analysis/2026-09-21-findata-native-15m-1h-dynamics.md` in full.
    Condition-level candles
    can mix complementary YES/NO prices; the token-consistent trade fallback is
    too sparse for a diverse seq64 cohort. Preserve raw rows, never invert
    prices, retain forward-confirmed persistent crashes, and leave quarantined
    timestamps as gaps. Phase 5 accepts the approved final pruning as correct
    offline cleaning and does not replay `quarantine_available_at`.
    Native-frequency forward-fill sensitivity is limited to one missing bar,
    flat OHLC, zero volume, and explicit imputation metadata; longer gaps stay
    missing, observed-only results remain primary, and stochastic augmentation
    must not enter canonical OHLCV or targets.
    Phase 5 selects the two fresh walk-specific top-50 clean native one-hour
    condition-candle cohorts under this explicit research assumption,
    causally fills only complete isolated one-hour gaps, uses synthetic rows as
    context only, requires observed decision/target endpoints, and breaks
    sequences at longer gaps. Native 15-minute data is a later sensitivity.
    The walk-specific sequence/label builder and identical comparator
    identities are implemented under `src/data_processing/` and exposed by
    `scripts_v3/`, with validated bundles under
    `experiments/phase5/data_preparation/`. The six canonical walk-specific
    neural encoder trajectories are complete under
    `docs/phase_plan/2026-09-21-phase-5-encoder-pretraining-amendment.md`.
    The two canonical feature stores and four seed-0 framework downstream runs
    are complete and replay-validated. Learned baselines and additional seeds
    remain gated on their separately frozen matrix and fairness checks.
    Encoder row eligibility must remain target-free; apply future-target
    existence, observed status, segment continuity, and maturity only when
    deriving downstream supervised train/evaluation rows.
    For feature extraction and the seed-0 framework probe, read
    `docs/phase_plan/2026-09-21-phase-5-feature-and-framework-downstream-amendment.md`.
    It freezes walk-local train-only feature standardization and the fixed
    probability-point regression unit `100 * delta`, inverted before raw-delta
    reporting. Learned baselines and additional seeds are later work.
    The completed post-primary regression sensitivities are governed by
    `docs/phase_plan/2026-09-21-phase-5-regression-sensitivities-amendment.md`:
    eight-hour raw change uses independently mature targets, while two-hour
    log return uses a train-only target transform and price-band reporting.
    The executed absolute-price probe is governed by
    `docs/phase_plan/2026-09-21-phase-5-absolute-price-h8-amendment.md`: it
    reuses the eight-hour rows, has no explicit current-price skip, and reports
    implied-movement Rank IC separately from level reconstruction.
12. For recent one-hour Phase 5 window capacity, read
    `docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md` in full.
    Use its capacity evidence while applying the later Phase 5 decision:
    `seq64` and the balanced two-walk schedule are frozen; retrospective
    selection is accepted; `seq256` remains a later sensitivity.

## Response Contract

Present the relevant parts of:

- The global 80/20 train/test split and why the test side remains locked.
- For Phase 5, the two fresh rolling global calendar-time walks, walk-specific encoder
  weights, and proof that later timestamps from any contract never train an
  earlier-walk model. Per-contract lifecycle fractions are reporting strata,
  not the primary split.
- Whether the raw-time boundary precedes fitted imputation/scaling and window
  generation; identical stored `.npz` rows do not establish leakage safety.
- The train/test-only rule: no validation split, no early stopping, and no test-driven checkpoint selection.
- The data allocated to encoders, aggregator, task heads, and baselines for training and evaluation.
- Any task-label constraints, including train-fitted thresholds/scalers and split-safe horizon alignment.
- For price prediction, whether the run uses the active contract-safe bundle or
  the legacy merged-array contract, including removal of terminal contract
  rows before fitting the feature standardiser.
- For probability-movement regression, the continuous signed probability-change
  two-hour horizon, fold-local eligibility, and exact zero-movement reference.
  Classification uses the same horizon and fixed `tau=0.001`. Treat
  conventional arithmetic-return regression as secondary unless its
  zero-price, scaling, robust-loss, and price-band contract is predeclared.
- The ordered workflow from raw data through final evaluation.
- The rules that prevent data leakage and preserve baseline fairness.

When reviewing a proposed workflow, identify any step that violates the documented leakage safeguards and explain the compliant alternative. If the documents disagree, report the conflict instead of silently choosing one.
