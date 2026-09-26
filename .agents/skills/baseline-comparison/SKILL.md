---
name: baseline-comparison
description: Design, run, and interpret fair comparisons between this project's representation-learning framework and internal or external baselines. Use when comparing framework metrics against Raw-OHLCV MLP, LSTM, Raw LSTM volatility, adapted GARCH--LSTM stacking, GINN limitation evidence, TA-MLP, ablations, decoder variants, or when planning claims that one model is better than another.
---

# Baseline Comparison

Use this skill after the data split and task definitions are known, and before selecting a model or writing a performance claim. Isolate representation quality from differences in data, labels, decoder capacity, training budget, and test-set tuning.

## Comparison Workflow

### 1. Establish the comparison contract

Read `docs/training_test_data_selection.md` and the relevant experiment-design sections of `docs/design.md`. Confirm:

- the processed bundle was generated under the corrected raw-time-first
  contract in `docs/data_processing_split_contract.md`; legacy bundles are
  characterization-only even when downstream row hashes match;
- identical processed `.npz` train/test partitions;
- chronological per-contract 80/20 split;
- identical task target builders, horizon, price index, and preprocessing;
- identical saved task label bundles and row identities for strict task comparisons;
- identical test samples and metric definitions;
- fixed epoch budgets and seeds;
- test split locked until final evaluation.

Follow the repository rule of train/test only: no validation split, early stopping, or test-driven checkpoint selection. If another document mentions validation, treat `docs/training_test_data_selection.md` as authoritative and report the conflict.

For price prediction, strict new comparisons must use
`data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz`. It removes the
terminal row of every contract, yielding 109,791 train / 27,450 test rows on
the current data. Phase-1 and early Phase-2 artifacts with `labels_npz: null`
used the legacy merged-array 109,840/27,499-row contract and retained 49
invalid cross-contract transitions per split. Treat those as legacy
characterisation evidence unless they are retrained on the saved bundle. See
`docs/price_prediction_label_contract.md`.

The historical volatility bundle at
`data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz` uses the old
four-hour shifted-window proxy and is characterization evidence only. Phase 6
strict volatility comparisons must instead use the new walk-specific
eight-hour future-interval realized-variance bundle frozen under
`docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`.
The Raw
MLP, Raw LSTM, adapted GARCH--LSTM, canonical framework, and temporal encoder
variants must share those exact aligned rows.

### 2. Define comparison levels

Use the Raw-OHLCV MLP as the direct internal baseline: it learns end-to-end from flattened raw OHLCV sequences and uses the shared task head. Compare it with the full framework using the same task head first.

Then run ablations to identify where gains come from:

- statistical branch only;
- transformed branch only;
- neural branch only;
- all branches in concat mode;
- all branches in gated mode.

For Phase 2 decoder claims, use the frozen D0--D4 matrix: shallow MLP,
branch-aware residual MLP, gated fusion, temporal LSTM, and temporal
Transformer. Keep the five Phase-1 branches fixed and use identical final task
rows. Treat end-to-end task benchmarks as contextual complete-system
comparisons rather than decoder-isolating controls.

For Phase 6 temporal encoder claims, use the precommitted 11-configuration
matrix in
`docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`.
Interpret same-dimensional temporal substitutions against the canonical H0
framework. Interpret heterogeneous additions against both H0 and the matched
duplicate-feature controls, so evidence for new temporal information is not
confused with evidence for a wider downstream head.

For Phase 6.5, follow
`docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`:
compare the two-layer LSTM primarily with its same-family one-layer reference,
and treat the GARCH--LSTM stack as a complete-system volatility comparator
built from chronological OOF meta-features. For Phase 7A, use both the frozen
single-branch and leave-one-branch-out comparisons in
`docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`; do not
interpret them as parameter-matched causal feature importance.

### 3. Run matched characterization sweeps

Use the same predeclared budgets for every model, such as `15,50,100`. Train from the same seed and report every budget. Do not call the lowest test error the selected model when the budget was chosen after reading test results. For a single final operating point, choose the budget from a training-only rule or commit to it before evaluation.

For regression, report MAE, RMSE, MSE, and Pearson correlation. For classification, report accuracy, macro-F1, per-class precision/recall/F1, and the confusion matrix; inspect class counts and include a majority-class reference when imbalance exists.

### 4. Quantify the claim

Report absolute metrics for every model and budget, then compute relative change using the correct direction:

```text
regression improvement = (baseline_error - framework_error) / baseline_error
correlation improvement = (framework_corr - baseline_corr) / baseline_corr
classification improvement = framework_metric - baseline_metric
```

Run multiple seeds when practical and report mean +/- standard deviation. Because models share test samples, use paired bootstrap confidence intervals for metric differences when making a strong claim. Be precise: beating Raw-OHLCV MLP supports an internal representation-value claim; it does not by itself establish state-of-the-art performance.

### 5. Preserve evidence

Store each run's configuration, dataset manifest, checkpoints, training history, predictions, metrics, and summary. Generate task-specific plots. Keep full sweep tables rather than deleting weaker budgets. Check that no checkpoint, scaler, neural encoder, or hyperparameter was influenced by test data.

## Interpretation Rules

- Lower MAE/RMSE/MSE is better; higher correlation is better.
- For trend classification, accuracy near the majority-class rate with low macro-F1 is not useful learning.
- Lower training loss with worse test metrics indicates overfitting or distribution mismatch, not framework superiority.
- Compare at matched budgets before comparing the best observed budget.
- Explain whether a result demonstrates representation improvement, decoder improvement, optimization improvement, or only a larger parameter budget.

## Project Paths

- Data rules: `docs/training_test_data_selection.md`
- Price-label transition: `docs/price_prediction_label_contract.md`
- Design and comparison paradigm: `docs/design.md`
- Phase 2 D0--D4 decoder contract:
  `docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md`
- Baseline plan: `src/baselines/mlp_baseline/EXPERIMENT_PLAN.md`
- Baseline runner: `src/baselines/mlp_baseline/run_experiment.py`
- Plotter: `src/baselines/mlp_baseline/plot_experiment.py`
- Phase 5 baseline contract:
  `docs/phase_plan/2026-09-22-phase-5-baseline-amendment.md`
- Phase 5 baseline implementation: `src/training/phase5_baselines.py`
- Phase 5 freeze/execute entry point:
  `scripts_v3/bootstrap_phase5_baselines.py`
- Phase 6 future-volatility comparison contract:
  `docs/phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md`
- Phase 6 eight-hour horizon freeze:
  `docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`
- Phase 6 temporal encoder comparison matrix:
  `docs/phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`
- Phase 6.5 LSTM capacity and adapted GARCH--LSTM contract:
  `docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`
- Phase 7A canonical representation ablation contract:
  `docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`
