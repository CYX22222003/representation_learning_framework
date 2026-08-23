---
name: baseline-comparison
description: Design, run, and interpret fair comparisons between this project's representation-learning framework and internal or external baselines. Use when comparing framework metrics against Raw-OHLCV MLP, LSTM, GINN, TA-MLP, standalone GARCH, ablations, decoder variants, or when planning claims that one model is better than another.
---

# Baseline Comparison

Use this skill after the data split and task definitions are known, and before selecting a model or writing a performance claim. Isolate representation quality from differences in data, labels, decoder capacity, training budget, and test-set tuning.

## Comparison Workflow

### 1. Establish the comparison contract

Read `docs/training_test_data_selection.md` and the relevant experiment-design sections of `docs/design.md`. Confirm:

- identical processed `.npz` train/test partitions;
- chronological per-contract 80/20 split;
- identical task target builders, horizon, price index, and preprocessing;
- identical saved task label bundles for strict classification comparisons;
- identical test samples and metric definitions;
- fixed epoch budgets and seeds;
- test split locked until final evaluation.

Follow the repository rule of train/test only: no validation split, early stopping, or test-driven checkpoint selection. If another document mentions validation, treat `docs/training_test_data_selection.md` as authoritative and report the conflict.

### 2. Define comparison levels

Use the Raw-OHLCV MLP as the direct internal baseline: it learns end-to-end from flattened raw OHLCV sequences and uses the shared task head. Compare it with the full framework using the same task head first.

Then run ablations to identify where gains come from:

- statistical branch only;
- transformed branch only;
- neural branch only;
- all branches in concat mode;
- all branches in gated mode.

For decoder-controlled claims, compare the end-to-end benchmark, the framework with the default MLP head, and the frozen framework with a benchmark-mirrored decoder. Use this only when the decoder architecture can be separated cleanly.

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
- Design and comparison paradigm: `docs/design.md`
- Baseline plan: `src/baselines/mlp_baseline/EXPERIMENT_PLAN.md`
- Baseline runner: `src/baselines/mlp_baseline/run_experiment.py`
- Plotter: `src/baselines/mlp_baseline/plot_experiment.py`
