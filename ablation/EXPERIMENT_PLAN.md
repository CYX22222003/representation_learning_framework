# Ablation Experiment Plan — Draft 1

## Purpose and boundary

This is an independent, parallel workstream. It is not a Phase-1 continuation
and it is not a Phase-2 experiment. Phase-1's frozen five-branch feature store
is an immutable input only. All ablation plans, checkpoints, predictions,
metrics, plots, and reports are written under `ablation/experiments/`.

The study asks two different questions:

1. **Standalone utility:** can each representation branch support a downstream
   task by itself?
2. **Conditional contribution:** how much does the full representation change
   when one branch is removed while all other branches remain?

Single-branch runs answer the first question. Leave-one-branch-out runs answer
the second. These estimands must not be conflated in the report.

## Locked v1 matrix

Inputs:

- processed sequences: `data/processed/market_4h_seq64_top50.npz`
- frozen features: `data/features/features_4h_seq64_top50_phase1.npz`
- trend labels: `data/task_labels/trend_classification/triclass_4h_seq64_top50.npz`
- volatility labels: `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`

Branches and dimensions are discovered from the feature store and validated
against the expected five names: statistical (70), transformed (55), VAE (64),
contrastive (128), and BYOL (128). Dimensions are recorded in `plan.json`; the
runner does not hard-code them for model construction.

For each of price prediction, volatility prediction, and trend classification:

- five single-branch concat probes;
- five leave-one-branch-out concat probes;
- one full five-branch concat control.

All configurations use the same task head class, hidden width, optimizer,
learning rate, batch size, train-fitted per-branch standardization, seed, and
fixed `15,50,100` epoch snapshots. The default matrix therefore contains 33
model-task configurations and 99 reported fixed-budget results.

The optional full-gated configuration is a fusion/capacity comparison, not a
branch ablation. It must be predeclared in a fresh study via `--include-gated`.

## Data and leakage contract

- Reuse the existing chronological per-contract 80/20 train/test split.
- Do not introduce a validation split or early stopping.
- Fit feature standardizers on training rows only and apply them unchanged to
  test rows.
- Use the saved label bundles and aligned row indices for trend and volatility.
- Construct price targets independently inside train and test splits.
- Write `plan.json` before training/evaluation and reject later matrix changes.
- Report every predeclared epoch budget. Never select a model or budget from
  locked-test performance.
- Frozen encoder inference has already occurred; encoders are not retrained.

## Metrics and comparisons

- Price: MAE, RMSE, with MSE and Pearson correlation retained as diagnostics.
- Volatility: MSE and Pearson correlation, with MAE/RMSE retained.
- Trend: accuracy and macro-F1, with weighted/per-class metrics and confusion
  matrices retained. Interpret accuracy against the majority-class reference.

Every variant is compared at the same task, seed, and epoch budget with the
full-concat control. Positive report deltas always mean that the variant did
better than full concat, regardless of metric direction.

For a leave-one-out variant, a worse result after removing branch `b` supports
a positive conditional contribution from `b`. A better result suggests that
`b` may add noise or harmful redundancy for that task. A strong single branch
shows standalone usefulness but does not establish that it adds value once the
other branches are present.

## Draft-1 limitations and next step

Draft 1 uses a common head width, which controls architecture form but not exact
parameter count because input dimensions vary. Report input dimension and model
parameters before making capacity-sensitive claims. Seed 0 is the initial
characterization matrix; strong claims require a separately predeclared
multi-seed confirmation and preferably paired bootstrap intervals over matched
test predictions.

The plan-only preflight has completed and the locked manifest is stored at
`ablation/experiments/4h_all5_v1/plan.json`. The immediate next action is to
execute that seed-0 matrix without changing configurations in response to test
results.
