# Phase 6.9 SGN Classification Replacement Amendment

**Date:** 2026-10-09
**Authority:** Owner-directed revision of the classification leg of
`2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`.
**Status:** Comparator replacement and independent-reimplementation direction
approved. Implementation specification, source/licence audit, runtime
admission, and training remain pending. This amendment authorizes planning,
not model implementation or experiment execution.

## 1. Decision and scope

Replace the unimplemented Phase 6.9 Monotone-VI classification adaptation
(`MVI-C-D0` and its uncommissioned source-classifier sensitivity) with an
independently authored **SGN classification adaptation**, provisionally named
`SGN-C` / `sgn_c`.

SGN is selected for its explicit supervised multivariate time-series
classification design, not because of any Polymarket evaluation result.
Monotone-VI is a sequence-representation method and remains a legitimate
representation comparator; it is not rejected as scientifically invalid.
Its optional Phase 6.8 `Monotone-VI-Frozen` scope remains unchanged.

The completed XM-C8 price leg, historical XM-MV8 evidence, reuse of the
completed strict Phase 6.5B GARCH--LSTM volatility stacks, optional Phase 6.8,
deferred Phase 6.6, and frozen Phase 7A are not changed or relaunched.

## 2. Sources and reimplementation boundary

Primary paper: Ying et al., *SGN: Shifted Window-Based Hierarchical Variable
Grouping for Multivariate Time Series Classification*, NeurIPS 2025.

- [Published paper](https://papers.nips.cc/paper_files/paper/2025/file/9b9aa183c3ab49380e0f306e9f130acf-Paper-Conference.pdf).
- [Official reference repository](https://github.com/colison/SGN).
- Owner-supplied local reading copy:
  `E:\zotero\storage\STZF8JRL\Ying et al. - 2025 - SGN Shifted Window-Based Hierarchical Variable Grouping for Multivariate Time Series Classification.pdf`.

The paper, official model, experiment runner, and launch configurations are
method and setting references, not a ready-made project training pipeline.
Write project-native model and experiment code independently. Do not copy or
redistribute upstream source before its licence/permission boundary is
resolved. Pin and record the reference revision during the formal source
audit; the initial reading is not a completed reproducibility or runtime gate.

Preserve the defining method family: variable grouping, group-level temporal
embedding and intra-/inter-group mixing, multi-scale convolution, shifted
periodic windows, hierarchical merging, and a native classification head.
Any omission or substantive alteration must be explicit and approved; do not
silently reduce SGN to a generic CNN. Treat the resulting method as an
adaptation, not an exact paper reproduction or a new methodological invention.

The audit must resolve known reference issues before coding: the model's
hard-coded TDBRAIN grouping file, precomputed similarity-matrix provenance,
configured rather than forward-selected period, paper/source embedding and
grouping semantics, temperature-state handling, and safe merging depth for a
64-step input. Preserve reference evidence and document each chosen behavior.

## 3. Comparison role and scientific question

The primary comparison becomes:

```text
SGN-C: historical raw OHLCV -> supervised SGN backbone -> native 3-class head
versus immutable H0-D0 and Raw LSTM on the original classification identities
```

This tests the competitiveness of one reusable target-free representation
system against a classification-specialized complete system. It does not
isolate representation quality, decoder capacity, variable-grouping
causality, or parameter-matched architecture effects. Do not freeze a
supervised SGN embedding and relabel it as target-free representation evidence.

Recognition of a state/activity within an observed sensor signal is not the
same problem as predicting future probability movement. The SGN transfer to
five-channel OHLCV is an empirical question, not a presumed performance win.
No universal SOTA, trading, or multi-seed robustness claim is authorized.

## 4. Unchanged task, rows, and causal allocation

Use the two original Phase 5 walk bundles and their companion manifests:

```text
experiments/phase5/data_preparation/walk{1,2}/market_1h_seq64_h2.npz
```

Input is `[B,64,5]` in the existing OHLCV order and preprocessing. Output is
three logits for the unchanged source labels:

```text
delta = close[t+2h] - close[t]
DOWN if delta < -0.001; STABLE if abs(delta) <= 0.001; UP if delta > 0.001
```

Reuse all original supervised training/evaluation rows, identities, ordering,
labels, gap/activity eligibility, observed endpoints, and target maturity.
Do not construct a future-path target, change the horizon to eight hours,
delete rows for SGN-specific convenience, or filter by future label values.
The expected train/evaluation counts are 37,864/30,340 for Walk 1 and
57,521/13,887 for Walk 2; verify them against the saved manifests before fitting.

Both walks retain their existing global-calendar cutoffs and independent
weights. Similarity/BDC initialization, K-means state, any FFT-derived period
statistics, and any additional permitted preprocessing use that walk's
classification training population only. A bounded training subsample for
quadratic-cost BDC must be predeclared and replayable. No upstream sensor
matrix, pooled train/evaluation fitting, later-walk state, or test-derived
class metadata may initialize the model.

Full historical-context convolution or shifting is not future leakage: all
input timestamps are already observed at the decision. No target-interval
candle may enter the input or a fitted preprocessing statistic.

## 5. Experiment lifecycle and controls

The bounded scope is two fresh supervised trajectories, one per walk, at
seed 0 with 50 epochs and retained snapshots at 5/15/50. Epoch 50 remains the
predeclared principal result. There is no validation split, early stopping,
evaluation-driven hyperparameter choice, restart selection, or best-on-test
checkpoint selection. Evaluate retained snapshots only after the fixed
trajectory completes, and report them all.

Use the established training-prior logit-adjusted classification protocol
(`lambda=1.0`) as the task-loss component. SGN's grouping regularization is a
separate, disclosed method component; its coefficient, reduction, and
implementation remain to be frozen. The paper's ordinary cross-entropy,
validation early stopping, 100-epoch budget, and dataset-specific settings
are references, not automatic overrides of the project protocol.

Audit original H0-D0 and Raw-LSTM configurations, source bundles, target and
row hashes, training recipes, checkpoint budgets, predictions, and replay
evidence before reuse. Raw MLP may be included as an existing identical-row
contextual reference without adding a new training trajectory. TA-MLP's
separate availability-intersection results are not silently inserted into
this full-row table. Reusing controls requires evidence, not just equal counts.

Macro-F1 is primary; report balanced accuracy, accuracy, per-class
precision/recall/F1, confusion matrix, predicted-class counts, collapse
diagnostics, and existing classification score metrics. Retain always-STABLE
and training-prior references, per-walk and contract-macro/subgroup results,
and pool only out-of-future predictions. Report trainable parameters,
initialization/training/inference time, peak memory, supervision, and all
paper/source/protocol deviations.

## 6. Decisions reserved for the implementation-plan discussion

No numerical architecture or optimizer choice is approved by this amendment.
The next discussion must resolve and then freeze:

1. BDC estimator, normalization, bounded sampling, K-means initialization,
   group count, empty-group behavior, and grouping-collapse diagnostics.
2. Global versus sample-dependent assignment and group embedding semantics,
   including shared versus independent embedding weights.
3. Fixed training-derived versus input-local period selection; FFT/DC,
   padding, constant-input, and non-periodic-input behavior.
4. Embedding width, kernel bank, stage depths, parameter sharing, shifting
   boundary behavior, and safe merge schedule for 64 steps.
5. Gumbel temperature schedule and checkpoint-safe training/evaluation state;
   grouping loss coefficient/reduction and native pooling/head details.
6. Optimizer, learning rate, weight decay, physical batch, dropout, clipping,
   precision, scheduler policy, and resource-only admission limits.
7. Paper/source discrepancy policy, attribution/licence boundary, test and
   replay tolerances, resumability, fingerprints, and complete reporting.

Use source settings and training-only diagnostics to justify choices. Neither
existing comparator test scores nor later SGN test scores may select them.
Do not commission a hyperparameter sweep, ablation matrix, additional seed,
frozen-SGN probe, or extra downstream task implicitly.

## 7. Ordered gates and proposed ownership

1. Discuss and approve a separate SGN implementation specification, supported
   by a pinned paper/source/settings/licence audit.
2. Author the project-native model and full-row experiment lifecycle; add
   CPU shape/loss/gradient, split-isolation, deterministic inference,
   temperature-state, merge-edge, checkpoint/resume, and replay tests.
3. Verify original rows and controls; freeze the two-run manifest and admit
   the declared local GPU recipe with bounded smoke/resource checks.
4. Launch only after explicit execution approval; preserve checkpoints,
   optimizer/RNG state, histories, logits/predictions, identities, and hashes.
5. Independently replay all six snapshots and produce the matched
   classification report, then the task-separated Phase 6.9 synthesis.

Proposed locations, not evidence of existing implementation:

```text
docs/baselines/SGN/
src/baselines/sgn/
src/training/                 # reusable Phase 6.9 lifecycle
scripts_v8/                  # manifest-first entry points
tests/baselines/sgn/
experiments/phase6_9/sgn_classification/
  feasibility/
  manifests/
  downstream/classification_h2/walk{1,2}/sgn_c/seed0/
  reports/seed0/
```

This planning change does not satisfy source audit, implementation, admission,
training, replay, or phase closure. Failure of an SGN gate requires an explicit
decision; it does not authorize a different comparator or altered task rows.
