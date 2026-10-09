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
built from chronological OOF meta-features. For the TA-MLP classification
benchmark, use the causal feature-availability-only common intersection and
compare its primary P2 run with retrained H0, Raw MLP, and Raw LSTM on
identical rows; keep its P1U source-paper sensitivity separate. For Phase 7A,
use both the frozen
single-branch and leave-one-branch-out comparisons in
`docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`; do not
interpret them as parameter-matched causal feature importance.

For Phase 6.5D, compare price-only residual-CNN substitutions with `H0` and
their additions with both `H0` and the completed same-family duplicate-CNN
control. Classification and volatility are deferred. For Phase 6.7, follow
`docs/phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md`.
For LWA specifically, also read the complete `docs/baselines/LWA/` dossier;
its paper/source audit and twelve owner decisions are complete, and its model
plus Stage 2--4 experiment lifecycle is complete: both caches, both two-stage
walk trajectories, both 384-wide stores, and all six downstream runs are
valid. Phase 6.7 is closed; its integrated epoch-50 comparison is recorded at
`experiments/phase6_7/reports/frozen_representation_seed0/summary.md`.
Treat LWA-Frozen and SaURL-TS-Frozen as the required direct external
representation baselines. Pretrain
each separately on each walk's target-free encoder population, freeze epoch-50
embeddings, and use the established lightweight heads on identical task rows.
SISSEL-Frozen and TimeDART-Frozen are peer optional post-core extensions, not
phase exit conditions. TimeDART was admitted only after its complete contract
was frozen and its full two-walk/three-task matrix is now replay-valid. Because
their admission may follow core-result review, label them exploratory and do
not use them to retroactively replace or redefine the core. Compare all
methods primarily with immutable `H0`; preserve native widths but report head
parameters and resource costs. Do not let evaluation metrics select the core
roster, optional admission, extraction point, adaptation, or checkpoint.
Existing raw, handcrafted, hybrid, and xLSTM-Mixer results are contextual
complete-system comparisons, not substitutes for this matrix.

For Phase 6.8, follow
`docs/phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`.
Treat Di-COT-Frozen and Monotone-VI-Frozen as recent representation-learning
baselines, not assumed empirical SOTA winners. Both are mandatory planned
entries and must use separate per-walk target-free fitting, native-width
frozen stores, exact task identities, and the common probes. The planned
inventory is four representation fits, four stores, 12 downstream
trajectories, and 36 probe snapshots beside six immutable `H0` references.
Do not report any of this inventory as implemented or executed yet.

For Phase 6.9, follow
`docs/phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`.
Keep its three task comparisons separate. For classification, also read
`docs/phase_plan/2026-10-09-phase-6-9-sgn-classification-amendment.md`.
Independently authored supervised SGN-C with its native head replaces the
unimplemented Monotone-VI common-probe leg. Compare with audited original
H0-D0/Raw-LSTM controls on every original h2/tau=0.001 row, with train-only
grouping/period initialization and the existing logit-adjusted task loss.
Disclose the grouping regularizer and native-head optimization freedom:
this is complete-system evidence, not a target-free representation control.
Read `docs/baselines/SGN/` before model coding: its static audit and all seven
decisions are complete, including `P=16` and depths `[2,2,2,1]`. The
independent model/lifecycle, initializers, matrix, tests, CUDA admission, both
50-epoch trajectories, six-snapshot replay, and matched report are complete.
SGN-C trails H0-D0/Raw LSTM/Raw MLP on principal macro-F1 in both walks and
its hard grouping collapses. Optional Phase 6.8
Monotone-VI-Frozen remains unchanged. For price, follow the Phase 6.9-owned
xLSTM-Mixer endpoint amendment in
`docs/phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md`: XM-C8
predicts only close[t+8h] with MSE on every original price row. Audit original
H0-D0/Raw-LSTM controls for source/identity/recipe/prediction reuse. The old
XM-MV8 path/intersection controls are superseded; its full-path results are
historical contextual evidence. The former Phase 6.6B listing is historical and
non-executable. For volatility, reuse rather than
retrain the replay-valid strict Phase 6.5B GARCH--LSTM stacks and retain their
mixed result interpretation. Do not average model ranks across tasks or call
the heterogeneous roster one direct representation matrix.

Before answering or acting on xLSTM-Mixer architecture, source faithfulness,
licence/dependency handling, implementation, runtime admission, or comparison,
also read the complete `docs/baselines/xLSTM-Mixer/` dossier. Its paper/source
audit and all fourteen owner decisions are complete, including exactly one
learned initial token. Both observed-path bundles pass source replay, and the
owner-directed local WSL vanilla-GPU runner replaces the Lumid prerequisite
while retaining selected-backend admission. The guarded infrastructure and
26 focused tests pass, including real-backend fixture resume/replay. Both
real-data seed-0 50-epoch trajectories and all six 5/15/50 snapshots are
complete and replay-valid for historical XM-MV8. The endpoint model,
resumable lifecycle, and 18 CPU tests pass. Both fresh XM-C8 50-epoch walks,
all six independently replayed snapshots, and matched comparison are complete
under `experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/`. XM-C8 beats
H0/Raw LSTM MAE/RMSE in both walks, but persistence wins MAE in both walks
and RMSE in Walk 2. SGN classification is now complete. The XM-MV8-only report under
`experiments/phase6_9/xlstm_mixer/reports/seed0/` is not the endpoint comparison;
retain its mixed persistence result and unconstrained forecast diagnostics.
In particular, the released `FULL` path
flips latent features rather than the variate-token axis.

TimeDART preparation was commissioned on 2026-10-05. Before answering or
acting on TimeDART architecture, training, extraction, feasibility, or
comparison questions, read the complete `docs/baselines/TimeDART/` dossier.
Its paper/source audit, owner-approved 170-dimensional extraction, all eleven
decisions, independent model, focused CPU tests, two encoder trajectories, two
stores, six downstream trajectories, and 18 snapshots are complete and replay-
valid. SISSEL remains optional and uncommissioned.

When summarizing the completed Phase 6.7 results, distinguish the immutable
`H0` direct comparison from the descriptive best-internal-variant envelope.
The envelope is metric-wise and is not one selected model. The strongest
framework result is future-price error; TimeDART has only a modest lead on
classification and realised-variance MAE. For volatility headlines, use MAE
and RMSE: do not use Spearman to claim a regression win, and disclose that
RMSE is effectively tied and mixed across walks.

Phase 6.6 is deferred. If it is reactivated, follow
`docs/phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md`.
Compare residual raw-sequence fusion primarily against the matched `F-H0`
projection control, not only against the simpler `H0-D0` head. Treat raw-only
`F-RL` as the second required comparison when discussing complementary
information. The active fusion and decoder matrices are future-price-only.
For Phase 6.6C decoder capacity, retain the simple D0 head as the
representation probe, compare residual projection D1-RP with D0, and interpret
branch-aware gated D2-BG as a complete-system fusion sensitivity.
Grouped SHAP is descriptive post-hoc attribution and cannot replace matched
ablation or select a model from evaluation results.

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
- Phase 6.5 capacity and task-benchmark contract:
  `docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`
- Phase 6.5B strict GARCH--LSTM implementation:
  `src/baselines/garch_lstm_stacking/phase6_5.py`
- Phase 6.5C causal TA-MLP implementation:
  `src/baselines/ta_mlp_baseline/phase6_5.py`
- Phase 6.6 price-focused raw-fusion and decoder contract:
  `docs/phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md`
- Phase 6.7 recent frozen-representation baseline contract:
  `docs/phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md`
- Phase 6.8 recent conference representation baseline contract:
  `docs/phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`
- Phase 6.9 task-specific competitiveness contract:
  `docs/phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`
- Phase 6.9 SGN classification replacement and implementation-discussion gate:
  `docs/phase_plan/2026-10-09-phase-6-9-sgn-classification-amendment.md`
- SGN paper/source audit, proposed specification, and owner decision record:
  `docs/baselines/SGN/`
- Phase 7A canonical representation ablation contract:
  `docs/phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`
