# Phase 7A Canonical Representation Ablation Plan

**Date:** 2026-09-26
**Status:** Approved planning contract; implementation and execution have not
started
**Predecessor:** `2026-09-26-phase-6-experiment-observation-and-outcomes.md`

## 1. Purpose and boundary

Phase 7A measures what each branch of the canonical five-branch representation
contributes to the three accepted downstream tasks. It uses both single-branch
and leave-one-branch-out probes because they answer different questions:

- a **single-branch** probe measures whether one branch is useful by itself;
  and
- a **leave-one-branch-out** probe measures whether removing that branch from
  the complete representation changes performance in the presence of the
  other four branches.

This study does not select one of the Phase 6 temporal configurations. The
canonical `H0` representation was fixed before Phase 6 evaluation and remains
the sole full-model reference:

```text
H0 = statistical + transformed + VAE + contrastive CNN + BYOL CNN
```

Phase 7A is independent of the Phase 6.5 outcomes. It may be implemented
before or after Phase 6.5, but its matrix must not be changed in response to a
Phase 6.5 result.

## 2. Research questions

The study asks:

1. Which canonical branches contain useful standalone information for each
   task and walk?
2. Which branches provide incremental information when the other four
   canonical branches are already present?
3. Is representation usefulness task-dependent across movement
   classification, future-price reconstruction/ranking, and future realised
   variance?

It does not estimate causal feature importance or Shapley values. Correlated
branches can substitute for one another, so a small leave-one-out change does
not prove that a branch contains no information.

## 3. Data and frozen representation sources

Phase 7A reuses the accepted Phase 5/6 one-hour global-calendar walks and
their exact task-specific row identities:

| Task | Frozen target/row contract |
|---|---|
| Movement classification | two-hour `DOWN/STABLE/UP`, `tau=0.001` |
| Future price | eight-hour future probability and implied movement |
| Volatility | eight-hour strictly future realised variance from raw probability changes |

The five source branches are the already replayed walk-specific epoch-50
canonical features:

| Branch | Symbol | Width |
|---|---|---:|
| Statistical AR/GARCH features | `S` | 70 |
| FFT/Haar transformed features | `T` | 55 |
| VAE latent mean | `V` | 64 |
| Contrastive CNN backbone | `C` | 128 |
| BYOL CNN online backbone | `B` | 128 |
| Full canonical concat | `H0` | 445 |

No encoder is retrained for Phase 7A. No target value, evaluation metric, or
Phase 6 result may alter branch extraction or row eligibility.

## 4. Frozen ablation matrix

### 4.1 Single-branch probes

| ID | Included branches | Width |
|---|---|---:|
| `A-S` | `S` | 70 |
| `A-T` | `T` | 55 |
| `A-V` | `V` | 64 |
| `A-C` | `C` | 128 |
| `A-B` | `B` | 128 |

These configurations test standalone downstream usefulness. Comparisons with
`H0` are descriptive complete-system comparisons; a narrower model is not
expected to match the full model merely to be considered informative.

### 4.2 Leave-one-branch-out probes

| ID | Removed from `H0` | Included width |
|---|---|---:|
| `A-noS` | statistical | 375 |
| `A-noT` | transformed | 390 |
| `A-noV` | VAE | 381 |
| `A-noC` | contrastive CNN | 317 |
| `A-noB` | BYOL CNN | 317 |

For each branch `j`, the primary marginal diagnostic is:

```text
metric(H0) - metric(H0 without branch j)
```

The sign must be interpreted using the metric direction: lower error is
better, whereas higher correlation, macro-F1, and balanced accuracy are
better.

### 4.3 Reference and excluded configurations

`H0` is an immutable reference. Its completed predictions may be reused only
after source feature, scaler, head, task-row, checkpoint, and prediction replay
succeeds.

The initial matrix excludes:

- Phase 6 LSTM/Transformer branches and Phase 6.5 deep-LSTM branches;
- duplicated-CNN controls;
- gated fusion or another learned aggregator;
- multiple simultaneous removals beyond the five declared leave-one-out rows;
- pairwise, exhaustive-subset, or Shapley-value enumeration;
- decoder variants;
- raw-input baselines beyond their already completed contextual comparisons;
- hyperparameter tuning; and
- alpha-factor construction.

Those exclusions keep Phase 7A focused on attribution within the canonical
representation rather than reopening architecture selection.

## 5. Downstream execution matrix

The ten new ablation configurations run on all three tasks and both walks:

```text
10 ablations x 3 tasks x 2 walks = 60 new trajectories
```

Together with the six immutable `H0` task/walk references, the Phase 7A report
contains 66 configuration-task-walk entries.

The initial matrix uses seed `0` to match the completed Phase 5/6 probing
contract. Additional seeds are a separate confirmation study and are not a
Phase 7A exit condition. Consequently, small differences must be described as
single-seed observations rather than stable rankings.

Each new trajectory retains the task's existing contract:

- task-training-only coordinate mean and population-standard-deviation
  scaling, followed by clipping to `[-10,10]`;
- the same lightweight `input -> 128 -> 64 -> output` task-head family;
- the same task-specific loss, output constraint, and non-learned references;
- batch size `512`, Adam learning rate `1e-4`, and weight decay `0`;
- one uninterrupted 50-epoch run with snapshots at 5, 15, and 50;
- epoch 50 fixed as the principal result; and
- no validation, early stopping, restart selection, or evaluation-driven
  configuration choice.

Different input widths necessarily change the first task-head layer's
parameter count. Record that count for every configuration. The experiment
therefore measures the practical information-plus-probe effect of including a
branch; it is not a perfectly parameter-matched attribution study.

## 6. Task metrics and comparisons

### 6.1 Movement classification

Primary metrics are macro-F1 and balanced accuracy. Supporting outputs remain
accuracy, weighted-F1, per-class precision/recall/F1, confusion matrix,
predicted counts, one-vs-rest ROC-AUC/PR-AUC, NLL, and Brier score. Always-
`STABLE` and repeated training-prior predictions remain references.

### 6.2 Eight-hour future price

Report price MAE/RMSE/MSE and correlation, current-price persistence, implied-
movement Pearson/Spearman/sign diagnostics, and timestamp-level cross-
sectional Rank IC. The fixed last-hour reversal diagnostic remains a reference.
Do not treat improved level reconstruction alone as directional alpha evidence.

### 6.3 Eight-hour future realised variance

Report raw-unit MAE/RMSE/MSE, Pearson/Spearman, contract-macro results, target
and lifecycle strata, imputation exposure, and tail-error concentration beside
zero, walk-training-median, and historical-volatility-persistence references.

### 6.4 Required report structure

For each task and walk, show:

1. absolute epoch-50 metrics for `H0` and every ablation;
2. all five single-branch results;
3. all five paired `H0` versus leave-one-out differences;
4. task-head parameter counts and feature widths;
5. epochs 5 and 15 as retained training snapshots; and
6. pooled out-of-future metrics only after walk-local predictions have been
   generated and reported separately.

Do not choose a branch from the evaluation results and then report only its
preferred tasks or walks. The complete predeclared matrix, including negative
and inconsistent results, is the evidence.

## 7. Interpretation rules

The conclusions must distinguish:

- **standalone usefulness:** a single branch performs meaningfully above the
  task's non-learned reference;
- **incremental usefulness:** removing a branch consistently worsens an `H0`
  metric under the matched task/walk comparison;
- **redundancy:** a branch is useful alone but removal has little effect in the
  full representation; and
- **interference or optimization effect:** removing a branch improves a metric,
  without implying that the branch contains intrinsically harmful information.

No one metric is sufficient for a universal branch ranking. A branch may help
classification while being redundant for future price or volatility. A full
representation need not beat every single branch on every metric for the
framework to remain useful, and one seed cannot support a strong ordering when
differences are small.

Because stride-one rows and future horizons overlap, ordinary row-wise IID
tests are inappropriate. A separately predeclared contract/calendar-block
resampling procedure is required before reporting paired uncertainty
intervals.

## 8. Artifacts and execution gates

New artifacts belong under:

```text
experiments/phase7/ablation/
  manifests/
  features/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/
  reports/canonical_five_branch_seed0/
```

The implementation must use branch selections from the replayed master feature
stores rather than recomputing canonical encoders. Reusable code belongs under
`src/`; orchestration belongs in the next script generation. The bootstrap
entry point is manifest-only by default and requires an explicit execution
flag to train.

Ordered gates are:

1. replay both source feature stores and all six `H0` task/walk references;
2. freeze branch order, widths, hashes, ten configurations, tasks, rows, seed,
   budgets, and expected artifact paths;
3. build ablation feature views and prove exact coordinate equality with their
   source branch blocks;
4. run CPU forward/loss/backward, scaler, occupied-path, and provenance-failure
   tests for every distinct input width and task head;
5. freeze the complete 66-entry manifest, including immutable `H0` references;
6. execute all 60 new trajectories without result-based truncation;
7. replay checkpoints, predictions, metrics, scalers, branches, and row
   identities; and
8. generate the complete per-walk and pooled report.

## 9. Phase 7B boundary

Phase 7B alpha research is intentionally not specified or authorized by this
document. Its research question, data allocation, search procedure, economic
evaluation, and claim boundary remain open pending further literature review.

Existing raw-OHLCV and direct-representation GP dry runs remain exploratory
implementation evidence only. Phase 7A outputs must not be repurposed into an
alpha search, and no claim of profitable alpha discovery follows from an
ablation result.

## 10. Exit conditions

Phase 7A is complete only when:

- all ten configurations use exact source coordinates and identical ordered
  task rows within each task/walk;
- all 60 new trajectories and six `H0` references pass independent replay;
- single-branch and leave-one-out tables are complete for all tasks and walks;
- feature widths and downstream parameter counts accompany performance;
- negative, redundant, and inconsistent branch effects are retained; and
- the report states the single-seed, correlated-branch, probe-capacity, and
  overlapping-row limitations.

The supported outcome is a task- and walk-specific account of canonical branch
usefulness. Phase 7A cannot by itself establish causal feature importance,
universal branch rankings, robust multi-seed superiority, or profitable alpha.
