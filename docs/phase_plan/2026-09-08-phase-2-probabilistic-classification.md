# Phase 2 Probabilistic Movement and Regime Classification Plan

Date: 2026-09-08
Status: Frozen for execution on 2026-09-12 before Phase 2 test-label statistics
or task-test metrics were generated. Primary `h=2`, `tau=0.005`, protocols
`P0/P1U/P1O/P2`, seeds `0/1/2`, and budgets `15/50/100` are fixed. The launcher
scope was corrected on 2026-09-12 to remove unintended ablation and full-row
runs; the frozen C1/C2/C5 definitions and protocols were unchanged.

> **Execution pause (2026-09-20):** The split-local movement labels, TA-row
> alignment, training-only sampling, priors, and downstream scalers remain
> correctly implemented. However, C1/C2 inputs inherit the shared upstream
> pipeline that fit volume normalisation and generated windows before the stored
> train/test split. Preserve the completed seed-0 artifacts, but do not run the
> remaining matrix until the corrected raw-time-first processed data, encoders,
> features, labels, and TA intersection are regenerated. See
> [`../data_processing_split_contract.md`](../data_processing_split_contract.md).

## Objective

Replace the Phase-1 stock-label transfer task as the primary prediction-market
classification experiment with a split-safe probability-movement task. The
realised target is a deterministic `DOWN`, `STABLE`, or `UP` class. Each model
must return a probability distribution over those classes rather than only an
argmax label. The experiment must also determine whether apparent classifier
collapse is caused by the target distribution, the training distribution, the
loss, the decision rule, or the model input family.

The existing TA-MLP-style `BUY/HOLD/SELL` bundle and all Phase-1 artifacts are
immutable historical controls. They remain evidence about stock-label transfer
and must not be overwritten or compared numerically with the new task as if the
labels were interchangeable.

This branch owns the label contract, classification metrics and saved scores,
classification baselines, anti-collapse controls, orchestration, and reports.
Decoder and encoder variants belong to Parts 1 and 2 under separate experiment
roots; they are not consumed or scheduled by this Part 3 launcher.

## Evidence motivating the change

The Phase-1 label rule is a poor primary contract for prediction markets:

- it uses percentage changes, which are unstable near probability zero;
- it maps moves above an upper cap back to `HOLD`, even though large probability
  jumps are often the most important information-arrival events;
- its class mix moves from 83.1% `HOLD` in training to 50.5% in test;
- the five-branch classifier remains below the exact majority-class accuracy;
  and
- adding BYOL by concatenation reduces accuracy and macro-F1 relative to the
  strictly matched four-branch run.

The last point is evidence about BYOL under the current concat probe, not proof
that BYOL is generally harmful. Relabelling, loss changes, and branch changes
must therefore be tested separately.

## Primary modelling decision

Use deterministic labels as ground truth and probabilistic model outputs. For
each row, the realised future movement is assigned exactly one hard class using
the predeclared horizon and threshold. Every classifier must save

```text
[P(DOWN | x), P(STABLE | x), P(UP | x)]
```

for every eligible row. Argmax labels are derived only for decision metrics and
confusion matrices.

Use train-prior logit-adjusted cross-entropy (`P2`) as the fixed primary
protocol for the C1/C2/C5 comparison. It keeps
all unique training rows while directly targeting majority-class domination.
Retain natural-sampling cross-entropy (`P0`) only as a minimal untreated
reference for estimating the empirical effect of imbalance; `P0` is not a
candidate method and is not eligible for recommendation or deployment.

The saved probabilities are useful scores for ranking and for deriving a class
decision, but confidence calibration is not a research objective in this phase.
Consequently:

1. logit-adjusted cross-entropy is the primary architecture-comparison
   protocol;
2. random undersampling is the source-paper-derived candidate data-level
   method;
3. random oversampling is a second candidate data-level method motivated by
   Buda et al.'s broader empirical comparison; and
4. natural-sampling cross-entropy is run only as an untreated reference needed
   to quantify whether the three candidate methods mitigate collapse on this
   dataset.

Generic label smoothing and post-hoc confidence calibration are excluded from
the initial matrix. Neither directly addresses the class-prior imbalance, and
including them would broaden the experiment beyond its classification focus.

## Frozen data and leakage contract

- Dataset: `data/processed/market_4h_seq64_top50.npz`.
- Feature store: `data/features/features_4h_seq64_top50_phase1.npz`.
- Scope: 4-hour bars, sequence length 64, top 50 contracts.
- Split: the existing chronological 80/20 per-contract boundary only.
- Neural encoders, classifiers, feature scalers, label parameters, and class
  priors use training rows only.
- Train and test labels are built independently. No forecast horizon may cross
  either split endpoint or a contract boundary.
- There is no validation split, early stopping, or test-driven checkpoint,
  threshold, loss, architecture, branch, or seed selection.
- Every strict comparison reuses the same saved label bundle and row identities.
- Phase 2 uses new paths and never overwrites Phase-1 artifacts.
- Because Phase-1 test results motivated this design, reuse of the current test
  partition is characterization evidence. A strong final claim requires a
  fresh, temporally later holdout.

## Proposed primary label contract

For the close probability at the end of an eligible sequence, define

$$
\Delta p_{t,h}=p_{t+h}-p_t.
$$

The proposed frozen values are:

| Setting | Proposed value | Reason |
|---|---:|---|
| Horizon `h` | 2 bars (8 hours) | Matches the Phase-1 forward horizon while changing only the movement definition |
| Threshold `tau` | 0.005 probability points | Interpretable half-percentage-point move with usable train-only class support |
| Classes | `DOWN=0`, `STABLE=1`, `UP=2` | Ordered and explicit |
| Boundary rule | `DOWN` if `delta < -tau`; `UP` if `delta > tau`; otherwise `STABLE` | Makes equality deterministic and replayable |

A train-only audit over the existing 4h rows found approximately 16.35%
`DOWN`, 66.94% `STABLE`, and 16.71% `UP` at `tau=0.005`. This audit did not use
the test rows. The threshold and horizon must be approved and frozen before the
new test labels or their class counts are generated.

`tau=0.01` may be retained as one explicitly named sensitivity bundle. It must
not replace the primary threshold after test inspection. A volatility-scaled
threshold is deferred until the fixed-threshold experiment is complete because
it changes the task semantics and adds a fitted component.

## Controlled imbalance and collapse study

The candidate methods are `P1U`, `P1O`, and `P2`. A minimal `P0` run uses
deterministic labels, ordinary cross-entropy, and unmodified sampling only to
measure the untreated behavior on this exact label bundle and row set. Prior
literature and Phase-1 results motivate expecting harm from imbalance, but do
not replace this within-experiment control.

Define the following protocols once and apply them through the shared
classification runner to the Raw-OHLCV MLP, fixed five-branch concat
classifier, and adapted TA-MLP:

| ID | Target, sampling, and loss | Purpose |
|---|---|---|
| `P0` | Deterministic targets + natural sampling + cross-entropy | Untreated reference only; excluded from candidate selection |
| `P1U` | Deterministic targets + majority-only random undersampling + cross-entropy | Adapt Parente et al.'s data-level imbalance treatment to the project split |
| `P1O` | Deterministic targets + fully balanced random oversampling + cross-entropy | Test Buda et al.'s generally stronger data-level alternative without discarding rows |
| `P2` | Deterministic targets + natural sampling + train-prior logit-adjusted cross-entropy, strength `lambda=1.0` | Test majority-class domination without changing row availability |

`P1U` retains every minority row and samples the unique majority class without
replacement down to the smaller minority-class count. This makes the three
counts approximately equal without silently discarding either minority class.
`P1O` samples with replacement until every class has the count of the largest
training class. Both operate on the training split only; the saved label bundle
and naturally distributed test rows remain unchanged. Save the sampler seed,
original counts, draw counts, retained source indices, and duplicate counts. Do
not combine undersampling, oversampling, class
weighting, focal loss, or logit adjustment in one run. That would make the cause
of any change uninterpretable. Focal loss may be a later named sensitivity; it
is not part of the initial matrix.

Parente et al. report a roughly 70% `HOLD` class and random undersampling of the
majority class to balance their dataset. Their paper then uses a random 70/30
train/test procedure and test-accuracy-driven model selection. `P1U` preserves
the paper's sampling idea, but applies it only after this project's fixed
chronological train split. It is therefore a paper-derived adaptation, not an
exact reproduction of the paper's full evaluation protocol.

For `P2`, compute the class prior `pi_y` from training labels only and optimize

$$
L_{LA}(y,f(x))=-\log
\frac{\exp(f_y(x)+\lambda\log\pi_y)}
{\sum_j\exp(f_j(x)+\lambda\log\pi_j)},\qquad \lambda=1.
$$

Use the unadjusted learned logits `f(x)` for inference. Adding the larger
majority-class prior inside the training loss forces the model to learn a larger
raw margin for a minority positive against a majority negative. This targets
balanced class error; it is not a confidence-calibration procedure. Do not also
apply post-hoc prior subtraction to `P2`, because that would double-adjust the
same prior.

Undersampling, oversampling, and logit adjustment change the training objective
away from the natural-frequency posterior. Their softmax outputs are decision
scores and must not be interpreted as natural-frequency probabilities.

For every epoch, record training-side:

- predicted class counts and mean predicted probabilities;
- predicted class concentration and maximum-class probability;
- per-class recall and macro-F1;
- logit/probability variance; and
- non-finite loss, logits, gradients, or probabilities.

Non-finite values or a constant-probability numerical failure stop the run.
Majority-only predictions are a substantive result and must be reported rather
than hidden by restarting or tuning from test behavior.

## Classification and score evaluation

Each model saves logits, softmax scores, argmax predictions, hard targets,
contract IDs, timestamps/window starts, and source row indices. Saving all three
scores does not make calibration a project objective; it enables ROC/PR ranking
analysis and reproducible alternative decision rules.

Primary metrics:

- macro-F1;
- balanced accuracy and per-class recall; and
- confusion matrix and predicted class counts to expose collapse.

Required secondary outputs:

- accuracy, weighted-F1, and per-class precision/recall/F1;
- macro one-vs-rest ROC-AUC and per-class ROC curves;
- macro average precision/PR-AUC and per-class precision-recall curves;
- predicted versus observed class proportions;
- NLL and multiclass Brier score for `P0` as compact score-quality diagnostics,
  not as a calibration study;
- per-contract metrics and class support; and
- paired row-level differences with uncertainty intervals where applicable.

Use two non-trained references:

1. exact always-`STABLE` prediction for decision metrics; and
2. the training class-prior score vector repeated on every row for the `P0`
   NLL and Brier diagnostics.

For `P0`, also report default argmax and threshold-swept one-vs-rest ROC and
precision-recall curves without changing the trained model.

The curves diagnose whether minority ranking information exists even when the
default argmax predicts mostly `STABLE`. They are characterization outputs, not
permission to select a test-optimal deployment threshold. Any single operating
threshold must be fixed from training-only information or externally specified
misclassification costs.

## Experiment matrices

Use seeds `0,1,2` and epoch budgets `15,50,100` for every learned model. The
framework and Raw-OHLCV MLP retain their predeclared batch size `512`, learning
rate `1e-4`, and train-fitted scaling. The adapted TA-MLP retains its published
project recipe of batch size `64`, learning rate `1e-3`, unchanged architecture,
and train-fitted TA-feature scaling. These model-specific optimization recipes
must be frozen before evaluation; they are not tuned against each other on the
test set. One uninterrupted trajectory per model and seed produces all three
epoch snapshots. Report the entire matrix; do not choose a best test epoch.

### Primary classification matrix

All learned models use the primary `P2` protocol unless their row explicitly
names another protocol.

| ID | Input/fusion/model | Role |
|---|---|---|
| `C0a` | Always `STABLE` | Decision floor |
| `C0b` | Train-prior probabilities | Probabilistic floor |
| `C1` | Raw-OHLCV MLP | Strict raw-input internal baseline |
| `C2` | Five Phase-1 branches, concat, shallow classifier | Canonical architecture on the new task |
| `C5` | Adapted TA-MLP using 36 TA features and the new shared movement-label bundle on the identical eligible rows | Required handcrafted-feature benchmark |

Run candidate protocols `P1U`, `P1O`, and `P2` on `C1`, `C2`, and `C5` as a
matched model-by-imbalance-protocol study. Also run one matched `P0` untreated
reference trajectory per model and seed; report it separately and never rank it
as a candidate method. This establishes whether each remedy improves over the
untreated condition, whether benefits generalize across raw, learned, and
handcrafted inputs, and whether the TA-MLP paper's undersampling choice remains
competitive. Report the complete predeclared matrix and do not use test results
to select one protocol for later models.

Full-row-only C1/C2 runs, representation ablations, `C3` (all-minus-BYOL),
`C4` (gated fusion), and alternative encoder/decoder variants are explicitly
outside the Part 3 classification matrix and are not generated or executed by
its launcher. Decoder and encoder variants remain in Phase 2 under Parts 1 and
2; the other listed configurations are not required by the current Phase 2
plan.

`C5` is an adaptation of the TA-MLP architecture and 36-feature input, not a
source-faithful reproduction of the paper's target or evaluation protocol. The
existing repository BUY/HOLD/SELL artifacts used natural sampling rather than
the paper's majority-class undersampling, so describe them as a historical
natural-sampling adaptation, not as a faithful reproduction. Do not compare
their metrics directly with `C5`, because the outcomes and eligible rows differ.

This document covers Part 3 only. Phase 2 decoder refinement and encoder
refinement are specified in the overall Phase 2 plan and write to separate
experiment roots.

## Implementation tasks

### Task 1: freeze the experiment specification

Approve or revise `h`, `tau`, seeds, budgets, metrics, and the matrix above
using training-side diagnostics only. Freeze `P2` as the primary protocol and
`P0` as the untreated reference before execution. Fix logit-adjustment strength
`lambda=1.0` rather than tuning it on the task test set. Record checksums for
the processed data and Phase-1 feature
store. After freeze, do not inspect new test label counts and then change these
choices.

### Task 2: implement and validate the label bundle

Add:

```text
src/tasks/probability_movement_labels.py
scripts/prepare_probability_movement_labels.py
tests/tasks/test_probability_movement_labels.py
```

Write the primary artifact to:

```text
data/task_labels/trend_classification/
  probability_movement_4h_h2_tau005_seq64_top50.npz
  probability_movement_4h_h2_tau005_seq64_top50.npz.manifest.json
```

Required arrays include deterministic train/test labels, `train_indices`,
`test_indices`, contract IDs, timestamps/window starts, current and future
close, realised `delta`, and class names. The manifest records the formula,
boundary convention, threshold provenance, dropped rows, per-contract/global
counts, source paths/checksums, and reconstruction counts.

Tests cover exact boundaries, zero and near-zero prices, split endpoints,
contract boundaries, missing/non-finite prices, deterministic replay, and
mismatch with processed split sizes.

### Task 3: make classification bundle loading task-generic

Refactor the trend-specific loader in `scripts/train_framework.py` into a
validated classification-label loader. Retain compatibility with the old
tri-class bundle. Add explicit label-mode provenance and store row identities
in every prediction artifact.

The framework runner may retain its validated `--branches` capability for other
studies, but the Part 3 bootstrap always uses all five canonical branches for
C2 and never schedules branch ablations.

### Task 4: add probabilistic losses and metrics

Extend the classification runner with explicit `--classification-loss` values
for `hard_ce` and `logit_adjusted_ce`, plus an independent
`--sampling` choice of `natural`, `majority_undersampling`, or
`balanced_oversampling`. Reject confounded combinations in the initial matrix.
Class priors are computed and saved from training labels only. Implement
balanced accuracy, macro/per-class ROC-AUC and
PR-AUC, curve data, predicted-class diagnostics, `P0` NLL/Brier diagnostics,
and the two reference predictors.

Unit-test metric values, extreme logits, empty classes, score normalization,
sampler determinism and draw counts, the exact logit-adjusted loss equation,
prior provenance, and the fact that test labels never influence loss, sampling,
or decision parameters.

### Task 5: migrate the Raw-OHLCV MLP to the shared multiclass bundle

Modify `src/baselines/mlp_baseline/run_experiment.py` to accept
`--labels-npz`, select raw sequences using the saved split-local indices, use a
three-logit `TrendClassifier`, train with the fixed classification loss, and
write the same probability/identity artifacts and metrics as the framework.
Keep the old binary trend path loadable; do not reinterpret its artifacts.
Support the three candidate protocols and the `P0` untreated reference through
the same shared loss and sampling implementation used by the framework.

### Task 6: build a strictly aligned TA-feature benchmark

Add a required bundle-aware path to the TA-MLP runner while retaining its
existing `triclass` and `binary` modes for provenance. The new mode accepts
`--labels-npz`, computes the unchanged 36-feature TA vector at each saved
sequence-end timestamp, uses `DOWN=0`, `STABLE=1`, and `UP=2` directly from the
shared movement bundle, and validates one-to-one contract/timestamp/order
alignment. It must never call `ta_labels.assign_labels` in this mode.

TA feature warm-up may make some label-bundle rows ineligible. Freeze a common
eligible-row intersection from feature availability alone, save its train/test
row identities, and run `C1`, `C2`, and `C5` on that exact intersection for
the strict TA comparison. Full-row-only C1/C2 runs are not part of this matrix.

Compute the `P1U`/`P1O` sampler counts and `P2` class priors separately from the
exact training rows in each matrix. In the strict TA-aligned matrix, all three
models must therefore share the same training-label hash, test-label hash, and
train-derived prior vector.

Expose the same `hard_ce` versus `logit_adjusted_ce` and `natural` versus
`majority_undersampling` versus `balanced_oversampling` controls as the
framework and Raw-OHLCV MLP. Execute candidate protocols `P1U`, `P1O`, and
`P2`, plus the separately reported `P0` untreated reference, with seeds `0,1,2`
and epoch budgets `15,50,100`. Keep the TA-MLP architecture and TA
feature definitions unchanged so this study changes only labels, eligible rows,
and the explicitly named imbalance protocol.

### Task 7: add orchestration, replay, and reporting

Add a classification Phase 2 runner that expands the frozen matrix, refuses
non-empty output directories unless explicitly overwritten, and verifies all
prediction targets and identities before comparison. Generate per-run,
multi-seed, imbalance-study, per-contract, and consolidated
reports under:

```text
experiments/framework/phase2/classification_relabelling/<run_name>/
```

Every learned run contains `config.json`, `dataset_manifest.json`, parameter
count, timing, training history, checkpoints, logits/scores/targets, metrics,
confusion matrices, curve data, and replay status.

### Task 8: run CPU smoke tests, then the CUDA matrix

Before GPU execution:

1. run focused label, loader, metric, framework, MLP, and TA tests;
2. build the primary bundle and run its validator;
3. run one-epoch CPU smoke tests for `C1`, `C2`, and `C5` under candidate
   protocols `P1U`, `P1O`, and `P2` plus the `P0` untreated reference;
4. replay all smoke artifacts and verify identical target/identity hashes; and
5. freeze the generated matrix manifest.

Then execute seeds `0,1,2` on CUDA. Train all snapshots before evaluating the
task test side, and report every predeclared entry.

### Task 9: write the research judgement and synchronize documents

The final report must separately answer:

- whether the new label contract is more defensible and temporally stable;
- whether the classifier ranks minority-class cases above the relevant
  train-prior and majority-class references;
- whether oversampling improves discrimination or mainly moves the decision
  operating point;
- whether the TA-MLP paper's undersampling control loses useful training
  information relative to natural sampling and oversampling;
- whether training-time logit adjustment improves macro-F1, balanced accuracy,
  and minority recall without simply reversing collapse toward a minority class;
  and
- which conclusions are characterization-only pending fresh later data.

After evidence exists, update `docs/design.md`, `docs/research_plan.md`,
`docs/schedule.md`, `docs/training_test_data_selection.md`, and `AGENTS.md` only
for implemented and executed behavior.

## Optional later regime task

Do not begin a multi-regime task until the three-class probability-movement
bundle and primary matrix are complete. A later training-only feasibility study
may define mutually exclusive regimes such as stable trading, gradual
repricing, information jump, and convergence/resolution. It must predeclare:

- causal, non-overlapping rules and precedence;
- jump, boundary, and convergence thresholds;
- horizon and required history;
- per-contract minimum support; and
- treatment of complementary YES/NO contracts.

If any regime lacks adequate training support, report that result and do not
merge classes after viewing test behavior. The regime task requires its own
bundle, baselines, and matrix; it must not silently replace the movement task.

## Completion gate

This branch's Phase 2 work is complete when:

- the primary deterministic label bundle passes replay and split-safety checks;
- all strict models use identical targets and row identities;
- candidate protocols `P1U`, `P1O`, and `P2` are completed for TA-aligned
  `C1`/`C2`/`C5`, with `P0` reported only as their matched untreated reference;
- the strict three-seed, fixed-budget C1/C2/C5 matrix is reported;
- score-ranking, decision, class-collapse, and per-contract diagnostics are
  present;
- old and new label contracts are interpreted as different tasks; and
- a consolidated judgement clearly labels current-test findings as
  characterization evidence and identifies the fresh-holdout confirmation
  requirement.

## Method references

- Parente, Rizzuti, and Trerotola, *A Profitable Trading Algorithm for Cryptocurrencies Using a Neural Network Model*, Expert Systems with Applications 238, 2024. The paper directly motivates the majority-class undersampling control.
- Menon et al., [Long-Tail Learning via Logit Adjustment](https://arxiv.org/abs/2007.07314), ICLR 2021.
- Maloof, *Learning When Data Sets are Imbalanced and When Costs are Unequal and Unknown*, ICML Workshop, 2003. This is broader classical imbalance background; it is not listed as a direct citation in the supplied Parente et al. PDF.
- Buda, Maki, and Mazurowski, *A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks*, Neural Networks 106, 2018. This is the imbalance study directly cited by Parente et al. at their undersampling step.
