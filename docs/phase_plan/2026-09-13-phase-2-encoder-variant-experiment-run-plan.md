# Phase 2 Encoder Variant Experiment Run Plan

Date: 2026-09-13  
Status: Frozen for implementation and execution before any Part 2 task-test run

## Purpose

This document is the execution specification for Part 2 of
`2026-09-08-phase-2-experiment-plan.md`. It turns the candidate scaffold in
`2026-09-13-phase-2-encoder-refinement-impl.md` into a fixed experiment matrix
for answering one question:

> Does an LSTM or compact Transformer temporal backbone produce a better frozen
> contrastive representation than the canonical CNN backbone when the
> self-supervised objective, representation width, fusion, downstream head,
> task rows, and training budgets are held fixed?

The current global task-test partition has already informed Phase 2. Results
from this matrix are therefore characterisation evidence. A strong
confirmatory claim requires a later temporal holdout that was not used to
design Phase 2.

## Scope and non-goals

The required variants are:

| ID | Branch used in the run | Backbone | Role |
|---|---|---|---|
| `E0` | `contrastive` | canonical CNN | immutable reference |
| `E1` | `contrastive_lstm` | one-layer LSTM | recurrent candidate |
| `E2` | `contrastive_transformer` | compact Transformer | attention candidate |

This experiment does not train BYOL variants, change the NT-Xent objective,
add an extra sixth branch, tune a decoder, change any task label, or select a
checkpoint from task-test performance. The deferred `byol_lstm` and
`byol_transformer` candidates remain behind the separately predeclared BYOL
ablation gate.

The primary comparison replaces exactly one branch:

```text
E0 = statistical + transformed + vae + contrastive             + byol
E1 = statistical + transformed + vae + contrastive_lstm        + byol
E2 = statistical + transformed + vae + contrastive_transformer + byol
```

Every primary representation remains 445-dimensional. Candidate and CNN
contrastive branches must never appear together in the primary bundle.

## Frozen experiment contract

### Data

- Processed input:
  `data/processed/market_4h_seq64_top50.npz`.
- Timeframe, sequence length, universe: `4h`, `64`, top `50` contracts.
- Expected stored split sizes: `109841` train and `27500` test sequences.
- The existing per-contract chronological 80/20 split is the only global
  split. It must not be rebuilt or changed.
- Encoder pretraining reads training sequences only. Reading the test array's
  shape for the manifest is allowed; test values must not enter optimisation
  or training diagnostics.
- Frozen encoder inference may run on both splits only after the complete
  candidate matrix and artifact names are frozen.

Before execution, record SHA-256 hashes for the processed input, canonical
Phase-1 feature bundle and index, canonical CNN checkpoint, VAE checkpoint,
BYOL checkpoint, volatility labels, probability-movement labels, and TA-row
alignment bundle in the Part 2 matrix manifest.

### Backbone and self-supervised settings

The implementation defaults are accepted as the fixed primary architectures:

| Setting | `E1` LSTM | `E2` Transformer |
|---|---:|---:|
| downstream backbone width | 128 | 128 |
| projector output width | 128 | 128 |
| recurrent/encoder layers | 1 | 2 |
| attention heads | n/a | 4 |
| feed-forward width | n/a | 256 |
| dropout | 0.0 | 0.1 |
| norm ordering | n/a | post-norm |
| position encoding | n/a | sinusoidal |
| readout | final hidden state | final token |
| maximum sequence length | 512 | 512 |

Both candidates retain:

- `models.contrastive.make_views` without variant-specific augmentations;
- the existing `nt_xent_loss` with temperature `0.2`;
- the distinction between the unnormalised 128-dimensional backbone state
  used downstream and the normalised projector output used by NT-Xent; and
- AdamW, batch size `256`, learning rate `1e-3`, and weight decay `1e-4`.

Architecture settings must not be revised after viewing downstream test
metrics. A failure such as non-finite loss or collapse permits a documented
bug fix or a separately named exploratory rerun, not a silent matrix change.

### Seeds, budgets, and checkpoint policy

- Encoder seeds: `0,1,2` for each candidate.
- Encoder trajectory: one uninterrupted 100-epoch run per candidate and seed.
- Recorded encoder snapshots: epochs `15,50,100`.
- Downstream probe seeds: `0,1,2`.
- Downstream trajectory: one uninterrupted 100-epoch run per configuration,
  task, and seed.
- Recorded downstream snapshots: epochs `15,50,100`.
- No validation split, early stopping, restart chosen by loss, or
  best-on-test selection is allowed.

Only the predeclared epoch-100 encoder snapshot feeds the downstream matrix.
Epochs 15 and 50 are retained as training-side diagnostics, not alternative
feature stores to be selected after inspection. This prevents an unnecessary
encoder-budget by probe-budget cross-product.

For `E1` and `E2`, encoder seed `s` is paired with downstream probe seed `s`.
For `E0`, the same immutable canonical seed-0 CNN feature bundle is probed with
downstream seeds `0,1,2`. This produces three matched downstream repetitions
without modifying or pretending to replicate the canonical Phase-1 encoder.
The report must state that candidate variability includes encoder
initialisation whereas the fixed CNN reference variability reflects only the
probe. Encoder-seed robustness is therefore directly measured for the
candidates but not for the historical CNN checkpoint.

### Downstream probe contract

All runs use concat fusion and the existing shallow task head with hidden width
`128`, batch size `512`, learning rate `1e-4`, weight decay `0`, train-fitted
feature standardisation, and clipping at `+/-10`. The encoder and all other
feature branches remain frozen.

Two representation views are required:

1. **Primary five-branch substitution:** `E0`, `E1`, and `E2` use the branch
   sets shown above. This is the basis for the main encoder-refinement claim.
2. **Branch-only diagnostic:** probe only `contrastive`,
   `contrastive_lstm`, or `contrastive_transformer`, respectively. This tests
   whether a candidate is independently useful and helps distinguish an
   encoder effect from redundancy with the other four branches. It is
   supporting evidence, not a replacement for the primary substitution.

The three tasks are fixed as follows:

| Task | Rows and target | Primary metrics |
|---|---|---|
| price prediction | existing split-local horizon-1 close target used by Phase 1 | MAE, RMSE |
| volatility prediction | `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz` | MAE, RMSE, MSE, Pearson correlation, negative-prediction fraction |
| probability-movement classification | `h=2`, `tau=0.005` label bundle on the saved TA-eligible row intersection; protocol `P2` only | macro-F1, balanced accuracy, per-class recall, confusion matrix and predicted counts; ROC-AUC, PR-AUC, NLL and Brier as diagnostics |

Classification uses
`data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz`
and alignment bundle
`data/features/phase2_ta_probability_movement_4h_h2_tau005.npz`. `P2`
logit-adjusted cross-entropy is fixed because Part 3 predeclares it for
architecture comparisons. `P0`, `P1U`, and `P1O` belong to the separate Part 3
imbalance study and are not repeated here.

The run count is therefore:

- pretraining: `2 candidates x 3 seeds = 6` uninterrupted trajectories;
- primary downstream: `3 encoders x 3 tasks x 3 paired seeds = 27`
  trajectories;
- branch-only downstream: another `27` trajectories; and
- each downstream trajectory yields the full `15/50/100` snapshot table.

## Required implementation before CUDA execution

The existing repository can train `E1`/`E2` and extract a single candidate
branch, but the complete Part 2 matrix is not runnable yet. Finish these items
before opening the task-test side:

1. Add a substitution-bundle builder that loads the canonical Phase-1 bundle,
   validates split sizes and provenance, removes only `contrastive`, inserts
   exactly one extracted candidate branch, and writes a new five-branch
   `FeatureBundle` plus index and manifest under `data/features/phase2/`.
2. Add a Part 2 bootstrapper that freezes the matrix and commands in a manifest,
   skips only demonstrably complete runs, and refuses partial or occupied run
   directories unless `--overwrite` is explicit.
3. Add a Part 2 reporter that validates dataset/label/row identity hashes,
   aggregates all seeds and budgets, calculates paired differences, and emits
   separate primary-substitution and branch-only tables.
4. Extend resource recording where necessary to include parameter count,
   total training time, frozen-feature inference time, and peak CUDA memory.
5. Ensure the volatility and classification runners save aligned row
   identities and that classification uses the existing P2 runner rather than
   the old BUY/HOLD/SELL `TrendClassifier` path.
6. Add focused tests for substitution rather than addition, split-index
   preservation, provenance mismatch rejection, bootstrap matrix size, resume
   behavior, and report-time row-identity rejection.

The bootstrapper should create a manifest without training by default and
require an explicit `--execute` flag, matching the safe Part 3 pattern.

## Execution procedure

### Gate 1 — repository and environment readiness

From the project root:

1. Confirm the worktree and record the commit in the matrix manifest. Do not
   overwrite or clean unrelated worktree changes.
2. Run the focused encoder, feature-store, framework, volatility-label, and
   Phase 2 classification tests.
3. Verify that WSL CUDA is available and record GPU model, driver, CUDA, PyTorch,
   and package versions.
4. Verify the source hashes and expected array shapes without rewriting any
   artifact.
5. Run one-epoch CPU smoke tests for both candidates into disposable smoke
   run names, then remove or clearly exclude those runs from the matrix.

No task-test metric may be inspected during this gate.

### Gate 2 — freeze the executable matrix

Run the new bootstrapper without `--execute`. Inspect its manifest and confirm:

- the two candidate architectures and all fixed hyperparameters above;
- seeds `0,1,2` and budgets `15,50,100`;
- the epoch-100-only feature-extraction rule;
- all expected pretraining, feature, primary, and branch-only commands;
- the exact task-label and alignment hashes; and
- distinct Phase 2 paths that do not overlap Phase 1 or Part 3 outputs.

After this review, treat the manifest as immutable. Any approved correction
must replace it before task-test execution and record why it changed.

### Gate 3 — candidate pretraining

For each seed, run the two existing trainers on CUDA. For seed 0 the commands
are:

```bash
.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_lstm \
  --run-name contrastive_lstm-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --temperature 0.2 \
  --device cuda

.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_transformer \
  --run-name contrastive_transformer-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --temperature 0.2 \
  --device cuda
```

Repeat with seeds 1 and 2 and matching run names. Do not pass `--overwrite`
during normal execution. After every trajectory, require:

- checkpoints, histories, and metrics at all three budgets;
- finite losses, gradients, embedding statistics, and model weights;
- backbone embedding standard deviation at or above `1e-3` at every recorded
  snapshot;
- dataset and configuration hashes matching the frozen manifest; and
- recorded parameter count, elapsed training time, and resource information.

A collapse warning or non-finite value blocks that run from feature extraction.
It does not authorise choosing another epoch from the task-test results.

### Gate 4 — frozen feature extraction and substitution bundles

Extract only the canonical copied epoch-100 checkpoint for each candidate and
seed. The seed-0 LSTM pattern is:

```bash
.venv/bin/python3 scripts/extract_phase2_encoder_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --checkpoint checkpoints/phase2/contrastive_lstm-4h-seq64-top50-seed0.pth \
  --out-path data/features/phase2/contrastive_lstm_4h_seq64_top50_seed0.npz \
  --batch-size 1024 \
  --device cuda
```

Repeat for both candidates and all seeds, then use the new substitution-bundle
builder. Every candidate bundle must contain exactly five branches with widths
`70,55,64,128,128`, preserve the Phase-1 train/test index, contain only finite
values, and record both the source Phase-1 bundle hash and candidate checkpoint
hash. Run `scripts/validate_feature_store.py` with the candidate's exact branch
list and compare row counts against the processed input.

Do not overwrite `data/features/features_4h_seq64_top50_phase1.npz` or any
canonical encoder checkpoint.

### Gate 5 — downstream matrix

Execute the bootstrapper's commands in this order:

1. all `E0` reference probes;
2. `E1` primary and branch-only probes;
3. `E2` primary and branch-only probes; and
4. price, volatility, and classification within each encoder/seed group.

The order is operational only and must not be used for selection. Each
trajectory trains on train rows, saves all fixed-budget checkpoints, and then
evaluates every predeclared snapshot on the unchanged task-test rows. Complete
the whole matrix even when an early result is weak. Resume only through the
bootstrapper's completeness checks; inspect a partial directory rather than
blindly overwriting it.

Expected roots are:

```text
experiments/framework/phase2/encoder_refinement/pretraining/<run_name>/
experiments/framework/phase2/encoder_refinement/<task>/primary/<encoder>/seed<seed>/
experiments/framework/phase2/encoder_refinement/<task>/branch_only/<encoder>/seed<seed>/
data/features/phase2/<encoder>_4h_seq64_top50_seed<seed>.npz
data/features/phase2/five_branch_<encoder>_4h_seq64_top50_seed<seed>.npz
```

### Gate 6 — replay, aggregation, and interpretation

Before comparing metrics, the reporter must reject any run with mismatched
processed-data hash, feature hash, label hash, branch set, seed, budget, target
row identity, or prediction length. Recompute metrics from saved predictions
and require replay agreement within the reporter's declared numerical
tolerance.

Report every epoch budget. For each task, view, encoder, and budget, report the
three seed-level results plus mean and sample standard deviation. Compare `E1`
and `E2` with `E0` at the same downstream seed and budget using saved,
row-aligned predictions. Include paired per-row error differences for
regression, paired correctness/class-recall changes for classification, and
95% paired bootstrap intervals. Where contract IDs are available, use
contract-cluster resampling and state the resampling seed and draw count.

Resource tables must include trainable parameters, encoder training time,
feature-extraction time, task training time, inference time, and peak CUDA
memory. Parameter mismatch is reported as a limitation; it is not corrected
after seeing task-test performance.

The conclusion must distinguish:

- the primary five-branch substitution result from branch-only diagnostics;
- representation effects from the unchanged shallow decoder;
- task-specific gains from consistency across all three tasks;
- candidate encoder-seed variability from fixed-CNN probe variability; and
- characterisation on the existing test split from later-data confirmation.

No single best epoch or universal winner is selected from this matrix. A
candidate may be described as promising only when its direction and magnitude
are reported across all budgets, seeds, tasks, and resource costs.

## Stop conditions and failure handling

- **Infrastructure or provenance failure:** stop before task-test evaluation,
  fix the defect, regenerate the dry-run manifest, and document the change.
- **Non-finite or collapsed encoder run:** quarantine the affected artifacts;
  do not substitute them downstream. The required matrix is incomplete until
  the predeclared run succeeds or the failure is reported as an unresolved
  candidate failure.
- **CUDA interruption:** retain the partial directory for inspection. The
  current trainer does not resume mid-trajectory, so rerunning requires an
  explicit, documented replacement of that exact run.
- **Metric replay or row mismatch:** exclude the affected comparison and fix
  the pipeline before reporting.
- **Weak performance:** continue the matrix and report it. Weak results are not
  a reason to tune the architecture on the locked test side.

## Completion gate

Part 2 encoder refinement is complete only when:

- all six candidate pretraining trajectories and their fixed snapshots pass
  health and provenance checks;
- all six epoch-100 candidate branch artifacts and substitution bundles pass
  finite-value, dimension, split-index, and hash validation;
- the complete `E0`/`E1`/`E2` primary and branch-only downstream matrices are
  evaluated for all three tasks, seeds, and budgets on identical eligible rows;
- saved predictions replay exactly enough to reproduce the reported metrics;
- multi-seed aggregate, paired-comparison, and resource-cost reports exist;
- no Phase-1 or Part 3 artifact was overwritten;
- conclusions remain specific to encoder substitution and acknowledge that the
  current test split provides characterisation evidence; and
- `docs/schedule.md` and other affected project documentation are synchronized
  from completed artifacts rather than from planned commands.

Implementation alone, a seed-0 result, successful pretraining without
downstream probes, or an incomplete task matrix does not satisfy this gate.
