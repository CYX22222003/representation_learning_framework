# Phase 3 Experiment Plan

Date: 2026-09-20  
Status: Encoder pretraining and seed-0 framework price matrix complete

> **Execution update (2026-09-20):** The leakage-safe 4-hour bundle was built
> and validated with 21,696 train windows, 3,085 test windows, 50 contracts,
> and processed SHA-256
> `a8eb367be99bb1485b34186073940024b05806409df68d52dd83aa4728e6693a`.
> All seven seed-0 encoder trajectories completed in the predeclared order and
> produced checkpoints at epochs 5, 15, and 50. The consolidated validation
> covers 21 snapshots with no collapse warnings. Numerical and visual records
> are stored under
> `experiments/phase3/reports/encoder_pretraining_seed0/`. Epoch 50 remains the
> predeclared checkpoint for later frozen feature extraction; downstream task
> metrics were not inspected or used during pretraining.

> **Price execution update (2026-09-20):** New `scripts_v2` entry points
> now build and replay contract-local horizon-1 labels, extract the two
> window-local deterministic branches and all seven frozen epoch-50 neural
> branches, train the complete H0/HC/HB substitution/addition/duplicate-control
> matrix, and replay/report its predictions. Statistical AR/GARCH and
> FFT/Haar features consume only their corresponding causal input window.
> Price labels require the next consecutive window inside the same contract and
> stored split, dropping each contract's terminal row. The validated label
> bundle contains 21,646 train and 3,035 test rows. The master nine-branch
> feature bundle and all 15 seed-0 price trajectories are complete, with
> snapshots at epochs 5, 15, and 50. Replay validation covers all 45 snapshots
> on identical held-out identities. The best observed row is `HB-ALT` at epoch
> 50 (MAE 0.009234, RMSE 0.022782, correlation 0.998993); this is exploratory
> locked-test evidence, not a test-selected deployment checkpoint.

> Phase 3 must not execute against any legacy processed bundle, label bundle,
> feature store, checkpoint, or experiment artifact. All runnable Phase 3 entry
> points belong in `scripts_v2/`. Existing files under `scripts/` may be read
> for ideas and reviewed source modules under `src/` may be imported, but a
> Phase 3 command must never invoke an existing `scripts/` entry point.

## 1. Purpose and scope

Phase 3 rebuilds the experimental evidence after the leakage and alignment
problems found during Phase 1 and Phase 2. It has three ordered subtasks:

1. self-supervised encoder pretraining;
2. frozen feature extraction and downstream evaluation; and
3. matched baseline experiments.

The initial execution scope is fixed to:

- 4-hour OHLCV data;
- sequence length 64;
- 50 contracts selected without test-period information;
- chronological 80/20 raw-time split independently per contract;
- seed `0` only;
- one uninterrupted training trajectory with snapshots at epochs `5`, `15`,
  and `50`;
- price prediction with a contract-local horizon-1 target; and
- three-class probability-movement classification using the Phase-2 idea of
  `DOWN`, `STABLE`, and `UP`.

Volatility prediction and its baselines are explicitly deferred. Legacy
results may motivate the design, but they are not Phase 3 comparison rows.

## 2. Research questions

Phase 3 initially asks:

1. Do CNN, LSTM, and Transformer backbones produce different useful frozen
   representations under the same contrastive or BYOL objective?
2. Do heterogeneous backbones within one SSL family provide complementary
   downstream information when used together?
3. Does the leakage-safe frozen representation improve price prediction over
   matched raw-input baselines?
4. Does it improve probability-movement classification over matched raw-OHLCV,
   handcrafted TA, and non-trained class-prior references?

Substitution and addition answer different questions. A substitution result is
evidence about encoder quality at fixed representation width. An additive
result is evidence about complementarity only when it also beats a matched
duplicate-feature width control.

## 3. Non-negotiable leakage contract

The following order is mandatory:

```text
raw contract timelines
    -> validate timestamps and OHLCV schema
    -> establish each contract's raw 80/20 boundary
    -> compute universe-selection statistics from training prefixes only
    -> freeze the top-50 universe
    -> fit imputation and volume scaling on selected training prefixes only
    -> transform train and test portions with frozen training parameters
    -> construct windows independently inside each split
    -> construct labels independently inside each contract and split
    -> validate identities, coverage, hashes, and boundary invariants
    -> pretrain encoders on training windows only
    -> freeze encoders
    -> extract train and test features separately
    -> fit downstream preprocessing and models on aligned training rows only
    -> evaluate the complete predeclared matrix on locked test rows
```

Identical stored row counts or hashes downstream do not prove that preprocessing
was leakage-safe. Every Phase 3 processed artifact must trace back to raw row
ranges and train-fitted preprocessing parameters.

### 3.1 Universe selection

The legacy full-file-size ranking is prohibited because later observations can
affect contract membership. Phase 3 uses a fixed 256-row selection prefix. A
candidate must contain at least 320 chronological rows, which guarantees that
all 256 selection rows lie within its eventual 80% training side. Rank by the
count of finite positive-volume observations in that fixed prefix, then by the
sum of `log1p(max(volume, 0))` over the same prefix, with the stable filename as
the final tie-breaker. Later rows may establish that the contract can produce
both train and test windows, but their OHLCV values, missingness, activity, or
outcomes must not affect the ranking score. The manifest records every
candidate's prefix length, two training-only activity scores, split boundary,
eligibility decision, rank, and source hash.

### 3.2 Preprocessing and windows

- Establish the raw boundary before fitted imputation, scaling, or windows.
- Fit fill values and volume statistics using the raw training prefix only.
- Apply causal forward filling in chronological order and never backfill from a
  future row. Forward filling may carry the last observed training value into
  leading missing cells on the test side because that observation is available
  at the forecast boundary. This is a causal boundary-imputation policy, not a
  shared-window policy. If no prior observed value exists, use the fill value
  fitted from the training prefix.
- Construct train and test windows independently. No raw observation may occur
  in both stored splits under the initial isolated-window policy.
- Preserve contract ID, window start/end raw indices, sequence-end timestamp,
  split, and preprocessing provenance for every window.

The isolation guarantee concerns raw window membership: stored training and
test windows use disjoint raw-row indices. It does not require discarding
causally available training history when imputing a leading missing test cell.
The critical direction remains strict: no test-period value may influence a
training fill value, fitted statistic, transformed training row, or training
window. The audited 4-hour top-50 source currently contains no missing or
non-finite OHLCV cells, so the boundary carry-forward rule does not alter its
generated values; the policy is documented for deterministic behavior if a
future source contains missing cells.

### 3.3 Labels

Price prediction uses the close at horizon 1. The final eligible input row of
every contract and split is dropped. A target must never cross a contract or
train/test boundary.

Probability movement uses:

```text
delta = close[t + 2] - close[t]
DOWN   when delta < -0.005
STABLE when abs(delta) <= 0.005
UP     when delta > 0.005
```

The horizon `2` and fixed threshold `0.005` are frozen for the initial Phase 3
matrix. The last two rows of each contract and split are ineligible. Labels,
current/future close, contract IDs, timestamps, and source row identities must
be saved. Test rows retain their natural class distribution and are never
resampled.

## 4. Training and model-selection policy

Phase 3 inherits the Phase 1/2 train/test-only policy:

- no validation partition;
- no early stopping;
- no restart selected by training or test behavior;
- no hyperparameter, threshold, epoch, or configuration selected from test
  metrics; and
- report every predeclared configuration and snapshot.

Each learned run initializes its model and optimizer once, trains continuously
through epoch 50, and saves snapshots after epochs 5, 15, and 50. These are
snapshots from one trajectory, not three independent runs.

For encoders, all three snapshots and training-only health diagnostics are
retained, but only the predeclared epoch-50 snapshot feeds the principal frozen
feature and downstream matrix. This prevents an encoder-snapshot by downstream-
snapshot search. Downstream heads and baselines report all `5/15/50` snapshots
without naming the lowest test error as a selected model.

Seed `0` is the only seed in the initial matrix. Results are therefore
single-seed characterization evidence. Multi-seed confirmation is a later,
separately frozen extension.

## 5. Subtask 1 — encoder pretraining

### 5.1 Encoder matrix

| Family | Variant | Downstream dimension | Role |
|---|---|---:|---|
| VAE | canonical MLP encoder-decoder | 64 | Fixed generative branch |
| Contrastive | CNN | 128 | Canonical NT-Xent reference |
| Contrastive | LSTM | 128 | Recurrent NT-Xent candidate |
| Contrastive | Transformer | 128 | Attention NT-Xent candidate |
| BYOL | CNN | 128 | Canonical BYOL reference |
| BYOL | LSTM | 128 | Recurrent BYOL candidate |
| BYOL | Transformer | 128 | Attention BYOL candidate |

VAE remains a single canonical MLP encoder-decoder. Phase 3 does not introduce
a VAE backbone matrix because changing only its encoder would make the
generative architecture asymmetric, while changing its encoder and decoder
would no longer be the same isolated backbone-substitution question used for
contrastive and BYOL.

Within each contrastive or BYOL family, keep the objective, view augmentation,
projector semantics, embedding width, optimizer recipe, batch size, and training
rows fixed wherever architecture permits. BYOL LSTM and Transformer variants
must retain distinct online and EMA target encoders plus the canonical
projector/predictor roles.

### 5.2 Encoder artifacts and health checks

Every encoder run stores its full configuration, source commit, processed-data
and manifest hashes, architecture, parameter count, optimizer state, training
history, `e5/e15/e50` checkpoints, elapsed time, peak device memory, and finite-
value checks. Contrastive and BYOL runs also store representation variance and
collapse diagnostics. VAE stores reconstruction and KL components.

The pretraining stage never evaluates test task labels. Frozen inference on
test sequences begins only after all seven encoder run specifications and
artifact paths are frozen.

## 6. Subtask 2 — feature extraction and downstream evaluation

### 6.1 Frozen branch inventory

The deterministic branches remain:

- statistical: 70 dimensions;
- transformed: 55 dimensions.

Together with the seven neural outputs, Phase 3 can construct explicitly named
feature bundles. Every bundle records source checkpoint hashes, branch order,
branch dimensions, train/test identity hashes, finite-value results, and total
width. A feature validator must reject an absent or unexpected branch, a source
hash mismatch, a split mismatch, non-finite values, or reordered identities.

### 6.2 Primary encoder-quality and complementarity matrix

The common reference is:

```text
H0 = statistical + transformed + VAE + contrastive CNN + BYOL CNN
```

Run the contrastive and BYOL families separately:

| ID pattern | Change from H0 | Purpose |
|---|---|---|
| `*-SL` | replace the family's CNN with LSTM | Fixed-width substitution |
| `*-ST` | replace the family's CNN with Transformer | Fixed-width substitution |
| `*-AL` | retain CNN and add LSTM | Single-branch complementarity |
| `*-AT` | retain CNN and add Transformer | Single-branch complementarity |
| `*-DC` | add one explicit duplicate CNN block | Matched-width control |
| `*-ALT` | retain CNN and add LSTM and Transformer | Three-backbone family |
| `*-DD` | add two explicit duplicate CNN blocks | Matched-width control |

Use prefix `HC` for contrastive and `HB` for BYOL. The two families are
reported independently. A later model combining temporal variants from both
families is outside the initial matrix and must be frozen separately.

Concat fusion and the same shallow task-head architecture are primary. Wider
additive bundles necessarily increase first-layer parameters, which is why the
duplicate controls are required. Gated or fixed-width fusion is deferred.

### 6.3 Downstream tasks

#### Price prediction

- Use the saved Phase 3 contract-local horizon-1 bundle.
- Select eligible rows before fitting the feature standardizer.
- Fit all standardization using selected training rows only.
- Report MAE, RMSE, MSE, Pearson correlation, per-contract results, and the
  exact saved predictions/targets/identities needed for replay.
- Include a no-training current-close persistence reference.

#### Probability-movement classification

- Use the frozen `h=2`, `tau=0.005` bundle.
- Use the common TA-eligible row intersection for strict framework, Raw-OHLCV
  MLP, and TA-MLP comparisons.
- Use training-only class priors and Phase 2's fixed `P2` logit-adjusted
  cross-entropy (`lambda=1.0`) as the primary learned-model protocol.
- Record a matched natural-sampling cross-entropy `P0` trajectory as an
  untreated reference, not as a candidate chosen from test results.
- Report macro-F1, balanced accuracy, accuracy, weighted-F1, per-class
  precision/recall/F1, confusion matrix, predicted class counts, macro/per-
  class ROC-AUC and PR-AUC, NLL, Brier score, and per-contract results.
- Include exact always-`STABLE` and repeated training-prior references.

All configurations in a strict comparison consume identical ordered targets
and row identities. Every metric must replay from saved predictions or logits.

## 7. Subtask 3 — baseline experiments

Phase 3 baseline results live under `experiments/phase3/baselines/`, not under
`src/baselines/` or any legacy experiment directory. Baselines must consume the
same Phase 3 processed bundle, saved labels, eligible rows, epoch snapshots,
seed, and metric functions as the framework.

### 7.1 Initial price baselines

1. current-close persistence reference;
2. Raw-OHLCV MLP using flattened aligned windows; and
3. stacked price LSTM rebuilt so raw splitting and label construction occur
   before supervised windows can overlap or cross boundaries.

The Raw-OHLCV MLP and framework shallow head should be capacity-described, but
they need not have identical parameter counts because one consumes raw values
and the other consumes frozen features. Claims must identify this as a complete-
system comparison rather than a decoder-isolating comparison.

### 7.2 Initial classification baselines

1. always-`STABLE` reference;
2. repeated training-prior probability reference;
3. Raw-OHLCV MLP; and
4. adapted TA-MLP using its fixed 36 handcrafted features and the shared Phase
   3 movement labels.

TA feature warm-up determines a common eligible-row intersection from feature
availability only. All strict classification models are rerun on that exact
intersection. Sampling, loss priors, and feature scaling use training rows
only. Test rows remain untouched.

Volatility baselines, GARCH--LSTM, Raw LSTM volatility, and GINN are not part of
this initial Phase 3 execution.

## 8. `scripts_v2/` implementation boundary

No existing `scripts/` file is a Phase 3 runner, even if inspection later finds
its logic correct. Phase 3 adds entry points on demand under `scripts_v2/`.
Stable implementation code should live in reviewed modules under `src/` and be
imported by the new entry points rather than importing executable `scripts/`
modules.

The planned entry-point responsibilities are:

```text
scripts_v2/
    prepare_phase3_sequences.py
    validate_phase3_sequences.py
    prepare_phase3_labels.py
    validate_phase3_labels.py
    train_phase3_encoder.py
    validate_phase3_checkpoint.py
    extract_phase3_features.py
    build_phase3_feature_bundles.py
    validate_phase3_features.py
    train_phase3_downstream.py
    run_phase3_price_baseline.py
    run_phase3_classification_baseline.py
    bootstrap_phase3.py
    validate_phase3_run.py
    report_phase3.py
```

Names may be split further during implementation, but every new entry point
must fail closed on missing manifests, legacy paths, hash mismatches, identity
mismatches, unexpected label contracts, non-finite inputs, occupied output
directories, or incomplete prerequisite artifacts.

`bootstrap_phase3.py` is manifest-first and non-executing by default. An
explicit execution flag is required to launch work. It may schedule only
commands from `scripts_v2/` and must reject paths under `_old` roots.

## 9. Artifact layout

Generated data inputs are isolated from legacy inputs:

```text
data/phase3/processed/
data/phase3/task_labels/
data/phase3/features/
```

All experiment records, including model checkpoints used by Phase 3, are
contained under:

```text
experiments/phase3/
    manifests/
    encoder_pretraining/<family>/<backbone>/seed0/
    downstream_evaluation/<task>/<matrix_id>/seed0/
    baselines/<task>/<baseline_id>/seed0/
    reports/<task_or_family>/
```

Every learned run contains configuration, environment and source provenance,
data/label/feature hashes, histories, all fixed-budget checkpoints, predictions
or scores, aligned targets and identities, metrics, replay status, parameter
count, runtime, and memory use. Existing content is never overwritten by
default.

## 10. Validation and execution gates

### Gate A — raw data and processed bundle

- Unit tests cover training-only universe ranking, raw-boundary placement,
  fitted preprocessing, disjoint raw-window coverage, causal boundary
  imputation, the invariant that test changes cannot affect training outputs,
  deterministic replay, and malformed timestamps/data.
- The full 4-hour top-50 artifact is generated once and independently audited.
- Provenance proves that no training input or fitted statistic depends on a
  test-period value.

### Gate B — labels and identities

- Price and movement targets reconstruct exactly from raw provenance.
- No horizon crosses a contract or split boundary.
- Dropped terminal rows and row counts reconcile per contract.
- Framework and baseline loaders produce identical identity and target hashes.

### Gate C — CPU smoke tests

- Every new model family completes a disposable one-epoch CPU smoke test.
- Smoke artifacts use clearly excluded paths and never count as matrix runs.
- Non-finite, shape, resume, occupied-directory, and provenance failures are
  tested.

### Gate D — freeze

- Generate the complete immutable matrix manifest without executing it.
- Record exact commands, paths, hashes, architectures, seed, budgets, metrics,
  and expected artifact inventory.
- Review the manifest before any CUDA training or locked-test inference.

### Gate E — encoder pretraining and feature extraction

- Train all seven seed-0 encoder configurations from training windows only.
- Validate every checkpoint and training history.
- Extract only predeclared epoch-50 embeddings for the primary matrix.
- Validate every branch and constructed bundle before downstream training.

### Gate F — downstream and baselines

- Complete training for the entire frozen matrix before comparative ranking.
- Evaluate and retain all `5/15/50` snapshots.
- Replay all metrics and verify row identities before reporting.
- A failed run is repaired only for a documented implementation fault; its
  configuration is not silently tuned from test behavior.

### Gate G — reporting

- Report the complete matrix, not only favorable configurations.
- Separate substitution, addition, and duplicate-control conclusions.
- Compare framework and baselines only on identical labels and rows.
- State compute and parameter costs alongside performance.
- Label all current-split results as characterization evidence.

## 11. Interpretation and confirmation limits

The existing test period influenced the lessons that produced Phase 3. A
leakage-safe rerun makes its measurements internally valid, but does not turn
the already-inspected period into a fresh confirmatory holdout. Phase 3 results
on this period are therefore characterization evidence.

A strong final model-selection or superiority claim requires a separately
frozen, temporally later holdout that has not influenced architecture, label,
baseline, or protocol decisions. Until then:

- do not call the best observed snapshot a selected model;
- do not combine winning branches after reading test metrics and present the
  combination as confirmatory;
- do not compare Phase 3 absolute metrics directly with legacy metrics as if
  preprocessing and target contracts were identical; and
- describe gains by task and comparison level rather than as universal model
  superiority.

## 12. Completion criteria

The initial Phase 3 scope is complete only when:

1. the 4-hour top-50 processed artifact passes the end-to-end raw provenance
   and leakage audit;
2. price and movement bundles pass contract/split boundary replay;
3. all seven encoder trajectories and their `5/15/50` snapshots are complete;
4. epoch-50 frozen features and every declared substitution/addition/control
   bundle pass validation;
5. the complete single-seed price and classification downstream matrices are
   evaluated on identical saved rows;
6. all initial price and classification baselines are rerun under the same
   data and target contracts;
7. every reported metric replays from immutable predictions and identities;
   and
8. the final report states limitations, resource costs, and the fresh-holdout
   requirement.
