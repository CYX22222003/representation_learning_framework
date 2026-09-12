# Phase 2 Experiment Plan

Date: 2026-09-08

## Purpose

Phase 2 studies how the completed Phase-1 framework can be improved without
losing the ability to identify why an improvement occurred. It has three
separate experiment parts:

1. decoder refinement;
2. encoder refinement; and
3. classification-task relabelling.

The alpha-research infrastructure continues in parallel as a separate research
track. It is not a Phase 2 task, completion gate, or fourth Phase 2
contribution.

Phase 2 extends rather than replaces Phase 1. The canonical Phase-1 system
remains the frozen five-branch reference:

| Branch | Current implementation | Dimension |
|---|---|---:|
| `statistical` | AR and GARCH features | 70 |
| `transformed` | FFT and Haar wavelet features | 55 |
| `vae` | MLP VAE encoder | 64 |
| `contrastive` | CNN backbone trained with NT-Xent | 128 |
| `byol` | CNN online/target backbone trained with BYOL | 128 |
| **Concat representation** | all five frozen branches | **445** |

The original checkpoints, feature bundle, task-label bundles, experiment
directories, and reports must remain unchanged and reproducible.

## Motivation from Phase-1 evidence

The Phase-1 results motivate the three parts but do not determine their
outcomes:

- The price representation beat the strict Raw-OHLCV MLP at the 15-epoch
  budget, but not at 50 or 100 epochs. Falling training loss alongside worsening
  test error makes uncontrolled increases in decoder capacity risky.
- On the shared volatility label rows, the frozen five-branch framework beat
  Raw LSTM at every recorded budget and was close to the GARCH--LSTM stack on
  RMSE/MSE at 15 and 50 epochs. The representation is therefore informative,
  while fusion and downstream extraction remain plausible bottlenecks.
- The current volatility head produced invalid negative predictions, so task
  constraints must be part of decoder design rather than a post-hoc clipping
  choice.
- The five-branch trend classifier was worse than the matched four-branch
  system, and the train/test class proportions shifted substantially. This
  motivates the encoder-refinement question and a prediction-market-specific
  label study; it does not establish that BYOL is generally harmful or that
  relabelling will automatically solve classification.

These observations were obtained from the existing task-test partition and
informed Phase 2. Consequently, Phase 2 results on that partition are
characterisation evidence. Strong confirmatory claims require a fresh,
temporally later holdout that was not used to design this plan.

## Research questions

Phase 2 answers three questions independently:

1. **Decoder:** Can branch-aware or temporal decoding extract information from
   the frozen Phase-1 representation that the static shallow MLP does not?
2. **Encoder:** Does changing the temporal backbone of a self-supervised branch
   improve its frozen representation while retaining the same learning
   objective and downstream evaluation contract?
3. **Classification target:** Does a prediction-market-specific probability
   movement target provide a more defensible and stable classification task
   than the transferred stock-market BUY/HOLD/SELL rule?

The plan must not answer one question by changing factors assigned to another.

### Experiment identifiers

| Phase 2 part | Identifiers | Scope |
|---|---|---|
| Part 1: decoder refinement | `D0`–`D4` | Fixed Phase-1 encoders; decoder changes only |
| Part 2: encoder refinement | named variants such as `contrastive_lstm` and `contrastive_transformer` | Fixed SSL objective, fusion, labels, and shallow probe |
| Part 3: classification relabelling | references `C0a`/`C0b`; learned models `C1`, `C2`, `C5` | Fixed probability-movement label contract and strict shared rows |

These identifiers are local to their experiment part. There is no executable
`A_*` or `B_*` Phase 2 family. The historical `C2_framework_full` artifacts
used the same probability-movement labels on all label-eligible rows, but that
secondary characterization is not in the canonical Part 3 matrix.

## Global experiment rules

The following rules apply to every Phase 2 part:

1. Use the existing chronological 80/20 split per contract. Do not re-split the
   stored Phase-1 data.
2. Train neural encoders, aggregators, decoders, scalers, and label thresholds
   on training rows only.
3. Preserve contract identity, chronological order, split boundaries, and
   forecast-horizon safety when constructing any new sequence or label bundle.
4. Use fixed epoch budgets with no validation split and no early stopping.
5. Predeclare the complete configuration matrix, seeds, budgets, and primary
   metrics before starting its task-test evaluation. Report the full matrix;
   do not select a checkpoint from test metrics.
6. Fit feature standardisers and any target transformations on training rows
   only, then reuse those fitted objects on test rows.
7. Strict comparisons must consume identical targets and row identities.
8. Preserve Phase-1 artifacts. Phase 2 uses new run names and directories and
   must not overwrite canonical checkpoints or reports.
9. Match parameter counts where practical and always report parameter counts,
   training time, and inference time so architectural gains are not presented
   without their resource cost.
10. Use multiple fixed seeds for the final predeclared matrix and report mean,
    standard deviation, and paired uncertainty where row-aligned predictions
    permit it.
11. Treat configurations created after inspecting Phase 2 task-test results as
    exploratory additions. Confirm any resulting combined design only on fresh
    later data.

Before execution, a separate experiment specification must freeze any values
left open by this overall plan, including temporal context length `K`, exact
hidden dimensions, dropout, seeds, epoch budgets, label horizon, and movement
threshold.

---

## Part 1: Decoder refinement

### Objective

Hold the five Phase-1 representation branches fixed and determine whether the
downstream limitation is static head capacity, branch fusion, or the absence of
temporal modelling across consecutive embeddings.

### Isolation rule

All decoder configurations use the same frozen Phase-1 checkpoints and the
same task rows. No VAE, contrastive, or BYOL parameter is updated during this
study. A decoder result therefore supports a claim about downstream extraction,
not improved self-supervised representation learning.

### Decoder matrix

| ID | Representation input | Decoder | Purpose |
|---|---|---|---|
| `D0` | one 445-d concat vector | Current shallow MLP | Preserve the Phase-1 reference |
| `D1` | projected named branches | Residual MLP | Control for additional nonlinear capacity and cross-branch interaction |
| `D2` | gated branch projections | Current shallow MLP | Test task-dependent branch weighting separately from temporal modelling |
| `D3` | `K` consecutive 445-d concat vectors | Compact LSTM | Primary recurrent temporal decoder |
| `D4` | `K` consecutive 445-d concat vectors | Compact Transformer | Attention-based temporal comparison |

`D1` and `D2` must not be described as temporal models. `D3` and `D4` must use
the same ordered embedding sequences, context length, targets, and row
eligibility. Their trainable parameter counts should be reasonably matched.
`D2` is a Part 1 decoder/fusion experiment and is distinct from the removed
Part 3 `C4` classification configuration.

### Temporal embedding-sequence contract

A genuine temporal decoder consumes a sequence of representations:

```text
K consecutive eligible OHLCV windows from one contract
        -> frozen five-branch representation for each window
        -> tensor [batch, K, 445]
        -> LSTM or Transformer decoder
        -> task prediction after the final context position
```

Sequence construction must:

- remain within one contract;
- use increasing timestamps with no shuffled temporal membership;
- never cross the train/test boundary;
- exclude contexts with missing predecessors or non-contiguous window starts;
- align the prediction target to the final context position using the existing
  task horizon; and
- save source row indices, contract IDs, timestamps/window starts, and the
  final target row for replay verification.

Applying an LSTM or Transformer to a single 445-dimensional vector does not
test temporal modelling and is outside this matrix.

### Recommended compact temporal designs

The implementation specification may refine these values before execution,
but the intended scale is:

- **LSTM:** one recurrent layer, hidden size near 128, optional input
  projection, dropout confined to explicitly supported locations, and the
  final valid hidden state passed to the task output layer.
- **Transformer:** input projection from 445 to approximately 128, positional
  encoding, two encoder blocks, four attention heads, feed-forward dimension
  near 256, modest dropout, and a predeclared final-token or pooled readout.

The Transformer context contains only information available by the prediction
time. If a causal attention mask is used, that choice must be fixed in the
experiment specification and applied consistently.

### Task-specific decoder requirements

- **Price prediction:** preserve the current shared price-target contract. A
  bounded probability output or residual probability-change formulation may be
  tested only as a separately named, predeclared target/output experiment.
- **Volatility prediction:** reuse the saved realised-volatility label bundle
  and use a nonnegative output transformation such as Softplus. Raw Phase-1
  linear-output results remain the historical reference; clipping new
  predictions after evaluation is not the primary method.
- **Classification:** do not use decoder results on the old stock-derived label
  task to choose the final Phase 2 classifier. The principal Phase 2
  classification comparison occurs after Part 3 creates the new label bundle.

### Decoder outputs and analysis

For each configuration, save configuration, seed, model state, training
history, parameter count, fitted standardiser, predictions, aligned targets,
row identities, metrics, and timing. Report:

- price MAE and RMSE;
- volatility MAE, RMSE, MSE, Pearson correlation, and negative-prediction
  fraction;
- classification accuracy, macro-F1, weighted-F1, per-class metrics, and the
  confusion matrix after Part 3; and
- paired per-row error differences and per-contract results where possible.

### Part 1 completion gate

Part 1 is complete when all predeclared decoder configurations have been run on
identical eligible rows for the supported tasks, replay checks pass, the full
multi-seed matrix is reported, and gains are attributed separately to static
capacity, branch weighting, recurrence, or attention.

---

## Part 2: Encoder refinement

### Objective

Test new temporal backbones as additional, explicitly named variants of the
current self-supervised encoders. The canonical CNN contrastive and CNN BYOL
models remain unchanged as reference branches.

### Isolation rule

Change one encoder backbone at a time. Keep the self-supervised objective,
augmentations, projector semantics, downstream embedding dimension, data,
training budget, and downstream probe fixed wherever architecture permits.
The primary encoder comparison uses the current Phase-1 shallow decoder rather
than a decoder selected from the same task-test results.

### Priority and candidate variants

| Priority | Variant | Fixed objective | New backbone | Status in Phase 2 |
|---:|---|---|---|---|
| 0 | `contrastive` | NT-Xent | Existing CNN | Immutable reference |
| 0 | `byol` | BYOL | Existing CNN | Immutable reference |
| 1 | `contrastive_lstm` | NT-Xent | LSTM | Primary recurrent encoder variant |
| 1 | `contrastive_transformer` | NT-Xent | Compact Transformer | Primary attention encoder variant |
| 2 | `byol_lstm` | BYOL | LSTM | Conditional secondary variant |
| 2 | `byol_transformer` | BYOL | Compact Transformer | Conditional secondary variant |

The two contrastive variants are the required encoder-refinement comparison.
BYOL variants are conditional secondary work and are not part of the Phase 2
completion gate unless they are separately predeclared before execution. This
does not convert the current single-seed trend result into a general rejection
of BYOL.

### Fair backbone comparisons

Each variant must:

- train only on the existing training sequences;
- retain a 128-dimensional frozen downstream representation unless the
  predeclared comparison explicitly studies dimension;
- retain the existing contrastive or BYOL projector/predictor role rather than
  treating the projector as the temporal backbone;
- use the same augmentations and loss definition as its CNN reference;
- use new checkpoints, feature keys, manifests, and run names; and
- record architecture and parameter-count differences explicitly.

The principal full-framework comparison substitutes one named variant for its
corresponding CNN branch while preserving the total branch count and embedding
dimension. For example:

```text
statistical + transformed + vae + contrastive_transformer + byol
```

This is an experimental substitution inside a new configuration, not a
repository-wide replacement of `contrastive`. Adding both CNN and Transformer
versions to the same bundle is a separate additive-branch experiment because
it changes representation size and redundancy; it must not be used as the
primary backbone comparison.

### Required encoder evaluations

For each new encoder variant:

1. run pretraining diagnostics and check for non-finite values or representation
   collapse;
2. extract frozen train/test embeddings with a provenance manifest;
3. substitute it into the matched five-branch framework and rerun the same
   fixed probe; and
4. compare against its CNN reference on identical task rows over all three
   tasks, using the relabelled task for the primary classification result.

### Part 2 completion gate

Part 2 is complete when the two primary contrastive variants and their CNN
reference have matched pretraining and downstream reports using the fixed
five-branch substitution design. Any BYOL variants are separately scoped
secondary work.

---

## Part 3: Classification-task relabelling

### Objective

Create a prediction-market-specific classification target while retaining the
existing TA-MLP-style BUY/HOLD/SELL task as a documented stock-label transfer
experiment.

The primary task uses deterministic realised movement labels as ground truth.
Every classifier outputs and saves a three-class probability vector; argmax is
used only for decision metrics. Training-time train-prior logit-adjusted
cross-entropy (`P2`) is the fixed primary protocol because it retains all unique
training rows while addressing majority-class domination. Paper-derived
majority-class random undersampling (`P1U`) and balanced random oversampling
(`P1O`) are the other candidate methods. Natural-sampling cross-entropy (`P0`)
is retained only as an untreated within-experiment reference and is excluded
from method selection. The treatments must not be combined in the first
comparison.
Generic label smoothing and post-hoc confidence calibration are outside the
initial classification scope.

### Primary target definition

Use absolute probability movement over a predeclared horizon:

\[
\Delta p_{t,h}=p_{t+h}-p_t.
\]

Define:

\[
y_{t,h}=\begin{cases}
\text{UP}, & \Delta p_{t,h}>\tau,\\
\text{DOWN}, & \Delta p_{t,h}< -\tau,\\
\text{STABLE}, & |\Delta p_{t,h}|\leq\tau.
\end{cases}
\]

The horizon `h` and threshold rule `tau` must be fixed before test-label
evaluation. `tau` may be a fixed number of probability points or fitted from
training rows only. A volatility-scaled threshold is a separately named
sensitivity experiment, not an unreported replacement for the primary label.

### Label-bundle contract

The new bundle must:

- build train and test labels independently so no horizon crosses the split;
- fit thresholds from training data only and store their provenance;
- retain contract IDs, timestamps/window starts, aligned feature-row indices,
  horizon, class names, and per-contract/global class counts;
- validate that prices and future movements are finite and eligible;
- record rows dropped because a future horizon is unavailable; and
- be reused unchanged by the framework, Raw-OHLCV baseline, and any adapted
  technical-feature benchmark used for a strict comparison.

### Classification matrix

The primary comparison holds the new label bundle fixed:

| Configuration | Purpose |
|---|---|
| Exact majority-class predictor | Minimum class-imbalance reference |
| Raw-OHLCV MLP | Strict raw-input internal baseline |
| Current five-branch concat + shallow classifier | Phase-1 architecture on the new task |
| Adapted TA-MLP using its unchanged 36 TA features and the new shared movement-label bundle | Required handcrafted-feature comparison |

The adapted TA-MLP must accept the saved label bundle rather than rebuilding
the upstream BUY/HOLD/SELL labels. Match contract IDs, sequence-end timestamps,
split-local indices, class ordering, and eligible rows one-to-one. If TA feature
warm-up removes rows, freeze a common eligible-row intersection and rerun the
Raw-OHLCV MLP, five-branch framework, and TA-MLP on that same subset. The
existing repository TA-MLP results used natural sampling, whereas Parente et
al. report majority-`HOLD` random undersampling. Existing artifacts are
therefore a historical natural-sampling adaptation, not a source-faithful
reproduction, and cannot be compared directly with the new-label results.

Apply the three candidate imbalance protocols—paper-derived majority
undersampling (`P1U`), balanced oversampling (`P1O`), and training-time
logit-adjusted cross-entropy (`P2`)—to the Raw-OHLCV MLP, five-branch framework,
and adapted TA-MLP. Run `P0` once per matched model and seed only as the
untreated reference needed to quantify each method's improvement on this task.
This tests whether an anti-collapse method generalizes across input
representations and whether Parente et al.'s data-level choice remains
competitive on the new task.

For this strict submatrix, derive sampling counts and priors from the identical
aligned training labels and require matching train/test identity hashes across
all three models. Retain each model's predeclared optimizer and batch-size
recipe rather than tuning them from test comparisons.

Results from the old and new label definitions answer different questions.
Accuracy or F1 values must not be compared as though the targets were
interchangeable. Instead, compare class balance, temporal stability,
per-contract support, confusion patterns, and performance relative to the
exact majority reference within each label contract.

For the new task, also save logits and the three softmax scores for reproducible
ranking and decision analysis. Primary evaluation uses macro-F1, balanced
accuracy, per-class recall, confusion matrices, and predicted-class counts;
macro/per-class one-vs-rest ROC-AUC and PR-AUC diagnose whether minority ranking
exists despite an argmax collapse. NLL and multiclass Brier score are compact
`P0` diagnostics, not a calibration research track. No test-optimal threshold
may be selected. Oversampled or logit-adjusted scores must not be described as
natural-frequency probabilities.

### Part 3 completion gate

Part 3 is complete when the label definition and bundle are documented and
validated, every strict model consumes identical rows, class-distribution
diagnostics are reported for train and test, and the full predeclared
classification matrix—including the TA-aligned candidate `P1U`, `P1O`, and
`P2` submatrix for the Raw-OHLCV MLP, five-branch framework, and adapted TA-MLP,
plus the separately reported matched `P0` untreated reference—is evaluated
across fixed seeds and budgets.

---

## Separation of the three parts

The central interpretation rule is:

```text
Decoder study:
    fixed Phase-1 encoders and labels -> vary decoder only

Encoder study:
    vary one backbone -> fixed SSL objective, fusion, labels, and probe

Relabelling study:
    fixed model configurations -> vary the label contract
```

The first reports for each part should remain factorially clean. A later
exploratory system may combine a promising encoder, decoder, and new label, but
its result cannot identify the contribution of any one change. Because such a
combination would be informed by prior task-test observations, its strong final
evaluation belongs on a fresh temporal holdout.

## Execution order and dependencies

1. **Freeze Phase-1 evidence and write Phase 2 specifications.** Preserve all
   existing artifacts and predeclare unresolved architecture/data values.
2. **Build shared Phase 2 infrastructure.** Add temporal embedding-sequence
   indexing, manifest validation, configuration naming, artifact replay, and
   common multi-seed reporting.
3. **Implement the new classification label bundle.** This can proceed in
   parallel with decoder implementation because it does not modify the
   representation branches.
4. **Execute decoder refinement first on price and volatility.** These tasks
   retain established target contracts and provide the cleanest initial
   decoder evidence.
5. **Execute primary encoder refinement with the fixed shallow probe.** Start
   with `contrastive_lstm` and `contrastive_transformer`; keep any BYOL variants
   as separately predeclared secondary work.
6. **Execute the relabelled classification matrix.** Run only the strict
   TA-aligned C1/C2/C5 configurations under P0/P1U/P1O/P2. Decoder and encoder
   variants remain in their own Part 1 and Part 2 experiment roots.
7. **Produce cross-part analysis.** Separate representation, fusion, temporal
   decoding, and label-contract conclusions.
8. **Run fresh temporal confirmation when suitable later data exists.** Freeze
   any combined Phase 2 candidate before this one-time evaluation.

Implementation and training may overlap where dependencies permit, but the
interpretation and artifact contracts must preserve this order.

## Proposed artifact organisation

Use explicit Phase 2 paths without modifying Phase-1 directories:

```text
data/features/phase2/
data/task_labels/trend_classification/probability_movement_*.npz
checkpoints/phase2/
experiments/framework/phase2/decoder_refinement/<task>/<run_name>/
experiments/framework/phase2/encoder_refinement/<task>/<run_name>/
experiments/framework/phase2/classification_relabelling/<run_name>/
```

Every experiment directory should contain at least:

```text
config.json
dataset_manifest.json
sweep_metrics.json
summary.md
report.md
e<budget>/metrics.json
e<budget>/predictions.npz
e<budget>/history.npz
```

Add architecture-specific manifests where needed, including temporal row maps,
encoder checkpoint provenance, and label-bundle provenance.

## Parallel alpha-research track

Alpha research is explicitly outside Phase 2 scope and does not block any of
the three completion gates.

Already demonstrated infrastructure includes train-only raw-OHLCV factor
screening, bounded GP, direct-representation exploration, and timestamped
discovery/confirmation reporting without use of the global task-test rows.
These remain infrastructure and exploratory evidence rather than a Phase 2
model comparison.

The intended framework-facing alpha path is not yet equivalent to those dry
runs. Remaining work includes chronological out-of-fold predictions from the
downstream heads, economically named prediction terminals, train-only symbolic
selection, and eventual evaluation on a fresh later holdout. Alpha work must
not use Phase 2 test results to select primitives or formulae.

## Phase 2 deliverables

Phase 2 should produce:

1. a decoder-refinement specification and complete D0–D4 comparison report;
2. validated temporal embedding-sequence contracts for the recurrent and
   attention decoders;
3. frozen `contrastive_lstm` and `contrastive_transformer` encoder variants,
   feature manifests, and matched downstream reports against the CNN reference;
4. a validated probability-movement classification label bundle;
5. a strict TA-aligned C1/C2/C5 classification report under P0/P1U/P1O/P2;
6. multi-seed aggregate tables, paired uncertainty estimates, and resource
   comparisons for each part; and
7. a consolidated Phase 2 judgement document that keeps decoder, encoder, and
   label-contract conclusions separate.

The Part 3 classification launcher does not run decoder variants, encoder
variants, full-row-only comparisons, or representation ablations. Parts 1 and
2 remain Phase 2 work and use their own experiment roots.

## Overall completion criteria

Phase 2 is complete when:

- Parts 1, 2, and 3 each satisfy their completion gates;
- decoder effects, encoder effects, and label-contract effects are evaluated
  in separate experiment matrices;
- all models in a strict comparison use identical eligible rows and targets;
- fixed-budget, multi-seed results and resource costs are reported without
  best-on-test selection;
- Phase-1 artifacts remain reproducible and are used as explicit controls;
- no result confuses an experimental backbone substitution with replacement of
  the canonical CNN branches;
- old and new classification labels are interpreted as different task
  contracts;
- a consolidated report states which conclusions are confirmatory,
  characterisation-only, or still provisional; and
- any combined final Phase 2 system is frozen before evaluation on fresh,
  temporally later data.

## Documentation synchronization during implementation

This plan records the approved Phase 2 direction. As each design becomes fixed
or completed, synchronize only the affected material in:

- `docs/Research_Ideas_Writeup.md` for architecture and evaluation framing;
- `docs/design.md` for implemented components and experiment contracts;
- `docs/research_plan.md` for the stable three-part Phase 2 roadmap;
- `docs/schedule.md` for evidence-backed status and next actions;
- `docs/training_test_data_selection.md` for temporal-sequence and new-label
  allocation rules; and
- `AGENTS.md` for runnable commands, branch dimensions, modules, and data
  contracts after implementation exists.

Do not mark a decoder, encoder variant, label bundle, or experiment complete
merely because it appears in this plan.
