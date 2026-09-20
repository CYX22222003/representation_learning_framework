# Phase 2 Horizontal Encoder-Branch Expansion Proposal

Date: 2026-09-20  
Status: Reviewed

## Purpose

This proposal defines the horizontal direction of the next encoder study:
retain the canonical representation branches and add frozen neural branches
with different encoding architectures.

The principal question is:

> Do heterogeneous encoders preserve complementary information that improves
> downstream performance when their frozen embeddings are used together?

This is an additive representation-diversity study. It is distinct from the
vertical capacity proposal, where one branch is replaced and the branch count
and total representation width remain fixed.

## Relationship to the current Phase 2 plan

The canonical five-branch Phase-1 representation remains:

```text
statistical (70)
+ transformed (55)
+ vae (64)
+ contrastive CNN (128)
+ BYOL CNN (128)
= 445 dimensions
```

The frozen selected-branch amendment uses substitution to determine whether an
LSTM or Transformer is a better backbone than its corresponding CNN. This
proposal does not supersede that comparison. It adds a separate question:
whether CNN, LSTM, and Transformer features are useful together.

The current task-test split has already informed Phase 2 and the creation of
this proposal. Any result on that split is characterisation evidence. A
combined additive system must be frozen before evaluation on a fresh,
temporally later holdout for a strong confirmatory claim.

## Hypotheses

The study separates four hypotheses:

1. **Complementarity hypothesis:** different encoder architectures preserve
   non-redundant task-relevant information.
2. **Dimension hypothesis:** performance changes because the concatenated
   representation has more coordinates.
3. **Decoder-parameter hypothesis:** performance changes because a wider input
   creates more parameters in the first task-head layer.
4. **Ensemble hypothesis:** combining several individually useful but
   different encoders is more robust across tasks and seeds than any single
   substitution.

An additive model supports the complementarity hypothesis only when it is
compared with width and duplicate-information controls.

## Why addition and substitution must both be retained

Substitution and addition answer different questions:

```text
Substitution:
    replace byol CNN with byol Transformer
    -> is the Transformer representation individually better?

Addition:
    retain byol CNN and add byol Transformer
    -> does the Transformer contribute information not already in BYOL CNN?
```

Substitution remains the primary encoder-quality comparison. Addition is the
primary branch-complementarity comparison. Neither result should be used as a
proxy for the other.

## Primary isolation rules

Keep fixed across the additive matrix:

- processed data, chronological split, task labels, and aligned rows;
- frozen encoder checkpoints within a seed;
- 128-dimensional output per neural branch;
- concat fusion for the primary study;
- shallow D0-style task-head architecture apart from its required input width;
- train-fitted feature scaling;
- downstream seeds, budgets, optimiser, and metrics; and
- no validation split, early stopping, or test-selected configuration.

All neural branches are pretrained on training sequences only. Frozen encoder
inference on test sequences is permitted only after the candidate matrix and
artifact names are fixed.

## Proposed contrastive-family matrix

The first additive matrix should vary one SSL family at a time while retaining
the other canonical CNN branch.

| ID | Branch configuration | Width | Purpose |
|---|---|---:|---|
| `H0` | canonical five branches | 445 | Reference |
| `HC-SL` | replace contrastive CNN with contrastive LSTM | 445 | LSTM substitution control |
| `HC-ST` | replace contrastive CNN with contrastive Transformer | 445 | Transformer substitution control |
| `HC-AL` | canonical five + contrastive LSTM | 573 | Test CNN+LSTM complementarity |
| `HC-AT` | canonical five + contrastive Transformer | 573 | Test CNN+Transformer complementarity |
| `HC-DC` | canonical five + duplicate contrastive CNN features | 573 | Width and decoder-parameter control |
| `HC-ALT` | canonical five + contrastive LSTM + contrastive Transformer | 701 | Required three-backbone interaction test |
| `HC-DD` | canonical five + two duplicate contrastive CNN branches | 701 | Width and decoder-parameter control for `HC-ALT` |

`HC-ALT` is a required part of the predeclared contrastive-family matrix. It
tests whether the LSTM and Transformer provide complementary information to
each other as well as to the CNN reference. It may be executed operationally
after the single-addition arms, but its inclusion, configuration, seeds, and
budgets must be frozen before any new task-test results are inspected.

`HC-DD` is required alongside `HC-ALT`. It matches the 701-dimensional input
width and downstream-head parameter count while adding no independently
encoded information. Without this control, an `HC-ALT` gain could not be
separated from the effect of two additional feature blocks and their extra
decoder parameters.

## Proposed BYOL-family matrix

The BYOL family mirrors the contrastive matrix:

| ID | Branch configuration | Width | Purpose |
|---|---|---:|---|
| `H0` | canonical five branches | 445 | Shared reference |
| `HB-SL` | replace BYOL CNN with BYOL LSTM | 445 | LSTM substitution control |
| `HB-ST` | replace BYOL CNN with BYOL Transformer | 445 | Transformer substitution control |
| `HB-AL` | canonical five + BYOL LSTM | 573 | Test CNN+LSTM complementarity |
| `HB-AT` | canonical five + BYOL Transformer | 573 | Test CNN+Transformer complementarity |
| `HB-DC` | canonical five + duplicate BYOL CNN features | 573 | Width and decoder-parameter control |
| `HB-ALT` | canonical five + BYOL LSTM + BYOL Transformer | 701 | Required three-backbone interaction test |
| `HB-DD` | canonical five + two duplicate BYOL CNN branches | 701 | Width and decoder-parameter control for `HB-ALT` |

`HB-ALT` is a required part of the predeclared BYOL-family matrix. It tests
whether the BYOL LSTM and Transformer contribute complementary information to
each other as well as to the BYOL CNN reference. As with `HC-ALT`, it may be
scheduled after the single-addition arms, but its complete configuration must
be frozen before any new task-test results are inspected.

`HB-DD` is required alongside `HB-ALT` to match its 701-dimensional input width
and downstream-head parameter count without adding independently encoded
information.

Contrastive and BYOL additive families should be reported separately. A model
that adds temporal variants from both families is a later combined experiment,
not part of either primary family-level conclusion.

## Duplicate-branch capacity control

Concat fusion increases the input dimension of the first task-head layer. A
new 128-dimensional branch therefore adds both features and trainable decoder
weights. `HC-DC` and `HB-DC` control for a single addition by repeating an
existing frozen branch. `HC-DD` and `HB-DD` extend the same control to the two
additions in `HC-ALT` and `HB-ALT`:

- the representation width matches the corresponding single-addition model;
- the downstream head receives the same number of input coordinates and has
  the same parameter count; but
- no genuinely new representation information is introduced.

If `HC-AL` or `HC-AT` improves over `H0` but not over `HC-DC`, the evidence is
consistent with input-width or optimisation effects rather than encoder
complementarity. If `HC-ALT` improves over the single-addition arms but not
over `HC-DD`, its apparent gain is likewise consistent with its larger input
and task head. Apply the same interpretation to `HB-AL`/`HB-AT` versus
`HB-DC`, and to `HB-ALT` versus `HB-DD`.

The duplicated branch must be stored or constructed with explicit provenance
and a distinct diagnostic name. It must never be described as an independent
encoder.

A frozen random or noise branch may be included as a secondary robustness
control, but it is not a replacement for the duplicate-information control.

## Representation-diversity diagnostics

Before downstream evaluation, calculate training-side diagnostics for every
candidate pair:

- linear Centered Kernel Alignment (`linear CKA`), using the fixed contract
  below;
- coordinate-correlation summaries;
- covariance spectrum and effective rank of individual and concatenated
  representations;
- nearest-neighbour agreement;
- temporal smoothness by contract;
- branch-only probe performance;
- incremental linear-probe performance when one branch is added; and
- non-finite, variance, and collapse checks.

### Linear CKA contract

Linear CKA is the required primary representation-similarity metric; it must
not be replaced by a metric chosen after inspecting downstream results. For
two aligned frozen representation matrices `X` and `Y`, centre every feature
using its mean over the shared training rows and compute:

```text
linear_CKA(X, Y) = ||X_c^T Y_c||_F^2
                   / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
```

Use the same ordered training-row identities for both matrices, operate on the
full aligned training split where feasible, and accumulate cross-products in
float64. Test rows must not be used for candidate filtering or architecture
decisions. Report the complete pairwise CKA matrix for every encoder seed,
plus the across-seed mean and sample standard deviation for each named pair.

At minimum, report pairwise CKA among the CNN, LSTM, and Transformer branches
within each SSL family. For `HC-ALT` and `HB-ALT`, also report their pairwise
CKA values alongside the corresponding duplicate controls. The duplicate
branches should produce CKA numerically equal or extremely close to `1.0`;
this acts as an implementation sanity check.

These diagnostics are descriptive and must not be used to choose a candidate
after seeing current task-test results. If they are used to reduce the matrix,
the filtering rule and threshold must be frozen using training data only.

Low similarity does not automatically imply useful complementarity, and high
similarity does not automatically imply redundancy for every task. The final
claim must use downstream paired comparisons as well as representation
diagnostics.

## Downstream evaluation contract

Use the unchanged shallow probe as the primary decoder. The input layer may
grow to accept 573 or 701 dimensions, but all subsequent hidden widths,
activations, losses, optimiser settings, and budgets remain fixed.

Use the same tasks and rows:

| Task | Required target contract | Primary metrics |
|---|---|---|
| Price prediction | Existing split-local horizon-1 close target | MAE, RMSE |
| Volatility prediction | Shared realised-volatility label bundle | MAE, RMSE, MSE, Pearson correlation |
| Probability movement | `h=2`, `tau=0.005`, common TA-eligible rows, P2 | Macro-F1, balanced accuracy, per-class recall, ROC-AUC and PR-AUC diagnostics |

Run downstream seeds `0,1,2`, retain every predeclared budget, and save aligned
predictions and row identities. Comparisons must use the same encoder seed,
probe seed, targets, and rows wherever applicable.

## Artifact-first recording and on-demand comparison

Complete and verify the raw artifacts for every predeclared run before making
cross-run comparisons. Each run must independently record its configuration,
source and row hashes, encoder and probe seeds, complete training history,
every fixed-budget checkpoint, raw predictions or class scores, aligned
targets and row identities, per-budget metrics, pairwise linear CKA results,
parameter counts, timing, memory, feature-store hashes, and extraction costs.

No dedicated comparison, aggregation, ranking, or winner-selection script is
required for this proposal. Comparisons may be produced on demand from the
immutable raw artifacts for a specific research question or report section.
Each derived note or table must identify the source artifact paths, included
configurations, seeds, budgets, metric definitions, and any uncertainty
calculation.

On-demand analysis does not permit selective reporting. A comparison must use
all predeclared configurations, seeds, and budgets relevant to its question,
and it must not choose a configuration because it has the lowest observed
task-test error.

## On-demand comparison questions

For each family, the retained raw artifacts must support:

1. substitution versus canonical reference;
2. addition versus canonical reference;
3. addition versus the corresponding substitution;
4. addition versus the duplicate-branch capacity control;
5. branch-only performance for every distinct encoder;
6. leave-one-branch-out performance for each added configuration; and
7. each required `ALT` configuration versus its corresponding `AL` and `AT`
   configurations to measure the marginal contribution of the other temporal
   encoder; and
8. `HC-ALT` versus `HC-DD`, and `HB-ALT` versus `HB-DD`, to separate
   multi-encoder complementarity from increased width and decoder
   parameterisation.

When one of these comparisons is requested, derive absolute metrics,
seed-level results, mean and sample standard deviation, and—when a strong claim
is made—paired row-level or contract-cluster bootstrap intervals. Include
parameter count, training and inference time, peak memory, feature storage,
and feature-extraction cost.

## Fixed-width fusion as a secondary analysis

The primary concat study deliberately exposes the practical cost of adding a
branch. A secondary fixed-width analysis may test whether complementarity
survives after all configurations are mapped to the same downstream width.

Possible methods include a predeclared branch-projection bottleneck or the
existing gated aggregator. This is not a pure branch-addition comparison
because it introduces learned fusion parameters. It must therefore use
separate identifiers and be reported as a fusion interaction study, not as the
primary evidence of branch complementarity.

## Execution stages

1. Complete or validate the required frozen encoder branches and checkpoints.
2. Freeze the additive, substitution, and duplicate-control matrix.
3. Run training-only diversity and sufficiency diagnostics, including all
   pairwise relationships inside `HC-ALT` and `HB-ALT`.
4. Build additive feature bundles with explicit branch names and provenance.
5. Validate dimensions, split indices, hashes, finite values, and exact branch
   membership.
6. Freeze all shallow-probe commands and run names.
7. Execute the complete matched multi-seed downstream matrix.
8. Verify that every run's raw metrics replay from its saved predictions and
   that the complete artifact inventory is present.
9. Produce cross-run comparisons only on demand from the immutable artifacts;
   no dedicated comparison script is required.
10. If justified, freeze a small combined-family or fixed-width fusion study.
11. Confirm any selected additive system only on fresh temporally later data.

## Artifact proposal

```text
experiments/framework/phase2/encoder_refinement_horizontal/
├── checkpoints/<encoder_variant>/seed<seed>.pth
├── features/branches/<encoder_variant>/seed<seed>.npz
├── features/supersets/<family>/seed<seed>.npz
├── diagnostics/linear_cka/<family>/seed<seed>/
└── probes/<task>/<configuration>/seed<seed>/
```

Each bundle manifest should record:

- every branch name and dimension;
- source encoder checkpoint and feature hash;
- whether a branch is independently encoded or duplicated;
- train/test sizes and index hashes;
- total representation width;
- downstream input-layer parameter count; and
- the canonical Phase-1 feature-bundle hash.

Raw per-run artifacts are the primary output. Derived comparison notes or
tables are secondary products and may be saved alongside the relevant research
write-up when requested; they do not require a permanent comparison script.

## Interpretation constraints

- An additive gain over `H0` alone does not establish complementarity.
- Beating the duplicate-branch control is the minimum evidence that the new
  branch contributes more than increased input width and decoder parameters.
- A substitution gain supports individual encoder quality; an additive gain
  supports complementarity only under the required controls.
- A branch that is weak alone may still be complementary. Conversely, two
  strong standalone branches may be redundant together.
- Improvements must be described per task unless they are consistent across
  tasks and seeds.
- The shallow probe remains primary. Testing several decoders belongs to a
  later encoder-decoder interaction experiment with separately frozen scope.
- Current-split results are characterisation evidence, not fresh confirmation.

## Frozen execution decisions

The executable matrix freezes the following choices before inspecting new
task-test results:

- both the contrastive and BYOL families are included;
- LSTM and Transformer candidate encoders use 128-dimensional outputs and
  epoch-100 checkpoints from the predeclared `15,50,100` training trajectory;
- encoder seed `s` is paired with downstream-probe seed `s` for `s=0,1,2`;
- `H0` and duplicate controls retain the existing canonical encoder features
  while varying the downstream seed;
- duplicate branches are explicit aliases named `contrastive_dup1`,
  `contrastive_dup2`, `byol_dup1`, and `byol_dup2`;
- leave-one-out coverage concerns the newly added temporal branches and is
  represented by each `ALT` arm versus its `AL` and `AT` arms;
- linear CKA is descriptive and does not filter the matrix;
- every downstream task retains budgets `15,50,100`; and
- fixed-width fusion is deferred to a separately frozen study.

The exact commands and source hashes are frozen in
`experiments/framework/phase2/encoder_refinement_horizontal/matrix_manifest.json`.
The launcher records raw artifacts only; cross-run comparison remains an
on-demand analysis and has no dedicated comparison script.
