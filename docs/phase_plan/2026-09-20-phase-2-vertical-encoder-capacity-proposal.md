# Phase 2 Vertical Encoder-Capacity Experiment Proposal

Date: 2026-09-20  
Status: Draft for review; blocked from execution by upstream data correction

> Do not freeze or execute this proposal against the legacy processed and
> feature bundles. Resume only after the raw-time-first split and preprocessing
> contract in [`../data_processing_split_contract.md`](../data_processing_split_contract.md)
> is implemented and validated.

## Purpose

This proposal defines the vertical direction of the next encoder study:
increase the capability of one self-supervised encoder while keeping the
number of representation branches and the downstream evaluation contract
fixed.

The principal question is:

> Does a deeper encoder, or a depth-matched encoder with residual connections,
> produce a more useful frozen representation than the current shallow
> backbone?

This study concerns within-branch representation capacity. It is separate from
the horizontal branch-expansion proposal, which asks whether several encoders
provide complementary features when used together.

## Relationship to the current Phase 2 plan

The canonical five-branch Phase-1 representation and the frozen Phase-2
selected-branch amendment remain unchanged. This document is a proposed
follow-up and does not supersede
`2026-09-14-phase-2-selected-branch-encoder-amendment.md`.

The existing seed-0 contrastive LSTM/Transformer pilot is prior
characterisation evidence. It showed that temporal backbones can improve price
or movement classification while degrading volatility, so lower
self-supervised loss or greater model capacity must not be treated as a
task-independent improvement.

The current task-test split has already informed the design of Phase 2.
Results from this proposal on that split are therefore characterisation
evidence. A strong final claim requires a fresh, temporally later holdout.

## Hypotheses

The study separates three hypotheses:

1. **Depth hypothesis:** additional nonlinear layers improve the information
   captured by the frozen representation.
2. **Residual hypothesis:** residual connections improve optimisation or
   information preservation beyond the effect of depth alone.
3. **Capacity hypothesis:** a larger residual model provides additional gains
   after the depth and residual effects have been separated.

A lower pretraining loss alone does not confirm any of these hypotheses. The
primary evidence is matched downstream probing plus representation-sufficiency
diagnostics.

## Isolation rules

For a primary comparison, change exactly one encoder backbone. Keep fixed:

- the processed `4h`, sequence-length-64, top-50 dataset;
- the chronological per-contract 80/20 train/test split;
- the branch's self-supervised objective and loss definition;
- view augmentations and their parameters;
- projector or BYOL projector/predictor semantics;
- the 128-dimensional unnormalised downstream embedding;
- the other four branches in the five-branch representation;
- concat fusion and the existing shallow D0-style task head;
- task targets, aligned rows, feature scaling, metrics, seeds, and reporting
  budgets; and
- the rule of no validation split, early stopping, or test-driven checkpoint
  selection.

The deeper candidate must replace its corresponding canonical branch. It must
not be added as a sixth branch in the vertical study.

## Proposed first architecture family

The first matrix should use CNN encoders so depth and residual structure can be
tested without simultaneously introducing recurrence or attention. Exact
kernel, activation, initialisation, and residual-block definitions must be
frozen before execution.

| ID | Proposed backbone | Role |
|---|---|---|
| `V0` | Current two-layer CNN | Immutable shallow reference |
| `V1` | Deeper plain CNN at width 128 | Test depth without residual connections |
| `V2` | Depth-matched residual CNN at width 128 | Test residual connections at matched depth and approximately matched parameters |
| `V3` | Larger residual CNN at width 128 | Test further capacity after establishing the residual effect |

A suitable provisional construction is:

- `V1`: one input stem followed by four sequential Conv1D transformations;
- `V2`: the same stem and number of Conv1D transformations arranged as two
  identity residual blocks; and
- `V3`: the same stem followed by four residual blocks.

`V1` and `V2` should use the same kernel sizes and channel width. Identity
shortcuts should be preferred so the residual mechanism adds no trainable
projection unless shape compatibility requires one. Report exact trainable
parameter counts rather than claiming perfect matching without evidence.

If normalisation, stochastic depth, squeeze-and-excitation, dilation, or other
mechanisms are desired, they belong to separately identified variants. They
must not be silently introduced into `V2`, because that would prevent a clean
residual-versus-plain interpretation.

## Contrastive and BYOL families

Contrastive and BYOL remain selected peer branches. The two objectives must be
reported as separate families:

```text
Contrastive family:
    statistical + transformed + vae + contrastive_<Vx> + byol

BYOL family:
    statistical + transformed + vae + contrastive + byol_<Vx>
```

No primary vertical run may replace both neural branches simultaneously. The
initial executable matrix may start with one family for feasibility, but a
claim about the overall selected-branch design requires matched treatment of
both families or an explicit scope limitation.

For BYOL candidates, the online and EMA target encoders must share the same
candidate architecture, and the existing projector, predictor, EMA update,
augmentation, and loss contracts must remain intact.

## Representation-sufficiency diagnostics

Before the task-test matrix is opened, each frozen candidate should be checked
with training-only diagnostics. Use fixed lightweight probes and report the
complete diagnostic matrix rather than selecting an architecture from one
favourable signal.

Required diagnostics should include:

- reconstruction of the current close and final OHLC values;
- recent close change over fixed in-window horizons;
- next-step delta or movement using split-safe training rows;
- realised-volatility probing;
- probability-movement class probing;
- per-coordinate and aggregate embedding variance;
- effective rank and covariance spectrum;
- temporal smoothness between consecutive contract-local embeddings;
- non-finite and collapse checks; and
- linear Centered Kernel Alignment (`linear CKA`) relative to the canonical
  branch and between every capacity variant.

Linear CKA is the required primary representation-similarity metric. For two
aligned frozen representation matrices `X` and `Y`, centre each feature using
the shared training rows and compute:

```text
linear_CKA(X, Y) = ||X_c^T Y_c||_F^2
                   / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
```

Use identical ordered training-row identities, the full aligned training split
where feasible, and float64 cross-product accumulation. Report every
candidate-to-reference and candidate-to-candidate value by encoder seed, with
across-seed mean and sample standard deviation. Do not use test rows to choose
a candidate or a CKA threshold.

Any diagnostic target that uses future observations must be built independently
inside the training split and must not cross a contract or split boundary.

These diagnostics distinguish a model that merely optimises its SSL objective
from one that preserves current level, short-horizon dynamics, or volatility
information required by downstream tasks.

## Pretraining contract

The frozen executable specification should define:

- seeds `0,1,2`;
- one uninterrupted trajectory per candidate and seed;
- fixed batch size, optimiser, learning rate, weight decay, and SSL loss;
- predeclared short, medium, and long snapshots;
- cumulative optimiser steps, examples seen, and learning-rate trajectory at
  every snapshot;
- training time, peak CUDA memory, and parameter count; and
- collapse, gradient, and finite-value checks.

When batch size and data rows are identical across this matrix, matched epochs
also imply matched optimiser steps. Nevertheless, both values must be recorded
so the result can be compared responsibly with models using other batch sizes.

Only one predeclared encoder snapshot should feed the primary downstream
matrix. Training-side diagnostics may be used to verify health, but task-test
metrics must not be used to choose the encoder checkpoint.

## Downstream evaluation contract

Each candidate replaces exactly one branch in a 445-dimensional five-branch
bundle. The primary probe is the existing shallow concat head so any change is
attributed to the representation rather than decoder capacity.

Use the same three tasks:

| Task | Required target contract | Primary metrics |
|---|---|---|
| Price prediction | Saved contract-safe horizon-1 bundle; terminal contract rows removed | MAE, RMSE |
| Volatility prediction | Shared realised-volatility label bundle | MAE, RMSE, MSE, Pearson correlation |
| Probability movement | `h=2`, `tau=0.005`, common TA-eligible rows, P2 | Macro-F1, balanced accuracy, per-class recall, ROC-AUC and PR-AUC diagnostics |

Run downstream seeds `0,1,2` and retain every predeclared probe budget. Use
identical target rows and saved row identities for paired comparisons.

Branch-only probes are supporting evidence and should be run for the canonical
and candidate branches. They help determine whether an apparent full-framework
gain comes from a better standalone representation or an interaction with the
other branches.

## Artifact-first recording and on-demand comparison

Complete and verify the raw artifacts for every predeclared run before making
cross-run comparisons. Each run must independently record its configuration,
source and row hashes, seed, complete training history, every fixed-budget
checkpoint, raw predictions or class scores, aligned targets and row
identities, per-budget metrics, linear CKA diagnostics, parameter count,
training and inference time, peak memory, and feature-extraction provenance.

No dedicated comparison, aggregation, ranking, or winner-selection script is
required for this proposal. Comparisons may be produced on demand from the
immutable raw artifacts when a specific research question or report section
requires them. The resulting note or table must identify its source artifact
paths, included configurations, seeds, budgets, metric definitions, and any
uncertainty calculation so it can be checked without rerunning training.

On-demand analysis does not permit selective reporting. A comparison must
include all predeclared configurations, seeds, and budgets relevant to its
question, and it must not designate a model by choosing the lowest observed
task-test error.

## On-demand comparison questions

For each SSL family, the retained raw artifacts must support:

1. `V1 - V0`: effect of additional depth;
2. `V2 - V1`: effect of residual structure at matched depth;
3. `V2 - V0`: combined depth and residual effect;
4. `V3 - V2`: marginal effect of further residual capacity; and
5. every candidate's branch-only and five-branch results.

When one of these comparisons is requested, derive absolute metrics,
seed-level values, mean and sample standard deviation, and—when a strong claim
is made—paired row-level or contract-cluster bootstrap intervals. Include
parameter count, training time, inference time, and peak memory. A result must
be described as task-specific unless its direction is consistent across all
tasks and seeds.

## Execution stages

1. Freeze exact block definitions and the complete candidate matrix.
2. Implement candidates and unit tests without reading new task-test results.
3. Run CPU smoke tests and training-only representation diagnostics.
4. Freeze run names, hashes, seeds, snapshots, and downstream commands.
5. Run CUDA pretraining and extract the predeclared frozen embeddings.
6. Build substitution bundles and validate branch count, dimensions, split
   indices, hashes, and finite values.
7. Run the fixed shallow probes on identical task rows.
8. Verify that every run's raw metrics replay from its saved predictions and
   that the complete artifact inventory is present.
9. Produce cross-run comparisons only on demand from the immutable artifacts;
   no dedicated comparison script is required.
10. Confirm any selected design only on fresh temporally later data.

## Artifact proposal

```text
checkpoints/phase2/encoder_capacity/<family>/<variant>/
data/features/phase2/encoder_capacity/<family>/<variant>/
experiments/framework/phase2/encoder_capacity/pretraining/<family>/<variant>/
experiments/framework/phase2/encoder_capacity/<task>/<family>/<variant>/seed<seed>/
```

Every run should preserve its configuration, source hashes, architecture
manifest, checkpoints, histories, representation diagnostics, predictions,
aligned row identities, replay result, metrics, timing, and resource use.
Raw artifacts are the primary experiment output; derived comparison tables are
secondary products created when needed.

## Interpretation constraints

- Lower SSL loss is not sufficient evidence of a better representation.
- A deeper model that improves one task and hurts another is task-specialised,
  not universally superior.
- `V2` must be compared with `V1` before attributing a gain to residual
  connections.
- Parameter and compute increases must be reported alongside task metrics.
- The shallow probe remains the primary encoder comparison. More expressive
  decoders belong to a later, separately frozen interaction study.
- Results on the current test split are characterisation evidence because it
  has already informed this proposal.

## Open decisions before freezing

- exact Conv1D depth, kernel sizes, padding, activation, and residual-block
  ordering;
- whether to include normalisation in all variants or none;
- the size of `V3` and the acceptable parameter/compute envelope;
- whether the first execution covers contrastive only or both SSL families;
- the encoder snapshot used for downstream extraction;
- short/medium/long pretraining budgets and their optimiser-step counts;
- the exact training-only diagnostic probe definitions.

This proposal becomes executable only after these decisions and the complete
matrix are frozen without reference to additional task-test results.
