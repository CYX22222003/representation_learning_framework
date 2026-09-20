# Phase 2 Selected-Branch Encoder Amendment

Date: 2026-09-14  
Status: Frozen design; execution paused pending upstream data correction

> Existing pilot artifacts are preserved, but this amendment must not execute
> against the legacy processed or feature bundles. Rebuild the encoder inputs
> under [`../data_processing_split_contract.md`](../data_processing_split_contract.md)
> before resuming.

## Decision

BYOL is a selected peer feature branch in the canonical five-branch
representation, not an optional add-on. Phase 2 therefore refines the
backbone of both selected self-supervised branches: contrastive and BYOL.

The existing seed-0 contrastive LSTM/Transformer results remain a recorded
pilot under the earlier contrastive-only plan. They must not be relabelled as
results from this amended matrix or used to choose a BYOL architecture.

## Fixed primary configurations

Every primary bundle remains 445-dimensional and changes exactly one branch:

| ID | Five-branch substitution |
|---|---|
| `R0` | statistical + transformed + vae + contrastive + byol |
| `C1` | statistical + transformed + vae + contrastive_lstm + byol |
| `C2` | statistical + transformed + vae + contrastive_transformer + byol |
| `B1` | statistical + transformed + vae + contrastive + byol_lstm |
| `B2` | statistical + transformed + vae + contrastive + byol_transformer |

`R0` is the shared immutable CNN reference. `C1`/`C2` answer the contrastive
backbone question; `B1`/`B2` answer the BYOL backbone question. No primary run
may contain both a temporal contrastive branch and a temporal BYOL branch.

## BYOL implementation contract

`byol_lstm` and `byol_transformer` must retain the canonical BYOL learning
mechanism: online and EMA target encoders, projector and predictor roles,
the existing two-view augmentation policy, loss semantics, optimizer recipe,
and a 128-dimensional unnormalised online-backbone embedding for downstream
use. The LSTM and Transformer backbone settings match the frozen temporal
design unless an incompatibility is documented before pretraining. They train
on the locked train split only, use seeds `0,1,2`, one uninterrupted 100-epoch
trajectory, and record snapshots at epochs `15,50,100`.

## Evaluation contract

For each candidate and seed, extract only the epoch-100 frozen branch and
substitute it into its matching five-branch bundle. Run the unchanged shallow
probe on price prediction, shared-label volatility, and P2 probability-movement
classification at downstream seeds `0,1,2` and budgets `15,50,100`.

Branch-only diagnostics are required for `contrastive`, `contrastive_lstm`,
`contrastive_transformer`, `byol`, `byol_lstm`, and `byol_transformer`. Each
comparison must use identical task targets and row identities, record resource
costs, replay saved metrics, and report seed-level values, mean, sample
standard deviation, and paired uncertainty. The immutable `R0` reference is
probed at all downstream seeds but is not retrained.

The execution inventory is 12 candidate pretraining trajectories (four
candidates × three seeds), 45 primary downstream trajectories (five
configurations × three tasks × three seeds), and 54 branch-only trajectories
(six branches × three tasks × three seeds). Existing pilot artifacts are not
counted toward this matrix unless regenerated or independently validated to
meet this amendment's manifest and replay requirements.

## Interpretation

Report the contrastive and BYOL families as separate comparisons against `R0`.
Do not call a temporal backbone universally superior because it helps one task,
and do not infer that a weak historical BYOL contribution removes BYOL from the
chosen representation. Any later bundle containing two winning temporal
branches is a separately named, post-selection exploratory combination and
requires a fresh later temporal holdout for a confirmatory claim.

## Completion gate

The encoder-refinement part is complete only after all four candidate families
have passing health/provenance checks, frozen feature and substitution bundles,
the complete primary and branch-only multi-seed matrices, metric replay,
resource reporting, and separate family-level aggregate reports. Results on
the already inspected task-test split remain characterisation evidence.
