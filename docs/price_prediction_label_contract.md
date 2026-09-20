# Price-Prediction Label Contract

Date: 2026-09-20
Status: Target contract retained; execution paused pending upstream data rebuild

> The saved bundle fixes cross-contract horizon-1 targets, but it was generated
> from the legacy processed data whose preprocessing occurred before the stored
> split. It must not be described as fully leakage-free until regenerated from
> the corrected raw-time-first pipeline. Existing contract-safe Phase-2 price
> artifacts remain useful within-pipeline characterisation evidence.

## Purpose

This document records the transition from the legacy merged-array
price-target helper to the saved, contract-safe horizon-1 price-label bundle.

> **Phase 4 transition (2026-09-20):** The Phase 3 version of this contract is
> leakage-safe and remains replayable, but its absolute next-close target is
> persistence-dominated and is no longer the primary regression task. Phase 4
> will define continuous probability movement
> `close[t+h] - close[t]` under global calendar-time walks, aligned in concept
> with movement classification but evaluated as regression. Contract lifecycle
> becomes a reporting stratum inside each walk. The horizon-1
> bundle and results are retained as negative characterisation evidence.
It is the comparison authority when an older Phase-1 or early Phase-2 price
result differs from a newer result using the same representation and decoder.

The global data split is unchanged: every contract is split chronologically
into its first 80% for training and final 20% for testing. The change occurs
after that split, when horizon-1 targets and eligible feature rows are built.

## Legacy merged-array contract

The legacy framework price path called the horizon helper separately on the
already merged `processed["train"]` and `processed["test"]` arrays. For a
merged split with `N` rows, it paired feature row `i` with the close value at
row `i+1` and dropped only the final row of the entire merged array.

This was split-safe with respect to the global train/test boundary, but it was
not contract-safe. At every internal contract join, the final row of one
contract was assigned the first close value of the next contract as its
target. Those transitions have no valid horizon-1 economic interpretation.

For the current 50-contract 4h data:

| Split | Processed rows | Legacy eligible rows |
|---|---:|---:|
| Train | 109,841 | 109,840 |
| Test | 27,500 | 27,499 |

The following completed results use this legacy contract unless explicitly
rerun with a saved label bundle:

- Phase-1 framework price experiments;
- the earlier Phase-2 contrastive LSTM/Transformer encoder-refinement price
  pilots under `experiments/framework/phase2/encoder_refinement/`; and
- price baselines whose manifests have `labels_npz: null` and the same merged
  target-building path.

These artifacts remain useful characterisation evidence. They must not be
presented as strict row-identical comparisons with new contract-safe runs.

## Active contract-safe label bundle

New price experiments must use:

```text
data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz
```

The bundle constructs horizon-1 targets independently inside every contract
and inside each stored train/test split. It stores aligned labels and row
identities, including row indices, contract IDs, window starts, and timestamps.

For each contract and split, the final row has no within-contract future close
at horizon 1 and is therefore ineligible. With 50 contracts, 50 rows are
excluded from each processed split:

| Split | Processed rows | Contract-safe eligible rows |
|---|---:|---:|
| Train | 109,841 | 109,791 |
| Test | 27,500 | 27,450 |

Relative to the legacy helper, this removes 49 additional rows per split: the
legacy helper already removed the last row of the final contract, but retained
the terminal rows of the preceding 49 contracts and incorrectly crossed their
boundaries.

## Required downstream procedure

For any new framework, encoder, decoder, ablation, or baseline price run:

1. Load the saved train/test row indices and targets from the price-label
   bundle; do not rebuild targets by shifting a merged array.
2. Select feature rows using those indices before fitting preprocessing.
3. Fit feature standardisation on the selected training rows only and apply
   it unchanged to the aligned test rows.
4. Train for every predeclared fixed budget without validation, early
   stopping, or test-driven checkpoint selection.
5. Save predictions with the aligned row indices, contract IDs, window starts,
   and timestamps so metrics can be replayed.
6. Require identical label-bundle and row identities for a strict comparison.

Example invocation:

```bash
.venv/bin/python3 scripts/train_framework.py \
  --task price_prediction \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --features-npz <feature-bundle.npz> \
  --labels-npz data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz \
  --run-root <experiment-run-root> \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --device cuda
```

## Comparison and interpretation rules

- Compare new horizontal encoder configurations only with the new horizontal
  `H0`, because all of them share the contract-safe rows and scaler procedure.
- Do not compare an old Phase-1 or early Phase-2 absolute metric directly with
  a new metric and attribute the difference to the encoder.
- Restricting old predictions to the new test row indices is a useful
  diagnostic, but it is not a strict rerun: the old decoder and standardiser
  were still fitted using the legacy training rows.
- A strict cross-generation comparison requires retraining the older
  configuration with this saved bundle and otherwise identical settings.
- Current-split results remain characterisation evidence because this test
  split has already informed later experiment design.

The Phase-2 decoder study applies an additional `K=8` contract-local context
eligibility rule. Its 109,441 train and 27,100 test price rows are a further
subset of this same contract-safe horizon-1 label contract, not a competing
label definition.
