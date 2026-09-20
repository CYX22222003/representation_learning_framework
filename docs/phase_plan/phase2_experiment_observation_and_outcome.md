# Phase 2 Experiment Observations and Outcomes

Date: 2026-09-20
Status: Phase 2 paused — upstream data correction required
Tracking issues: [#14 — Correct price baseline setup and redesign volatility target](https://github.com/CYX22222003/representation_learning_framework/issues/14); [#15 — Rebuild split-safe data pipeline before resuming Phase 2](https://github.com/CYX22222003/representation_learning_framework/issues/15)

## Purpose

This document records findings discovered while executing and auditing Phase 2.
Phase 2 is paused before additional training because the shared processed-data
pipeline allowed the later test period to influence preprocessing before the
stored train/test split. Encoder-refinement results and the final Phase 2
judgement remain incomplete.

The findings below concern experiment validity rather than model quality. Keep
all affected artifacts for provenance, but label them as historical
characterisation evidence and do not use them for strict performance claims.

## Observation 0: the shared preprocessing boundary is invalid

The most serious finding occurs before task labels or downstream train/test
loaders. The legacy pipeline preprocesses each complete contract, fits volume
normalisation on that complete contract, generates every stride-one window, and
only then splits the windows 80/20.

The audit confirmed that, for all 50 active 4-hour contracts:

- full-contract volume statistics differ from training-prefix statistics, so
  the stored training inputs are influenced by the later test period; and
- the last training and first test windows share 63 of 64 raw observations.

The fitted-normalisation issue is direct leakage. Window overlap is not by
itself always invalid in rolling forecasting, because a test forecast can use
past context, but the project did not define such a context policy and the
current split is a split over already-generated windows rather than the claimed
raw-time holdout. The combined implementation therefore does not satisfy the
documented locked-test contract.

This affects every Phase-2 component derived from
`data/processed/market_4h_seq64_top50.npz`, including encoder pretraining,
frozen feature extraction, decoder studies, horizontal encoder probes, and the
latest probability-movement classification experiment. Correct downstream
row alignment cannot repair information introduced upstream.

### Outcome and required action

- Pause all further Phase-2 training and evaluation on the legacy processed and
  feature bundles.
- Preserve every existing checkpoint, feature bundle, prediction, metric, and
  report without overwriting it.
- Implement and validate the raw-time-first contract in
  [`docs/data_processing_split_contract.md`](../data_processing_split_contract.md).
- Rebuild processed data, retrain neural encoders, regenerate frozen features
  and task bundles, and then rerun the required Phase-2 comparisons.

## Observation 1: price-prediction baselines require rerunning

The existing price-prediction baseline results were produced under erroneous or
incompatible sample-construction contracts and therefore need to be rerun.

Two related problems were identified:

1. The legacy merged-array horizon helper shifted the concatenated contract
   arrays. At internal contract joins, the terminal row of one contract was
   paired with the first close from the next contract. This affected legacy
   price runs whose manifests have `labels_npz: null`, including older
   Raw-OHLCV MLP/framework comparisons.
2. The existing price LSTM baseline constructed labelled sliding windows before
   applying its train/test split. Its adjacent stride-one train and test samples
   consequently overlap at the split boundary and do not match the active
   framework price rows.

The active contract-safe price bundle is:

```text
data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz
```

It constructs horizon-1 targets independently inside each contract and stored
split and removes the terminal row of every contract. On the current dataset it
contains 109,791 training rows and 27,450 test rows. Feature selection must use
these row indices before fitting train-only preprocessing.

### Outcome and required action

- Treat existing price LSTM baseline checkpoints and results as legacy
  characterisation evidence.
- Treat any price baseline or framework result using the merged-array target
  path as legacy characterisation evidence.
- Correct the price baseline data pipeline so splitting and target construction
  obey the upstream per-contract contract and the saved label bundle.
- Rerun each baseline required for a final price comparison on identical saved
  label rows, preprocessing rules, budgets, seeds, and metrics.
- Do not attribute differences between a legacy result and a contract-safe
  Phase-2 result to an encoder or decoder change.

The detailed transition rules are recorded in
[`docs/price_prediction_label_contract.md`](../price_prediction_label_contract.md).

## Observation 2: the volatility target must be redesigned

The current volatility bundle uses an MVP next-window target. For an input
close-price window

```text
[p_j, ..., p_{j+63}]
```

the target is realised volatility over

```text
[p_{j+1}, ..., p_{j+64}].
```

The input and target therefore share 63 of 64 prices, and 62 of the 63 target
returns are already observable from the input. Only the final return is new.
This is better described as a one-step rolling-volatility update or nowcast than
as prediction of a genuinely unseen future volatility window.

The target was retained during Phase 1 and early Phase 2 to keep the framework,
Raw LSTM, and adapted GARCH--LSTM comparisons row-identical. That engineering
parity does not establish that the target is an adequate forecasting task.

### Outcome and required action

- Redesign and predeclare the volatility prediction target before further
  confirmatory volatility experiments.
- The revised target must use a genuinely future, non-overlapping or otherwise
  explicitly justified forecast interval, constructed independently within
  every contract and split.
- Record the forecast origin, horizon, aggregation formula, annualisation or
  scaling convention, eligible rows, and dropped boundary rows in a new saved
  label-bundle manifest.
- Add simple persistence/current-window references so a downstream model is not
  credited for information already present in the input.
- Regenerate aligned volatility labels and rerun the framework, Raw LSTM,
  GARCH--LSTM, and any Raw-OHLCV MLP comparator required for strict claims.
- Treat results using `rv_4h_seq64_top50.npz` as evidence about the historical
  MVP next-window proxy only, not general future-volatility forecasting.

The exact replacement target remains an open design decision. It must be fixed
before new results are inspected rather than selected using test performance.

## Current conclusion and Phase 2 decision

The upstream defect changes the earlier judgement. Existing Phase-2 encoder
pretraining, frozen features, linear CKA, decoder runs, and downstream probes
are reproducible evidence about the legacy pipeline only. Matched rows still
make within-pipeline diagnostics informative, but they do not make the results
leakage-free.

- Recent horizontal price probes fixed the separate cross-contract target bug,
  but still inherit the upstream preprocessing defect.
- Current volatility results additionally characterize an overlapping MVP
  proxy rather than a genuine future-volatility forecast.
- Phase-2 classification has a sound split-local label and alignment design,
  but its raw and frozen inputs still inherit the upstream defect.
- No current Phase-2 result should support a confirmatory model-superiority
  conclusion.

## Why the defect was not detected earlier

The earlier checks started from the already-produced `.npz` files. They
verified shapes, stored train/test counts, contract-local task horizons, row
identity hashes, train-only downstream scalers, and prediction replay. Those
checks correctly found later target/alignment errors but did not reconstruct
which raw observations or fitted statistics produced each stored window.

Several factors allowed the problem to survive:

1. documentation and code comments described the per-contract window split as
   a chronological train/test split, masking the fact that preprocessing and
   window generation occurred first;
2. processed bundles did not store raw boundary indices, window coverage, or
   preprocessing-fit provenance;
3. unit tests concentrated on task-label and downstream alignment rather than
   an end-to-end raw-to-window leakage invariant; and
4. all framework variants reused the same processed bundle, so internal row
   consistency looked like experimental validity.

The earlier audit should have traced the pipeline back to raw feather rows when
the LSTM boundary discrepancy was first found. Restricting that audit to label
construction and downstream row matching was too narrow.

## Pending additions

After the data pipeline is corrected and the revised experiments finish, update
this document with:

1. the completed horizontal and vertical encoder matrices;
2. raw downstream results for every declared configuration;
3. linear CKA results and their interpretation;
4. matched comparisons under valid task contracts;
5. compute/resource observations; and
6. the final Phase 2 outcome, limitations, and recommended next experiment.
