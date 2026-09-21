# Source Package

This directory contains the implementation of the representation-learning
framework, its downstream task heads, evaluation utilities, and comparison
baselines.

The main framework flow is:

```text
data_processing -> features/models -> aggregation -> tasks -> evaluation
```

External raw-source ingestion is isolated in `data_acquisition/`. It writes
provenance-bearing raw data under ignored storage roots and does not perform
experiment splitting, fitted preprocessing, label construction, or training.
For FinData Polymarket data, condition-level candles are noncanonical because
they can mix outcome-token orientations. A versioned
forward-confirmed quarantine screen may additionally create exploratory clean
artifacts: raw rows stay immutable, persistent crashes/repricings are retained,
suspected transient rows are not price-corrected, their timestamps remain gaps,
and a 1% batch-level removal budget fails closed. Each forward-looking decision
records when its confirmation became available. Phase 5 treats the approved
final pruning as offline retrospective cleaning and does not replay those
availability times. The
token-consistent path aggregates trades after mapping the declared `Yes`
outcome to its token ID; only that path provides direct outcome provenance.
Phase 5 nevertheless selects the clean native one-hour condition candles for
an explicitly exploratory walk-forward loop, with isolated-one-bar causal
filling, observed decision/target endpoints, longer-gap sequence breaks, and
affected-contract/source-limitation sensitivities.

The Phase 5 walk builder is implemented in `data_processing/phase5_walks.py`.
Its thin executables are under `scripts_v3/`, and its replayable data artifacts
are under `experiments/phase5/data_preparation/`.

Baseline implementations live under `baselines/` and are kept close to the
framework code so they can share the same processed data, target builders, and
metrics.
