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
records when its confirmation became available so earlier walk cutoffs cannot
use it. The
token-consistent path aggregates trades after mapping the declared `Yes`
outcome to its token ID; only that path provides direct outcome provenance.
Phase 5 nevertheless selects the clean native one-hour condition candles for
an explicitly exploratory walk-forward loop, with isolated-one-bar causal
filling, observed decision/target endpoints, longer-gap sequence breaks, and
affected-contract/source-limitation sensitivities.

Baseline implementations live under `baselines/` and are kept close to the
framework code so they can share the same processed data, target builders, and
metrics.
