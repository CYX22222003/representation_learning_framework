# Source Package

This directory contains the implementation of the representation-learning
framework, its downstream task heads, evaluation utilities, and comparison
baselines.

The main framework flow is:

```text
data_processing -> features/models -> aggregation -> tasks -> evaluation
```

Baseline implementations live under `baselines/` and are kept close to the
framework code so they can share the same processed data, target builders, and
metrics.
