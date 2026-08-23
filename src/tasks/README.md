# Tasks

This package contains downstream target builders and lightweight task heads.

The supported tasks are price prediction, realised-volatility prediction, and
trend classification. Each module defines how labels are built from processed
sequences and provides a simple MLP head used by the framework and internal
baselines.
