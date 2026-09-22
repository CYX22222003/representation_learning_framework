# Phase 5 Downstream Add-ons

This directory contains downstream tasks added after the canonical two-hour
raw-change regression and movement classification matrix was frozen. The
canonical runs remain under `experiments/phase5/downstream/`.

## Layout

- `shared/h8/data/`: independently eligible observed/mature eight-hour rows.
- `shared/h8/features/`: canonical five-branch features aligned to those rows.
- `shared/h8/feature_scalers/`: walk-local train-only coordinate scalers shared
  by the eight-hour raw-change and absolute-price tasks.
- `tasks/raw_delta_h8/`: eight-hour signed probability-change regression.
- `tasks/log_return_h2/`: two-hour ordinary log-return regression; it reuses
  the canonical two-hour dataset, features, and scaler.
- `tasks/absolute_price_h8/`: sigmoid-bounded eight-hour future-price task.
- `manifests/`: frozen add-on execution matrices.
- `reports/`: pooled predictions, Rank IC diagnostics, plots, and summaries.

The `shared/h8/features/` store is not an alternative feature method. It uses
the same five branches and walk-specific epoch-50 encoders as the canonical
store, but follows the independently eligible eight-hour row identities. Rows
were reused only after exact identity and context matching; contexts absent
from the two-hour stores were freshly extracted.
