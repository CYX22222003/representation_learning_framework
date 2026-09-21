# Data Processing

This package turns raw Polymarket OHLCV feather files into fixed-length sequence
tensors for training and evaluation.

It selects source files, establishes a chronological per-contract 80/20 boundary
on raw rows, fits causal imputation and volume scaling from the training prefix,
and only then builds sliding windows. Train windows remain before the boundary;
test windows are built solely from the test suffix (`isolated_test_windows`).

`scripts/prepare_sequences.py` writes a new `_split_safe.npz` by default and a
JSON provenance manifest. The NPZ includes `train` and `test` plus contract IDs,
raw window starts/ends, and window-end timestamps. The manifest records source
hashes, raw boundaries, fitted preprocessing parameters, contract ordering, and
replay hashes. Existing files are not overwritten unless `--overwrite` is given.

## Phase 5 global-calendar walks

`phase5_walks.py` is the reusable implementation for the two recent native
one-hour Phase 5 walks. It consumes the approved bounded-fill Parquet artifact,
validates raw volume/mask invariants, fits volume scaling on the walk training
interval only, and constructs separate encoder-training, mature supervised-
training, and next-interval evaluation populations. Regression and
classification share exact row identities and two-hour targets. Raw OHLCV,
scaled OHLCV, imputation metadata, availability times, lifecycle reporting
fields, and replay hashes are stored together.

Encoder row construction never creates or inspects a future target. Downstream
row construction separately applies target existence, observed-endpoint,
same-segment, and maturity rules. Unit tests perturb both target observation
status and post-cutoff gaps and require encoder identities and tensors to
remain unchanged.

Thin entry points live under `scripts_v3/`; canonical generated bundles live
under `experiments/phase5/data_preparation/`. Phase 5 implementation must not
be added to the already crowded `scripts_v2/` directory.
