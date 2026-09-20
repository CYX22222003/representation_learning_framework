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
