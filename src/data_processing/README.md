# Data Processing

This package turns raw Polymarket OHLCV feather files into fixed-length sequence
tensors for training and evaluation.

It selects source files, cleans and aligns OHLCV rows, applies the chronological
per-contract 80/20 train/test split, builds sliding windows, and saves processed
`.npz` files with `train` and `test` arrays.
