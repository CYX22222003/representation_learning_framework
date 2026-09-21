# Features

This package contains deterministic feature extractors and feature-store
utilities.

`statistical.py` builds AR and GARCH features per OHLCV column. `transform.py`
builds FFT and Haar wavelet features. `feature_store.py` stores deterministic
features together with named frozen neural branches such as `vae` or
`contrastive`.

`phase5_features.py` extracts and replay-validates the canonical five-branch
Walk 1/Walk 2 stores. It preserves the source supervised identities, verifies
byte-identical encoder-context membership, hashes each branch, and uses only
the separately trained walk-local epoch-50 encoders.

For the eight-hour sensitivity, the extractor may reuse a primary feature row
only after exact identity and context equality and identical checkpoint hashes
are verified. Rows absent from the primary store are freshly extracted and the
reuse counts and source hashes are recorded in the sensitivity manifest.
