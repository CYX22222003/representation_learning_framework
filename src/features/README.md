# Features

This package contains deterministic feature extractors and feature-store
utilities.

`statistical.py` builds AR and GARCH features per OHLCV column. `transform.py`
builds FFT and Haar wavelet features. `feature_store.py` stores deterministic
features together with named frozen neural branches such as `vae` or
`contrastive`.
