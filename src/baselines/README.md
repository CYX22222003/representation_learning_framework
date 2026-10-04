# Baselines

This package contains internal and external comparison models used to evaluate
the representation-learning framework.

The baselines are trained on the same processed train/test splits as the
framework. Current baselines include a Raw-OHLCV MLP, a stacked LSTM price
benchmark, a Raw LSTM volatility benchmark, an adapted GARCH--LSTM stacking
volatility benchmark, a GINN volatility benchmark, and a TA-MLP
trend-classification benchmark. Phase 6.7 additionally contains independent
paper-guided adapters for SaURL-TS and LWA. The LWA package currently covers
only the Stage 1 architecture, deterministic transforms, objectives, mapper
stage, and efficient frozen extractor; its training orchestration is a later
review-gated stage.
