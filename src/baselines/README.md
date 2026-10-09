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

Phase 6.9 now plans an independently authored SGN-C classification adaptation
using the paper and official source/settings as references. It replaces the
unimplemented Monotone-VI classification leg, not optional Phase 6.8's frozen
representation comparison. SGN source/settings/licence audit and a separate
implementation specification must be approved before model coding; no SGN
package or training exists yet. See
[`SGN classification amendment`](../../docs/phase_plan/2026-10-09-phase-6-9-sgn-classification-amendment.md).
