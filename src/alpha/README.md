# Alpha research skeleton

This package is an implementation starting point for the deferred alpha-factor
capability. It consumes aligned, chronological out-of-fold predictions from
the price, volatility, and trend heads. It does not mine arbitrary latent
coordinates and it does not select formulas on the current Phase-1 test set.

The intended future flow is: fit heads on expanding train folds → collect OOF
predictions with `oof.generate_oof_predictions` → build terminals with
`build_alpha_primitives` → enumerate a shallow protected grammar → score IC,
rank-IC, temporal stability, and redundancy on OOF rows → freeze a small set →
evaluate once on a fresh holdout or later data. A cost-aware contract-level
backtest is required before any profitability claim.
