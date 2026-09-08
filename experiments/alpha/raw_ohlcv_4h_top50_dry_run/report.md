# Raw-OHLCV alpha dry run

This is a training-only factor diagnostic, not a profitability claim or cost-aware backtest.
The original per-contract 20% test partition was not used. Each original training partition was split chronologically into 60% discovery and 20% confirmation.

## Protocol

- Universe: top 50 raw Polymarket 4h contracts by existing file-size activity proxy.
- Target: next-bar close-to-close return; all factors use same-bar or earlier OHLCV only.
- Selection: top three predeclared formulae by absolute discovery RankIC; direction is frozen from discovery.
- Evaluation: timestamp-level cross-sectional IC/RankIC and equal-weight top-minus-bottom quintile return, without costs.

## Results

| Formula | Discovery RankIC | Direction | Confirmation RankIC | Confirmation IC | Top-bottom return | Dates |
|---|---:|---|---:|---:|---:|---:|
| reversal_1 | 0.2052 | long_high | 0.2449 | 0.2923 | 0.051029 | 440 |
| intraday_reversal | 0.1627 | long_high | 0.2026 | 0.1585 | 0.030656 | 440 |
| open_volume_corr_10 | 0.0206 | long_high | 0.0107 | 0.0143 | 0.001160 | 440 |
| vwap_high_corr_10 | -0.0170 | long_low | -0.0011 | -0.0018 | 0.000485 | 440 |
| volume_weighted_momentum_3 | -0.1918 | long_low | 0.2241 | 0.2524 | 0.046183 | 440 |

## Selected for a future fresh holdout

- `reversal_1`
- `volume_weighted_momentum_3`
- `intraday_reversal`

A selected result is only a candidate. Before any trading claim, evaluate these frozen formulae on a new later period and use contract-aware execution, liquidity, spread, and fee assumptions.
