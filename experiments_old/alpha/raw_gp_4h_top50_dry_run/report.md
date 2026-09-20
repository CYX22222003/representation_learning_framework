# Raw-OHLCV GP dry run

Train-only constrained GP search; not a backtest.

| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Top-bottom return |
|---|---:|---|---:|---:|---:|
| `(intraday_reversal sub (volume_weighted_momentum_3 div open_volume_corr_10))` | 0.1395 | long_high | 0.1698 | 0.0952 | 0.023927 |
| `((open_volume_corr_10 add open_volume_corr_10) add (intraday_reversal add volume_weighted_momentum_3))` | 0.1173 | long_high | 0.1320 | 0.0991 | 0.017849 |
| `(vwap_high_corr_10 mul intraday_reversal)` | -0.1133 | long_low | 0.1454 | 0.1211 | 0.026505 |

Global test rows were sliced away before terminal construction. Formula selection used discovery only; confirmation is chronological and fixed-formula. Costs, spreads, liquidity and a fresh final holdout remain required.
