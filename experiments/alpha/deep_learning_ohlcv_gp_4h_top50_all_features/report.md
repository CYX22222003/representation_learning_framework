# Representation + OHLCV GP dry run

Exploratory only: all saved deep-learning coordinates and causal OHLCV terminals are GP inputs.

| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Spread |
|---|---:|---|---:|---:|---:|
| `(((vae_62 sub byol_103) add contrastive_15) sub ohlcv_intraday_reversal)` | -0.1388 | long_low | 0.1660 | 0.1345 | 0.028760 |
| `(((vae_62 sub vae_55) add contrastive_15) sub ohlcv_intraday_reversal)` | -0.0999 | long_low | 0.1153 | 0.1117 | 0.029565 |
| `(ohlcv_volume_weighted_momentum_3 div byol_111)` | 0.0564 | long_high | 0.0544 | -0.0401 | -0.013994 |

The global test partition was not used. This is not an interpretable or tradeable alpha claim; it is a direct empirical check of representation coordinates.
