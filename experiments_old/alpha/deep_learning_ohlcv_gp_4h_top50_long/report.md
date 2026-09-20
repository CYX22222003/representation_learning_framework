# Representation + OHLCV GP dry run

Exploratory only: all saved deep-learning coordinates and causal OHLCV terminals are GP inputs.

| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Spread |
|---|---:|---|---:|---:|---:|
| `((ohlcv_volume_weighted_momentum_3 div (abs(contrastive_120) sub byol_63)) div contrastive_53)` | 0.1301 | long_high | 0.0942 | 0.0506 | 0.018737 |
| `(ohlcv_volume_weighted_momentum_3 div (abs(contrastive_120) sub (contrastive_82 mul byol_105)))` | -0.1268 | long_low | 0.1150 | 0.0819 | 0.026734 |
| `((ohlcv_volume_weighted_momentum_3 div (abs(contrastive_120) sub (contrastive_82 mul contrastive_46))) div contrastive_53)` | 0.1257 | long_high | 0.1240 | 0.0449 | 0.020891 |

The global test partition was not used. This is not an interpretable or tradeable alpha claim; it is a direct empirical check of representation coordinates.
