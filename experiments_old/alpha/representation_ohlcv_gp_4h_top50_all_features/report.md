# Representation + OHLCV GP dry run

Exploratory only: all saved representation coordinates and causal OHLCV terminals are GP inputs.

| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Spread |
|---|---:|---|---:|---:|---:|
| `(statistical_13 sub ((transformed_34 sub ohlcv_reversal_1) sub (transformed_6 div transformed_3)))` | 0.0541 | long_high | 0.0803 | 0.0590 | 0.025500 |
| `(contrastive_84 sub (ohlcv_reversal_1 div (transformed_31 div transformed_53)))` | -0.0332 | long_low | 0.0308 | 0.0289 | 0.010533 |
| `(contrastive_22 add ((contrastive_33 sub vae_31) div (statistical_24 sub statistical_15)))` | 0.0091 | long_high | -0.0051 | -0.0135 | 0.002645 |

The global test partition was not used. This is not an interpretable or tradeable alpha claim; it is a direct empirical check of representation coordinates.
