# Phase 5 Matched Raw-Sequence Baselines (Seed 0)

Epoch 50 is the predeclared principal checkpoint. Both walks were fitted independently and pooled only after out-of-future predictions were generated.

## Two-hour movement regression

| Model | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Five-branch framework | 0.00271136 | 0.01083505 | 0.080745 | -0.000518 |
| raw_ohlcv_mlp | 0.00240460 | 0.01072570 | 0.277389 | 0.004498 |
| raw_ohlcv_lstm | 0.00240600 | 0.01078520 | 0.009041 | 0.003955 |

## Two-hour movement classification

| Model | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Five-branch framework | 0.454116 | 0.472953 | 0.636873 |
| raw_ohlcv_mlp | 0.441711 | 0.459794 | 0.623782 |
| raw_ohlcv_lstm | 0.430663 | 0.474314 | 0.559093 |

## Eight-hour absolute future price

| Model | Price MAE | Price Pearson | Implied-movement Spearman | Mean cross-sectional Rank IC |
|---|---:|---:|---:|---:|
| Five-branch framework | 0.00862989 | 0.993783 | 0.128726 | 0.126384 |
| raw_ohlcv_mlp | 0.00819293 | 0.996199 | 0.095304 | 0.119290 |
| raw_ohlcv_lstm | 0.00513661 | 0.996974 | 0.220031 | 0.208024 |

The fixed causal last-hour reversal reference has pooled mean cross-sectional Rank IC `0.279444` on the same h8 rows.

These comparisons use identical task rows and source preprocessing. No best-on-evaluation checkpoint selection is performed.
