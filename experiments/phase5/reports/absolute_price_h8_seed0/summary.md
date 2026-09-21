# Phase 5 Eight-Hour Absolute-Price Probe (Seed 0)

| Walk | Price MAE | Price RMSE | Price Pearson | Price Spearman | Persistence MAE | Implied-delta Pearson | Implied-delta Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00798540 | 0.01890583 | 0.996198 | 0.951078 | 0.00297826 | 0.209421 | 0.147363 |
| 2 | 0.01005353 | 0.03007260 | 0.987541 | 0.982321 | 0.00442618 | -0.003541 | 0.099988 |

## Walk-level movement ranking

- Walk 1: implied-movement mean cross-sectional Rank IC `0.143956`; last-hour reversal reference `0.301238`; non-zero sign agreement `0.558459`.
- Walk 2: implied-movement mean cross-sectional Rank IC `0.094730`; last-hour reversal reference `0.240011`; non-zero sign agreement `0.554901`.

## Pooled result

- Price level: MAE `0.00862989`, RMSE `0.02297542`, Pearson `0.993783`, Spearman `0.964441`.
- Persistence: MAE `0.00342947`, RMSE `0.01505473`.
- Persistence-relative MSE skill: `-1.329061`.
- Implied movement: Pearson `0.102503`, Spearman `0.128726`, sign agreement `0.421527`.
- Cross-sectional implied-movement Rank IC: mean `0.126384`, median `0.136364`, positive fraction `0.6583` across `2821` timestamps.
- Last-hour reversal reference cross-sectional Rank IC: mean `0.279444`, median `0.298969`, positive fraction `0.8330`.

Direct price training recovers a consistent implied-movement ranking signal, but the simple last-hour reversal reference is materially stronger. High price-level correlation remains state reconstruction rather than evidence of superior forecasting.
