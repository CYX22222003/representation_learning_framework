# Phase 6.5D Residual-CNN Future-Price Report

Epoch 50 is the predeclared principal snapshot. The report is written only after replaying all four residual encoders, both feature stores, eight candidate price trajectories, same-family duplicate-width controls, H0, the Raw MLP/LSTM context baselines, and both CKA diagnostics.

## Principal result

Across the fixed evaluation populations (43,340 rows), `HC-SR` reduces row-weighted MAE by 7.17% and pooled RMSE by 14.99% relative to H0. The error improvement occurs in both walks.

`HC-AR` reduces row-weighted MAE by 8.45% and pooled RMSE by 14.67% relative to H0. Against the same-width `HC-DC` control, the reductions are 10.82% MAE and 17.93% RMSE, so the price-error gain is not explained by width alone.

The Contrastive result is bounded: `HC-AR` still trails Raw LSTM (MAE 0.007901 versus 0.005137) and current-price persistence (MAE 0.003429, pooled RMSE 0.015055). Its walk-weighted cross-sectional Rank IC (0.1235) is below H0 (0.1286) and last-hour reversal (0.2822). The BYOL substitutions/additions do not improve price error consistently across walks.

`epoch50_weighted_walk_summary.csv` weights MAE/MSE by evaluation-row count and derives RMSE from the weighted MSE. Its correlation and Rank-IC columns are explicitly walk-weighted descriptive summaries, not correlations recomputed after merging the two walk populations.

## Claim boundary

Same-width substitutions are compared with H0. Heterogeneous additions are compared with both H0 and the matching duplicate-CNN width control. Persistence and last-hour reversal remain non-learned references. Results are seed-0, two-walk, future-price characterisation only; they do not establish profitable trading or universal representation superiority.
