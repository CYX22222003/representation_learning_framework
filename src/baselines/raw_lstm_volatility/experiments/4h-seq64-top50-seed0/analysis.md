# Raw LSTM Volatility Experiment Analysis

## Run contract

- Run: `4h-seq64-top50-seed0`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU via CUDA
- Model: two-layer unidirectional LSTM with 209,537 trainable parameters
- Data: 4-hour, sequence length 64, top 50 contracts
- Aligned rows: 109,791 train and 27,450 test
- Training: one seed-0 trajectory through epoch 100, with snapshots at epochs 15, 50, and 100
- Evaluation: all snapshots evaluated only after epoch 100; replay verification succeeded
- Dataset ID: `ad6c2c56db78ad396697f1d09b8d7f7bf2598751be2dbf530def1b705d9851e8`

The processed-data digest in the run manifest matches the volatility-label manifest. Labels are constructed per contract and per split, so no target crosses a contract boundary or the global train/test boundary. These three budgets are predeclared characterization points, not test-selected candidate models.

## Pooled test results

| Epoch | Train MSE | MAE | RMSE | MSE | Pearson corr. | Prediction mean | Prediction std. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.0014646461 | 0.0515061431 | 0.1086248457 | 0.0117993578 | 0.6898544431 | 0.0467447639 | 0.0537415668 |
| 50 | 0.0007425623 | 0.0479481108 | 0.0981066823 | 0.0096249217 | 0.6769673824 | 0.0752889067 | 0.0973582417 |
| 100 | 0.0003482254 | 0.0470426343 | 0.1031050608 | 0.0106306542 | 0.6399757266 | 0.0666011721 | 0.0911311433 |

The target mean is 0.0804843232 and its standard deviation is 0.1327154338. All predictions are finite and non-negative at every budget, as expected from the Softplus output. No global or per-contract prediction series is constant, so the LSTM learned meaningful variation rather than collapsing to a fixed value.

From epoch 15 to 50, pooled MSE decreases by 18.43% and RMSE by 9.68%, while correlation decreases slightly from 0.690 to 0.677. From epoch 50 to 100, training MSE decreases by another 53.11%, but pooled test MSE increases by 10.45% and correlation decreases to 0.640. MAE continues to decrease slightly. This divergence indicates that longer training improves typical absolute errors while worsening some larger errors and reducing linear association on the held-out split. It is characterization evidence for overfitting or train/test distribution mismatch, not a basis for selecting epoch 50 after observing the test set.

## Contract-level behavior

| Epoch | Macro MAE | Macro RMSE | Macro MSE | Mean per-contract corr. | Top contract SSE share | Top 5 SSE share |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.047365 | 0.063987 | 0.009380 | 0.335373 | 29.27% | 70.13% |
| 50 | 0.044497 | 0.063050 | 0.008212 | 0.228537 | 19.91% | 61.40% |
| 100 | 0.044480 | 0.065042 | 0.009314 | 0.317152 | 15.78% | 62.72% |

Global squared-error metrics are materially influenced by a small group of contracts. At epoch 50, the five largest contributors account for 61.40% of total squared error; the largest alone accounts for 19.91%. The pooled MSE of 0.009625 is 17.20% above the unweighted macro per-contract MSE of 0.008212. Therefore, the pooled error is not attributable to a single contract, but it is concentrated enough that macro and per-contract results must accompany it.

Pooled correlation is also much higher than mean per-contract correlation at every budget. At epoch 50 the values are 0.677 pooled versus 0.229 macro. This implies that a substantial part of pooled correlation comes from separating contracts or regimes with different volatility levels; within-contract temporal association is appreciably weaker.

All 50 contracts have finite correlations, non-constant targets, non-constant predictions, and 327 to 1,154 test samples. The five largest epoch-50 squared-error contributors are contract IDs 30, 3, 35, 36, and 13. Their individual metrics remain available in [`e50/per_contract_metrics.json`](e50/per_contract_metrics.json).

## Calibration and tails

The model compresses the target distribution. At epoch 50, prediction standard deviation is 0.09736 versus target standard deviation 0.13272, and the predicted 95th percentile is 0.28965 versus 0.37184 for the target. The maximum prediction is 0.43433 while the maximum target is 0.88525. The time-ordered and scatter plots show that ordinary volatility regimes are tracked more closely than abrupt high-volatility spikes.

The epoch-15 model underpredicts the global mean most strongly. Epoch 50 is better calibrated in the mean, while epoch 100 again underpredicts both the mean and upper tail. Residual medians stay close to zero, so the larger RMSE is driven more by tail misses than by broad median bias.

## Interpretation boundary

This is a valid Raw-OHLCV sequence baseline for the shared volatility task, but it does not establish long-horizon forecasting performance. The target is the realised volatility of the next stored stride-one sequence, and adjacent length-64 windows overlap by 63 timesteps. The processed dataset also inherits full-contract volume normalization performed before the chronological split. Any future normalization correction must be applied to the framework and all volatility baselines together before comparing rerun results.

The seed-0 sweep measures training-duration behavior only. Strong model-comparison claims require a common predeclared budget across the Raw LSTM, framework, and stacking benchmark, followed by multiple seeds and paired uncertainty estimates on the shared test rows.

## Artifacts

- [Sweep metrics](sweep_metrics.json)
- [Dataset manifest](dataset_manifest.json)
- [Sweep plot](images/sweep_metrics.png)
- [Per-contract MSE plot](images/per_contract_mse.png)
- [Epoch-15 diagnostics](e15/summary.md)
- [Epoch-50 diagnostics](e50/summary.md)
- [Epoch-100 diagnostics](e100/summary.md)
