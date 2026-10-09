# Endpoint-only xLSTM-Mixer comparison

Seed 0, fixed epoch 50; original h8 price rows and endpoint-only MSE. Both trajectories and all six snapshots independently replay-valid.

| Walk | Method | Rows | MAE | RMSE | Movement Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | XM-C8 | 29834 | 0.003100233 | 0.011040398 | 0.328457 |
| 1 | h0_d0 | 29834 | 0.007985399 | 0.018905834 | 0.147363 |
| 1 | raw_lstm | 29834 | 0.004500579 | 0.012151731 | 0.263878 |
| 2 | XM-C8 | 13506 | 0.004759289 | 0.020759157 | 0.259229 |
| 2 | h0_d0 | 13506 | 0.010053535 | 0.030072603 | 0.099988 |
| 2 | raw_lstm | 13506 | 0.006541574 | 0.022129913 | 0.146491 |

XM-C8 improves MAE/RMSE over H0-D0 and Raw LSTM in both walks. Persistence remains stronger on MAE in both walks; XM-C8 improves persistence RMSE only in Walk 1. These are endpoint-matched complete-system results, not isolated representation or architecture gains.

| Walk | Persistence MAE | Persistence RMSE | Timestamp Rank IC | Outside [0,1] | Training seconds | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002978262 | 0.011871641 | 0.306897 | 0.000000 | 97.60 | 284.12 |
| 2 | 0.004426176 | 0.020395383 | 0.230508 | 0.000000 | 128.95 | 284.22 |

The inverse-RevIN output is unclipped, unlike the sigmoid controls; XM-C8 also retains clip norm 1.0. This is an endpoint adaptation, not the paper's full-path reproduction. The protocol correction followed review of XM-MV8 results; fresh holdouts or separately approved replications are required for strong confirmatory claims. No checkpoint was selected by evaluation. Historical XM-MV8 artifacts are unchanged. No universal, multi-seed, significance, or trading claim follows.

The original launcher completed training and replay but its final writer encountered `KeyError: spearman` because H0 stores subgroup metrics. This standalone report reads the correct existing field and independently replays all artifacts without changing fingerprinted training code, checkpoints, predictions, or metrics.
