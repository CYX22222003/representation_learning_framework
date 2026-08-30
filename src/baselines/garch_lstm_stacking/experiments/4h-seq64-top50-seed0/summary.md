# GARCH--LSTM Stacking Volatility Benchmark

This is a Peter-et-al.-inspired adapted stack, not an exact paper reproduction. All epoch budgets are reported; no test metric is used for model selection.

| LSTM epoch | Model | MAE | RMSE | MSE | Pearson corr. | Macro-contract MSE | Raw negative fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 15 | Raw LSTM | 0.0515061431 | 0.1086248457 | 0.0117993578 | 0.6898544431 | 0.0093797307 | 0.0000000000 |
| 15 | GARCH--LSTM stack | 0.0329557471 | 0.0890257880 | 0.0079255905 | 0.8058654070 | 0.0061535035 | 0.2413114754 |
| 50 | Raw LSTM | 0.0479481108 | 0.0981066823 | 0.0096249217 | 0.6769673824 | 0.0082120049 | 0.0000000000 |
| 50 | GARCH--LSTM stack | 0.0326731317 | 0.0865056813 | 0.0074832332 | 0.8100172877 | 0.0056865145 | 0.1727504554 |
| 100 | Raw LSTM | 0.0470426343 | 0.1031050608 | 0.0106306542 | 0.6399757266 | 0.0093139153 | 0.0000000000 |
| 100 | GARCH--LSTM stack | 0.0304159932 | 0.0833720863 | 0.0069509046 | 0.8218969107 | 0.0053103617 | 0.0753369763 |

| LSTM epoch | Intercept | GARCH coefficient | LSTM coefficient | Interaction coefficient | GARCH fallback rate | GARCH cap rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.0121885186 | 0.0445922524 | 0.0084989310 | -0.0028959680 | 0.3926047359 | 0.0000000000 |
| 50 | 0.0118272566 | 0.0421697971 | 0.0085493484 | -0.0018855078 | 0.3926047359 | 0.0000000000 |
| 100 | 0.0113244683 | 0.0391764025 | 0.0121341867 | -0.0016709065 | 0.3926047359 | 0.0000000000 |
