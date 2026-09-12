# Framework price prediction sweep

All epoch budgets are reported as a characterization sweep; no checkpoint is selected from test performance.

| epoch | mae | rmse | mse | corr | train_loss |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.0574953221 | 0.0954704508 | 0.0091146072 | 0.9780145288 | 0.0010047419 |
| 50 | 0.0694802403 | 0.1045774966 | 0.0109364530 | 0.9763363600 | 0.0005326392 |
| 100 | 0.0720483065 | 0.1058754772 | 0.0112096164 | 0.9760525823 | 0.0004076131 |

See `comparison.md` for baseline context. The current LSTM numbers are genuine
held-out test results, but they are treated as external benchmark
characterization until the LSTM runner is refit to load the processed `.npz`
split and shared price-target builder.
