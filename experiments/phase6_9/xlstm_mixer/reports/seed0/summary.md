# XM-MV8 seed-0 training diagnostics

Full OHLCV supervision at t+1..t+8; headline uses close[t+8]. Epoch 50 is predeclared.

This is not a completed fair baseline comparison. Matched H0-D0 and Raw-LSTM reruns remain required.

| Walk | Epoch | MAE | RMSE | Movement Rank IC | Full-path MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.00333226 | 0.0116688 | 0.27278 | 0.015412 |
| 1 | 15 | 0.00319481 | 0.0110782 | 0.316926 | 0.0146852 |
| 1 | 50 | 0.00305484 | 0.0108527 | 0.352519 | 0.0143795 |
| 2 | 5 | 0.00542224 | 0.0235149 | 0.193657 | 0.0180559 |
| 2 | 15 | 0.00505056 | 0.0225073 | 0.237339 | 0.0171302 |
| 2 | 50 | 0.00478029 | 0.0218588 | 0.257461 | 0.0167378 |
