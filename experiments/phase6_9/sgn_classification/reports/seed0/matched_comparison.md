# SGN-C matched classification comparison

Seed 0; every method uses the exact original h2/tau=0.001 rows. Epoch 50 is the predeclared principal result.

| Walk | Method | Accuracy | Macro-F1 | Balanced accuracy | NLL | Brier |
|---:|---|---:|---:|---:|---:|---:|
| 1 | sgn_c | 0.585959 | 0.398941 | 0.442285 | 1.001004 | 0.582889 |
| 1 | h0_d0 | 0.647956 | 0.453469 | 0.465787 | 0.850085 | 0.481465 |
| 1 | raw_ohlcv_lstm | 0.542057 | 0.425204 | 0.469376 | 0.906327 | 0.519657 |
| 1 | raw_ohlcv_mlp | 0.628378 | 0.444806 | 0.460938 | 0.919250 | 0.521975 |
| 2 | sgn_c | 0.410960 | 0.313716 | 0.356374 | 1.088838 | 0.643581 |
| 2 | h0_d0 | 0.612659 | 0.453202 | 0.489162 | 0.770604 | 0.446391 |
| 2 | raw_ohlcv_lstm | 0.596313 | 0.446326 | 0.484626 | 0.809239 | 0.471694 |
| 2 | raw_ohlcv_mlp | 0.613739 | 0.432459 | 0.457089 | 0.796726 | 0.456765 |

## Interpretation

SGN-C does not beat H0-D0 on principal macro-F1 or balanced accuracy in either walk. It also trails Raw LSTM on those metrics in both walks and trails Raw MLP on macro-F1 in both walks. Walk 2 degrades substantially by epoch 50; the stronger epoch-5 result is retained but not selected post hoc.

Hard assignments collapse all five variables into one group by the principal checkpoint in both walks. This is reported as a method diagnostic and is not retuned after evaluation.

SGN-C is a supervised complete-system comparator, not target-free representation evidence. No universal, multi-seed, significance, or trading claim follows.
