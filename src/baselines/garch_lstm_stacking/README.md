# GARCH--LSTM Stacking Volatility Benchmark

This benchmark is an adapted Peter-et-al.-inspired hybrid for Polymarket volatility prediction, not an exact reproduction of the source paper. It fuses causal guarded GARCH forecasts and Raw LSTM volatility forecasts with a fixed ElasticNet meta-learner.

## Protocol

- Scope: `4h`, top-K `50`, sequence length `64`, seed `0`.
- Labels: shared realised-volatility bundle at `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`.
- Base features: `[garch, lstm, garch * lstm]`.
- Meta-learner: `ElasticNet(alpha=1e-4, l1_ratio=0.5, fit_intercept=True, max_iter=10000)`.
- Scaling: `RobustScaler` fit only on out-of-fold training meta-features.
- Cross-fitting: five expanding contract-aware OOF folds after one burn-in block; these folds create stacking training features and are not validation or model selection.
- Final LSTM test predictions are reused exactly from `src/baselines/raw_lstm_volatility/experiments/<run-name>/`.

## GARCH Alignment

For an input close-price window `p[j : j + 64]`, the target is realised volatility over `p[j + 1 : j + 65]`. The causal GARCH feature uses the 62 already-observed target-window returns plus one one-step-ahead variance forecast:

```text
known = diff(log(p[j + 1 : j + 64]))
g_j = sqrt((sum(known ** 2) + v_next) / 63)
```

GARCH parameters, return scaling, sanity checks, and caps are fitted from the allowed training prefix only. Failed, near-static, non-finite, constraint-violating, or implausibly scaled fits fall back to a deterministic EWMA variance forecast.

## Commands

```bash
PYTHONPATH=src .venv/bin/python3 src/baselines/garch_lstm_stacking/run_experiment.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --labels-npz data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --raw-lstm-run src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0 \
  --run-name 4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --crossfit-folds 5 \
  --seed 0 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --elasticnet-alpha 1e-4 \
  --elasticnet-l1-ratio 0.5 \
  --device cuda

PYTHONPATH=src .venv/bin/python3 src/baselines/garch_lstm_stacking/run_experiment.py \
  --verify-run src/baselines/garch_lstm_stacking/experiments/4h-seq64-top50-seed0

.venv/bin/python3 src/baselines/garch_lstm_stacking/plot_experiment.py \
  src/baselines/garch_lstm_stacking/experiments/4h-seq64-top50-seed0
```

## Result Table

Canonical 4h run: `experiments/4h-seq64-top50-seed0/`. All rows are characterization evidence; no epoch is selected from test performance.

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

## Interpretation

If the stack beats Raw LSTM on matched rows and budgets, the supported claim is that the adapted hybrid stack adds predictive value under this protocol. It does not prove standalone GARCH superiority. A near-zero GARCH or interaction coefficient means the global ElasticNet found little incremental use for that scaled feature in this run.
