# Raw LSTM Volatility Benchmark

This benchmark trains a two-layer unidirectional LSTM directly on processed raw OHLCV sequences for realised-volatility prediction. It is an external-style end-to-end comparator for the representation framework, not a tuned model-selection workflow.

## Protocol

- Input: `data/processed/market_4h_seq64_top50.npz`, shape `[N, 64, 5]`.
- Labels: shared contract-aware bundle at `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`.
- Target: realised volatility of the next stored stride-one close-price sequence.
- Split rule: train/test only, using the existing per-contract chronological 80/20 split.
- Budgets: one seed-0 training trajectory through epoch 100, with snapshots at 15, 50, and 100. Test evaluation happens only after training finishes.
- Limitation: the next-window target has stride-one overlap; for `seq_len=64`, adjacent input and target windows overlap by 63 timesteps. It should not be interpreted as a non-overlapping long-horizon forecast.
- Inherited data limitation: the current processed data z-scores volume per full contract before splitting.

## Commands

```bash
.venv/bin/python3 scripts/prepare_volatility_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --timeframe 4h \
  --seq-len 64 \
  --top-k 50 \
  --overwrite

PYTHONPATH=src .venv/bin/python3 src/baselines/raw_lstm_volatility/run_experiment.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --labels-npz data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --run-name 4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --device cuda

PYTHONPATH=src .venv/bin/python3 src/baselines/raw_lstm_volatility/run_experiment.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --labels-npz data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz \
  --verify-run src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0 \
  --device cuda

.venv/bin/python3 src/baselines/raw_lstm_volatility/plot_experiment.py \
  src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0
```

## Results

Shared label bundle generated for the canonical 4h top-50 data:

- train labels: `109791`
- test labels: `27450`
- train contracts: `50`
- test contracts: `50`
- overlap note: `next stored stride-one sequence; 63 of 64 timesteps overlap when seq_len=64`

| Epoch | MAE | RMSE | MSE | Pearson corr. | Negative fraction | Test rows |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | TBD | TBD | TBD | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD | TBD | TBD | TBD |

Report all rows as characterization evidence. Do not pick the lowest test error as the canonical checkpoint.
