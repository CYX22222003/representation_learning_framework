# XM-MV8 Walk 2 — seed 0

Completed 50 epochs on 2026-10-09 using local WSL, vanilla sLSTM on CUDA,
and the dedicated `.venv-xlstm-mixer` environment. Checkpoint, prediction,
identity, and metric replay pass at all three frozen snapshots.

This model receives full five-channel OHLCV supervision at horizons t+1..t+8;
the headline extracts `close[t+8]`. It is a contextual complete-system baseline,
not a target-matched representation control. Epoch 50 was fixed before evaluation.

| Epoch | Future-close MAE | RMSE | Implied-movement Spearman | Full-path MAE |
|---:|---:|---:|---:|---:|
| 5 | 0.005422236 | 0.023514863 | 0.193657 | 0.018055930 |
| 15 | 0.005050564 | 0.022507327 | 0.237339 | 0.017130170 |
| 50 | 0.004780288 | 0.021858833 | 0.257461 | 0.016737754 |

Training/evaluation rows: 53,112/12,115; evaluation contracts: 25.
Fixed recipe: seed 0, batch 512 (104 batches, final batch 376), Adam 1e-4,
no validation, early stopping, scheduler, AMP, or test-driven checkpoint selection.
Parameters: 94,992; summed epoch training time: 120.017 seconds (excludes
preflight, serialization, evaluation, and replay); peak training CUDA allocation:
298,917,376 bytes; epoch-50 inference time: 0.212 seconds.

At epoch 50, persistence MAE/RMSE are 0.004498082/0.021350637: XM-MV8
has 6.27% worse MAE and 2.38% worse RMSE. Implied-movement Pearson is
0.016708, sign agreement 0.499629, and mean timestamp-level Rank IC 0.217973
over 1,512 eligible timestamps. Last-hour reversal Spearman is 0.258007,
effectively tied with XM-MV8's 0.257461. Contract-macro MAE/RMSE are
0.005701409/0.013992779. Positive ranking does not imply improved price error.

Unclipped full-path diagnostics are retained: invalid OHLC ordering in 54.17%
of paths, negative inverse-transformed raw volume in 6.41%, and probability
outside [0,1] in 0%. No post-hoc constraint repair or retraining was applied.

Detailed artifacts: [metrics](e50/metrics.json), [replay](replay_validation.json),
[all snapshots](sweep_metrics.json), and [completion](training_complete.json).
Matched H0-D0 and Raw-LSTM intersection reruns remain required before any
learned-comparator competitiveness claim.
