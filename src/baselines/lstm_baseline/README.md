# LSTM Baseline

External benchmark for the price prediction task. A 3-layer stacked LSTM that takes 64 prior close prices and predicts the next close. Univariate by design — used as a comparison point for the multi-branch framework.

## Splitting convention

This baseline uses **train / test only** — no validation split, no early stopping. This matches the framework's standard pipeline (`build_from_file_list` in `src/data_processing/data_processing.py`), which also produces train + test only. Training runs for a fixed number of epochs supplied via `--epochs`.

## Multi-run sweeps

We run the model at several `--epochs` budgets to **characterize how the baseline behaves under different training durations** — e.g. how train loss decays, when test performance flattens or starts to drift, how stable the model is at long horizons. The purpose is comparison and description, **not selection of a "best" run**. Reporting in this baseline therefore covers the whole sweep (every run's metrics), not a hand-picked winner; picking the epoch count by inspecting test metrics would turn the test set into a tuning set.

Seeds are fixed (`--seed`) across the sweep so that differences between runs reflect the epoch budget rather than initialization noise.

### Canonical sweep

Current epoch budgets used for the LSTM baseline characterization sweep:

| `--epochs` | Purpose |
|---|---|
| 15 | Around where prior early-stopped runs typically halted |
| 20 | Near the train-loss plateau seen in prior runs |
| 25 | Slightly past plateau — does extra training still move things? |
| 50 | Mid-horizon — well past plateau |
| 100 | Long-horizon — does test error drift up under overtraining? |

All five runs share the same seed (`--seed 0`) so observed differences reflect the training budget alone.

## Files

| Path | Purpose |
|---|---|
| `lstm_model.py` | Model (`LSTMModel`), loss (`RMSELoss`), data pipeline (`build_lstm_dataset`, `PriceDataset`), training (`train_model`), evaluation (`evaluate`), and CLI runner |
| `plot_experiment.py` | Generates `training_curve.png` and `pred_vs_actual.png` for a given run directory |
| `experiments/` | One subdirectory per run; each is self-contained |

Each `experiments/<run-name>/` directory contains:

```
log.log            ← stdout of the training run (captured via shell redirection)
summary.md         ← hand-written results summary
history.npz        ← per-epoch train_loss, val_loss, best_epoch
predictions.npz    ← preds, targets on the test set
checkpoint.pth     ← saved model state_dict
images/
  training_curve.png
  pred_vs_actual.png
```

## Running a new experiment

**The commands below use paths relative to the project root** (the directory that contains `.venv/`, `src/`, `data/`, `checkpoints/`). Run them from there — do not `cd` into `src/baselines/lstm_baseline/` first, or the relative paths to `.venv/bin/python3` and `PYTHONPATH=src` will resolve to the wrong place and the run will fail silently with the log only containing a `No such file or directory` line.

A defensive guard at the top of the snippet catches that mistake:

```bash
# Always run from the project root
test -d .venv && test -d src/baselines/lstm_baseline || {
    echo "ERROR: run this from the project root (the directory that contains .venv/ and src/)" >&2
    return 1 2>/dev/null || exit 1
}

RUN=2026-06-22-v4-rmse
RUN_DIR="src/baselines/lstm_baseline/experiments/$RUN"
mkdir -p "$RUN_DIR"

PYTHONPATH=src .venv/bin/python3 -u src/baselines/lstm_baseline/lstm_model.py \
    --run-name "$RUN" \
    --epochs 30 \
    --seed 0 \
    2>&1 | tee "$RUN_DIR/log.log"

.venv/bin/python3 src/baselines/lstm_baseline/plot_experiment.py "$RUN_DIR"
```

`--epochs` controls the fixed training budget. `--seed` controls weight init, DataLoader shuffling, numpy / random / CUDA — fixing it makes runs reproducible and makes an epoch sweep actually comparable (otherwise epoch effects and seed effects are confounded).

Then write `$RUN_DIR/summary.md` describing the setup and results.

If you prefer not to rely on CWD at all, anchor everything to an absolute project path:

```bash
PROJECT=/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework
RUN=2026-06-22-v4-rmse
RUN_DIR="$PROJECT/src/baselines/lstm_baseline/experiments/$RUN"
mkdir -p "$RUN_DIR"

cd "$PROJECT"
PYTHONPATH="$PROJECT/src" "$PROJECT/.venv/bin/python3" -u \
    "$PROJECT/src/baselines/lstm_baseline/lstm_model.py" \
    --run-name "$RUN" \
    2>&1 | tee "$RUN_DIR/log.log"

"$PROJECT/.venv/bin/python3" "$PROJECT/src/baselines/lstm_baseline/plot_experiment.py" "$RUN_DIR"
```

## Existing runs

| Run | Loss | Test MAE | Test RMSE | Notes |
|---|---|---|---|---|
| `2026-06-21-v1-mape` | MAPE | — | — | MAPE blew up on near-zero Polymarket prices; test MAPE 0.587 unreadable |
| `2026-06-21-v2-rmse` | RMSE | 0.0079 | 0.0165 | First MAE/RMSE run; no artifacts saved |
| `2026-06-21-v3-rmse` | RMSE | 0.0118 | 0.0179 | Adds saved history + predictions + plots |
| `2026-06-21-v4-rmse` | RMSE | 0.0110 | 0.0176 | First run on the new `--run-name` CLI layout |
| `2026-06-21-v5-e15`  | RMSE | 0.0089 | 0.0166 | v5 sweep, 15 epochs, seed=0; see `experiments/2026-06-21-v5-summary.md` |
| `2026-06-21-v5-e20`  | RMSE | 0.0072 | 0.0157 | v5 sweep, 20 epochs, seed=0 |
| `2026-06-21-v5-e25`  | RMSE | 0.0103 | 0.0171 | v5 sweep, 25 epochs, seed=0 |
| `2026-06-21-v5-e50`  | RMSE | 0.0116 | 0.0185 | v5 sweep, 50 epochs, seed=0 |
| `2026-06-21-v5-e100` | RMSE | 0.0093 | 0.0167 | v5 sweep, 100 epochs, seed=0 |

## Current latest checkpoint

`checkpoints/lstm_baseline_4h_seq64.pth` at project root is a copy of `experiments/2026-06-21-v3-rmse/checkpoint.pth`.
