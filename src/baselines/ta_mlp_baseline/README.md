# TA-MLP Baseline

External benchmark for the **trend classification** task. A 4-layer MLP (36→128→64→32→n_classes, LeakyReLU) over 36 TA-Lib technical indicators (oscillators, moving-average ratios, calendar features, 23 candlestick patterns). Used as a comparison point for the multi-branch framework on a non-regression downstream task.

## Splitting convention

Same as the LSTM baseline: **train / test only**, no validation split, no early stopping. Each contract is split chronologically 80 / 20; per-contract `StandardScaler`-style z-score is fitted on train rows only, then applied to both halves. Training runs for a fixed number of epochs supplied via `--epochs`.

## Label modes

Two modes are available; the canonical sweep uses `triclass`.

| `--label-mode` | Classes | Source |
|---|---|---|
| `triclass` (default) | BUY / HOLD / SELL (0 / 1 / 2) | Upstream paper's labeling — see `ta_labels.py` and `REFERENCE/` |
| `binary` | DOWN / UP (0 / 1) | Whether close rises over the next `--horizon` candles |

For `triclass`, the thresholds `alpha` and `beta` are quantiles of `|pct_change|` (defaults: 0.85 / 0.997), fitted **per contract on training rows only** to avoid leaking future-period statistics into the labels. The labeling formula and the original constants are documented in `ta_labels.py`.

The triclass distribution is heavily HOLD-dominated by design (~80% HOLD, ~10% each BUY/SELL). Accuracy alone is therefore not a meaningful headline — **macro-F1 and per-class recall are the load-bearing metrics**.

## Multi-run sweeps

We run the model at several `--epochs` budgets to **characterize how the baseline behaves under different training durations** — e.g. how train loss decays, when test performance flattens or starts to drift, how stable the model is at long horizons. The purpose is comparison and description, **not selection of a "best" run**. Reporting in this baseline therefore covers the whole sweep (every run's metrics), not a hand-picked winner; picking the epoch count by inspecting test metrics would turn the test set into a tuning set.

Seeds are fixed (`--seed`) across the sweep so that differences between runs reflect the epoch budget rather than initialization noise.

### Canonical sweep

Current epoch budgets used for the TA-MLP baseline characterization sweep:

| `--epochs` | Purpose |
|---|---|
| 15 | Around where prior early-stopped runs typically halted |
| 20 | Near the train-loss plateau seen in prior runs |
| 25 | Slightly past plateau — does extra training still move things? |
| 50 | Mid-horizon — well past plateau |
| 100 | Long-horizon — does test error drift up under overtraining? |

All five runs share the same seed (`--seed 0`) so observed differences reflect the training budget alone. The canonical sweep also uses default `--label-mode triclass`, `--b-window 5`, `--f-window 2`, `--hold-q 0.85`, `--buy-sell-q 0.997` (matching the upstream paper).

## Files

| Path | Purpose |
|---|---|
| `ta_mlp_model.py` | Model (`TAMLPClassifier`), training (`train_model`), evaluation (`evaluate`), and CLI runner |
| `ta_features.py` | 36-feature extractor + `build_ta_dataset` with `label_mode={"binary","triclass"}` |
| `ta_labels.py` | Vectorized BUY/HOLD/SELL labeling and quantile thresholds |
| `plot_experiment.py` | Generates `training_curve.png` and `confusion_matrix.png` for a given run directory |
| `REFERENCE/` | Verbatim copy of the upstream labeling code, kept for provenance only — not imported |
| `experiments/` | One subdirectory per run; each is self-contained |

Each `experiments/<run-name>/` directory contains:

```
log.log                  ← stdout of the training run (captured via shell redirection)
summary.md               ← hand-written results summary
history.npz              ← per-epoch train_loss, epochs, seed
predictions.npz          ← preds (argmax), logits, targets on the test set
confusion_matrix.npz     ← cm (n×n int64), class_names
checkpoint.pth           ← saved model state_dict
images/
  training_curve.png
  confusion_matrix.png
```

## Running a new experiment

**The commands below use paths relative to the project root** (the directory that contains `.venv/`, `src/`, `data/`, `checkpoints/`). Run them from there — do not `cd` into `src/baselines/ta_mlp_baseline/` first, or the relative paths to `.venv/bin/python3` and `PYTHONPATH=src` will resolve to the wrong place and the run will fail silently with the log only containing a `No such file or directory` line.

A defensive guard at the top of the snippet catches that mistake:

```bash
# Always run from the project root
test -d .venv && test -d src/baselines/ta_mlp_baseline || {
    echo "ERROR: run this from the project root (the directory that contains .venv/ and src/)" >&2
    return 1 2>/dev/null || exit 1
}

RUN=2026-06-22-v2/e30
RUN_DIR="src/baselines/ta_mlp_baseline/experiments/$RUN"
mkdir -p "$RUN_DIR"

PYTHONPATH=src .venv/bin/python3 -u src/baselines/ta_mlp_baseline/ta_mlp_model.py \
    --run-name "$RUN" \
    --epochs 30 \
    --seed 0 \
    --label-mode triclass \
    2>&1 | tee "$RUN_DIR/log.log"

.venv/bin/python3 src/baselines/ta_mlp_baseline/plot_experiment.py "$RUN_DIR"
```

`--epochs` controls the fixed training budget. `--seed` controls weight init, DataLoader shuffling, numpy / random / CUDA — fixing it makes runs reproducible and makes an epoch sweep actually comparable (otherwise epoch effects and seed effects are confounded). `--label-mode` chooses between the triclass (default) and binary labeling formulations.

Then write `$RUN_DIR/summary.md` describing the setup and results.

If you prefer not to rely on CWD at all, anchor everything to an absolute project path:

```bash
PROJECT=/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework
RUN=2026-06-22-v2/e30
RUN_DIR="$PROJECT/src/baselines/ta_mlp_baseline/experiments/$RUN"
mkdir -p "$RUN_DIR"

cd "$PROJECT"
PYTHONPATH="$PROJECT/src" "$PROJECT/.venv/bin/python3" -u \
    "$PROJECT/src/baselines/ta_mlp_baseline/ta_mlp_model.py" \
    --run-name "$RUN" \
    2>&1 | tee "$RUN_DIR/log.log"

"$PROJECT/.venv/bin/python3" "$PROJECT/src/baselines/ta_mlp_baseline/plot_experiment.py" "$RUN_DIR"
```

## Existing runs

| Run | Mode | Accuracy | Macro-F1 | F1 BUY | F1 HOLD | F1 SELL | Notes |
|---|---|---|---|---|---|---|---|
| `2026-06-22-v1/e15`  | triclass | 0.7116 | 0.4546 | 0.2974 | 0.8625 | 0.2039 | v1 sweep, 15 epochs, seed=0; see `experiments/2026-06-22-v1/summary.md` |
| `2026-06-22-v1/e20`  | triclass | 0.7256 | 0.4524 | 0.3029 | 0.8647 | 0.1897 | v1 sweep, 20 epochs, seed=0 |
| `2026-06-22-v1/e25`  | triclass | 0.7115 | 0.4651 | 0.2936 | 0.8587 | 0.2429 | v1 sweep, 25 epochs, seed=0 |
| `2026-06-22-v1/e50`  | triclass | 0.7118 | 0.4570 | 0.2881 | 0.8542 | 0.2285 | v1 sweep, 50 epochs, seed=0 |
| `2026-06-22-v1/e100` | triclass | 0.7065 | 0.4595 | 0.2674 | 0.8457 | 0.2654 | v1 sweep, 100 epochs, seed=0 |
