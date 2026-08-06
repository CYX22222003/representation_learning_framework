# GINN Baseline Experiment Design

- **Date:** 2026-08-04
- **Status:** Approved design
- **Initial scope:** Polymarket 4-hour data only
- **Future extension:** The same interfaces must support 1-hour and 1-day data without redesign.

## 1. Purpose

This work completes the GINN external benchmark for the volatility-prediction task. The benchmark retains the defining GINN pipeline—autoregressive mean modelling, GARCH volatility modelling, and an LSTM trained with fused realised-volatility and GARCH supervision—while aligning its data split, target, metrics, and experiment protocol with the representation-learning framework.

The primary goal is a fair framework comparison, not a byte-for-byte reproduction of the reference implementation in `../baselines/GINN`. The reference is used to preserve the model identity and default architecture; project rules take precedence where its protocol conflicts with the framework.

## 2. Fixed Experiment Decisions

- Use the top-50 Polymarket contracts for the 4-hour timeframe.
- Use a chronological 80/20 train/test split per contract.
- Use no validation set and no early stopping.
- Fit every learned or estimated statistical parameter on training-visible history only.
- Use 64-timestep inputs.
- Use all five OHLCV-derived residual channels as LSTM input.
- Predict the framework's next-shifted-window realised volatility target.
- Evaluate against realised volatility with MSE and Pearson correlation.
- Run a fixed-epoch characterization sweep at `[15, 20, 25, 50, 100]` with seed `0`.
- Report every epoch budget; do not choose a best checkpoint from test metrics.
- Cache prepared GINN arrays once and reuse the identical cache for every epoch budget.

## 3. Relationship to the Reference GINN

The reference implementation uses a learned univariate AR(90), GARCH on its residuals, a two-layer LSTM with hidden size 256, and pointwise squared residuals as the variance target. It also uses a validation split and validation-based checkpoint selection.

The framework-aligned benchmark intentionally changes the following:

| Concern | Reference | Framework-aligned GINN |
|---|---|---|
| Input | Close-only residuals | Five OHLCV-derived residual channels |
| AR method | Learned linear AR(90) | Per-channel least-squares AR(5) |
| Sequence length | 90 | 64 |
| Target | Pointwise squared AR residual | Next-shifted-window realised close volatility |
| GARCH supervision | Conditional variance | Window volatility derived from conditional variance |
| Split | 85/15 train/validation | 80/20 train/test per contract |
| Model selection | Best validation loss | Fixed epoch budgets; no selection |
| Primary metrics | Validation fused loss | Test MSE and Pearson correlation against realised volatility |

The LSTM architecture and fused-loss idea remain the benchmark's defining elements.

## 4. Source Structure

The GINN implementation will be separated by responsibility:

```text
src/baselines/ginn_baseline/
  ginn_data.py          # Pure contract-level transforms and cache construction
  prepare_data.py       # CLI for creating and validating the cached dataset
  ginn_model.py         # LSTM, fused loss, training epoch, and evaluation
  run_experiment.py     # Reproducible fixed-epoch sweep and artifact writing
  plot_experiment.py    # Per-budget and cross-sweep visualisations
  README.md             # Protocol, commands, artifact contract, and result index
```

Tests will use synthetic arrays and temporary directories only:

```text
tests/baselines/ginn_baseline/
  test_ginn_data.py
  test_ginn_model.py
  test_ginn_experiment.py
  test_ginn_plots.py
```

## 5. Contract-Level Data Preparation

### 5.1 Input selection and cleaning

Each contract is processed independently so no AR window, GARCH recursion, target, or metric spans two contracts.

For each filename in the existing 4-hour top-50 file list:

1. Read the raw Feather file.
2. Require `open`, `high`, `low`, `close`, and `volume` columns.
3. Apply the project's forward-fill followed by interpolation convention.
4. Reject the contract if it remains non-finite or is too short to form the required samples.

The experiment uses raw contract files rather than the merged processed NPZ because the current processed format discards contract boundaries. It nevertheless uses the same contract list, chronological split convention, timeframe, sequence length, and target definition as the framework.

### 5.2 Stationary series

Create a length-`T` transformed series aligned with the original rows:

- For each OHLC column, clip values to `1e-8`, take the natural logarithm, and first-difference it.
- For volume, clip negative values to zero, apply `log1p`, and first-difference it.
- Set the first transformed value in each channel to zero so transformed arrays preserve the original row indexing.

### 5.3 Split indices and training-visible history

For `T` cleaned rows and sequence length `L = 64`, framework-aligned sample starts are:

```text
start = 0, 1, ..., T - L - 1
```

There are `N = T - L` possible next-shifted-window samples. Define:

```text
n_train = floor(0.8 * N)
train starts = [0, n_train)
test starts  = [n_train, N)
```

The final training sample has start `n_train - 1`; its shifted target ends at raw row `n_train + L - 1`. Therefore statistical fitting may use raw rows in the exclusive range:

```text
[0, fit_end), where fit_end = n_train + L
```

This is the complete history visible to training inputs and training targets. No later row may influence AR coefficients, GARCH parameters, centring values, or initialization values.

AR warm-up starts below `ar_order` are removed from the training-start list. The chronological test boundary remains `n_train`; test samples are not moved into training to replace removed warm-up samples.

### 5.4 AR fitting and residual generation

Fit an independent AR(5) model to each transformed channel using only values before `fit_end`. The least-squares model follows the project's existing lag convention and has no intercept.

Store the five coefficient vectors. Apply them unchanged to the full transformed contract series to obtain a length-`T`, five-channel residual array. Residual values before the fifth lag are zero and are never used as model samples.

Changing rows at or after `fit_end` must not change fitted AR coefficients or any prepared training array.

### 5.5 GARCH fitting and conditional variance

Use only the close-channel AR residuals before `fit_end` to fit GARCH(1,1). Store:

- `omega`, `alpha`, and `beta`;
- the close-residual mean calculated on fitting history;
- the initial variance calculated on fitting history;
- optimizer convergence or fallback status.

Apply the stored parameters chronologically to the full close-residual series. Test residuals may update later conditional variances through the fixed GARCH recursion, but they must never update fitted parameters or fitting statistics.

If constrained GARCH optimization fails, use the project's valid deterministic initialization fallback and mark the contract as a fallback in the manifest. AR fitting failure, irrecoverable non-finite output, or insufficient length causes the contract to be skipped with an explicit reason.

### 5.6 Sample and target construction

For a valid sample start `i`:

```text
X[i] = AR residuals[i : i + 64, :]
```

The ground-truth target exactly follows the framework's shifted-window definition:

```text
future_close = close[i + 1 : i + 65]
y_gt[i] = sqrt(mean(diff(log(clip(future_close, 1e-8))) ** 2))
```

The GARCH supervision target covers the same shifted window and uses volatility units:

```text
future_sigma2 = conditional_variance[i + 1 : i + 65]
y_garch[i] = sqrt(mean(future_sigma2))
```

Taking the square root is required. The current local implementation mixes realised volatility with mean conditional variance, which gives the two fused-loss terms incompatible units.

Construct train and test samples from their respective start-index lists before merging contracts. Assignment is determined only by the sample start: no test-start sample enters the training arrays, and no sample or target crosses a contract boundary. Because the existing framework splits stride-one sliding windows after window construction, the final training windows and first test windows share some raw timesteps. This documented overlap is preserved for parity; it must not be confused with fitting AR or GARCH parameters on rows after `fit_end`.

## 6. Cached Dataset Contract

Data preparation writes:

```text
data/processed/ginn_4h_seq64_top50.npz
data/processed/ginn_4h_seq64_top50.manifest.json
```

The compressed NPZ contains:

```text
X_train             float32 [N_train, 64, 5]
y_gt_train          float32 [N_train, 1]
y_garch_train       float32 [N_train, 1]
contract_id_train   int32   [N_train]
X_test              float32 [N_test, 64, 5]
y_gt_test           float32 [N_test, 1]
y_garch_test        float32 [N_test, 1]
contract_id_test    int32   [N_test]
```

The JSON manifest records:

- a stable `dataset_id` derived from the preprocessing configuration, ordered source list, and SHA-256 digest of each included source file;
- timeframe, sequence length, top-K, split ratio, AR order, and GARCH order;
- transformed-series definitions and target definitions;
- ordered contract-ID-to-filename mapping;
- per-contract raw-row, training-sample, and test-sample counts;
- skipped contracts and reasons;
- GARCH convergence/fallback status per included contract;
- total array shapes and dtypes;
- non-finite and negative-target counts, which must both be zero.

The preparation command validates all arrays immediately after writing them. Training rejects a cache whose manifest disagrees with its requested timeframe, sequence length, top-K, AR order, GARCH order, shape, dtype, or `dataset_id`.

## 7. Model and Loss

The model retains the local GINN LSTM architecture:

```text
Input: [batch, 64, 5]
2-layer LSTM(input_size=5, hidden_size=256, dropout=0.1)
last-layer final hidden state
Linear(256 -> 128)
BatchNorm1d(128)
ReLU
Linear(128 -> 1)
```

The final layer remains unconstrained to preserve the reference architecture. Evaluation records the number of negative predictions as a diagnostic.

For prediction `p`, realised target `y_gt`, GARCH target `y_garch`, and `lambda_garch = 0.3`:

```text
loss_gt = MSE(p, y_gt)
loss_garch = MSE(p, y_garch)
loss_total = (1 - lambda_garch) * loss_gt
           + lambda_garch * loss_garch
```

Training history must retain all three loss components rather than only the total.

## 8. Training Protocol

Default configuration:

| Setting | Value |
|---|---|
| Timeframe | `4h` |
| Sequence length | `64` |
| Input channels | `5` |
| Hidden size | `256` |
| LSTM layers | `2` |
| LSTM dropout | `0.1` |
| Batch size | `64` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| GARCH loss weight | `0.3` |
| Seed | `0` |
| Epoch budgets | `[15, 20, 25, 50, 100]` |
| Validation | None |
| Early stopping | None |

Seed Python, NumPy, PyTorch, CUDA, and the training DataLoader generator. Shuffle training samples with the seeded generator; never shuffle test samples.

Train one trajectory through epoch 100. At epochs 15, 20, 25, 50, and 100:

1. Save a checkpoint immediately.
2. Preserve the loss-history prefix through that epoch.
3. Do not inspect test data while training continues.

After training finishes, load and evaluate every saved checkpoint in one final sweep-evaluation session. This produces the same seeded training trajectory that separate fixed-budget runs would produce while avoiding five redundant training jobs.

Every checkpoint is reported. Test metrics must not determine which checkpoint is retained, highlighted as best, or used for later framework comparison. A later comparison must predeclare its epoch budget independently of these test results.

## 9. Evaluation

Official metrics compare model predictions only with `y_gt_test`:

- mean squared error;
- Pearson correlation.

Diagnostics include:

- MAE and RMSE against `y_gt_test`;
- GARCH-only MSE and Pearson correlation from `y_garch_test` against `y_gt_test`;
- number and fraction of negative model predictions;
- prediction, target, and error summary statistics;
- per-contract MSE and correlation where a contract has at least two non-constant samples.

If Pearson correlation is undefined because either vector is constant or contains fewer than two values, store JSON `null` and record a warning. Non-finite predictions or targets fail evaluation rather than being silently removed.

## 10. Experiment Artifacts

The initial run family uses:

```text
src/baselines/ginn_baseline/experiments/2026-08-04-v1/
```

The root contains:

```text
config.json
dataset_manifest.json
log.log
sweep_metrics.json
summary.md
images/
  epoch_sweep.png
```

Each budget directory (`e15`, `e20`, `e25`, `e50`, `e100`) contains:

```text
checkpoint.pth
history.npz
predictions.npz
metrics.json
summary.md
images/
  training_curve.png
  pred_vs_actual.png
  error_distribution.png
```

### 10.1 Machine-readable schemas

`history.npz` contains:

```text
train_total_loss
train_gt_mse
train_garch_mse
epochs
seed
```

`predictions.npz` contains:

```text
preds
targets
garch_targets
contract_ids
```

`metrics.json` contains the official metrics, diagnostics, sample count, epoch, seed, dataset ID, and model configuration. `sweep_metrics.json` contains one ordered entry per epoch budget.

The checkpoint contains model state, completed epoch, architecture configuration, training configuration, seed, and dataset ID. Optimizer state is included so a run can be resumed for diagnostic purposes, but resumed training does not replace the canonical sweep.

### 10.2 Visual records

- `training_curve.png`: total fused loss, realised-target MSE, and GARCH-target MSE through the saved epoch.
- `pred_vs_actual.png`: predicted versus realised volatility with an identity line, MSE, correlation, and sample count.
- `error_distribution.png`: histogram of `prediction - target` with a zero reference line and error summary.
- `epoch_sweep.png`: MSE and Pearson correlation for every epoch budget, with no best-run annotation.

### 10.3 Written records

Each budget summary records configuration, sample counts, final training losses, official metrics, diagnostics, artifact paths, and caveats. The root summary reports the complete sweep table and characterizes behavior across budgets. It must not describe the lowest test error as the selected or optimal run.

## 11. Command-Line Behavior

All commands run from the project root and follow the repository's Python-environment guidance.

`prepare_data.py` accepts explicit timeframe, sequence length, top-K, AR order, output path, manifest path, and overwrite controls. It prints a manifest summary and exits non-zero on validation failure.

`run_experiment.py` accepts the cache path, manifest path, run name, epoch budgets, seed, batch size, learning rate, GARCH loss weight, and device. It refuses to overwrite a non-empty run directory unless an explicit overwrite flag is supplied.

`plot_experiment.py` accepts a run root or one budget directory and generates visuals only from saved artifacts. Plot generation never retrains or rereads raw market data.

## 12. Error Handling and Auditability

- Data-preparation errors are associated with a contract filename and recorded in the manifest.
- A missing required column, insufficient contract length, failed AR fit, or non-finite prepared output cannot be silently ignored.
- A GARCH fallback is allowed only when explicitly recorded.
- Empty train/test arrays, incompatible shapes, non-float32 model arrays, non-int32 contract IDs, negative targets, or non-finite values abort preparation or training.
- Existing experiment directories are protected from accidental overwrite.
- Logs include the resolved device, dataset ID, seed, full configuration, sample counts, model parameter count, checkpoint paths, and metric paths.
- The user's existing modification to `docs/design.md` is outside this work and must remain untouched.

## 13. Verification Strategy

Tests do not read the repository's real `data/` directory.

### 13.1 Data tests

- Synthetic contracts produce arrays with the documented shapes and dtypes.
- Contract IDs prove no sample crosses a contract boundary.
- Start-index tests prove train and test arrays follow the declared per-contract boundary and contain only the documented stride-one raw-window overlap.
- Perturbing only rows at or after `fit_end` does not change AR coefficients, GARCH parameters, or prepared training arrays.
- Realised and GARCH targets match hand-calculated volatility examples.
- Both targets are finite, non-negative, and expressed in volatility units.
- GARCH fallback and skipped-contract reasons appear in the manifest.
- Repeating preparation with identical inputs produces the same dataset ID and arrays.

### 13.2 Model tests

- The LSTM maps `[batch, 64, 5]` to `[batch, 1]`.
- `lambda_garch = 0` reduces the fused loss to realised-target MSE.
- `lambda_garch = 1` reduces it to GARCH-target MSE.
- A training epoch updates model parameters and returns all three finite loss components.
- Evaluation reproduces hand-calculated MSE and handles undefined correlation as specified.

### 13.3 Experiment and plotting tests

- A one-epoch CPU smoke run on a synthetic cache creates the complete artifact schema.
- Checkpoints reload and reproduce stored predictions within numerical tolerance.
- Manifest mismatch causes a clear failure before training starts.
- A protected non-empty run directory is not overwritten.
- Plot generation creates every expected non-empty PNG from saved synthetic artifacts.

## 14. Real Experiment Sequence

1. Implement and pass the synthetic data-preparation tests.
2. Prepare `ginn_4h_seq64_top50.npz` once.
3. Audit its manifest, sample shapes, target distributions, skipped contracts, and GARCH fallback count.
4. Run a one-epoch CPU or GPU smoke experiment in a clearly named non-reportable smoke directory.
5. Run the canonical seed-0 trajectory with checkpoints at 15, 20, 25, 50, and 100 epochs.
6. Evaluate all checkpoints after the training trajectory completes.
7. Generate every per-budget and cross-sweep visual.
8. Verify machine-readable metrics against saved predictions.
9. Write per-budget summaries and the aggregate sweep summary.
10. Update the GINN README with commands, artifact locations, and the complete result table.

## 15. Success Criteria

The GINN baseline is considered trained and recorded when:

- the cached 4-hour dataset passes all validation checks;
- AR and GARCH fitting use training history only;
- the smoke run passes;
- all five canonical checkpoints and artifact sets exist;
- every checkpoint has MSE and Pearson correlation against the same realised-volatility test target;
- all plots render from saved artifacts;
- summaries report the complete sweep without test-based checkpoint selection;
- the process is reproducible from documented project-root commands;
- 1-hour and 1-day extension requires only different CLI arguments and new caches, not source redesign.
