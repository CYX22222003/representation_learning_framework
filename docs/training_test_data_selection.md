# Training and Test Data Selection

This document specifies exactly which data subset each model component uses for training and evaluation. The rules here exist to prevent data leakage and ensure all models are compared fairly on the same held-out test set.

---

## The Global Split (already implemented)

The 80/20 chronological split applied per contract in `src/data_processing/data_processing.py` is the **only** split boundary that determines what is "seen" vs. "unseen" during model development.

```
Per-contract time series  →  first 80% = train portion
                          →  last  20% = test portion

After merging across all top-50 contracts:
  data/processed/*.npz  →  key "train"  shape [N_train, seq_len, 5]
                        →  key "test"   shape [N_test,  seq_len, 5]
```

**The test split is locked.** No model parameters — supervised or unsupervised — may be influenced by test sequences until evaluation under the predeclared model × task × epoch-budget matrix.

---

## Train / Test only — no validation split

This project deliberately uses **train and test partitions only**. There is no validation split, no early stopping, and no model-selection step that reads test metrics.

**Why no validation split:**

1. **Architectures are fixed from the start.** For external benchmarks (LSTM, Raw LSTM volatility, GARCH--LSTM stacking, GINN, TA-MLP) the architecture and most hyperparameters come from the upstream paper or a predeclared adaptation plan — we are not tuning them. For the framework, the design is fixed by `src/aggregation/`, `src/models/`, and `src/tasks/` and is not driven by test-loss feedback. There is therefore nothing meaningful for a validation set to *select between*.
2. **Cross-baseline comparison stays clean.** If one baseline used a val split for early stopping and another did not, the comparison would conflate "better representation" with "better stopping rule." Forcing every model to a fixed-epoch budget removes that confound.
3. **Test-set integrity.** A val split inevitably gets watched alongside training. Pretending it's separate is fragile. Removing it keeps the test-set protocol unambiguous: evaluate each predeclared configuration and epoch budget, report the complete matrix, and do not add or select configurations after reading its test metrics.

**What replaces a val split for training control:**

- **Fixed epoch budgets.** Every training run specifies `--epochs N` explicitly. No early stopping.
- **Characterization sweeps.** To understand a model's behavior across training durations, run the same configuration at several epoch budgets (canonical sweep: `[15, 20, 25, 50, 100]` with one fixed seed). Report the full sweep — do **not** pick the best-on-test entry, because that turns the test set into a tuning set. See `src/baselines/lstm_baseline/README.md` and `src/baselines/ta_mlp_baseline/README.md` for the operational details.
- **Multi-seed runs for noise calibration.** Optionally, run the same epoch budget at multiple seeds to estimate the noise band on test metrics. Report mean ± std so claims like "the framework beats the baseline" can be assessed against that noise.

**Implication for components that classically *want* a val signal** (unsupervised pretraining of VAE / contrastive encoders): use a fixed training budget for those too. If a stopping criterion is genuinely needed during development, monitor the **training loss curve** itself (does it plateau?) rather than introducing a val split. Once the project commits to a recipe, lock it in as the canonical pretraining config and apply it uniformly.

---

## Data Usage Per Component

| Component | Trains on | Evaluated on |
|---|---|---|
| VAE encoder | train data (fixed-epoch pretraining) | — (frozen after pretraining) |
| Contrastive encoder | train data (fixed-epoch pretraining) | — (frozen after pretraining) |
| BYOL encoder | train data (fixed-epoch pretraining) | — (frozen after pretraining) |
| Additional neural encoders (TBD) | train data | — (frozen after pretraining) |
| Statistical features | *(deterministic — no fitting)* | — |
| Transformation features | *(deterministic — no fitting)* | — |
| RepresentationAggregator | train feature bundles (fixed-epoch) | test feature bundles |
| Task heads (price, volatility, trend) | train feature bundles + train task labels/targets (fixed-epoch) | test feature bundles + test task labels/targets |
| LSTM baseline | train sequences (fixed-epoch sweep) | test sequences |
| Raw LSTM volatility benchmark | train sequences + shared volatility label bundle (fixed-epoch sweep) | test sequences + shared volatility label bundle |
| GARCH--LSTM stacking volatility benchmark | train-only expanding OOF base predictions for fixed ElasticNet meta-features; GARCH fit/scaling/caps use allowed training prefixes only | locked test rows using reused Raw LSTM test predictions and train-fitted GARCH/meta parameters; a complementary hybrid comparator, not a replacement for Raw LSTM |
| TA-MLP baseline (trend classification) | train TA-feature rows + train tri-class labels (fixed-epoch sweep) | test TA-feature rows + test tri-class labels |
| Raw-OHLCV MLP baseline | train sequences (fixed-epoch); volatility comparison must consume the shared volatility bundle | test sequences; legacy volatility artifacts are characterization-only until migrated to the shared bundle |
| Single-branch ablations | train feature bundles (fixed-epoch) | test feature bundles |
| Raw-OHLCV alpha / GP dry runs | only the original per-contract train portion, internally divided into chronological 60% discovery and 20% confirmation; the global test rows are sliced away before terminal/factor construction; GP fitness, evolution, sign choice, and selection use discovery only | no global-test evaluation; frozen candidates require a fresh later holdout and a cost-aware backtest |
| Direct-representation GP exploration | saved Phase-1 feature rows aligned back to original train-only contract timestamps; a fixed coordinate lattice is standardized on discovery only, then GP fit/selection occurs only on discovery | chronological confirmation only; no global-test use and no interpretation as a tradeable or economically named factor |
| Exhaustive representation + OHLCV GP exploration | all 445 saved coordinates and five causal OHLCV terminals, aligned to the representation window end; all terminals seed the initial discovery-only GP population, with discovery-fitted scaling | chronological confirmation only; exploratory comparison against raw factors, not a final alpha or trading evaluation |
| Future additional symbolic alpha mining (outside current budget) | chronological OOF predictions from downstream heads on aligned training rows; GP fits/selects formulas only on those rows | requires a fresh, still-unseen holdout or temporally later data once the current test split has been used for task evaluation |

All entries share the same `data/processed/*.npz` train/test split. There is no per-component val split.

---

## Feature Extraction for the Test Set

After neural encoders are trained and frozen, their weights are fixed. Running the frozen encoder on test sequences is **not leakage** — it is equivalent to applying a fitted scaler. The encoder parameters contain no information derived from test sequences.

```
Frozen VAE encoder + Frozen contrastive encoder + Frozen BYOL encoder
        │
        ├── inference on train sequences → named train neural embeddings
        └── inference on test sequences  → named test neural embeddings

Statistical + transform features computed independently for both splits.

Combined into FeatureBundle:
  data/features/*.npz
    keys: statistical, transformed, and one key per frozen neural branch
          such as vae, contrastive, or byol
  data/features/*.npz.index.npz
    keys: train_size, test_size
```

The `NpzFeatureStore` handles branch-aware save/load. Feature arrays are stored as train+test concatenated rows, and the companion `.index.npz` stores `train_size` and `test_size` so downstream code can recover the split boundary. Legacy stores with an empty or packed `neural` key remain loadable, but new frozen neural embeddings should be stored as separate branch keys.

## Task Label Bundles

Task labels and targets must respect the same split boundary as the processed sequences. Build train labels from `processed["train"]` only and test labels from `processed["test"]` only. Do not concatenate train and test sequences before applying a future horizon, because the final train rows would then look across the train/test boundary.

The current trend-classification MVP uses a saved TA-MLP-style tri-class label bundle:

```text
data/task_labels/trend_classification/triclass_4h_seq64_top50.npz
data/task_labels/trend_classification/triclass_4h_seq64_top50.npz.manifest.json
```

Its labels are BUY/HOLD/SELL classes. Thresholds are fit per contract using training rows only, then applied to train and test rows. The final `f_window` rows are dropped inside each split because their future target would not be available within that split. The bundle stores train/test labels and aligned feature-row indices so downstream framework runs and baselines can use identical rows.

Strict comparison with the TA-MLP benchmark should reuse this saved label contract or regenerate TA-MLP labels with the same thresholds and row alignment. Existing TA-MLP results remain useful characterization evidence, but they are not a fully strict row-by-row comparison until this alignment is enforced.

The volatility benchmark label bundle is saved under:

```text
data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz
data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz.manifest.json
```

It stores realised-volatility targets plus train/test row indices, contract IDs, and window starts. The Raw LSTM volatility benchmark, GARCH--LSTM stacking benchmark, future framework volatility task, and Raw-OHLCV MLP volatility baseline must reuse this bundle so targets and row identities match exactly. The stacking benchmark's expanding cross-fitting creates honest out-of-fold training meta-features for ElasticNet; it is not a validation split, early-stopping signal, or hyperparameter-selection mechanism. The existing Raw-OHLCV MLP volatility sweep predates this contract and uses the legacy merged-array target helper; it remains characterization evidence only and is not a strict row-by-row comparison with the two completed volatility benchmarks.

---

## Order of Operations

The sequence below must be followed to avoid leakage.

```
1. data/processed/*.npz already exists
        │
        ▼
2. Pretrain VAE on the full train split for a fixed epoch budget
   (no val split; track training-loss curve for sanity, not for stopping)
   Save checkpoint: checkpoints/vae_<timeframe>.pth
        │
        ▼
3. Pretrain contrastive encoder on the full train split for a fixed epoch budget
   Save checkpoint: checkpoints/contrastive_<timeframe>.pth
        │
        ▼
   Pretrain BYOL encoder on the full train split for a fixed epoch budget
   Save checkpoint: checkpoints/byol_<timeframe>.pth
        │
        ▼
   (repeat for any additional neural encoders — TBD)
        │
        ▼
4. Extract features for ALL sequences using frozen encoders:
     - statistical + transform: computed directly (no encoder needed)
     - neural branches: run frozen encoder inference and store by branch name
   Save: data/features/*.npz plus data/features/*.npz.index.npz
        │
        ▼
5. Build task labels/targets from each split independently:
     - price targets from train/test processed sequences
     - volatility targets from the shared contract-aware volatility label bundle
     - trend labels from the saved tri-class task label bundle
   Fit any label thresholds or target scalers using train data only
        │
        ▼
6. Train RepresentationAggregator + task head on the full train feature bundle
   for a fixed epoch budget; save checkpoint per task
        │
        ▼
7. Train all baseline models on the full train sequences (or full train feature
   bundles for single-branch ablations) for a fixed epoch budget.
   For external benchmarks, run a small characterization sweep across epoch
   budgets at one fixed seed — report the full sweep, not a "best" run.
        │
        ▼
8. ── CURRENT-BUDGET TASK EVALUATION ──
   Load every predeclared task checkpoint for each fixed epoch budget; run
   inference on test feature bundles / test sequences. Record and report the
   complete model × task × epoch-budget metric matrix for price, volatility,
   and trend. Do not select or add a configuration from these test results.
        │
        ▼
9. Future additional alpha-research capability (not part of the current budget):
   - The raw-OHLCV dry run may screen a predeclared, small formula set inside
     the train portion only (60% discovery / 20% confirmation per contract).
     It must slice away global test rows before rolling-factor construction and
     cannot be reported as final alpha evaluation or trading performance.
   - The raw-OHLCV GP dry run obeys the same split: all evolutionary fitness,
     formula direction, and non-redundancy selection are discovery-only, with
     confirmation used solely for frozen-tree diagnostics.
   - Direct representation-coordinate GP is an exploratory exception to the
     economic-terminal preference. It must predeclare its coordinate set, fit
     scaling and selection only on discovery, and report negative as well as
     positive confirmation results without calling coordinates alpha factors.
   - Generate chronological OOF training predictions from each selected head.
     A head must not train on the row it predicts or any later row.
   - Fit symbolic GP only on these OOF, decision-time primitive predictions
     (for example return/movement, volatility, trend probabilities, and
     confidence margins) and aligned training objectives.
   - Predeclare the grammar, depth/window limits, selection metric, and
     redundancy threshold. Purge/embargo rows whose forecast horizons overlap
     a fold boundary when necessary.
   - Refit the fixed heads on the entire train split. Do not use the current
     task-evaluation test rows, ICs, or backtests to choose primitives or
     formulas. A future factor evaluation requires a fresh holdout or later
     temporal data.
```

---

## Aggregator Mode and Task Head Input Dimension

`RepresentationAggregator` supports two modes (set at construction time):

- **`mode="concat"` (default)**: branches are concatenated; `output_dim = sum of branch dims`. No learnable parameters in the aggregator itself — the task head does all the supervised learning.
- **`mode="gated"`**: each branch is projected to `out_dim`, then a gating network produces softmax weights. `output_dim = out_dim`. Adds learnable parameters; useful as a comparison once concat baseline results are available.

In both modes, `agg.output_dim` returns the correct input dimension for the task head:

```python
agg = RepresentationAggregator(branch_dims, mode="concat")
task_head = nn.Linear(agg.output_dim, n_outputs)  # works for both modes
```

Both modes are trained on the same data splits and evaluated identically, making them directly comparable in the ablation study.

---

## Rules Summary

1. **Split once, at the start.** The 80/20 per-contract split is already done. Do not re-split.
2. **Train and test only — no validation split.** Every model uses a fixed epoch budget. No early stopping.
3. **Unsupervised ≠ exempt from the split.** VAE, contrastive, and BYOL encoders are trained on `train_data` only, never on test sequences.
4. **Never pick a run by reading test metrics.** Characterization sweeps across epoch budgets are reported in full; selecting the best-on-test entry turns the test set into a tuning set.
5. **Task labels, thresholds, and scalers are fit from training data only.** Test labels may be computed only after all threshold/scaler parameters are fixed from train data, and label horizons must not cross the split boundary.
6. **All strict comparisons use identical train/test partitions and label rows** — same `.npz` files, same split indices, and the same saved task label bundle where one exists. Legacy artifacts that predate a bundle are characterization evidence, not strict comparison evidence.
7. **Test evaluation follows a predeclared matrix.** Evaluate each fixed model, task, seed, and epoch budget once; report the complete matrix. Do not add configurations, select a best-on-test run, or otherwise change the protocol after reading test metrics.
8. **Frozen encoder inference on test sequences is valid.** Encoder weights are fixed; no test-set gradient flows back.
9. **Alpha-factor research is training-only model selection.** Raw-OHLCV screens and bounded GP may use a chronological discovery/confirmation split inside the original train portion, while the framework-facing symbolic search must use OOF economically meaningful downstream predictions rather than arbitrary latent coordinates. All require a fresh holdout or later data for final evaluation.
