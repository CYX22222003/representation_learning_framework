# FYP Progress and Schedule

**Last updated:** 2026-08-09

---

## 1. Current Achievements

### Stage 1 — Data Collection and Processing
| Task | Status |
|---|---|
| Collect Polymarket OHLCV feather files | ✅ Done |
| Select top-50 active contracts per timeframe (1h, 4h, 1d) | ✅ Done |
| Forward-fill and interpolate missing values | ✅ Done |
| Z-score normalise volume | ✅ Done |
| Sliding window segmentation → `[N, seq_len, features]` | ✅ Done |
| 80/20 chronological train/test split per contract | ✅ Done |
| Merge sequences across contracts | ✅ Done |
| Exploratory data analysis (distributions, volatility regimes) | ⬜ Not started |

### Stage 2 — Framework Implementation and Training
| Task | Status |
|---|---|
| **Statistical features** | |
| AR(p) coefficients + residual statistics per column | ✅ Done |
| GARCH(1,1) MLE fit → 7 features per column | ✅ Done |
| **Transformation features** | |
| FFT top-k magnitude coefficients per column | ✅ Done |
| Haar wavelet detail energy (multi-level) per column | ✅ Done |
| **Neural encoders** | |
| VAE architecture (MLP encoder-decoder, β-VAE loss) | 🔄 Implemented with pretraining/report scripts, not trained |
| Contrastive encoder (CNN backbone, NT-Xent loss, augmentations) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| **Aggregation and downstream tasks** | |
| `RepresentationAggregator` (N-branch gated fusion, dict API) | 🔄 Implemented, not trained |
| `PriceRegressor` task head (MAE/RMSE) | 🔄 Implemented, not trained |
| `VolatilityRegressor` task head (MSE, correlation) | 🔄 Implemented, not trained |
| `TrendClassifier` task head (accuracy, F1) | 🔄 Implemented, not trained |
| Branch-aware `FeatureBundle` + `NpzFeatureStore` (save/load pipeline) | ✅ Done; legacy packed `neural` stores remain loadable |
| End-to-end training script | ⬜ Not started |

### Stage 3 — Benchmark and Baseline Implementation and Training

**External benchmarks** (from prior work)
| Task | Status |
|---|---|
| Stacked LSTM benchmark (3-layer, 4h data) | ✅ Trained on unified splits (v5 epoch sweep, seed=0; MAE 0.007–0.012, RMSE 0.016–0.018) |
| GINN benchmark (AR→GARCH→LSTM, volatility) | ✅ Trained on 4h data at 15 epochs; further sweep deferred because of documented GARCH-target failure |
| TA-MLP benchmark (FreqTrade, trend classification) | ✅ Trained on unified splits (v1 triclass epoch sweep, seed=0; acc 0.71–0.73, macro-F1 0.45–0.47) |
| Additional benchmarks from literature review (TBD) | ⬜ TBD |

**Internal baselines** (designed within this project)
| Task | Status |
|---|---|
| Raw-OHLCV MLP (no representation learning) | ✅ Trained on 4h data for price, volatility, and trend; price/volatility sweeps at 15/50/100 epochs |
| Statistical-only ablation | ⬜ Not started |
| Transformation-only ablation | ⬜ Not started |
| VAE-only ablation | ⬜ Not started |
| Contrastive-only ablation | ⬜ Not started |
| Additional neural encoder ablations (per TBD methods) | ⬜ TBD |
| Standalone GARCH for volatility prediction task | ⬜ Not started |
| Additional internal baselines (TBD) | ⬜ TBD |

### Stage 4 — Experiments and Benchmarking
| Task | Status |
|---|---|
| Evaluation harness (unified test loop for all models) | ⬜ Not started; baseline runners evaluate independently |
| Price prediction benchmark (MAE, RMSE) | 🔄 Raw-OHLCV MLP and LSTM results recorded; framework comparison pending |
| Volatility prediction benchmark (MSE, correlation) | 🔄 Raw-OHLCV MLP and GINN results recorded; framework comparison pending |
| Trend classification benchmark (accuracy, F1) | 🔄 Raw-OHLCV MLP and TA-MLP results recorded; framework comparison pending |
| Transferability analysis (across markets and timeframes) | ⬜ Not started |
| Ablation study (per-branch contribution) | ⬜ Not started |
| Result tables and visualisations | 🔄 Raw-OHLCV MLP sweep plots generated; final cross-model tables pending |

---

## 2. Summary

The data pipeline and all three processed timeframes are complete. The deterministic feature extraction code now produces the documented AR+GARCH and FFT+wavelet dimensions (`70` and `55` for 5 OHLCV columns), but the existing `data/features/features_*_seq64_top50.npz` artifacts were generated with older dimensions (`35` and `40`) and should be regenerated before framework evaluation. The contrastive encoder has completed a 4h CUDA pretraining sweep at epoch budgets `[15, 20, 25, 50, 100]`, with checkpoints, histories, metrics, plots, and a report under `experiments/contrastive_encoder/contrastive-4h-seq64-top50/`; the canonical checkpoint is `checkpoints/contrastive_4h_seq64_top50.pth`. The VAE encoder, aggregator, and task heads are implemented but have not yet produced a full end-to-end framework result. The external LSTM and TA-MLP benchmarks have completed fixed-epoch 4h sweeps; GINN has a completed 15-epoch 4h run. The Raw-OHLCV MLP has completed 4h runs for all three tasks, including 15/50/100-epoch sweeps for price and volatility, with training curves, predictions, error plots, and sweep summaries.

The current project state is **Phase B — first working loop**, with the baseline side still ahead of the framework side. One neural encoder pretraining prerequisite is now satisfied by the contrastive checkpoint, but Phase B is not complete because the framework has not yet produced a downstream result. The Raw-OHLCV MLP price sweep reports MAE/RMSE of `0.0834/0.1067` at 15 epochs, `0.0600/0.0805` at 50 epochs, and `0.0454/0.0683` at 100 epochs. For volatility, 15 epochs is the strongest observed budget (`MAE 0.0404`, `RMSE 0.0931`, correlation `0.7279`); longer training reduces train loss but worsens test metrics, indicating overfitting or distribution mismatch. These are characterization results, not test-selected checkpoints. No claim that the framework is better than the baseline is currently supported because no framework task result exists on the same test split.

The immediate blocker is the first reproducible framework loop: regenerate deterministic feature bundles with the current feature extractors, extract frozen train/test features from the pretrained contrastive encoder as a named `contrastive` branch, combine them in the branch-aware feature store, train the aggregator plus task head, and evaluate with the same target builders and metrics. VAE pretraining remains pending if the first full multi-branch run should include both neural branches. After that, build a unified comparison table, add branch ablations, and run multi-seed confirmation before making superiority claims. The repository-local `baseline-comparison` and `wsl-cuda-experiments` skills document the fairness and WSL/CUDA execution procedures.

### Recent contrastive encoder progress

The contrastive encoder was successfully trained on the NVIDIA GPU through WSL using the unified 4h, sequence-length-64 dataset. The run used only the locked training split (`109841` sequences) and recorded the test split shape (`27500` sequences) for traceability only. No test sequences were used for training, early stopping, or checkpoint selection.

| epoch budget | train NT-Xent loss | best train loss so far | elapsed seconds |
|---:|---:|---:|---:|
| 15 | 2.5918 | 2.5918 | 214.70 |
| 20 | 2.5496 | 2.5496 | 287.62 |
| 25 | 2.5225 | 2.5225 | 364.87 |
| 50 | 2.4600 | 2.4599 | 727.67 |
| 100 | 2.4100 | 2.4085 | 1460.38 |

The loss decreased consistently and plateaued gradually, so the result is meaningful as unsupervised pretraining evidence. It is not yet downstream performance evidence; the next required check is frozen-embedding probing through the framework task loop.

### Recent GINN progress

The GINN baseline was successfully executed on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. Two seed-0, 15-epoch
characterisation runs were recorded:

| output transform | MSE | Pearson correlation | RMSE | negative prediction fraction |
|---|---:|---:|---:|---:|
| linear | 1887.2247 | 0.0488 | 43.4422 | 0.6284 |
| softplus | 1313.9036 | -0.0703 | 36.2478 | 0.0000 |

The softplus transform removed invalid negative volatility predictions but did
not restore predictive quality. Investigation found that one near-static,
sparsely traded contract generated a numerically converged but implausibly
large GARCH target, which dominated the fused loss. This limitation is recorded
in [`src/baselines/ginn_baseline/LIMITATIONS.md`](../src/baselines/ginn_baseline/LIMITATIONS.md).
The result is retained as a documented GINN limitation rather than used to
justify a broad claim against GARCH or silently modified for the main benchmark.

---

## 3. Proposed Schedule

The schedule is structured as three phases. Phase A is a hard prerequisite. Phases B and C are iterative — the framework and baselines grow in complexity together, with a working end-to-end loop established as early as possible.

---

### Phase A — Data (prerequisite for everything)

- [ ] Write a preprocessing script that runs `build_from_file_list` and saves results to `.npz` (bridging `data_processing.py` → `reader.py`)
- [ ] Verify tensor shapes and data integrity across all three timeframes (1h, 4h, 1d)

**Exit condition:** all three timeframes have clean `.npz` files that load correctly.

---

### Phase B — First working loop (framework milestone)

The goal of this phase is a single end-to-end run: train one neural encoder, train the aggregator on one task, and compare against one baseline — enough to confirm the pipeline works and produce a first reference number.

**Framework side**
- [x] Pretrain one neural encoder on train data: contrastive 4h sweep complete with checkpoint/report artifacts
- [x] Write training/report scripts for the VAE encoder; training checkpoint pending
- [ ] Write `train_framework.py`: frozen encoder → feature bundle → aggregator + task head; train on price prediction task first
- [ ] Write `evaluate.py` with consistent metrics for all models (build this early so every result is comparable from the start)

**Baseline side** (run in parallel once data is ready)
- [x] Train `RawOHLCVMLP` baseline (flattened OHLCV, no representation); 4h price, volatility, and trend artifacts exist under `src/baselines/mlp_baseline/experiments/`
- [ ] Run statistical-only ablation (AR + GARCH features only, no aggregator)

**Exit condition:** framework and at least two baselines produce numbers on the same test split.

---

### Phase C — Iterative expansion (after Phase B)

From here, both sides grow in parallel. Add one method at a time; re-run evaluation after each addition to track whether it helps.

**Expand neural encoders** (order by complexity)
- [x] Train/report contrastive encoder on the 4h split; checkpoint and report complete
- [ ] Evaluate: does adding the contrastive branch improve over VAE-only?
- [ ] Identify additional unsupervised methods from literature (masked autoencoder, self-supervised Transformer, etc.); integrate promising ones one at a time following the same pattern

**Expand external benchmarks** (wire into evaluation harness one at a time)
- [x] Retrain LSTM benchmark on unified `.npz` data splits; record results *(v5 sweep, see `src/baselines/lstm_baseline/experiments/`)*
- [x] Train GINN benchmark on the unified 4h split at 15 epochs; document the GARCH-target failure and defer further GINN sweeps while selecting a more suitable volatility benchmark
- [x] Train TA-MLP benchmark *(v1 triclass sweep, see `src/baselines/ta_mlp_baseline/experiments/2026-06-22-v1/`)*; wire into evaluation harness (trend classification task) — wiring still pending
- [ ] Additional benchmarks from literature (TBD after literature review) — retrain each on same data splits

**Expand internal baselines** (order by complexity)
- [ ] Transformation-only ablation (FFT + Wavelet features only)
- [ ] VAE-only and contrastive-only ablations
- [ ] Standalone GARCH baseline for the volatility task
- [ ] Additional internal baselines (TBD) — add as identified; no fixed list

**Expand tasks**
- [ ] Extend framework training and evaluation to volatility prediction and trend classification tasks
- [ ] Transferability experiment: embed with model trained on one timeframe, evaluate on another
- [ ] *(time permitting)* Decoder-controlled comparison for price prediction: three configurations (benchmark end-to-end / framework + MLP head / framework + benchmark-mirrored head) on the same test split; extend to other tasks if time allows

**Exit condition:** all planned methods (both sides) have been trained and evaluated on all three tasks; ablation table is complete.

**Current Phase C exit gap:** the framework has not yet produced a downstream test result, the unified evaluation harness is absent, and the branch ablations have not been run.

---

### Phase D — Final experiments and report

- [ ] Run full benchmark sweep: all models × all tasks × all metrics
- [ ] Produce result tables, gating weight distributions, embedding visualisations (t-SNE/UMAP)
- [ ] Write final report: motivation, related work, architecture, experiments, discussion, conclusion

---

## 4. Dependency Structure

```
Phase A (data .npz files)
        │
        ▼
Phase B (first end-to-end loop)
  ┌─────┴──────┐
  Framework    Baselines     ← run in parallel
  (VAE + agg)  (MLP + stat-only)
        │
        ▼
Phase C (iterative expansion — both sides grow together)
  add encoders ──┐
  add baselines ─┤  ← interleaved; evaluate after each addition
  add tasks ─────┘
        │
        ▼
Phase D (final experiments + report)
```

There is no fixed ordering within Phase C. Add whichever method is ready next, evaluate immediately, and use the result to inform what to prioritise.
