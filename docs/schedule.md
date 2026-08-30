# FYP Progress and Schedule

**Last updated:** 2026-08-30

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
| VAE architecture (MLP encoder-decoder, β-VAE loss) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| Contrastive encoder (CNN backbone, NT-Xent loss, augmentations) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| BYOL encoder (CNN online/target encoder, EMA target update) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| **Aggregation and downstream tasks** | |
| `RepresentationAggregator` (N-branch gated fusion, dict API) | ✅ Implemented; concat mode used in price and trend MVP framework runs |
| `PriceRegressor` task head (MAE/RMSE) | ✅ Implemented and evaluated in 4h framework MVP |
| `VolatilityRegressor` task head (MSE, correlation) | 🔄 Implemented, framework task run pending |
| `TrendClassifier` task head (accuracy, macro-F1) | ✅ Implemented and evaluated in 4h tri-class framework MVP |
| Branch-aware `FeatureBundle` + `NpzFeatureStore` (save/load pipeline) | ✅ Done; legacy packed `neural` stores remain loadable |
| End-to-end training script | ✅ `scripts/train_framework.py` supports price prediction and trend classification; volatility run pending |

### Stage 3 — Benchmark and Baseline Implementation and Training

**External benchmarks** (from prior work)
| Task | Status |
|---|---|
| Stacked LSTM benchmark (3-layer, 4h data) | ✅ Trained on unified splits (v5 epoch sweep, seed=0; MAE 0.007–0.012, RMSE 0.016–0.018) |
| Raw LSTM volatility benchmark | ✅ Trained on shared 4h realised-volatility label bundle at 15/50/100 epochs |
| Adapted GARCH--LSTM stacking volatility benchmark | 🔄 Implemented with unit-tested GARCH, expanding OOF fold assignment, fixed ElasticNet replay, runner, and plots; canonical stack training pending |
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
| BYOL-only ablation | ⬜ Not started |
| Additional neural encoder ablations (per TBD methods) | ⬜ TBD |
| Additional internal baselines (TBD) | ⬜ TBD |

### Stage 4 — Experiments and Benchmarking
| Task | Status |
|---|---|
| Evaluation harness (unified test loop for all models) | 🔄 Framework runner records configs, manifests, metrics, predictions, summaries, and comparisons; baseline runners still evaluate independently |
| Price prediction benchmark (MAE, RMSE) | ✅ MVP framework, Raw-OHLCV MLP, and LSTM results recorded on locked 4h test split; stricter shared-target LSTM alignment remains future work |
| Volatility prediction benchmark (MSE, correlation) | 🔄 Raw-OHLCV MLP, Raw LSTM volatility, and GINN limitation results recorded; GARCH--LSTM stack implemented but canonical stack run and framework comparison pending |
| Trend classification benchmark (accuracy, macro-F1) | ✅ MVP framework result recorded on locked 4h test split; Raw-OHLCV MLP and TA-MLP context available, strict TA-MLP label-bundle alignment pending |
| Transferability analysis (across markets and timeframes) | ⬜ Not started |
| Ablation study (per-branch contribution) | ⬜ Not started |
| Result tables and visualisations | 🔄 Framework price/trend summaries and comparisons generated; final cross-model tables, branch ablations, and embedding visualisations pending |

---

## 2. Summary

The data pipeline and all three processed timeframes are complete. The 4h framework feature bundle has been regenerated and validated at the current dimensions: `statistical` (`137341 x 70`), `transformed` (`137341 x 55`), `vae` (`137341 x 64`), and `contrastive` (`137341 x 128`), with `109841` train rows and `27500` test rows recorded in the companion index/manifest. The VAE, contrastive, and BYOL encoders have completed fixed-budget 4h CUDA pretraining sweeps and provide the canonical checkpoints `checkpoints/vae_4h_seq64_top50.pth`, `checkpoints/contrastive_4h_seq64_top50.pth`, and `checkpoints/byol_4h_seq64_top50.pth`. BYOL downstream feature extraction remains pending, so the current feature store does not yet contain the `byol` branch.

The current project state is **Phase C — iterative expansion after the first working framework loop**. Phase B is achieved for price prediction: the framework now trains `RepresentationAggregator(mode="concat") + PriceRegressor` on frozen statistical, transformed, VAE, and contrastive features and evaluates on the locked 4h test split. The price framework run under `experiments/framework/price_prediction/4h_stat_transform_vae_contrastive_concat/` reports MAE/RMSE of `0.0575/0.0955` at 15 epochs, `0.0695/0.1046` at 50 epochs, and `0.0720/0.1059` at 100 epochs. These are directly comparable to the Raw-OHLCV MLP sweep on the same processed split (`0.0834/0.1067`, `0.0600/0.0805`, `0.0454/0.0683` at 15/50/100 epochs). The LSTM benchmark uses the held-out test side, but it rebuilds close-only windows from raw feather inputs and currently has a one-row target alignment difference, so it should be treated as external context until re-wired to the exact shared target builder.

Trend classification has also reached an MVP framework result. The TA-MLP-style tri-class BUY/HOLD/SELL label bundle is saved under `data/task_labels/trend_classification/triclass_4h_seq64_top50.npz`, with thresholds fit from training data only and final horizon rows dropped per split. The framework trend run under `experiments/framework/trend_classification/4h_triclass_stat_transform_vae_contrastive_concat/` reports accuracy/macro-F1 of `0.4730/0.3875` at 15 epochs, `0.4953/0.4164` at 50 epochs, and `0.5071/0.4232` at 100 epochs. The majority-HOLD reference on the same label bundle is `0.5046` accuracy and `0.2236` macro-F1. The existing TA-MLP sweep remains useful context (`0.71-0.73` accuracy, `0.45-0.47` macro-F1), but strict comparison requires reusing the saved framework label bundle and identical row alignment.

The next priority is to finish the remaining MVP surface before deeper claims: extract the frozen BYOL embeddings into the branch-aware feature store, add the volatility framework task run, run the canonical GARCH--LSTM stack, run single-branch ablations, align any remaining external baselines to shared task targets where needed, and then produce final comparison tables. Current results support the implementation claim that the frozen multi-branch features contain useful downstream information; they do not yet support a superiority claim over task-specific baselines without ablations, stricter baseline alignment, and multi-seed confirmation.

### Recent VAE encoder progress

The VAE encoder was successfully trained on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. The run used only the locked
training split (`109841` sequences) and recorded the test split shape (`27500`
sequences) for traceability only. No test sequences were used for training,
early stopping, or checkpoint selection.

| epoch budget | total loss | reconstruction MSE | KL divergence | best train loss so far | elapsed seconds |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.0999 | 0.0700 | 0.0300 | 0.0809 | 19.96 |
| 20 | 0.0866 | 0.0669 | 0.0196 | 0.0809 | 26.10 |
| 25 | 0.0841 | 0.0659 | 0.0182 | 0.0809 | 32.58 |
| 50 | 0.0828 | 0.0647 | 0.0180 | 0.0809 | 64.44 |
| 100 | 0.0802 | 0.0626 | 0.0176 | 0.0800 | 125.33 |

The run recovered from a transient early instability at epochs 13-14 and then
improved gradually through the 100-epoch budget. The final checkpoint is close
to the best observed training point (`0.0800388432` at epoch 98), so it is a
reasonable current VAE encoder candidate. As with contrastive pretraining, this
is unsupervised pretraining evidence only; the checkpoint now feeds the MVP
price and trend probing runs, while branch-specific contribution still needs
ablation.

### Recent contrastive encoder progress

The contrastive encoder was successfully trained on the NVIDIA GPU through WSL using the unified 4h, sequence-length-64 dataset. The run used only the locked training split (`109841` sequences) and recorded the test split shape (`27500` sequences) for traceability only. No test sequences were used for training, early stopping, or checkpoint selection.

| epoch budget | train NT-Xent loss | best train loss so far | elapsed seconds |
|---:|---:|---:|---:|
| 15 | 2.5918 | 2.5918 | 214.70 |
| 20 | 2.5496 | 2.5496 | 287.62 |
| 25 | 2.5225 | 2.5225 | 364.87 |
| 50 | 2.4600 | 2.4599 | 727.67 |
| 100 | 2.4100 | 2.4085 | 1460.38 |

The loss decreased consistently and plateaued gradually, so the result is meaningful as unsupervised pretraining evidence. This checkpoint now feeds the MVP frozen-embedding probing runs for price prediction and trend classification; branch-specific contribution still needs ablation.

### Recent BYOL encoder progress

The BYOL encoder was successfully trained on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. The run used only the locked
training split (`109841` sequences) and recorded the test split shape (`27500`
sequences) for traceability only. No test sequences were used for training,
early stopping, or checkpoint selection.

| epoch budget | train BYOL loss | view cosine | embedding std | collapse warning | elapsed seconds |
|---:|---:|---:|---:|:---:|---:|
| 15 | 0.0755 | 0.9623 | 0.6185 | false | 191.52 |
| 20 | 0.0810 | 0.9595 | 0.7010 | false | 250.28 |
| 25 | 0.0797 | 0.9602 | 0.7765 | false | 315.87 |
| 50 | 0.0776 | 0.9612 | 0.9908 | false | 635.14 |
| 100 | 0.0526 | 0.9737 | 1.3357 | false | 1283.12 |

The loss reached an early minimum before rising as the exponential-moving-average
target and representation scale evolved, then declined through the final budget.
Because the BYOL target changes during training, the early minimum is recorded as
a diagnostic rather than used for checkpoint selection. Embedding standard
deviation increased from `0.1457` after epoch 1 to `1.3357` after epoch 100,
remaining well above the configured collapse threshold (`0.001`). The full raw
histories, checkpoint metrics, plots, and report are stored under
`experiments/byol_encoder/byol-4h-seq64-top50/`; the canonical checkpoint is
`checkpoints/byol_4h_seq64_top50.pth`. Downstream BYOL feature extraction and
branch ablation remain pending.

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

The schedule is structured as four phases. Phase A is a hard prerequisite. Phases B and C are iterative — the framework and baselines grow in complexity together, with a working end-to-end loop established as early as possible.

---

### Phase A — Data (prerequisite for everything)

- [x] Write a preprocessing script that runs `build_from_file_list` and saves results to `.npz` (bridging `data_processing.py` → `reader.py`)
- [x] Verify tensor shapes and data integrity across all three timeframes (1h, 4h, 1d)

**Exit condition:** all three timeframes have clean `.npz` files that load correctly.

---

### Phase B — First working loop (framework milestone)

The goal of this phase is a single end-to-end run: train one neural encoder, train the aggregator on one task, and compare against one baseline — enough to confirm the pipeline works and produce a first reference number.

**Framework side**
- [x] Pretrain one neural encoder on train data: contrastive 4h sweep complete with checkpoint/report artifacts
- [x] Train/report VAE encoder on train data: 4h sweep complete with checkpoint/report artifacts
- [x] Write `train_framework.py`: frozen encoder feature bundle → aggregator + task head; price and trend task runs complete
- [ ] Write a unified cross-model `evaluate.py`/comparison harness for all framework and baseline artifacts

**Baseline side** (run in parallel once data is ready)
- [x] Train `RawOHLCVMLP` baseline (flattened OHLCV, no representation); 4h price, volatility, and trend artifacts exist under `src/baselines/mlp_baseline/experiments/`
- [ ] Run statistical-only ablation (AR + GARCH features only, no aggregator)

**Exit condition:** framework and at least two baselines produce numbers on the same test split. **Status:** achieved for the first price-prediction loop; trend MVP is also implemented, while strict external-baseline row alignment remains pending.

---

### Phase C — Iterative expansion (after Phase B)

From here, both sides grow in parallel. Add one method at a time; re-run evaluation after each addition to track whether it helps.

**Expand neural encoders** (order by complexity)
- [x] Train/report contrastive encoder on the 4h split; checkpoint and report complete
- [ ] Evaluate: does adding the contrastive branch improve over VAE-only? *(full VAE+contrastive concat result exists; isolated branch ablation pending)*
- [x] Implement, train, and report the BYOL encoder on the 4h split; checkpoint and diagnostic artifacts complete
- [ ] Evaluate: does adding the BYOL branch improve over the current VAE + contrastive branch set?
- [ ] Identify additional unsupervised methods from literature (masked autoencoder, self-supervised Transformer, etc.); integrate promising ones one at a time following the same pattern

**Expand external benchmarks** (wire into evaluation harness one at a time)
- [x] Retrain LSTM benchmark on unified `.npz` data splits; record results *(v5 sweep, see `src/baselines/lstm_baseline/experiments/`)*
- [x] Train Raw LSTM volatility benchmark on the shared realised-volatility label bundle; record matched 15/50/100 epoch artifacts
- [ ] Run the adapted GARCH--LSTM stacking volatility benchmark using Raw LSTM predictions and fixed ElasticNet meta-learning
- [x] Train GINN benchmark on the unified 4h split at 15 epochs; document the GARCH-target failure and defer further GINN sweeps while selecting a more suitable volatility benchmark
- [x] Train TA-MLP benchmark *(v1 triclass sweep, see `src/baselines/ta_mlp_baseline/experiments/2026-06-22-v1/`)*; strict comparison should reuse the saved framework tri-class label bundle
- [ ] Additional benchmarks from literature (TBD after literature review) — retrain each on same data splits

**Expand internal baselines** (order by complexity)
- [ ] Transformation-only ablation (FFT + Wavelet features only)
- [ ] VAE-only, contrastive-only, and BYOL-only ablations
- [ ] Additional internal baselines (TBD) — add as identified; no fixed list

**Expand tasks**
- [x] Extend framework training and evaluation to trend classification
- [ ] Extend framework training and evaluation to volatility prediction
- [ ] Transferability experiment: embed with model trained on one timeframe, evaluate on another
- [ ] *(time permitting)* Decoder-controlled comparison for price prediction: three configurations (benchmark end-to-end / framework + MLP head / framework + benchmark-mirrored head) on the same test split; extend to other tasks if time allows

**Exit condition:** all planned methods (both sides) have been trained and evaluated on all three tasks; ablation table is complete.

**Current Phase C exit gap:** price and trend framework MVP runs exist, but volatility is still missing, BYOL has not yet been extracted into the full feature store, external baseline alignment needs tightening, and branch ablations/multi-seed confirmation have not been run.

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
  Framework             Baselines     ← run in parallel
  (VAE/contrastive+agg) (MLP + stat-only)
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
