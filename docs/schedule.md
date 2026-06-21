# FYP Progress and Schedule

**Last updated:** 2026-06-22

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
| VAE architecture (MLP encoder-decoder, β-VAE loss) | 🔄 Implemented, not trained |
| Contrastive encoder (CNN backbone, NT-Xent loss, augmentations) | 🔄 Implemented, not trained |
| **Aggregation and downstream tasks** | |
| `RepresentationAggregator` (N-branch gated fusion, dict API) | 🔄 Implemented, not trained |
| `PriceRegressor` task head (MAE/RMSE) | 🔄 Implemented, not trained |
| `VolatilityRegressor` task head (MSE, correlation) | 🔄 Implemented, not trained |
| `TrendClassifier` task head (accuracy, F1) | 🔄 Implemented, not trained |
| `FeatureBundle` + `NpzFeatureStore` (save/load pipeline) | ✅ Done |
| End-to-end training script | ⬜ Not started |

### Stage 3 — Benchmark and Baseline Implementation and Training

**External benchmarks** (from prior work)
| Task | Status |
|---|---|
| Stacked LSTM benchmark (3-layer, 4h data) | ✅ Trained on unified splits (v5 epoch sweep, seed=0; MAE 0.007–0.012, RMSE 0.016–0.018) |
| GINN benchmark (AR→GARCH→LSTM, volatility) | 🔄 Implemented, not trained |
| TA-MLP benchmark (FreqTrade, trend classification) | ✅ Trained on unified splits (v1 triclass epoch sweep, seed=0; acc 0.71–0.73, macro-F1 0.45–0.47) |
| Additional benchmarks from literature review (TBD) | ⬜ TBD |

**Internal baselines** (designed within this project)
| Task | Status |
|---|---|
| Raw-OHLCV MLP (no representation learning) | 🔄 Implemented, not trained |
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
| Evaluation harness (unified test loop for all models) | ⬜ Not started |
| Price prediction benchmark (MAE, RMSE) | ⬜ Not started |
| Volatility prediction benchmark (MSE, correlation) | ⬜ Not started |
| Trend classification benchmark (accuracy, F1) | ⬜ Not started |
| Transferability analysis (across markets and timeframes) | ⬜ Not started |
| Ablation study (per-branch contribution) | ⬜ Not started |
| Result tables and visualisations | ⬜ Not started |

---

## 2. Summary

The data pipeline and all feature extraction code (statistical, transformation) are complete. The aggregator, all three task heads, and both initial neural encoders (VAE and contrastive) are implemented but not yet trained. All three external benchmarks (LSTM, GINN, TA-MLP) and the Raw-OHLCV MLP internal baseline are implemented. **Two of the three external benchmarks (LSTM, TA-MLP) have now been trained on unified data splits** using a fixed-epoch sweep methodology (no validation split, no early stopping, seed-controlled), with per-run + cross-run summaries recorded under each baseline's `experiments/` directory.

The data `.npz` files now exist for all three timeframes — Phase A is complete. Phase B is partially underway: two benchmarks have first numbers on the 4h test split, but the framework side (train the VAE encoder, write `train_framework.py`, write `evaluate.py`) is not started. Immediate next steps are to train the VAE encoder, produce the first framework number on price prediction, and decide whether to add a constant-predictor floor + multi-seed runs for the baselines before the framework comparison (see `src/baselines/ta_mlp_baseline/IMPROVEMENTS.md`). The exact set of additional neural encoders and benchmarks will be decided incrementally as the literature review progresses.

---

## 3. Proposed Schedule

The schedule is structured as three phases. Phase A is a hard prerequisite. Phases B and C are iterative — the framework and baselines grow in complexity together, with a working end-to-end loop established as early as possible.

---

### Phase A — Data (prerequisite for everything)

- [ ] Write a preprocessing script that runs `build_from_file_list` and saves results to `.npz` (bridging `data_processing.py` → `reader.py`)
- [ ] Verify tensor shapes and data integrity across all three timeframes (1h, 4h, 1d)

**Exit condition:** all three timeframes have clean `.npz` files that load correctly.

---

### Phase B — First working loop (simplest implementations on both sides)

The goal of this phase is a single end-to-end run: train one neural encoder, train the aggregator on one task, and compare against one baseline — enough to confirm the pipeline works and produce a first reference number.

**Framework side**
- [ ] Write training script for the VAE encoder; train and save checkpoint
- [ ] Write `train_framework.py`: frozen encoder → feature bundle → aggregator + task head; train on price prediction task first
- [ ] Write `evaluate.py` with consistent metrics for all models (build this early so every result is comparable from the start)

**Baseline side** (run in parallel once data is ready)
- [ ] Train `RawOHLCVMLP` baseline (flattened OHLCV, no representation)
- [ ] Run statistical-only ablation (AR + GARCH features only, no aggregator)

**Exit condition:** framework and at least two baselines produce numbers on the same test split.

---

### Phase C — Iterative expansion (add complexity incrementally)

From here, both sides grow in parallel. Add one method at a time; re-run evaluation after each addition to track whether it helps.

**Expand neural encoders** (order by complexity)
- [ ] Add contrastive encoder training script; register as second neural branch
- [ ] Evaluate: does adding the contrastive branch improve over VAE-only?
- [ ] Identify additional unsupervised methods from literature (masked autoencoder, self-supervised Transformer, etc.); integrate promising ones one at a time following the same pattern

**Expand external benchmarks** (wire into evaluation harness one at a time)
- [x] Retrain LSTM benchmark on unified `.npz` data splits; record results *(v5 sweep, see `src/baselines/lstm_baseline/experiments/`)*
- [ ] Train GINN benchmark; wire into evaluation harness (volatility task)
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
