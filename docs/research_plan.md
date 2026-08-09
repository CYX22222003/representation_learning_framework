# Research Plan

This document outlines the four-stage research plan for developing and evaluating the unified representation learning framework.

**Progress tracking note:** This document is the stable research roadmap. It should change when the project direction, planned stages, comparison scope, task definitions, or evaluation methodology changes. Routine implementation status, checkpoint evidence, experiment observations, and next-action tracking belong in `docs/schedule.md` and the corresponding experiment reports.

---

## Stage 1 — Data Collection and Processing

- Collect historical OHLCV data from *Polymarket* event prediction markets.
  - Select top-*K* most active contracts per timeframe (1-hour, 4-hour, 1-day) by trading volume.

- Clean and standardise each contract's time series:
  - Handle missing timestamps and volume gaps via forward-fill and interpolation.
  - Z-score normalise volume; retain raw OHLC prices (bounded to [0, 1] probability scale).

- Segment into fixed-length sequences using a sliding window:
  - Apply sliding window of length `seq_len` to produce sequences of shape `[seq_len, features]`.
  - Split each contract chronologically: first 80% for training, last 20% for testing.
  - Concatenate sequences across all selected contracts to form the final training and test sets.

- Perform exploratory analysis to understand distribution, volatility regimes, and event-driven price jumps.

---

## Stage 2 — Framework Implementation and Training

This stage implements and trains all components of the proposed unified representation learning framework.

### 2.1 Statistical Feature Extraction

- Autoregressive (AR) model — fit AR(*p*) per OHLCV column via least squares; output coefficients and residual statistics (mean, std).
- GARCH(1,1) model — fit via MLE on first-differenced series per column; extract 7 features: ω, α, β, persistence (α+β), unconditional variance, mean and std of conditional variance.

### 2.2 Transformation-based Feature Extraction

- Fourier Transform (FFT) — top-*k* magnitude coefficients per column.
- Haar Wavelet — multi-level detail energy per column.

### 2.3 Neural Encoder Pretraining (Unsupervised)

The neural branch is designed to accommodate multiple unsupervised learning methods. Each method is implemented as a separate encoder and registered as its own named branch in the aggregator. The two methods below are the initial set; additional methods may be integrated as the literature review progresses.

- **Variational Autoencoder (VAE)**:
  - MLP encoder-decoder architecture trained with β-VAE reconstruction + KL-divergence loss.
  - Pretrain on training sequences; freeze encoder weights for downstream use.

- **Contrastive Encoder**:
  - CNN backbone with projector head; trained via NT-Xent loss on augmented view pairs (jitter, scaling, time masking).
  - Pretrain on training sequences; freeze encoder weights for downstream use.

- **Additional methods (TBD)** — candidates include masked autoencoders, self-supervised Transformer encoders, or other self-supervised objectives identified during the literature review. Each new encoder registers a new key in the aggregator's `branch_dims` without requiring any changes to existing components.

### 2.4 Representation Aggregation and Downstream Task Training

- Implement `RepresentationAggregator` — a flexible N-branch fusion module that accepts an arbitrary set of named branches via a `dict[str, Tensor]` API.
  - **Concat mode** (default): branches are concatenated into a single higher-dimensional vector; no learnable parameters in the aggregator itself. Output dimension equals the sum of all branch dimensions.
  - **Gated mode**: each branch is projected to a shared `out_dim`, then a gating network produces per-branch softmax weights. Output dimension equals `out_dim`.
  - The `output_dim` property returns the correct task-head input size regardless of mode.
  - Default branch set: `statistical` (70-d), `transformed` (55-d), `vae` (64-d), `contrastive` (128-d). Adding a new neural encoder requires only registering a new key in `branch_dims`.
  - Concat serves as the primary implementation and as an ablation baseline for gated mode.

- Train the aggregator jointly with each downstream task head using supervised task losses:
  - Price prediction — MLP regressor, MSE/MAE loss.
  - Volatility prediction — MLP regressor, MSE loss on realised volatility targets.
  - Trend classification — MLP classifier, binary cross-entropy loss.

- Write end-to-end training scripts connecting data loading, feature extraction, encoder inference, aggregation, and task training.

---

## Stage 3 — Baseline and Benchmark Implementation

All comparison models must be trained on the **same data splits and preprocessing** as the framework to ensure fair comparison. The exact set of models is provisional and will be finalised once the literature review is complete.

Two categories of comparison models are used:

| Term | Definition |
|---|---|
| **External benchmark** | Model from prior work (task-specific, end-to-end trained). Shows the framework is competitive with the state-of-the-art. |
| **Internal baseline** | Model designed within this project. Shows each framework component contributes. |

**External benchmarks:**

- **Stacked LSTM** — 3-layer LSTM trained directly on raw OHLCV sequences as the primary external benchmark for price prediction.
- **GINN** *(AR→GARCH→LSTM with fused loss)* — three-stage hybrid: AR mean prediction, GARCH(1,1) volatility estimation, LSTM variance predictor trained with GARCH-fused loss. Primary benchmark for the volatility prediction task.
- **TA-MLP** *(FreqTrade-based)* — 4-layer LeakyReLU MLP trained on 36 TA-Lib technical indicator features (RSI, Bollinger Bands, candlestick patterns, etc.). Primary benchmark for the trend classification task. Labels follow the upstream paper's tri-class BUY/HOLD/SELL formulation (`src/baselines/ta_mlp_baseline/ta_labels.py`); thresholds are quantiles of `|pct_change|` fit per contract on training rows only.
- **Additional benchmarks (TBD)** — further models may be added based on the literature review.

**Internal baselines:**

- **Raw-OHLCV MLP** — 5-layer MLP trained directly on flattened OHLCV sequences with no representation learning; serves as the minimum competence reference.

- **Single-branch ablations** — run each active representation branch independently (no aggregation) through the same task heads. Will include at minimum:
  - Statistical-only (AR + GARCH features)
  - Transformation-only (FFT + Wavelet features)
  - VAE-only (latent embeddings from the pretrained VAE)
  - Contrastive-only (embeddings from the pretrained contrastive encoder)
  - One ablation per additional neural encoder that is integrated (TBD)

  These ablations isolate each branch's individual contribution and verify that the aggregated framework outperforms any single branch.

- **Standalone GARCH** — classical GARCH(1,1) used as a dedicated baseline for the volatility prediction task.

- **Additional internal baselines (TBD)** — further baselines may be added as identified.

---

## Stage 4 — Experiments and Benchmarking

The framework is evaluated using **probing**: frozen multi-branch encoders + a lightweight MLP task head trained on extracted features. Keeping the task head simple is intentional — representation quality, not decoder complexity, should drive performance.

- Evaluate all models (framework, benchmarks, internal baselines) on the held-out test splits using consistent metrics:
  - Price prediction: MAE, RMSE
  - Volatility prediction: MSE, Pearson correlation of predicted vs. realised volatility
  - Trend classification: Accuracy, F1-score

- **Decoder-controlled comparison (optional, time permitting, all three tasks):** three configurations run on the same test split to isolate encoder quality from decoder choice. Two decoder types are used: the **default decoder** (task head — simple MLP from `src/tasks/`) and the **mirrored decoder** (benchmark's own FC architecture, detached and retrained on frozen framework embeddings).

  | Configuration | Encoder | Decoder | Trained |
  |---|---|---|---|
  | Benchmark end-to-end | Task-specific, end-to-end | Benchmark's FC head | End-to-end |
  | Framework + default decoder | Multi-branch concat (frozen) | Task head (simple MLP) | Head only |
  | Framework + mirrored decoder | Multi-branch concat (frozen) | Benchmark FC architecture (retrained) | Head only |

  Configurations 1 vs 3 isolate the encoder (same decoder architecture); configurations 2 vs 3 isolate the decoder (same encoder). Applies to all three tasks, subject to availability of a separable benchmark decoder per task.

- **Transferability analysis** — evaluate whether embeddings trained on one subset of tasks or markets transfer effectively to held-out tasks, contract types, or timeframes without retraining.

- **Ablation study** — compare the full aggregated framework against each single-branch baseline to quantify each branch's marginal contribution.

- Summarise all results in tables and visualisations (embedding scatter plots, metric comparisons, gating weight distributions).

---

## Final Step — Documentation and Reporting

- Write final report covering: motivation, related work, model design, experimental results, analysis of limitations, and future directions.
- Prepare codebase documentation and ensure reproducibility.
