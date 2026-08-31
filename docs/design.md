# Model Design

## Architecture Design

The proposed model processes raw OHLCV time-series data through an extensible set of named representation branches: a `statistical` branch (AR and GARCH features), a `transformed` branch (FFT and Haar wavelet features), and neural branches (`vae`, `contrastive`, and `byol`; additional unsupervised methods may be added). These representations are fused by a `RepresentationAggregator` into a unified embedding *h_i*, which is passed to lightweight MLP task heads for three downstream tasks: price prediction, volatility prediction, and trend classification.

The aggregator supports two fusion modes. In *concat mode* (default), branches are concatenated into a single higher-dimensional vector with no learnable parameters; the task head absorbs all supervised learning. In *gated mode*, each branch is projected to a shared dimension and a gating network produces per-branch softmax weights. Concat mode serves as the primary implementation and as an ablation comparison for gated mode. A detailed architecture diagram is provided in the Appendix.

## Experiment Design

The experimental setup is designed to evaluate the effectiveness of the unified representation learning framework on event prediction market data, focusing on two downstream tasks: regression/prediction (probability or return forecasting) and classification (trend direction or event outcome).

### Data Preparation

The dataset consists of OHLCV time-series data from approximately 72,222 event contracts from *Polymarket*, with varying timesteps (1-hour, 4-hour, and 1-day). The data preparation process is designed to produce training-ready sequences for representation learning while preserving temporal order and market-specific dynamics. We use the top 50 most active contracts per timeframe, selected by trading volume.

- **Timestep Separation:** Markets are grouped by their time resolution (1h, 4h, 1d) to handle differing temporal dynamics. Each group is processed independently.

- **Market-level Preprocessing:** For each contract market:
  - Missing values are handled via interpolation or forward-filling to maintain continuous sequences.
  - Features (OHLCV) are normalized or standardized to ensure numerical stability.
  - Sliding window segmentation is applied to generate sequences of fixed length (`seq_len`), preserving chronological order.
  - Minor noise augmentation can be added to improve robustness of learned embeddings.

- **Train-Test Split:** Each contract market's sequences are split chronologically:
  - First 80% of sequences → **training set**
  - Last 20% of sequences → **testing set**

- **Merging Across Markets:** Sequences from all contract markets within the same timestep group are concatenated to form the final training and testing datasets:

  ```
  Training Set = Seqs(contract1, train) + Seqs(contract2, train) + ...
  Testing Set  = Seqs(contract1, test)  + Seqs(contract2, test)  + ...
  ```

- **Final Tensor Shape:** The processed dataset is represented as a tensor of shape `[N, seq_len, features]`, where:
  - `N` = total number of sequences across all markets of the same timestep
  - `seq_len` = sequence length
  - `features` = number of features (OHLCV + optional transformed features)

### Representation Learning

- **Baseline Transformations:**
  - Autoregressive (AR) model to capture linear temporal dependencies. Per column, the fitted AR(*p*) coefficients and residual statistics (mean, std) form the mean-structure representation.
  - GARCH(1,1) model to capture conditional variance dynamics. Fitted via maximum likelihood on the first-differenced series per column, producing a 7-element feature vector: `[ω, α, β, α+β, ω/(1−α−β), mean(σ²), std(σ²)]`, encoding the volatility level, shock sensitivity, persistence, and long-run variance.
  - Wavelet and Fourier transform to extract time-frequency features.

- **Unsupervised Neural Embeddings:**
  - **Variational Autoencoder (VAE)** — MLP encoder-decoder trained with β-VAE loss; encoder frozen after pretraining.
  - **Contrastive Encoder** — CNN backbone with projector head, trained via NT-Xent loss on augmented view pairs (time masking, jittering, scaling).
  - **BYOL Encoder** — CNN online/target encoder with projector and predictor heads, trained by bootstrap prediction on augmented view pairs. The target encoder is updated by exponential moving average; frozen online-backbone embeddings form the downstream branch.
  - **Additional methods (TBD)** — further unsupervised approaches (e.g. masked autoencoders, self-supervised Transformers) may be integrated based on the literature review. Each new method is registered as an independent branch in the aggregator.

### Training Procedure

- All model training — supervised and unsupervised — uses only the training split (80% per contract). The test split is held out until evaluation under the predeclared model × task × epoch-budget matrix.

- **Train and test only — no validation split, no early stopping.** Every model uses a fixed epoch budget (`--epochs N`); for external benchmarks a small characterization sweep across epoch budgets is run at one fixed seed, and the full sweep is reported rather than a best-on-test entry. See `docs/training_test_data_selection.md` for the rationale and the full set of rules.

- Neural encoders (VAE, contrastive, BYOL, and any additional methods) are pretrained unsupervised on training sequences only, then their weights are frozen.

- Frozen encoders are used to extract neural embeddings for both training and test sequences. Running inference through a frozen encoder on test data is not leakage — the encoder parameters contain no information derived from test sequences.

- The aggregator and task heads are trained on training feature bundles (statistical + transformed + separately named frozen neural branches) for fixed epoch budgets; each predeclared checkpoint receives one test pass and the complete epoch-budget matrix is reported without selecting a best-on-test run.

- Training is conducted separately for each timestep group (1-hour, 4-hour, 1-day) to account for differing temporal dynamics.

- All baseline models use the identical train/test partitions as the framework. See `docs/training_test_data_selection.md` for the complete data allocation rules.

### Evaluation Process

The evaluation is designed to assess both the **effectiveness** and **transferability** of the learned embeddings compared to baseline approaches. The process is structured as follows.

**Terminology:**

| Term | Definition |
|---|---|
| **External benchmark** | Model from prior work or a predeclared paper-inspired adaptation (end-to-end or task-specific). Current set: Stacked LSTM, Raw LSTM volatility, adapted GARCH--LSTM stacking, GINN limitation evidence, TA-MLP. |
| **Internal baseline** | Model designed within this project (Raw-OHLCV MLP, single-branch ablations). Shows each framework component contributes. |
| **Default decoder** | Task head (`PriceRegressor`, `VolatilityRegressor`, `TrendClassifier`) — simple MLP from `src/tasks/`. Used by the framework and all internal baselines. |
| **Mirrored decoder** | Benchmark's own FC architecture retrained on frozen framework embeddings. Used only in the decoder-controlled comparison experiment. |

**Evaluation paradigm (probing):** The framework uses frozen representation extractors. After pretraining, the named branch features remain fixed while a lightweight task head, and the aggregator when it is learnable, are trained for each task. Keeping the task head simple is intentional — if the representations are powerful, the decoder should not need to be complex. Any benchmark comparison is against an end-to-end trained model, which has more optimisation freedom; matching or beating it with frozen representations + a simple head is the primary claim.

- **Benchmark Retraining:** Each benchmark model is retrained on the same event prediction market dataset, using the same sliding window sequences, train/test split, and temporal ordering. This project does not use a validation split or early stopping; see `docs/training_test_data_selection.md`.

- **Volatility benchmark adaptation:** The strict volatility comparison uses a shared realised-volatility label bundle. Raw LSTM volatility is the direct end-to-end neural benchmark. The adapted GARCH--LSTM stack is a complementary, stronger hybrid benchmark: it fuses causal guarded GARCH forecasts with the same Raw LSTM forecasts through fixed ElasticNet meta-features `[g, l, g*l]`. Its expanding cross-fitting is used only to create out-of-fold training features for the meta-learner; it is not validation or model selection. The existing Raw-OHLCV MLP volatility sweep uses a legacy merged-array target helper and is characterization-only until it is migrated to this shared bundle; the framework volatility task must also consume this bundle before any strict comparison. A framework result should therefore report its relationship to both benchmarks rather than treating the stack as evidence that standalone GARCH is superior.

- **Embedding-based Model Training:**
  - Deterministic branches (statistical, transformed) require no training; neural branches are pretrained unsupervised and their encoder weights are frozen.
  - All branch embeddings are extracted into a branch-aware `FeatureBundle`: deterministic arrays are saved as `statistical` and `transformed`, and neural embeddings are saved under their encoder names such as `vae`, `contrastive`, and `byol`. The `RepresentationAggregator` receives these named branch tensors and fuses them into a unified embedding *h_i* per sequence.
  - **Downstream Task Preparation:**
    - **Regression Task:** supervised pairs (X, y), where X is the sequence embedding and y is the target return or probability at a future timestep.
    - **Classification Task:** labels such as trend direction or event outcome mapped to embeddings as input-output pairs. The current trend-classification MVP uses TA-MLP-style tri-class BUY/HOLD/SELL labels, with thresholds fit from training data only and saved as a reusable task label bundle.
  - A lightweight MLP task head is trained on these (X, y) pairs.

- **Performance Comparison:**
  - Evaluate strict task comparisons on the same held-out test sequences and aligned label rows.
  - Consistent metrics: Regression → MAE, RMSE; Classification → Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix.
  - Comparison axes:
    - Benchmarks (end-to-end, task-specific) vs. framework (frozen encoder + MLP head)
    - Single-branch ablations vs. full aggregated framework
    - Transferability: embeddings trained on one timeframe evaluated on another without retraining

- **Decoder-Controlled Comparison (optional, time permitting, all three tasks):**

  To isolate encoder quality from decoder choice, three configurations are compared per task, holding the decoder architecture constant:

  | Configuration | Encoder | Decoder | Trained |
  |---|---|---|---|
  | Benchmark end-to-end (e.g. LSTM) | Task-specific, end-to-end | Benchmark's FC head | End-to-end |
  | Framework + default decoder | Multi-branch concat (frozen) | Task head (simple MLP) | Head only |
  | Framework + mirrored decoder | Multi-branch concat (frozen) | Benchmark FC architecture (retrained) | Head only |

  Configurations 1 vs 3 isolate the encoder (same decoder architecture); configurations 2 vs 3 isolate the decoder (same encoder). Applies to all three tasks (price prediction, volatility prediction, trend classification), subject to availability of a separable benchmark decoder per task.

### Additional Alpha-Research Capability

Alpha research is a future downstream capability test outside the current task-evaluation budget, not an additional representation branch or the framework's central contribution. The task heads transform the shared representation \(z\) into economically named predictions; these predictions, rather than arbitrary latent coordinates \(z_j\), are the terminals for a deliberately small symbolic search:

\[
X \rightarrow z \rightarrow \{\hat r,\hat\sigma,p_{bull},p_{bear},\ldots\}
\rightarrow \mathcal F_0 \rightarrow \{\alpha_1,\ldots,\alpha_K\}.
\]

The initial primitive set contains predicted return/price movement, volatility, trend probabilities, and confidence margins such as \(p_{bull}-p_{bear}\). A shallow GP grammar may compose them with protected arithmetic, ranking, delay, and bounded rolling operators. Formula selection must use chronological out-of-fold predictions on the training portion only, with target-horizon purge/embargo where needed. Once the current test split has been used for task evaluation, future factor evaluation requires a fresh holdout or temporally later data. Evaluate factor IC/rank-IC, stability, quantile spreads, and redundancy; do not claim tradable profitability without a cost-aware, contract-aware backtest.
