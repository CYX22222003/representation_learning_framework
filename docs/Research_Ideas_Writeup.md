# Research Idea V2

## 1. Main Topic

A unified representation learning framework for time-series data that integrates statistical, transformation-based, and deep learning features for transferable multi-task applications

Evaluate this framework on cryptocurrency market data across tasks such as price prediction, volatility forecasting, and alpha factor discovery tasks.

## 2. Literature Review (Brief \& Informal)

### 2.1 Current Research on Deep Learning in Cryptomarket

Existing research on cryptocurrency time-series modeling can be broadly categorized into two main directions: (1) handcrafted feature-based approaches and (2) task-specific deep learning or hybrid models. While these approaches have demonstrated decent performance and effectiveness in specific settings, they exhibit limitations in representation learning and transferability across tasks.

#### 2.1.1 Handcrafted features

A large portion of existing work relies on handcrafted features derived from technical indicators and domain-specific heuristics, without incorporating learned representations from raw data. For example:

Parente et al. (2024) build a 3-class crypto price trend classification MLP using 36 handcrafted inputs, exclusively technical indicators (Bollinger Bands, RSI, EMA crossovers) and basic temporal features (day of week, hour). Their SHAP analysis confirms the model relies entirely on these predefined technical rules, with handcrafted candlestick patterns showing near-zero predictive value.

Barnwal et al. (2019) construct a stacked ensemble model using 40+ handcrafted technical indicators across trend, momentum, volume, and volatility categories, plus basic Twitter sentiment features, with no automated feature learning component.

Anguiano & García-Medina (2025) evaluates classical technical trading strategies (EMA crossover, MACD+ADX), all of which use handcrafted technical indicator inputs with no learned market embeddings.

**Core limitations:**

- While handcrafted features offer interpretability and are grounded in financial intuition, they are inherently constrained by predefined assumptions about market behavior.

- They may struggle to capture high-frequency dynamics and complex interactions present in cryptocurrency markets.

- Emergent patterns that are not explicitly encoded in technical indicators are difficult to represent.

- Generalization across different assets, timeframes, or market regimes often requires manual feature redesign.

#### 2.1.2 Stacked/hybird models: 

More recent approaches adopt deep learning or hybrid architectures (e.g., CNN, LSTM, Transformer, or ensemble models) to automatically extract temporal features from data.

Mahdi et al. (2025) propose a hybrid Transformer+GRU model purpose-built for single-step BTC/ETH price regression, with the Transformer’s attention and GRU’s sequence learning tightly coupled to the price prediction loss function; there is no mechanism to reuse the learned temporal features for other tasks (e.g., volatility forecasting) without full model retraining.

Gautam (2025) design a two-stage LSTM+XGBoost hybrid, where the LSTM extracts temporal features exclusively for price direction classification, with no transferability to other trading-related tasks.

Halder (2022)’s FinBERT-LSTM model integrates news sentiment and price data, but is built exclusively for price level prediction, with sentiment features integrated only for this narrow use case.

Hu et al. (2024) build an Autoencoder-CNN-GAN pipeline optimized solely for predicting extreme BTC price movements, with the denoising and feature extraction components locked to the single task of large price swing detection.

**Core limitations:**

- These models are typically optimized for a single predefined task (e.g., price prediction or trend classification), resulting in representations that are tightly coupled to specific objectives.

- As a consequence, the learned features are not directly transferable to other related tasks, such as volatility forecasting or alpha factor discovery.

- Adapting these models to new tasks or changing market conditions often requires full retraining, leading to increased computational cost and reduced flexibility.

- While these methods improve predictive performance, they do not explicitly aim to learn reusable or general-purpose representations of market dynamics.

#### 2.1.3 Conclusion

This highlights a broader limitation in current research: the focus remains on task-specific modeling rather than learning transferable representations that can generalize across multiple financial tasks.

### 2.2 Current Research on Time-Series Embedding

Time-series embedding has emerged as an effective paradigm for transforming raw temporal sequences into compact, information-rich representations that preserve key temporal dynamics for downstream tasks. Below are some approaches in time series representation learning

#### 2.2.1 model-based statistical approach

Early approaches to time-series embedding are rooted in statistical modeling, where model parameters or latent states serve as compact representations of temporal structure. These methods excel at capturing linear dependencies, stationary trends, and interpretable statistical properties of time series, serving as foundational benchmarks for temporal representation learning. 

Autoregressive (AR) and ARIMA models are traditional examples, where the fitted coefficients encode the linear temporal dependency structure of the series (Irani et al., 2025). Complementing these, the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model captures the volatility dynamics of a time series. The GARCH(1,1) model defines the conditional variance as:

σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

where ω is the baseline variance, α (ARCH coefficient) quantifies sensitivity to past shocks, and β (GARCH coefficient) measures variance persistence. The parameters are estimated via maximum likelihood. The persistence α+β indicates how long volatility shocks endure, and the unconditional variance ω/(1−α−β) reflects the long-run variance level. For financial time series such as event prediction markets, GARCH features provide a compact, interpretable representation of the volatility regime that complements the mean-structure captured by AR models. More complex statistical methods like Hidden Markov Models (HMMs) capture the probabilistic transitions between different latent states in the series (Irani et al., 2025).

#### 2.2.2 Transformation-based approach

Transformation-based methods further demonstrate that meaningful representations can be obtained by projecting time series into alternative domains. These methods excel at capturing periodic patterns, spectral signatures, and multi-scale transient features of time series, with strong interpretability and computational efficiency for stationary/non-stationary temporal data. 

The Fourier Transform decomposes a series into its constituent frequencies, making it suitable for analyzing periodic components (Sneddon, 1995). However, it assumes stationarity, limiting its effectiveness for non-stationary data. The Wavelet transformation (Irani et al., 2025) is introduced as a more versatile alternative, capturing both time and frequency information, making it more suitable for analyzing non-stationary and transient signals.

#### 2.2.3 Unsupervised Deep learning approach

More recently, unsupervised deep learning has significantly advanced time-series embedding by enabling the learning of flexible, high-dimensional representations directly from data without manual feature design. Deep neural networks have been used to learn hierarchical, nonlinear representations from unlabeled time series via unsupervised tasks (reconstruction, contrastive learning, masked prediction). These methods excel at capturing complex long-range dependencies, nonlinear dynamics, and multi-scale patterns in time series, and support universal representation learning across diverse downstream tasks. I want to highlight in the following three paradigms

---

**Autoencoder-based methods**

Autoencoder-based methods learn embeddings via an encoder-decoder architecture, where the encoder maps raw time series to latent embeddings, and the decoder reconstructs the original input from the embeddings. The core idea is that valid embeddings should preserve sufficient information to reconstruct the original temporal sequence.

**Contrastive Learning-based Embeddings**

Contrastive learning is the most popular unsupervised time-series embedding paradigm in recent years. It learns embeddings by maximizing the similarity between positive sample pairs (different views of the same sequence) and minimizing the similarity between negative sample pairs (different sequences), without relying on reconstruction tasks.

**Transformer-based Embeddings**

Transformer-based methods use self-attention mechanisms to model long-range dependencies in time series, and learn universal temporal embeddings via unsupervised pre-training tasks (masked prediction, contrastive learning). 

---

#### 2.2.4 Conclusion

Together, these approaches demonstrate that time-series embedding is a well-established and versatile paradigm, capable of capturing temporal dynamics from multiple perspectives. This provides a strong foundation for constructing richer representations by combining complementary embedding strategies.

### 2.3 Crypto Maket Characteristics

Cryptocurrency markets exhibit unique structural properties that distinguish them from traditional financial markets, making them a particularly suitable and challenging domain for representation learning.

1. Extreme volatility

Cryptocurrency markets are characterized by high volatility and frequent regime shifts, driven by speculative trading, macroeconomic events, regulatory changes, and market sentiment. Models trained for a specific regime or task may quickly become outdated, highlighting the need for representations that can generalize across different market conditions.

2. Continuous 24/7 trading and high-frequency dynamics

Unlike traditional financial markets, cryptocurrency markets operate continuously without trading interruptions. It has a rich intraday and high frequency patterns and a large volume of streaming data. This creates complex temporal dependencies across multiple time scales, requiring representations that can capture both short-term fluctuations and long-term trends.

3. Multi-task nature of crypto-related decision making

In real-world trading and quantitative finance, multiple tasks (price prediction, volatility forecasting, alpha factor discovery) are inherently interconnected. 

4. Multi-scale and multi-structure dynamics

Cryptocurrency time series exhibit patterns across multiple temporal and structural scales, including short-term noise, medium-term trends, and long-term cycles.

These characteristics highlight a fundamental mismatch between the complexity of cryptocurrency markets and the predominantly task-specific modeling approaches in existing research. This motivates the development of a unified representation learning framework that can capture diverse temporal dynamics and support transferable multi-task applications.

## 3. Model Design

### 3.1 What will the proposed model address?

**Task-specific overfitting**: Existing models are trained for a single task (price prediction, volatility forecasting, or alpha mining) and cannot generalize to other tasks without retraining

**Limited feature diversity**: Traditional approaches rely either on handcrafted indicators or a single deep learning method, failing to capture multi-scale, multi-structural dynamics.

**Transferability gap**: There is no systematic approach to generate reusable representations that can be shared across tasks, coins, or timeframes.

### 3.2 Model Architecture

```
Data processing for training and testing
        ||
        VV
Raw OHLCV Time Series (Polymarket event contracts)
        |
        |-- [Branch: statistical]
        |      |_ AR(p) Coefficients + Residual Statistics
        |      |_ GARCH(1,1): omega, alpha, beta, persistence,
        |                     uncond_var, mean/std conditional variance
        |
        |-- [Branch: transformed]
        |      |_ FFT top-k magnitude coefficients
        |      |_ Haar wavelet detail energies (multi-level)
        |
        |-- [Branch: vae]
        |      |_ VAE encoder (MLP, pretrained unsupervised)
        |
        |-- [Branch: contrastive]
        |      |_ Contrastive encoder (CNN, pretrained via NT-Xent)
        |
        |-- [Branch: ...TBD]
               |_ Additional unsupervised methods to be identified
               |  from literature (masked autoencoder, self-supervised
               |  Transformer, etc.) — each adds one new branch key
                    |
        RepresentationAggregator -> h_i
          Two modes (selected at construction):
          • concat (default): branches concatenated into one
            higher-dimensional vector; no learnable parameters;
            output_dim = sum of branch dims
          • gated: each branch projected to out_dim, then
            softmax gating weights computed from concatenated
            projections; output_dim = out_dim
                    ||
                    || Pass to downstream task heads
                    VV
     -------------------------------------------------------
     |                    |                                |
Price Prediction     Volatility Prediction       Trend Classification
(MLP regressor)      (MLP regressor)             (MLP classifier)
     |                    |                                |
  MAE, RMSE           MSE, Pearson corr.           Accuracy, F1
```

### 3.3 Training

1. Pretrain Base Embeddings: 

Statistical and transformation-based features are computed directly from historical data. For each sliding window sequence, the statistical branch fits an AR(p) model per OHLCV column (capturing linear mean dynamics) and a GARCH(1,1) model on the first-differenced series per column (capturing conditional variance dynamics). Transformation-based features are extracted via FFT and Haar wavelet decomposition. Deep learning embeddings are trained using unsupervised methods (autoencoder reconstruction, contrastive learning, masked prediction). Base embeddings are frozen during downstream training to preserve general-purpose representations.

2. Train Aggregator on Downstream Tasks: 

With frozen encoder weights, the `RepresentationAggregator` is trained jointly with a shallow MLP task head for each downstream task (price prediction, volatility prediction, trend classification). The aggregator’s learned softmax gating adaptively weights contributions from each branch to minimise the task loss. Each task is trained and evaluated independently, sharing the same pretrained encoder checkpoints.

3. Data: 

- Source: Polymarket event prediction market OHLCV data (top-50 most active contracts per timeframe)
- Timeframes: 1-hour, 4-hour, 1-day
- Features: raw OHLCV (5 columns) only — no order book or external data

### 3.4 Innovation

**Extensible multi-branch design**: The `RepresentationAggregator` accepts an arbitrary number of named branches via a dictionary API. Adding a new unsupervised encoder (e.g. a Transformer-based method) requires only registering a new key in `branch_dims` — no changes to the aggregator or any other component.

**Hybrid representation**: Combines deterministic statistical and transformation features (no training required) with neural encoders that are pretrained unsupervised, then fuses all branches via learned gating.

**Task-transferable aggregator**: The same frozen encoder checkpoints and aggregator are reused across all downstream tasks (price prediction, volatility, trend classification), with only a lightweight task head trained per task.

**Semi-/unsupervised support**: Neural encoders are trained without labels (reconstruction, contrastive objectives), requiring only unlabeled OHLCV sequences.

## 5. Evaluation

The framework is evaluated on three downstream tasks. In each case the same test split is used for all models. The exact set of comparison models is provisional and will be finalised based on the literature review.

### 5.1 Evaluation Paradigm

The framework operates as a **frozen encoder evaluated via probing**: multi-branch features are extracted using frozen encoders, then a lightweight MLP task head is trained on those features. Keeping the task head simple is intentional — if the representations are powerful, the decoder should not need to be complex. This is the standard paradigm for evaluating representation quality in the self-supervised learning literature.

### 5.2 Terminology

| Term | Definition |
|---|---|
| **External benchmark** | Model from prior work (end-to-end trained, task-specific). Shows the framework is competitive with the state-of-the-art. Current set: Stacked LSTM, GINN, TA-MLP. |
| **Internal baseline** | Model designed within this project. Shows each framework component contributes. Current set: Raw-OHLCV MLP, single-branch ablations. |
| **Default decoder** | The task head (`PriceRegressor`, `VolatilityRegressor`, `TrendClassifier`) — a simple MLP from `src/tasks/` used by the framework and internal baselines. Intentionally lightweight. |
| **Mirrored decoder** | A benchmark's own FC architecture detached from its encoder and retrained on top of the frozen framework encoder. Used only in the decoder-controlled comparison experiment. |

### 5.3 Tasks and Metrics

1. **Price Prediction**
   - Metrics: MAE, RMSE
   - External benchmarks: Stacked LSTM; additional TBD from literature review
   - Internal baselines: Raw-OHLCV MLP, single-branch ablations

2. **Volatility Prediction**
   - Metrics: MSE, Pearson correlation of predicted vs. realised volatility
   - External benchmarks: GINN; additional TBD from literature review
   - Internal baselines: Standalone GARCH(1,1), single-branch ablations

3. **Trend Classification**
   - Metrics: Accuracy, F1-score
   - External benchmarks: TA-MLP; additional TBD from literature review
   - Internal baselines: Raw-OHLCV MLP, single-branch ablations

### 5.4 Decoder-Controlled Comparison (optional, time permitting, all three tasks)

An additional three-configuration experiment isolates encoder quality from decoder choice by holding the decoder architecture constant. Applies to all three tasks (price prediction, volatility prediction, trend classification) if time permits, subject to availability of a separable benchmark decoder per task:

| Configuration | Encoder | Decoder | Trained |
|---|---|---|---|
| Benchmark end-to-end (e.g. LSTM) | Task-specific encoder, end-to-end | Benchmark's FC head | End-to-end |
| Framework + default decoder | Multi-branch concat (frozen) | Task head (simple MLP) | Head only |
| Framework + mirrored decoder | Multi-branch concat (frozen) | Benchmark FC architecture (retrained) | Head only |

- Comparing configurations **1 vs 3**: same decoder architecture, only the encoder differs — the cleanest test of encoder quality.
- Comparing configurations **2 vs 3**: same encoder, different decoder — isolates whether decoder choice matters.
- If all three configurations produce similar numbers, it confirms the representations are doing the heavy lifting regardless of decoder design.

### 5.5 Ablation Study and Transferability

**Ablation study**: each branch is run independently (no aggregation) against the full N-branch framework to quantify marginal contribution.

**Transferability analysis**: embeddings trained on one timeframe are evaluated on another without retraining, to assess generalisation across temporal scales.

*Note: Alpha factor mining and anomaly detection are out of scope for the current phase. They may be revisited as extensions if time permits.*

## 6. Inspiration and Motivation

Thie idea is inspired by the principle articulated in Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:

"Many information processing tasks can be very easy or very difficult depending on how the information is presented… A good representation is one that makes a subsequent learning task easier. The choice of representation will usually depend on the choice of the subsequent learning tasks."

In other words, the right representation can dramatically simplify downstream tasks.

While representation learning is common in vision, NLP, and general time series, it is rarely applied systematically to cryptocurrency markets, which feature high volatility, continuous 24/7 trading, multi-scale dynamics, and sparse labeling.

# Requirements

1) What is missing?
2) Proposal on design
3) Current progress
4) Pick the baselines and benchmarks

(For Final Report)

1) Analysis of the results 
2) Codebase