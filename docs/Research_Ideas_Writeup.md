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

Autoregressive (AR) and ARIMA models are traditional examples, while more complex methods like Hidden Markov Models (HMMs) capture the probabilistic transitions between different states in the series (Irani et al., 2025).

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
        ||
        ||
        ||
        ||
        VV
Raw OHLCV Time Series/ Level 1 order book data if we have
        |
        |-- Statistical Representations
        |      |_ AR / ARIMA / HMM Statistics
        |
        |-- Transformation-based Representations
        |      |_ FFT Coefficients / Wavelet Features
        |
        |-- Neural Representations
               |_ Autoencoder Encoder (CNN / LSTM / Transformer)
               |_ Contrastive Encoder (CNN / LSTM / Transformer)
                    |
        Representation Aggregation -> h_i
              (Weighted sum or Shallow MLP)
                    ||
                    ||
                    || Pass to downstream tasks
                    ||
                    ||
                    VV
           ---------------------------------------------------------------------------
           |                            │                                            |
  Alpha Factor Mining -> a_i     Price/Volatility Prediction → y_i          Trend/Trading Signal Classicication 
           |                                   |                                            |
    Evaluation Metrics                 Evaluation Metrics                           Evaluation Metrics
    (Information Coefficient)            (MSE, RMSE)                                    (Binary Cross-Enthropy)
            |
    Strategy Formulation
    (Buy/Hold/Sell Signals)
            |
    Backtest (ROI)
```

### 3.3 Training

1. Pretrain Base Embeddings: 

Statistical and transformation-based features are computed directly from historical cryptocurrency data. Deep learning embeddings are trained using unsupervised methods (autoencoder reconstruction, contrastive learning, masked prediction). Base embeddings are frozen during downstream training to preserve general-purpose representations.

2. Train Aggregator on Selected Tasks: 

For each downstream task (e.g., price prediction, volatility forecasting, alpha factor mining), we adopt baseline models from literature (e.g., LSTM, CNN+LSTM, XGBoost). The aggregator module (weighted sum or shallow MLP) learns to combine the heterogeneous embeddings such that the downstream models’ performance is improved. 

Task-specific losses from these baseline models guide the aggregator training, allowing it to adaptively weight features from statistical, transformation-based, and neural embeddings. Some downstream tasks are used exclusively to train the aggregator, while others are reserved for later evaluation to measure transferability.

3. Data selection: 

- Coins: mix of large-cap (BTC, ETH) and mid-cap (LTC, ADA, SOL)
- Timeframes: 5-min, 4-hour, 1-day OHLCV
- Market regimes: bull, bear, range-bounding up, range-bounding down
- Level-1 order book features/OHLCV data

### 3.4 Innovation

Hybrid Representation Learning: Combines statistical, transformation-based, and neural embeddings to capture multi-scale, multi-structural market dynamics.

Task-Transferable Aggregator: Learns to merge heterogeneous embeddings into general-purpose representations, reusable across multiple downstream tasks, coins, and timeframes.

Practical Impact: Supports alpha factor discovery, price/volatility prediction, and trading signal generation in a single unified framework.

Semi-/Unsupervised Support: Can learn from unlabeled crypto price/volatility data, addressing scarcity of supervised labels.

## 5. Evaluation

To demonstrate the effectiveness and transferability of the proposed framework, the model will be benchmarked against multiple downstream tasks using both standard metrics and financial performance indicators:

1. Price Prediction:
- Metrics: MAE, RMSE
- Benchmarks: Stacked LSTM+CNN, LSTM+XGBoost, handcrafted-feature MLPs

2. Volatility Prediction:
- Metrics: MSE, correlation of predicted vs. realized volatility
- Benchmarks: GARCH, LSTM-based volatility models

3. Alpha Factor Mining:
- Metrics: Information Coefficient (IC), Sharpe Ratio
- Benchmarks: Existing factor-mining pipelines using handcrafted or neural features

4. Anomaly Detection
- Benchmarks: GCN-GRN model for anomaly detection

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