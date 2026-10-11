# A Unified Representation Learning Framework for Multi-Task Time-Series Modeling in Event Prediction Markets

**Report:** Research Motivation Report

**Why Are Financial Representations Useful, and What Can Our Research Contribute?**

*A literature-based motivation for a heterogeneous time-series representation framework in prediction markets*

## Executive Summary

Financial modeling and trading rely on time-series data, including historical prices, returns, and trading volumes. These observations show how markets evolve and support many applications, from price forecasting to risk assessment. The central challenge is to extract information that is useful for these different objectives from data that are often noisy and complex.

Researchers use statistical features, signal transformations, and supervised deep-learning models to analyze financial time series. These methods can be effective, but their features are shaped either by manually selected assumptions or by particular prediction targets. Financial targets themselves are not always straightforward: researchers must choose forecasting horizons, labeling rules, and prediction objectives, and those choices may vary across applications or market conditions.

Financial research can improve both the definition of prediction labels and the features used to predict them. Representation learning has been used for both purposes: learned reconstruction can support label construction, while pretrained encoder outputs can serve as features for downstream models. Our project focuses on feature extraction. We use fixed target definitions and evaluate whether the same extracted features are useful for several prediction tasks.

Previous studies in conventional financial markets report benefits from learned features, additional pretraining objectives, and the reuse of pretrained encoders. Our research evaluates these ideas in **prediction markets**, using a heterogeneous framework that considers classical statistical features, signal transformations such as FFT, and deep representation learning.

We distinguish two questions throughout this report:

1. **Research motivation:** Why might broader and reusable financial representations be valuable?
2. **Research objective:** Can our heterogeneous representations provide useful, complementary, and reusable information for prediction-market forecasting?

Within each chronological walk, our neural encoders are pretrained and then frozen. Separate prediction heads use their features for future-price prediction, movement classification, and future realised-variance prediction. The research evaluates how well these shared features perform against suitable baselines. Canonical branch complementarity will be examined in the later ablation analysis.

---

# Part I. Financial Time-Series Modeling and Existing Methods

## 1. Why Financial Time Series Matter

A single price or volume observation describes one moment in a market. A historical sequence shows how that market changes. Such sequences can reveal trends, reversals, changing volatility, and shifts in trading activity. Financial modeling uses these temporal relationships to support predictions and decisions.

The same history may be relevant to different questions:

| Financial application | Example objective | Information of interest |
|---|---|---|
| Price or return forecasting | Estimate a future price or return | Expected future value or change |
| Movement classification | Predict whether a price will rise or fall | Directional information |
| Volatility forecasting | Estimate future price or return variability | Size and persistence of fluctuations |
| Risk assessment | Estimate the chance or size of a loss | Downside and extreme outcomes |
| Anomaly detection | Identify unusual market behavior | Departures from typical patterns |
| Trading decisions | Decide whether and how to trade | Expected outcomes, uncertainty, and costs |

These applications share historical market observations but do not require identical information. **Extracting useful features from the time series is therefore a common problem underlying many financial tasks.**

## 2. How Financial Features Are Currently Extracted

Financial research uses several approaches:

| Approach | What it does | Examples |
|---|---|---|
| Classical statistical features | Summarizes selected properties of past observations | Moving averages, momentum, volatility, autocorrelation |
| Signal transformations | Expresses the sequence in another domain | FFT, wavelet or frequency-domain features |
| Task-specific supervised modeling | Learns features while optimizing a selected prediction target | LSTM, CNN, Transformer forecasters and classifiers |
| Representation learning and pretraining | Learns features through reconstruction, auxiliary, contrastive, or other objectives before downstream prediction | Autoencoders, self-supervised encoders, latent-factor models |

All four approaches can be useful. Statistical indicators are often simple and interpretable. FFT makes frequency components explicit, although it does not create new information. Supervised deep networks can fit complex relationships and perform strongly when the task is clear. Pretrained representations offer a different possibility: separating feature extraction, at least partly, from the final task.

In this report, **representation framework** refers broadly to the system that organizes statistical, transformed-domain, and learned features. Not all its components are representation-*learning* methods: FFT and manually computed statistical summaries are not learned encoders.

An **encoder** is a trained model that converts an input sequence into features. **Pretraining** trains that encoder before the downstream prediction model. A **frozen encoder** keeps its weights unchanged while a separate prediction model learns to use its output.

---

# Part II. Challenges That Motivate Broader Representations

## 3. Challenge One: Handcrafted Features Capture Properties Chosen in Advance

Traditional indicators summarize aspects of price behavior that analysts consider useful. Their advantages are simplicity, interpretability, and established statistical meaning. However, selecting the right indicators may require expertise, and a fixed set of indicators may overlook complex nonlinear or temporal relationships.

**Evidence from previous research.** Ying et al. (2024) [1] explicitly motivate their SMART model by the dependence of stock-trend prediction on expert-designed features and the difficulty of uncovering hidden information from price sequences. They use self-supervised stock-sequence features and report favorable stock-trend prediction results, including a financial-service deployment. Self-FTS (2022) [2] similarly studies self-supervised financial representations as an alternative or complement to handcrafted features, with downstream stock-trading evaluation.

**What this supports:** Data-driven feature extraction is a credible complement to traditional indicators. It does not mean that handcrafted features should always be replaced.

## 4. Challenge Two: Financial Prediction Tasks and Labels Are Difficult to Define

Financial time-series modeling does not always have one natural, universally accepted prediction target. Researchers or practitioners must decide what they want to predict and how the target should be measured.

For example, the same historical observations might be used to construct different prediction targets:

| Task | Possible target |
|---|---|
| Price forecasting | Contract price one hour ahead |
| Movement classification | Whether the price increases by more than 1% over the next hour |
| Volatility forecasting | Realized price or return variability over the next 24 hours |
| Anomaly detection | Whether current activity differs substantially from a reference pattern |
| Risk assessment | Probability of a large adverse price change over a chosen horizon |

Even within the same task, researchers must choose the prediction horizon, labeling thresholds, class definitions, and treatment of small fluctuations.

These choices determine how training labels are constructed. Changing them may affect which patterns are useful for prediction and how well a model performs.

**The difficulty is that the most appropriate target definition may not be obvious in advance.** Furthermore, a suitable labeling rule for one dataset or market condition may not be equally appropriate for another.

### Evidence from previous research

**Ma, Ventre, and Polukarov (2021/2022) [3]** investigate the difficulty of constructing reliable classification labels from noisy financial price sequences.

Their denoising autoencoder learns to reconstruct smoothed prices using moving-average series as reconstruction targets. They calculate returns from the reconstructed prices and apply thresholds to generate movement labels. An SVM classifier is then trained using those labels. The experiments cover daily S&P 500 prices, one-minute Bitcoin prices, and high-frequency Google order-book data. [Paper, Section 4](https://arxiv.org/html/2112.10139v1)

The authors report higher classification F1 scores under the denoised-label scheme. Both training and evaluation labels are derived from the reconstructed prices, so this result concerns a changed labeling scheme. It does not establish more accurate prediction of the original unsmoothed labels.

Beyond label quality, previous research also shows that financial labeling rules may need to account for changing market conditions.

**López de Prado (2018) [4]**, in *Advances in Financial Machine Learning*, discusses limitations of fixed-horizon labeling and introduces volatility-adjusted thresholds through the triple-barrier method.

For example, a fixed 1% price movement may represent a significant event during a low-volatility period but an ordinary fluctuation during a high-volatility period.

This motivates using thresholds that account for prevailing volatility rather than relying exclusively on fixed price-change boundaries.

More recently, **Koukorinis et al. (2025) [14]**, in *Generative-Discriminative Machine Learning Models for High-Frequency Financial Regime Classification*, explicitly investigate dynamic financial labeling.

The authors construct classification labels using empirical quantile thresholds that adapt to local volatility dynamics and market conditions.

Their method also allows different labeling configurations to capture short-term or longer-term behavior and binary or multi-class market regimes.

These studies give concrete reasons to examine label construction. Their methods address the question of how to define the outcome a model should predict.

### Key Argument

Prediction horizons, movement thresholds, and classification rules determine the labels used to train and evaluate a model. A useful labeling rule therefore needs to match the question being asked and the data being studied.

Improving labels is one research direction. Improving the features supplied to the prediction model is another. The examples above concern label construction; Section 6 explains how representation learning can also be used for feature extraction.

Different target definitions give us a reason to test features across several tasks. They do not establish that a particular representation will work well on those tasks.

## 5. Challenge Three: Supervised Feature Learning Depends on the Selected Target

Supervised deep-learning models learn features while optimizing a selected prediction target. A model trained to predict future prices learns features that help with price prediction. Those features may also help predict volatility, but this needs to be tested.

Changing the target, forecasting horizon, or movement threshold can change which features help the model. A model trained for one target may therefore need adaptation or retraining for another.

This motivates testing whether features learned before downstream training can support several prediction tasks. A separate prediction model is trained on the extracted features for each task. Their usefulness is assessed through prediction results.

For example, two market conditions may have the same expected future return but very different volatility. A model trained only to predict the expected return is not explicitly required by its loss to distinguish those volatility conditions. It may still learn that information, but retaining it is not guaranteed by the training objective.

Task-specific models can be strong predictors. The research question is whether shared features can also support useful predictions, and where specialized models perform better.

**Evidence from previous research.** Wang, Ma, and Cohen (2025) [5] propose SSPT, which uses stock-code classification, sector classification, and moving-average prediction as pretraining objectives before stock selection. Their KDD study reports improvements in investment-return and Sharpe-ratio metrics over evaluated baselines. This provides evidence that features learned beyond the final prediction objective can benefit that objective.

Ma and Tan (2022) [6] find that auxiliary supervised tasks can help stock ranking. Although this is multi-task learning rather than target-independent representation pretraining, it independently supports the value of learning from more than one supervision signal.

**What this supports:** A single prediction loss is not always the only useful source of learning signals. Additional objectives and broader feature representations may help, although none is guaranteed to improve every task.

## 6. Why These Challenges Matter Together

Financial prediction depends on both the definition of the target and the features used to predict it. Existing research uses representation learning in at least two ways to address these parts of prediction.

### 6.1 Improving Labels

This approach changes how prediction labels are constructed. The denoising study [3] in Section 4 uses a learned reconstruction to generate movement labels before classification.

Volatility-adjusted thresholds [4] provide another way to revise labels. Label improvement can use learned methods or explicit rules.

### 6.2 Improving Features

This approach changes the features supplied to a downstream prediction model. An encoder is first trained using an objective such as reconstruction or contrastive learning. Its output is then evaluated for other tasks.

Zhong, Lin, and Yang [7] train limit-order-book encoders through reconstruction. They replace the reconstruction decoder with simple downstream models for price-trend prediction and missing-data imputation. They also test freezing the encoder and training only the downstream models. Reconstruction trains the feature extractor; it is not used to generate replacement movement labels. [Paper, Section 4.2](https://arxiv.org/html/2505.02139v1#S4.SS2)

Dolphin, Smyth, and Dong [8] learn asset embeddings from historical returns. Return similarity across time windows is used to choose positive and negative asset pairs. Contrastive training draws similar assets together in the embedding space and separates dissimilar assets. The embeddings are evaluated for industry classification and portfolio hedging. [Paper, Sections 3 and 4](https://arxiv.org/html/2407.18645v1)

These studies explain how features can be trained before their downstream use is evaluated. The pretraining objective still shapes what the encoder learns. Useful features for every task are not guaranteed.

### 6.3 Our Project's Focus

Our project focuses on feature extraction and reuse. We use fixed definitions for future price, movement classification, and future realised variance. The labels are constructed from the accepted price data rather than from a learned denoised reconstruction.

Within each chronological walk, the same pretrained encoder weights are frozen and reused across tasks. Each task has its own prediction head, which is the model trained to predict that task's target from the extracted features.

We ask:

> Can the same extracted features support several prediction tasks, and how well do they perform compared with suitable baselines?

Label improvement and feature improvement can be used together, but neither guarantees the other. A useful representation does not repair a poorly defined target, and a suitable target does not ensure that a model learns useful features.

---

# Part III. Why a Financial Representation Framework Is Worth Studying

## 7. Evidence of Real Value in Financial Research

Published studies provide three reasons to investigate representation learning in finance.

### 7.1 Discovering Financial Features Beyond Handcrafted Indicators

SMART [1] and Self-FTS [2] investigate learned features for stock-trend prediction and trading-related evaluation. Gu, Kelly, and Xiu (2021) [9] use autoencoder-based nonlinear latent factor modeling for asset pricing and report lower out-of-sample pricing errors than leading factor-model alternatives.

These results support the value of learning financial structure not fully captured by preselected indicators or restricted linear models. They do not establish that learned representations universally outperform handcrafted features.

### 7.2 Learning Information Beyond One Prediction Objective

SSPT [5] shows that stock-specific pretraining objectives can improve final stock-selection performance in the evaluated settings. This is particularly important for the research motivation: pretraining can have value **even for a single downstream application**, rather than only when features are reused across tasks.

### 7.3 Reusing Features Across Tasks and Datasets

Zhong, Lin, and Yang (2025) [7] introduce LOBench for limit-order-book representation learning. As described in Section 6, they examine reconstruction, price-trend prediction, and imputation. Their frozen-encoder experiment reports performance comparable to joint encoder-and-decoder training, with more training epochs. They also examine cross-stock transfer while adapting a prediction head. These results support studying feature reuse, with performance interpreted under the specific training conditions.

Dolphin, Smyth, and Dong (2024) [8] evaluate their return-based embeddings for industry classification and portfolio hedging. The loss function that performs strongly on classification is not the strongest for hedging. Their results show both the potential for reuse and the importance of evaluating each downstream task.

**Overall lesson:** Other financial researchers have found practical reasons to learn and evaluate reusable representations. This is a recognized empirical research problem, not merely a preference for a particular model architecture.

## 8. Why Investigate Heterogeneous Representations?

A deep encoder is one way to extract features from a financial time series. Our framework also includes statistical and transformed-domain features. Each method is included for a specific reason:

| Method | What it is designed to describe or learn |
|---|---|
| AR and GARCH features | Fitted summaries of local dependence and conditional variation |
| FFT and wavelet features | Frequency content and variation at different scales |
| Variational autoencoder | A latent representation trained to reconstruct the input, with a regularization term |
| Contrastive encoder | Features trained to agree across selected augmented views and distinguish other examples |
| BYOL encoder | Features trained through prediction between online and target networks using augmented views |

These design differences motivate comparison, but they do not prove that the methods capture different useful signals. Reconstruction can preserve details that do not help forecasting. Augmentations can encourage useful consistency but can also remove details needed by a task. Statistical and transformed features can overlap with neural features.

Complementarity therefore needs experimental evidence. We test whether using feature sets together improves a matched downstream comparison. Increasing the feature width or observing different embeddings alone does not establish added predictive value. Changes in downstream model capacity must also be considered.

Sun et al. (2025) [10] combine time-domain and frequency-domain components for stock prediction and report improvements over their selected baselines. This provides a precedent for examining multiple feature views in finance; it does not guarantee that the same combination is helpful for prediction-market data.

Our framework therefore has a broader purpose than training a self-supervised encoder. It aims to **compare and, where appropriate, combine heterogeneous features** under a shared evaluation protocol.

The completed controlled encoder-addition studies provide evidence for selected additions. The later ablation analysis will compare individual canonical branches, the full combination, and combinations with one branch removed. That analysis will examine which branches contribute to each task and whether their combination helps.

## 9. What Is Motivation, and What Is the Research Objective?

The distinction matters:

| Motivation from financial practice and literature | Objective that our experiments investigate |
|---|---|
| Handcrafted features may omit useful patterns | Are the evaluated representations informative for prediction-market forecasting? |
| A chosen supervised objective may not preserve everything other tasks need | Can common representations support several different downstream tasks? |
| Financial targets and label choices differ across applications and contexts | Does the framework show cross-task reuse under the selected fixed evaluation definitions? |
| Statistical, spectral, and neural methods may highlight different properties | Do the evaluated representations provide complementary predictive information? |

The experiments evaluate these objectives under fixed task definitions. Their claim boundaries are stated together in Section 16.

---

# Part IV. Why Prediction Markets?

## 10. Prediction Markets Have Distinctive Time-Series Behavior

Much existing representation-learning research concerns equities, foreign exchange, and limit-order-book trading. Prediction markets differ because contracts are tied to future event outcomes. Binary YES-share prices commonly lie between zero and one and can be interpreted, with qualifications, as market-implied probabilities. Many contracts have finite lifetimes and settle when the relevant event is resolved.

Their historical series may be affected by sudden event-related information, changing time to resolution, uneven liquidity and trading activity, and differences among event categories. These characteristics mean that findings from conventional financial markets should not simply be assumed to apply unchanged.

Our forecasting tasks concern future traded prices and their changes. They do not directly evaluate the accuracy of a contract price as an estimate of the event's true probability.

## 11. What Earlier Prediction-Market Research Shows

**Historical price behavior is informative.** Sung et al. (2019) [11] analyze more than 8.4 million price observations from 6,058 betting-exchange markets. They find relationships between historical price trends and errors in market-implied event probabilities and use these relationships to improve probability forecasts. This gives a direct domain-specific reason to examine information in historical prediction-market time series rather than only the most recent price.

**Feature and representation methods have already appeared in prediction-market research.** Saguillo et al. (2025) [12] study arbitrage on Polymarket using market relationships and topical similarity to help identify relevant contracts. Capponi, Gliozzo, and Zhu (December 2025, original preprint) [13] use semantic analysis of contract descriptions to cluster markets and identify relationships for trading signals.

These works show that structured information extraction is valuable to researchers working on prediction markets. However, their central emphasis is on relationships among contracts or market-implied probabilities, rather than a systematic evaluation of **statistical, frequency-domain, and deep-learned temporal representations reused across multiple forecasting targets**.

## 12. Research Gap

The reviewed studies give reasons to investigate representation learning in finance and historical price information in prediction markets. They leave open the question of how useful a shared combination of statistical, transformed-domain, and pretrained neural features is for several prediction-market forecasting tasks.

Our study evaluates this question through common data and evaluation rules. It compares the framework with raw-input models, other pretrained representations, and suitable task-specific models. The later ablation analysis will examine whether the canonical feature branches add predictive value when combined.

The contribution is evidence about those comparisons in prediction-market forecasting. Building a combined architecture or applying it to a new dataset alone would not answer the research questions.

---

# Part V. Research Objectives, Questions, and Contributions

## 13. Research Objective

We aim to develop and evaluate a unified heterogeneous representation framework for historical prediction-market time series. It considers classical statistical features, signal transformations such as FFT, and deep representation learning, including self-supervised methods.

We evaluate three downstream tasks using a common experimental setting:

1. **Future-price prediction:** Forecast a future contract price.
2. **Movement classification:** Predict the direction of a future contract-price change.
3. **Volatility prediction:** Forecast realised variance over a strictly future interval, calculated from squared changes in contract prices.

The focus is on how useful the extracted information is, not on assuming one representation is best for every task.

## 14. Research Questions

**RQ1 — Representation Effectiveness and Complementarity**

*Can heterogeneous representations extract useful and complementary information from prediction-market time series for downstream forecasting?*

This asks whether statistical, transformed-domain, and learned features are useful individually and whether their combination adds predictive information beyond the individual approaches.

The completed comparisons address feature usefulness and selected controlled encoder additions. The later ablation analysis will examine individual canonical branches and their combined contribution.

**RQ2 — Representation Reusability**

*Can the same frozen features support several prediction-market forecasting tasks, and how well do they perform compared with suitable baselines?*

Within each chronological walk, the same encoder weights are reused for price forecasting, movement classification, and realised-variance forecasting. A separate prediction head is trained for each task. Comparisons with raw-input models, other frozen representations, and specialized models assess predictive usefulness as well as the ability to reuse features. Transferring one walk's encoder to another walk is a different question and is outside these experiments.

## 15. Research Contributions and Remaining Analysis

**Contribution 1 — A common prediction-market evaluation.** We implement a heterogeneous feature framework and compare it with raw-input models, external frozen representations, and task-specific models under documented chronological evaluation rules. Each comparison uses matched task rows or an explicitly defined common intersection.

**Contribution 2 — Evidence on feature reuse.** We evaluate the same frozen encoders with separate prediction heads for future price, movement, and realised variance. The completed comparisons identify where shared features remain competitive and where other models perform better.

**Contribution 3 — Analysis of representation choices and complementarity.** Completed controlled encoder studies examine substitution, addition, and capacity. The later single-branch and leave-one-branch-out ablations will examine the contributions of the canonical branches. This remains a planned analysis stage for the final report and research paper.

The research contribution depends on clear comparisons and informative analysis. Mixed results can identify which representation choices help particular tasks and which do not.

## 16. How We Will Support the Claims

| Proposed claim | Experimental evidence required |
|---|---|
| Extracted features are useful | Compare against raw-input and simple handcrafted-feature baselines |
| Different representations complement one another | Compare single-method features with feature fusion, using ablations |
| Features can be reused effectively | Evaluate the same frozen encoders with different downstream heads and compare predictive performance with suitable baselines |
| The approach is competitive | Compare with relevant supervised, task-specific, and representation-learning baselines under consistent evaluation rules |
| Results are credible | Use chronological splits, avoid future leakage, and examine variation across available contracts or periods |

### Scope of the Claims

The experiments evaluate forecasting under fixed target definitions and reuse across tasks within each walk. They do not establish improved financial labels, automatic adaptation to changing targets or market regimes, transfer to unseen market categories, or reduced computational or engineering cost.

Predictive performance is assessed separately for each task and metric. Stronger forecasting metrics do not establish profitable trading after spreads, fees, and execution costs. Canonical branch complementarity will be assessed through the later ablation analysis.

---

# Conclusion

Financial prediction requires suitable targets and useful features. Existing research has used learned reconstruction to improve label construction, and reconstruction or contrastive objectives to train features for downstream tasks.

Our project focuses on feature extraction. It combines statistical features, signal transformations, and pretrained neural encoders, then evaluates the same frozen encoders through separate prediction heads for future price, movement classification, and future realised variance.

The comparisons assess how well those features support each task and where raw-input or specialized models perform better. The later ablation analysis will examine which canonical branches contribute and whether their combination adds predictive value.

**The project contributes experimental evidence about feature usefulness and reuse in prediction-market forecasting, with branch complementarity reserved for the later ablation analysis.**

---

# References

[1] Ying, Z., Cheng, D., Chen, C., et al. (2024). *Predicting stock market trends with self-supervised learning.* **Neurocomputing**, 568, 127033. https://doi.org/10.1016/j.neucom.2023.127033

[2] Sun, J., Qing, Y., Liu, C., & Lin, J. (2022). *Self-FTS: A Self-Supervised Learning Method for Financial Time Series Representation in Stock Intraday Trading.* **IEEE INDIN 2022**. https://doi.org/10.1109/INDIN51773.2022.9976077

[3] Ma, Y., Ventre, C., & Polukarov, M. (2022; first preprint 2021). *Denoised Labels for Financial Time Series Data via Self-Supervised Learning.* **ACM ICAIF 2022**. https://doi.org/10.1145/3533271.3561687 ; https://arxiv.org/abs/2112.10139

[4] López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 3: Labeling. **Wiley**. https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml

[5] Wang, M., Ma, T., & Cohen, S. B. (2025). *Pre-training Time Series Models with Stock Data Customization.* **KDD 2025**, 3019–3030. https://doi.org/10.1145/3711896.3737005

[6] Ma, T., & Tan, Y. (2022). *Stock Ranking with Multi-Task Learning.* **Expert Systems with Applications**, 199, 116886. https://doi.org/10.1016/j.eswa.2022.116886

[7] Zhong, M., Lin, Y., & Yang, P. (2025). *Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking.* **arXiv:2505.02139**. https://arxiv.org/abs/2505.02139

[8] Dolphin, R., Smyth, B., & Dong, R. (2024). *Contrastive Learning of Asset Embeddings from Financial Time Series.* **ACM ICAIF 2024**. https://arxiv.org/abs/2407.18645

[9] Gu, S., Kelly, B., & Xiu, D. (2021). *Autoencoder Asset Pricing Models.* **Journal of Econometrics**, 222(1), 429–450. https://doi.org/10.1016/j.jeconom.2020.07.009

[10] Sun, W., Mei, J., Liu, S., Yuan, C., & Zhao, J. (2025). *Research on deep learning model for stock prediction by integrating frequency domain and time series features.* **Scientific Reports**, 15, 30386. https://doi.org/10.1038/s41598-025-14872-6

[11] Sung, M.-C., McDonald, D. C. J., Johnson, J. E. V., Tai, C.-C., & Cheah, E.-T. (2019). *Improving prediction market forecasts by detecting and correcting possible over-reaction to price movements.* **European Journal of Operational Research**, 272(1), 389–405. https://doi.org/10.1016/j.ejor.2018.06.024

[12] Saguillo, O., Ghafouri, V., Kiffer, L., & Suarez-Tangil, G. (2025). *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets.* **AFT 2025**. https://arxiv.org/abs/2508.03474

[13] Capponi, A., Gliozzo, A., & Zhu, B. (2025). *Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets.* **arXiv:2512.02436, version 1 (December 2025)**. https://arxiv.org/abs/2512.02436v1

[14] Koukorinis, A., Peters, G. W., & Germano, G. (2025). **Generative-Discriminative Machine Learning Models for High-Frequency Financial Regime Classification.** *Methodology and Computing in Applied Probability*, 27, Article 36. https://doi.org/10.1007/s11009-025-10148-8
