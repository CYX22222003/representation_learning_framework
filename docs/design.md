# Model Design

> **Phase 5 authority (2026-09-21):** The initial architecture and experiment
> design evolved after issues discovered in Phases 1--3; Phase 4 was a
> data-analysis phase and ran no model training. For the active main-experiment
> data, tasks, assumptions, and scope, read
> [`phase_plan/2026-09-21-phase-5-experiment-plan.md`](phase_plan/2026-09-21-phase-5-experiment-plan.md)
> first. It supersedes conflicting older Phase 5 handoff language below.

## Architecture Design

The proposed model processes raw OHLCV time-series data through an extensible set of named representation branches: a `statistical` branch (AR and GARCH features), a `transformed` branch (FFT and Haar wavelet features), and neural branches (`vae`, `contrastive`, and `byol`; additional unsupervised methods may be added). These representations are fused by a `RepresentationAggregator` into a unified embedding *h_i*, which is passed to lightweight MLP task heads for three downstream tasks: probability-movement regression, volatility prediction, and movement/trend classification. Absolute next-close prediction is retained as a completed negative characterisation study rather than the primary regression probe.

The aggregator supports two fusion modes. In *concat mode* (default), branches are concatenated into a single higher-dimensional vector with no learnable parameters; the task head absorbs all supervised learning. In *gated mode*, each branch is projected to a shared dimension and a gating network produces per-branch softmax weights. Concat mode serves as the primary implementation and as an ablation comparison for gated mode. A detailed architecture diagram is provided in the Appendix.

## Experiment Design

The experimental setup is designed to evaluate the effectiveness of the unified representation learning framework on event prediction market data, focusing on two downstream tasks: regression/prediction (probability or return forecasting) and classification (trend direction or event outcome).

### Data Preparation

> **Correction required (2026-09-20):** The legacy implementation normalised
> and windowed each complete contract before splitting generated windows. Phase
> 2 is paused because this allowed test-period information into fitted volume
> scaling and left the raw holdout boundary ambiguous. The bullets below state
> the required replacement design; see `docs/data_processing_split_contract.md`.

> **Phase 5 contract (2026-09-21):** Phase 4 concluded the
> data-selection and exploratory-analysis stage without launching models. Phase
> 5 preserves raw-time-first preprocessing and global-calendar walk-forward
> evaluation and uses the two fresh walk-specific top-50 clean native one-hour
> FinData cohorts with isolated-one-bar filling. Retrospective universe
> selection is accepted and approved final pruning is assumed correct offline
> cleaning. Every walk has a separate model; later information cannot train an
> earlier-walk model. Contract lifecycle remains a reporting stratum.

The dataset consists of OHLCV time-series data from approximately 72,222 event contracts from *Polymarket*, with varying timesteps (1-hour, 4-hour, and 1-day). The data preparation process is designed to produce training-ready sequences for representation learning while preserving temporal order and market-specific dynamics. Legacy and Phase 3 experiments use top-50 cohorts. The unexecuted Phase 4 archive design studied cutoff-local four-hour top-80 selection. Phase 5 uses the already acquired fresh Walk 1 and Walk 2 top-50 one-hour cohorts and accepts their retrospective catalog selection as an FYP assumption.

A separate read-only FinData acquisition module can collect newer Polymarket
rows under Git-ignored `data_new/`. The expanded December-2025--August-2026
audit stores condition-level candles for source diagnosis and token-identified
YES trades for valid orientation. Condition candles may mix YES/NO prices; the
trade-only fallback is sparse and currently supplies complete seq64+h2 rows
for only two related contracts. A raw-first 50-contract audit uses a versioned
forward-confirmed quarantine that preserves persistent crashes/repricings,
removes rather than repairs transient complementary or unsupported extreme-range
rows, leaves timestamp gaps, and fails if removal exceeds 1% of either native
resolution. Its 116/325,730 15-minute and 147/107,635 hourly removals are below
that budget; each decision records its causal availability time. Acquisition or
99% retention does not independently prove token identity. Phase 5 explicitly
assumes the approved pruning is correct and treats the final clean native
one-hour candles as the canonical corrected historical source for a
confirmatory comparison within that assumption. It does not replay pruning by
`quarantine_available_at`.

For the selected Phase 5 one-hour source, at most one complete missing hourly
bar may be filled causally as a flat, zero-volume candle with explicit
`is_imputed` and time-since-observation metadata. Longer gaps split sequences,
and decision and target endpoints remain observed. Native 15-minute filling is
a resolution sensitivity only. Observed-only movement is primary, filled-grid
results remain diagnostics, and stochastic price augmentation is not written
into canonical OHLCV or targets.

The canonical model input remains five-channel OHLCV. In both fresh walk
artifacts, `volume == 0` exactly identifies imputed rows and all observed rows
have positive volume, so volume is the implicit missingness signal. Explicit
imputation metadata remains mandatory for validation, eligibility, and
imputation-exposure reporting but is not an additional model channel.

- **Timestep Separation:** Markets are grouped by their time resolution (1h, 4h, 1d) to handle differing temporal dynamics. Each group is processed independently.

- **Market-level boundary and preprocessing:** For each contract market:
  - Establish the chronological raw-time 80/20 boundary first.
  - Missing values are handled via interpolation or forward-filling to maintain continuous sequences.
  - Any imputation or scaling parameters are fitted on training history only and
    applied causally with frozen parameters.
  - Sliding windows are constructed after the boundary under an explicit context
    policy; no test-period observation may enter a training window or target.
  - Training-only augmentation may be applied to transient self-supervised
    views after the causal split. It never rewrites canonical OHLCV, fills a
    missing FinData candle, or creates a decision or target endpoint.

- **Train-Test Split:** Phase 3 used a per-contract raw chronological 80/20
  split before fitted preprocessing and window generation. Phase 5 instead
  uses fixed-duration rolling global calendar-time walks. At cutoff `T_k`, every pooled
  contract contributes only information available before `T_k`; the next
  calendar interval is evaluation-only. This prevents a later observation
  from one contract training a model scored on an earlier observation from
  another. The exact timestamps, two-walk count, universe eligibility, minimum
  history, activity mask, and target-maturity rules must be frozen in the Phase
  5 contract derived from the Phase 4 conclusion.

- **Lifecycle diagnostic:** On the selected top-50 4-hour contracts, exact
  zero movement rose from `30.00%` in the early lifecycle third to `58.06%` in
  the late third, while the zero-movement baseline MAE fell from `0.010659` to
  `0.004268`. Saved feature branches also contain contract-general lifecycle
  information. This motivates lifecycle-stratified evaluation and encoder
  adaptation tests; it does not by itself prove that different stages require
  different architectures. See
  `docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md`.

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
  - **Phase 2 temporal variants** — contrastive and BYOL are selected peer branches. Each has named LSTM and compact-Transformer substitutions that preserve its own SSL objective, augmentations, and projector/predictor contract and expose a 128-dimensional frozen backbone state. A primary comparison replaces exactly one corresponding CNN branch; it never adds a sixth branch or changes both neural branches together.
  - **Additional methods (TBD)** — further unsupervised approaches (e.g. masked autoencoders, self-supervised Transformers) may be integrated based on the literature review. Each new method is registered as an independent branch in the aggregator.

### Training Procedure

- All model training—supervised and unsupervised—uses only causally permitted
  training information. Legacy and Phase 3 runs use the documented
  per-contract split; each Phase 5 walk uses one global calendar cutoff. Its
  following interval is held out until evaluation under the predeclared model
  × task × epoch-budget matrix.

- **Train and test only — no validation split, no early stopping.** Every model uses a fixed epoch budget (`--epochs N`); for external benchmarks a small characterization sweep across epoch budgets is run at one fixed seed, and the full sweep is reported rather than a best-on-test entry. See `docs/training_test_data_selection.md` for the rationale and the full set of rules.

- Neural encoders (VAE, contrastive, and BYOL) are
  pretrained unsupervised on training sequences only, then their weights are
  frozen. The Phase 5 primary adaptive evaluation trains separate encoder
  weights per global walk.

- A lifecycle-conditioned shared encoder/head or predeclared stage-specific
  experts, encoder variants, gated fusion, and temporal-transfer comparisons
  are deferred to Phase 6.

- Frozen encoders are used to extract neural embeddings for both training and test sequences. Running inference through a frozen encoder on test data is not leakage — the encoder parameters contain no information derived from test sequences.

- The aggregator and task heads are trained on training feature bundles (statistical + transformed + separately named frozen neural branches) for fixed epoch budgets; each predeclared checkpoint receives one test pass and the complete epoch-budget matrix is reported without selecting a best-on-test run.

- New price-prediction runs use the saved contract-safe horizon-1 label
  bundle. Target construction is independent within every contract and split,
  so the terminal row of each contract is removed instead of being paired
  across a merged-array boundary. This yields 109,791 training and 27,450 test
  rows on the current 50-contract data. Earlier price runs with
  `labels_npz: null` use the legacy 109,840/27,499-row contract and are not
  strict comparisons with new runs. See
  `docs/price_prediction_label_contract.md`.

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
| **Refined decoder** | Phase-2 static residual, gated, recurrent, or attention decoder trained on the unchanged frozen five-branch representation. |

**Evaluation paradigm (probing):** The framework uses frozen representation extractors. After pretraining, the named branch features remain fixed while a lightweight task head, and the aggregator when it is learnable, are trained for each task. Keeping the task head simple is intentional — if the representations are powerful, the decoder should not need to be complex. Any benchmark comparison is against an end-to-end trained model, which has more optimisation freedom; matching or beating it with frozen representations + a simple head is the primary claim.

- **Benchmark Retraining:** Each benchmark model is retrained on the same event
  prediction market data using the same sliding-window identities and global
  calendar walk. Framework encoders, heads, and baselines all obey the same
  cutoff. This project does not use a validation split or early stopping; see
  `docs/training_test_data_selection.md`.

- **Volatility benchmark adaptation:** The strict volatility comparison uses a shared realised-volatility label bundle. Raw LSTM volatility is the direct end-to-end neural benchmark. The adapted GARCH--LSTM stack is a complementary, stronger hybrid benchmark: it fuses causal guarded GARCH forecasts with the same Raw LSTM forecasts through fixed ElasticNet meta-features `[g, l, g*l]`. Its expanding cross-fitting is used only to create out-of-fold training features for the meta-learner; it is not validation or model selection. The existing Raw-OHLCV MLP volatility sweep uses a legacy merged-array target helper and is characterization-only until it is migrated to this shared bundle; the framework volatility task must also consume this bundle before any strict comparison. A framework result should therefore report its relationship to both benchmarks rather than treating the stack as evidence that standalone GARCH is superior.

- **Embedding-based Model Training:**
  - Deterministic branches (statistical, transformed) require no training; neural branches are pretrained unsupervised and their encoder weights are frozen.
  - All branch embeddings are extracted into a branch-aware `FeatureBundle`: deterministic arrays are saved as `statistical` and `transformed`, and neural embeddings are saved under their encoder names such as `vae`, `contrastive`, and `byol`. The `RepresentationAggregator` receives these named branch tensors and fuses them into a unified embedding *h_i* per sequence.
  - **Downstream Task Preparation:**
    - **Regression Task:** supervised pairs (X, y), where X is the sequence
      embedding and `y = close[t+h] - close[t]` is continuous future
      probability movement. Labels are contract- and fold-local. Absolute
      next-close prediction remains a Phase 3 diagnostic only.
    - **Classification Task:** labels such as trend direction or event outcome mapped to embeddings as input-output pairs. Phase 1 retains its TA-MLP-style tri-class BUY/HOLD/SELL bundle. The isolated Phase 2 task uses hard `DOWN/STABLE/UP` labels from absolute probability movement over a split-safe horizon and saves three-class scores. Its candidate imbalance protocols are majority undersampling (`P1U`), balanced oversampling (`P1O`), and train-prior logit-adjusted cross-entropy (`P2`); natural cross-entropy (`P0`) is an untreated reference only.
  - A lightweight MLP task head is trained on these (X, y) pairs.

- **Performance Comparison:**
  - Evaluate strict task comparisons on the same global-calendar walk identities
    and aligned label rows, then stratify results by contract lifecycle stage.
  - Consistent metrics: Regression → MAE, RMSE; Classification → Accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, predicted-class counts, one-vs-rest ROC-AUC/PR-AUC, NLL, and multiclass Brier score. The Phase 2 classification focus is imbalance and collapse rather than confidence calibration.
  - Comparison axes:
    - Benchmarks (end-to-end, task-specific) vs. framework (frozen encoder + MLP head)
    - Single-branch ablations vs. full aggregated framework
    - Transferability: embeddings trained on one timeframe evaluated on another without retraining

- **Phase 2 decoder refinement:**

  Keep the five Phase-1 branches frozen and compare `D0` shallow MLP, `D1`
  branch-aware residual MLP, `D2` gated fusion, `D3` compact temporal LSTM,
  and `D4` compact temporal Transformer. `D3` and `D4` consume the same `K=8`
  contract-local representation sequences; D0--D2 use their identical final
  target rows. Price and volatility are the first execution stage, while
  movement classification is a later P2-only extension outside the Part-3
  launcher. See
  `docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md` for the exact
  architecture and artifact contract. The Stage-1 data preparation, D0--D4
  models, trainer, bootstrapper, reporting path, and tests are implemented; the
  frozen CUDA matrix is partially executed; 14 of 30 trajectories currently
  have complete sweep artifacts, with aggregate reporting deferred until the
  remaining runs finish.


### Additional Alpha-Research Capability

Alpha research is a future downstream capability test outside the current task-evaluation budget, not an additional representation branch or the framework's central contribution. The task heads transform the shared representation \(z\) into economically named predictions; these predictions, rather than arbitrary latent coordinates \(z_j\), are the terminals for a deliberately small symbolic search:

\[
X \rightarrow z \rightarrow \{\hat r,\hat\sigma,p_{bull},p_{bear},\ldots\}
\rightarrow \mathcal F_0 \rightarrow \{\alpha_1,\ldots,\alpha_K\}.
\]

The initial primitive set contains predicted return/price movement, volatility, trend probabilities, and confidence margins such as \(p_{bull}-p_{bear}\). A shallow GP grammar may compose them with protected arithmetic, ranking, delay, and bounded rolling operators. Formula selection must use chronological out-of-fold predictions on the training portion only, with target-horizon purge/embargo where needed. Once the current test split has been used for task evaluation, future factor evaluation requires a fresh holdout or temporally later data. Evaluate factor IC/rank-IC, stability, quantile spreads, and redundancy; do not claim tradable profitability without a cost-aware, contract-aware backtest.
