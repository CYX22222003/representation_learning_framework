# Model Design

> **Phase 5 authority (2026-09-21):** The initial architecture and experiment
> design evolved after issues discovered in Phases 1--3; Phase 4 was a
> data-analysis phase and ran no model training. For the active main-experiment
> data, tasks, assumptions, and scope, read
> [`phase_plan/2026-09-21-phase-5-experiment-plan.md`](phase_plan/2026-09-21-phase-5-experiment-plan.md)
> first. It supersedes conflicting older Phase 5 handoff language below.
>
> **Phase 6 authority (2026-09-22):** For the current future-interval realised-
> variance task and temporal encoder matrix, read
> [`phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md`](phase_plan/2026-09-22-phase-6-volatility-forecasting-plan.md)
> and
> [`phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`](phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md).
> The primary future-realised-variance horizon is frozen to eight hours by
> [`phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`](phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md).
> The completed seed-0 matrix and its claim boundaries are interpreted in
> [`phase_plan/2026-09-26-phase-6-experiment-observation-and-outcomes.md`](phase_plan/2026-09-26-phase-6-experiment-observation-and-outcomes.md).
> These documents supersede conflicting Phase 6 scope statements below.
>
> **Next-scope authority (amended 2026-10-09):** Phase 6.5 LSTM capacity,
> strict adapted GARCH--LSTM, current-task TA-MLP classification, and price-
> focused residual-CNN SSL encoders are frozen in
> [`phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`](phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md).
> The matched recent frozen-representation comparison is closed in
> [`phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md`](phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md).
> It freezes mandatory LWA-Frozen and SaURL-TS-Frozen across the same three
> tasks and two walks as canonical `H0`; the commissioned TimeDART-Frozen
> extension is also complete. SISSEL-Frozen remains uncommissioned optional
> scope and was not a phase exit condition. The optional Phase 6.8 extension is
> frozen in
> [`phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`](phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md).
> It adds Di-COT-Frozen and Monotone-VI-Frozen under the same target-free,
> two-walk, three-task common-probe design. Its roster/comparison contract is
> frozen, but implementation and execution have not started. Price-focused
> Raw-OHLCV/`H0` residual fusion, a source-aligned recent
> xLSTM-Mixer forecasting candidate, and canonical decoder-capacity sensitivity
> are frozen separately in
> [`phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md`](phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md),
> with grouped SHAP deferred to a later analysis amendment. Phase 6.9 is the
> current planned handoff in
> [`phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`](phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md).
> It uses classification-only Monotone-VI, a Phase 6.9-owned xLSTM-Mixer
> contract for price, and the completed strict GARCH--LSTM stack for
> volatility. The Phase 6.6 raw-fusion and decoder-capacity studies remain
> deferred; their former xLSTM-Mixer listing is superseded.
> Canonical single-branch and leave-one-out attribution is frozen in
> [`phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`](phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md).
> Phase 6.5A--D are executed for their frozen seed-0 scopes. Phase 6.5D's four
> residual-CNN encoders, two feature stores, eight price trajectories, CKA,
> resources, subgroup tables, and complete report are replay-valid. Phase 6.7
> now has a tested SaURL-TS model plus its gated pretraining, frozen-feature,
> replay, and common-probe infrastructure. Its CUDA/resource gate and both
> walk-specific 50-epoch pretraining trajectories are complete and replay-
> valid. Both master stores and all six staged SaURL probes are also complete
> and replay-valid; SaURL is generally weaker than H0, with only isolated
> metric-specific improvements. LWA's paper/source audit and eight-document
> independent-adaptation dossier are complete under `docs/baselines/LWA/`,
> and all twelve adaptation decisions are approved. Its Lumid/CUDA runtime,
> pinned `PyWavelets==1.8.0` CWT path, model, both caches, both 50+50
> trajectories, both 384-wide stores, and all six downstream runs are complete
> and replay-valid. The integrated Phase 6.7 epoch-50 comparison is under
> `experiments/phase6_7/reports/frozen_representation_seed0/summary.md`.
> TimeDART execution is documented under
> `docs/baselines/TimeDART/`: its source/design audit, all eleven owner
> decisions, independent model, focused CPU tests, two encoder trajectories,
> two 170-wide stores, six downstream trajectories, and 18 snapshots are
> complete and replay-valid. SISSEL remains uncommissioned optional scope.
> Phase 6.9 is the current planned handoff. Optional Phase 6.8 and frozen
> Phase 7A remain unimplemented; Phase 6.6 likewise remains planning evidence.
> The xLSTM-Mixer paper/source audit is complete under
> `docs/baselines/xLSTM-Mixer/`; all owner decisions are complete, including
> one learned initial token, while the common-path artifact and Lumid runtime
> admission remain pending. Phase 7B alpha research remains
> unspecified pending further literature review.

## Architecture Design

The proposed model processes raw OHLCV time-series data through an extensible set of named representation branches: a `statistical` branch (AR and GARCH features), a `transformed` branch (FFT and Haar wavelet features), and neural branches (`vae`, `contrastive`, and `byol`; additional unsupervised methods may be added). These representations are fused by a `RepresentationAggregator` into a unified embedding *h_i*, which is passed to lightweight MLP task heads for three downstream tasks: probability-movement regression, volatility prediction, and movement/trend classification. Absolute next-close prediction is retained as a completed negative characterisation study rather than the primary regression probe.

The aggregator supports two fusion modes. In *concat mode* (default), branches are concatenated into a single higher-dimensional vector with no learnable parameters; the task head absorbs all supervised learning. In *gated mode*, each branch is projected to a shared dimension and a gating network produces per-branch softmax weights. Gated mode remains an implemented historical comparison capability, but it is not part of the active Phase 6 matrix. A detailed architecture diagram is provided in the Appendix.

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

> **Implementation location:** Phase 5 reusable train/test construction lives
> in `src/data_processing/phase5_walks.py`, thin executable entry points live
> in `scripts_v3/`, and generated experiment data lives under
> `experiments/phase5/data_preparation/`. `scripts_v2/` remains the earlier
> experiment and acquisition generation.

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
  - **Phase 2 temporal variants** — contrastive and BYOL are selected peer branches. Each has named LSTM and compact-Transformer substitutions that preserve its own SSL objective, augmentations, and projector/predictor contract and expose a 128-dimensional frozen backbone state. A primary comparison replaces exactly one corresponding CNN branch; it never adds a sixth branch or changes both neural branches together. The active Phase 6 implementation is described in detail in [`phase6_temporal_encoder_architecture.md`](phase6_temporal_encoder_architecture.md).
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

- Phase 6 completed walk-specific LSTM/Transformer substitutions and
  heterogeneous single additions under concat, with duplicated-CNN width
  controls. Phase 6.5 completed one two-layer LSTM capacity candidate per SSL
  family, both strict GARCH--LSTM stacks, and the ten-run current-task TA-MLP
  comparison. Phase 6.5B/C artifacts and replay-valid 5/15/50 snapshots live
  under `experiments/phase6_5/`; their compact principal-result report is
  generated, while expanded contract-macro/subgroup reporting remains.
  Phase 6.5D's deeper residual-CNN candidates under both SSL families and all
  eight future-price trajectories are complete and replay-valid. Contrastive
  residual variants reduce price error against H0 in both walks, while BYOL
  does not improve consistently and movement ranking remains weaker. Phase
  6.7 completed the canonical H0 comparison with three recent target-free
  frozen representations under the same simple heads and three tasks. Phase
  Optional Phase 6.8 adds Di-COT-Frozen and Monotone-VI-Frozen through that
  same probing boundary after method-specific source/licence/adaptation gates.
  Phase 6.9 instead adds a task-specific comparison for classification, price,
  and volatility. Phase 7A separately runs canonical single/leave-one-out
  attribution. The otherwise deferred Phase 6.6
  retains a matched frozen-H0/raw-sequence residual fusion system and two
  richer static canonical heads on future price only. Phase 6.9 owns the
  source-aligned xLSTM-Mixer full-path benchmark. Its additional supervision
  makes it a contextual complete-system comparison. The completed simple head
  remains the primary representation probe. Lifecycle conditioning,
  stage-specific experts, temporal decoder variants, fixed-first-walk
  transfer, and additional seeds remain outside these plans.

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
| **External representation baseline** | Target-free prior-work representation frozen before the shared lightweight probes. Completed Phase 6.7: required LWA-Frozen and SaURL-TS-Frozen plus optional TimeDART-Frozen. Optional planned Phase 6.8: Di-COT-Frozen and Monotone-VI-Frozen. SISSEL-Frozen remains uncommissioned optional scope. |
| **Task-specific external benchmark** | End-to-end or paper-inspired comparator selected for one downstream task. Phase 6.9 uses a classification-oriented Monotone-VI adaptation, xLSTM-Mixer for future price, and the completed adapted GARCH--LSTM stack for volatility. Other context includes Stacked LSTM, Raw LSTM volatility, GINN limitation evidence, and TA-MLP. |
| **Internal baseline** | Model designed within this project (Raw-OHLCV MLP, single-branch ablations). Shows each framework component contributes. |
| **Default decoder** | Task head (`PriceRegressor`, `VolatilityRegressor`, `TrendClassifier`) — simple MLP from `src/tasks/`. Used by the framework and all internal baselines. |
| **Refined decoder** | A controlled downstream-capacity variant trained on unchanged frozen features. Phase 6.6C freezes static `D1-RP` residual projection and `D2-BG` branch-gated projection heads; earlier Phase-2 recurrent/attention variants remain historical scope. |

**Evaluation paradigm (probing):** The framework and Phase 6.7/6.8 external
representation methods use frozen extractors or fitted representations. After fitting, the features
remain fixed while the same lightweight task-head family is trained for each
task. Keeping the head simple is intentional: the primary comparison concerns
representation quality. Task-specific end-to-end benchmarks have more
optimisation freedom and remain contextual complete-system comparisons.

- **Benchmark Retraining:** Each benchmark model is retrained on the same event
  prediction market data using the same sliding-window identities and global
  calendar walk. Framework encoders, heads, and baselines all obey the same
  cutoff. This project does not use a validation split or early stopping; see
  `docs/training_test_data_selection.md`.
  The implemented Phase 5 baseline matrix uses a flattened Raw-OHLCV MLP and
  a three-layer five-channel raw OHLCV LSTM on the exact h2/h8 rows for the
  regression, classification, and future-price tasks. Its 12 seed-0
  trajectories are complete and replay-validated; see
  `phase_plan/2026-09-22-phase-5-baseline-amendment.md`.

- **Classification benchmark adaptation:** Phase 6.5C restores the fixed
  36-feature `36-128-64-32-3` TA-MLP on the current h2/tau=0.001
  `DOWN/STABLE/UP` task. A causal feature-availability intersection is shared
  by retrained H0, Raw MLP, Raw LSTM, and TA-MLP `P2` runs. The paper-derived
  majority-undersampling rule is a separate TA-only training sensitivity; the
  natural evaluation distribution remains untouched. Legacy BUY/HOLD/SELL
  metrics are not current-task evidence.

- **Recent frozen-representation comparison:** Phase 6.7 pretrained each
  admitted external encoder separately on the two target-free walk
  populations, freezes epoch-50 embeddings, and uses the established simple
  heads on the exact movement-classification, future-price, and future-RV
  rows. The core matrix has 12 new trajectories and six immutable `H0`
  references. The admitted TimeDART extension added six trajectories; SISSEL
  remains optional and uncommissioned. Native widths, parameters, time,
  memory, source deviations, and negative results are preserved. No evaluation
  metric selected an optional method's extraction point, recipe, checkpoint,
  or matrix truncation.

- **Optional recent conference extension:** Phase 6.8 keeps Phase 6.7 closed and adds
  Di-COT-Frozen (ICML 2026) plus Monotone-VI-Frozen (ICLR 2025). Both fit
  separately on each unchanged target-free walk population, return one native-
  width embedding for every established row, and use the exact common probes.
  The planned inventory is four representation fits, four stores, 12 probes,
  and 36 downstream snapshots beside six immutable `H0` references. Di-COT
  and Monotone-VI are both treated as recent representation-learning
  baselines; neither is presumed to be an empirical SOTA winner.

- **Task-specific competitiveness demonstration:** Phase 6.9 compares the
  same canonical `H0` system separately with a source-aligned comparator for
  each task. Monotone-VI is classification-only and must pass an inductive
  train-only embedding gate before it uses the common simple classifier probe.
  xLSTM-Mixer supplies the future-price complete-system comparison under the
  Phase 6.9 full-path contract. The replay-valid strict GARCH--LSTM
  stack supplies the volatility comparison without retraining. These rows
  support task-level competitiveness claims, not one homogeneous cross-task
  architecture ranking.

- **Deferred decoder-capacity sensitivity:** Phase 6.6C keeps canonical frozen H0
  features, future-price rows, scaler, target, loss, and output transform fixed.
  It compares the immutable simple probe with a residual projection head and
  a branch-aware gated projection head across both walks. The gated row is a
  complete-system sensitivity because it changes both supervised capacity and
  fusion.

- **Deferred raw/representation fusion sensitivity:** Phase 6.6A projects frozen H0 to
  width 128 and optionally adds a learned residual correction from an LSTM or
  historical-context-only BiLSTM over the matching 64-by-5 OHLCV sequence.
  `F-H0` is the matched supervised-capacity control, while `F-RL` is required
  to assess whether H0 adds value beyond raw temporal modelling. Phase 6.5D
  separately tests deeper residual-CNN SSL encoders with the simple future-
  price probe. Grouped SHAP is post-hoc description only and cannot select
  either matrix.

- **Phase 6.9 recent multivariate forecasting benchmark:** Phase 6.9 evaluates the
  NeurIPS 2025 xLSTM-Mixer as a source-aligned complete system. It maps the
  same 64-hour five-channel context to the full next-eight-hour OHLCV path and
  extracts `close[t+8]` for the existing price metrics. A deterministic
  metadata join freezes the known reduced common row contract first; matched
  H0 and Raw LSTM comparator reruns use that same intersection. Its five-channel,
  eight-horizon supervision is reported explicitly and is not treated as a
  target-matched decoder or representation contrast. This candidate is
  resolved for the Phase 6.9 price leg before any result is reported. If the
  broader Phase 6.6 programme later resumes, it cannot rerun or own this
  method; xLSTM-Mixer remains outside the direct
  representation-comparison claim. The completed dossier at
  `docs/baselines/xLSTM-Mixer/` shows that the released second view flips the
  latent feature axis rather than variate order and that paper/source RevIN,
  token-count, release, packaging, and dependency details differ. The phrase
  “source-aligned” therefore means the approved pinned-source core plus the
  disclosed project protocol. All owner decisions are complete; implementation
  has not started.

- **Volatility benchmark adaptation:** The historical four-hour volatility bundle is an overlapping shifted-window proxy and remains characterisation evidence only. Phase 6 completed the strict comparison on a walk-specific shared bundle of eight-hour realised variance over the strictly future interval `(t,t+8h]` from observed raw probability changes. H=8 was frozen from the training-period-only audit before label construction. Raw LSTM volatility is the direct end-to-end neural benchmark. The strict adapted GARCH--LSTM Phase 6.5 complementary hybrid is now trained and replay-valid for both walks: it fuses causal guarded raw-change GARCH forecasts with matched Raw LSTM forecasts through fixed ElasticNet meta-features `[g, l,g*l]`. Its expanding cross-fitting is used only to create out-of-fold training features for the meta-learner; it is not validation or model selection. Epoch-50 MSE improves only marginally while MAE and Spearman worsen in both walks, so the evidence does not support broad hybrid superiority. The Raw-OHLCV MLP, canonical framework, temporal configurations, and stack consume the identical replacement evaluation rows. See `docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md` and `docs/phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`.

- **Embedding-based Model Training:**
  - Deterministic branches (statistical, transformed) require no training; neural branches are pretrained unsupervised and their encoder weights are frozen.
  - All branch embeddings are extracted into a branch-aware `FeatureBundle`: deterministic arrays are saved as `statistical` and `transformed`, and neural embeddings are saved under their encoder names such as `vae`, `contrastive`, and `byol`. The `RepresentationAggregator` receives these named branch tensors and fuses them into a unified embedding *h_i* per sequence.
  - **Downstream Task Preparation:**
    - **Regression Task:** supervised pairs (X, y), where X is the sequence
      embedding. The clearest Phase 5 transfer target is the sigmoid-bounded
      eight-hour future probability `close[t+8h]`. Its implied movement is
      evaluated with Pearson/Spearman, sign agreement, and cross-sectional
      Rank IC. Direct raw-change and log-return heads remain aligned diagnostic
      tasks showing that target formulation changes what the frozen
      representation exposes. Labels are contract- and fold-local.
    - **Classification Task:** labels such as trend direction or event outcome mapped to embeddings as input-output pairs. Phase 1 retains its TA-MLP-style tri-class BUY/HOLD/SELL bundle. The isolated Phase 2 task uses hard `DOWN/STABLE/UP` labels from absolute probability movement over a split-safe horizon and saves three-class scores. Its candidate imbalance protocols are majority undersampling (`P1U`), balanced oversampling (`P1O`), and train-prior logit-adjusted cross-entropy (`P2`); natural cross-entropy (`P0`) is an untreated reference only.
  - A lightweight MLP task head is trained on these (X, y) pairs.

- **Performance Comparison:**
  - Evaluate strict task comparisons on the same global-calendar walk identities
    and aligned label rows, then stratify results by contract lifecycle stage.
  - Consistent metrics: Regression → MAE, RMSE, Pearson/Spearman, implied-
    movement Rank IC and sign agreement; Classification → Accuracy, macro-F1,
    balanced accuracy, per-class precision/recall/F1, confusion matrix,
    predicted-class counts, one-vs-rest ROC-AUC/PR-AUC, NLL, and multiclass
    Brier score. The Phase 2 classification focus is imbalance and collapse
    rather than confidence calibration.
  - Broad comparison axes, only when separately approved for the applicable
    phase:
    - Benchmarks (end-to-end, task-specific) vs. framework (frozen encoder + MLP head)
    - Canonical single-branch and leave-one-branch-out ablations vs. the full
      aggregated framework
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
