# Research Plan

This document outlines the four-stage research plan for developing and evaluating the unified representation learning framework.

**Progress tracking note:** This document is the stable research roadmap. It should change when the project direction, planned stages, comparison scope, task definitions, or evaluation methodology changes. Routine implementation status, checkpoint evidence, experiment observations, and next-action tracking belong in `docs/schedule.md` and the corresponding experiment reports.

**Experiment-phase transition (2026-09-21):** Numbered Phase 4 concluded the
data-selection and exploratory-analysis loop without launching new models.
Numbered Phase 5 will implement the two fresh top-50 recent one-hour FinData
walks, retrain walk-specific canonical five-branch encoders, evaluate shared
two-hour movement regression/classification targets, and run matched baselines.
This is a natural revision of the initial roadmap after issues discovered in
Phases 1--3; Phase 4 was the data-analysis phase. The authoritative contract is
`docs/phase_plan/2026-09-21-phase-5-experiment-plan.md`. The four stable
research stages below remain broad workstreams rather than experiment-phase
specifications.

---

## Stage 1 — Data Collection and Processing

- Collect historical OHLCV data from *Polymarket* event prediction markets.
  - Select top-*K* most active contracts per timeframe (1-hour, 4-hour, 1-day) by trading volume.

- Clean and standardise each contract's time series:
  - Handle missing timestamps and volume gaps via forward-fill and interpolation.
  - Establish the chronological raw-time boundary before fitting any imputer or
    scaler.
  - Fit volume normalisation and any data-dependent preprocessing on permitted
    training history only, then apply the frozen rule causally; retain raw OHLC
    prices (bounded to [0, 1] probability scale).

- Segment into fixed-length sequences using a sliding window:
  - Apply sliding window of length `seq_len` to produce sequences of shape `[seq_len, features]`.
  - Construct train/test windows only after the raw chronological boundary is
    fixed, under an explicit historical-context policy that prevents any test
    observation from entering a training sample.
  - Concatenate sequences across all selected contracts to form the final training and test sets.

- For walk-forward evaluation, replace the single final-20% lifecycle tail with two
  predeclared fixed-duration rolling global calendar-time walks. Each walk fits
  preprocessing and constructs windows using only information available before
  one shared cutoff across contracts; later-walk history must never enter an
  earlier model. Report early/middle/late contract lifecycle strata inside
  each calendar interval rather than using per-contract fractions as the
  primary split.

- Phase 5 executable orchestration is isolated under `scripts_v3/`, with
  reusable walk preparation under `src/data_processing/` and generated
  experiment artifacts under `experiments/phase5/`. The implemented data
  bundles keep encoder-training rows separate from mature supervised rows and
  freeze shared framework/baseline identities before any model launch.

- Continue exploratory analysis of distributions, volatility regimes, and
  event-driven price jumps. The targeted top-50 4-hour lifecycle diagnostic is
  complete: later contract stages are more persistent and boundary-concentrated,
  and saved feature branches encode lifecycle state. Broader 1-hour/1-day and
  event-level analysis remains pending; see
  `docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md`.

- Maintain recent external acquisition separately from experiment processing.
  The expanded FinData audit covers December 2025 through August 2026 over 50
  retrospectively selected markets. Condition-level candles can mix YES/NO
  prices. A forward-confirmed, raw-preserving quarantine retains persistent
  crashes and 99.9644% of 15-minute rows plus 99.8634% of hourly rows under a
  1% fail-closed budget. Phase 5 assumes the approved final pruning is correct
  retrospective cleaning and therefore does not replay decisions by their
  availability timestamps. The safe
  token-identified fallback returned 28,359 YES trades from 12 conditions but
  only 448 complete four-hour `seq64+h2` rows from two related contracts. It
  remains source-feasibility evidence only; see
  `docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md`.
  The completed native-frequency audit selects the clean one-hour series for
  the Phase 5 training loop. Exactly one missing hourly bar may be
  filled causally with flat OHLC, zero volume, and explicit imputation/time-
  since-observation metadata; longer gaps split sequences and observed
  decision/target endpoints remain primary. Native 15-minute data is retained
  as a resolution sensitivity because its lower coverage and stronger
  fill-induced staleness make it less suitable as the primary source. The
  condition-candle token-identity limitation and an affected-contract
  exclusion sensitivity remain mandatory; see the
  [Phase 4 conclusion](phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md).

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

- **BYOL Encoder**:
  - CNN online/target encoder with projector and predictor heads; trained by bootstrap prediction on augmented view pairs.
  - Target encoder is updated by exponential moving average; pretrain on training sequences and freeze the online backbone for downstream use.

- **Phase 2 temporal backbone refinement**:
  - Treat contrastive and BYOL as selected peer feature branches. Within each branch's own SSL objective, keep augmentations, projector/predictor semantics, 128-dimensional downstream output, data, budgets, and shallow probing contract fixed while comparing its immutable CNN reference with named LSTM and compact-Transformer substitutions.
  - Store `contrastive_lstm`, `contrastive_transformer`, `byol_lstm`, and `byol_transformer` as separate experimental branch artifacts. Replace only the corresponding CNN branch in each primary five-branch comparison rather than increasing the branch count or changing both neural branches together.

- **Additional methods (TBD)** — candidates include masked autoencoders, self-supervised Transformer encoders, or other self-supervised objectives identified during the literature review. Each new encoder registers a new key in the aggregator's `branch_dims` without requiring any changes to existing components.

Frozen neural embeddings are stored as separate named feature arrays, not as one packed neural matrix. This preserves branch identity for concat aggregation, gated aggregation, and single-branch ablations.

Under Phase 5, the primary evaluation trains separate canonical VAE,
contrastive-CNN, and BYOL-CNN weights at each global calendar cutoff, freezes
them, and trains that walk's downstream heads on embeddings from the same
history. Encoder variants, fixed-first-walk transfer, gating, and branch
ablations move to Phase 6. No Phase 3 encoder weights are Phase 5 inputs.

The Phase 5 core does not include a representation-transfer comparison.
Fixed-first-walk reuse, lifecycle-conditioned shared models, stage-specific
experts, encoder variants, and branch/fusion ablations are deferred to Phase 6.
Representation drift alone is not evidence that a different architecture is
needed in each lifecycle stage.

### 2.4 Representation Aggregation and Downstream Task Training

- Implement `RepresentationAggregator` — a flexible N-branch fusion module that accepts an arbitrary set of named branches via a `dict[str, Tensor]` API.
  - **Concat mode** (default): branches are concatenated into a single higher-dimensional vector; no learnable parameters in the aggregator itself. Output dimension equals the sum of all branch dimensions.
  - **Gated mode**: each branch is projected to a shared `out_dim`, then a gating network produces per-branch softmax weights. Output dimension equals `out_dim`.
  - The `output_dim` property returns the correct task-head input size regardless of mode.
  - Default branch set: `statistical` (70-d), `transformed` (55-d), `vae` (64-d), `contrastive` (128-d), `byol` (128-d). Adding a new neural encoder requires only registering a new key in `branch_dims`.
  - Concat serves as the primary implementation and as an ablation baseline for gated mode.

- Train the aggregator jointly with each downstream task head using supervised task losses:
  - Probability-movement regression — MLP regressor for continuous
    contract-local `close[t+h] - close[t]`; absolute next-close regression is
    retained only as historical/negative characterisation evidence.
  - Volatility prediction — MLP regressor, MSE loss on realised volatility targets.
  - Trend classification — MLP classifier trained with cross-entropy on TA-MLP-style tri-class BUY/HOLD/SELL labels.

- Write end-to-end training scripts connecting data loading, feature extraction, encoder inference, aggregation, and task training.

---

## Stage 3 — Baseline and Benchmark Implementation

All comparison models must be trained on the **same data splits and preprocessing** as the framework to ensure fair comparison. Strict task comparisons must also use the same target definition and aligned label rows. The exact set of models is provisional and will be finalised once the literature review is complete.

Two categories of comparison models are used:

| Term | Definition |
|---|---|
| **External benchmark** | Model from prior work (task-specific, end-to-end trained). Shows the framework is competitive with the state-of-the-art. |
| **Internal baseline** | Model designed within this project. Shows each framework component contributes. |

**External benchmarks:**

- **Stacked LSTM** — 3-layer LSTM trained directly on raw OHLCV sequences as the primary external benchmark for price prediction.
- **Raw LSTM volatility** — LSTM trained directly on raw OHLCV sequences and the shared realised-volatility label bundle. This is the direct end-to-end neural benchmark for volatility prediction.
- **Adapted GARCH--LSTM stacking** — paper-inspired parallel hybrid for volatility prediction. Causal guarded GARCH forecasts and Raw LSTM forecasts are fused with fixed ElasticNet meta-features `[g, l, g*l]` using train-only expanding OOF features. It complements, rather than replaces, the direct Raw LSTM benchmark: the former tests a task-specific hybrid and the latter tests direct end-to-end sequence prediction.
- **GINN** *(AR→GARCH→LSTM with fused loss)* — retained as volatility limitation evidence after the initial run exposed an implausibly scaled GARCH target failure; it is no longer the planned headline volatility comparison.
- **TA-MLP** *(Parente et al., 2024 / FreqTrade-based)* — 4-layer LeakyReLU MLP trained on 36 TA-Lib technical indicator features (RSI, Bollinger Bands, candlestick patterns, etc.). Primary benchmark for the trend classification task. Labels follow the paper's tri-class BUY/HOLD/SELL formulation (`src/baselines/ta_mlp_baseline/ta_labels.py`); thresholds are quantiles of `|pct_change|` fit per contract on training rows only. The paper reports random majority-`HOLD` undersampling, while the existing repository v1 sweep used natural sampling and is therefore an adaptation rather than an exact reproduction. Strict framework-vs-TA-MLP comparison should reuse the saved task label bundle and explicitly name the training-only sampling protocol so rows, thresholds, class definitions, and imbalance treatment are traceable. For the Phase 2 movement-label task, candidate protocols are majority undersampling (`P1U`), balanced oversampling (`P1O`), and logit-adjusted cross-entropy (`P2`); `P2` is fixed for architecture comparisons, while natural cross-entropy (`P0`) is an untreated reference only.
- **Additional benchmarks (TBD)** — further models may be added based on the literature review.

**Internal baselines:**

- **Raw-OHLCV MLP** — 5-layer MLP trained directly on flattened OHLCV sequences with no representation learning; serves as the minimum competence reference. Its existing volatility sweep predates the contract-aware realised-volatility bundle and is characterization evidence only; it must be migrated to the shared bundle before strict volatility comparison.

- **Single-branch ablations** — run each active representation branch independently (no aggregation) through the same task heads. Will include at minimum:
  - Statistical-only (AR + GARCH features)
  - Transformation-only (FFT + Wavelet features)
  - VAE-only (latent embeddings from the pretrained VAE)
  - Contrastive-only (embeddings from the pretrained contrastive encoder)
  - BYOL-only (embeddings from the pretrained BYOL encoder)
  - One ablation per additional neural encoder that is integrated (TBD)

  These ablations isolate each branch's individual contribution and verify that the aggregated framework outperforms any single branch.

- **Additional internal baselines (TBD)** — further baselines may be added as identified.

---

## Stage 4 — Experiments and Benchmarking

The framework is evaluated using **probing**: frozen multi-branch encoders + a lightweight MLP task head trained on extracted features. Keeping the task head simple is intentional — representation quality, not decoder complexity, should drive performance.

Phase 3 concluded after the leakage-safe encoder and framework next-close
matrix. Its test tail was dominated by near-settlement persistence, and the
causal no-change reference substantially outperformed every framework row.
Phase 4 therefore concluded by selecting continuous probability movement,
global calendar-time walks, and bounded-forward-filled recent one-hour inputs
for the next experimental loop. Phase 5 will execute that loop with performance
stratified by contract lifecycle. This prevents cross-contract calendar
lookahead in the pooled representation model. Movement classification remains
a related but distinct directional task, and conventional arithmetic-return
regression remains a secondary exploratory target because low prices strongly
distort its scale.

The executed Phase 5 exploratory regression sensitivities additionally test
eight-hour raw probability change and two-hour ordinary log return. They are
diagnostic horizon/target-unit probes rather than replacements selected from
evaluation performance; neither recovered stable signed correlation.

Phase 2 contains three separate experiment parts whose effects must not be
mixed in the first comparison: (1) decoder refinement with the Phase-1
encoders fixed, (2) encoder refinement through matched new temporal-backbone
variants with the shallow decoder fixed, and (3) probability-movement
classification relabelling. The Part 3 launcher is restricted to the strict
TA-aligned Raw-OHLCV MLP, five-branch framework, and adapted TA-MLP matrix;
Parts 1 and 2 use separate experiment roots.

The canonical specifications are `docs/phase_plan/2026-09-08-phase-2-experiment-plan.md`
for the complete three-part programme,
`docs/phase_plan/2026-09-14-phase-2-decoder-refinement.md` for the frozen D0--D4
implementation contract, and
`docs/phase_plan/2026-09-08-phase-2-probabilistic-classification.md` for the
Part 3 execution contract.

- Evaluate all models (framework, benchmarks, internal baselines) on identical
  global-calendar walk identities using consistent metrics:
  - Probability-movement regression: MAE, RMSE/MSE, Pearson and Spearman
    correlation, sign agreement, per-contract, per-global-walk, and
    per-lifecycle-stage metrics, with exact zero movement as the primary
    reference
  - Volatility prediction: MSE, Pearson correlation of predicted vs. realised volatility
  - Trend classification: Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix. Accuracy is reported as a supporting metric because the HOLD class can dominate.

- Reuse saved task-label bundles and their aligned rows whenever a task has
  one. The Phase 3 horizon-1 absolute-price bundle remains immutable negative
  characterisation evidence. Phase 5 must create a new fold-aware continuous
  probability-movement bundle whose horizons never cross fold or contract
  boundaries. Raw LSTM, GARCH--LSTM stacking, the future framework volatility
  run, and the Raw-OHLCV MLP volatility rerun must consume the same
  contract-aware realised-volatility bundle.

- For volatility, retain both the Raw LSTM and the adapted GARCH--LSTM stack in the final table. Beating or approaching Raw LSTM indicates competitiveness with direct neural sequence prediction; beating or approaching the stack is stronger hybrid-comparator evidence. The stack comparison must be described as a complete-system comparison, not a standalone-GARCH result.

- **Phase 2 decoder refinement:** keep all five Phase-1 branches frozen and
  compare `D0` shallow MLP, `D1` branch-aware residual MLP, `D2` gated fusion,
  `D3` temporal LSTM, and `D4` temporal Transformer on identical eligible rows.
  Price and volatility are the first execution stage; movement classification
  is a later P2-only extension kept separate from the Part-3 C1/C2/C5 matrix.
  The exact architecture, `K=8`, seeds, budgets, row-map, and replay contract
  are frozen in the dedicated decoder specification.

- **Transferability analysis** — evaluate whether embeddings trained on one subset of tasks or markets transfer effectively to held-out tasks, contract types, or timeframes without retraining.

- **Ablation study** — compare the full aggregated framework against each single-branch baseline to quantify each branch's marginal contribution.

- **Additional alpha-research downstream capability (deferred beyond the current task-evaluation budget)** — a future extension may test whether interpretable formulaic factors can be composed from downstream predictions rather than latent dimensions. The representation-learning framework remains the contribution; GP/symbolic regression is a small-scale established search tool, not a claimed algorithmic novelty.
  - Primitive set \(\mathcal F_0\): predeclared downstream outputs available at decision time, initially predicted return/price movement, predicted realised volatility, trend probabilities, and confidence margins such as \(p_{bull}-p_{bear}\). Multiple horizons are optional and must use split-safe targets.
  - Before implementation, lock whether the directional primitive is future probability change or return, and make its horizon, eligible contract universe, and factor objective consistent. A price-level forecast is not a directly comparable cross-contract factor.
  - Exclude raw embedding coordinates \(z_j\) as GP terminals because they have no guaranteed individual financial interpretation.
  - Use a shallow, bounded grammar (protected arithmetic, ranks, delays, rolling statistics/time-series ranks) and predeclare depth/window/population/generation limits.
  - Build training primitives with chronological OOF predictions from heads that did not train on the predicted rows. Search and select formulas only on those OOF training rows; retain a small non-redundant set by predeclared IC/stability criteria. If this future extension is funded, refit heads on full training data and evaluate factors once on a fresh holdout or temporally later data, not on the current task-evaluation test split.
  - Report IC/rank-IC, temporal stability, quantile/spread monotonicity, and factor redundancy. A trading backtest is outside the core scope unless contract mechanics, fees, liquidity, and position constraints are explicitly modelled.

- Summarise all results in tables and visualisations (embedding scatter plots, metric comparisons, gating weight distributions).

---

## Final Step — Documentation and Reporting

- Write final report covering: motivation, related work, model design, experimental results, analysis of limitations, and future directions.
- Prepare codebase documentation and ensure reproducibility.
