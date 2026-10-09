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

**Current transition (2026-10-09):** Phase 6.7 is closed after comparing
canonical `H0` with required LWA/SaURL and optional TimeDART under the same
three tasks, two walks, and lightweight heads. SISSEL remains optional and
uncommissioned. Phase 6.9 is the current approved planning handoff and adds a
task-specific competitiveness demonstration: Monotone-VI for classification,
xLSTM-Mixer for future price, and the completed strict GARCH--LSTM stack for
future realised variance. Phase 6.8 remains a frozen optional representation
extension; it is unimplemented and does not block Phase 6.9. Phase 7A remains
frozen and unimplemented. Phase 6.6 raw fusion and decoder capacity remain
deferred, while Phase 6.9 is the sole xLSTM-Mixer technical/artifact authority. The new contract is
`docs/phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md`.
The xLSTM-Mixer paper/source audit is complete under
`docs/baselines/xLSTM-Mixer/`; all owner decisions are complete, including one
learned initial token. Its model, guarded runtime/training/replay lifecycle,
and observed-path builder follow the canonical contract. The owner-directed
local WSL vanilla-GPU amendment removes Lumid as a prerequisite while
preserving selected-backend resource admission and source replay. The later
`docs/phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md` replaces
full-path supervision/intersection reruns with XM-C8: direct close[t+8h],
endpoint MSE, original price rows, and verified original control reuse.
Historical XM-MV8 results remain separate. Current execution evidence belongs in `docs/schedule.md`.

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
history. No Phase 3 encoder weights are Phase 5 inputs.

The Phase 5 core does not include a representation-transfer comparison.
Phase 6 tests walk-specific contrastive and BYOL LSTM/Transformer backbones as
fixed-width substitutions and as heterogeneous single additions under concat.
Same-width duplicated-CNN features control for the wider downstream head.
Movement classification, future price, and the independently frozen future-
realised-variance task form the downstream matrix. That seed-0 matrix is now
complete. Phase 6.5 has completed one two-layer, 128-wide LSTM capacity
candidate under both SSL families, a strict H=8 adapted GARCH--LSTM benchmark,
and a classification-only TA-MLP benchmark on the current h2/tau=0.001 task.
Phase 6.5D is also complete: four deeper residual-CNN SSL encoders, two
walk-specific feature stores, eight future-price probes, CKA, resources,
subgroups, and the complete report are replay-valid. The Contrastive
substitution and addition improve price MAE/RMSE over H0 in both walks, and
the addition beats its same-width duplicate control, but Raw LSTM and
persistence remain stronger error references and movement ranking weakens.
The simple head remains the primary representation probe. Phase 6.7 is now
complete: mandatory LWA-Frozen and SaURL-TS-Frozen and commissioned optional
TimeDART-Frozen were pretrained separately for both walks, frozen, and
evaluated with the same simple heads on movement classification, future price,
and future realised variance. SISSEL-Frozen remains optional and
uncommissioned. Optional Phase 6.8 adds Di-COT-Frozen and Monotone-VI-Frozen as
recent conference representation comparators under the same two-walk,
three-task common probes. Both are recent representation-learning baselines;
their inclusion does not assume empirical SOTA performance. Phase 6.9 instead
tests task-level competitiveness using classification-only Monotone-VI,
xLSTM-Mixer for price, and the completed GARCH--LSTM stack for volatility.
Phase 7A separately tests the canonical five branches through single-branch
and leave-one-branch-out probes. Phase 6.6 is otherwise deferred; it retains a matched supervised
fusion of canonical `H0` with raw-sequence LSTM/BiLSTM towers and two
richer static canonical decoders on eight-hour future price. XM-C8 is an
endpoint-matched xLSTM-Mixer complete-system adaptation; historical XM-MV8
had additional full-path supervision. Neither isolates reusable representation quality. Grouped
SHAP-style attribution remains deferred and cannot select the model matrix.
Fixed-first-walk reuse,
lifecycle-conditioned models, stage-specific experts, temporal decoder
variants, and additional seeds remain outside these active follow-ups. See
`phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`.
The follow-up contracts are
`phase_plan/2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md` and
`phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md` and
`phase_plan/2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md` and
`phase_plan/2026-10-09-phase-6-9-task-specific-competitiveness-plan.md` and
`phase_plan/2026-09-29-phase-6-6-raw-fusion-and-residual-cnn-plan.md` and
`phase_plan/2026-09-26-phase-7a-representation-ablation-plan.md`.
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
  - Volatility prediction — MLP regressor on future interval realised variance,
    defined from raw probability changes as
    `sum_{j=1..8} (p[t+j] - p[t+j-1])^2` over `(t,t+8h]`. The historical
    shifted-window proxy remains characterisation evidence only. The Phase 6
    eight-hour horizon was frozen from training-period capacity and target
    diagnostics before label construction or model evaluation; see
    `phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`.
  - Movement/trend classification — the current task uses two-hour
    `DOWN/STABLE/UP` labels at `tau=0.001` with train-prior logit-adjusted
    cross-entropy. The older TA-MLP-style `BUY/HOLD/SELL` task remains
    historical stock-label-transfer characterisation.

- Write end-to-end training scripts connecting data loading, feature extraction, encoder inference, aggregation, and task training.

---

## Stage 3 — Baseline and Benchmark Implementation

All comparison models must be trained on the **same data splits and preprocessing** as the framework to ensure fair comparison. Strict task comparisons must also use the same target definition and aligned label rows. The exact set of models is provisional and will be finalised once the literature review is complete.

Three categories of comparison models are used:

| Term | Definition |
|---|---|
| **External representation baseline** | Target-free method from prior work, frozen and evaluated with the common lightweight task heads. Directly tests reusable representation quality. |
| **Task-specific external benchmark** | End-to-end or paper-inspired model from prior work. Provides task-level complete-system context but does not isolate representation quality. |
| **Internal baseline** | Model designed within this project. Shows each framework component contributes. |

**External representation baselines:**

- **Learning Without Augmenting-Frozen** *(NeurIPS 2025)* — multi-domain time/Fourier/time-frequency representation evaluated through its frozen source-style extraction path and the common probes.
- **SaURL-TS-Frozen** *(Pattern Recognition 2026)* — mandatory adaptive time/frequency bootstrap representation.
- **SISSEL-Frozen** *(Information Fusion 2026, optional)* — scale-independent multi-autoencoder representation retained as uncommissioned possible future scope.
- **TimeDART-Frozen** *(ICML 2025, optional)* — completed autoregressive denoising representation extension with two encoders, two 170-wide stores, and six common-probe trajectories.
- **Di-COT-Frozen** *(ICML 2026, Phase 6.8 planned)* — direct recent
  augmentation-free temporal representation baseline, to be fitted separately
  per walk and evaluated through the unchanged common probes.
- **Monotone-VI-Frozen** *(ICLR 2025, Phase 6.8 planned)* — recent sequence
  representation-learning baseline, subject to its
  source/licence/adaptation and row-embedding feasibility gate.

The Phase 6.7 and optional planned Phase 6.8 methods use the exact existing task/walk
rows and simple-head contracts. They are project `-Frozen` adaptations unless
every source detail is reproduced; their native embedding widths and compute
costs are reported.

**Task-specific external benchmarks:**

- **Stacked LSTM** — 3-layer LSTM trained directly on raw OHLCV sequences as the primary external benchmark for price prediction.
- **Raw LSTM volatility** — LSTM trained directly on raw OHLCV sequences and the shared realised-volatility label bundle. This is the direct end-to-end neural benchmark for volatility prediction.
- **Adapted GARCH--LSTM stacking** — paper-inspired parallel hybrid for volatility prediction. Its legacy four-hour run is preserved; the strict H=8 Phase 6.5B adaptation is now complete and replay-valid for both walks. Causal guarded GARCH forecasts and Raw LSTM forecasts are fused with fixed ElasticNet meta-features `[g, l, g*l]` using train-only expanding OOF features. It complements, rather than replaces, the direct Raw LSTM benchmark: the former tests a task-specific hybrid and the latter tests direct end-to-end sequence prediction. The completed stack gives only marginal MSE gains while worsening MAE and Spearman, so it is not a broad win.
- **GINN** *(AR→GARCH→LSTM with fused loss)* — retained as volatility limitation evidence after the initial run exposed an implausibly scaled GARCH target failure; it is no longer the planned headline volatility comparison.
- **TA-MLP** *(Parente et al., 2024 / FreqTrade-based)* — 4-layer LeakyReLU MLP trained on 36 TA-Lib technical indicator features (RSI, Bollinger Bands, candlestick patterns, etc.). Primary handcrafted-feature benchmark for classification. The legacy experiment used the paper's tri-class BUY/HOLD/SELL formulation and natural sampling, so it is historical characterisation rather than a current-task comparison. Phase 6.5C instead preserves the `36 -> 128 -> 64 -> 32 -> 3` architecture while consuming the exact h2/tau=0.001 `DOWN/STABLE/UP` labels on a causal TA-feature-availability intersection. Its primary `P2` matrix retrains canonical H0, Raw MLP, Raw LSTM, and TA-MLP on identical rows with train-prior logit-adjusted cross-entropy; a separate TA-only `P1U` run applies the paper-derived majority undersampling to training rows only. This remains an adaptation rather than a reproduction of the paper's random split or model-selection procedure.
- **xLSTM-Mixer** *(NeurIPS 2025, Phase 6.9 planned)* — a Phase 6.9-owned
  endpoint-only future-price complete-system adaptation, XM-C8, with one
  close[t+8h] target and MSE on original price rows. It is not a direct
  frozen-representation comparison. The completed full-path XM-MV8 run is
  separate contextual evidence. The old Phase 6.6B listing is superseded. The paper/source
  audit and owner decisions are complete; training obeys the data/runtime
  gates in `docs/baselines/xLSTM-Mixer/`, including the owner-directed local
  vanilla-GPU execution amendment.
- **Monotone-VI classification adaptation** *(ICLR 2025, Phase 6.9 planned)* —
  classification-oriented representation baseline using the common simple
  probe on exact h2/tau=0.001 rows. Admission requires an inductive procedure
  fitted only on walk-training sequences; joint train/evaluation embedding is
  prohibited.
- **Additional benchmarks (TBD)** — deferred beyond the approved Phase 6.9
  roster; any further baseline requires a separate amendment.

**Internal baselines:**

- **Raw-OHLCV MLP** — 5-layer MLP trained directly on flattened OHLCV sequences with no representation learning; serves as the minimum competence reference. The older four-hour volatility sweep is characterisation evidence, while the strict walk-specific H=8 Phase 6 run is complete on the replacement future-interval labels.

- **Canonical branch ablations** — run each canonical representation branch
  independently and remove each branch once from `H0`, using the same task
  heads and rows. The frozen Phase 7A scope includes:

  - Statistical-only (AR + GARCH features)
  - Transformation-only (FFT + Wavelet features)
  - VAE-only (latent embeddings from the pretrained VAE)
  - Contrastive-only (embeddings from the pretrained contrastive encoder)
  - BYOL-only (embeddings from the pretrained BYOL encoder)

  The five matching leave-one-out rows remove statistical, transformed, VAE,
  contrastive CNN, and BYOL CNN in turn. These probes distinguish standalone
  usefulness from marginal usefulness in the correlated full representation;
  they do not require the full framework to beat every branch on every task.
  Temporal encoder branches are excluded because their substitution and
  addition roles were already tested in Phase 6.

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

The executed Phase 5 exploratory additional regression tasks additionally test
eight-hour raw probability change and two-hour ordinary log return. They are
diagnostic horizon/target-unit probes rather than replacements selected from
evaluation performance; neither recovered stable signed correlation.

An auxiliary Phase 5 eight-hour absolute-price probe is also complete. It
reconstructs future probability levels but remains worse than persistence on
level error. Its implied movement has positive Rank IC, though a simple
last-hour reversal score is stronger; the result motivates matched raw
temporal baselines rather than a representation-superiority claim.

The Phase 5 intermediate observation therefore promotes eight-hour future-
price prediction as the clearest regression transfer task for the final
framework narrative. Direct movement and return heads remain diagnostic
target-formulation evidence. Financial evaluation emphasizes the implied-
movement Rank IC rather than treating level error as the sole objective. The
last-hour reversal finding is a candidate empirical factor pending a fresh
holdout and cost-aware evaluation.

The matched Phase 5 comparison is complete as a 12-run seed-0 matrix: Raw-
OHLCV MLP and three-layer raw OHLCV LSTM models for two-hour movement
regression, two-hour movement classification, and eight-hour absolute future
price in each walk. Both consume the exact saved task rows and all 5/15/50
checkpoints replay. The framework leads h2 classification macro-F1, while the
raw LSTM leads h8 future-price error and implied-movement Rank IC. TA-MLP, TCN,
additional seeds, and the exploratory raw/log-return targets remain outside
this baseline stage. See
`phase_plan/2026-09-22-phase-5-baseline-amendment.md`.

Phase 6.5C adds TA-MLP back only for the current two-hour movement-
classification task. Because long-window TA indicators may exclude early
rows, it freezes a feature-availability-only common intersection and reruns
H0, Raw MLP, Raw LSTM, and TA-MLP under the matched `P2` protocol. The
paper-derived training-only undersampling variant is a separately labelled
sensitivity, not the primary architecture comparison. This matrix is now
complete and replay-valid: TA-P2 leads H0 on Walk 1 macro-F1 but trails H0 on
Walk 2, while P1U is not consistently better across walks.

Phase 6.5D tested whether a deeper residual CNN produces more useful future-
price representations under the existing Contrastive and BYOL objectives.
Its two substitutions and two additions use the simple probe on both walks.
The Contrastive result supports a narrow price-level improvement, while BYOL
is not consistently better and no movement-ranking or trading claim follows;
classification and volatility extensions remain deferred.

Phase 6.7 tested the central transfer claim against the closest recent prior
work. Mandatory LWA-Frozen and SaURL-TS-Frozen used
the same two target-free walk populations, then freeze one
embedding per row and train the established lightweight heads on the exact
classification, future-price, and future-realised-variance identities. The
primary matrix contains 12 new downstream trajectories and six immutable `H0`
references. The separately commissioned TimeDART extension added six complete
trajectories after its source, licence, extraction, and hardware feasibility
were frozen. SISSEL remains optional and uncommissioned. No core or optional
evaluation metric selected an extension recipe or truncated its frozen matrix.

Optional Phase 6.8 extends the external representation evidence with Di-COT-Frozen and
Monotone-VI-Frozen. It reuses the exact target-free walk populations, native-
width frozen stores, identical task rows, and lightweight probes. The planned
matrix contains four representation fits, four stores, 12 new downstream
trajectories, 36 retained probe snapshots, and six immutable `H0` references.
Its source/licence/adaptation and compute dossiers must be frozen before
implementation, and neither source is selected or tuned from Polymarket
evaluation results.

Phase 6.9 adds a separate task-specific competitiveness demonstration. It
does not force one comparator across unrelated tasks: Monotone-VI is adapted
for movement classification, xLSTM-Mixer is used for future price, and the
completed strict GARCH--LSTM stack is reused for future realised variance.
The `H0` representation remains the common system evaluated on all three
tasks. This separates evidence about reusable representation quality from
task-level complete-system competitiveness.

Phase 7A separately provides internal branch evidence through the
precommitted canonical single-branch and leave-one-out matrix. Phase 6.8 is
not a prerequisite for Phase 6.9, and Phase 7A is not part of Phase 6.9's exit
conditions.

The deferred Phase 6.6C separately tests whether the intentionally simple probe limits what
the canonical frozen representation can expose. It compares immutable `D0`
with a residual projection head and a branch-aware gated projection head on
eight-hour future price in both walks. These are decoder/complete-system
sensitivities; they do not replace the simple-head representation evidence or
change any encoder.

The deferred Phase 6.6A uses the same price-only boundary for matched `F-H0`, raw-LSTM,
`H0`+LSTM, and `H0`+BiLSTM systems. Classification already supplies the
clearest representation advantage, while volatility remains metric- and walk-
dependent; the new work therefore prioritises the task on which the Raw LSTM,
persistence, and reversal references leave the framework's edge least
convincing.

Phase 6.6C no longer depends on the historical Phase 6.6B listing. Phase 6.9
solely owns xLSTM-Mixer. Its endpoint-only amendment supersedes the former
observed-path intersection and control reruns. XM-C8 reuses all original
price identities, endpoint labels, and MSE, with fresh weights and verified
original controls. Freeze the recipe before endpoint evaluation; no endpoint
result may select it. Disclose that the correction follows XM-MV8 review.

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
  - Volatility prediction: MAE, RMSE/MSE, Pearson and Spearman correlation of
    predicted versus future interval realised variance, together with zero and
    historical-volatility-persistence references
  - Trend classification: Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix. Accuracy is reported as a supporting metric because the HOLD class can dominate.

- Reuse saved task-label bundles and their aligned rows whenever a task has
  one. The Phase 3 horizon-1 absolute-price bundle remains immutable negative
  characterisation evidence. Phase 5 must create a new fold-aware continuous
  probability-movement bundle whose horizons never cross fold or contract
  boundaries. The old volatility bundle is an overlapping shifted-window
  proxy. Phase 6 must create a new walk-specific future-interval realised-
  variance bundle whose component closes are observed, consecutive, contract-
  local, gap-segment-local, and mature within the applicable walk boundary.
  Raw LSTM, GARCH--LSTM stacking, the future framework volatility run, and the
  Raw-OHLCV MLP volatility rerun must consume its identical rows and targets.

- For volatility, retain both the Raw LSTM and the adapted GARCH--LSTM stack in the final table. Beating or approaching Raw LSTM indicates competitiveness with direct neural sequence prediction; beating or approaching the stack is stronger hybrid-comparator evidence. The stack comparison must be described as a complete-system comparison, not a standalone-GARCH result.

- **Phase 2 decoder refinement:** keep all five Phase-1 branches frozen and
  compare `D0` shallow MLP, `D1` branch-aware residual MLP, `D2` gated fusion,
  `D3` temporal LSTM, and `D4` temporal Transformer on identical eligible rows.
  Price and volatility are the first execution stage; movement classification
  is a later P2-only extension kept separate from the Part-3 C1/C2/C5 matrix.
  The exact architecture, `K=8`, seeds, budgets, row-map, and replay contract
  are frozen in the dedicated decoder specification.

- **Transferability analysis** — evaluate whether embeddings trained on one subset of tasks or markets transfer effectively to held-out tasks, contract types, or timeframes without retraining.

- **Ablation study** — use single-branch probes for standalone usefulness and
  matched leave-one-branch-out probes for marginal usefulness relative to the
  canonical full representation.

- **Additional alpha-research downstream capability (Phase 7B, intentionally
  deferred)** — its research question, data allocation, search protocol, and
  economic evaluation will be reconsidered only after further literature
  review. A possible future extension may test whether interpretable formulaic
  scores can be composed from downstream predictions rather than latent
  dimensions. The representation-learning framework remains the contribution;
  GP/symbolic regression would be an established search tool, not a claimed
  algorithmic novelty, and no profitable-alpha claim is currently proposed.
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
