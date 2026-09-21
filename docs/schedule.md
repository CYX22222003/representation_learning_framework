# FYP Progress and Schedule

**Last updated:** 2026-09-21

> **Current phase:** Phase 4 data exploration and selection are concluded.
> Phase 4 ran no encoder, downstream, or baseline training. Its final decision
> led to the canonical
> [Phase 5 plan](phase_plan/2026-09-21-phase-5-experiment-plan.md). Phase 5 uses
> the two fresh top-50 clean native one-hour walk cohorts, accepts their
> retrospective selection, assumes the approved pruning is correct offline
> cleaning, and does not require quarantine-availability replay. Its primary
> contract is `seq64`, two rolling global-calendar walks, separately trained
> canonical five-branch weights per walk, and shared two-hour movement
> regression/classification targets. The walk-specific train/test builder,
> shared labels, common comparator identities, and replayable data manifests
> are implemented and validated. The six canonical neural encoders, two frozen
> five-branch feature stores, two train-only feature scalers, and four seed-0
> framework downstream trajectories are also complete and replay-validated.
> Learned baselines and additional seeds remain pending and require a separately
> frozen identical-row matrix before execution.

---

## 1. Current Achievements

### Stage 1 — Data Collection and Processing
| Task | Status |
|---|---|
| Collect Polymarket OHLCV feather files | ✅ Done |
| Recent FinData acquisition (Dec 2025--Aug 2026) | ✅ Phase 4 source audit complete: raw-first 50-market recollection, forward-confirmed pruning, native 15m/1h dynamics, gap distribution, and one-/four-bar fill sensitivities are recorded; clean one-hour data is selected for exploratory Phase 5 training with explicit token-identity limitations |
| Recent FinData Phase 5 walk-capacity audit | ✅ No-training feasibility complete: balanced two-walk `seq64` candidate retains 31,828/37,173 active training rows and 21,401/10,086 supported evaluation rows; retrospective selection and final pruning are now accepted Phase 5 assumptions rather than blockers |
| Fresh Phase 5 Walk 1/Walk 2 top-50 FinData exploration | ✅ Primary Phase 5 cohorts frozen: two independent searches/downloads complete with training-interval eligibility probes and evaluation-end history; approved quarantine, post-pruning gaps, separate one-bar bounded fill, staleness, and 200 contract-resolution plots replay-validated. Walk 2 hourly coverage is 75.40% versus 82.97% in Walk 1. |
| Phase 5 walk-specific sequence/label preparation | ✅ Implemented in `src/data_processing/phase5_walks.py`, exposed only through `scripts_v3/`, and replay-validated under `experiments/phase5/data_preparation/`. Target-free encoder selection is separated from downstream target eligibility. Walk 1 has 39,070 encoder / 37,864 supervised-train / 30,340 supported-evaluation rows; Walk 2 has 58,473 / 57,521 / 13,887. Raw and train-scaled OHLCV, common regression/classification labels, causal activity, endpoint maturity, gap metadata, and source/artifact hashes are preserved. |
| Phase 5 canonical encoder pretraining | ✅ Six independent seed-0 CUDA trajectories complete: VAE, contrastive CNN, and BYOL CNN for each walk, one continuous 50-epoch run with snapshots at 5/15/50. All checkpoints and histories replay-valid, epoch-50 inference passes, and no collapse warning was raised. Weights live under `experiments/phase5/encoder_pretraining/`; epoch 50 is fixed for feature extraction. |
| Phase 5 five-branch feature extraction | ✅ Two walk-local 445-dimensional stores complete under `experiments/phase5/features/`: 37,864/30,340 train/evaluation rows for Walk 1 and 57,521/13,887 for Walk 2. Branch arrays, row identities, source bundles, and epoch-50 encoder hashes replay; every supervised training context is byte-identical to its mapped encoder context and evaluation identities are disjoint. |
| Phase 5 framework downstream probing | ✅ Four seed-0 CUDA trajectories complete under `experiments/phase5/downstream/`: regression and classification for both walks, 50 epochs with 5/15/50 snapshots. Train-only coordinate scalers, CPU smoke tests, checkpoint/prediction replay, non-trained references, breakdowns, pooled predictions, and report are complete. |
| Phase 5 exploratory additional regression tasks | ✅ Eight-hour raw probability change and two-hour ordinary log return are implemented and executed for both walks at seed 0. Horizon-specific maturity/gap rows, reused-or-fresh frozen features, train-only log-target scaling, reconstructed probabilities, starting-price/lifecycle/imputation breakdowns, 5/15/50 checkpoints, and CPU replay are complete. Neither target recovered stable signed correlation. |
| Phase 5 eight-hour absolute-price probe | ✅ Two walk-specific sigmoid heads completed at seed 0 and 5/15/50 epochs. Price-level MAE remains worse than current-price persistence, but implied movement has positive pooled Spearman `0.1287` and mean cross-sectional Rank IC `0.1264`. A simple last-hour reversal score is stronger (`0.2794` pooled mean IC), so representation value is not established. |
| Select top-50 active contracts per timeframe (1h, 4h, 1d) | ✅ Done |
| Causal missing-value handling | 🔄 Implemented, not run at full scale; active top-50 1h/4h/1d audit found zero missing/non-finite OHLCV cells |
| Z-score normalise volume | 🔄 Corrected to fit the raw training prefix only; full top-50 rebuild/audit pending |
| Sliding window segmentation → `[N, seq_len, features]` | 🔄 Raw-time-first isolated-window policy implemented; full top-50 rebuild/audit pending |
| 80/20 chronological train/test split per contract | 🔄 Raw boundary now precedes fitted preprocessing and windows; full top-50 rebuild/audit pending |
| Merge sequences across contracts | ✅ Done |
| Exploratory data analysis (distributions, volatility regimes) | ✅ Phase 4 scope complete: pre-Dec-2025 lifecycle/calendar evidence and recent FinData 15m/1h staleness, gap, pruning, and forward-fill analyses are consolidated; broader 1d/event-level EDA is optional future work rather than a Phase 4 exit condition |

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
| VAE architecture (MLP encoder-decoder, β-VAE loss) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| Contrastive encoder (CNN backbone, NT-Xent loss, augmentations) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| BYOL encoder (CNN online/target encoder, EMA target update) | ✅ Trained on 4h split with fixed-budget CUDA sweep; checkpoint/report complete |
| Phase-2 temporal encoder refinement | ⏸️ Paused; completed checkpoints/probes are retained as legacy-pipeline characterisation evidence |
| **Aggregation and downstream tasks** | |
| `RepresentationAggregator` (N-branch gated fusion, dict API) | ✅ Implemented; concat mode evaluated in five-branch Phase-1 price, trend, and volatility runs |
| `PriceRegressor` task head (MAE/RMSE) | ✅ Evaluated in the four-branch MVP and the five-branch Phase-1 price sweep |
| `VolatilityRegressor` task head (MSE, correlation) | ✅ Evaluated in the five-branch Phase-1 shared-label volatility sweep; unconstrained negative outputs documented for follow-up |
| `TrendClassifier` task head (accuracy, macro-F1) | ✅ Evaluated in four-branch MVP and five-branch Phase-1 tri-class sweeps |
| Branch-aware `FeatureBundle` + `NpzFeatureStore` (save/load pipeline) | ✅ Done; legacy packed `neural` stores remain loadable |
| End-to-end training script | ✅ `scripts/train_framework.py` supports price, trend, and contract-safe volatility prediction; five-branch Phase-1 execution complete for all three tasks |

### Stage 3 — Benchmark and Baseline Implementation and Training

**External benchmarks** (from prior work)
| Task | Status |
|---|---|
| Stacked LSTM benchmark (3-layer, 4h data) | ✅ Trained on unified splits (v5 epoch sweep, seed=0; MAE 0.007–0.012, RMSE 0.016–0.018) |
| Raw LSTM volatility benchmark | ✅ Trained on shared 4h realised-volatility label bundle at 15/50/100 epochs |
| Adapted GARCH--LSTM stacking volatility benchmark | ✅ Trained on shared 4h realised-volatility label bundle at 15/50/100 epochs; replay verification and plots complete |
| GINN benchmark (AR→GARCH→LSTM, volatility) | ✅ Trained on 4h data at 15 epochs; further sweep deferred because of documented GARCH-target failure |
| TA-MLP benchmark (FreqTrade, trend classification) | ✅ Natural-sampling adaptation trained on unified splits (v1 triclass epoch sweep, seed=0; acc 0.71–0.73, macro-F1 0.45–0.47); paper-derived training-only undersampling remains pending |
| Additional benchmarks from literature review (TBD) | ⬜ TBD |

**Internal baselines** (designed within this project)
| Task | Status |
|---|---|
| Raw-OHLCV MLP (no representation learning) | ✅ Trained on 4h data for price, volatility, and trend; price/volatility sweeps at 15/50/100 epochs. The volatility sweep predates the shared contract-aware label bundle and is characterization-only pending a strict rerun. |
| Statistical-only ablation | ⬜ Not started |
| Transformation-only ablation | ⬜ Not started |
| VAE-only ablation | ⬜ Not started |
| Contrastive-only ablation | ⬜ Not started |
| BYOL-only ablation | ⬜ Not started |
| Additional neural encoder ablations (per TBD methods) | ⬜ TBD |
| Additional internal baselines (TBD) | ⬜ TBD |

### Stage 4 — Experiments and Benchmarking
| Task | Status |
|---|---|
| Evaluation harness (unified test loop for all models) | 🔄 Framework runner records configs, manifests, metrics, predictions, summaries, and comparisons; baseline runners still evaluate independently |
| Absolute next-close benchmark (MAE, RMSE) | ⚠️ Phase 3 is leakage-safe but persistence-dominated; retained as negative characterisation evidence and superseded as the primary regression probe |
| Eight-hour absolute-price add-on | 🔄 Phase 5 recent-data probe complete: price Pearson is high (`0.9938`) but level errors trail persistence; implied-movement Rank IC is positive across both walks but weaker than last-hour reversal. This is useful state-reconstruction/ranking evidence, not a return-forecasting win. |
| Probability-movement regression | ⚠️ Phase 5 framework seed-0 probe complete: pooled out-of-future MAE/RMSE `0.002711/0.010835`; Pearson `0.0807`, Spearman approximately zero. Eight-hour raw change (`Pearson 0.0142`) and two-hour log return (`Pearson -0.0103`) did not restore signed predictability. Learned baselines and additional seeds remain pending. |
| Probability-movement classification | ✅ Phase 5 framework seed-0 probe complete: pooled macro-F1/balanced accuracy `0.4541/0.4730` versus always-STABLE `0.2831/0.3333`; accuracy is lower (`0.6369` versus `0.7382`) because the logit-adjusted model predicts minority classes. Learned comparisons remain pending. |
| Temporal encoder adaptation/transfer | ⏭️ Deferred to Phase 6; Phase 5 trains the same canonical architecture independently in each walk without a fixed-first-walk transfer ablation |
| Volatility prediction benchmark (MSE, correlation) | ⚠️ Row-aligned artifacts are preserved, but they inherit the upstream defect and use the overlapping MVP target |
| Trend classification benchmark (accuracy, macro-F1) | ⚠️ Artifacts are preserved, but their raw/frozen inputs inherit the upstream defect |
| Phase 2 decoder refinement | ⏸️ Paused; 14 of 30 trajectories are preserved, but no remaining run should execute before the upstream rebuild |
| Phase 2 encoder refinement | ⏸️ Paused; seed-0 checkpoints, frozen features, CKA, and probes are preserved as legacy-pipeline evidence |
| Phase 2 probability-movement classification | ⏸️ Paused; task-local labels/alignment are sound, but completed seed-0 runs inherit the upstream processed-data defect |
| Transferability analysis (across markets and timeframes) | ⬜ Not started |
| Ablation study (per-branch contribution) | ⬜ Not started |
| Additional alpha-research capability (OOF downstream predictions → shallow symbolic factors) | 🔄 Train-only raw-OHLCV Alpha101-style and bounded GP dry runs are archived under `experiments_old/alpha/raw_ohlcv_4h_top50_dry_run/` and `experiments_old/alpha/raw_gp_4h_top50_dry_run/`; the 20-coordinate direct-representation run found no useful confirmation signal, while the exhaustive 445-coordinate + OHLCV GP run found only weak mixed signal under `experiments_old/alpha/representation_ohlcv_gp_4h_top50_all_features/`; downstream-head OOF symbolic mining and a fresh-holdout evaluation remain unrun |
| Result tables and visualisations | 🔄 Phase-1 price, trend, and volatility summaries, comparisons, and plots generated; final cross-model tables, branch ablations, and embedding visualisations pending |

Canonical Phase 2 scope is maintained in
`docs/phase_plan/2026-09-08-phase-2-experiment-plan.md`; the frozen Part 3
classification matrix is maintained in
`docs/phase_plan/2026-09-08-phase-2-probabilistic-classification.md`.

The concluded leakage-safe rerun is specified in
`docs/phase_plan/2026-09-20-phase-3-experiment-plan.md`. Its raw-time-first
4-hour bundle is built and validated at 21,696 train / 3,085 test windows over
50 contracts. The seed-0 encoder-pretraining matrix is complete: canonical VAE,
contrastive CNN/LSTM/Transformer, and BYOL CNN/LSTM/Transformer each have one
uninterrupted 50-epoch trajectory and checkpoints at `5/15/50`. All 21
snapshots passed the consolidated provenance/history check with no collapse
warnings; numerical tables and six plots are stored under
`experiments/phase3/reports/encoder_pretraining_seed0/`. Phase 3 price-label,
master feature-extraction, 15-configuration framework training, manifest-first
launch, and replay/report scripts are implemented under `scripts_v2/`. The
contract-local horizon-1 labels (21,646 train / 3,035 test), nine-branch master
feature bundle, and all 15 seed-0 framework price trajectories are complete.
All 45 snapshots passed common-row prediction replay; the exploratory best
held-out result is HB-ALT at epoch 50 (MAE 0.009234, RMSE 0.022782, correlation
0.998993). A subsequent exact persistence audit obtained MAE 0.001063 and RMSE
0.003789 on the same 3,035 test identities, so no Phase 3 framework row added
value over copying the current close. The test tail contained 75.45% unchanged
targets and 77.83% prices near 0/1, versus 43.74% and 36.10% in training.
Phase 3 is therefore closed before its originally planned classification and
baseline execution; those runs are superseded by the Phase 4 global-calendar
and probability-movement redesign that Phase 5 will execute. Existing `scripts/`
entry points and all `_old` artifacts remain excluded from new execution.

---

## 2. Summary

The current project state is **Phase 4 concluded; Phase 5 framework-only seed-0
probing complete, with learned baselines pending**. Phase 3 proved that raw-time-first construction,
training-only fitted preprocessing, isolated windows, contract-local labels,
and artifact replay can remove the Phase 1/2 future leakage. Phase 4 then
established that the single final-20% tail is lifecycle-biased, absolute
next-close regression is dominated by persistence, and per-contract-relative
walks are insufficient for a pooled cross-contract model. It completed the
pre-December-2025 lifecycle/frequency studies and the recent FinData source,
staleness, gap, pruning, and forward-fill audits. No Phase 4 model training was
run. The Phase 5 plan now freezes the two fresh walk-specific cohorts, `seq64`,
two-hour movement targets, `tau=0.001` classification, the canonical
five-branch concat model, and separate weights per walk. The sequence/label
builder, encoders, feature stores, train-only scalers, framework heads, and
5/15/50 prediction replay are now implemented, executed, and validated. The
pooled regression probe is weaker than exact zero on MAE/RMSE, while
classification improves macro-F1 and balanced accuracy over always-STABLE at
the cost of accuracy. The immediate priority is to freeze and execute the
learned-baseline matrix on identical rows before making comparative claims;
additional framework seeds remain later work.

The targeted top-50 4-hour lifecycle audit now provides supporting evidence:
from early to late contract thirds, exact-zero movement increased from `30.00%`
to `58.06%`, near-boundary prices from `29.62%` to `55.97%`, and the
zero-movement baseline MAE fell from `0.010659` to `0.004268`. All saved feature
branches contained lifecycle-separable information in contract-grouped probes.
This justifies the fixed-versus-adaptive encoder test and within-walk lifecycle
reporting, but not a conclusion that different encoder architectures are
already required. See
`docs/data_analysis/2026-09-20-phase4-calendar-lifecycle-exploration.md`.

The matching top-80 one-hour audit found 192,658 raw rows with a clean
`datetime64[ns]` hourly `date` field, zero duplicate/gapped timestamps, and no
missing or non-finite OHLCV cells. At a duration-matched 256-step context, the
candidate rolling folds provide 75,241--75,262 contract-relative training rows
and 97,511--130,054 global-calendar training rows, well above the 21,696-row
Phase 3 four-hour reference. However, the middle interval of an equally spaced
three-walk global schedule contains only two contracts; the two-walk schedule
has healthier evaluation coverage of 20 and 15 contracts. See
`docs/data_analysis/2026-09-20-phase4-top80-1h-timestamp-capacity.md`.

The subsequent one-hour activity audit found that aggregate capacity
overstates usable information. Across 171,618 duration-matched eligible rows,
46.23% of eight-hour targets are exactly unchanged and 69.57% remain within
0.005. Twenty contracts have meaningful movement in fewer than 10% of rows,
and 19 contracts have a trailing constant-price tail covering at least one
quarter of the file. A causal prior-24h price-change filter retains 130,878
rows across all 80 contracts, lowers the exact-zero share to 29.61%, and leaves
104,629/90,504 training rows across 59 contracts and 12,876/13,278 evaluation
rows across 14/13 contracts in the two calendar walks. This freezes causal
recent-activity eligibility as a Phase 5 requirement; implementation remains pending. See
`docs/data_analysis/2026-09-20-phase4-top80-1h-activity-suitability.md`.

The duration-matched archive comparison found that top-80 one-hour and
four-hour data have aligned-target correlation `0.9690`, exact delta agreement
`88.12%`, and nearly identical stable shares and zero-baseline errors. This
supported four-hour top-80 as the compute-efficient primary configuration in
the unexecuted 2024--2025 Phase 4 design. It remains qualitative feasibility
evidence, but it is superseded for the next training loop by the recent
FinData one-hour selection. See
`docs/data_analysis/2026-09-20-phase4-top50-top80-1h-4h-comparison.md`.

The legacy data pipeline produced all three processed timeframes, but the
raw-time split and fitted-preprocessing audit has invalidated their status as
clean held-out bundles. The existing 4h bundles, encoder checkpoints, frozen
features, and task evaluations remain preserved and structurally replayable;
they are not valid inputs for further confirmatory execution.

The Phase B/C implementation remains available, but Phase 5 must use new
`scripts_v3` entry points and artifacts under `experiments/phase5/` tied to
global-walk provenance. Shared preparation logic belongs under
`src/data_processing/`; `scripts_v2/` is not extended for Phase 5.
Each primary walk needs its own preprocessing state, encoder checkpoint,
feature bundle, downstream head, and baseline weights. Phase 3 artifacts remain
immutable characterisation evidence; they are not silently relabelled as
Phase 5 inputs.

A separate recent-data audit covers `[2025-12-01, 2026-09-01)` through the lab
FinData API. The raw-first 50-market collection contains 325,730 native
15-minute rows and 107,635 native hourly rows. Forward-confirmed quarantine
flags 116 and 147 rows and retains 99.9644%/99.8634%. After pruning, observed
native movement is non-zero for 68%--75% of valid short-horizon targets. The
hourly grid has 84.01% observed internal coverage and reaches 87.56% after
causally filling 4,534 isolated one-hour gaps; synthetic rows then comprise
4.05% of retained rows and direct one-hour exact-zero movement rises from
29.66% to 32.78%. The 15-minute grid is only 63.64% observed; a four-bar
sensitivity reaches 79.97% coverage but makes 20.42% of retained rows
synthetic and raises direct 15-minute zero movement from 31.53% to 46.13%.
Phase 4 therefore selects bounded-forward-filled clean one-hour data for the
next exploratory loop and retains 15-minute data as a sensitivity. The
condition endpoint's missing token identity remains an explicit limitation;
the correctly oriented trade fallback is too sparse. See the
[Phase 4 conclusion](phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md).

The five-branch Phase-1 price sweep is archived under `experiments_old/framework/phase1/price_prediction/4h_phase1_all5_concat/`. It uses the validated 445-dimensional concat representation (`statistical`, `transformed`, `vae`, `contrastive`, `byol`), seed 0, and fixed 15/50/100 budgets. Its MAE/RMSE are `0.0512/0.0908`, `0.0651/0.0993`, and `0.0678/0.1009`, respectively. The saved comparison records a strict 27,499-row match to the Raw-OHLCV MLP within the legacy merged-array price contract and shows Phase-1 lower MLP-matched MAE/RMSE at epoch 15 only (`38.6%`/`14.9%` relative error reduction); later budgets are worse. New Phase-2 price runs instead use the contract-safe bundle with 109,791 train / 27,450 test rows, removing the terminal row of each contract and the 49 invalid internal boundary transitions retained by the legacy helper. Phase-1 absolute metrics are therefore not strict comparators for new price runs. The LSTM table is contextual rather than strict because it uses 27,500 close-only rows with a documented one-row target alignment difference. All fixed-budget results are retained; no epoch is selected from locked-test performance. See `docs/price_prediction_label_contract.md`.

Trend classification has both four-branch and five-branch framework results on the same TA-MLP-style tri-class BUY/HOLD/SELL label bundle. The Phase-1 run archived under `experiments_old/framework/phase1/trend_classification/4h_phase1_all5_concat/` reports accuracy/macro-F1 of `0.4550/0.3757` at 15 epochs, `0.4761/0.3942` at 50 epochs, and `0.4765/0.3935` at 100 epochs. It remains below the exact majority-HOLD accuracy (`0.5046`) but above its macro-F1 (`0.2236`), showing non-trivial minority-class predictions. The strict matched four-branch comparison is stronger at every budget, so the current BYOL-plus-concat configuration does not improve trend classification. The existing TA-MLP sweep remains contextual until it consumes the identical saved rows.

The five-branch Phase-1 volatility run, Raw LSTM, and adapted GARCH--LSTM stack use the same realised-volatility bundle and exactly identical `27,450` locked test targets. Phase-1 records MSE/correlation of `0.00793/0.767`, `0.00749/0.770`, and `0.00769/0.764` at 15/50/100 epochs. It improves on Raw LSTM at every matched budget (about `22-33%` lower MSE), while the stack remains stronger overall; Phase-1 and the stack are nearly tied on RMSE/MSE at 15 and 50 epochs, but the stack has lower MAE and higher correlation. The framework head also produces `5.8-8.4%` negative predictions because its output is unconstrained; raw results remain primary, zero-clipping is diagnostic only, and a predeclared nonnegative decoder rerun is needed before final claims. The legacy Raw-OHLCV MLP remains contextual pending migration to the shared bundle.

The five-branch Phase-1 task matrix is complete for price, trend, and volatility, with reports and plots saved for all three. Phase 2 has three independent parts: decoder refinement, encoder refinement, and classification relabelling. The D0--D4 Stage-1 decoder pipeline is implemented at `K=8`, seeds `0/1/2`, and budgets `15/50/100`; its contract-safe temporal index and price labels pass real-data replay. Fourteen of the 30 frozen CUDA trajectories have complete sweep artifacts (42 evaluated snapshots); D3 price seed 2 and all 15 volatility trajectories remain, so aggregate reporting and decoder-performance conclusions are still pending. Horizontal encoder refinement now has seed-0 contrastive and BYOL LSTM/Transformer checkpoints, frozen feature bundles, and linear-CKA diagnostics. Its contract-safe price stage has complete raw/replay-verified `H0`, `HC-SL`, `HC-ST`, and `HC-AL` trajectories; the remaining contrastive controls/additions and BYOL-family price runs are pending, and volatility/classification have not started. The earlier contrastive price pilots remain legacy-label characterisation evidence rather than members of this strict matrix. The Part 3 classification specification is frozen at `h=2`, `tau=0.005`, seeds `0/1/2`, and budgets `15/50/100`; its only learned trajectories are the strict TA-aligned C1 Raw-OHLCV MLP, C2 five-branch framework, and C5 adapted TA-MLP under P0/P1U/P1O/P2. Decoder and encoder variants are excluded from the Part 3 launcher but remain the subjects of Parts 1 and 2. Full-row-only runs, representation ablations, C3, and C4 are not required by the current Phase 2 plan. The earlier launcher was stopped after this scope correction; completed intended C1/C2/C5 artifacts remain available for a clean resume. Other priorities remain completing and reporting the decoder and classification matrices, completing the horizontal encoder price matrix before later tasks, migrating the Raw-OHLCV MLP volatility baseline, confirming a nonnegative framework volatility decoder, and finishing external-baseline alignment. Train-only raw-OHLCV Alpha101-style and bounded-GP dry runs demonstrate timestamped factor scoring and evolutionary formula construction without touching the global task-test partition; downstream-head OOF mining and fresh-holdout alpha evaluation remain outside this budget. Current framework evidence does not support universal superiority over task-specific baselines.

The canonical phase documents are maintained under `docs/phase_plan/`: the
Phase-1 product-readiness plan and experiment-observation judgement, plus the
complete Phase-2 plan, frozen Part 1 decoder implementation specification, and
frozen Part 3 classification contract. The Phase 3 experiment plan separately
defines the leakage-safe rerun, seven-encoder pretraining matrix, downstream
feature study, and matched price/classification baselines.

### Recent VAE encoder progress

The VAE encoder was successfully trained on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. The run used only the locked
training split (`109841` sequences) and recorded the test split shape (`27500`
sequences) for traceability only. No test sequences were used for training,
early stopping, or checkpoint selection.

| epoch budget | total loss | reconstruction MSE | KL divergence | best train loss so far | elapsed seconds |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.0999 | 0.0700 | 0.0300 | 0.0809 | 19.96 |
| 20 | 0.0866 | 0.0669 | 0.0196 | 0.0809 | 26.10 |
| 25 | 0.0841 | 0.0659 | 0.0182 | 0.0809 | 32.58 |
| 50 | 0.0828 | 0.0647 | 0.0180 | 0.0809 | 64.44 |
| 100 | 0.0802 | 0.0626 | 0.0176 | 0.0800 | 125.33 |

The run recovered from a transient early instability at epochs 13-14 and then
improved gradually through the 100-epoch budget. The final checkpoint is close
to the best observed training point (`0.0800388432` at epoch 98), so it is a
reasonable current VAE encoder candidate. As with contrastive pretraining, this
is unsupervised pretraining evidence only; the checkpoint now feeds the MVP
price and trend probing runs, while branch-specific contribution still needs
ablation.

### Recent contrastive encoder progress

The contrastive encoder was successfully trained on the NVIDIA GPU through WSL using the unified 4h, sequence-length-64 dataset. The run used only the locked training split (`109841` sequences) and recorded the test split shape (`27500` sequences) for traceability only. No test sequences were used for training, early stopping, or checkpoint selection.

| epoch budget | train NT-Xent loss | best train loss so far | elapsed seconds |
|---:|---:|---:|---:|
| 15 | 2.5918 | 2.5918 | 214.70 |
| 20 | 2.5496 | 2.5496 | 287.62 |
| 25 | 2.5225 | 2.5225 | 364.87 |
| 50 | 2.4600 | 2.4599 | 727.67 |
| 100 | 2.4100 | 2.4085 | 1460.38 |

The loss decreased consistently and plateaued gradually, so the result is meaningful as unsupervised pretraining evidence. This checkpoint now feeds the MVP frozen-embedding probing runs for price prediction and trend classification; branch-specific contribution still needs ablation.

### Recent BYOL encoder progress

The BYOL encoder was successfully trained on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. The run used only the locked
training split (`109841` sequences) and recorded the test split shape (`27500`
sequences) for traceability only. No test sequences were used for training,
early stopping, or checkpoint selection.

| epoch budget | train BYOL loss | view cosine | embedding std | collapse warning | elapsed seconds |
|---:|---:|---:|---:|:---:|---:|
| 15 | 0.0755 | 0.9623 | 0.6185 | false | 191.52 |
| 20 | 0.0810 | 0.9595 | 0.7010 | false | 250.28 |
| 25 | 0.0797 | 0.9602 | 0.7765 | false | 315.87 |
| 50 | 0.0776 | 0.9612 | 0.9908 | false | 635.14 |
| 100 | 0.0526 | 0.9737 | 1.3357 | false | 1283.12 |

The loss reached an early minimum before rising as the exponential-moving-average
target and representation scale evolved, then declined through the final budget.
Because the BYOL target changes during training, the early minimum is recorded as
a diagnostic rather than used for checkpoint selection. Embedding standard
deviation increased from `0.1457` after epoch 1 to `1.3357` after epoch 100,
remaining well above the configured collapse threshold (`0.001`). The full raw
histories, checkpoint metrics, plots, and report are stored under
`experiments_old/byol_encoder/byol-4h-seq64-top50/`; the archived checkpoint is
`checkpoints_old/byol_4h_seq64_top50.pth`. Downstream BYOL feature extraction and
branch ablation remain pending.

### Recent GINN progress

The GINN baseline was successfully executed on the NVIDIA GPU through WSL using
the unified 4h, sequence-length-64 dataset. Two seed-0, 15-epoch
characterisation runs were recorded:

| output transform | MSE | Pearson correlation | RMSE | negative prediction fraction |
|---|---:|---:|---:|---:|
| linear | 1887.2247 | 0.0488 | 43.4422 | 0.6284 |
| softplus | 1313.9036 | -0.0703 | 36.2478 | 0.0000 |

The softplus transform removed invalid negative volatility predictions but did
not restore predictive quality. Investigation found that one near-static,
sparsely traded contract generated a numerically converged but implausibly
large GARCH target, which dominated the fused loss. This limitation is recorded
in [`src/baselines/ginn_baseline/LIMITATIONS.md`](../src/baselines/ginn_baseline/LIMITATIONS.md).
The result is retained as a documented GINN limitation rather than used to
justify a broad claim against GARCH or silently modified for the main benchmark.

---

## 3. Proposed Schedule

The schedule is structured as four phases. Phase A is a hard prerequisite. Phases B and C are iterative — the framework and baselines grow in complexity together, with a working end-to-end loop established as early as possible.

> **Numbered experiment-phase handoff:** The A--D sections below remain useful
> workstream history, but they no longer name the current experiment phase.
> Numbered Phase 4 is concluded. Numbered Phase 5 now owns the recent one-hour
> walk-forward builder, fold-specific encoder/downstream/baseline loop, and
> replayed comparison. Unchecked legacy A--D items are not evidence that Phase
> 4 remained open.

---

### Phase A — Data (prerequisite for everything)

- [x] Write a preprocessing script that runs `build_from_file_list` and saves results to `.npz` (bridging `data_processing.py` → `reader.py`)
- [x] Verify tensor shapes and data integrity across all three timeframes (1h, 4h, 1d)
- [ ] Move the per-contract split to the raw timeline before fitted
  preprocessing and window generation
- [ ] Record raw boundaries, window coverage, train-fitted preprocessing
  parameters, and the test-context policy in the processed manifest
- [ ] Add end-to-end tests that reject test-influenced preprocessing or test
  observations inside training windows/targets
- [ ] Rebuild and validate processed bundles before retraining encoders

**Exit condition:** all three timeframes have replayable `.npz` files that load
correctly and prove that preprocessing, training windows, and training targets
use no test-period observations. **Status: not currently achieved.**

---

### Phase B — First working loop (framework milestone)

The goal of this phase is a single end-to-end run: train one neural encoder, train the aggregator on one task, and compare against one baseline — enough to confirm the pipeline works and produce a first reference number.

**Framework side**
- [x] Pretrain one neural encoder on train data: contrastive 4h sweep complete with checkpoint/report artifacts
- [x] Train/report VAE encoder on train data: 4h sweep complete with checkpoint/report artifacts
- [x] Write `train_framework.py`: frozen encoder feature bundle → aggregator + task head; Phase-1 price, trend, and volatility task runs complete
- [ ] Write a unified cross-model `evaluate.py`/comparison harness for all framework and baseline artifacts

**Baseline side** (run in parallel once data is ready)
- [x] Train `RawOHLCVMLP` baseline (flattened OHLCV, no representation); legacy 4h price, volatility, and trend artifacts are archived under `src/baselines/mlp_baseline/experiments_old/`. The volatility artifact predates the shared bundle and requires a strict rerun.
- [ ] Run statistical-only ablation (AR + GARCH features only, no aggregator)

**Exit condition:** framework and at least two baselines produce numbers on the same test split. **Status:** achieved for the first price-prediction loop; trend MVP is also implemented, while strict external-baseline row alignment remains pending.

---

### Phase C — Iterative expansion (after Phase B)

From here, both sides grow in parallel. Add one method at a time; re-run evaluation after each addition to track whether it helps.

**Expand neural encoders** (order by complexity)
- [x] Train/report contrastive encoder on the 4h split; checkpoint and report complete
- [ ] Evaluate: does adding the contrastive branch improve over VAE-only? *(full VAE+contrastive concat result exists; isolated branch ablation pending)*
- [x] Implement, train, and report the BYOL encoder on the 4h split; checkpoint and diagnostic artifacts complete
- [ ] Evaluate: does adding the BYOL branch improve over the current VAE + contrastive branch set?
- [ ] Identify additional unsupervised methods from literature (masked autoencoder, self-supervised Transformer, etc.); integrate promising ones one at a time following the same pattern
- [ ] Pretrain and evaluate the Phase-2 `contrastive_lstm` and `contrastive_transformer` candidates using the fixed shallow probe *(basic implementation and CPU tests complete; final experiment specification and CUDA matrix pending)*

**Expand external benchmarks** (wire into evaluation harness one at a time)
- [x] Retrain LSTM benchmark on unified `.npz` data splits; legacy results archived *(v5 sweep, see `src/baselines/lstm_baseline/experiments_old/`)*
- [x] Train Raw LSTM volatility benchmark on the shared realised-volatility label bundle; record matched 15/50/100 epoch artifacts
- [x] Run the adapted GARCH--LSTM stacking volatility benchmark using Raw LSTM predictions and fixed ElasticNet meta-learning
- [x] Train GINN benchmark on the unified 4h split at 15 epochs; document the GARCH-target failure and defer further GINN sweeps while selecting a more suitable volatility benchmark
- [x] Train TA-MLP natural-sampling adaptation *(legacy v1 triclass sweep, see `src/baselines/ta_mlp_baseline/experiments_old/2026-06-22-v1/`)*; paper-derived training-only undersampling and strict saved-label alignment remain pending
- [ ] Additional benchmarks from literature (TBD after literature review) — retrain each on same data splits

**Expand internal baselines** (order by complexity)
- [ ] Transformation-only ablation (FFT + Wavelet features only)
- [ ] VAE-only, contrastive-only, and BYOL-only ablations
- [ ] Additional internal baselines (TBD) — add as identified; no fixed list

**Expand tasks**
- [x] Extend framework training and evaluation to trend classification
- [x] Implement the isolated Phase 2 probability-movement classification pipeline and strict C1/C2/C5 row-alignment contract; full multi-seed experiment execution remains pending
- [x] Execute and compare the Phase-1 framework volatility experiment on the shared label bundle
- [ ] Transferability experiment: embed with model trained on one timeframe, evaluate on another
- [ ] Execute and report the implemented Phase 2 D0--D4 decoder matrix on price
  and volatility using the validated `K=8` temporal row map; movement
  classification remains a later P2-only extension outside the Part-3 launcher

**Exit condition:** all planned methods (both sides) have been trained and evaluated on all three tasks; ablation table is complete.

**Current exit gap:** implement the raw-time-first split/preprocessing contract,
store replayable raw-window provenance, add end-to-end leakage tests, rebuild
the processed and frozen-feature artifacts, and regenerate task row identities.
Only then may the remaining decoder, encoder, and classification trajectories
resume. Existing completion counts describe preserved legacy-pipeline artifacts,
not valid completion of the revised Phase-2 matrix.

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
  Framework             Baselines     ← run in parallel
  (VAE/contrastive+agg) (MLP + stat-only)
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
