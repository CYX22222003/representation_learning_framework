# Proposal: integrate SaURL-TS into Phase 6.7

**Status:** SaURL model and Stage 0--4 pipeline implemented and CPU-tested;
formal CUDA/resource admission and all experiment execution remain pending  \
**Baseline ID:** `SAURL-F` / SaURL-TS-Frozen  \
**Scientific role:** direct external frozen-representation baseline  \
**Primary comparator:** immutable canonical `H0`  \
**Fallback:** `SISSEL-Frozen`, exactly as pre-approved by the Phase 6.7 plan

## 1. Decision rule

Proceed with an independently authored `SAURL-F` using the approved Questions
4--11 reconstruction. The public repository remains audit evidence only: do
not copy, adapt, import, or vendor its unlicensed source. Report the candidate
as **SaURL-TS-Frozen (paper-guided reimplementation)** rather than an official
reproduction.

Activate `SISSEL-Frozen` only if the independent adapter fails its frozen CPU/
CUDA correctness or resource gate before downstream evaluation. Do not use
evaluation results to trigger the fallback.

The project owner approved staged execution on 2026-10-04: after its own
feasibility, pretraining, and feature gates pass, SaURL may run all six
task/walk probes before LWA is implemented. Those results may not change the
later LWA or optional TimeDART contract, and the final core report must retain
the immutable SaURL trajectories.

## 2. Comparison contract

If admitted, SaURL-TS must obey the existing Phase 6.7 contract without exception:

- train separate encoder weights for Walk 1 and Walk 2;
- use only the saved target-free `encoder_train_sequences` population for the applicable walk;
- never use movement, future-price, or future-volatility targets during encoder training;
- fit any input normalization only on that walk's encoder-training population;
- train one uninterrupted seed-0 trajectory for 50 full training-set passes;
- save epochs 5, 15, and 50, with epoch 50 fixed in advance for extraction;
- freeze the encoder before fitting any downstream scaler or head;
- return an embedding for every established task row; and
- reuse exact H0 task identities, heads, losses, metrics, and downstream budgets.

No validation split, early stopping, best-checkpoint selection, evaluation-driven restart, or model-specific task-row filter is allowed.

## 3. Data mapping

### Walk-specific pretraining inputs

| Walk | Source artifact | Encoder population | Permitted interval |
|---:|---|---|---|
| 1 | `experiments/phase5/data_preparation/walk1/market_1h_seq64_h2.npz` | `encoder_train_sequences` | `[2025-12-02, 2026-04-01)` |
| 2 | `experiments/phase5/data_preparation/walk2/market_1h_seq64_h2.npz` | `encoder_train_sequences` | `[2026-02-16, 2026-06-16)` |

Each tensor is finite `[N,64,5]` in `open, high, low, close, volume` order. OHLC values are unscaled probabilities. Volume already uses a walk-training-only global z-score from unique bounded-fill candles.

### Candidate input normalization

SaURL's paper forecasting pipeline standardizes training variables. The selected adaptation therefore fits one additional five-coordinate mean/std scaler on the flattened walk-local `encoder_train_sequences` and replays it unchanged on every extraction row for that walk. This choice must be frozen in the feasibility manifest and tested for finite output.

This scaler is candidate-specific input preprocessing, not the downstream embedding scaler. The latter is still fitted separately on each task's supervised training embeddings.

### Task-aligned extraction inputs

| Task | Dataset artifact pattern | Target |
|---|---|---|
| Movement classification | `experiments/phase5/data_preparation/walk{1,2}/market_1h_seq64_h2.npz` | `DOWN/STABLE/UP`, `h=2`, `tau=0.001` |
| Future price | `experiments/phase5/downstream_addons/shared/h8/data/walk{1,2}/market_1h_seq64_h8.npz` | absolute `close[t+8]` |
| Future realised variance | `experiments/phase6/volatility_prediction/data_preparation/walk{1,2}/volatility_1h_seq64_h8.npz` | raw-change RV over `(t,t+8h]` |

The encoder sees only each row's historical 64-hour context. Future labels and endpoint metadata are used only by the existing supervised evaluation path.

## 4. Frozen SaURL adaptation

The adaptation should preserve the paper wherever the project contract permits:

| Item | Frozen setting | Rationale |
|---|---|---|
| Input | `[B,64,5]`, walk-local channel standardization | source-compatible and leakage-safe |
| Representation width | 128 | paper native width |
| Domains | time, phase-preserving magnitude-augmented frequency, and cross-domain | core paper contribution plus selected source-evidenced reconstruction |
| Frequency view | `rfft` over time; transform magnitude only; retain original phase; reconstruct with `irfft(..., n=64)` before `E_F` | selected 2026-10-04; keeps every encoder input real `[B,64,5]` |
| Frequency inference | pass normalized real `[B,64,5]` input directly to frozen `E_F` | equivalent to an unchanged FFT/inverse-FFT round trip; branch specialization comes from pretraining |
| Backbone | six residual dilated-CNN blocks, hidden 64, output 128, kernel 3, dilations `1,2,4,8,16,32` | covers the 64-step context without source layers whose dilation exceeds it |
| Sequence pooling | global maximum over encoder timestamps | selected source-grounded row extraction |
| RwAM | eight 16-coordinate regions; avg/max regional pooling; shared `Conv1d(3,1,1) -> ReLU -> Conv1d(1,3,1)`; sigmoid; weighted sum | explicit reconstruction of underspecified paper block |
| SaDA view heads | separate temporal/frequency modules; shared width-16 depth-1 factorizer within a domain; two independent transform-head pairs | paper/code-informed view separation |
| Hard mask | one deterministic `1[sigmoid(logit)>0.5]` value per position, broadcast over channels and shared by both views; straight-through gradient estimator; no mask noise | follows the paper's `1 x T` threshold mask rather than the repository's stochastic sampler |
| Cross pair | temporal and reconstructed-frequency views of the same sample | selected interpretation of paper/code ambiguity |
| Update schedule | SaDA first every two minibatches; SaSSL every minibatch; parameter-isolated alternating updates | repository-evidenced interpretation of Algorithm 1 |
| BYOL projector/predictor | `128 -> 128 -> 128` with GELU | reuses project-native normalized symmetric prediction contract |
| Fusion | RwAM-weighted sum | required extraction point; remains 128-wide |
| Optimizers | Adam, SaSSL `1e-4`, SaDA `1e-2` | paper settings |
| Batch size | 32 | paper setting; smaller final batch allowed without dropping rows |
| Dropout | 0.1 | paper setting |
| SaDA weights | `alpha=0.1`, `beta=0.01`, `gamma=0.5`, `lambda=1.25` | paper sensitivity optimum, frozen without project tuning |
| BYOL EMA | 0.99 | selected from source evidence |
| Epoch budget | 50 with 5/15/50 snapshots | Phase 6.7 override for fair characterization |
| Seed | 0 | Phase 6.7 scope |
| Extraction | online encoders plus RwAM, projectors/predictors removed | direct frozen representation |

The project epoch budget is an explicit adaptation because the public ETTh1 source configuration uses a different number of epochs. It is acceptable only because all Phase 6.7 methods use the same predeclared full-pass budget and epoch 50 is selected before evaluation.

The encoder residual block is fixed as two same-width kernel-3 dilated
convolutions with padding equal to dilation, GELU after the first convolution
and residual addition, and dropout 0.1 after both convolutions. A `1x1`
convolution maps five inputs to hidden width 64, and another maps 64 to output
width 128. No normalization layer or public-code multi-kernel bank is used.
Projectors and predictors are two-layer `128 -> 128 -> 128` GELU MLPs without
batch normalization.

Each augmentation factorizer maps five channels to width 16 and uses one
dilation-1 residual block. One shared `Linear(16,1)` supplies factor logits;
four view-specific `Linear(16,1)` heads supply sigmoid informative/irrelevant
scales. One deterministic sigmoid mask is thresholded at 0.5 and shared by
the two view-specific transform-head pairs. The binary forward value uses a
straight-through gradient estimator, but no logistic/Gumbel mask noise is
sampled. This follows the paper's stated mask equation; the stochastic sampler
observed in the older public repository is deliberately not adopted.

### 4.1 Frozen loss realization

The implementation follows the paper's input-domain MMD definitions rather
than the public snapshot's undocumented encoder-feature MMD. For temporal
source `S_t=X` and frequency source `S_f=abs(rfft(X))`, each view separates an
informative component with a straight-through hard mask, applies separate
sigmoid transforms to informative and complementary components, and sums them.
The frequency result is recombined with the original phase and inverse-
transformed only after its magnitude-domain SaDA losses are calculated.

Use one independently implemented five-kernel Gaussian MMD. Flatten each
sample; derive a detached base bandwidth from the clamped mean non-diagonal
pairwise squared distance; use scales `{1/4,1/2,1,2,4}`; and calculate the
biased batch estimate. For each of the two views:

```text
L_A_time = L_mask + 0.1*L_preserve + 0.01*L_continuity + 0.5*L_dissimilar
L_A_freq = L_mask + 0.1*L_preserve                    + 0.5*L_dissimilar

L_mask       = mean(straight-through hard mask)
L_preserve   =  MMD(source, transformed informative component)
L_continuity = mean absolute adjacent-mask difference
L_dissimilar = -MMD(original irrelevant, transformed irrelevant)
L_diversity  = -MMD(view 1, view 2)

L_SaDA = mean over two views of (L_A_time + L_A_freq)
         + 1.25*(L_diversity_time + L_diversity_freq)
```

For SaSSL, define:

```text
Bundle A = {E_T(V_t1), E_F(V_f1), E_C(V_t1)}
Bundle B = {E_T(V_t2), E_F(V_f2), E_C(V_f1)}
```

Each encoder output is globally max-pooled, and RwAM jointly weights the three
branch vectors in each bundle. Each weighted branch then enters its own
projector/predictor path. The cross branch therefore predicts between a
temporal and frequency-derived view of the same sample. Its use of the first
view from each domain is a disclosed repository-evidenced reconstruction
choice. The per-branch objective is symmetric normalized squared-L2
(equivalently `2-2*cosine`), and:

```text
L_SaSSL = L_time_BYOL + L_frequency_BYOL + L_cross_BYOL
```

The target side uses frozen EMA encoders/projectors and the current shared
RwAM under `no_grad`; RwAM has no target copy. Adaptive loss weights and SWA
are excluded.

### 4.2 Frozen alternating optimizer order

There is no separate SaDA-to-completion warm-up. `global_step` spans epoch
boundaries and starts at zero:

1. On steps `0,2,4,...`, enable only SaDA gradients, generate both domain-view
   pairs, compute `L_SaDA`, and step separate temporal/frequency Adam
   optimizers at `1e-2`.
2. On every step, freeze SaDA and regenerate all views using its current
   parameters under `no_grad`. Do not reuse views made before the SaDA step.
3. Build Bundles A/B, compute the unweighted three-branch `L_SaSSL`, and update
   only online encoders/projectors/predictors and RwAM with Adam at `1e-4`.
4. After the SaSSL optimizer step, update target encoders/projectors with EMA
   `target = 0.99*target + 0.01*online`.
5. Fail on non-finite tensors, losses, or gradients. Do not silently skip,
   clip, restart, or tune the trajectory.

Every epoch is one seeded shuffled no-drop pass. A singleton remainder is
merged into the preceding batch so every encoder row is used once and MMD has
at least two samples. Primary pretraining is float32 without AMP. Checkpoints
at epochs 5/15/50 include all online/target/SaDA/RwAM states, all three
optimizers, the input scaler, global step, all RNG states, configuration/data
hashes, history, and frozen-probe replay tensors.

The full normative pseudocode and equations are in
[the clarification record](upstream_clarification_request.md#83-full-alternating-training-algorithm),
and the module/test/checkpoint mapping is in
[the implementation plan](implementation_plan.md).

### 4.3 Adaptation disclosure

The final report must distinguish:

- **paper-stated:** three dilated-CNN domains, magnitude-domain augmentation,
  RwAM-weighted 128-wide sum, deterministic `1[sigmoid(logit)>0.5]` mask,
  SaDA coefficients, batch 32, dropout 0.1, and SaDA/SaSSL learning rates;
- **repository-evidenced:** original-phase inverse-FFT reconstruction,
  separate transform heads, alternating cadence, cross temporal/frequency
  pairing, and EMA 0.99; its hard stochastic mask sampler is audited but not
  adopted; and
- **project-selected:** six-block dilation schedule, global max pooling, exact
  eight-region RwAM, input-domain five-kernel MMD estimator, normalized
  symmetric BYOL realization, deterministic straight-through gradient
  estimator for the paper's binary mask, view-1 cross pair, 50-epoch Phase 6.7
  budget, and project-native checkpoint/replay boundary.

## 5. Representation-store design

Create one master store per admitted method and walk, so SaURL contributes two stores:

```text
experiments/phase6_7/features/saurl_frozen/walk1/representations.npz
experiments/phase6_7/features/saurl_frozen/walk2/representations.npz
```

Each store should contain separate groups for the six task/split populations:

```text
classification_h2_{train,test}_features
absolute_price_h8_{train,test}_features
realised_variance_{train,test}_features
<task>_<split>_{condition_ids,decision_date_ns,...identity fields}
```

The manifest records source checkpoint hash, input-scaler hash, ordered identity hashes, output width 128, array hashes, finiteness, extraction device, and elapsed time. Repeated identities may be de-duplicated internally only if reconstruction preserves exact task order and hashes.

## 6. Downstream comparison

SaURL contributes:

```text
1 representation x 3 tasks x 2 walks = 6 downstream trajectories
6 trajectories x epochs {5,15,50} = 18 evaluated snapshots
```

Together with LWA, Phase 6.7 has 12 new trajectories; the six immutable H0 task/walk trajectories remain the direct references.

For SaURL embeddings:

1. fit a coordinatewise embedding mean/std on the task's supervised training rows only;
2. replace standard deviations below `1e-8` by 1;
3. standardize and clip to `[-10,10]` using frozen training statistics;
4. construct the existing task head with input width 128 and hidden width 128; and
5. retain the established task-specific loss, output transform, batch size 512, learning rate `1e-4`, seed 0, and epochs 5/15/50.

The current Phase 6 standardizer accepts only widths 445 and 573. Phase 6.7 must implement a generic finite two-dimensional standardizer rather than weakening that completed contract in place.

## 7. Metrics and interpretation

Reuse the complete existing metrics:

- classification: accuracy, macro-F1, balanced accuracy, class metrics, confusion matrix, prediction counts, and collapse diagnostics;
- future price: MAE, RMSE/MSE, Pearson/Spearman, persistence skill, implied-movement correlations/sign agreement, global and timestamp-level Rank IC, and existing subgroups;
- volatility: MAE, RMSE/MSE, Pearson/Spearman, persistence/median/zero references, contract-macro, tail, and subgroup diagnostics; and
- resources: encoder/probe/total parameters, native width, pretraining/extraction/head time, inference time, and peak memory.

The allowed conclusion is local: under identical rows and probes, SaURL-F was stronger or weaker than H0 for a named task and walk. Native-width results do not isolate parameter count, and one task win does not imply universal superiority.

## 8. Leakage and fairness safeguards

- A walk's input scaler and all SaDA statistics are fitted only on its target-free encoder rows.
- Walk 2 cannot update or reinterpret Walk 1.
- Task target maturity is applied only by the existing task bundles, after encoder pretraining.
- Evaluation rows cannot select the source revision, RwAM definition, learning rates, checkpoint, fallback, or retry.
- If any established row yields a non-finite embedding, the run fails; the row set is not reduced.
- Projectors, predictors, EMA targets, and downstream labels never enter the frozen feature vector.
- Failed/collapsed trajectories and every checkpoint remain reportable evidence.

## 9. Artifact plan

```text
experiments/phase6_7/
  feasibility/saurl_ts/
  manifests/
  encoder_pretraining/saurl_frozen/walk{1,2}/seed0/
  features/saurl_frozen/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/saurl_frozen/walk{1,2}/seed0/
  reports/frozen_representation_seed0/
```

Reusable method code belongs under `src/baselines/saurl_ts/`. Shared Phase 6.7 training, feature, validation, and reporting logic belongs under `src/`. Orchestration belongs under `scripts_v6/`. Existing Phase 5/6 artifacts and scripts remain immutable.

## 10. Risks and resolutions

| Risk | Resolution |
|---|---|
| No software licence | independently author the adapter; never copy, modify, import, or vendor upstream code |
| RwAM missing from source | use the approved documented reconstruction and disclose it in reporting |
| 2024 source differs from 2026 paper | treat it only as behavioural evidence and report every selected deviation |
| Cross input was ambiguous | use temporal and reconstructed-frequency views of the same sample as the cross BYOL pair |
| Frequency reconstruction is implemented incorrectly | assert magnitude-only transformation, unchanged phase, explicit `n=64`, real finite output, and identity round-trip tolerance |
| SaDA destabilizes short OHLCV windows | report collapse/non-finite behavior; no evaluation-driven retuning |
| Learned views violate OHLC identities | views are transient; record diagnostics, but do not add financial constraints without an amendment |
| Batch size 32 is expensive | measure in the pre-evaluation resource smoke test; resource failure activates the documented fallback |
| Native 128 width gives a smaller head than H0 445 | retain native widths and report probe parameters as required by the plan |

## 11. Admission checklist

- [x] Public software licence absence recorded; direct upstream code reuse prohibited.
- [x] Audited reference commit pinned as evidence, not implementation source.
- [x] RwAM and final 128-dimensional extraction frozen by project-owner decision.
- [x] Frequency view path selected: magnitude-only transformation, original-phase preservation, inverse-FFT reconstruction before `E_F`.
- [x] Cross-domain input pair selected.
- [x] Exact SaDA/SaSSL alternating update schedule selected.
- [x] CPU synthetic `[4,64,5]` forward/backward/extract test passes.
- [ ] CUDA small real training-only batch test passes within the frozen resource budget.
- [ ] Candidate manifest is frozen before downstream metrics.
- [ ] If any required item fails by the deadline, `SISSEL-Frozen` is activated and recorded.
