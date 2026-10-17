# Phase 7A Ablation, Branch Overlap, and Complementarity Analysis Plan

**Date:** 2026-10-12  
**Status:** Superseded on 2026-10-17 by the trimmed Phase 7A branch-contribution plan. Retained as the fully specified deferred-extension design.  
**Branch:** `brainstorm/phase7-ablation-result-analysis`  
**Execution status:** This full A+B+C scope was not implemented or executed before supersession.

> **Current authority:** Read
> [the 2026-10-17 trimmed Phase 7A plan](2026-10-17-phase-7a-trimmed-branch-contribution-analysis-plan.md).
> The required scope is now Package A plus B-Core: 60 single/removal seed-0
> heads, six immutable H0 references, standardized classification-population
> CKA, and calendar-walk/contract/future-RV groups. Bootstrap uncertainty is
> optional future work and does not gate completion.
> Resource and error-overlap analyses are excluded. Pairwise probes and the heavier analyses below are deferred
> future extensions, not current completion requirements.

## 1. Purpose, authority, and scope

The central research question is:

> What does combining the five canonical representation branches contribute,
> what structures do those branches share, and under which conditions is their
> combination useful?

This document defines the experiments, estimators, artifacts, and interpretation
rules needed to answer that question. It supersedes
[the original Phase 7A plan](2026-09-26-phase-7a-representation-ablation-plan.md)
as the full ablation-analysis authority. The owner clarified that the September
document is the outdated original ablation study and this document is the more
complete replacement plan. The inherited single-branch and leave-one-out matrix
is one component of the current study, alongside structural diagnostics,
pairwise complementarity, task/condition analysis, uncertainty, and resources.
Packages A+B+C form the current planned scope; D and E remain optional.
Numerical settings below are the planning specification and must be recorded
in a versioned implementation manifest before analysis executes. No numerical
recipe change may be selected from the new study's evaluation results.

The original plan said Phase 7A follows Phase 6.8. The current
[schedule](../schedule.md) records this owner-adopted replacement and retains
Phase 6.8 as optional. This owner-directed replacement establishes the current
Phase 7A planning direction and supersedes the historical Phase 6.8 dependency.
Optional Phase 6.8 is not a prerequisite for the full ablation-analysis study.
Implementation and model execution remain separate work from adopting this plan.

The simple concat probe remains the primary representation experiment. The study
does not select a new encoder architecture, optimize a subset on evaluation data,
estimate causal feature importance, or authorize alpha discovery. Phase 6.6 raw
fusion/decoder studies and Phase 7B remain separate.

### 1.1 Work packages

| Package | Contents | Scope status | New neural-head trajectories |
|---|---|---|---:|
| A | Five single branches and five leave-one-out configurations, three tasks, two walks | Inherited experiment matrix within the current plan | 60 |
| B | Structural similarity, effective rank, reconstruction, descriptor probes, error overlap, subgroup reporting, uncertainty, resources | Required analysis package in the current plan | 0; auxiliary ridge fits are counted separately |
| C | All ten two-branch concatenations, three tasks, two walks | Required complementarity package in the current plan | 60 |
| D | Two deterministic duplicate-width controls for every pair | Optional capacity sensitivity | 120 |
| E | Complete A+C+H0 matrix at head seeds 1 and 2 beside seed 0 | Optional head-seed confirmation | 252 |

The current integrated scope is A+B+C. D and E are fully specified optional
follow-ups. Packages may execute in stages, but completing A alone, or A+B,
does not complete the full plan. Without C, practical pair complementarity
remains unresolved.

## 2. Research questions and evidence map

| ID | Research question | Main evidence | Required package |
|---|---|---|---|
| Q1 | What can each branch predict on its own? | Single-branch task performance and skill against non-learned references | A |
| Q2 | What changes when each branch is removed from H0? | Paired H0-versus-removal metric differences | A |
| Q3 | Which branches overlap, and which structures are shared? | CKA, spectra/effective rank, directional reconstruction, descriptor probes | B |
| Q4 | Do different branches provide useful complementary information? | Error overlap plus pairwise gains over both singles | B+C; D strengthens capacity interpretation |
| Q5 | Is branch usefulness task-specific? | Complete branch-by-task contribution tables | A; C adds pair specialization |
| Q6 | Does usefulness depend on the walk, contract, or market conditions? | Paired subgroup, contract, and calendar-block results | A+B+C |
| Q7 | Are the observed gains reliable and worth their cost? | Paired uncertainty, resource tables, performance-cost comparisons | B; D/E address additional uncertainties |

Structural diversity, downstream usefulness, and complementarity are different
properties. A branch can have low similarity to the others and still contain
mostly noise. A branch can resemble another globally while retaining useful
task-specific detail. None of Q3's metrics replaces Q1/Q2/Q4.

## 3. Definitions and notation

- A **branch** is one frozen feature block computed for a historical OHLCV context.
- A **probe** is a separately trained downstream head on fixed branch features.
- A **trajectory** is one uninterrupted 50-epoch head-training run.
- A **snapshot** is the checkpoint, predictions, and metrics at epoch 5, 15, or 50.
- **H0** is the immutable five-branch canonical concat representation and its
  audited task/walk head; it is not the best observed internal variant.
- A **single-branch experiment** uses only one block and measures standalone usefulness.
- A **leave-one-out experiment** retrains the probe on the remaining four blocks.
  It is not inference-time masking of an already trained H0 head.
- **Structural overlap** means similar sample geometry or linearly recoverable
  feature variation. It does not mean identical task information.
- **Practical complementarity** means a combined probe improves a declared task
  metric over both individual probes. It does not establish information-theoretic synergy.
- **Held-out** means the evaluation rows were not used to fit that model or mapping.
- Let `X_j` be branch `j`'s feature matrix, `N` its row count, and `d_j` its width.
- Let `p_t` be the raw close probability at the decision endpoint, `y_i` a true
  task target, `yhat_i` a prediction, and `e_i = yhat_i - y_i` a signed residual.
- `M` denotes a higher-is-better metric; `E` denotes a lower-is-better error metric.

The canonical branch order is always `S,T,V,C,B`:

| Symbol | Stored branch key | Meaning | Width | H0 coordinate interval, zero-based and half-open |
|---|---|---|---:|---|
| S | `statistical` | AR coefficients/residual summaries and GARCH features | 70 | `[0,70)` |
| T | `transformed` | FFT magnitudes and Haar detail energies | 55 | `[70,125)` |
| V | `vae` | VAE latent mean | 64 | `[125,189)` |
| C | `contrastive` | Contrastive CNN backbone embedding | 128 | `[189,317)` |
| B | `byol` | BYOL online CNN backbone embedding | 128 | `[317,445)` |
| H0 | All five keys | Canonical concat | 445 | `[0,445)` |

Use named branch arrays and audited manifests to select coordinates. The
intervals are a verification aid, not permission to ignore branch metadata.

## 4. Data, tasks, and fitting boundaries

### 4.1 Walks and information allocation

| Walk | Training interval, UTC and half-open | Evaluation interval, UTC and half-open |
|---:|---|---|
| 1 | `[2025-12-02,2026-04-01)` | `[2026-04-01,2026-06-16)` |
| 2 | `[2026-02-16,2026-06-16)` | `[2026-06-16,2026-09-01)` |

Reuse the accepted clean native one-hour cohorts, 64-hour contexts, isolated
one-hour bounded filling, causal activity rules, observed decision/target
endpoints, and longer-gap sequence breaks. Retrospective cohort selection and
approved offline pruning remain the declared research assumptions.

Each walk owns its encoder checkpoints, feature stores, scalers, heads, and
auxiliary mappings. Walk 2 can train on history that was evaluation data for
Walk 1 because it is historical at the second cutoff. That state never updates
Walk 1's model or predictions. There is no validation split, early stopping,
evaluation-driven tuning, or best-checkpoint selection.

### 4.2 Task and source inventory

Paths below are relative to the repository root; `{w}` is 1 or 2.

| Task ID | Target | Label/data bundle | Canonical feature store |
|---|---|---|---|
| `classification_h2` | DOWN if `p[t+2]-p[t] < -0.001`; STABLE if absolute change is at most 0.001; UP if change exceeds 0.001 | `experiments/phase5/data_preparation/walk{w}/market_1h_seq64_h2.npz` | `experiments/phase5/features/walk{w}/five_branch_epoch50.npz` |
| `absolute_price_h8` | `p[t+8]` | `experiments/phase5/downstream_addons/shared/h8/data/walk{w}/market_1h_seq64_h8.npz` | `experiments/phase5/downstream_addons/shared/h8/features/walk{w}/five_branch_epoch50.npz` |
| `realised_variance` | `sum(k=1..8, (p[t+k]-p[t+k-1])^2)` | `experiments/phase6/volatility_prediction/data_preparation/walk{w}/volatility_1h_seq64_h8.npz` | `experiments/phase6/volatility_prediction/features/walk{w}/five_branch_epoch50_h8.npz` |

The realised-variance target is variance in squared probability units, not its
square root and not a log-return volatility proxy. Every component future close
obeys the frozen observed/consecutive/maturity rules.

Reuse these H0 runs only after independent source/recipe/prediction replay:

- Classification: `experiments/phase5/downstream/walk{w}/classification/seed0/`.
- Price: `experiments/phase5/downstream_addons/tasks/absolute_price_h8/walk{w}/seed0/`.
- Variance: `experiments/phase6/volatility_prediction/runs/framework_h0/walk{w}/seed0/`.

Record actual counts and hashes from replayed manifests, not copied planning
counts. Row identity includes walk, condition ID, window-start timestamp,
decision timestamp, decision availability, and the task's target identity.

All models within a task/walk use identical ordered rows and targets. The three
tasks have different eligibility populations: never compare their raw scores
as though they came from one shared prediction problem. Primary structural
diagnostics use the classification feature population; fixed price/variance
population repeats are reported as task-population sensitivities, not new samples
independent of the primary population. Existing supervised eligibility is reused
without additional label-based filtering.

## 5. Experiment matrices and training contract

### 5.1 Package A: inherited single/removal matrix

| ID | Included branches | Width |
|---|---|---:|
| `A-S` | S | 70 |
| `A-T` | T | 55 |
| `A-V` | V | 64 |
| `A-C` | C | 128 |
| `A-B` | B | 128 |
| `A-noS` | T,V,C,B | 375 |
| `A-noT` | S,V,C,B | 390 |
| `A-noV` | S,T,C,B | 381 |
| `A-noC` | S,T,V,B | 317 |
| `A-noB` | S,T,V,C | 317 |

Inventory: `10 x 3 tasks x 2 walks = 60` new trajectories, 180 new snapshots,
plus six reused H0 trajectories and their 18 snapshots. Core reporting has 66
task/walk/configuration entries. No encoder is retrained.

### 5.2 Package C: pairwise complementarity matrix

| ID | Included branches | Width |
|---|---|---:|
| `P-ST` | S,T | 125 |
| `P-SV` | S,V | 134 |
| `P-SC` | S,C | 198 |
| `P-SB` | S,B | 198 |
| `P-TV` | T,V | 119 |
| `P-TC` | T,C | 183 |
| `P-TB` | T,B | 183 |
| `P-VC` | V,C | 192 |
| `P-VB` | V,B | 192 |
| `P-CB` | C,B | 256 |

Inventory: 60 additional trajectories and 180 additional snapshots. A+C together
have 120 new trajectories, 360 new snapshots, and six reused H0 trajectories;
the combined report has 126 entries and 378 snapshots including H0. Run every
pair on every task and walk; CKA or earlier metrics do not select the pairs.

### 5.3 Shared probe recipe

- Seed 0; Adam; learning rate `1e-4`; weight decay 0; batch size 512;
  shuffle training rows deterministically and keep the remainder batch.
- One continuous 50-epoch run; snapshots at 5/15/50; epoch 50 principal.
- Coordinate mean and population standard deviation fit on the complete
  supervised training population for that task/walk.
- For coordinate `k`, use scale `s_k = 1` if training std is below `1e-8`,
  otherwise use the actual std. Apply `(x_k-mu_k)/s_k`, then clip to `[-10,10]`.
  Save the mean, raw std, scale, clip limits, and array hashes.
- Preserve constant coordinates in downstream probes to match the original
  feature contract. Report their counts and clipping fractions.
- Concat has no learned parameters. The head uses `d -> 128 -> 64 -> output`
  with the task's exact existing activations and regularization.

| Task | Output/loss | Head details |
|---|---|---|
| Classification | Three logits; training-prior logit-adjusted cross-entropy, strength 1 | GELU; dropout 0.2 after first hidden activation; no evaluation-time logit adjustment added |
| Price | Scalar sigmoid; endpoint MSE in raw probability units | GELU; dropout 0.1 after first hidden activation; no current-price skip |
| Variance | Scalar Softplus; Smooth L1, beta 1, on `10000 * RV` | GELU; no dropout; gradient norm clip 5; divide predictions by 10000 for metrics |

Preserve all other audited task-specific optimizer and gradient behavior from
the frozen source recipe; do not silently add clipping or regularization.
Generalize reusable loaders/scalers to declared widths rather than changing
their semantics. Fresh heads are trained for every new configuration.

For classification let `pi_k` be class k's frequency on the complete task-training
rows. The training loss is `CE(logits + log(pi), target)`; inference uses the
original logits. For variance, with scaled residual `r`, Smooth L1 at beta 1
is `0.5*r^2` when `abs(r)<1`, otherwise `abs(r)-0.5`, averaged over the batch.
Classification and price add no gradient clipping; variance clips norm to 5.

For output width `o`, the listed head has
`128*d + 8384 + 65*o` trainable parameters. Verify by counting model parameters;
record output transforms and dropout even though they add no parameters.

## 6. Metric definitions shared by Q1, Q2, Q4, Q5, and Q6

### 6.1 Regression metrics

For `e_i = yhat_i-y_i` on the specified evaluation group:

```text
MAE  = mean(abs(e_i))
MSE  = mean(e_i^2)
RMSE = sqrt(MSE)
bias = mean(e_i)
```

Pearson is the correlation between predictions and targets. Spearman is Pearson
correlation between average-tie ranks. Correlation is undefined for constant
arrays or fewer than two rows; save an explicit null and reason, not zero.
Primary price and variance error metrics are MAE and RMSE; MSE, bias, and
correlations support interpretation. Report raw units throughout.

### 6.2 Classification metrics

Class order is `DOWN,STABLE,UP`. Compute probabilities using softmax of the
unadjusted inference logits and predict with argmax, retaining the source
pipeline's deterministic tie convention.

```text
precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 * precision_k * recall_k / (precision_k + recall_k)
macro-F1    = mean(F1_DOWN, F1_STABLE, F1_UP)
balanced accuracy = mean(recall_DOWN, recall_STABLE, recall_UP)
accuracy    = correctly classified rows / all rows
weighted-F1 = sum(class support_k * F1_k) / N
NLL         = -mean(log(max(probability assigned to true class, 1e-12)))
Brier       = mean(sum_k((probability_ik - one_hot_target_ik)^2))
```

Use zero for a zero-denominator per-class precision/F1 as in the existing
implementation. Overall evaluation must retain all three observed classes;
record class counts. In subgroup reporting retain the fixed-three-class scores
with missing-class flags, and additionally show observed-class-only balanced
accuracy. Do not treat a subgroup missing classes as evidence of improvement
on the full three-class problem. One-vs-rest ROC-AUC is undefined when a class
has no positives or negatives; average precision is undefined with no positives.
Reuse the source metric code, including average precision rather than silently
substituting a different PR-curve area convention.

Macro-F1 and balanced accuracy are primary. Required supporting outputs are
accuracy, weighted-F1, per-class precision/recall/F1, confusion matrix, observed
and predicted counts, one-vs-rest ROC-AUC/average precision, NLL, and Brier.
NLL/Brier describe probability quality; no post-hoc recalibration is fitted.

### 6.3 Future-price movement diagnostics

```text
predicted movement = predicted future close - current close
actual movement    = actual future close - current close
nonzero sign agreement = mean(sign(predicted movement) == sign(actual movement))
                         on rows whose actual movement is nonzero
```

Report implied-movement Pearson/Spearman and all-row sign agreement separately
from the nonzero-target sign agreement. Sign of exact zero is zero; do not tune
an evaluation-dependent deadband.

At each decision timestamp, calculate cross-sectional Rank IC as Spearman
between predicted and actual movement across eligible conditions. Require at
least five distinct conditions. Reuse the existing minimum-five contract; rank
ties by their average rank. Exclude undefined ICs with a reason and count.

For paired comparisons require the same usable timestamp set for every member
of the declared comparison. Report each model's individual timestamp coverage
as well: a constant model cannot gain apparent skill by selectively losing
timestamps. Report mean/median IC, its standard deviation, positive-IC fraction,
and `mean(IC)/std(IC)` as an unannualized descriptive ratio. Do not present this
ratio as a trading Sharpe ratio or independent-sample significance statistic.

### 6.4 Non-learned references

| Task | References and definitions |
|---|---|
| Classification | Always STABLE for class metrics; logits `[-30,0,-30]` in DOWN/STABLE/UP order for probability metrics, disclosed as implementation-defined; repeated training class priors for probabilities and their argmax for class metrics |
| Price | Current-price persistence `yhat=p_t`; fixed last-hour reversal score `-(p_t-p[t-1])` for movement ranking only |
| Variance | Zero; median of that walk/task's training RV; historical RV over the last eight historical intervals, using nine context closes |

Historical RV may include bounded-filled context prices; label it accordingly
and show observed-only-context results. Never impute future RV components.
Persistence implies a constant zero movement score, so its movement correlation
and Rank IC are undefined. Last-hour reversal is a ranking reference, not an
eight-hour price forecast with a fitted multiplier.

For each error reference with `E_ref > 0`, report
`skill = 1 - E_model/E_ref`. Positive means improvement. Record undefined skill
when the reference error is zero. For classification use absolute metric
differences, including percentage points, rather than relative percentages.

## 7. Q1 and Q2: standalone and incremental usefulness

### 7.1 Standalone measurement

For each single branch, save all task metrics and reference-relative skill for
every task/walk/snapshot. Compare with H0 descriptively as well. A narrower
branch need not beat H0 to contain useful predictive information.

### 7.2 Removal contribution

For every branch and metric define a benefit-positive contribution:

```text
Delta_j(M) = M(H0) - M(H0 without j)      # higher-is-better metric
Delta_j(E) = E(H0 without j) - E(H0)      # error metric
```

Report absolute differences; for error metrics additionally divide the error
difference by `E(H0)` when positive/nonzero to obtain relative degradation.
Compute metrics from complete paired prediction arrays, not from an average
of per-row F1 scores. Recompute nonlinear metrics within each resampling draw.

Deliver single-branch tables and removal plots, with direction labels, feature
width, parameters, counts, and pointwise uncertainty. A negative contribution
means removal improved the observed metric; it does not prove harmful information.

## 8. Q3: structural overlap and feature structures

### 8.1 Populations, samples, and normalization

Compute the primary diagnostics on the classification feature population in
each walk, separately for training and evaluation. Repeat the declared
diagnostics on price and RV populations as fixed population sensitivities.
Never pool walks or mix training/evaluation into one structural matrix.

For CKA and spectra use up to 4096 rows per task/walk/split. Sampling procedure:

1. Sort condition IDs and rows by their stable identity fields.
2. Seed a NumPy PCG64 generator with 170012; record its version and the
   ordered processing sequence in the manifest: task order classification,
   price, variance; within each task walk 1 then 2, and training then evaluation.
3. Permute row positions without replacement within each condition.
4. Select one row per condition in sorted round-robin order until 4096 rows
   or the population is exhausted; continue past exhausted conditions.
5. Freeze sampled row IDs and hashes. Every branch uses exactly that sample.

The balanced diagnostic sample represents condition diversity; downstream
metrics still use every frozen evaluation row with their natural frequencies.
Save per-condition sample counts and temporal coverage. Adjacent/overlapping
windows can remain; the sample is not declared independent.

Primary structural features are the task-training-standardized, clipped
coordinates used by downstream heads. Raw unscaled branch arrays are a fixed
sensitivity because standardization and clipping change geometry. Remove
coordinates whose full-training std is below `1e-8` from structural and ridge
diagnostics only; save the excluded coordinate list. Selection never uses
evaluation variance. A constant whole branch yields undefined similarity/rank,
with a collapse flag. Sample centering of an evaluation matrix for a descriptive
covariance/CKA calculation does not refit the predictive scaler.

### 8.2 Linear centered kernel alignment (CKA)

For aligned centered matrices `X` and `Y`:

```text
CKA(X,Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```

`||.||_F` is the square root of the sum of squared matrix entries. Compute in
float64 using the existing `centered_linear_cka` implementation. CKA compares
sample relationships across different widths and is invariant to orthogonal
coordinate rotations and one global nonzero scale; it is not invariant to
arbitrary per-coordinate scaling. Higher CKA means more similar geometry,
not a percentage of shared information.

There are ten unique pairs. Across three populations, two walks, and two splits,
report 120 standardized pair scores and 120 raw sensitivities. Diagonal entries
are one only for noncollapsed branches. Do not impose an arbitrary high/low
CKA cutoff, choose pairs from CKA, or interpret low CKA as task relevance.

### 8.3 Effective rank and variance spectra

Effective rank measures how broadly a branch distributes variance across
directions. It is not its embedding width, an integer count of independent
features, or a predictive-quality score.

Let `Sigma = X_centered^T X_centered/(N-1)` and let its nonnegative eigenvalues
be `lambda_k`. Define the covariance-entropy version explicitly:

```text
p_k = lambda_k / sum_j(lambda_j)
effective_rank = exp(-sum_k(p_k * log(p_k)))
normalized_effective_rank = effective_rank / retained_coordinate_count
```

Use natural logs and `0*log(0)=0`. Remove only tiny negative eigenvalues caused
by roundoff: the tolerance is `1e-12 * max(1,max(abs(lambda)))`; set a
negative value within that tolerance to zero, record its magnitude, and fail
on a value below the negative tolerance.
If total variance is zero, report undefined rank and a collapse diagnostic.
For nonzero variance, rank lies between 1 and the available matrix rank.

Also save the complete sorted normalized eigenvalue spectrum and the smallest
number of directions explaining 90% and 95% of variance. Compute 60 spectra
and rank summaries for standardized features and 60 raw sensitivities.
This is the entropy effective rank of the covariance matrix; applying the
entropy formula to feature-matrix singular values instead would be a different
estimator and must not be substituted silently.

Interpretation examples: exact copies of one varying signal give rank 1;
equally strong uncorrelated directions give rank equal to their count. High
rank can reflect noise. Low rank can reflect useful compression.

### 8.4 Directional held-out reconstruction

For each ordered distinct branch pair `A -> B`, fit a fixed ridge map on the
complete task-training population:

```text
minimize_(W,b) (1/N_train) * ||Z_B - Z_A W - b||_F^2 + alpha * ||W||_F^2
alpha = 0.01; intercept b is not penalized
```

`Z` denotes the standardized/clipped retained coordinates. Solve in float64;
center training inputs/targets to fit the unpenalized intercept, then retain
the fitted coefficients and intercept. No validation or alpha sweep is used.
Using an unnormalized sum of errors requires the equivalent penalty
`N_train * alpha`; record the objective to prevent library-convention drift.

Predict B from evaluation A, without refitting any state. For target coordinate k:

```text
SSE_k = sum_i((predicted_B_ik - actual_B_ik)^2)
SSE_mean_k = sum_i((training_mean_B_k - actual_B_ik)^2)
skill_k = 1 - SSE_k/SSE_mean_k
```

Save per-coordinate MSE and skill, the equal-coordinate mean skill, and a
pooled skill `1 - sum(SSE_k)/sum(SSE_mean_k)`. These differ: pooled skill weights
coordinates by baseline error. Undefined denominators are flagged, never set
to zero. Compute the training mean after clipping, since it need not be zero.
Show training reconstruction diagnostically and held-out reconstruction as the
primary result. No clipping is added to the reconstructed outputs.

There are 20 directed maps per task/walk, hence 120 planned fits across all
three populations and both walks. Show the classification-population results
first and population sensitivities separately. Mapping A to B need not have
the same skill as mapping B to A. Skill 0.8 means 80% lower squared error than
the specified mean predictor, not 80% shared information or predictive alpha.

### 8.5 Named feature-block overlap

The deterministic blocks permit interpretable substructure diagnostics without
new downstream subset training:

- S consists of five 14-coordinate OHLCV-column blocks. Within each block,
  offsets `[0,7)` are five AR coefficients plus residual mean/std; `[7,14)`
  are seven GARCH quantities. Combined AR/residual width is 35; GARCH width is 35.
- T consists of five 11-coordinate blocks. Offsets `[0,8)` are FFT magnitude
  bins; `[8,11)` are three Haar detail energies. Combined FFT width is 40;
  Haar width is 15.
- The current FFT extractor keeps the first eight rFFT magnitude bins, including
  DC; it does not select the eight largest magnitudes. Haar stores mean squared
  details at three scales. Match the implemented source semantics.

Compare each of these four named subblocks with V,C,B using CKA and the same
samples/scaler coordinates: 12 comparisons per task/walk/split, 144 standardized
scores in total. These explain location of overlap; no AR-only/GARCH-only/
FFT-only/Haar-only downstream experiment is added to Package A.

### 8.6 Descriptor probes: what structures are encoded?

Use the classification population as the common descriptor population. Derive
the following ten continuous descriptors only from each saved historical raw
context and its metadata; no future prices, task labels, or eventual resolution
are descriptor inputs or targets.

| Descriptor | Exact definition |
|---|---|
| Current price level | Last context close `p_t` |
| Context price level | Mean of the 64 context closes |
| Last-hour movement | `p_t-p[t-1]` |
| Historical eight-hour movement | `p_t-p[t-8]` |
| Historical eight-hour RV | Sum of squared changes over the last nine closes |
| Recent volume | `log1p(sum(volume over last 24 context bars))` using raw volume |
| Recent update fraction | Fraction of the last 24 hourly close differences that are nonzero; requires the last 25 closes |
| Recent candle range | Mean `high-low` over the last 24 bars |
| Imputation exposure | Number of imputed context bars divided by 64 |
| Age of last close change | Number of consecutive zero close differences ending at the endpoint, capped at 63; a fully flat context receives 63 |

Filled historical prices are retained consistently. Metadata distinguishes
imputation from an observed zero-change candle; the two must not be conflated.

For each branch and H0, fit one multi-output ridge map to the ten descriptors
on training rows using the same normalized objective and alpha 0.01. Fit
descriptor means/stds on training rows; do not clip descriptor targets. Invert
the target scaling before reporting descriptor-unit MAE, mean-predictor skill,
and Spearman per descriptor. A training-constant descriptor is unavailable and
is listed explicitly. Primary fits: six representations x two walks = 12;
report every descriptor, even weak or negative results.

Descriptor predictability means the information is linearly accessible to this
probe. It does not identify a latent coordinate's unique meaning or establish
why the future-task head performs well. No task-conditioned decoder tuning,
neural reconstruction model, or nonlinear mutual-information estimator is added.

## 9. Q4: complementarity in errors and paired probes

### 9.1 Prediction-error overlap

For all ten single-branch pairs on every task/walk:

- Regression: report Pearson/Spearman of signed residuals and Pearson of
  absolute errors. High residual correlation means common errors, but shared
  targets and bias can induce it; low correlation is only a diagnostic.
- Classification: let `w_A` indicate an incorrect class prediction. Report
  `mean(w_A*w_B)` (both wrong), `mean((1-w_A)*w_B)` (A-only correct),
  `mean(w_A*(1-w_B))` (B-only correct), and both-correct frequency. The four
  proportions sum to one. Also report error Jaccard
  `count(both wrong)/count(at least one wrong)` and predicted-class disagreement.
- Calculate classification overlap overall and by true class so the STABLE
  majority does not conceal different minority-class errors. Undefined Jaccard
  when neither errs is flagged. Report error rates beside overlap: different
  errors from weak predictors are not automatically useful.

This produces 60 pair/task/walk overlap records plus class strata. Do not
fit an ensemble using evaluation outcomes or report a hindsight oracle as an
achievable predictor.

### 9.2 Pairwise performance gains

Train every Package C pair under Section 5.3. Compare it with both singles,
not just with the weaker one. For each declared metric:

```text
gain_pair(M) = M(A+B) - max(M(A), M(B))
gain_pair(E) = min(E(A), E(B)) - E(A+B)
```

Also report the two individual signed gains, pair-versus-H0 differences, and
error-relative gain against `min(E(A),E(B))` when nonzero. The best-single
envelope is a descriptive comparator, not one test-selected deployable model.
Bootstrap the min/max again inside each draw; do not fix its identity after
looking at the full evaluation score.

Positive pair gain supports practical complementarity for that metric/task/walk.
It includes increased input width and first-layer capacity. It is not a formal
synergy estimate, cannot be added across pairs, and need not imply that the same
pair helps the other tasks or walks.

### 9.3 Optional Package D: deterministic duplicate-width controls

For pair `A+B` of width `d=d_A+d_B`, construct two control views:

```text
D-AB-A[:,k] = A[:,k mod d_A] for k=0..d-1
D-AB-B[:,k] = B[:,k mod d_B] for k=0..d-1
```

The first copy appears in canonical coordinate order and remaining coordinates
repeat cyclically. This introduces no new source information and matches the
pair's input width, first-layer shape, total head parameters, and task recipe.
Prove exact coordinate duplication and freeze the coordinate map before fitting.

Two controls x ten pairs x three tasks x two walks = 120 new trajectories and
360 snapshots. A+C+D totals 240 new trajectories and 720 snapshots plus six
reused H0 trajectories/18 snapshots; reporting has 246 entries.

Compare each pair with both controls, using benefit-positive differences.
Duplication changes feature weighting and optimization, so these are informative
but imperfect capacity controls. Even a win over both does not establish causal
information synergy. This package is outside the original frozen Phase 7A core.

## 10. Q5 and Q6: task specialization and conditional usefulness

### 10.1 Task specialization

Create one branch-by-task table per walk for standalone skill and removal
contributions. If C runs, create analogous pair-by-task tables. Use percentage
points for classification, relative error changes for MAE/RMSE, and absolute
changes for Rank IC. Do not average them into one cross-task score.

Report the complete matrix. A branch helping price MAE but hurting movement
Rank IC represents different strengths, not one net improvement. Task-population
differences must accompany any claim of task specialization.

### 10.2 Predeclared reporting groups

Recompute model and reference metrics on exactly the same subgroup rows.

| Grouping | Rule |
|---|---|
| Walk | Report Walk 1 and Walk 2 separately before pooling |
| Contract | Every condition ID; natural row-weighted and equal-contract summaries |
| Starting probability | `p<=0.01`, `0.01<p<=0.10`, `0.10<p<0.90`, `0.90<=p<0.99`, `p>=0.99`; disjoint and exhaustive |
| Lifecycle | Existing source codes unknown/early/middle/late; retrospective reporting only |
| Context imputation | Zero versus at least one imputed context row; also report fraction continuously |
| Activity | Training tertiles of historical 24-hour update fraction |
| Historical variance | Training tertiles of historical eight-hour RV |
| Volume | Training tertiles of log recent raw volume |
| Calendar | Consecutive seven-day blocks anchored at that walk's evaluation start; retain final partial block |
| Realised target RV | Below training 90th percentile, 90th-to-99th, at/above 99th; target strata are retrospective diagnostics only |

Training quantiles use NumPy's linear quantile convention on the complete task-
training population. Store edges. Use lower-inclusive/upper-exclusive intervals
apart from the final inclusive interval; collapsed edges produce explicitly
empty groups, not evaluation-driven alternative cutoffs. Source lifecycle
metadata and future-target RV strata never enter model inputs or eligibility.

Show row, contract, timestamp, calendar-block, and class counts as applicable.
Keep point estimates for every nonempty group. Mark groups with fewer than 100
rows, three conditions, or five populated calendar blocks as sparse; do not
promote them to broad conclusions. Per-contract tables are naturally single-
condition and are interpreted as case-specific evidence. Correlations and
class metrics have the additional availability rules in Section 6.

Contract-macro means an equal-weight average of the per-contract metric,
excluding only undefined values with counts/reasons. Macro RMSE is the mean
of contract RMSEs, not the square root of macro MSE. Classification contract
summaries retain missing-class flags; show the all-three-classes subset beside
the complete descriptive table. No result-based condition exclusion is allowed.

For subgroup Rank IC, recompute using the conditions available within that
subgroup at each timestamp and retain the minimum-five rule. Report unavailable
groups rather than lowering the minimum to obtain an IC.

### 10.3 Tail-error concentration

For each training-defined RV tail threshold `q` and each variance predictor:

```text
tail_SSE_share(q) = sum_(y_i>=q)(e_i^2) / sum_all(e_i^2)
```

Report tail row fraction, tail MAE/RMSE/bias, and tail SSE share at the training
90th and 99th percentiles. Zero total SSE gives undefined share. Keep full-target
metrics primary: neither tail rows nor difficult contracts are removed or
winsorized. This explains why RMSE can change little while typical-case MAE differs.

### 10.4 Pooling

Pool only already generated out-of-future predictions after walk-local results
are complete. Include walk in identities; verify that evaluation intervals are
disjoint and each condition/timestamp contributes once within its walk. Recompute
metrics from concatenated arrays; do not average walk RMSE or F1 as a substitute.
Report pooled natural row-weighted metrics and the equal-walk average separately.
For contract-macro pooling use `(walk,condition)` groups and disclose repeated
conditions across walks. A pooled gain cannot conceal an opposite walk-local effect.

## 11. Q7: uncertainty, optimization, and practical cost

### 11.1 Paired temporal-block intervals

The primary uncertainty procedure is synchronous calendar-block resampling,
retaining all conditions together within a selected block. This respects shared
calendar events and preserves cross-sectional samples for Rank IC. It estimates
temporal evaluation-sample variability conditional on these cohorts and fitted
models; it does not measure encoder/head seed variability or random-market sampling.

Fixed procedure to materialize in the implementation manifest:

1. Partition the full evaluation interval into nonoverlapping seven-day blocks
   anchored at its UTC start. Retain the final partial block and empty blocks.
2. For each of 2000 draws, sample the original number of blocks with replacement
   using PCG64 seed 170013. Use one draw manifest for all configurations in that
   task/walk and retain every row of each sampled block, including multiplicity.
3. Recompute the relevant metric and paired difference on the sampled rows.
   Pair gain re-evaluates its min/max comparator within each draw.
4. For Rank IC use `(draw-block-occurrence,original timestamp)` as the timestamp
   key so repeated blocks are replicated rather than merged into duplicate
   conditions at one timestamp. Apply the common-valid timestamp rule.
5. Take the 2.5th/97.5th percentiles of valid paired differences as a pointwise
   95% percentile interval. Save the full draw differences or replayable hashes.
6. Repeat at fixed block lengths of three and fourteen days as dependence-
   sensitivity analyses; never choose the interval that gives a preferred result.

Block duration exceeds the forecast horizons, but overlapping 64-hour contexts
and longer market dependence remain. These approximate intervals are descriptive,
especially with few calendar blocks. Report number of blocks and valid draws;
if fewer than five blocks contain rows or fewer than 90% of draws yield a defined
metric, withhold the interval and explain why. Keep the point estimate.

For pooled metrics resample within each walk separately and then concatenate,
preserving walk-local fitting. Do not independently resample timestamp rows,
cross-sectional members, or model predictions.

Intervals are pointwise across many comparisons, not simultaneous confidence
statements. No formal family-wide significance or equivalence claim is part of
this plan. Such claims require a predeclared multiplicity procedure and, for
equivalence, practical margins. An interval spanning zero does not prove redundancy.

### 11.2 Contract heterogeneity

Report paired metric differences for every contract, their median/interquartile
range, and the fraction with benefit-positive differences. These summarize
heterogeneity rather than independent repeated experiments. Keep temporal-block
uncertainty separate from any later contract-sampling inference; Rank IC should
not be bootstrapped by duplicating contracts within timestamps.

### 11.3 Training progression and optional Package E

Show all 5/15/50 snapshot metrics and training losses, preserving epoch 50 as
primary. Compare training/evaluation predictions in deterministic evaluation
mode if describing a generalization gap; a dropout-active batch training loss
is not directly equivalent to an evaluation-mode score.

Package E repeats the complete 21-configuration A+C+H0 matrix at head seeds
1 and 2: `21 x 3 tasks x 2 walks x 2 seeds = 252` new trajectories and 756
snapshots. Seed-0 H0 references remain immutable. Summarize three-seed means,
sample standard deviations, paired seed differences, and sign consistency.
No promising subset is selected from seed-0 evaluation. D is excluded from E
unless a separate complete seed/control matrix is declared.

All encoder features remain frozen from their original seed-0 fits. E estimates
head initialization/minibatch variability conditional on those features, not
full-pipeline seed robustness. Encoder-seed confirmation remains separate scope.

### 11.4 Resource measurements

For every representation/probe report:

- Feature width, retained diagnostic width, constant-coordinate count, head
  parameter count, and logical uncompressed bytes per row (`4*d` for float32).
- Actual feature-store bytes, disclosing compression/shared-view reuse so
  physical storage is not confused with logical width.
- Total head-training time from complete histories and peak allocated GPU memory.
- Feature extraction/pretraining cost only from traceable existing logs;
  missing measurements are unavailable, not inferred zero. Reusing features
  makes this stage's incremental extraction cost zero, not the method's total cost.
- Head-only inference and separately available feature-plus-head inference.
  Use one fixed device/software environment, resident input tensors, batch 512,
  evaluation mode, ten warm-up batches, and thirty measured batches. Synchronize
  CUDA immediately before/after measurement, include the actual row count, and
  report median throughput, median batch latency, and peak allocated memory.
  The benchmark seed and sampled rows are fixed. If CPU is used, identify it
  explicitly and do not compare its timings directly with GPU timings.

Plot MAE/RMSE or macro-F1 against parameters, time, and memory independently.
Define descriptive Pareto dominance as no worse declared performance and no
greater cost, with at least one strict improvement. Uncertain/negligible gaps
do not establish a robust Pareto ordering. Do not collapse performance and
cost into an arbitrary weighted score.

## 12. Interpretation and claim boundaries

| Pattern | Supported interpretation | What it does not establish |
|---|---|---|
| Single branch beats a reference | Standalone usefulness under this probe/data protocol | Universal forecasting superiority |
| Removing a branch worsens H0 | Positive practical incremental contribution | Parameter-matched causal feature importance |
| Strong single; small removal difference | Evidence consistent with substitution/redundancy | Proof the branch has no unique information |
| Weak single; positive removal contribution | Evidence consistent with conditional usefulness | Formal interaction or Shapley value |
| High CKA/high reconstruction | Shared geometry/linearly recoverable feature variation | Identical task-relevant information |
| Low CKA/low reconstruction | Structural differences under these estimators | Useful complementary signal |
| Pair beats both singles | Practical pair complementarity for that metric/task/walk | Capacity-independent information synergy |
| Pair also beats both duplicate controls | Stronger evidence beyond these width controls | A perfectly isolated causal mechanism |
| Removal improves H0 | Possible interference, regularization, or optimization effect | Intrinsically harmful source information |
| Effect reverses across walks | Cohort/time sensitivity needing explanation | One stable branch ranking |
| High effective rank | Broad variance distribution | More useful features; noise can also raise rank |

Use numerical effect sizes rather than uncommitted labels such as strong,
negligible, or equivalent. Report consistency across walks and tasks without
treating two walks as independent model-seed replications. Preserve negative
results and all predeclared cells. No test-driven subset is promoted as a new
model without a separately specified fresh evaluation.

## 13. Artifact and report contract

Use the existing root `experiments/phase7/ablation/`, with separate package
namespaces and one manifest relating all dependencies:

```text
experiments/phase7/ablation/
  manifests/                         # sources, rows, config matrix, analysis settings
  features/walk{1,2}/                # exact branch views and coordinate maps
  downstream/<task>/<config>/walk<w>/seed<s>/
    config.json
    dataset_manifest.json
    architecture_manifest.json
    feature_standardizer.npz
    history.npz
    e{5,15,50}/                       # checkpoint, predictions, metrics
    replay_validation.json
  analysis/
    structural_samples/<task>/walk<w>/<split>/
    cka/<task>/walk<w>/
    spectra/<task>/walk<w>/
    reconstruction/<task>/walk<w>/<source>_to_<target>/
    descriptors/walk<w>/<representation>/
    error_overlap/<task>/walk<w>/
    uncertainty/<task>/walk<w>/
    resources/
  reports/canonical_five_branch_seed0/
  reports/branch_structure_seed0/
  reports/pairwise_complementarity_seed0/   # only if C executes
  reports/duplicate_width_controls_seed0/  # only if D executes
  reports/head_seed_confirmation/          # only if E executes
```

Every prediction file retains ordered identities, targets, predictions, and
classification probabilities/logits or regression-unit metadata. Every auxiliary
fit retains its coefficients, training-derived state, source hashes, objective,
predictions, and metric definitions. JSON uses null plus a reason for undefined
metrics; CSV includes availability flags. No invalid metric is silently imputed.

Required report products:

1. Absolute metrics for every task/walk/configuration and every snapshot.
2. Single-branch reference skill and all H0 removal differences.
3. Standardized/raw CKA heatmaps, named-subblock tables, spectra, effective rank.
4. Directional reconstruction matrices and descriptor-probe tables.
5. Error-overlap tables and, when C runs, pair gains against both singles/H0.
6. Per-contract, subgroup, calendar-block, pooled, and tail diagnostics with counts.
7. Pointwise paired intervals and block-length sensitivity, with validity counts.
8. Resource tables and performance-cost plots; optional D/E reports kept separate.
9. A seven-question discussion separating measured evidence from hypotheses.

## 14. Implementation, execution, and replay order

Reusable code belongs under `src/`; thin orchestration belongs in the next
available script generation. At implementation time inspect the repository
before choosing that directory name. Existing CKA, identity, metric, and replay
utilities can be reused after auditing their semantics; no new script is claimed
to exist by this plan.

Ordered gates:

1. Record the current A+B+C scope and any separately commissioned optional D/E
   packages in the implementation manifest. Phase 6.8 does not gate this study.
2. Replay the six task/walk source feature stores and six H0 references; record
   exact hashes, row populations, original head recipes, and reference metrics.
3. Freeze configuration/coordinate maps, all task/walk identities, seed/budgets,
   metric conventions, structural samples, descriptor definitions, ridge objective,
   subgroup edges, bootstrap settings, resource procedure, and expected inventory.
4. Build exact branch views and prove equality to source coordinates. Fit
   training-only scalers and verify they equal the corresponding canonical
   per-coordinate state; test evaluation-perturbation invariance.
5. Run meaningful CPU integration checks for every distinct width/task family:
   forward/loss/backward, scaler replay, identity mismatch rejection, branch-map
   corruption, constant features, invalid metrics, occupied-path protection,
   and interrupted-run recovery. Verify auxiliary objectives and metric formulas
   on synthetic examples with known rank, duplication, and reconstruction skill.
6. Generate manifest-only launch artifacts. Resource-admit execution, use
   persistent logs, and require explicit `--execute` for real neural trajectories.
7. Execute the complete required A/C matrices and any separately commissioned
   D/E matrices without result-driven truncation; reuse completed runs only
   after provenance and prediction replay.
8. Replay checkpoints, scalers, branch coordinates, predictions, metrics, source
   hashes, row identity, and auxiliary coefficients/samples/draws. Preserve
   backend/dtype-specific tolerances in a written replay contract.
9. Generate all declared tables/figures, including negative and unavailable cells.
10. Write the outcome discussion and update progress documents from executed
    evidence. Planned files, successful smoke tests, and existing utilities are
    not evidence that the matrix is complete.

The 2026-10-12 owner clarification replaces the original Phase 7A authority.
Keep the supersession notice, schedule, research/design documents, data-allocation
rules, AGENTS.md, and affected skill pointers consistent with this current plan.
Implementation and execution statuses remain not started until repository
evidence establishes progress. Optional D/E remain distinct from required A+B+C.

## 15. Completion criteria

### 15.1 Package A

- All 60 new trajectories and six H0 references pass independent replay.
- All 180 new snapshots and 18 reference snapshots are accounted for.
- Single/removal results cover all three tasks and both walks with exact rows.
- Width, parameter count, references, negative results, and limitations accompany metrics.

### 15.2 Package B

- All declared CKA/spectrum/subblock populations and raw sensitivities exist.
- All 120 cross-branch ridge fits and 12 descriptor fits replay, with unavailable
  dimensions/descriptors explicitly accounted for.
- All 60 primary error-overlap records and their classification strata exist.
- Subgroup/tail/contract/pooling rules are replayable; no evaluation-defined bins.
- Uncertainty and resource reports cover all required A+C configurations and
  any commissioned optional configurations, with
  availability reasons and the stated limits on inference.

### 15.3 Package C and optional D/E

- C: all 60 additional trajectories/180 snapshots replay; every pair is compared
  against both singles and H0 across all tasks/walks.
- D, if commissioned: all 120 controls/360 snapshots replay with exact duplication.
- E, if commissioned: all 252 additional trajectories/756 snapshots replay and
  summaries distinguish head-seed from encoder-seed variability.
- Uncommissioned D/E do not prevent full-plan completion. Packages A, B, and C
  must all satisfy their criteria; unexecuted C means an incomplete A+B stage,
  with pair complementarity unresolved.

The final contribution is a task- and condition-specific account of standalone
usefulness, incremental contribution, structural overlap, and practical
complementarity. Causal attribution, universal rankings, encoder-seed robustness,
SOTA, and profitable trading are not conclusions of this experiment.

## 16. References and repository authorities

- [Historical original Phase 7A matrix, now superseded](2026-09-26-phase-7a-representation-ablation-plan.md).
- [Phase 5 canonical data/evaluation authority](2026-09-21-phase-5-experiment-plan.md).
- [Phase 5 downstream probing contract](2026-09-21-phase-5-feature-and-framework-downstream-amendment.md).
- [Eight-hour future-price contract](2026-09-21-phase-5-absolute-price-h8-amendment.md).
- [Strict variance horizon](2026-09-24-phase-6-volatility-horizon-freeze-amendment.md)
  and [loss/head/unit freeze](2026-09-25-phase-6-volatility-model-matrix-freeze.md).
- [Training/test allocation](../training_test_data_selection.md),
  [raw-time causality contract](../data_processing_split_contract.md), and
  [current schedule](../schedule.md).
- Kornblith et al. (2019),
  [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html):
  CKA as a representation-geometry diagnostic.
- Roy and Vetterli (2007),
  [The Effective Rank: a Measure of Effective Dimensionality](https://doi.org/10.5281/zenodo.40328):
  entropy-based effective rank; this plan explicitly applies that principle to covariance eigenvalues.
- Relevant existing implementations: `src/features/phase5_features.py`,
  `src/features/statistical.py`, `src/features/transform.py`,
  `src/evaluation/phase6_encoder_variants.py`,
  `src/evaluation/phase6_encoder_variant_reporting.py`,
  `src/training/phase5_downstream.py`, `src/training/phase5_absolute_price.py`,
  `src/training/phase6_volatility.py`,
  `src/training/phase6_encoder_variant_downstream.py`, and
  `src/tasks/phase2_classification/metrics.py`.
