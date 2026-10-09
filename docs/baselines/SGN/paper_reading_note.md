# SGN paper reading note

**Title:** SGN: Shifted Window-Based Hierarchical Variable Grouping for
Multivariate Time Series Classification
**Authors:** Zenan Ying; Jinke Wang; Zhi Zheng; Tong Xu; Wei Chen; Qi Liu;
Huijun Hou
**Venue:** Advances in Neural Information Processing Systems 38 (NeurIPS
2025), Main Conference
**Primary source read:** owner-supplied 32-page proceedings PDF
**Local SHA-256:**
`b8148c25a156d51e4a34de82ef52d903e71083254633a7fc10473f7aa045924d`

## 1. One-paragraph summary

SwinGroupNet (SGN) is a supervised multivariate time-series classifier that
tries to occupy the middle ground between channel-independent models and
models that mix every channel globally. It initializes a learnable variable-to-
group assignment from Brownian distance correlation and K-means, embeds the
resulting group signals, extracts temporal patterns with an averaged bank of
odd-width depthwise convolutions, mixes information first within and then
across groups using pointwise convolutions, alternates ordinary and half-period
shifted windows, and hierarchically merges adjacent windows before global
pooling and classification. The paper reports strong results on four main
datasets and UEA classification collections. Its main relevance here is as a
classification-specialized complete system; its sensor/activity recognition
results do not directly predict performance on future Polymarket movement.

## 2. Problem

The paper argues that two common multivariate strategies are incomplete:

- treating every variable independently preserves variable identity but loses
  dependencies; and
- mixing all variables together captures dependencies but may blur
  heterogeneous variable semantics and over-smooth the representation.

The proposed answer is structured interaction: assign variables to a small
number of groups, model detailed interaction inside each group and coarser
interaction across groups, and combine that variable hierarchy with a
period-aware temporal hierarchy.

## 3. Main idea

For input `X in R^(C x L)`, SGN has three named components.

1. **Variable Group Embedding (VGE):** initialize a global `C x G`
   assignment from Brownian distance correlation (BDC) and K-means; learn it
   with soft Gumbel-Softmax assignments during training and hard assignments
   during evaluation.
2. **Multi-Scale Group Window Mixing (MGWM):** divide the sequence into
   period-length windows; apply an averaged bank of grouped temporal
   convolutions; then use pointwise convolutions to mix within groups and
   across groups.
3. **Periodic Window Shifting and Merging (PWSM):** alternate unshifted and
   half-period shifted windows so neighboring windows communicate, and merge
   neighboring period windows to grow the temporal receptive field.

The model pools the final feature map and applies a native classification
projection. It is trained end to end with task cross-entropy plus a variable-
similarity grouping penalty.

## 4. Contributions

- **Methodological:** replace all-independent/all-mixed channel handling with
  learnable intra-group and inter-group interaction.
- **Methodological:** use shifted period windows and hierarchical merging to
  extend a convolutional model's temporal reach efficiently.
- **Empirical:** evaluate on four large domain datasets and UEA multivariate
  classification datasets, with component ablations and limited resource
  analysis.

The simplest accurate novelty statement is: SGN combines a learnable global
channel grouping prior with group-structured pointwise mixing inside a
shifted, hierarchically merged temporal-convolution network.

## 5. Method

### 5.1 Variable grouping

Paper Section 3.1 and Appendix A compute a dependence matrix between channel
signals using Brownian distance covariance/correlation. Each row of the
`C x C` matrix is treated as a variable's dependence profile, and K-means
assigns the `C` variables to `G` initial clusters.

For variable `j` and group `i`, Equation 2 defines a Gumbel-Softmax assignment
with temperature `tau`. Training uses soft assignments plus Gumbel noise;
evaluation uses one-hot assignments. `tau` decays exponentially.

Equation 3 adds:

```text
L_sim = sum_(i,j) S_ij * ||M_i - M_j||^2
L     = L_task + beta * L_sim
```

so variables with high current cosine similarity are penalized for receiving
dissimilar assignments. The paper's prose does not fully define the
observation axis used by BDC across a dataset of windows, the exact BDC sample
budget, K-means initialization, empty-group handling, or the normalization of
group fusion. Those details matter for the project adaptation.

### 5.2 Period selection and partitioning

Section 3.2 applies an FFT along time, averages the amplitude spectrum, takes
top-frequency candidates, and writes the selected window as the smallest
period among the top candidates. The sequence is zero-padded to a multiple of
the chosen period `P` and reshaped to `C x N x P`, with `N=ceil(L/P)`.

This account is not internally complete. Appendix D reports dataset-specific
period candidates, while Table 7 and the released launch scripts use manually
configured periods that are not consistently the literal minimum described by
Equation 5. The released forward path performs no FFT selection.

### 5.3 Multi-scale group-window mixing

Equation 6 averages `K` temporal convolution branches with odd kernels
`1,3,...,2K-1`. The paper calls these grouped/depthwise convolutions. The
subsequent pointwise layers implement two interaction levels:

- **intra-group:** mix the latent coordinates within each group while keeping
  groups separate; and
- **inter-group:** transpose the group/latent axes conceptually and mix groups
  for each latent coordinate.

Residual connections preserve the input feature map. In the source, temporal
convolution is followed by BatchNorm, GELU, dropout-bearing pointwise blocks,
and another residual addition.

### 5.4 Shifted windows

Section 3.3 cyclically shifts the time axis left by `P/2`, partitions and
processes the shifted windows, and shifts back. Alternating unshifted and
shifted blocks enables cross-window interaction without global attention.

The paper describes a pure cyclic shift. The source additionally restores the
first half-window from the unprocessed input after shifting back, preventing a
fully cyclic oldest/newest boundary interaction. This is a material behavioral
detail requiring an explicit project decision.

### 5.5 Hierarchical merging

Adjacent period windows are concatenated in the latent dimension and reduced
back to the original width. For more than four windows, the paper/source use
non-overlapping pairwise merging, carrying an odd final window forward. For
four or fewer windows, overlapping adjacent pairs make the number of windows
decay linearly (`N -> N-1`). The stage count must therefore be chosen so it
never attempts to merge a single remaining window.

### 5.6 Pooling, head, and objective

After the last stage, SGN flattens the remaining window positions, applies
LayerNorm over `G*D`, averages over time, and applies one linear classification
head. The paper uses ordinary cross-entropy plus `beta*L_sim`. Phase 6.9 must
replace only the task-loss component with the project's training-prior
logit-adjusted cross-entropy while keeping the grouping term separately
reported.

## 6. Experiments

### Main datasets and protocol

The four main datasets are TDBRAIN (`C=33,L=256`, binary), PTB-XL
(`C=12,L=250`, five classes), FLAAP (`C=6,L=100`, ten classes), and UCI-HAR
(`C=9,L=128`, six classes). The paper reports accuracy, precision, recall,
F1, AUROC, and AUPRC over five seeds `41..45`, using fixed train/validation/
test splits and validation-F1 early stopping. Main SGN settings use two to five
stages, widths 32 or 64, seven convolution kernels, ratio 2, `beta=0.1`,
Adam, and SGN learning rate `1e-3`.

### Headline results

- Table 2 reports SGN best on every displayed main-dataset metric. The
  accuracy lead over the second-best method is about 4.8 points on TDBRAIN,
  0.5 on PTB-XL, 4.5 on FLAAP, and 2.2 on UCI-HAR.
- Table 3 reports the best average rank (`2.56`) and 14 top-1 finishes over 25
  UEA datasets, but SGN is not best on every dataset.
- Figure 4/Table 8 report the best average accuracy among the shown UEA-10
  methods.
- Tables 4 and 5 report worse results when variable grouping, intra-group
  interaction, inter-group interaction, shifting, or merging is removed.
- Appendix D reports sensitivity to group count, width, kernel count, and
  period. Width 32/64 is generally stronger than 128/256; seven kernels is the
  reported default.
- Figure 9 presents an accuracy/time/memory trade-off on TDBRAIN and FLAAP,
  but the paper does not provide a complete numeric resource table.

### What the ablations establish

The ablations support the claim that the complete grouped/shifted/merged model
is useful on the tested datasets. They do not fully isolate the causal effect
of each idea because removing a group interaction changes both parameter
connectivity and optimization, and dataset-specific hyperparameters were
chosen with validation sets.

## 7. Evidence versus claims

| Claim | Assessment | Reason |
|---|---|---|
| Structured grouping improves the tested main datasets | Strongly supports | Direct component ablations worsen accuracy/F1 on all four displayed main datasets. |
| Shifted and merged period windows improve the tested model | Strongly supports | Table 5 removes each mechanism and reports consistent degradation. |
| SGN is broadly competitive for multivariate classification | Strongly supports | Four main datasets plus broad UEA tables cover varied domains and baselines. |
| SGN universally outperforms prior MTSC methods | Does not clearly establish | It loses or ties on several UEA datasets, and settings/protocols differ across tables. |
| BDC grouping is the cause of the gains | Partially supports | Grouping ablations help, but there is no matched alternative-initializer study and no statistical test for the grouping itself. |
| SGN will be strong for future financial movement | Does not establish | The paper evaluates recognition/classification, not forward prediction, OHLCV, or prediction-market shift. |

## 8. Strengths

- The variable interaction hypothesis is clear and operational.
- Convolutional group mixing is cheaper and easier to inspect than dense
  global attention.
- Shift/merge mechanics give a transparent path from local to longer-range
  temporal interaction.
- The paper includes useful component and hyperparameter ablations.
- Five-seed main-dataset reporting is stronger than a single-run claim.

## 9. Limitations

- The paper's strongest tasks are mostly activity/signal recognition, whereas
  this project predicts a future movement label from a historical context.
- The paper uses validation-based early stopping and dataset-specific
  settings, incompatible with this project's fixed train/test-only lifecycle.
- BDC observation semantics and bounded computation are underspecified.
- Period selection differs between Equation 5, Appendix D, Table 7, and the
  released source.
- “Independent group embeddings” is ambiguous; the source uses one shared
  embedding after summing variables into group series.
- The released repository has no software licence and has several execution
  defects; see `official_code_audit.md`.
- The paper does not report significance tests, confidence intervals for
  pairwise improvements, or a complete numeric compute table.
- Only classification is evaluated; forecasting and imputation are named as
  future work in Appendix G.

## 10. Relation to prior work

SGN borrows multi-period/multi-kernel temporal ideas associated with TimesNet-
style CNNs and shifted hierarchical windows associated with Swin-style
architectures. Its distinctive step is to insert learnable, globally shared
variable grouping between channel-independent and fully mixed processing, and
to realize group-local and cross-group interaction with two pointwise
convolution layouts.

For this project, SGN differs from H0 most importantly in supervision and
comparison level: SGN learns its full temporal backbone and native head from
classification labels, while H0 reuses a target-free representation with a
lightweight task head. Any SGN win would demonstrate task-specific system
competitiveness, not superior reusable representation.

## 11. Key takeaways

1. SGN's core is structured channel interaction, not merely a CNN with FFT.
2. The source behavior is more specific—and sometimes different—than the
   paper equations.
3. Five OHLCV channels naturally support two groups, not the four-to-six
   groups used by many paper datasets.
4. A fixed period must be frozen from training-only evidence; per-input period
   selection would make architecture and replay unnecessarily unstable.
5. The adaptation must retain the native supervised head and disclose its
   optimization freedom relative to H0-D0 and Raw LSTM.
6. The paper supports trying SGN-C; it does not predict that SGN-C will win.

## 12. Decisions after reading

The owner approved `G=2`, fixed `P=16` with depths `[2,2,2,1]`, source-style summed group fusion with one shared
embedding, source first-half shift restoration, the proposed width/kernel/
loss/temperature package, source-specific Adam `1e-3` with physical batch 256
subject to resource admission, and independent-only implementation. All seven
paper/source adaptation decisions are resolved.

## 13. Relevance to this project

SGN-C fills Phase 6.9's classification slot because it is explicitly designed
for multivariate classification and retains a native supervised head. The
project adaptation must consume the exact saved `[N,64,5]` contexts and
`DOWN/STABLE/UP` labels, fit every BDC/period/grouping state on each walk's
training population only, and predict every unchanged evaluation row. It
should be compared separately with immutable H0-D0 and Raw LSTM, with macro-F1
primary and collapse/per-class/resource diagnostics preserved. No frozen-SGN
feature store or additional task is implied.
