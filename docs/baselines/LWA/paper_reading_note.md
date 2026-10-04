# Learning Without Augmenting paper reading note

**Title:** Learning Without Augmenting: Unsupervised Time Series Representation
Learning via Frame Projections
**Authors:** Berken Utku Demirel; Christian Holz
**Venue:** 39th Conference on Neural Information Processing Systems (NeurIPS
2025)
**Version read:** arXiv:2510.22655v2, revised 2026-01-15
**Local SHA-256:**
`3594c7eb09fd354062bc0ef6db911548712099ac4556ee086fc501dd96abc8a6`

## 1. One-paragraph summary

The paper replaces stochastic, task-specific time-series augmentations with
three deterministic views of the same sample: the original time series, its
orthonormal Fourier projection, and an overcomplete localized time-frequency
projection computed with a Gabor/Morlet wavelet. Three domain-specific
encoders and projectors are trained jointly through pairwise instance
discrimination, while embedding-space mappers learn to predict the Fourier and
wavelet projected embeddings from the time-domain embedding. A second stage
freezes the domain encoders and trains two lightweight representation-space
mappers from the time representation to the Fourier and wavelet
representations. At inference, the auxiliary Fourier and wavelet transforms,
encoders, and all projectors are discarded. The time representation and two
mapped representations are concatenated into a 384-dimensional feature.

## 2. Problem

The paper addresses two limitations of augmentation-based self-supervision for
temporal signals:

- useful augmentations are task- and modality-dependent;
- augmentation-induced invariances can suppress features needed by an unknown
  downstream task.

The authors seek reusable representations without randomly perturbing samples
or selecting a different augmentation family for every dataset.

## 3. Main idea

For an input `x` with length `L` and `C` channels, construct:

1. the unchanged time-domain view `x`;
2. the orthonormally normalized positive-frequency DFT `F(x)`; and
3. a localized Gabor/Morlet continuous-wavelet view `W(x)` at 48 log-spaced
   scales from 1 to 128.

Separate encoders map the views to `h_t`, `h_F`, and `h_W`, each with width
128. Separate two-layer projectors produce `z_t`, `z_F`, and `z_W`. Pairwise
NT-Xent losses align matching samples across all three domain pairs while
repelling other batch samples. Nonlinear embedding mappers predict `z_F` and
`z_W` from `z_t`. After this joint stage, new nonlinear representation
mappers learn `h_t -> h_F` and `h_t -> h_W` with the encoders frozen.

The inference representation in Algorithm 2 is:

```text
h_LWA = concat(h_t, Phi_h_t_to_F(h_t), Phi_h_t_to_W(h_t))
        in R^(128 + 128 + 128) = R^384
```

## 4. Contributions

- **Methodological:** replace random augmentation views with fixed time,
  Fourier, and localized time-frequency projections.
- **Methodological:** learn nonlinear latent mappings so only the time encoder
  is needed at inference.
- **Empirical:** evaluate nine datasets spanning heart-rate regression,
  activity recognition, cardiovascular classification, step counting, and
  sleep staging.
- **Analytical:** provide arguments about unintended invariance under NT-Xent
  and the possible geometric differences between high-dimensional latent
  spaces.

## 5. Method

### 5.1 Frame projections

Equation 1 uses a normalized DFT and a Gabor wavelet transform. The Fourier
view supplies global frequency information. The wavelet view supplies
localized frequency information for transient or non-stationary behavior.
Appendix E.3.3 specifies `torch.fft.rfft(..., norm="ortho")` and a complex
Morlet CWT at `np.geomspace(1, 128, num=48)`.

Important qualification: the paper often calls the transformations unitary or
isometric. The actual source feeds magnitude and phase from a one-sided real
FFT to one encoder and only the magnitude of the complex CWT to the wavelet
encoder. Those concrete tensors do not make the paper's preservation claim
automatic, particularly after CWT phase is discarded.

### 5.2 Pairwise instance discrimination

Equations 2 and 3 define NT-Xent across the time/Fourier, time/wavelet, and
Fourier/wavelet domain pairs. The released implementation realizes each pair
symmetrically by concatenating both batches and treating the corresponding
cross-domain row as the positive. It uses normalized embeddings and
temperature `0.15`.

### 5.3 Embedding mapping during joint pretraining

Two nonlinear mappings predict transformed-domain projected embeddings from
the time-domain projected embedding. Equation 6 adds two L1 mapping terms to
the unweighted instance-discrimination objective:

```text
L_joint = L_ID
        + mean_over_samples ||Phi_z_t_to_F(z_t) - z_F||_1
        + mean_over_samples ||Phi_z_t_to_W(z_t) - z_W||_1
```

The paper says there is no additional weighting. The exact reduction remains
important because the official code does not implement this expression
literally; see the source audit.

### 5.4 Representation mapper stage

Algorithm 1 then freezes the three encoders, discards or omits the projectors
and embedding mappers for the second stage, and trains two new nonlinear
convolutional mappings on encoder representations with L1 error. The paper is
clear that this occurs after joint pretraining, but it does not separately
state the number of representation-mapper epochs.

### 5.5 Architecture

Section 3.3 and Appendix E.3 specify:

- time encoder: eight-block 1D ResNet with width-128 output;
- Fourier encoder: separate amplitude and phase convolution/residual branches,
  each reduced to 64 values and concatenated to width 128;
- wavelet encoder: a 2D multiscale convolutional encoder over channel, scale,
  and time dimensions, reduced to width 128;
- projector: two fully connected layers;
- each mapping: a stride-2 `Conv1d(1,64,3)`, ReLU, and stride-2
  `ConvTranspose1d(64,1,3)` that preserves the 128-coordinate length.

Batch normalization follows convolutional blocks, and the time ResNet uses
dropout `0.5` inside residual blocks.

### 5.6 Training and inference

The paper reports Adam with learning rate `0.003`, batch size `1024`, 256
epochs, and cosine learning-rate decay. Results use three runs. At downstream
time the encoders/mappers are frozen and a single linear layer is trained.
The paper reports 128 dimensions per LWA manifold and sets competing baseline
encoders to width 384 for downstream-head capacity matching.

Only the time encoder and the two representation mappers survive inference.
FFT, CWT, auxiliary encoders, projectors, and embedding mappers are
pretraining-only.

## 6. Experiments

### Datasets and tasks

The study covers nine biomedical/wearable datasets and five tasks: three
heart-rate datasets, two activity-recognition datasets, two cardiovascular
classification datasets, one step-counting dataset, and one sleep-staging
dataset (Section 3.1).

### Baselines

The paper compares with SimCLR, BYOL, VICReg, Barlow Twins, CLIP, TS-TCC,
TF-C, SimMTM, and TS2Vec, plus supervised FCN and ResNet references. Several
baselines are adapted to the authors' shared backbone or augmentation choices,
so the tables are not all exact reproductions of each baseline's original
system.

### Headline evidence

- Tables 1--3 show strong average ranking and substantial gains on several
  datasets, but LWA is not best on every individual metric.
- Tables 4--6 show that using both transformed views and representation
  mappers often helps, but removing one view or the mappers improves some
  dataset/metric cells.
- Tables 8--10 support embedding-space mapping on average, while also showing
  exceptions and showing that a larger nonlinear MLP mapper can outperform the
  selected lightweight convolutional mapper on some datasets.
- Appendix D.3 reports that retaining all three encoders at inference improves
  performance by roughly 4--5%, at the cost of much higher inference expense.
- Figure 8 reports the best average rank over 27 dataset/metric cells using a
  Friedman/Nemenyi analysis.
- Appendix F reports approximately 680 total GPU hours including ablations on
  24 GB RTX 4090 GPUs. The core method averaged 18.67 seconds per epoch in its
  timing comparison, excluding one-time cached FFT/CWT preprocessing.

## 7. Evidence versus claims

| Claim | Assessment | Reason |
|---|---|---|
| Fixed frame projections can replace random augmentations on the evaluated temporal tasks | Strongly supports | Broad nine-dataset evaluation and matched ablations directly test this setting. |
| Complementary transformed domains improve the representation | Partially supports | Combined views often help, but several ablation cells improve after removing a view. |
| Lightweight latent mappings recover useful transformed-domain geometry at inference | Partially supports | Mapper ablations are relevant, but gains are inconsistent and full auxiliary encoders perform better. |
| The learned spaces are distinct because of transformation-induced geometry | Partially supports | Distance/angle plots are suggestive; Proposition 2.2 relies on an independent-uniform-sphere approximation that is conjectured for learned coupled encoders. |
| The method avoids unintended invariance in general | Does not clearly establish | Proposition 2.1 conditions its lower bound on the existence of `K` near-positive negatives caused by an invariance; it does not establish that every unintended invariance creates those negatives. |
| LWA is universally superior to augmentation-based SSL | Does not clearly establish | Evidence is limited to selected temporal/biomedical tasks and source-adjusted baselines; individual metrics contain losses. |

## 8. Strengths

- The pretraining views are deterministic and do not require downstream-task
  augmentation knowledge.
- The inference path is efficient: only one encoder plus two very small
  mappings is retained.
- Global and localized frequency information are both represented during
  pretraining.
- Ablations isolate transformed views, embedding mappers, representation
  mappers, and the efficient-inference trade-off.
- The native 384-dimensional feature is directly usable by the Phase 6.7
  frozen probing design.

## 9. Limitations

- The paper does not evaluate financial OHLCV data or the project's exact
  forecasting targets.
- Most empirical evidence is classification-oriented; the authors explicitly
  identify forecasting as future work.
- The theoretical analysis does not explain the reported performance gains,
  which the paper acknowledges in Section 5.
- Preprocessing and split protocols vary by dataset, and some source paths use
  validation/model selection that cannot be imported into this project.
- A post-publication source commit corrected an HHAR subject-overlap data leak;
  the paper does not state whether its reported HHAR table was regenerated
  after that fix.
- CWT computation and storage are substantial. For this project's encoder
  populations, an uncompressed float32 CWT cache is approximately 2.236 GiB
  for Walk 1 and 3.346 GiB for Walk 2.
- Several architectural and loss details conflict with the released code.

## 10. Relation to prior work

LWA is closest to multi-view contrastive approaches such as TF-C, but removes
stochastic time/frequency perturbations and adds a third localized
time-frequency view. Its distinctive inference contribution is not merely
three-domain contrastive learning: it distills the auxiliary latent geometries
into mappings applied to the time representation so the extra encoders are not
needed downstream.

## 11. Key takeaways

1. LWA is a multi-domain contrastive method, not a no-view method.
2. Its 384-dimensional inference feature comes from one actual time encoder
   output and two learned approximations of auxiliary encoder outputs.
3. There are two training stages: joint encoder/projector/embedding-map
   pretraining, then frozen-encoder representation-map training.
4. FFT/CWT are pretraining-only under the efficient inference path.
5. The paper is sufficiently specific for an independent implementation, but
   source conflicts must be frozen before coding.

## 12. Questions after reading

- Does “256 epochs” apply independently to both training stages or only to the
  joint encoder stage?
- Should Equation 6 be reduced as mean per-sample L1 sum, coordinate mean, or
  the source's additional batch division?
- Was the paper evaluated with the stated 64-hidden-channel inference mapper
  or the source's one-hidden-channel instantiation?
- Were headline HHAR results regenerated after the 2026-01-16 leakage fix?
- Is cosine decay intended for the second mapper stage as well as joint
  pretraining?

These questions were resolved for the project adaptation on 2026-10-04. The
approved contract is recorded in `upstream_clarification_request.md`; the
paper questions remain noted here as reproducibility limitations rather than
open implementation choices.

## 13. Relevance to Phase 6.7

LWA directly tests a useful contrast with the canonical framework and SaURL:
its views are fixed mathematical projections, its objective is contrastive,
and its efficient inference feature is distilled from auxiliary domains. The
project will retain the native 384-dimensional representation and use the
same lightweight task heads, rows, losses, and metrics as immutable `H0` and
SaURL. This supports a narrow statement about representation quality under
the project's frozen-probing protocol; it is not a reproduction of the
paper's biomedical results or a state-of-the-art claim.
