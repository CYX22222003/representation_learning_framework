# SaURL-TS upstream clarification request

Suggested message to the authors or repository maintainer:

> I am adapting SaURL-TS as a frozen unsupervised representation baseline for an academic comparison on 64-step, five-channel OHLCV time series. I want to preserve the published method rather than infer missing details. Could you please clarify or provide the following?
>
> 1. What software licence applies to <https://github.com/YusenL/SAURL-TS>?
> 2. Which commit or release produced the results in the 2026 Pattern Recognition paper?
> 3. Is there a newer implementation containing the paper's Representation-wise Attention Mechanism (RwAM)?
> 4. What is the exact frozen inference representation: pooling rule, output width, and included modules? Is it the 128-dimensional weighted sum in Equation 17?
> 5. What tensors enter the frequency encoder at training and inference: complex FFT, magnitude, or an inverse-FFT reconstruction?
> 6. What two views enter the cross encoder?
> 7. What are the RwAM pooling axes/regions, Conv1d kernel sizes, reduction ratio, and tensor layout?
> 8. How are the SaDA loss, diversity loss, and three SaSSL losses combined or alternated? Which parameter groups update in each step?
> 9. Are the two ADT views produced by shared or separate transformation heads, and is the hard mask stochastic during training?
> 10. What EMA decay, dilated-CNN depth/width, projector/predictor dimensions, and checkpoint epoch were used for the published 128-dimensional model?
> 11. The public code computes adaptive branch-loss weights and then overwrites the weighted loss with an unweighted sum. Which behavior is intended?
> 12. May the source be adapted and redistributed with attribution for a non-commercial academic project?
>
> I can cite the paper and repository and will clearly label any necessary project adaptation. A tagged release or minimal extraction example for an input shaped `[batch, time, channels]` would be especially helpful.

## Resolution record

When a response is received, record:

- date and communication channel;
- responder name/role;
- licence text or permission scope;
- source archive/commit hash;
- answers to questions 3–10;
- any files supplied and their SHA-256 hashes; and
- the resulting Phase 6.7 decision: admit SaURL, use SISSEL fallback, or propose a separately named reconstruction amendment.

## Internal clarification exchange — 2026-10-04

This section records the project owner's reply to the questions above and the
resulting technical clarification. It is an internal design discussion, not a
response from the SaURL-TS authors. The project owner approved the resolutions
to Questions 4--11 on 2026-10-04; they now form the frozen implementation
contract for the independently authored Phase 6.7 adaptation.

### Project-owner reply

> 1. I don't know, and I failed to find an alternative repository other than
>    the provided one.
> 2. I also don't know. That link was extracted directly from the 2026
>    *Pattern Recognition* paper.
> 3. No, I cannot find a newer implementation.
> 4. and 5. I don't understand these questions. Are you referring to pooling
>    in RwAM? How is it related to the loss function? For 5, what do you mean
>    by tensors entering the frequency encoder? I thought it was the sequence
>    after Fourier transformation. If that is not correct, clarify.
> 6. If I am not wrong, from the paper, it refers to the time and frequency
>    views.
> 7. I don't know. This is not specified in the paper. Could you check the
>    paper and repository and decide a suitable number?
> 8. I thought this part was specified in the paper. Why was this question
>    raised? What are some alternative actions?
> 9. They should be separated, right? I am pretty sure the paper specified
>    this. Please read the paper and verify the code again.
> 10. Check the codebase and paper and propose some settings.
> 11. That means the code is contradictory with the paper. Any recommendation?
> 12. The source should be adapted and referred to. I will not directly
>    copy-and-paste the paper.

### Agreed interpretation of source authority

The Phase 6.7 candidate should be an independently written, paper-guided
reimplementation. The public repository is useful reconstruction evidence,
but it is not treated as the specification because it is unlicensed, does not
contain the paper's complete RwAM, has broken extraction paths, and contains
loss logic that contradicts its own adaptive-weight calculation.

The reporting label should therefore be:

> **SaURL-TS-Frozen (paper-guided reimplementation)**

It must not be described as an official reproduction. No upstream source code
should be copied into this project; the paper and repository should instead be
attributed as design references. Existing project-native BYOL, EMA,
checkpointing, feature-store, replay, and downstream-comparison machinery can
be reused.

### 4. RwAM pooling versus final representation pooling

There are two separate pooling operations:

1. **Pooling inside RwAM** is used to calculate attention weights for the
   temporal, frequency, and cross-domain representations. It is part of the
   trainable architecture.
2. **Final temporal pooling** converts an encoder output such as
   `[batch, 64, 128]` into one frozen vector `[batch, 128]` for each Phase 6.7
   decision row.

Pooling is not itself a loss. It is related to training because the paper's
SaSSL losses operate on attention-weighted representations. Consequently,
gradients from the BYOL losses pass through RwAM and train its attention
weights.

**Selected project interpretation:** each dilated CNN first produces sequence
states, global maximum pooling produces a 128-dimensional branch vector, and
RwAM weights and sums the three branch vectors. The BYOL projectors and
predictors are discarded during frozen feature extraction; the fused
128-dimensional pre-projector representation is retained. Global maximum
pooling is source-grounded by the public repository's intended `encode`
behaviour, although that extraction implementation is incomplete.

### 5. Tensor entering the frequency encoder

The project owner's understanding is substantially correct: the frequency
view begins with the Fourier-transformed sequence. The unresolved detail is
that an FFT produces complex values, whereas an ordinary `Conv1d` expects
real-valued inputs.

**Repository evidence:** the public implementation applies `rfft`, separates
amplitude and phase, applies frequency SaDA to the amplitude, combines the
transformed amplitude with the original phase, and applies `irfft`. The
frequency encoder therefore receives a real-valued, frequency-derived
sequence rather than a complex FFT tensor:

```text
normalised X
  -> rFFT -> amplitude A and phase phi
  -> frequency SaDA transforms A into A'
  -> combine A' with the original phi
  -> inverse rFFT
  -> real frequency-derived view V_f shaped [batch, 64, 5]
  -> frequency encoder
```

Other technically possible choices are magnitude-only input, separate
real/imaginary channels, or a complex-valued neural network. **Project-owner
decision (2026-10-04):** the reimplementation will use the repository-evidenced
inverse-FFT reconstruction. Frequency SaDA receives and transforms magnitude
only; each view is recombined with the original phase, inverse-transformed to
`[batch,64,5]`, and then supplied to the frequency encoder. At frozen
inference, the original normalized real sequence is supplied directly to that
encoder because an unchanged FFT/inverse-FFT round trip would reconstruct the
same input within numerical tolerance. The frequency encoder remains distinct
because its independent weights were trained under spectral augmentations.

### 6. Cross-domain encoder views

The project owner's interpretation is supported: cross-domain refers to the
temporal and frequency-derived views.

The paper's notation `E_C(x_t, x_f)` does not establish whether the inputs are
concatenated, processed by two towers, or treated as two positive views. The
public repository's most coherent behaviour uses one temporal SaDA view and
one reconstructed spectral SaDA view as the positive pair for a single cross
encoder.

**Proposed project interpretation:** `E_T` aligns two temporal SaDA views,
`E_F` aligns two spectral SaDA views, and `E_C` aligns a temporal SaDA view
with a frequency-derived SaDA view of the same sample. The cross property is
therefore supplied by its training objective, not by concatenating the inputs
or changing the encoder to ten input channels.

### 7. Selected RwAM dimensions and layout

**Paper evidence:** RwAM uses fine-grained average and maximum pooling, a
shared MLP formed from two 1-D convolutions, dimensionality reduction and
restoration, ReLU after the first convolution, addition of the average and
maximum paths, and sigmoid attention. The paper does not give all tensor axes,
region sizes, convolution kernel sizes, or the reduction ratio.

**Selected project setting:**

- stack the three branch vectors as `[batch, 3, 128]`;
- partition each 128-dimensional branch into eight regions of 16 coordinates;
- average-pool and max-pool inside each region;
- pass both pooled summaries through a shared
  `Conv1d(3, 1, kernel_size=1) -> ReLU -> Conv1d(1, 3, kernel_size=1)`
  bottleneck;
- add the two paths and apply sigmoid;
- expand the eight regional weights over their corresponding 16 coordinates;
- multiply each branch vector by its weights and sum the three branches.

Sigmoid, rather than softmax, follows the paper: the three branch weights are
not forced to sum to one. Normalised versions may be computed afterwards for
diagnostic plots. Eight regions is an explicit project adaptation and must be
frozen before evaluation metrics are read.

### 8. Loss composition and update schedule

The paper specifies the component objectives but does not fully specify their
optimizer interaction. It gives the temporal SaDA objective

```text
L_A = L_k + alpha L_t + beta L_r + gamma L_d,
```

a frequency variant without reconstruction, a diversity objective, and the
three-branch SaSSL objective

```text
L_L = L_t^BYOL + L_f^BYOL + L_c^BYOL.
```

It does not unambiguously provide a single total-loss equation, update order,
parameter groups, or gradient boundaries. Plausible implementations include:

1. one simultaneous update of SaDA and SaSSL from a combined loss;
2. alternating SaDA and SaSSL updates; or
3. a bilevel/meta-learning procedure.

**Selected project interpretation:** SaDA is not trained to completion and
then frozen permanently. Instead, use parameter-isolated alternating updates,
informed by the usable part of the repository. On every second minibatch,
freeze the representation networks and update temporal and frequency SaDA
first from their paper-defined input-domain objectives. On every minibatch,
freeze SaDA, regenerate and detach the current views, update the online
encoders, RwAM, projectors, and predictors with the equal-weight three-branch
BYOL sum, and then update target networks by EMA. Thus SaDA is trained before
SaSSL consumes its views on a joint-update batch, but both components co-evolve
throughout the 50-epoch trajectory. Use the paper's diversity coefficient
`lambda = 1.25`.

#### 8.1 Frozen view and loss definitions

For a normalized minibatch `X: [B,64,5]`, define the temporal-domain source as
`S_t = X`. Define the frequency-domain source by applying `rfft` along the
64-step axis and separating its magnitude and phase:

```text
C = rfft(X, dim=time)        # complex [B,33,5]
S_f = abs(C)                 # real magnitude [B,33,5]
phi = angle(C)               # real phase [B,33,5]
```

For domain `d in {t,f}`, its domain-specific SaDA first computes one shared
embedding `e_d: [B,L,16]` and factor logits `a_d: [B,L,1]`. For view
`k in {1,2}`, draw an independent `u_d^k ~ Uniform(1e-6,1-1e-6)` and use the
straight-through logistic mask below. Each view has its own informative and
irrelevant linear transform heads, both applied to `e_d`:

```text
embedding                   e_d = AugEncoder_d(S_d)
factor logits               a_d = Linear_d(e_d)
logistic noise              eps_d^k = log(u_d^k) - log(1-u_d^k)
soft mask                   m_d^k = sigmoid((a_d + eps_d^k) / 1.0)
hard mask                   b_d^k = 1[m_d^k >= 0.5]
ST hard mask                h_d^k = b_d^k - stopgrad(m_d^k) + m_d^k
informative source          I_d^k = h_d^k * S_d
irrelevant source           N_d^k = (1 - h_d^k) * S_d
informative scale           s_info_d^k = sigmoid(Linear_info_d^k(e_d))
irrelevant scale            s_irr_d^k = sigmoid(Linear_irr_d^k(e_d))
transformed informative     Ibar_d^k = s_info_d^k * I_d^k
transformed irrelevant      Nbar_d^k = s_irr_d^k * N_d^k
complete domain view        U_d^k = Ibar_d^k + Nbar_d^k
```

The factorizer is shared between the two views of one domain, but the mask
noise and transformation heads are separate. The temporal encoder view is
`V_t^k = U_t^k`. The frequency encoder view preserves the original phase:

```text
V_f^k = irfft(U_f^k * (cos(phi) + i sin(phi)), n=64, dim=time)
```

All MMD terms use a project-native five-kernel Gaussian MMD on flattened
per-sample domain tensors. Its base bandwidth is the detached mean non-diagonal
pairwise squared distance of the combined samples, clamped to `1e-8`; the five
bandwidths are `{base/4, base/2, base, 2*base, 4*base}`. The biased batch
estimate averages `Kxx + Kyy - Kxy - Kyx`. This is a frozen implementation
choice: the paper specifies an RKHS/MMD objective but not its kernel estimator.

For each view:

```text
L_k(d,k) = mean(h_d^k)
L_t(d,k) = MMD(S_d, Ibar_d^k)
L_d(d,k) = -MMD(N_d^k, Nbar_d^k)
L_r(t,k) = mean(abs(h_t^k[:,1:] - h_t^k[:,:-1]))

L_A(t,k) = L_k + alpha*L_t + beta*L_r + gamma*L_d
L_A(f,k) = L_k + alpha*L_t              + gamma*L_d
L_D(d)   = -MMD(U_d^1, U_d^2)

L_SaDA = 0.5 * sum_k(L_A(t,k) + L_A(f,k))
         + lambda * (L_D(t) + L_D(f))
```

The fixed coefficients are `alpha=0.1`, `beta=0.01`, `gamma=0.5`, and
`lambda=1.25`. MMD is evaluated directly in the applicable time or magnitude
domain; the public code's undocumented encoder-feature MMD is not reproduced.

#### 8.2 Frozen SaSSL view bundles and loss

The first and second multi-domain bundles are:

```text
Bundle A inputs: time=V_t^1, frequency=V_f^1, cross=V_t^1
Bundle B inputs: time=V_t^2, frequency=V_f^2, cross=V_f^1
```

Thus the time and frequency branches align their two within-domain SaDA views,
while the cross branch aligns a temporal and a reconstructed-frequency view of
the same sample. Selecting the first generated temporal/frequency views for
the cross pair follows the clearest executable repository evidence; symmetric
BYOL evaluates both prediction directions.

For each bundle and online/target path:

1. Run the appropriate branch encoder and globally max-pool its 64 timestamp
   states to `[B,128]`.
2. Stack time, frequency, and cross vectors as `[B,3,128]`.
3. Apply the selected RwAM to obtain three attention-weighted branch vectors.
4. Sum those vectors for the fused diagnostic representation; apply the BYOL
   projector/predictor separately to each weighted branch vector.

The same current RwAM is used for target-side weighting under `no_grad`; RwAM
does not have a separate EMA copy because the paper defines the EMA target as
encoder plus projector. For branch `d in {t,f,c}`:

```text
L_d^BYOL = 0.5 * [
    2 - 2*cos(p_d(q_d(ztilde_d^A)), stopgrad(qbar_d(ztildebar_d^B)))
  + 2 - 2*cos(p_d(q_d(ztilde_d^B)), stopgrad(qbar_d(ztildebar_d^A)))
]

L_SaSSL = L_t^BYOL + L_f^BYOL + L_c^BYOL
```

Projector outputs and predictions are normalized inside the loss. No adaptive
branch-loss weights or SWA parameters are used.

#### 8.3 Full alternating training algorithm

```text
Inputs:
  walk-specific encoder_train_sequences only
  epochs = 50, batch_size = 32, seed = 0
  SaDA cadence r = 2 global minibatches
  Adam SaDA optimizers: lr=1e-2, weight_decay=0
  Adam SaSSL optimizer: lr=1e-4, weight_decay=0
  EMA tau = 0.99

Initialize:
  fit the five-channel scaler on this walk's encoder population only
  initialize temporal/frequency SaDA independently
  initialize three online encoder/projector/predictor paths
  copy online encoder/projector parameters into their target paths
  freeze target parameters permanently
  initialize RwAM
  initialize global_step = 0

For epoch = 1,...,50:
  iterate one seeded shuffled, no-drop pass over the encoder population

  For each normalized minibatch X:

    # Phase A: SaDA update first on global steps 0,2,4,...
    if global_step mod 2 == 0:
      enable gradients only for temporal/frequency SaDA
      zero both SaDA optimizers with set_to_none=True
      generate both domain-view pairs with independent ST mask samples
      compute L_SaDA in the input/magnitude domains
      fail immediately on a non-finite view or loss
      backpropagate L_SaDA
      fail immediately on a non-finite SaDA gradient
      step both SaDA optimizers

    # Phase B: SaSSL update on every minibatch
    freeze SaDA parameters
    regenerate V_t^1, V_t^2, V_f^1, V_f^2 using the current SaDA
      with stochastic masks but under no_grad; detach all four views
    enable gradients for online encoders, projectors, predictors, and RwAM
    keep target encoders/projectors frozen and in eval mode
    zero the SaSSL optimizer with set_to_none=True
    build Bundles A and B
    compute online weighted branch representations for A and B
    under no_grad, compute target weighted branch representations for A and B
    compute L_t^BYOL, L_f^BYOL, L_c^BYOL and their unweighted sum L_SaSSL
    fail immediately on a non-finite representation or loss
    backpropagate L_SaSSL
    fail immediately on a non-finite SaSSL gradient
    step the SaSSL optimizer
    update each target encoder/projector:
      target = 0.99*target + 0.01*online

    log losses, mask rates, view distances, branch norms/stds,
      RwAM summaries, learning rates, elapsed time, and memory
    global_step += 1

  at epochs 5, 15, and 50:
    save all model/optimizer states, scaler, epoch/global_step,
      configuration and data hashes, and Python/NumPy/Torch CPU/CUDA RNG states
    replay a frozen CPU probe batch before marking the checkpoint valid

After epoch 50:
  discard SaDA, projectors, predictors, and target paths for feature extraction
  put online E_T, E_F, E_C and RwAM in eval mode and freeze them
  for normalized X, feed the same real X independently to all three encoders
  global-max-pool each branch, apply RwAM, and save their 128-wide weighted sum
```

The data loader must not drop encoder rows. If the final remainder would be a
singleton, merge it into the preceding batch so MMD always receives at least
two samples while every row appears exactly once per epoch. Primary training
uses float32 without AMP. There is no validation split, early stopping,
evaluation-driven restart, or best-checkpoint selection.

### 9. Separation and sharing of the two ADT views

The project owner's expectation is mostly correct, with one distinction:
the two views should use separate transformation heads, but the evidence does
not support duplicating the entire factorisation network.

**Paper evidence:** the transformation is repeated to produce the second
view. **Repository evidence:** temporal SaDA and frequency SaDA are completely
separate networks; within each domain, the embedding/factorisation machinery
is shared, while the two views have separate transformation heads and receive
independent stochastic mask samples.

**Selected project interpretation:** temporal and frequency SaDA are separate.
Within each domain, use one shared width-16, depth-1 embedding/factor network;
a factor head emits one logit per temporal or frequency position; and the two
views have separate informative and irrelevant transformation-head pairs.
Sample independent stochastic straight-through hard masks for the two views at
threshold `0.5` and temperature `1.0`. Use mask shapes `[B,64,1]` for time and
`[B,33,1]` for RFFT magnitude, broadcasting across five channels. This follows
the paper's `1 x T` mask rather than the public code's per-channel behaviour.
The width-16 augmentation encoder is one residual block with dilation 1; the
factor and four transformation heads are independent `Linear(16,1)` layers
apart from the factor head being shared by both views.

### 10. Selected initial hyperparameters

These are project adaptations inferred from the paper, public repository, the
64-hour Phase 6.7 input, and the existing project BYOL contract. They are not
all reported paper hyperparameters.

| Component | Proposed setting |
|---|---:|
| Input | 64 timestamps x 5 normalised OHLCV channels |
| Encoders | Three independent dilated residual CNNs |
| Hidden width | 64 |
| Representation width | 128 |
| Residual blocks | 6 |
| Dilations | `1, 2, 4, 8, 16, 32` |
| Kernel size | 3 |
| Dropout | 0.1 |
| Temporal pooling | Global maximum |
| RwAM regions | 8 regions x 16 coordinates |
| Projector output | 128 |
| Projector/predictor hidden width | 128 |
| EMA coefficient | 0.99 |
| Batch size | 32 |
| SaSSL learning rate | `1e-4` |
| SaDA learning rate | `1e-2` |
| SaDA update cadence | Every two batches |
| SaDA embedding | width 16, depth 1 |
| Mask | shared factor head, independent view samples, threshold 0.5, temperature 1.0 |
| Optimizer | Adam, zero weight decay |
| Training budget | 50 epochs, retaining epochs 5/15/50 |
| Paper loss weights | `alpha=0.1`, `beta=0.01`, `gamma=0.5`, `lambda=1.25` |
| Frozen downstream feature | Fused 128-dimensional pre-projector vector |

Six dilated residual blocks give a receptive field that covers the complete
64-hour input. This avoids retaining repository layers whose dilations greatly
exceed the sequence length. No model width, epoch, checkpoint, or optional
component may be selected by reading evaluation metrics.

### 11. Contradictory adaptive branch-loss weighting

The public code computes adaptive branch-loss weights and then overwrites the
weighted result with an ordinary unweighted sum. The adaptive weights are
therefore dead code in the observed implementation.

**Recommendation:** do not reproduce the unused adaptive-loss module. Use the
paper's explicit equal-weight objective

```text
L_L = L_t^BYOL + L_f^BYOL + L_c^BYOL.
```

RwAM should adapt representation contributions; it should not be conflated
with learnable loss coefficients. Repository-only SWA behaviour should also
be omitted unless separately justified, and the broken extraction path should
be replaced with a project-native `encode` interface.

### Reuse boundary and remaining review status

The implementation does not need to rebuild the complete training and
evaluation stack. Existing project-native components can supply BYOL
projection/prediction, symmetric cosine loss, EMA, deterministic checkpoints,
epoch 5/15/50 replay, frozen feature stores, row-identity validation, common
downstream heads, and comparison reporting.

The SaURL-specific implementation scope is limited to:

- temporal SaDA;
- spectral SaDA plus Fourier reconstruction;
- the dilated-CNN branch backbone;
- RwAM; and
- alternating SaDA/SaSSL orchestration.

### Approval record

On 2026-10-04 the project owner accepted the resolutions to Questions 4--6,
delegated the missing RwAM dimensions in Question 7 to the documented project
decision, accepted the alternating interpretation of Question 8 after
clarification, accepted the explicit settings for Questions 9--10, and
accepted the paper-aligned unweighted loss resolution for Question 11.
Sections 4--11 are therefore frozen for implementation before any downstream
evaluation metric is read.
