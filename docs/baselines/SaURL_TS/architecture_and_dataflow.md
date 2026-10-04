# SaURL-TS architecture and data flow

This note reconstructs the architecture intended by the 2026 paper. Statements marked **paper** are explicit in the article; **source** refers to the audited public commit; **project decision** is the independently authored Phase 6.7 adaptation approved on 2026-10-04. The distinctions matter because the public code does not implement the final paper exactly.

## 1. Intended end-to-end flow

```text
                                  Self-adaptive Data Augmentation (SaDA)
                             +----------------------------------------------+
                             |                                              |
X [B,T,F] -------------------+--> time ADT --> Vt(1), Vt(2) --> E_T -------+ |
  |                          |                                              | |
  +--> RFFT --> (A, phi) --> freq ADT(A) --> A(1), A(2)                    | |
  |                              |                                         | |
  |                         restore original phi                           | |
  |                              |                                         | |
  |                         IRFFT --> Vf(1), Vf(2) --> E_F ----------------|-+--> RwAM
  |                                                                         |      |
  +-------------------------- time/frequency cross views --> E_C -----------+      v
                                                                          weighted sum
                                                                          Z [B,128]

During pretraining: each E_i participates in an online/projector/predictor path
and an EMA target/projector path. During extraction: discard projectors and
predictors; freeze the encoders and RwAM; emit Z only.
```

For this project, `T=64` hourly bars and `F=5` channels ordered as OHLCV.

## 2. Component inventory

| Component | Input | Output | Learned during pretraining | Retained for frozen extraction |
|---|---:|---:|---:|---:|
| Walk-local input scaler | `[B,64,5]` | `[B,64,5]` | fitted statistics only | yes |
| Temporal ADT | time sequence | two temporal views | yes | no |
| Spectral ADT | RFFT magnitude `[B,33,5]`; original phase is retained | two inverse-RFFT reconstructed real views `[B,64,5]` | yes | no |
| Time encoder `E_T` | temporal views | paper branch vector `[B,128]`; source intermediate `[B,64,128]` | yes | yes |
| Frequency encoder `E_F` | real `[B,64,5]` views reconstructed after magnitude-only spectral augmentation | paper branch vector `[B,128]`; source intermediate `[B,64,128]` | yes | yes |
| Cross encoder `E_C` | temporal view 1 and reconstructed frequency view 1 of the same sample as its symmetric BYOL pair | branch vector `[B,128]` | yes | yes |
| RwAM | three same-width representations | three element-wise weight tensors | yes | yes |
| Fusion | weighted branch outputs | `[B,64,128]`, then one row vector | no parameters beyond RwAM | yes |
| BYOL projector/predictor | branch embeddings | projection/prediction vectors | yes | no |
| EMA targets | augmented views | target projections | EMA only | no |

The article defines the branch outputs as row representations and the final representation as a weighted **sum**, so its width is 128. The project decision applies global maximum pooling to each `[B,64,128]` encoder state sequence before RwAM. Concatenating three branches into 384 dimensions is a different model.

## 3. Self-adaptive Data Augmentation

### 3.1 Factorization

For each domain, an embedding network feeds a factor head. The factor head produces a thresholded mask at `kappa=0.5`:

```text
H = hard_threshold(sigmoid(FactorHead(Embed(X))))
X_info = H * X
X_irrel = (1-H) * X
```

The paper's notation specifies a `1 x T` mask broadcast across variables. The
audited source instead configures its heads to produce one value per input
channel. The project decision follows the paper: `[B,64,1]` for temporal views
and `[B,33,1]` for RFFT-magnitude views, broadcast over five channels.

### 3.2 Learned transforms and paired views

The informative and irrelevant parts have different sigmoid transformation heads:

```text
V_info  = sigmoid(G_info(X_info)) * X_info
V_irrel = sigmoid(G_irrel(X_irrel)) * X_irrel
V       = V_info + V_irrel
```

A second transformation path creates the paired view. In the source snapshot the factor network is shared and the two view paths have distinct transformation projectors.

### 3.3 Frequency path

The source snapshot implements the following concrete operation:

```text
RFFT(X) -> magnitude A and phase P
ADT acts on A
recombine transformed A with original P
IRFFT -> frequency-derived time-domain view
```

The paper explicitly says augmentation acts on magnitude, but does not
explicitly freeze the phase/reconstruction rule. The audited source provides a
coherent realization of that missing step. **Project decision (2026-10-04):**
the paper-guided reimplementation will transform magnitude only, retain the
original phase for each view, reconstruct the complex spectrum, and apply
`irfft(..., n=64)` before `E_F`. Thus `E_F` receives a real `[B,64,5]`
sequence, not a complex spectrum or a magnitude-only `[B,33,5]` tensor.

At frozen inference there is no learned SaDA transformation. The normalized
real sequence is passed directly to `E_F`; applying an unchanged
`irfft(rfft(X))` would be equivalent up to numerical tolerance. The frequency
branch remains distinct because its independent weights were pretrained using
the two spectrally augmented, phase-preserving reconstructed views.

### 3.4 Augmentation losses

The intended objectives are:

```text
L_A_time = L_k + alpha L_t + beta L_r + gamma L_d
L_A_freq = L_k + alpha L_t            + gamma L_d
L_D      = -MMD(V_1, V_2)
```

Paper sensitivity values are `alpha=0.1`, `beta=0.01`, `gamma=0.5`, and diversity weight `lambda=1.25`. These values were selected using paper validation analysis and therefore can be adopted only as fixed source-derived settings, never retuned on Phase 6.7 evaluation data.

## 4. Multi-domain bootstrap learning

Each domain has an online and target path:

```text
online: view -> E_theta -> projector q_theta -> predictor p_theta
target: view -> E_xi    -> projector q_xi
xi <- tau * xi + (1-tau) * theta
```

The symmetric BYOL loss for a branch should predict view 2 from view 1 and view 1 from view 2. Branch losses are summed. The public implementation uses EMA decay `0.99`; the paper only says it is close to 1.

The three encoders are described as multi-layer dilated CNNs. The source snapshot's concrete backbone is:

- input linear projection;
- ten residual convolutional blocks with kernel size 3 and exponentially increasing dilation `2^i`;
- a final output-width residual block;
- parallel temporal convolutions with kernels `[1,2,4,8,16,32,64,128]`; and
- mean aggregation across those kernel outputs.

This is useful reconstruction evidence, not yet licensed implementation code.

## 5. Representation-wise attention

The paper's intended RwAM is analogous to a shared channel-attention block:

1. partition or pool each branch representation at a fine granularity;
2. compute average- and max-pooled summaries;
3. pass summaries through a shared two-layer 1D-convolutional MLP;
4. reduce then restore dimensionality with ReLU between layers;
5. combine the two pooled paths and apply sigmoid; and
6. multiply each branch representation by its attention and sum the branches.

The paper does not specify enough dimensions to reconstruct the exact block,
and the public repository contains no RwAM module. The project decision stacks
the three pooled branch vectors as `[B,3,128]`, reshapes them to
`[B,3,8,16]`, average- and max-pools each 16-coordinate region, and sends both
`[B,3,8]` summaries through one shared
`Conv1d(3,1,1) -> ReLU -> Conv1d(1,3,1)` MLP. Their sum passes through sigmoid,
is expanded back to `[B,3,128]`, and weights the three branches before their
sum. No softmax is used.

## 5.1 Selected update schedule

SaDA and SaSSL co-evolve rather than running as two complete sequential
training stages. Every second minibatch first updates temporal and frequency
SaDA from the paper's input-/magnitude-domain objectives. Every minibatch then freezes SaDA,
detaches its generated views, updates the online encoders, RwAM, projectors,
and predictors, and finally applies EMA `0.99` to the target paths. The three
branch prediction losses are summed with equal weight.

The exact five-kernel MMD realization, two view bundles, gradient boundaries,
optimizer order, checkpoint state, and extraction algorithm are frozen in
[`upstream_clarification_request.md`](upstream_clarification_request.md#83-full-alternating-training-algorithm).

## 6. Training versus extraction boundary

### Pretraining-only components

- both ADT modules and all view-generation heads;
- online projectors and predictors;
- EMA target encoders/projectors; and
- MMD and bootstrap losses.

### Frozen representation components

- input normalization frozen from the walk's encoder-training population;
- online time, frequency, and cross-domain encoders at epoch 50;
- RwAM; and
- global maximum sequence-to-row pooling followed by the selected RwAM.

The downstream representation must not include task labels, task-specific projectors, target-period values, or projectors/predictors used only to optimize SSL.

## 7. Phase 6.7 tensor contract

The approved independent adapter contract is:

```text
input              float32 [B,64,5], finite
input order        open, high, low, close, volume
normalization      channelwise statistics fitted on encoder_train_sequences for that walk
encoder output     float32 [B,128], finite
extraction point   RwAM-weighted sum after global maximum row pooling
checkpoint         epoch 50, seed 0
gradient status    disabled for all downstream extraction and probes
```

The model is trained independently for Walk 1 and Walk 2. No weights, scalers, attention statistics, or decisions transfer from the later walk into the earlier walk.

## 8. Architecture invariants to test

- Both views have the same shape as their input and remain finite.
- Frequency SaDA changes magnitude only and leaves the source phase unchanged.
- Each frequency view safely reconstructs through `rfft/irfft` to finite
  `[B,64,5]` real values.
- Identity spectral transformation satisfies
  `irfft(rfft(X), n=64) ~= X` within a frozen numerical tolerance.
- A forward/backward step updates SaDA during its update and SaSSL during its update without unintended cross-updates.
- EMA parameters receive no gradient and change only by the frozen EMA rule.
- RwAM weights are finite, bounded in `[0,1]`, and have broadcast-compatible shapes.
- The final representation has native width 128, not 384 or 960.
- Projector and predictor tensors do not enter saved downstream features.
- CPU and CUDA extraction produce the same row order and numerically close embeddings.
