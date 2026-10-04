# Learning Without Augmenting architecture and data flow

This document describes the owner-approved Phase 6.7 adaptation frozen on
2026-10-04.

## 1. End-to-end flow

LWA has a training-only multi-domain path and a smaller inference path. The
Fourier/wavelet transforms, their encoders, the three projectors, and the two
embedding-space mappings are used only during pretraining. Frozen inference
uses the time encoder and the two separately trained representation-space
mappings.

### 1.1 Pretraining-data workflow

```text
accepted walk bundle
    |
    +--> select target-free encoder_train_sequences only
    |      shape [N,64,5]
    |      no movement, price, or volatility target is loaded
    |
    +--> fit one five-channel mean/std scaler on this walk's encoder rows
    |
    +--> apply the frozen scaler to the same ordered rows
    |      x_t [N,64,5]
    |
    +--> construct deterministic, row-aligned training views
           |
           +--> time view:    x_t                         [N,64,5]
           +--> Fourier view: rfft(x_t over time)         [N,5,33] complex
           +--> wavelet view: |CWT_cmor1-1(x_t)|          [N,5,48,64]
    |
    +--> save/replay view-cache manifest
           row order + scaler hash + transform settings + content hashes
    |
    +--> shuffled, row-matched minibatches
           (x_t[i], x_F[i], x_W[i]) always describe the same sample i
```

The deterministic Fourier and wavelet tensors may be cached once because they
do not change across epochs. They are training views, not additional samples:
one source row still contributes one time/Fourier/wavelet triplet. The cache
contains training rows only and is never fitted or selected using a downstream
target.

### 1.2 Pretraining Stage A — joint multi-domain learning

```text
time view x_t              Fourier view x_F             wavelet view x_W
     |                           |                            |
     v                           v                            v
time encoder f_t           Fourier encoder f_F          wavelet encoder f_W
     |                           |                            |
 h_t [B,128]                h_F [B,128]                   h_W [B,128]
     |                           |                            |
projector g_t              projector g_F                 projector g_W
     |                           |                            |
 z_t [B,128]                z_F [B,128]                   z_W [B,128]
     |                           |                            |
     +---------------- normalize each z ---------------------+
                                 |
       normalized z_t, normalized z_F, normalized z_W
                    |                         |
                    |                         +--> three pairwise symmetric
                    |                              NT-Xent losses --> L_ID
                    |
                    +--> Phi_z^(t->F)(z_t) --> z_F_hat --L1 vs z_F--+
                    |
                    +--> Phi_z^(t->W)(z_t) --> z_W_hat --L1 vs z_W--+--> L_map_z

                         L_joint = L_ID + L_map_z
```

Stage A jointly updates all three encoders, all three projectors, and both
embedding-space mappings `Phi_z`. Its output checkpoint contains those
components, but `g_t/g_F/g_W` and `Phi_z` are not part of the final extractor.

### 1.3 Pretraining Stage B — representation-space mappings

Stage B reuses the same ordered, standardized training triplets and their
cached transformed views. It freezes the final Stage-A encoders and obtains
fixed mapping targets:

```text
x_t --> frozen f_t --> h_t [B,128]
                         |                 \
                         |                  +--> Phi_h^(t->W) --> h_W_hat
                         +--> Phi_h^(t->F) --> h_F_hat

x_F --> frozen f_F --> h_F [B,128] --detach--> target for h_F_hat
x_W --> frozen f_W --> h_W [B,128] --detach--> target for h_W_hat

L_map_h = L1(h_F_hat, h_F) + L1(h_W_hat, h_W)
```

Only the newly initialized `Phi_h^(t->F)` and `Phi_h^(t->W)` parameters are
updated in Stage B. These are distinct modules from the Stage-A `Phi_z`
mappings: `Phi_z` operates on projected embeddings `z`, whereas `Phi_h`
operates on encoder representations `h` and survives inference.

### 1.4 Frozen inference and feature extraction

For any established downstream training or evaluation row:

```text
raw task-aligned OHLCV row [B,64,5]
        |
        v
apply the already frozen walk-local LWA input scaler
        |
       x_t
        |
        v
frozen time encoder f_t
        |
     h_t [B,128]
       / \
      /   +--> frozen Phi_h^(t->W) --> h_W_hat [B,128]
     +-------> frozen Phi_h^(t->F) --> h_F_hat [B,128]
        |
        v
concat [h_t, h_F_hat, h_W_hat] --> LWA feature [B,384]
```

Inference does **not** compute an FFT or CWT and does not load `f_F`, `f_W`,
the three projectors, or either `Phi_z`. The resulting 384-dimensional feature
is frozen before the task-specific coordinate scaler and common lightweight
head are fitted.

## 2. Input and view contract

```text
input shape       [B,64,5]
input order       open, high, low, close, volume
dtype             float32, finite
normalization     per-channel mean/std fitted only on the walk encoder rows
time dimension    axis 1 in [B,T,C]
channel dimension axis 2 in [B,T,C]
```

The project OHLC coordinates enter unscaled as probabilities and volume has
already been z-scored from unique walk-training candles. The proposed LWA
standardizer is nevertheless candidate-specific: it brings all five channels
onto comparable scale before constructing the transformed views. Standard
deviations below `1e-8` become 1.0. The fitted statistics are checkpointed and
replayed unchanged on every row.

### Time view

`x_t` is the standardized real input, transposed internally as required by
`Conv1d`.

### Fourier view

Apply `torch.fft.rfft(x.transpose(1,2), dim=-1, norm="ortho")`. This produces
complex `[B,5,33]`. The Fourier encoder processes `abs` and `angle` in two
separate branches. No negative frequencies are materialized.

### Gabor/Morlet wavelet view

For every sample and channel, apply the complex Morlet wavelet `cmor1-1` at 48
log-spaced scales `geomspace(1,128,48)` and retain coefficient magnitude. The
model receives `[B,5,48,64]`.

The paper calls this a Gabor wavelet frame; the official source realizes it
through PyWavelets' complex Morlet family. Sampling frequency changes the
reported physical-frequency coordinates but does not change the coefficient
calculation for a fixed sampling period convention. For one-hour bars, the
project should record `fs = 1/hour` semantically while preserving the source
scales.

## 3. Time encoder

The source-style time encoder is:

```text
Conv1d same: 5 -> 32, kernel 5, stride 1
BatchNorm + ReLU
8 residual blocks, each with two same-padded kernel-5 convolutions
  downsample on blocks 2,4,6,8 via stride 2
  increase channels 32 -> 64 at block 5
  BatchNorm, ReLU, Dropout(0.5) inside non-first residual blocks
final BatchNorm + ReLU
global average over time
Linear: 64 -> 128
```

Output: `h_t [B,128]`.

## 4. Fourier encoder

Magnitude and phase each use:

```text
Conv1d: 5 -> 16, kernel 3, stride 1, padding 1
ReLU
Residual block: 16 -> 32, kernel 3, stride 2
Residual block: 32 -> 64, kernel 3, stride 2
Linear over remaining frequency length: 9 -> 1
```

Each branch returns `[B,64]`; concatenation gives `h_F [B,128]`.

The source's `FourierAutoencoder` decoders are not used by the objective or
inference. The independent adapter should not add unused reconstruction
parameters.

## 5. Wavelet encoder

Starting with `[B,5,48,64]`:

```text
Conv2d block: 5 -> 32, kernel 5, stride 1
Conv2d block: 32 -> 64, kernel 5, stride 2
concat AvgPool2d(input, 2): 64 + 5 -> 69
Conv2d block: 69 -> 96, kernel 5, stride 2
concat AvgPool2d(input, 4): 96 + 5 -> 101
Conv2d block: 101 -> 128, kernel 5, stride 2
result: [B,128,6,8]
Linear over time: 8 -> 1
Linear over scales: 6 -> 1
```

Every convolutional block is Conv2d, BatchNorm2d, ReLU. Output:
`h_W [B,128]`.

## 6. Projectors and mappings

Each domain projector is:

```text
Linear(128,128) -> ReLU -> Linear(128,128)
```

Each nonlinear mapping proposed by the paper is:

```text
reshape [B,128] -> [B,1,128]
Conv1d(1,64,kernel=3,stride=2,padding=1)
ReLU
ConvTranspose1d(64,1,kernel=3,stride=2,padding=1,output_padding=1)
squeeze -> [B,128]
```

Two such mappings operate on projected embeddings during joint pretraining.
Two newly initialized mappings operate on raw representations in the second
stage and survive inference. The approved hidden width is 64; the released
inference code's width-1 call is treated as an implementation inconsistency.

## 7. Joint objective

Normalize each projected embedding along its coordinate dimension. For every
unordered pair `(t,F)`, `(t,W)`, `(F,W)`, compute a symmetric NT-Xent loss with
temperature `0.15`; matching rows are positives and all non-matching rows from
both domains are negatives.

The proposed paper-form mapping loss is:

```text
L_map_z = mean_i sum_j |Phi_z_F(z_t)[i,j] - z_F[i,j]|
        + mean_i sum_j |Phi_z_W(z_t)[i,j] - z_W[i,j]|

L_joint = L_NTX(t,F) + L_NTX(t,W) + L_NTX(F,W) + L_map_z
```

This approved reduction intentionally does not use the source's extra
`/ batch_size` after an already averaged L1 loss.

## 8. Representation-mapper objective

After the fixed final joint checkpoint:

- freeze all three encoders;
- initialize new `Phi_h_F` and `Phi_h_W` mappings;
- minimize ordinary L1 mapping error from raw `h_t` to detached raw `h_F` and
  `h_W`; and
- do not update projectors or embedding mappings.

The implementation uses the sum of the two mean per-sample L1 norms. It trains
50 mapper epochs with constant-learning-rate Adam after the final 50-epoch
joint checkpoint.

## 9. Inference boundary

Retained:

- walk-local input standardizer;
- time ResNet;
- representation mapper `Phi_h_F`;
- representation mapper `Phi_h_W`.

Discarded:

- rFFT and CWT preprocessing caches;
- Fourier and wavelet encoders;
- all three projectors;
- both embedding mappings; and
- all contrastive-loss state.

The approved output order is `[h_t, mapped_h_F, mapped_h_W]`, matching
Algorithm 2. The source integrated encoder's alternative coordinate order is
not used.

## 10. Phase 6.7 invariants

- Input and every intermediate/output tensor is finite.
- Every encoder and mapper output has shape `[B,128]`.
- Physical batch size is fixed to 128 for both walks with source-style
  `drop_last=True`; this omits 30 shuffled Walk 1 rows and 105 shuffled Walk 2
  rows per epoch. No gradient accumulation or automatic batch fallback is
  allowed.
- Frozen inference has shape `[B,384]` and does not invoke FFT/CWT or auxiliary
  encoders.
- The inference output is invariant to batching in evaluation mode within the
  frozen replay tolerance.
- Fourier and wavelet views transform the time axis, never the channel axis.
- Each walk owns independent scaler, encoder, mapper, optimizer, RNG, and
  checkpoint state.
- No supervised target or evaluation row influences either LWA training stage.
