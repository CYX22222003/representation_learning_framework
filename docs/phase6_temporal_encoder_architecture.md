# Phase 6 Temporal Encoder Architecture

**Status:** Implementation reference for the frozen Phase 6 temporal-encoder
matrix. The canonical experiment authority remains
[`phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md`](phase_plan/2026-09-22-phase-6-temporal-encoder-variants-plan.md).
This document explains the implemented architecture; it does not change the
frozen matrix or authorize execution.

## 1. Scope and terminology

Phase 6 evaluates two temporal backbone architectures:

1. a one-layer unidirectional LSTM; and
2. a compact two-layer Transformer encoder.

Each backbone is used under two self-supervised learning (SSL) families:

| SSL family | LSTM branch | Transformer branch |
|---|---|---|
| Contrastive / NT-Xent | `contrastive_lstm` | `contrastive_transformer` |
| BYOL | `byol_lstm` | `byol_transformer` |

Thus, “two encoder variants” refers to two backbone types, while the complete
Phase 6 matrix contains four named temporal branches per walk. The backbone
turns a complete OHLCV sequence into a 128-dimensional state. The surrounding
contrastive or BYOL wrapper determines how that backbone is trained.

The reusable implementation is split between:

- [`src/models/temporal_backbones.py`](../src/models/temporal_backbones.py),
  which defines the LSTM, sinusoidal position encoding, and Transformer;
- [`src/models/encoder_variants.py`](../src/models/encoder_variants.py), which
  defines the frozen settings and the contrastive/BYOL wrappers; and
- [`src/training/phase6_encoder_variants.py`](../src/training/phase6_encoder_variants.py),
  which applies the walk-specific Phase 6 training and replay contract.

## 2. Shared input and output contract

Both backbones receive the same tensor:

```text
x: [B, 64, 5]

B  = batch size
64 = consecutive one-hour observations ending at the decision point
5  = open, high, low, close, volume
```

OHLC values remain unscaled probabilities. Volume is z-scored using statistics
fitted only on the applicable walk's training interval. The sequence builder
may fill a complete isolated one-hour gap using flat OHLC and zero raw volume;
longer gaps break the sequence. Imputation and time-since-observation arrays
are retained as metadata, but they are not additional inputs to these
five-channel encoders.

The input contains historical context only. It never contains a classification,
future-price, or future-realised-variance target, nor any token from the future
target interval. Encoder eligibility is therefore target-independent.

Both backbones return:

```text
h: [B, 128]
```

`h` is the unnormalised backbone state saved as the frozen downstream branch.
The SSL projector output is used to train the encoder but is not saved as the
downstream representation.

## 3. LSTM backbone

### 3.1 Data flow

```text
[B, 64, 5]
     |
     v
one-layer LSTM
input_size = 5
hidden_size = 128
num_layers = 1
dropout = 0.0
     |
     v
hidden states [B, 64, 128]
     |
     v
final hidden state h_64
     |
     v
[B, 128]
```

The implementation calls `nn.LSTM` with `batch_first=True`. Internally, the
LSTM traverses the 64 observations in chronological order while reusing the
same recurrent parameters at every step. It is unidirectional: information
flows from the oldest observation toward the decision-time observation.

For timestep `t`, the LSTM conceptually computes:

$$
i_t=\sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}),
$$

$$
f_t=\sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}),
$$

$$
g_t=\tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg}),
$$

$$
o_t=\sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho}),
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,
\qquad
h_t=o_t\odot\tanh(c_t).
$$

The input, forget, candidate, and output gates control how new OHLCV
information updates the memory cell. Only the last-layer final hidden state,
`hidden[-1]`, is returned. With one layer this is the final state after the
64th observation and has shape `[B,128]`. The final cell state is not exposed
as a downstream feature.

### 3.2 Why one recurrent layer

One recurrent layer still processes all 64 timesteps; “one layer” does not mean
one observation. The Phase 6 question is a controlled practical substitution
of backbone type, not a depth search. Adding another recurrent layer would
change depth, parameter count, regularisation, and optimisation at the same
time. Optimal depth is explicitly outside the current experiment.

PyTorch's built-in LSTM `dropout` is inter-layer dropout. Because the frozen
architecture has only one recurrent layer, the implementation sets it to
`0.0`; a nonzero value would not provide recurrent dropout within that layer.

## 4. Transformer backbone

### 4.1 Data flow

```text
[B, 64, 5]
     |
     v
Linear input projection: 5 -> 128
     |
     v
[B, 64, 128]
     |
     + deterministic sinusoidal position encoding
     |
     v
2 x Transformer encoder block
  - 4 attention heads (32 coordinates per head)
  - feed-forward network 128 -> 256 -> 128
  - GELU activation
  - dropout 0.1
  - post-norm residual structure
     |
     v
[B, 64, 128]
     |
     v
select final token [:, -1]
     |
     v
LayerNorm(128)
     |
     v
[B, 128]
```

### 4.2 Input projection

Each five-dimensional OHLCV observation is independently projected into the
128-dimensional Transformer model space:

$$
e_t = W_{in}x_t+b_{in}.
$$

This is a learned projection shared by all 64 positions. It changes feature
width but does not mix information between different timesteps.

### 4.3 Sinusoidal position encoding

Self-attention has no inherent concept of order, so a deterministic position
vector is added to every projected observation. For position `p` and model
coordinate pair `i`:

$$
PE(p,2i)=\sin\left(p\,10000^{-2i/128}\right),
$$

$$
PE(p,2i+1)=\cos\left(p\,10000^{-2i/128}\right).
$$

The 128 coordinates oscillate at different frequencies. This allows attention
to distinguish early context from recent context and to learn relationships
at multiple temporal offsets. The encoding is stored as a non-trainable buffer
of shape `[1,512,128]`; Phase 6 uses only positions `0` through `63`. It moves
with the model between CPU and GPU and is saved in the model state, but it is
not updated by the optimizer.

This encoding represents ordinal position inside the window. It does not
encode wall-clock time, market lifecycle, missingness, or contract identity.

### 4.4 Multi-head self-attention

Each encoder block forms queries, keys, and values from its input. With four
heads and model width 128, each head operates on 32-dimensional projections:

$$
Q_j=XW_j^Q,\qquad K_j=XW_j^K,\qquad V_j=XW_j^V,
$$

$$
\operatorname{Attention}_j(X)
=
\operatorname{softmax}\left(\frac{Q_jK_j^\top}{\sqrt{32}}\right)V_j.
$$

The four head outputs are concatenated and projected back to width 128. This
lets different heads learn different temporal relationships, such as recent
movement, longer-range state, or volume-price interactions; those
interpretations are possibilities, not guaranteed semantic assignments.

No causal attention mask is applied inside the 64-hour context. Every context
token may attend to every other context token because all 64 observations are
already available at decision time. This bidirectional historical attention
does not permit target leakage because future target-interval tokens are never
loaded.

### 4.5 Feed-forward and residual structure

Each of the two encoder blocks uses PyTorch's post-norm
`TransformerEncoderLayer` (`norm_first=False`):

```text
a = MultiHeadSelfAttention(x)
y = LayerNorm(x + Dropout(a))
f = Linear(256->128)(Dropout(GELU(Linear(128->256)(y))))
z = LayerNorm(y + Dropout(f))
```

The position-wise feed-forward network expands each token from 128 to 256
coordinates, applies GELU, and projects it back to 128. It operates on every
position independently; cross-position interaction occurs in self-attention.
The configured `0.1` dropout is used within attention, inside the feed-forward
path, and on the sublayer outputs. Residual connections preserve the input to
each sublayer, while post-layer normalisation controls the scale of the
combined result.

### 4.6 Final-token readout

There is no learned classification token and no mean pooling. After both
encoder blocks, the implementation selects `encoded[:, -1]`, corresponding to
the decision-time token. Because that token has attended to the full historical
window, it acts as the sequence summary. A final `LayerNorm(128)` produces the
returned `[B,128]` state.

## 5. Contrastive wrapper

The LSTM and Transformer backbones can each be placed inside
`TemporalContrastiveEncoder`:

```text
original sequence [B,64,5]
       |                       |
       v                       v
view 1: jitter -> scaling   view 2: scaling -> time mask
       |                       |
       v                       v
shared backbone             shared backbone
       |                       |
      h1                      h2            [B,128]
       |                       |
       v                       v
Linear 128->128, GELU, Linear 128->128
       |                       |
       v                       v
L2-normalised z1            L2-normalised z2
       \_______________________/
                  |
          NT-Xent, temperature 0.2
```

The two augmented views use the same backbone and projector parameters.
NT-Xent treats the two views of the same sequence as a positive pair and other
sequences in the batch as negatives. The projector is part of SSL training,
but `encode(x)` bypasses it and returns the unnormalised backbone state `h` for
downstream feature extraction.

## 6. BYOL wrapper

The same two backbones can instead be placed inside `TemporalBYOLEncoder`:

```text
view 1 ----------------> online backbone -> h1 -> projector -> predictor p1
view 2 ----------------> online backbone -> h2 -> projector -> predictor p2

view 1 ----------------> target backbone -> target projector -> z1_target
view 2 ----------------> target backbone -> target projector -> z2_target

loss = 0.5 * [D(p1, z2_target) + D(p2, z1_target)]
```

The online path contains:

- the selected LSTM or Transformer backbone;
- a `128 -> 128 -> 128` GELU projector; and
- a `128 -> 128 -> 128` GELU predictor.

The target path is initialized as a copy of the online backbone and projector.
It receives no gradients. After every optimizer step, its parameters are
updated using exponential moving average:

\[
\theta_{target}
\leftarrow
0.99\,\theta_{target}+0.01\,\theta_{online}.
\]

`D` is negative cosine regression: prediction and detached target are
L2-normalised before their cosine agreement is measured. BYOL uses no explicit
in-batch negatives. As with the contrastive wrapper, `encode(x)` returns the
online backbone's unnormalised state `h`, not a projector or predictor output.

## 7. Frozen architecture table

| Setting | LSTM | Transformer |
|---|---:|---:|
| Input | `[B,64,5]` | `[B,64,5]` |
| Backbone width | 128 | 128 |
| Recurrent/encoder layers | 1 | 2 |
| Direction/attention scope | oldest to newest | full historical window |
| Attention heads | n/a | 4, width 32 each |
| Feed-forward width | n/a | 256 |
| Backbone dropout | 0.0 | 0.1 |
| Position encoding | n/a | sinusoidal, maximum length 512 |
| Readout | final hidden state | final token, then LayerNorm |
| Backbone output | `[B,128]` | `[B,128]` |

The widths are matched, but parameter counts are deliberately not matched.
Every run records trainable and total parameter counts in
`architecture_manifest.json`; the surrounding run artifacts record training
time, inference time, and peak device memory where required. Phase 6 therefore
compares practical fixed-width architectures rather than equal-capacity
models.

## 8. Training and downstream lifecycle

For each walk, SSL family, and backbone, training uses only that walk's
target-free `encoder_train_sequences`. The frozen recipe is:

- seed `0`;
- batch size `256`, shuffled, incomplete final batch dropped;
- AdamW with learning rate `1e-3` and weight decay `1e-4`;
- one uninterrupted 50-epoch trajectory;
- snapshots at epochs 5, 15, and 50;
- NT-Xent temperature `0.2` for contrastive variants; and
- BYOL target decay `0.99` for BYOL variants.

Epoch 50 is the only checkpoint used to create downstream temporal features.
Epochs 5 and 15 are health and replay snapshots, not candidates selected from
downstream results. Encoders are trained separately for Walk 1 and Walk 2;
later-walk data never update or select an earlier-walk model.

The extracted 128-dimensional state enters the Phase 6 configuration matrix
in one of two ways:

- **substitution:** replace the corresponding 128-dimensional CNN branch,
  retaining a five-branch, 445-dimensional concat representation; or
- **addition:** retain the CNN branch and append the temporal branch, producing
  a six-branch, 573-dimensional concat representation compared against the
  corresponding duplicated-CNN width control.

Only the unchanged shallow downstream task head is trained. Feature scaling is
fitted coordinatewise on the task's training rows and then frozen for the
evaluation rows.

## 9. Validation and interpretation boundaries

The Phase 6 implementation validates:

- exact `[N,64,5]` input and `[N,128]` output shapes;
- finite losses, gradients, weights, and embeddings;
- non-collapse through average coordinate standard deviation;
- walk, dataset, identity, configuration, checkpoint, and prediction replay;
- absence of target and evaluation values during encoder training; and
- complete epoch-5/15/50 snapshot artifacts.

The comparison can test whether either frozen temporal backbone substitutes
effectively for its same-family CNN and whether it adds information beyond a
same-width duplicate control. It does not establish the optimal LSTM depth,
Transformer depth, attention-head count, hidden width, or a universally best
architecture. Results remain single-seed characterisation evidence and must be
reported separately by SSL family, downstream task, and walk.
