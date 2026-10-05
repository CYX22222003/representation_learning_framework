# TimeDART architecture and data flow

**Status:** owner-approved architecture; independent model and focused tests
complete  
**Target input:** 64 hourly OHLCV observations  
**Frozen feature width:** 170 (160 pooled states plus 10 instance statistics)

This document describes the owner-approved Phase 6.7 adaptation. The model
is implemented under `src/baselines/timedart/`; experiment infrastructure
is held for owner evaluation.

## What comes from the paper, source, and project

| Operation | Paper | Audited source | Project proposal |
|---|---|---|---|
| Keep embedding and encoder; discard denoising decoder | Section 3.3 | Forecast/classification fine-tuning models | Keep embedding and encoder frozen |
| Attention when using the encoder downstream | Figure 1 shows a noncausal downstream encoder, without a detailed tensor rule | `Model.forecast` and `ClsModel.forecast` pass unshifted clean embeddings with `is_mask=False` | Use the source forecast input/attention path; owner approved |
| Reduce patch states | Forecast uses a flatten head; classification uses max pooling | Forecast flattens `[N,D]` for each channel; classification max-pools a different joint-channel encoder | Max-pool `[N,D]` separately for each of the five channels: 160 coordinates; this combination is new |
| Restore the input scale | Paper says instance statistics are restored after reconstruction | Forecast output is multiplied by per-window std and shifted by per-window mean | Append five means and five stds to the shared feature, giving approved width 170; appending is new |
| Train downstream | Main tables fine-tune the encoder; Table 8 includes selected frozen linear probes | Ordinary trainer updates encoder and head | Freeze encoder and train the repository's common heads |

The paper does **not** specify the 170-coordinate feature. It is the
owner-approved Phase 6.7 adaptation. The source
forecasting head itself also stays channelwise; only the proposed common
project head can combine the five pooled channel vectors.

## 1. End-to-end workflow

```text
accepted walk bundle
    |
    +--> target-free encoder_train_sequences only [N,64,5]
    |
    +--> accepted OHLC probabilities + already walk-scaled volume [N,64,5]
    |
    +--> train TimeDART pretrainer for fixed 50 epochs
    |      no labels, no validation, no evaluation rows
    |
    +--> retain final epoch-50 patch embedding + Transformer encoder
    |      discard corruption, SOS training path, decoder, projector
    |
    +--> extract one 170-wide vector for every established task row
    |
    +--> train the unchanged Phase 6.7 native-width common probes
```

Each walk owns an independent model, optimizer, scheduler, RNG, and
checkpoints. The input bundle already owns its walk-specific volume scaler.

## 2. Proposed pretraining tensors

### 2.1 Accepted inputs and TimeDART instance normalization

Start with finite input in canonical order:

```text
x_input [B,64,5] = [open, high, low, close, walk-scaled volume]
```

The owner declined an additional walk-level five-channel scaler. The accepted
bundle retains unscaled OHLC probabilities and its existing training-interval
volume scaling. Inside TimeDART, compute detached row-local mean and
`sqrt(var + 1e-5)` over 64 timestamps separately for every channel. This is
the paper/source instance-normalization step and is replayed independently on
each historical input row.

```text
x_norm   [B,64,5]
```

No additional input clipping or fitted OHLCV scaling is applied. The shared
downstream probe still fits its own coordinate scaler on *training embeddings*
for each task and walk.

### 2.2 Channel independence and patch embedding

The approved channel path follows the paper's forecasting/source finance
path:

```text
x_norm                    [B,64,5]
permute/reshape           [B*5,64,1]
unfold P=2, stride=2      [B*5,32,2]
shared Linear(2,32)       [B*5,32,32]
```

Here “channel independent” means that each of the five single-channel
sequences passes through the same linear patch embedding and the same
Transformer weights. The official forecasting head also acts channelwise but
is not retained. The paper/source classification path uses a different
joint-channel convolutional embedding. Question 3 in the decision record
approves the shared per-channel forecasting encoder.

### 2.3 Shifted clean causal path

```text
clean embeddings e        [B*5,32,32]
concat(SOS,e[:,:-1])      [B*5,32,32]
+ sinusoidal position     [B*5,32,32]
2-layer causal encoder    [B*5,32,32]
```

At position `j`, direct patch embeddings contain only the learned SOS and
clean patches before `j`. The first position contains no clean patch. This
does **not** imply strict statistical independence from later raw values:
the source-style instance mean/std are computed over the whole 64-hour
window before patching, so normalized earlier patches change when a later
raw patch changes. The independent implementation preserves this source
behavior and tests both the direct mask boundary and normalization effect.

### 2.4 Independent cosine corruption

Register the 1,000-step cosine cumulative schedule as model buffers. Sample:

```text
t       [B*5,32] integers in [0,999]
epsilon [B*5,32,2]
x_t     = sqrt(gamma[t])*x_0 + sqrt(1-gamma[t])*epsilon
```

Every channel-patch pair samples its own timestep. `t` determines the
corruption strength but is not an input to the encoder or denoiser. The model
predicts the clean patch in one pass; it performs no iterative reverse
sampling. These are standard diffusion terms used to explain the audited
paper/source path; Question 7 adds no new mechanism.

Noisy patches use the same `Linear(2,32)` and positional encoding as the clean
path.

### 2.5 Self-only denoising decoder

Use one Transformer decoder block, eight heads, width 32, feed-forward width
64, and dropout 0.2.

For decoder position `j`:

- self-attention can attend only to noisy query `j`;
- cross-attention can attend only to causal encoder state `j`; and
- encoder state `j` summarizes clean patches strictly before `j`.

The proposed paper-aligned projector is `Linear(32,2)` applied independently
to every decoder position. Concatenate its 32 length-2 outputs back to length
64 per channel, restore `[B,64,5]`, undo instance normalization, and calculate
mean MSE against the accepted clean input.

The active source instead applies `Linear(32*32,64)` across every channel's
complete decoder state grid. The owner approved the paper-aligned patchwise
projector on 2026-10-05.

## 3. Proposed architecture

| Component | Setting |
|---|---|
| Input | accepted `[B,64,5]` with unscaled OHLC and already walk-scaled volume |
| Instance normalization | per row/channel over 64 timestamps |
| Channel handling | shared channel-independent encoder |
| Patch length / stride | `2 / 2` |
| Patch count | 32 |
| Model width | 32 |
| Encoder | 2 source-style post-norm residual Transformer blocks |
| Heads | 8 |
| Feed-forward width | 64 |
| Encoder/decoder dropout | 0.2 |
| Denoising decoder | 1 layer, self-only self/cross masks |
| Noise | 1,000-step cosine cumulative schedule; independent per patch |
| Clean projection | approved patchwise `32 -> 2` |
| Reconstruction loss | mean squared clean-input error |

The independent implementation should reproduce the paper/source operation,
not copy upstream class layouts or code.

## 4. Frozen extraction

For every task-aligned historical context:

```text
x_input [B,64,5]
  -> row-local instance normalization; retain mean/std [B,5] each
  -> channel independence                    [B*5,64,1]
  -> non-overlapping patches                 [B*5,32,2]
  -> frozen shared embedding + positions     [B*5,32,32]
  -> frozen encoder, no attention mask        [B*5,32,32]
  -> maximum over 32 patch positions          [B*5,32]
  -> restore channels                         [B,5,32]
  -> concat pooled O,H,L,C,V                  [B,160]
  -> append five means and five stds          [B,170]
```

The approved extraction uses the official forecasting downstream behavior:
no SOS shift and no causal mask. The official classification path also removes
the causal mask but uses a different joint-channel embedding and max pooling.
The project adapter keeps the forecasting encoder and borrows per-channel
max pooling. It appends the ten row-local normalization statistics that source
forecasting uses after its head to restore absolute scale, producing one
170-dimensional vector for all three tasks. Its
attention sees only the historical 64-hour input and is decision-time safe,
but the no-mask path differs from pretraining. Question 9 records approval.

Retaining the pretraining shift and causal mask unchanged would omit the last
two-hour clean patch from the encoder states because its target patch normally
enters only as the decoder's noisy query. Using a causal mask without the
shift would include the last patch but introduce a third, untrained attention
pattern. The recommended unmasked path follows the released forecasting
transfer path, although this project will keep the encoder frozen.

Discarded at extraction:

- diffusion schedule and random noise;
- learned SOS token;
- denoising decoder;
- reconstruction projector; and
- optimizer/scheduler state.

## 5. Output coordinate contract

```text
feature[:,   0: 32] = max-pooled open-channel states
feature[:,  32: 64] = max-pooled high-channel states
feature[:,  64: 96] = max-pooled low-channel states
feature[:,  96:128] = max-pooled close-channel states
feature[:, 128:160] = max-pooled volume-channel states
feature[:, 160:165] = window means for open, high, low, close, volume
feature[:, 165:170] = window stds for open, high, low, close, volume
```

This ordering must be recorded in the feature manifest and replayed exactly.

## 6. Training proposal

| Item | Proposed value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `0` |
| Scheduler | exponential decay, gamma `0.95` per epoch |
| Physical batch | 16 |
| Drop last | true |
| Walk 1 omitted remainder | 14 shuffled rows/epoch |
| Walk 2 omitted remainder | 9 shuffled rows/epoch |
| Epochs | 50 |
| Snapshots | 5, 15, 50 |
| Extraction checkpoint | final epoch 50 |
| Seed | 0 |
| Precision | float32, no AMP in primary run |

The architecture/optimizer settings follow the source's Exchange finance
configuration where possible. The project overrides source seed 2024 and its
validation selection with the shared seed-0, fixed-final-checkpoint contract.

## 7. Required invariants

- `64` is divisible by patch length `2`; patches never overlap.
- The shifted clean embedding at position `j` contains no direct embedding of
  clean patch `j` or later patches. Full-window instance normalization limits
  a stronger causal-independence claim.
- Noise timesteps vary independently across patch positions.
- Self-only decoder masks have exactly one permitted position per query.
- Schedule tensors are finite registered buffers and survive checkpoint load.
- Every walk and task row produces one finite `[170]` vector.
- Feature slice order is O/H/L/C/V and is batching-invariant in eval mode.
- Frozen extraction never constructs the decoder or samples noise.
- No target, evaluation row, or downstream metric influences pretraining.
