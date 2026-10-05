# TimeDART official-code audit

**Audit date:** 2026-10-05  
**User-supplied repository:** <https://github.com/ustc-time-series/TimeDART>  
**Paper/README repository URL:** <https://github.com/Melmaphother/TimeDART>  
**Audited commit:** `e658a648cf6b04612ca643d10a634b45e136194c`  
**Commit date:** 2025-12-17  
**Remote refs:** one public `main`; no tags observed  
**History at audit:** 36 commits  
**Software licence:** none found

The checkout was read-only under `/tmp/TimeDART-official`. No upstream source
was copied into this project.

## 1. Repository contents

The repository is a Time-Series-Library-style experiment tree containing the
original TimeDART, later TimeDART-v2/Qwen additions, dataset loaders, generic
layers, pretrain/fine-tune scripts, and experiment runners. The key original
method files are:

- `models/TimeDART.py` — forecasting and classification model paths;
- `layers/TimeDART_EncDec.py` — channel independence, shifted SOS input,
  Transformer blocks, diffusion noise, decoder masks, and heads;
- `layers/Embed.py` — non-overlapping patching, linear patch embedding, and
  sinusoidal positions;
- `utils/masking.py` — causal, self-only, and partial masks;
- `exp/exp_timedart.py` — pretraining, validation selection, fine-tuning, and
  tests;
- `scripts/pretrain/*.sh` — dataset-specific architecture/training settings;
  and
- `run.py` — seed, arguments, and orchestration.

The original `models/TimeDART.py` path was last changed at commit
`fe63995d9c8dd06ab235d8c618f55c8a86fe01b8` on 2025-04-18; the audited head
also contains later TimeDART-v2 additions. Commit pinning is therefore
required even though the proposed adapter targets original TimeDART only.

## 2. Paper-to-code crosswalk

| Paper component | Official implementation | Audit result |
|---|---|---|
| Per-sample/channel instance normalization | `Model.pretrain` and `forecast` | Present for forecasting path |
| Non-overlapping patches | `Patch.unfold`, scripts set `patch_len=stride` | Present |
| Shared clean/noisy embedding | one `PatchEmbedding` instance | Present |
| Shifted SOS causal input | `AddSosTokenAndDropLast` + causal mask | Present |
| Independent noise step per patch | `sample_time_steps(x.shape[:2])` | Present |
| Cosine schedule | `Diffusion._cosine_beta_schedule` | Present |
| Self-only denoising mask | partial mask with default `mask_ratio=1.0` | Present under default scripts |
| One-layer denoising decoder | source scripts use `d_layers=1` | Present |
| Patchwise projection in Equation 8 | active `FlattenHead` maps all patch states jointly | Mismatch; patchwise head exists only as commented alternative |
| Timestep-conditioned reverse model | no timestep embedding or conditioning | Missing from both active model and paper equations |
| Discard decoder downstream | fine-tune model constructs encoder plus head | Present |
| Frozen generic representation | no single reusable extraction API | Missing/ambiguous |

## 3. Critical findings

### 3.1 No software licence

No `LICENSE`, `COPYING`, `NOTICE`, package licence declaration, or README
software licence was found. The ICML paper does not grant a software licence
for the repository. Direct copying, modification, or vendoring is not
admitted. Any implementation must be independently authored from the paper
and disclosed behavioral observations.

### 3.2 Repository identity is not release-pinned

The paper and README name `Melmaphother/TimeDART`, while the user supplied and
audited remote is `ustc-time-series/TimeDART`. The audited checkout reports
the latter as its origin but retains the former in documentation. There are no
tags or paper-matching releases. The exact commit and both public URLs must be
recorded; the work must not claim an official release reproduction.

### 3.3 Forecasting and classification are different encoder APIs

The forecasting `Model` treats each channel as an independent sample:
`[B,T,C] -> [B*C,T,1]`, applies one shared patch encoder, and reshapes states
back by channel for its head. The classification `ClsModel` uses a
multi-channel `Conv1d(C,D,kernel=P,stride=P)` embedding and max-pools the
resulting patch states. The paper does not select one of these as the generic
cross-task frozen representation.

“Channel independent” refers to both the shared linear patch embedding and
the shared Transformer encoder over one-channel sequences. The forecasting
head is also applied per channel but will not be retained in Phase 6.7.

### 3.4 The active reconstruction head differs from Equation 8

Paper Equation 8 maps decoder state `j` back to patch `j`. The active
forecasting code reshapes all decoder states and applies a linear layer from
`N*D` to the complete length `L` separately for each channel. This lets every
decoder position influence every reconstructed patch. A patchwise
`Linear(D,P)` implementation exists as `ARFlattenHead` but is commented out.

This difference changes both parameter count and the locality of the pretext
task. It requires an owner decision before independent implementation.

### 3.5 The diffusion step is never supplied to the model

The source samples one integer timestep per patch and uses it to construct the
noisy patch. It returns `t`, but `Model.pretrain` never passes `t` to the
embedding, encoder, decoder, or projection. No timestep embedding exists in
the original path. The network must infer corruption severity from the noisy
values alone. There is also no iterative reverse sampling.

### 3.6 Noise schedule tensors are not registered buffers

The official `Diffusion` class stores beta/alpha/gamma as ordinary tensors on
the construction device. They do not participate in `state_dict` and do not
follow a later `.to(device)` automatically. An independent implementation
should register them as non-trainable buffers while preserving their numeric
definition.

### 3.7 Decoder masking is source-default dependent

`DenoisingPatchDecoder` calls `generate_partial_mask` independently for its
self-attention and cross-attention masks. With the source default
`mask_ratio=1.0`, both are deterministic self-only masks as described by the
paper. At smaller ratios, mask sampling is stochastic and the two masks can
differ. Every provided original pretraining script relies on the default
`1.0`; the project should freeze self-only explicitly.

### 3.8 Downstream inference removes the causal mask

Both official forecasting and classification fine-tuning call the encoder
with `is_mask=False` and do not prepend the SOS token. This matches Figure 1's
visual distinction between the causal pretrainer and downstream Transformer,
but the paper prose does not emphasize it. A frozen extractor therefore needs
an explicit causal/unmasked decision.

The independently authored project extractor now uses the owner-approved
unmasked forecasting path and returns 170 coordinates: 160 channelwise pooled
encoder states plus the ten per-window statistics carried by source
forecasting outside its encoder. The latter concatenation is a project
adaptation, not a released source vector.

### 3.8a Full-window normalization limits strict autoregression

`Model.pretrain` computes each channel's mean and standard deviation over the
entire input window before patching and causal attention. Changing a later
clean patch changes those statistics and therefore the normalized values of
earlier patches. The SOS shift and causal mask prevent direct attention to
later *patch embeddings*, but they do not remove this statistical dependence.
The independent model preserves the audited behavior and has a focused test
for both effects. Downstream decision-time inputs contain only the historical
64-hour window, so this is a pretext-task interpretation caveat rather than
future-label leakage in the project evaluation.

### 3.9 Source checkpointing is incompatible with Phase 6.7

Pretraining constructs a validation loader, saves `ckpt_best.pth` when
validation reconstruction improves, and also saves every tenth epoch. Fine-
tuning monitors validation metrics, computes test metrics during every epoch,
and uses early stopping. Phase 6.7 permits none of these. The project must use
one uninterrupted training-only 50-epoch trajectory and fixed epoch 50.

The paper's Table 8 separately reports frozen linear probing, but the audited
`Exp_TimeDART` runner exposes no dedicated original-TimeDART linear-probe
switch or frozen-encoder training branch. Its ordinary fine-tuning optimizer
receives all model parameters. The project common probes therefore have a
paper-level precedent but are not a direct replay of a released Table 8 path.

### 3.10 Checkpoints omit part of the pretraining model

The source saves only parameters whose names contain `encoder` or
`enc_embedding`. This excludes the denoising decoder and head as intended, but
also excludes the learned SOS token. Downstream happens not to use SOS. The
project should save complete training state for replay and separately export
the retained extractor state.

### 3.11 Hyperparameters are dataset-specific

The source scripts vary `d_model` from 8 to 512, patch length from 1 to 75,
learning rate from `1e-4` to `1e-3`, and batch from 8 to 128. The closest
published source analogue by domain is Exchange (finance): `D=32`, `d_ff=64`,
`P=S=2`, two encoder layers, one decoder layer, eight heads, learning rate
`1e-3`, batch 16, 50 epochs, and exponential decay `0.95`. Selecting this
analogue is a project adaptation, not a paper-mandated setting.

### 3.12 Source requirements do not prove the paper environment

The current repository pins Python 3.10+ packages including Torch 2.6.1. The
paper reports one RTX 4090 but not an exact Python/Torch stack. The current
requirements were updated after ICML acceptance and cannot be treated as the
historical result environment.

## 4. Read-only CPU smoke

The official source was imported without modification using the project's
existing virtual environment. On synthetic `[2,64,5]` with the proposed
finance-like source settings:

```text
pretrain output                 [2,64,5]
official downstream states      [10,32,32] = [B*C,N,D]
finite forward/backward         pass
official pretraining params     104,160
retained embedding+encoder      25,632
all-state flatten width         5,120
per-channel pooled concat width 160
global pooled width             32
```

This establishes static/runtime feasibility only. It is not a source-result
reproduction or an authorization to reuse source code.

## 5. Feasibility verdict

**Decision:** technically feasible; independently authored model implementation
and focused CPU checks complete. The owner is evaluating this model before
experiment infrastructure proceeds.

Recommended admission is an independently authored TimeDART model evaluated
with a frozen encoder. The internal artifact ID remains `timedart_frozen`:

- original TimeDART, not TimeDART-v2;
- explicit non-overlapping `P=2` patches and finance-oriented width 32;
- paper-aligned patchwise reconstruction;
- source-aligned independent cosine corruption and self-only decoder;
- an approved 170-wide channel-preserving frozen vector consisting of 160
  pooled encoder coordinates and ten source-used instance statistics;
- no additional five-channel input scaler on the already prepared Phase 5
  OHLCV bundle;
- fixed 50-epoch training-only trajectories; and
- the full two-walk/three-task optional extension matrix.

The owner resolved all eleven decisions on 2026-10-05; Question 7 was a
terminology clarification. The independent model and ten focused CPU tests
are complete, with a finite real-row forward/backward smoke. The model has
30,242 pretraining parameters and retains 17,248 encoder parameters. No
downstream metric has been inspected to form these decisions, and no
TimeDART training trajectory has been launched.
