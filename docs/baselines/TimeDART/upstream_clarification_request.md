# TimeDART clarification and decision record

**Status:** all eleven owner decisions approved on 2026-10-05; Question 7
closed as terminology clarification  
**Purpose:** freeze paper/source ambiguities before any TimeDART implementation
or training  
**Downstream metrics consulted:** none

The questions below are internal adaptation decisions. The owner confirmed
independent implementation using the source as reference, original TimeDART,
the paper's patchwise projector, the proposed width and patch settings, no
additional walk-level input scaler, deterministic self-only masks, the fixed
training recipe, severity-aware replay, and the full optional matrix. The
The owner subsequently approved the forecasting channel-independent encoder
and the revised 170-coordinate frozen extraction. Question 7 explains
standard diffusion terms and adds no TimeDART mechanism.

## Recommended decision set

### 1. Reuse boundary

**Finding:** no software licence was found in either named repository path.

**Recommendation:** independently implement original TimeDART from the paper
and audited behavior. Use the source as an attributed reference and author
the project code independently. Keep `timedart_frozen` as an internal artifact
identifier for the frozen-encoder probing protocol. In prose, call it
“TimeDART (frozen encoder; independent implementation).” The suffix describes
how this project evaluates the representation; it does not assert that the
published model has a separate “Frozen” variant.

**Owner decision:** approved on 2026-10-05 with the naming clarification
above. The source is a behavioral reference, not code to transplant.

### 2. Original TimeDART versus TimeDART-v2

**Finding:** the audited repository includes later Qwen-based TimeDART-v2
work, but the ICML method and Phase 6.7 rationale concern the compact original
causal Transformer.

**Recommendation:** implement only original TimeDART. Exclude Qwen,
generative-tuning additions, and all v2 scripts.

**Owner decision:** approved on 2026-10-05. Original ICML TimeDART only.

### 3. Multivariate encoder path

**Finding:** source forecasting first reshapes `[B,64,5]` into
`[B*5,64,1]`. Every single-channel sequence then passes through the *same*
`Linear(P,D)` patch embedding and Transformer weights; this is what
“channel independent” describes. The source forecast head later applies one
shared linear map per channel, but that head is excluded from the project's
frozen representation. The source classification path instead uses a
different joint-channel `Conv1d(5,D,P,stride=P)` embedding and its own
pretraining path. The project needs one encoder for all three common probes.

**Recommendation:** use the forecasting/channel-independent path because the
project's price and volatility probes are forecasting-oriented and the
closest source finance configuration uses this path. Preserve channel identity
in the final vector instead of averaging channels.

**Owner decision:** approved on 2026-10-05 after clarification: use the shared
per-channel forecasting encoder for pretraining and all three frozen probes.

### 4. Reconstruction projector

**Finding:** paper Equation 8 specifies one projection per decoder patch. The
active source flattens all patch states and predicts the complete channel
sequence jointly; a patchwise head exists only as commented code.

**Recommendation:** follow the paper and use shared `Linear(D,P)` patchwise
projection. This preserves the intended local denoising task and prevents the
reconstruction head from mixing all target patches. Record the active source
head as a reproducibility discrepancy.

**Owner decision:** approved on 2026-10-05. Use one shared `Linear(D,P)` per
decoder patch; document the active source's all-patch flattening head.

### 5. Architecture and patch length

**Finding:** the paper/source settings are dataset-specific. The Exchange
finance script is the closest domain analogue and uses `P=2`, `D=32`,
`d_ff=64`, two encoder layers, one decoder layer, and eight heads.

**Recommendation:** freeze those settings for both walks. With length 64 this
gives 32 non-overlapping patches. Do not tune patch length or width from any
Phase 6.7 evaluation metric.

**Owner decision:** approved on 2026-10-05.

### 6. Input normalization

**Finding:** the accepted Phase 5 input already contains unscaled OHLC
probabilities and walk-training-scaled volume. Official forecasting data
loaders standardize the data before the model's own per-instance/channel
normalization, but an additional walk-level five-channel standardizer is not
needed for this project.

**Approved adaptation:** consume the accepted `[B,64,5]` bundle directly.
Retain TimeDART's in-model, detached per-instance/channel normalization
(`sqrt(var + 1e-5)`) and reverse it before computing reconstruction MSE
against the accepted input. Apply that same in-model normalization during
feature extraction. Do not fit another walk-level OHLCV scaler. The common
downstream probe still fits its separate embedding-coordinate scaler on task
training rows only.

**Owner decision:** approved on 2026-10-05: no additional walk-level input
scaler; existing volume scaling and TimeDART instance normalization remain.

### 7. Diffusion semantics

**Finding:** paper and source use independent patch timesteps and cosine
corruption. “Timestep conditioning” would pass the sampled noise level `t`
into a denoiser, usually through a learned embedding, so it can use a
different rule at each noise severity. “Iterative reverse sampling” would
start with noise and repeatedly apply a denoiser across timesteps to generate
a sample. The released TimeDART path does neither: `t` chooses the noise
strength for one training corruption, the model predicts the clean patch in
one pass, and only the encoder survives downstream.

**Recommendation:** preserve these exact semantics. Register schedules as
buffers for reliable replay, but do not add timestep embeddings, noise
prediction, iterative sampling, or another “fix” that would create a different
method.

**Resolution:** informational clarification on 2026-10-05. The terms describe
operations in conventional diffusion models and in Appendix C's generic
derivation; they are absent from the audited TimeDART architecture and source
execution path. No additional owner choice is needed to reproduce that path.

### 8. Decoder mask

**Finding:** source `mask_ratio=1.0` realizes paper self-only attention; lower
ratios create stochastic partial masks.

**Recommendation:** use deterministic self-only masks for both decoder
self-attention and encoder cross-attention. Do not expose `mask_ratio` as a
training choice in the frozen run.

**Owner decision:** approved on 2026-10-05.

### 9. Frozen extraction boundary

**Finding:** the paper transfers the embedding and encoder and discards the
denoising decoder, but does not define one generic frozen vector. In
pretraining the encoder receives `[SOS,e_1,...,e_31]` with a causal mask, so
state `j` predicts clean patch `j` from earlier patches. In official forecast
fine-tuning it instead receives `[e_1,...,e_32]` with the attention mask
removed; all 32 states are flattened into a newly trained task head and the
encoder weights are updated. Official classification fine-tuning uses its
separate joint-channel embedding, also removes the causal mask, max-pools
states, and updates the encoder. The project needs one frozen vector for three
common heads, so direct reproduction of either task path is impossible.

**Revised recommendation:** remove noise, SOS shift, decoder, and
reconstruction head; run the clean embedding plus encoder with no attention
mask as in official forecasting downstream inference; max-pool the 32 patch
states separately per channel and concatenate O/H/L/C/V. Also append the five
detached per-window means and five `sqrt(var + 1e-5)` values that source
forecasting uses to restore its predictions to the input scale. This gives
`5*32 + 5 + 5 = 170` coordinates. The 160 learned-state coordinates plus ten
deterministic instance statistics are a disclosed project extraction. The
encoder is *not* fine-tuned. Because all 64 input hours precede the decision
time, unmasked attention and the appended statistics do not expose a future
target. Unmasked attention does change the pattern from pretraining.

**Provenance boundary:** the paper specifies removing the decoder and using
task-specific flatten or max-pool heads, but never defines this 170-coordinate
vector. Unmasked, unshifted forecasting encoder input and instance
denormalization are explicit in the released code. Per-channel max pooling
borrows the reduction idea from the different classification path; appending
mean/std to a reusable vector is a new project choice. Neither combination is
claimed to be the paper's original downstream architecture.

This addition matters for the project's absolute-price task. Instance
normalization makes two windows that differ mainly by price level look nearly
identical to the encoder. The released forecasting model can still produce an
absolute-scale prediction because it keeps the original instance mean and
standard deviation outside the encoder. A 160-coordinate encoder-only vector
would discard that information. The accepted training close-window means
span approximately `0.0010`--`0.9979` in Walk 1 and `0.0013`--`0.9950` in
Walk 2; this is not a negligible scale range.

The main alternative is to retain the shifted SOS input and causal mask
exactly as pretrained. Its 32 states then omit the final input patch from the
clean-history encoder path; that patch is normally supplied as the noisy
decoder query, which is removed at extraction. This would omit the most
recent two hours from the representation. A causal, unshifted path could
include all 64 hours, but it is a third inference pattern, different from
both the pretraining and released downstream paths.

Other extraction choices considered:

- global channel-and-patch pooling (`32` wide) discards channel identity;
- flattening every state (`5,120` wide) greatly enlarges the common probe;
- encoder-only per-channel max pooling (`160` wide) drops the original scale
  metadata used by source forecasting;
- concatenating latent mean and max (`320` wide before instance statistics)
  invents a new pooling rule not used by the source.

**Owner decision:** approved on 2026-10-05 after the paper/source/project
crosswalk: use the revised 170-coordinate feature and the common frozen
probing heads. The owner accepts that per-channel pooling and appending
instance statistics are project adaptations absent from the paper.

### 10. Pretraining recipe and row use

**Finding:** the Exchange source uses Adam `1e-3`, batch 16, 50 epochs, and
exponential LR decay `0.95`; source loaders shuffle and drop the final short
batch. The project requires seed 0 and snapshots 5/15/50.

**Recommendation:** use physical batch 16, `drop_last=True`, Adam `1e-3`, no
weight decay, gamma `0.95`, float32, 50 epochs, seed 0, and final epoch 50.
Record the per-epoch shuffled remainders (14 Walk 1, 9 Walk 2). No gradient
accumulation, validation, early stopping, or automatic batch fallback.

**Owner decision:** approved on 2026-10-05.

### 11. Replay tolerance and complete state

**Finding:** the source saves only part of the model and does not provide a
standalone replay contract.

**Approved adaptation:** checkpoints 5/15/50 save full model, optimizer,
scheduler, RNG/sampler, data/config/source hashes, and fixed probes. There is
no extra input scaler to save. Same-device replay should be exact where
deterministic. A strict elementwise CPU/CUDA miss becomes a recorded warning
and the Lumid launcher continues when relative L2 is `<=5e-4`, cosine is
`>=0.999999`, and all structural/provenance checks pass. Missing or corrupt
artifacts, wrong rows/configuration, non-finite values, and material numerical
drift remain fatal; treating those as warnings would invalidate downstream
artifacts.

**Owner decision:** approved on 2026-10-05 with this severity distinction.

### 12. Optional-extension scope

**Finding:** TimeDART is post-core exploratory scope, but the user has now
commissioned its preparation. Partial task execution after seeing an
intermediate result would bias the extension.

**Recommendation:** once admitted, freeze the complete inventory before the
first run: two walk-specific encoder trajectories, two master stores at the
width fixed by Question 9, and six downstream trajectories across
classification, future price, and
future realised variance, with all 5/15/50 snapshots retained. TimeDART cannot
replace or redefine the completed H0/SaURL/LWA core.

**Owner decision:** approved on 2026-10-05 for the complete two-walk,
three-task scope. Feature width remains contingent on Question 9.

## Implementation gate

The model and extraction decisions are complete. Independently authored model
implementation may begin. Training still requires the later CUDA/resource
gate and the staged implementation/replay checks.
