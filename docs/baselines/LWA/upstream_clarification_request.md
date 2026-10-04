# Learning Without Augmenting clarification and decision record

**Status:** all twelve decisions approved by the project owner on 2026-10-04
**Purpose:** freeze paper/source ambiguities before any LWA implementation or
training
**Downstream metrics consulted:** none

The questions below are internal implementation decisions. Contacting the
authors could improve provenance, but Phase 6.7 does not depend on receiving an
upstream reply if the project owner approves a disclosed independent
adaptation.

## Resolution record

The project owner approved Recommendations 1--8 and 11--12 as written. For
Recommendation 9, the owner selected an explicit `PyWavelets` dependency and
the proposed source-faithful, chunked float32 CWT cache. For Recommendation
10, the owner delegated one authoritative practical batch choice rather than a
batch-size sweep; physical batch size `128` is frozen for both walks. A
single-batch forward/backward smoke may confirm that this fixed configuration
runs, but it may not search batch sizes or change the value automatically.

## Recommended decision set

### 1. Reuse boundary

**Finding:** the official repository has no software licence.

**Recommendation:** independently implement LWA from the paper and audited
behavior. Do not copy upstream source, weights, or source-derived code blocks.
Report the method as `LWA-Frozen (paper-guided independent implementation)`,
not an official reproduction.

**Owner decision:** approved on 2026-10-04.

### 2. Inference-mapper hidden width

**Finding:** Appendix Table 15 and the paper's “about 500 parameters per
mapper” statement imply hidden width 64. Joint embedding mappers use 64 in the
source. The released `train_mappers` function uses hidden width 1 for the
actual inference mappers.

**Recommendation:** follow the paper and use hidden width 64 for both
representation mappers. Treat the source width 1 as an implementation
inconsistency.

**Owner decision:** approved on 2026-10-04.

### 3. Joint mapping-loss reduction

**Finding:** Equation 6 specifies `1/N` times an L1 norm and states there is no
additional weighting. The source uses coordinate-mean L1 and divides by batch
size again.

**Recommendation:** implement the paper expression as mean per-sample L1 sum:

```text
abs(prediction - target).sum(dim=1).mean()
```

for each of the two mappings, then add both terms unweighted to the three
pairwise symmetric NT-Xent losses. Do not reproduce the source's extra
`1/B` factor.

**Owner decision:** approved on 2026-10-04.

### 4. Two-stage 50-epoch budget

**Finding:** the official command's `n_epoch` is used once for joint
pretraining and again for frozen-encoder representation-mapper training. The
paper does not say whether its reported 256 epochs means 256+256 or 256 total.
Phase 6.7 requires a fixed 50-epoch characterization.

**Recommendation:** define one LWA trajectory as two predeclared stages:

1. 50 joint encoder/projector/embedding-mapper epochs, snapshots 5/15/50;
2. freeze the final joint epoch-50 encoders, then train representation mappers
   for 50 epochs, snapshots 5/15/50.

The frozen feature checkpoint is joint epoch 50 plus mapper epoch 50. Report
the resulting 100 total training-population traversals explicitly rather than
presenting it as a single 50-pass method.

**Owner decision:** approved on 2026-10-04.

### 5. Fixed checkpoint versus minimum training loss

**Finding:** the source uses the minimum joint training-loss checkpoint. The
project precommits epoch 50.

**Recommendation:** use the final joint epoch-50 checkpoint and final mapper
epoch-50 checkpoint. Training losses remain diagnostics only. Do not select a
checkpoint from training, validation, or evaluation metrics.

**Owner decision:** approved on 2026-10-04.

### 6. Learning-rate schedules

**Finding:** source joint pretraining uses Adam + cosine decay; the second
mapper stage uses separate Adam optimizers with no scheduler. The paper's
general implementation paragraph mentions cosine decay without separating
the stages.

**Recommendation:** preserve the explicit source behavior: joint Adam at
`3e-3`, weight decay `1e-6`, cosine decay to zero over 50 epochs; mapper-stage
Adam at `3e-3`, weight decay `1e-6`, constant learning rate. Save both
optimizer states.

**Owner decision:** approved on 2026-10-04.

### 7. Length-64 encoder adaptation

**Finding:** source transformed-domain encoders hard-code other lengths.

**Recommendation:** preserve their topology and set the analytically implied
reductions:

- Fourier magnitude and phase: `Linear(9,1)` after the two stride-2 blocks;
- wavelet: `Linear(8,1)` over time, then `Linear(6,1)` over scale.

Do not interpolate the input to a source-supported length and do not add an
adaptive pool absent from the source architecture.

**Owner decision:** approved on 2026-10-04.

### 8. Input normalization

**Finding:** source datasets use heterogeneous preprocessing and do not define
one universal OHLCV scaler. Phase 6.7 permits candidate-specific training-only
normalization.

**Recommendation:** fit one five-channel mean/std standardizer over every
timestamp of that walk's target-free encoder population. Replay it on all
training and extraction rows. Replace std below `1e-8` with 1.0. This matches
the already approved SaURL input boundary without sharing fitted statistics
between methods or walks.

**Owner decision:** approved on 2026-10-04.

### 9. CWT implementation and cache

**Finding:** source uses PyWavelets `cmor1-1`, 48 geometric scales 1--128,
and magnitude coefficients. `PyWavelets` is absent from the project
environment.

**Recommendation:** add `PyWavelets` as an explicit project dependency during
the implementation stage, compute CWT channelwise on CPU from normalized
training-only inputs, and store it as chunked float32 disk-backed cache data
with row/source/scaler hashes. Freeze the installed version in the runtime
environment manifest. Store no CWT for downstream rows because transformed
encoders are absent at inference.

**Owner decision:** approved on 2026-10-04. Include the dependency and the
chunked float32 cache.

### 10. Physical contrastive batch size

**Finding:** paper batch size 1024 was run on 24 GB; this project has 8,188 MiB.
NT-Xent changes when physical batch size changes, so gradient accumulation is
not equivalent.

**Recommendation:** use physical batch size `128` for both walks as one
authoritative FYP configuration. Preserve source `drop_last=True` and record
the omitted remainder rows per epoch: 30 of 39,070 shuffled rows in Walk 1 and
105 of 58,473 shuffled rows in Walk 2. Gradient accumulation, automatic batch
fallback, and a batch-size performance or resource sweep are excluded. A
single forward/backward smoke may verify that batch 128 runs on the selected
container; failure stops the gate and requires an explicit owner-approved
amendment rather than silently changing the batch.

**Owner decision:** approved on 2026-10-04 with fixed physical batch size 128.

### 11. Final concatenation order

**Finding:** paper Algorithm 2 uses `[time, Fourier-map, wavelet-map]`; source
uses `[Fourier-map, time, wavelet-map]`.

**Recommendation:** follow Algorithm 2 and the existing Phase 6.7 wording:
`[h_t, mapped_h_F, mapped_h_W]`. Freeze names and slice boundaries in the
feature manifest so the ordering is replayable.

**Owner decision:** approved on 2026-10-04.

### 12. Second-stage target normalization

**Finding:** the source trains representation mappers on raw encoder outputs,
not normalized representations. The joint stage normalizes projected
embeddings before both contrastive and embedding-map loss.

**Recommendation:** preserve that distinction: normalize projector outputs in
joint pretraining; train inference mappers from raw `h_t` to detached raw
`h_F/h_W`. Do not add target standardization.

**Owner decision:** approved on 2026-10-04.

## Approval record

Questions 1--12 are approved and must be copied into
`source_manifest.json`, `architecture_and_dataflow.md`,
`phase6_7_integration_proposal.md`, and `implementation_plan.md`. The
documentation decision gate is complete. Model implementation may begin only
when explicitly requested; training remains blocked on implementation,
dependency/environment verification, and the fixed-batch correctness/resource
smoke.
