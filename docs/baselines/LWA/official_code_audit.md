# Learning Without Augmenting official-code audit

**Repository:** <https://github.com/eth-siplab/Learning-with-FrameProjections>
**Audited commit:** `4461e916a107e7a64003286e72c866c62c647ba3`
**Commit date:** 2026-01-24
**Branches/tags:** one public `main` branch; no tags
**History at audit:** 20 commits
**Software licence:** none found

The checkout was read only. No upstream source was copied into this project.

## 1. Repository contents

The repository includes one general runner, a shared training/linear-evaluation
module, dataset-specific preprocessing, model definitions, baseline methods,
and unpinned environment files. The LWA implementation is internally named
`IsoAlign`.

The most relevant files are:

- `models/frameworks.py:301-469` — joint LWA model and integrated inference
  encoder;
- `models/models_nc.py:504-624` — time-domain ResNet;
- `models/models_nc.py:683-788` — Fourier encoder and unused reconstruction
  wrapper;
- `models/models_nc.py:798-829` — convolutional mapping;
- `models/models_nc.py:831-933` — wavelet encoder;
- `models/backbones.py:523-569` — projectors;
- `trainer_SSL_LE.py:365-448` — joint pretraining;
- `trainer_SSL_LE.py:639-727` — representation-mapper training;
- `utils.py:11-124` — CWT and orthonormal real FFT; and
- `main.py:9-239` — arguments, seeds, and end-to-end orchestration.

## 2. Paper-to-code crosswalk

| Paper component | Official implementation | Audit result |
|---|---|---|
| Eight-block time ResNet, width 128 | `ResNet1D(..., n_block=8, output_dim=128)` | Present |
| Fourier amplitude/phase encoder | `FourierEncoder` | Present, with length-specific reductions |
| 48-scale Gabor/Morlet view | `WaveletTransform(wavelet='cmor1-1')`, magnitude only | Present |
| Wavelet 2D encoder | `UNET_2D_simp` encoder path | Present, length-specific |
| Three two-layer projectors | three `Projector(model='IsoAlign')` instances | Present |
| Pairwise three-domain NT-Xent | `IsoAlign.cont_loss` over three unordered pairs | Present; each pair is symmetric |
| Embedding mappings `Phi_z` | two `ConvMapping(... hidden_channels=64)` modules | Present |
| Representation mappings `Phi_h` | `train_mappers` | Present, but capacity differs from paper |
| Efficient 384-wide inference | `IntegratedEncoder` | Present |
| Adam and cosine schedule | joint stage uses Adam + cosine | Present only for joint stage |
| Frozen linear probing | `lock_backbone` + linear classifier | Present, with dataset-dependent validation logic |

## 3. Critical implementation findings

### 3.1 No software licence

No `LICENSE`, `COPYING`, `NOTICE`, or repository licence declaration was found.
The paper PDF is distributed under CC BY-NC-ND 4.0 on arXiv, but that does not
supply a software licence for the repository. Direct source reuse is therefore
not admitted. A clean, independently authored implementation is required.

### 3.2 The source revision postdates the paper revision

The paper version read is arXiv v2 from 2026-01-15. The audited source commit
is from 2026-01-24. The final commit changes the representation-mapper Fourier
target from the time representation to the Fourier representation. This is
consistent with the paper, but it means source commit identity is essential.

### 3.3 A post-publication data-leak fix affects provenance

Commit `a3e09f7fa5c29f2233c2dd4f929c7021e7bd56d6` on 2026-01-16 changes an HHAR
split branch from `if` to `elif` to prevent a test user also entering the
training set. The repository does not provide regenerated result artifacts or
a release tying Table 2 to the corrected split. This does not affect the
project's walk data, but it limits exact reproduction claims about the paper.

### 3.4 Sequence length 64 is unsupported directly

The time ResNet is length-agnostic after global average pooling. The Fourier
encoder, however, defines final linear sizes only for input lengths 100, 128,
200, 480, and 1000. The wavelet encoder similarly defines reductions for a
small hard-coded set of time sizes. Length 64 has no branch in either class.

For `[B,64,5]`, the source topology yields:

- `rfft`: `[B,5,33]`, then two stride-2 residual reductions to length 9;
- CWT: `[B,5,48,64]`, then three stride-2 2D reductions to
  `[B,128,6,8]`.

The minimal topology-preserving adaptation is therefore Fourier `Linear(9,1)`
for each magnitude/phase branch, wavelet `Linear(8,1)` over time, then
`Linear(6,1)` over scales.

### 3.5 Paper/source inference-mapper capacity conflict

Appendix Table 15 and the parameter discussion specify each mapping as
`Conv1d(1,64,3) -> ReLU -> ConvTranspose1d(64,1,3)`, approximately 449
parameters for the source implementation. Joint embedding mappings use this
configuration.

`train_mappers`, which creates the actual inference mappings, instead passes
`hidden_channels=1`. Each such source mapper has only approximately eight
parameters. The paper's claim of roughly 500 parameters per mapper and roughly
1,000 additional inference parameters agrees with hidden width 64, not 1.

### 3.6 Mapping-loss scaling conflict

The paper's Equation 6 is an unweighted sum of mean per-sample L1 norms. In
`IsoAlign.calc_loss`, the source first calls `torch.nn.functional.l1_loss`
with default mean reduction and then divides each mapping loss by batch size.
This adds another factor of `1/B` and also averages coordinates, making the
relative mapping weight depend strongly on batch size. At `B=1024`, this term
is extremely small relative to the three contrastive losses.

The second mapper stage uses ordinary mean L1 loss without the extra division.

### 3.7 The published epoch count maps to two source loops

The runner first executes `train(... n_epoch)` for joint encoders/projectors/
embedding mappings. It then executes `train_mappers(... n_epoch)` after
freezing the joint model. A command with `--n_epoch 256` therefore performs
256 joint epochs plus 256 representation-mapper epochs. The paper says the
models are trained for 256 epochs but does not explicitly describe this total
of 512 training-population traversals.

### 3.8 Source checkpoint selection conflicts with Phase 6.7

Joint training deep-copies the minimum training-loss model across epochs and
returns it rather than the final fixed-budget model. This is training-only
selection, but it differs from the project's precommitted epoch-50 extraction
rule. The representation-mapper stage also tracks a minimum-loss state, but
does not restore it: the returned `IntegratedEncoder` contains the final
mapper parameters.

In addition, the mapper stage writes its mapper-only state to the same
`results/<model_name>_best.pt` path used by joint pretraining, overwriting the
standalone source checkpoint file.

### 3.9 The second stage has different optimizer semantics

Joint pretraining uses Adam, weight decay `1e-6`, and cosine annealing. The two
representation mappers use separate Adam optimizers at the same learning rate
and weight decay but no scheduler. The paper's general cosine-decay statement
does not distinguish the stages.

### 3.10 Paper/source inference ordering differs

Algorithm 2 presents `[h_t, mapped h_F, mapped h_W]`. `IntegratedEncoder`
concatenates `[mapped h_F, h_t, mapped h_W]`. This is a coordinate permutation
and does not change representational content, but it must be frozen for
deterministic feature hashes.

### 3.11 Source downstream selection is incompatible with the project

Some source dataset cases save the final linear classifier, while others pick
the minimum validation loss. The project permits neither a validation split
nor evaluation-driven model selection. Only the frozen LWA representation is
relevant; downstream probing must reuse the established Phase 6.7 head
contract.

### 3.12 Resource and dependency mismatch

The paper used RTX 4090 GPUs with 24 GB. At the static source-audit stage, the
fallback local device was an RTX 4060 Laptop GPU with 8,188 MiB. The admitted
Lumid runtime subsequently provided a 24 GiB RTX PRO 4000 Blackwell GPU.
Physical batch size still cannot be silently tuned after reading downstream
metrics; the owner-frozen value remains 128 and requires its model-specific
forward/backward smoke.

The official requirements are unpinned. The environment available during the
static audit lacked `PyWavelets` and `einops`. The admitted container now pins
and verifies `PyWavelets==1.8.0`; the independent adapter does not require
`einops`.

## 4. Static feasibility verdict

**Decision:** admit an independent paper-guided LWA-Frozen implementation.
The owner resolved all conflicts in `upstream_clarification_request.md` on
2026-10-04, including explicit PyWavelets use, chunked float32 CWT caching,
and authoritative physical batch size 128. The professor-container runtime
and exact CWT dependency are verified. The independent Stage 1 model and
focused CPU tests are now implemented. Training remains gated on owner review,
remote transform/cache replay, and the fixed-batch forward/backward smoke.

The method is technically feasible for 64-by-5 OHLCV:

- the time encoder supports five channels without structural changes;
- Fourier and CWT views are well-defined channelwise over 64 timestamps;
- the hard-coded reduction sizes can be generalized analytically;
- the inference representation remains exactly 384 dimensions; and
- auxiliary transformed-domain computation disappears at inference.

A source-shaped static parameter estimate for the 64-by-5 adaptation is about
0.91 million parameters during joint pretraining. With the paper's
64-hidden-channel mappings, the retained time encoder plus two inference
mappers is about 0.207 million parameters. Exact counts were generated from
the independently implemented model before the training gate.

The completed independent model has 898,022 Stage-A parameters and 206,978
retained inference parameters; Stage B trains only 898 new mapper parameters.
The Stage-A count is 9,155 below the earlier source-shaped estimate because the
adapter deliberately omits audited source parameters that never participate in
LWA output: an unused Fourier linear layer, disabled/unused normalization
parameters, and the time backbone's unused three-class head.

Uncompressed training-only caches are estimated at:

| Walk | Encoder rows | CWT float32 | complex64 rFFT |
|---:|---:|---:|---:|
| 1 | 39,070 | 2.236 GiB | 49.2 MiB |
| 2 | 58,473 | 3.346 GiB | 73.6 MiB |

Caching policy and free-disk checks belong to the later feasibility script.

## 5. Direct-reuse boundary

Direct reuse would require an explicit software licence and a source release
that resolves the paper/code conflicts, supports length 64, freezes an exact
two-stage schedule, and preserves complete checkpoints without validation or
best-on-evaluation selection. Those conditions are not currently met.
