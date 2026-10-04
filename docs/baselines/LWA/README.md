# Learning Without Augmenting baseline dossier

**Project label:** `LWA-Frozen` (`LWA-F`)
**Phase:** 6.7 mandatory core external representation baseline
**Dossier status:** paper/source audit and all twelve owner decisions complete
**Implementation status:** model, cache/pretraining, frozen-feature,
downstream, replay, and launcher infrastructure implemented
**Experiment status:** both caches, both 50+50 trajectories, both 384-wide
master stores, and all six downstream trajectories are complete and valid

This directory freezes the evidence and proposed adaptation for:

> Berken Utku Demirel and Christian Holz, “Learning Without Augmenting:
> Unsupervised Time Series Representation Learning via Frame Projections,”
> NeurIPS 2025, arXiv:2510.22655v2.

The paper, official repository, and Phase 6.7 contract agree on the main
inference boundary: retain the time-domain encoder and two learned mappings
from the time representation into Fourier- and Gabor-domain representation
spaces, then concatenate the three 128-dimensional vectors into a
384-dimensional frozen representation.

## Bottom line

- LWA is suitable for the required Phase 6.7 frozen-representation comparison.
- The official repository has no software licence. No upstream source file is
  to be copied or modified into this project; implementation must be
  independently authored from the paper and audited behavior.
- The source does not support the project input length of 64 directly. Its
  Fourier and wavelet encoders contain length-specific linear reductions.
- The approved implementation follows the paper's approximately 500-parameter
  inference mapper (`1 -> 64 -> 1`) rather than the released mapper-training
  function's approximately 8-parameter `1 -> 1 -> 1` mapper.
- The approved mapping loss follows the paper's mean per-sample L1 sum rather
  than the source's coordinate mean followed by another batch-size division.
- The released workflow trains the domain encoders and embedding mappers first,
  then freezes them and trains new representation mappers for another complete
  epoch schedule. The project contract currently says one 50-epoch encoder
  trajectory and therefore needs an explicit two-stage interpretation.
- The source selects the minimum training-loss encoder checkpoint, whereas
  Phase 6.7 fixes the final epoch-50 checkpoint before evaluation.
- The paper used batch size 1024. The owner-approved project adaptation fixes
  physical batch size 128 for both walks, with no batch-size sweep, gradient
  accumulation substitute, or automatic fallback.
- `PyWavelets==1.8.0` is the frozen explicit dependency for complex Morlet
  CWT. Its metadata/runtime version and the approved `cmor1-1`, 48-by-64 CWT
  shape were verified in the professor-provided container. Training views will
  use chunked float32 disk-backed caches.
- The professor-provided Lumid container is the admitted execution platform.
  Its isolated Python 3.12 environment, CUDA-enabled PyTorch 2.12.1, 24 GiB
  RTX PRO 4000 Blackwell GPU, persistent project path, and CWT runtime have
  been audited. The independent Stage 1 package is implemented under
  `src/baselines/lwa/`; `PyWavelets==1.8.0` is installed locally and remotely,
  package metadata matches runtime versions, and all 17 focused tests pass in
  both environments, including the CWT path. Both full caches replay and the
  fixed-batch-128 admission smoke passed on Lumid.
- Exact implemented counts are 898,022 parameters in Stage A, 898 trainable
  parameters in Stage B, and 206,978 retained parameters at inference. The
  independent model omits audited upstream layers that are instantiated but
  unused by LWA outputs.

Both full FFT/CWT caches and both joint/mapper trajectories were created on
Lumid on 2026-10-05. Their 5/15/50 checkpoints and histories are retained.
The first launcher stopped before Walk 2 because the original strict
elementwise CPU replay check rejected a Wavelet probe whose maximum absolute
difference was `9.39e-5`, relative L2 was `3.02e-5`, and cosine similarity was
effectively one. Replay now records such scale-small cross-device drift as a
warning under the frozen relative-L2/cosine bounds; material, non-finite,
structural, provenance, and artifact-integrity failures remain fatal.

Both 384-dimensional master stores and all six common-head downstream runs
then completed. Every 5/15/50 downstream snapshot is valid. At epoch 50, LWA
does not consistently outperform immutable H0: Walk 1 classification and both
price MAEs are weaker, while isolated Walk 2 price/volatility RMSE and
correlation improvements are insufficient for a broad superiority claim. The
method-local result is retained under
`experiments/phase6_7/reports/lwa_staged_seed0/`.

## Documents

- `paper_reading_note.md` — method, evidence, limitations, and relevance.
- `official_code_audit.md` — exact source revision and paper/code differences.
- `architecture_and_dataflow.md` — tensor-level reconstruction for 64-by-5
  OHLCV.
- `upstream_clarification_request.md` — decisions requiring owner approval
  before implementation.
- `phase6_7_integration_proposal.md` — fair adaptation to the two walks and
  three common tasks.
- `implementation_plan.md` — staged code and experiment plan; no execution is
  authorized by the document itself.
- `source_manifest.json` — immutable pre-execution paper, source, licence,
  hardware, and gate record. Its historical pending flags are intentionally
  retained because experiment admission artifacts hash this file; completed
  runtime status is recorded by the feasibility and experiment manifests.

## Primary sources

- Local paper:
  `/mnt/e/zotero/storage/JNPX6KNG/Demirel and Holz - 2025 - Learning Without Augmenting Unsupervised Time Series Representation Learning via Frame Projections.pdf`
- arXiv v2: <https://arxiv.org/abs/2510.22655v2>
- NeurIPS page: <https://neurips.cc/virtual/2025/poster/118514>
- Official repository: <https://github.com/eth-siplab/Learning-with-FrameProjections>
- Audited source commit:
  `4461e916a107e7a64003286e72c866c62c647ba3`
