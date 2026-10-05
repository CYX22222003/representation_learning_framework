# TimeDART baseline dossier

**Project label in prose:** TimeDART (frozen encoder; independent implementation)
**Artifact method ID:** `timedart_frozen` (`TD-F` in the Phase 6.7 plan)
**Phase:** 6.7 optional post-core external representation extension
**Dossier status:** paper/source audit and all eleven owner decisions complete;
Question 7 was an informational clarification
**Implementation status:** model plus TimeDART-only data inventory,
pretraining/resume, frozen-feature, native-width downstream, advisory replay,
and one-run launcher infrastructure complete; 15 focused CPU tests pass
**Experiment status:** no TimeDART training, feature extraction, or downstream
evaluation has run

This directory records the evidence and proposed project adaptation for:

> Daoyu Wang, Mingyue Cheng, Zhiding Liu, and Qi Liu, “TimeDART: A
> Diffusion Autoregressive Transformer for Self-Supervised Time Series
> Representation,” ICML 2025, PMLR 267.

## Bottom line

- TimeDART is technically feasible on the project's finite `[N,64,5]` OHLCV
  encoder populations and directly tests a useful causal-autoregressive plus
  denoising pretraining alternative.
- The official repository has no software licence. No upstream source file is
  to be copied, modified, or vendored into this project. The safe candidate is
  an independently authored, paper-guided implementation with the official
  code used as an attributed behavioral reference.
- `Frozen` names this project's encoder evaluation protocol, not a distinct
  published TimeDART variant. The paper's main downstream runs fine-tune the
  encoder; our common probes keep it fixed. The method ID retains the suffix
  for artifact consistency, while prose uses “TimeDART (frozen encoder).”
- The paper and source do not define a unique generic frozen vector. The
  forecasting path is channel-independent and flattens all patch states into
  its supervised head, while the classification path jointly embeds channels
  and max-pools patch states. Phase 6.7 requires one reusable representation
  for all three tasks.
- The owner-approved project extraction is 170-dimensional: with the
  finance-oriented source configuration (`patch_len=2`, `d_model=32`), max-pool
  the 32 patch states separately for each OHLCV channel, concatenate them in
  `open, high, low, close, volume` order, and append the five per-window means
  and five standard deviations used by the source forecast path.
- The active source reconstruction head globally flattens all decoder patch
  states, whereas paper Equation 8 describes a patchwise projection. The
  owner approved the paper's patchwise `32 -> 2` projector.
- The accepted Phase 5 bundle already has walk-training-scaled volume and
  unscaled OHLC probabilities. The owner declined another walk-level input
  scaler. TimeDART's own per-instance/channel normalization remains.
- Neither the paper nor source conditions the denoiser explicitly on the
  sampled diffusion step, and the method performs no iterative reverse
  sampling. The project should preserve and disclose this behavior rather than
  silently converting TimeDART into a different diffusion model.
- The official trainer uses validation loss, best-checkpoint selection, and
  downstream early stopping. Those procedures are incompatible with the
  project's train/test-only fixed-budget contract and will not be reused.
- The independent model lives under `src/baselines/timedart/`. Its experiment
  path lives under `src/data_processing/phase6_7_timedart_data.py`,
  `src/training/phase6_7_timedart.py`,
  `src/features/phase6_7_timedart_features.py`, and `scripts_v6/`. Fifteen
  focused CPU tests pass. No trajectory, feature store, or downstream head
  has been run.
- Data preparation inventories the exact six existing Phase 5/6 bundles
  without copying them. Encoder training opens only target-free sequences and
  identities; feature extraction opens only task contexts and identities;
  downstream labels are loaded only by the corresponding common probe.
- Replay validators are advisory entry points: they emit structured warnings
  and return successfully. Functional stage failures such as absent inputs,
  unreadable checkpoints, non-finite training, or inability to create a
  required output still stop that stage.
- Full-window instance normalization happens before the causal pretraining
  mask. It lets later values affect earlier normalized patches through the
  window mean/std, even though direct attention remains causal. This limits
  a literal strict-autoregression claim for the pretext task; the complete
  input window is historical at downstream decision time.
- The persistent Lumid environment is ready for the next implementation
  stage: the TimeDART branch and six required data files match local hashes;
  all 30 container requirements match exactly; Python 3.12.3, CUDA-enabled
  PyTorch 2.12.1, and the 24 GiB RTX PRO 4000 Blackwell pass the focused tests
  and a real batch-16 CUDA forward/backward smoke. GitHub SSH uses the
  persistent identity under `/home/personai-korolev-tes/.ssh/`. No training
  trajectory was launched during this readiness audit.

## Documents

- `paper_reading_note.md` — method, evidence, limitations, and project relevance.
- `official_code_audit.md` — pinned source revision, executable smoke evidence,
  and paper/source discrepancies.
- `architecture_and_dataflow.md` — proposed tensor-level `[64,5]` adaptation.
- `upstream_clarification_request.md` — eleven resolved owner decisions and
  one terminology clarification.
- `phase6_7_integration_proposal.md` — fair two-walk/three-task comparison
  contract.
- `implementation_plan.md` — staged implementation, testing, execution, and
  replay plan and current implementation status.
- `source_manifest.json` — machine-readable paper/source/gate record.

## Primary sources

- Local ICML paper:
  `/mnt/e/zotero/storage/Z9X88Y5S/Wang et al. - 2025 - TimeDART A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation.pdf`
- PMLR paper page: <https://proceedings.mlr.press/v267/wang25as.html>
- arXiv: <https://arxiv.org/abs/2410.05711>
- User-supplied official repository: <https://github.com/ustc-time-series/TimeDART>
- Repository URL printed in the paper/README:
  <https://github.com/Melmaphother/TimeDART>
- Audited commit: `e658a648cf6b04612ca643d10a634b45e136194c`
