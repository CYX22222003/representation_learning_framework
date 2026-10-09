# SGN baseline dossier

**Project label:** SGN-C (independently authored supervised adaptation)
**Artifact method ID:** `sgn_c`
**Phase:** 6.9 task-specific classification comparison
**Dossier status:** paper reading and pinned static source/settings/licence audit
complete; all seven owner decisions approved
**Implementation status:** complete for model, initialization, resumable
training/replay lifecycle, scripts, and 17 focused CPU tests
**Experiment status:** complete for the frozen two-walk seed-0 scope; both
50-epoch trajectories and all six snapshots pass same-backend replay

This directory records the evidence and proposed project adaptation for:

> Zenan Ying, Jinke Wang, Zhi Zheng, Tong Xu, Wei Chen, Qi Liu, and Huijun
> Hou, “SGN: Shifted Window-Based Hierarchical Variable Grouping for
> Multivariate Time Series Classification,” NeurIPS 2025.

## Bottom line

- SGN is a supervised multivariate time-series classifier. It is suitable as
  Phase 6.9's classification-specific complete-system comparator, not as a
  target-free frozen-representation baseline.
- Its defining mechanisms are train-initialized variable grouping,
  group-wise temporal embedding, multi-scale depthwise convolution,
  intra-/inter-group pointwise mixing, alternating shifted windows,
  hierarchical period merging, global pooling, and a native linear head.
- The paper reports strong classification evidence, including the best mean
  results on four main datasets and the best average rank on its 25-dataset
  UEA table. That evidence does not establish performance on causal future
  movement prediction or five-channel OHLCV.
- The official repository is pinned at commit
  `c6d1b573dcb8c4255cde59b988f74334ea5da503`. It has no repository-level
  software licence, so no upstream implementation is to be copied, modified,
  or vendored. The proposed implementation is independent and paper-guided;
  the source is an attributed behavioral reference only.
- The released repository is not runnable as published without repair: its
  launcher omits required `num_groups` and `kernel_size` arguments, imports a
  package layout absent from the checkout, the model always loads the
  TDBRAIN grouping file, and the source uses a configured period instead of
  the paper's FFT selection rule.
- A read-only diagnostic used only the saved classification training rows.
  Under a source-like BDC estimator, both walks separate OHLC from volume for
  `G=2`. Under the paper's FFT idea, both walks have the same leading
  non-full-length period candidates: `32, 21, 16, 13, 11` hours.
- The approved project contract is `G=2`, `D=64`, fixed training-derived
  `P=16`, four stages with depths `[2,2,2,1]`, seven odd kernels (`1..13`),
  ratio `2`, dropout `0.1`, and a native `128 -> 3` head.
- SGN-C must use every original h2/`tau=0.001` row: Walk 1
  `37,864/30,340` train/evaluation and Walk 2 `57,521/13,887`. It keeps the
  existing train-prior logit-adjusted task loss, adds a separately disclosed
  grouping regularizer, trains separate seed-0 weights per walk for 50 fixed
  epochs, retains 5/15/50, and keeps epoch 50 primary.
- The independently authored implementation has 275,853 trainable parameters.
  Both 2,000-row train-only initializers replay exactly and recover
  `[OHLC]`/`[volume]`; both retain FFT periods `32,21,16,13,11`. The selected
  RTX 4060/PyTorch runtime admits physical batch 256 at 778,075,136 bytes peak
  allocation with a one-row remainder and same-backend checkpoint replay.
- The principal epoch-50 macro-F1 is `0.3989`/`0.3137` in Walks 1/2, below
  H0-D0 (`0.4535`/`0.4532`), Raw LSTM (`0.4252`/`0.4463`), and Raw MLP
  (`0.4448`/`0.4325`). Hard assignments collapse all five variables into one
  group at both primary checkpoints.

## Documents

- `paper_reading_note.md` — method, evidence, critique, and project relevance.
- `official_code_audit.md` — pinned source, licence boundary, and
  paper/source discrepancies.
- `architecture_and_dataflow.md` — proposed tensor-level `[64,5]` adaptation.
- `upstream_clarification_request.md` — complete seven-decision owner record.
- `phase6_9_integration_proposal.md` — fair two-walk classification contract.
- `implementation_plan.md` — staged implementation, testing, admission,
  execution, replay, and reporting plan.
- `source_manifest.json` — machine-readable source and planning record.
- `docs/phase_plan/2026-10-10-phase-6-9-sgn-execution.md` — completed
  execution evidence and result judgement.

## Implementation

- `src/baselines/sgn/` — frozen configuration, BDC/K-means initializer,
  learned assignments, group embedding, MGWM/PWSM hierarchy, and native head.
- `src/training/phase6_9_sgn.py` — initialization replay, admission, exact
  resume, two-walk training, snapshot evaluation, and CPU prediction replay.
- `scripts_v8/*phase6_9_sgn*` — prepare, validate, bootstrap, report, and
  one-run shell entry points.
- `tests/baselines/sgn/` — 17 focused unit/lifecycle tests.

## Primary sources

- Owner-supplied local paper:
  `/mnt/e/zotero/storage/STZF8JRL/Ying et al. - 2025 - SGN Shifted Window-Based Hierarchical Variable Grouping for Multivariate Time Series Classification.pdf`
- [NeurIPS 2025 proceedings page](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9b9aa183c3ab49380e0f306e9f130acf-Abstract-Conference.html)
- [Published paper](https://papers.nips.cc/paper_files/paper/2025/file/9b9aa183c3ab49380e0f306e9f130acf-Paper-Conference.pdf)
- [Official reference repository](https://github.com/colison/SGN)
- DOI: `10.52202/085713-3609`

## Current gate

The classification leg and final Phase 6.9 task-separated synthesis are
complete. No retuning or earlier-checkpoint selection is authorized from the
SGN result.
