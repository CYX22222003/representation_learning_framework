# xLSTM-Mixer baseline dossier

**Project method ID:** `XM-MV8`

**Phase:** Phase 6.9 technical, implementation, artifact, and reporting
authority

**Dossier status:** paper and official-source audit complete; decisions 1--4
and 6--14 resolved; initial-token decision plus CUDA admission remain pending

**Implementation status:** not started

**Experiment status:** not started

This directory records the evidence and proposed project adaptation for:

> Maurice Kraus, Felix Divo, Devendra Singh Dhami, and Kristian Kersting,
> “xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar
> Memories,” NeurIPS 2025.

## Bottom line

- xLSTM-Mixer is a suitable recent task-specific baseline for the eight-hour
  future-price task. It is a supervised multivariate forecaster, not a frozen
  representation baseline, so it answers a different question from Phase 6.7.
- The paper predicts a complete multivariate future path. The fair project
  adaptation therefore consumes the same `[B,64,5]` OHLCV contexts, predicts
  `[B,8,5]`, and evaluates only the extracted `close[t+8]` against the existing
  price endpoint. Its additional four channels and seven intermediate horizons
  must remain disclosed.
- The method performs time mixing with a shared NLinear forecast, embeds each
  variate's preliminary eight-step path, recurrently mixes the variate tokens
  with sLSTM blocks, combines two views, and reverses RevIN.
- The primary adapter follows the released `FULL` path:
  `torch.flip(x, [-1])` reverses latent feature coordinates, not the
  variate-token axis. The paper's different variate-order interpretation is
  disclosed but is not implemented as a second sensitivity.
- The adapter follows source `RevIN(..., affine=False)`. The paper specifies
  one initial token while supplied scripts tune zero to four; the token count
  is the only owner choice still awaiting confirmation.
- The official repository is MIT-licensed, but its pinned `xlstm==1.0.3`
  dependency is AGPL-3.0. The project should not copy the entire training
  repository. A minimal project-native adapter plus an explicit dependency and
  licence decision is recommended.
- The source is not installable through its included `setup.py`: the referenced
  `xlstm_mixer/VERSION` file is absent, and the requirements call contains
  additional filename/type defects. Running from a checkout after separately
  installing requirements is the documented practical path.
- The audited main model code was added in October 2024 and has not changed;
  there is no release tag tying it to the November 2025 arXiv v4/NeurIPS paper.
  The later classification and GIFT-Eval additions in the paper are absent
  from the released xLSTM-Mixer path.
- The source selects a best validation checkpoint. Phase 6.9 forbids a
  validation split and checkpoint selection, so the project adaptation must
  retain fixed epoch 50 and snapshots 5/15/50.
- The experiment will run in the persistent Lumid Sandbox. The Docker and
  devcontainer environments are references for development/workflow use, not
  the experiment runtime. No package was installed and no training was
  launched during the local static audit; a sandbox CUDA/build smoke remains
  mandatory.
- All active implementation and canonical artifacts belong to Phase 6.9 under
  `experiments/phase6_9/xlstm_mixer/`. Phase 6.6 is a superseded historical
  reference for this baseline.
- The existing absolute-price rows do not all have observed intermediate
  paths. The frozen Phase 6.9 intersection therefore requires matched H0-D0
  and Raw-LSTM reruns; the decision record contains the observed counts.

## Documents

- `paper_reading_note.md` — method, empirical evidence, limitations, and
  project relevance.
- `official_code_audit.md` — pinned revision, implementation behavior,
  packaging/runtime findings, and paper/source discrepancies.
- `architecture_and_dataflow.md` — exact source tensors and proposed
  `[64,5] -> [8,5]` project mapping.
- `upstream_clarification_request.md` — resolved decision record and the one
  remaining initial-token confirmation required before implementation.
- `phase6_9_integration_proposal.md` — fair future-price comparison contract
  and sole Phase 6.9 artifact boundary.
- `implementation_plan.md` — staged implementation, resource admission,
  execution, replay, and reporting plan.
- `source_manifest.json` — machine-readable source, hash, licence, environment,
  and gate record.

## Primary sources

- Local paper:
  `/mnt/e/zotero/storage/GYKJQH4I/Kraus et al. - 2025 - xLSTM-Mixer Multivariate Time Series Forecasting by Mixing via Scalar Memories.pdf`
- NeurIPS proceedings:
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/09e38101e74a89129ccd0d0756ed36b3-Abstract-Conference.html>
- arXiv: <https://arxiv.org/abs/2410.16928>
- Official repository: <https://github.com/mauricekraus/xlstm-mixer>
- Audited repository commit:
  `730b0531aa9456e498765028f3c22ca3677de42e`
- Pinned xLSTM dependency: <https://github.com/NX-AI/xlstm/tree/v1.0.3>
