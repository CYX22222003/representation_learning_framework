# xLSTM-Mixer baseline dossier

**Active project method ID:** `XM-C8` (direct eight-hour close endpoint).

**Owner-approved contract correction (2026-10-09):** Read
[`../../phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md`](../../phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md).
XM-C8 takes `[B,64,5]`, predicts only `[B,1] = close[t+8h]`, and uses
endpoint-only MSE on **all original price rows**: 36,773/29,834 and
56,652/13,506 train/test rows. No future-path join, extra labels, or
intersection-control reruns. Exact-source/recipe/prediction audits confirm
the original H0-D0 and Raw LSTM controls share those bundles.

The direct model and resumable guarded lifecycle are implemented in
`src/baselines/xlstm_mixer/endpoint.py` and
`src/training/phase6_9_xlstm_endpoint.py`; 18 new endpoint CPU tests pass.
Its RevIN inverse remains unclipped (a disclosed difference from the sigmoid
controls). This is an endpoint adaptation, not full-path paper reproduction.
Fresh endpoint admission passed on the local RTX 4060 Laptop GPU: batch 512
and one-row backward/update/reload probes, 289,603,584 bytes peak allocation.
The manifest-only pipeline exits successfully and launches no trajectories.
Both fresh endpoint walks complete 50 epochs and all six 5/15/50 snapshots
pass independent replay. Epoch-50 MAE/RMSE are 0.003100233/0.011040398 and
0.004759289/0.020759157. The matched report under
`experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/` shows lower errors
than H0/Raw LSTM in both walks, but persistence wins MAE in both walks and
RMSE in Walk 2. Endpoint training is separate from completed XM-MV8 runs.
The standalone reporter corrects H0's nested movement-metric schema without
changing the admitted training fingerprint or checkpoints. The updated shell
launcher independently validates/reports completed runs and exits zero.
The focused suite passes 45 tests; one opt-in historical GPU fixture is skipped.

```bash
# CPU smoke, exact original-data/control audit, and immutable matrix; no GPU/training.
.venv-xlstm-mixer/bin/python3 scripts_v8/run_phase6_9_xlstm_endpoint.py
# Active endpoint runner: CUDA admission and matrix only by default.
bash scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh
# Fresh/resumed XM-C8 training; never resumes old XM-MV8 weights.
bash scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh --execute
# Standalone endpoint replay and comparison after completed training.
.venv-xlstm-mixer/bin/python3 scripts_v8/report_phase6_9_xlstm_endpoint.py \
  --device cuda --backend vanilla
```

New artifacts belong under `experiments/phase6_9/xlstm_mixer_endpoint/`.
The sections below preserve the **historical XM-MV8 full-path dossier**.
The endpoint amendment supersedes their project target/row/loss/launch
requirements; paper/source observations and historical execution evidence
are not rewritten. The old individual `prepare/audit/bootstrap/validate/report`
entry points still address only legacy XM-MV8. The shared shell runner now
addresses only XM-C8.

**Phase:** Phase 6.9 technical, implementation, artifact, and reporting
authority

**Dossier status:** paper and official-source audit complete; all fourteen
owner decisions resolved; model, guarded training lifecycle, runtime audit,
and replay validator implemented; local data and runtime readiness are checked

**Implementation status:** Stages 1--4/6 and local orchestration are
implemented under `src/baselines/xlstm_mixer/`, `src/data_processing/`,
`src/training/phase6_9_xlstm_mixer.py`, and `scripts_v8/`; 26 focused tests
pass, including real vanilla-GPU fixture interrupt/resume and replay.
Both real-data bundles pass source replay. Both seed-0 trajectories complete
50 epochs; all six 5/15/50 snapshots pass standalone same-backend replay.
Matched controls and final comparative reporting remain unexecuted.

**Experiment status:** XM-MV8 training/replay complete; matched comparison open

Results are recorded under `experiments/phase6_9/xlstm_mixer/reports/seed0/`
in `summary.md`, `summary.json`, and `execution_record.md`. At the predeclared
epoch 50, future-close MAE/RMSE are 0.003054837/0.010852719 (Walk 1) and
0.004780288/0.021858833 (Walk 2), with implied-movement Spearman
0.352519/0.257461. Persistence has lower MAE in both walks; XM-MV8 improves
RMSE only in Walk 1. Extra full-path supervision is disclosed, and invalid
OHLC ordering/negative raw-volume forecasts are retained without repair.

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
- The adapter follows source `RevIN(..., affine=False)` and uses exactly one
  learned initial token, as specified by the paper. Supplied scripts tune zero
  to four, but the project does not run that search.
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
- The owner-directed 2026-10-09 amendment selects local WSL vanilla sLSTM on
  CUDA tensors. Lumid, Docker, and a compiled CUDA extension are not required.
  Data/source replay and selected-backend GPU resource admission remain gates.
- The optional `xlstm-compat` target in `docker/Dockerfile` now adds the
  pinned dependency and CUDA build tools. A separate synthetic-input script
  checks CPU or CUDA compatibility before data preparation; its GPU checks
  include batch 512, a one-row remainder, backward/update, and checkpoint
  reload. The first GPU check defaults to the vanilla backend on CUDA tensors;
  `--backend cuda` explicitly selects the later compiled-kernel check. The
  expected Python 3.12.3/PyTorch `2.12.1+cu132`/CUDA 13.2 runtime and owner-
  supplied RTX PRO 4000 GPU inventory are recorded in `source_manifest.json`.
  Instructions are in `docker/README.xlstm-mixer.md`. Docker build
  and GPU compatibility remain unverified because Docker Desktop's WSL
  integration is unavailable in the current session.
- All active implementation and canonical artifacts belong to Phase 6.9 under
  `experiments/phase6_9/xlstm_mixer/`. Phase 6.6 is a superseded historical
  reference for this baseline.
- The independently authored minimal adapter now implements the frozen
  non-affine RevIN, shared NLinear/up/down projections, exactly one learned
  token, pinned `xlstm==1.0.3` stack boundary, released feature-axis reverse
  view, full-path L1 helper, endpoint extraction, and checkpoint provenance
  checks. The canonical `[2,64,5] -> [2,8,5]` path also passes a bounded
  vanilla-backend forward probe against the pinned dependency checkout.
- The training bootstrap is manifest-only unless `--execute` is supplied and
  refuses execution without matching data and CUDA-admission hashes. Training
  is per-walk, epoch-resumable, retains 5/15/50 atomically, evaluates only
  after the 50-epoch trajectory finishes, and supports same-backend standalone
  checkpoint/prediction/identity/metric replay. No run has been launched.
- The existing absolute-price rows do not all have observed intermediate
  paths. The frozen Phase 6.9 intersection therefore requires matched H0-D0
  and Raw-LSTM reruns; the decision record contains the observed counts.

## Dedicated local environment

At the owner's request, local WSL execution uses `.venv-xlstm-mixer/`,
separate from the existing project `.venv/` (no shared site-packages). The
reproducible installer is:

```bash
bash scripts_v8/install_xlstm_mixer_local.sh
```

This pins PyTorch `2.12.1+cu126`, `xlstm==1.0.3`, and the support packages in
`scripts_v8/requirements-xlstm-local.txt`. The installer verifies the original
compiler-loader SHA256 before applying the documented
`scripts_v8/xlstm-1.0.3-lazy-cuda-init.patch`. The patch removes import-time
CUDA development-toolkit discovery: vanilla sLSTM can run on CUDA tensors
without `nvcc`. It does not change model computation, parameters, loss, or
data. The modified installed dependency retains its AGPL-3.0 licence; see
`src/baselines/xlstm_mixer/THIRD_PARTY_NOTICES.md`.

Use the dedicated interpreter explicitly, or activate it:

```bash
source .venv-xlstm-mixer/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-xlstm-mixer/bin/python3 \
  scripts_v8/check_phase6_9_xlstm_mixer_environment.py \
  --device cuda --backend vanilla --batch-size 512 \
  --output .venv-xlstm-mixer/environment_check.json
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-xlstm-mixer/bin/python3 -m pytest tests/baselines/xlstm_mixer -q
```

Verified on 2026-10-09 with Python 3.10.12 and the local RTX 4060 Laptop GPU:
all 19 focused tests pass; the installed dependency imports normally; CPU and
GPU forward/backward, optimizer update, and checkpoint replay pass. GPU
batches 512 and 1 pass with 289,852,928 bytes (about 276 MiB) peak allocated
memory. The result and test report are saved locally as
`.venv-xlstm-mixer/environment_check.json` and `model_tests.xml`; installed
package versions are captured in `scripts_v8/requirements-xlstm-local-lock.txt`.
The patched loader SHA256 is
`c62be35147ba02a8da1b391e70f215bc4038082b369b31e98171f274225c7555`.

The canonical trainer now accepts the admitted vanilla backend on a CUDA
device (default), or an explicitly admitted compiled `cuda` backend.
Installing this environment alone does not launch or admit a full experiment.
The earlier sandbox expectations in the compatibility report are informational
and need not match this local GPU. Docker and Lumid are not needed for these
commands.

## Historical XM-MV8 training launcher and execution evidence

```bash
# Build/replay data, test GPU admission, and freeze the matrix; no trajectories.
# The shared shell runner now uses XM-C8; historical stage commands are below.
.venv-xlstm-mixer/bin/python3 scripts_v8/bootstrap_phase6_9_xlstm_mixer.py

# Explicitly train/resume both independent 50-epoch walk trajectories.
# Historical individual bootstrap remains XM-MV8-only, not the active comparison.
.venv-xlstm-mixer/bin/python3 scripts_v8/bootstrap_phase6_9_xlstm_mixer.py \
  --device cuda --execute
```

The runner uses `.venv-xlstm-mixer/bin/python3`, `--device cuda`, and
`--backend vanilla`; `--help` lists interpreter, device, log, and resource
ceiling options. Defaults are 6 GiB and 600 seconds per bounded admission probe.
It writes persistent timestamped logs under
`experiments/phase6_9/xlstm_mixer/logs/`, stops on gate failures, and reuses
unchanged admission/data/matrix artifacts. Checkpoints include model,
optimizer, CPU/CUDA RNG, sampler state, identities, and source/code provenance.
Interrupts resume at the last complete epoch; partially processed epochs are
repeated with restored RNG. Completion is written only after prediction replay.
The launcher holds a single-process lock to prevent overlapping pipelines.

Local real-data admission and the full manifest-only pipeline passed on
2026-10-09. Both walks passed batch-512 and one-row backward/update/checkpoint
replay; final admission maximum allocation was 295,512,576 bytes (about 282 MiB), below the
6 GiB ceiling. Admission is saved in `feasibility/cuda_admission.json`; the
two-run freeze is in `manifests/training_seed0.json`. No real-data model
trajectory was launched by this readiness check.

The subsequent explicit `--execute` launch on 2026-10-09 completed both real
trajectories. Standalone replay validates all snapshots. Peak training CUDA
allocation was 298,847,744/298,917,376 bytes for Walk 1/2 (about 285 MiB);
summed epoch training times were 79.064/120.017 seconds, excluding preflight,
serialization, evaluation, and replay. These are distinct from admission probes.

The data builder saves `[64,5]` contexts, observed `[8,5]` targets, accepted
train-only volume normalization, imputation/identity metadata, and original
h8 price-row indices. The counts are 32,470/27,786 for Walk 1 and
53,112/12,115 for Walk 2 (train/evaluation). No new scaler is fitted.
Original float64 current/target endpoint labels are retained for exact
comparator metrics; model contexts, full-path targets, and optimization stay
float32. Endpoint validation compares their float32 tensor representations.
The runner preserves snapshots 5/15/50 and reports all three, with epoch 50
predeclared as primary. Evaluation runs after training, never for selection.

The launcher executes **XM-MV8 only**, not the required matched H0-D0/Raw-LSTM
reruns. Its `report_phase6_9_xlstm_mixer.py` output is a training diagnostic,
not a completed fair comparison; the extra full-path supervision is disclosed.

## Dossier documents

- `paper_reading_note.md` — method, empirical evidence, limitations, and
  project relevance.
- `official_code_audit.md` — pinned revision, implementation behavior,
  packaging/runtime findings, and paper/source discrepancies.
- `architecture_and_dataflow.md` — exact source tensors and proposed
  `[64,5] -> [8,5]` project mapping.
- `upstream_clarification_request.md` — complete fourteen-item owner decision
  record.
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
