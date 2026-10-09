# xLSTM-Mixer official-code audit

**Repository:** <https://github.com/mauricekraus/xlstm-mixer>

**Audited commit:** `730b0531aa9456e498765028f3c22ca3677de42e`

**Commit date:** 2025-06-12

**Branches/tags:** public `main`; no tags

**History at audit:** 13 commits

**Repository licence:** MIT

**Pinned core dependency:** `xlstm==1.0.3`, AGPL-3.0

The checkout was audited read-only under `/tmp`. No upstream code was copied
into the project, no dependency was installed, and no training was launched.

## 1. Repository contents

The repository combines the xLSTM-Mixer model with a broad derivative of the
Time-Series-Library data/model stack and a Lightning CLI. Relevant files are:

- `xlstm_mixer/models/xlstm_mixer.py` — model and ablation modes;
- `xlstm_mixer/layers/StandardNorm.py` — RevIN implementation;
- `xlstm_mixer/exp/exp.py` — Lightning forecasting losses/metrics;
- `xlstm_mixer/cli_helper.py` — optimizer, scheduler, checkpoint defaults;
- `xlstm_mixer/cli.py` — fit/test and tuning orchestration;
- `xlstm_mixer/lit/data.py` and `data_provider/` — public benchmark loaders;
- `scripts/long_term_forecasting/` — 28 dataset/horizon run blocks;
- `.docker/Dockerfile` and `.devcontainer/devcontainer.json` — environment
  recipes;
- `requirements.txt` and `lightning_requirements.txt` — dependencies; and
- `setup.py` — currently broken package metadata.

No unit-test suite, CI workflow, trained checkpoint, prediction artifact, or
paper result table is included. The notebooks concern dataset statistics and
xLSTM exploration rather than an automated reproduction contract.

## 2. Paper-to-code crosswalk

| Paper component | Official implementation | Audit result |
|---|---|---|
| RevIN around complete model | `Normalize(enc_in, affine=False)` | Present, but affine parameters in paper are disabled |
| Shared NLinear `T -> H` | one `self.Linear` applied after variate transpose | Present |
| Shared up-projection `H -> D` | `self.pre_encoding` | Present |
| sLSTM-only stack over variate tokens | `xLSTMBlockStack`, `slstm_at="all"` | Present; backend defaults to custom CUDA |
| One learned initial token | `num_mem_tokens` parameter | Generalized; constructor defaults 0, scripts use 0--4 |
| Original and reverse view | `torch.flip(x, [-1])` in `FULL` | Reverses feature coordinates, not variate tokens |
| Shared two-view projection `2D -> H` | `self.fc` | Present |
| Inverse RevIN | final `Normalize(..., "denorm")` | Present |
| MAE training; MAE/MSE reporting | Lightning `L1Loss`; torchmetrics | Present |
| Three seeds | shell scripts use 2021/2022/2023 | Present |
| Classification outlook | generic trainers exist | xLSTM-Mixer path itself is not implemented |
| GIFT-Eval quantile evaluation | none found | Absent |

## 3. Critical implementation findings

### 3.1 The audited source does not correspond to a tagged final-paper release

The model and scripts entered the repository in commit `9200c4b8` on
2024-10-24. `git blame` shows that the core forward path, including reversal,
is unchanged since that commit. The only later source-level repository commit
adds the MIT licence in June 2025. There are no tags or GitHub release markers.

The paper read is arXiv v4 dated 20 November 2025 and published at NeurIPS
2025. The repository citation still identifies a 2024 arXiv article, and its
classification and GIFT-Eval paper extensions are not represented in the
code. Exact final-paper reproduction cannot be claimed from this checkout.

### 3.2 Released view reversal is not reversed variate order

After up-projection, normal `FULL` tensors have shape `[B,M+V,D]`. The code
sets:

```python
dim = 1 if ablation_mode == FULL_TIME else -1
x_reversed = torch.flip(x, [dim])
```

Thus `FULL` reverses `D`, the latent-feature coordinate axis. Both sLSTM calls
still traverse memory/variate tokens in the same order. Concatenation also
occurs on the last feature axis, and the reverse output is not flipped back.

This is compatible with the paper's literal phrase “order of latent
dimensions ... is inverted,” but not with its repeated interpretation as an
ensemble over different variate orderings. Appendix H's lower-triangular
attribution is also consistent with a single variate direction. The current
Phase 6.6 prose saying “reversed variate-order views” is therefore not a
faithful statement of the released code.

### 3.3 Paper/source RevIN mismatch

Paper Section 3.1 includes learned `gamma` and `beta`. The source creates:

```python
self.reversible_instance_norm = RevIN(enc_in, affine=False)
```

Statistics are detached, variance uses `unbiased=False`, and epsilon is
`1e-5`. Project reproduction must choose paper-affine or source-non-affine
behavior explicitly.

### 3.4 Initial-token count is tuned rather than fixed

The method text specifies one token `eta`. `xLSTMMixer` defaults
`num_mem_tokens=0`; supplied scripts use values 0, 1, 2, 3, and 4 depending on
dataset and horizon. This makes “one initial token” a paper-aligned choice but
not a universal released-script choice.

### 3.5 No source configuration covers the project regime

The official scripts cover `H in {96,192,336,720}` and lookbacks
`{336,512,768}`. Across scripts they use:

- width `{64,128,256,768,1024}`;
- one to four blocks and four to 32 heads;
- kernel `{0,2,4}` and dropout `{0.1,0.25}`;
- batch `{16,32,64,128,256}`; and
- 40, 50, or 60 epochs.

There is no source-selected setting for `T=64,H=8,V=5`. Any Phase 6.9 recipe
is necessarily a predeclared adaptation rather than an exact run-script reuse.

### 3.6 Package installation is broken

Running `python3 setup.py --name` fails immediately because
`xlstm_mixer/VERSION` is absent. Even if that were repaired:

- `read_requirements` expects a path string but receives a Python list; and
- `setup.py` names `lightning-requirements.txt`, while the repository file is
  `lightning_requirements.txt`.

The README separately misspells the file as `lighting_requirements.txt`.
Practical use requires running from the checkout after manual dependency
installation or writing a project-native package adapter.

### 3.7 Environment declarations disagree and are weakly pinned

The README says Python 3.11, PyTorch 2.4, Ubuntu 22.04, and CUDA 12.1. The
Dockerfile installs PyTorch 2.3.1. The paper's extracted implementation text
does not preserve a complete PyTorch patch version. All dependencies except
`xlstm==1.0.3` are unpinned.

The Dockerfile further:

- uses an external base image without a digest;
- ends Lightning dependency installation with `|| echo`, masking failure;
- sets `PYTHONPATH` to `src`, although this repository has no `src` package
  root; and
- relies on a bind-mounted checkout at runtime.

The Docker/devcontainer files are useful references for development and Lumid
workflow images but are not the Phase 6.9 experiment runtime. The experiment
will run in a persistent Lumid Sandbox with its own frozen admission manifest.

### 3.8 The pinned xLSTM core has a separate copyleft licence

`requirements.txt` pins `xlstm==1.0.3`. The corresponding official tag points
to commit `1ff240242795062e56b4e39b43023cce61e8e88c`, whose repository licence
is AGPL-3.0. Its sLSTM configuration defaults to `backend="cuda"`, with a
`vanilla` implementation also available. The README requires NVIDIA compute
capability at least 8.0 for the CUDA implementation.

MIT licensing of the xLSTM-Mixer wrapper does not replace the dependency's
AGPL terms. Before public distribution or service use, the project needs an
explicit dependency/licence decision and proper notices. This audit is not
legal advice.

### 3.9 Source training selection conflicts with the project contract

The Lightning CLI logs validation metrics, saves the best validation-MSE
checkpoints, and `fit_and_test` tests
`trainer.checkpoint_callback.best_model_path`. The paper also performs
Optuna tuning. Phase 6.9 permits neither validation allocation nor
evaluation-driven choice. Reusing the architecture with a fixed epoch-50
checkpoint is a disclosed protocol adaptation.

### 3.10 Source preprocessing uses two normalization layers

Dataset loaders fit `StandardScaler` on the training partition. The model then
applies per-instance RevIN and reverses it before loss. Therefore the source
L1 loss is computed after instance denormalization but still in globally
standardized channel units.

The Phase 6.9 owner decision does not add a second channel scaler: OHLC remains
in `[0,1]` and the existing walk-training-only volume transform is reused for
contexts and future targets. This is a disclosed project-protocol adaptation
rather than an exact copy of the source preprocessing stack.

### 3.11 Time features are ignored

Data loaders create calendar markers and pass them to the model, but
`xLSTMMixer.forecast` never uses `x_mark_enc`. The Phase 6.9 adapter does not
need hour/day embeddings to preserve released model behavior.

### 3.12 The released model has no output-domain constraints

The final layer is linear. The model does not impose `[0,1]` probability
bounds, `high >= max(open,close)`, `low <= min(open,close)`, or nonnegative
volume. Adding projection or clipping would be a project modification and may
not be selected after evaluation. Preserve unconstrained full paths and report
any invalid-output diagnostics separately.

### 3.13 Classification and probabilistic code in the final paper is absent

The `BaseModel` classification branch raises `NotImplementedError` for
`xLSTMMixer`, and no xLSTM-Mixer classification scripts exist. Searches found
no GIFT-Eval, CRPS, or quantile-head implementation. Generic classification
files inherited for other models do not reproduce Appendix J.

## 4. Static environment audit

The current project WSL virtual environment reported:

```text
Python 3.10.12
PyTorch 2.12.1+cu126
xlstm: absent
einops: absent
CUDA available to this audit process: false
```

This is neither a failure nor an admission result because the intended
experiment platform is the persistent sandbox. It establishes only that the
existing local environment cannot run the official model without dependency
changes. No packages were installed during the audit.

## 5. Static feasibility verdict

**Decision:** method suitable; implementation and training not yet admitted.

The architecture is technically compatible with `[B,64,5] -> [B,8,5]`:

- both linear mixers accept arbitrary configured `T` and `H`;
- the sLSTM sequence has only five variate tokens plus the approved single
  initial token;
- the project has enough historical and target metadata to audit a continuous
  eight-bar path; and
- the existing price evaluator can extract the eighth close without a learned
  adapter.

Admission still requires:

1. the now-complete owner decision record;
2. a documented AGPL dependency/reuse choice;
3. a persistent-runtime build of `xlstm==1.0.3` or an approved independent
   equivalent;
4. CPU shape/backward tests using the vanilla path;
5. a real fixed-batch CUDA forward/backward smoke with memory/time evidence;
6. same-backend CUDA checkpoint/fixed-probe replay; and
7. the Phase 6.9 common-intersection target artifact.

Failure of the CUDA or licence gate is a documented rejection of this
candidate. It is not permission to substitute ordinary LSTM while continuing
to call the result xLSTM-Mixer.

## 6. Recommended reuse boundary

Do not import the repository's full Lightning/Time-Series-Library stack. Build
a minimal project-native adapter with tests and manifests, using the MIT
xLSTM-Mixer source as an attributed behavioral reference. Resolve separately
the adapter around the approved AGPL `xlstm==1.0.3` package, preserve source
and licence notices, record exact hashes, and label every project deviation.
