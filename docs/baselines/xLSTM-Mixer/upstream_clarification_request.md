# xLSTM-Mixer clarification and decision record

**Status:** all fourteen owner decisions resolved on 2026-10-09

**Purpose:** freeze paper/source/project ambiguities before implementation or
training

**Downstream metrics consulted:** none

These are internal adaptation decisions. They do not ask the paper authors to
change the method. Once resolved, the answers must be copied into the source
manifest, Phase 6.9 architecture text, implementation configuration, and
experiment manifests before any evaluation result is read.

## Recommended decision set

### 1. Reference identity

**Finding:** the audited paper is NeurIPS 2025/arXiv v4, while the untagged
repository's model code dates to October 2024 and omits final-paper additions.

**Recommendation:** cite the NeurIPS 2025 paper as the scientific source and
pin official commit `730b0531...` as the behavioral source. Label the project
model “xLSTM-Mixer (`XM-MV8`; NeurIPS 2025 architecture adapted from the
pinned official source).” Do not call it an exact reproduction.

**Owner decision (2026-10-09):** approved, without claiming strict
line-by-line reproduction. Cite the paper and pin the official source as the
behavioral reference for the implemented path.

### 2. Reuse and licence boundary

**Finding:** the wrapper repository is MIT, but `xlstm==1.0.3` is AGPL-3.0.
The full repository also contains large unrelated Time-Series-Library code and
a broken package installer.

**Recommendation:** author a minimal project-native adapter and tests. Use the
official wrapper only as an attributed behavioral reference. Initially import
the exact `xlstm==1.0.3` dependency, preserve its licence/notices, and keep the
project repository's distribution compliant; if that boundary is unacceptable,
pause and separately scope a clean sLSTM implementation rather than silently
changing the dependency. This is a project decision, not legal advice.

**Owner decision (2026-10-09):** approved. Use a minimal project-native
adapter around the pinned dependency and preserve the applicable notices.

### 3. Reverse-view semantics

**Finding:** the paper both mentions reversed latent dimensions and interprets
views as different variate orderings. Released `FULL` code flips `dim=-1`, the
latent feature axis, and retains the same variate-token order. The current
Phase 6.6 diagram says reversed variate order.

**Recommendation:** for the source-aligned `XM-MV8` baseline, preserve the
released feature-axis reversal exactly. Correct the Phase 6.6 diagram after
approval and disclose the paper's inconsistent interpretation. A true
variate-order reversal would be a separately named paper-guided sensitivity,
not the primary run.

**Owner decision (2026-10-09):** approved. The primary run follows the
released feature-axis reversal; the paper/source interpretation discrepancy
is disclosed and is not turned into a second sensitivity.

### 4. RevIN affine behavior

**Finding:** the paper equation uses learned affine parameters; source sets
`affine=False`.

**Recommendation:** preserve source `affine=False`, detached per-instance
statistics, `unbiased=False`, and epsilon `1e-5`. This is unambiguous released
behavior and avoids adding parameters absent from the official forward path.

**Owner decision (2026-10-09):** approved. Follow the released non-affine
RevIN behavior.

### 5. Number of initial tokens

**Finding:** the paper says one `eta`; source default is zero and run scripts
tune zero through four.

**Recommendation:** use exactly one learned initial token. It follows the
scientific method text and avoids validation-based selection among script
settings. Initialize it with source `Normal(0,0.01)`.

**Owner decision (2026-10-09):** approved. Use exactly one learned initial
token, initialized with the source `Normal(0,0.01)` rule.

### 6. Architecture for `T=64,H=8,V=5`

**Finding:** no source script covers this short project regime. Hyperparameters
were tuned independently for every public dataset and long horizon.

**Recommendation:** predeclare the compact full model for both walks:

```text
D=128
one sLSTM block
eight heads
conv1d kernel disabled
dropout=0.1
packing=1
num_tokens_per_variate=1
NLinear backbone
two-view/backcast enabled
```

This uses common low-variate ETT script values without a task-result sweep.
Do not alter the configuration between walks.

**Owner decision (2026-10-09):** approved. Adapt this compact full model to
the project's comparison pipeline without changing its xLSTM-Mixer core.

### 7. Channel order and target path

**Finding:** recurrence over variates makes ordering meaningful. The project
stores `[open, high, low, close, volume]`, and the headline endpoint is the
eighth future close.

**Recommendation:** retain exactly `[open, high, low, close, volume]` for both
input and output. Predict all observed bars `t+1,...,t+8`; extract output
`[:,7,3]`. Do not reorder channels, use separate contracts, permit imputed
targets, or set `H=1` and relabel it `t+8`.

**Owner decision (2026-10-09):** approved.

### 8. Global scaler and loss domain

**Finding:** source loaders standardize channels on training data before the
model's RevIN. Raw project volume would otherwise dominate full-path L1.

**Recommendation:** do not introduce a second xLSTM-specific channel scaler.
Consume the project's accepted units directly: OHLC probabilities remain in
`[0,1]`, and volume uses the already fitted walk-training-only Phase 5 volume
transform for both contexts and future targets. Train with unweighted mean L1
over all 40 outputs in those accepted units. The existing volume-scaler
identity and hash remain part of the data manifest.

**Owner decision (2026-10-09):** approved. The owner considers the existing
price bounds and fixed upstream volume transform sufficient.

### 9. Training schedule and batch semantics

**Finding:** official runs use 40--60 epochs and dataset-specific schedules.
The project already precommits epoch 50 and snapshots 5/15/50.

**Recommendation:** use the project's fixed comparison lifecycle: float32,
seed 0, Adam `lr=1e-4`, no weight decay, physical batch 512 with
`drop_last=False`, 50 epochs, and retained epochs 5/15/50. Preserve the
method-specific full-path L1 loss and gradient clipping at norm 1.0. A smaller
physical batch is permitted only if the pre-training Lumid CUDA smoke shows
that 512 is infeasible; it must then be frozen before training and shared by
both walks. Do not tune the schedule from evaluation results.

**Owner decision (2026-10-09):** approved. Project comparison settings take
precedence over reproducing the paper's dataset-specific search schedule.

### 10. Checkpoint selection

**Finding:** official code tests a best validation-MSE checkpoint. The project
has no validation split and cannot tune from evaluation results.

**Recommendation:** retain complete epochs 5/15/50, use epoch 50 as the fixed
primary result, and never restore a best-loss epoch. Save model, optimizer,
RNG/sampler, configuration, data/source hashes, and fixed
replay probes atomically.

**Owner decision (2026-10-09):** approved. Epoch 50 is primary and 5/15/50
are reported without best-checkpoint selection.

### 11. CUDA and CPU replay

**Finding:** the pinned xLSTM package defaults to its custom CUDA sLSTM but
also supplies a vanilla backend. The source wrapper does not expose backend
selection. The current project environment lacks the dependency, and CUDA was
not available to the static audit.

**Recommendation:** expose backend only as an execution setting, not a model
hyperparameter. Train with the pinned CUDA backend on the persistent Lumid
Sandbox after a fixed-batch forward/backward smoke. Do not require numerical
equivalence between the CUDA and vanilla CPU backends. Retain only lightweight
same-runtime integrity checks: reload each retained checkpoint on the same
CUDA backend, replay one frozen probe, and recompute saved predictions and
metrics by identity.

**Owner decision (2026-10-09):** approved in this reduced form. There is no
cross-backend replay gate, but checkpoint and artifact replay are retained.

### 12. Full-path row availability

**Finding:** source forecasting assumes a regular complete grid; the project
price endpoint alone does not guarantee observed OHLCV at every intermediate
bar. A read-only identity check on 2026-10-09 confirmed this: the strict
fully observed intersections are 32,470/36,773 training and 27,786/29,834
evaluation rows in Walk 1, and 53,112/56,652 training and 12,115/13,506
evaluation rows in Walk 2.

**Recommendation:** reuse the existing strict Phase 6 observed-path identity
logic to freeze the Phase 6.9 common intersection while constructing the
eight-by-five targets. This is a deterministic metadata join and assertion,
not a new horizon-selection study. Rerun H0-D0 and Raw LSTM on exactly the
intersected training and evaluation rows. Record row/contract loss. No
model-specific deletion is allowed after training.

**Owner decision (2026-10-09):** the proposed no-audit shortcut is not safe
because the existing bundles contain intermediate imputed rows. The bounded
metadata join and matched-control reruns are therefore required.

### 13. Output constraints

**Finding:** official outputs are unconstrained and can violate probability
bounds or OHLC identities.

**Recommendation:** retain the unconstrained source head. Save raw predictions
and report invalid-probability/OHLC/volume rates as diagnostics. Do not clip or
project headline predictions unless a separate pre-evaluation sensitivity is
commissioned.

**Owner decision (2026-10-09):** approved.

### 14. Phase/artifact ownership

**Finding:** xLSTM-Mixer was previously listed in Phase 6.6B and Phase 6.9.

**Recommendation:** make Phase 6.9 the sole technical, implementation, and
artifact authority. Train exactly one model per walk and store its source,
configuration, data, checkpoint, and prediction artifacts under
`experiments/phase6_9/xlstm_mixer/`. The old Phase 6.6B text is retained only
as a superseded historical reference and cannot launch a second model.

**Owner decision (2026-10-09):** approved. All active xLSTM-Mixer work belongs
to Phase 6.9; Phase 6.6 is outdated for this baseline.

## Implementation gate

Paper and source reading and all owner decisions are complete. Implementation
may begin. Training additionally requires the frozen common-row/target artifact
and Lumid Sandbox CUDA admission.
