# xLSTM-Mixer clarification and decision record

**Status:** recommendations prepared; owner decisions pending

**Purpose:** freeze paper/source/project ambiguities before implementation or
training

**Downstream metrics consulted:** none

These are internal adaptation decisions. They do not ask the paper authors to
change the method. Once resolved, the answers must be copied into the source
manifest, Phase 6.6B architecture text, implementation configuration, and
experiment manifests before any evaluation result is read.

## Recommended decision set

### 1. Reference identity

**Finding:** the audited paper is NeurIPS 2025/arXiv v4, while the untagged
repository's model code dates to October 2024 and omits final-paper additions.

**Recommendation:** cite the NeurIPS 2025 paper as the scientific source and
pin official commit `730b0531...` as the behavioral source. Label the project
model “xLSTM-Mixer (`XM-MV8`; NeurIPS 2025 architecture adapted from the
pinned official source).” Do not call it an exact reproduction.

**Owner decision:** pending.

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

**Owner decision:** pending.

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

**Owner decision:** pending.

### 4. RevIN affine behavior

**Finding:** the paper equation uses learned affine parameters; source sets
`affine=False`.

**Recommendation:** preserve source `affine=False`, detached per-instance
statistics, `unbiased=False`, and epsilon `1e-5`. This is unambiguous released
behavior and avoids adding parameters absent from the official forward path.

**Owner decision:** pending.

### 5. Number of initial tokens

**Finding:** the paper says one `eta`; source default is zero and run scripts
tune zero through four.

**Recommendation:** use exactly one learned initial token. It follows the
scientific method text and avoids validation-based selection among script
settings. Initialize it with source `Normal(0,0.01)`.

**Owner decision:** pending.

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

**Owner decision:** pending.

### 7. Channel order and target path

**Finding:** recurrence over variates makes ordering meaningful. The project
stores `[open, high, low, close, volume]`, and the headline endpoint is the
eighth future close.

**Recommendation:** retain exactly `[open, high, low, close, volume]` for both
input and output. Predict all observed bars `t+1,...,t+8`; extract output
`[:,7,3]`. Do not reorder channels, use separate contracts, permit imputed
targets, or set `H=1` and relabel it `t+8`.

**Owner decision:** pending.

### 8. Global scaler and loss domain

**Finding:** source loaders standardize channels on training data before the
model's RevIN. Raw project volume would otherwise dominate full-path L1.

**Recommendation:** fit one five-coordinate scaler from permitted walk-
training accepted candles, after the existing upstream volume transform.
Apply it to every context and target bar in that walk. Train with unweighted
mean L1 over all 40 standardized outputs, then invert both RevIN and the
channel scaler for artifacts and diagnostics. Store scaler population,
parameters, and hashes.

**Owner decision:** pending.

### 9. Training schedule and batch semantics

**Finding:** official runs use 40--60 epochs and dataset-specific schedules.
The project already precommits epoch 50 and snapshots 5/15/50.

**Recommendation:** use float32, seed 0, Adam `lr=1e-3`,
`beta=(0.9,0.999)`, no weight decay, gradient clip 1.0, and the released
warmup/constant/cosine scheduler with warmup 5, constant 2, gamma 0.98, and
cosine 15. Use a provisional physical batch 128 with `drop_last=False`, subject
only to the pre-training CUDA resource smoke. No gradient accumulation or
automatic batch fallback. Any resource-mandated smaller batch must be frozen
before training and used for both walks.

**Owner decision:** pending.

### 10. Checkpoint selection

**Finding:** official code tests a best validation-MSE checkpoint. The project
has no validation split and cannot tune from evaluation results.

**Recommendation:** retain complete epochs 5/15/50, use epoch 50 as the fixed
primary result, and never restore a best-loss epoch. Save model, optimizer,
scheduler, scaler, RNG/sampler, configuration, data/source hashes, and fixed
replay probes atomically.

**Owner decision:** pending.

### 11. CUDA and CPU replay

**Finding:** the pinned xLSTM package defaults to its custom CUDA sLSTM but
also supplies a vanilla backend. The source wrapper does not expose backend
selection. The current project environment lacks the dependency, and CUDA was
not available to the static audit.

**Recommendation:** expose backend only as an execution/replay setting, not a
model hyperparameter. Train with the pinned CUDA backend after a persistent-
sandbox build and fixed-batch forward/backward admission. Validate state-dict
compatibility and replay the same fixed probes with the vanilla backend on
CPU. Freeze exact tolerances from the smoke before training. If the two paths
are structurally incompatible or drift materially, reject CPU numerical replay
and require same-backend CUDA replay; do not invent a tolerance afterward.

**Owner decision:** pending.

### 12. Full-path row availability

**Finding:** source forecasting assumes a regular complete grid; the project
price endpoint alone does not guarantee observed OHLCV at every intermediate
bar.

**Recommendation:** run the Phase 6.6B metadata-only audit before model code
execution. If any established price row is incomplete, freeze one common
intersection and rerun H0-D0 and Raw LSTM on exactly the intersected training
and evaluation rows. Record row/contract loss. No model-specific deletion is
allowed after training.

**Owner decision:** pending.

### 13. Output constraints

**Finding:** official outputs are unconstrained and can violate probability
bounds or OHLC identities.

**Recommendation:** retain the unconstrained source head. Save raw predictions
and report invalid-probability/OHLC/volume rates as diagnostics. Do not clip or
project headline predictions unless a separate pre-evaluation sensitivity is
commissioned.

**Owner decision:** pending.

### 14. Phase/artifact ownership

**Finding:** xLSTM-Mixer is deliberately listed in Phase 6.6B and Phase 6.9.

**Recommendation:** train exactly one model per walk. Store all source,
configuration, scaler, checkpoint, and prediction artifacts under
`experiments/phase6_6/recent_forecasting_baseline/`. The Phase 6.9 report must
reference those exact hashes and must not launch a second model.

**Owner decision:** pending.

## Implementation gate

Paper and source reading are complete. Implementation remains gated until all
fourteen decisions are approved or replaced explicitly. Training has the
additional full-path data and CUDA/runtime gates. No evaluation metrics may be
read to settle any open choice.
