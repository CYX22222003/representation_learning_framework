# SaURL-TS Phase 6.7 implementation plan

This is an execution-ready plan, not authorization to start training. The
Stage 1 model adapter and SaURL-specific Stage 0--4 audit, pretraining,
feature-store, replay, and downstream code are implemented and pass focused
CPU tests. The Stage 0 CUDA/resource gate and both 50-epoch SaURL pretraining
trajectories are complete and replay-valid at epochs 5/15/50. Both master
stores and all six SaURL downstream trajectories are also complete and replay-
valid. The LWA adapter, full shared core manifest, and final reporting stage
remain pending.
All bootstrap commands are manifest-only unless `--execute` is explicitly
supplied.
Staged SaURL-only downstream execution was approved by the project owner on
2026-10-04. The later LWA implementation remains required and must be frozen
independently of SaURL results.

## Stage 0 — freeze provenance and independent-implementation boundary

1. Record the paper identity/licence and audited repository commit in
   `experiments/phase6_7/feasibility/saurl_ts/`.
2. Record that no software licence was found and prohibit copying, modifying,
   importing, or vendoring the public implementation.
3. Freeze the approved Questions 4--11 reconstruction and reporting label
   `SaURL-TS-Frozen (paper-guided reimplementation)`.
4. Run CPU/CUDA shape, gradient-isolation, replay, and resource smoke tests
   before any downstream evaluation metric is read. If the adapter fails, stop
   SaURL and record the failure rather than silently changing the method.

**Exit:** a signed-off feasibility manifest with the upstream no-reuse boundary,
the complete independent architecture contract, and a passing smoke/resource
gate. Otherwise stop SaURL implementation and record the failed gate.

## Stage 1 — implement the method adapter

Only after Stage 0 passes, add:

```text
src/baselines/saurl_ts/
  __init__.py
  config.py             # immutable typed architecture/training contract
  augmentation.py       # time/frequency ADT and learned-view losses
  encoder.py            # independently authored selected dilated CNN
  attention.py          # independently authored documented RwAM reconstruction
  losses.py             # five-kernel MMD and symmetric normalized BYOL loss
  model.py              # SaDA + three online/EMA paths
  adapter.py            # [B,64,5] project interface and extraction

tests/baselines/saurl_ts/
  test_augmentation.py
  test_losses.py
  test_encoder_attention.py
  test_alternating_updates.py
  test_checkpoint_replay.py
  test_adapter.py
```

Required public interface:

```python
model = build_saurl(config)
views_and_parts = model.make_views(batch)
sada_losses = model.sada_losses(views_and_parts)
sassl_outputs = model.sassl_forward(views_and_parts.detached_views())
embedding = model.encode(batch)  # [B, 128]
```

These are forward interfaces only. The walk runner, not `model.py`, owns
optimizer zeroing/stepping, trainability boundaries, global-step cadence, EMA,
checkpointing, and resume. This keeps the alternating order directly testable
and prevents an opaque `pretrain_step` from updating the wrong parameter set.

The adapter must never import upstream task evaluators or data loaders. It receives tensors from the project's validated walk bundles.

The three encoders use hidden width 64, output width 128, kernel size 3, and
six residual blocks with dilations `1,2,4,8,16,32`, followed by global maximum
pooling. RwAM uses eight 16-coordinate regions and the selected shared
`Conv1d(3,1,1) -> ReLU -> Conv1d(1,3,1)` average/max paths before sigmoid and
weighted summation.

Temporal and frequency SaDA are separate. Within each, a shared width-16,
depth-1 embedding/factor network emits a one-position mask broadcast across
five channels. The two views use separate informative/irrelevant transform
heads and share the paper-specified deterministic hard mask at threshold
`0.5`. A straight-through estimator preserves the binary forward mask while
providing gradients through the threshold; no logistic/Gumbel mask noise is
used.

### Exact module definitions

Each of `E_T`, `E_F`, and `E_C` is independent and has:

```text
input [B,64,5]
transpose to [B,5,64]
Conv1d(5,64,kernel=1)
six residual blocks at dilations 1,2,4,8,16,32
  y = Conv1d(64,64,kernel=3,padding=d,dilation=d)(x)
  y = GELU(y); y = Dropout(0.1)(y)
  y = Conv1d(64,64,kernel=3,padding=d,dilation=d)(y)
  y = Dropout(0.1)(y)
  x = GELU(x + y)
Conv1d(64,128,kernel=1)
transpose to [B,64,128]
global maximum over time -> [B,128]
```

No normalization layer or multi-kernel source-only convolution bank is added.
Convolutions retain bias and use PyTorch default initialization. Each online
branch has a projector `Linear(128,128) -> GELU -> Linear(128,128)` and a
predictor of the same shape. Target encoder/projector modules are deep copies
of their online counterparts; they have no predictor and never receive
gradients.

Each domain-specific augmentation encoder uses
`Conv1d(5,16,1)` followed by one dilation-1 version of the residual block above
at width 16. A shared `Linear(16,1)` emits factor logits. Four separate
`Linear(16,1)` heads emit sigmoid informative/irrelevant scales for views 1
and 2. At every position form `m=Sigmoid(logit)`,
`b=1[m>0.5]`, and `h=b-stopgrad(m)+m`. The same domain mask is used for both
view-specific transform-head pairs. Temporal masks are `[B,64,1]`; spectral
masks are `[B,33,1]`; both broadcast across five channels.

The input scaler is coordinatewise over the five channels using every
timestamp of the applicable walk's `encoder_train_sequences`. Replace a
standard deviation below `1e-8` with 1.0 and replay the saved statistics
unchanged. Do not fit clipping thresholds or inspect downstream rows.

The frequency-view implementation is selected and must not be replaced by a
magnitude-only encoder input: apply `rfft` over the 64-step time axis, separate
magnitude and phase, apply frequency SaDA only to magnitude, recombine every
transformed magnitude with the corresponding original phase, and use
`irfft(..., n=64)` to create each real `[B,64,5]` frequency view passed to
`E_F`. Frozen inference passes normalized real `[B,64,5]` input directly to
`E_F`; the redundant identity FFT round trip is omitted.

`losses.py` implements the paper's input-domain MMD semantics independently of
the public source. Flatten each per-sample domain tensor; use a five-kernel
Gaussian MMD with a detached, clamped mean off-diagonal bandwidth and scales
`{1/4,1/2,1,2,4}`; and use the biased `Kxx + Kyy - Kxy - Kyx` batch estimate.
Temporal continuity is the mean adjacent absolute difference of the
straight-through mask. The exact SaDA equations and coefficients are frozen in
[the clarification record](upstream_clarification_request.md#81-frozen-view-and-loss-definitions).

The cross-domain pair is also fixed: the first temporal SaDA view and first
phase-preserving reconstructed frequency view enter the two sides of the
cross-branch BYOL loss. Time and frequency branches use their respective
view-1/view-2 pairs. RwAM weights the three branch vectors before each
branch-specific projector; the target side uses the current shared RwAM under
`no_grad`, with no EMA copy of RwAM.

**Tests:**

- exact tensor shapes for `B in {2,4}`, `T=64`, `F=5`;
- finite time and frequency views;
- frequency SaDA receives magnitude but never modifies phase;
- safe `rfft/irfft` reconstruction with explicit output length 64;
- identity spectral transformation reconstructs the input within a frozen
  numerical tolerance;
- the frequency encoder receives real `[B,64,5]` tensors in training and
  inference, never complex or `[B,33,5]` magnitude tensors;
- deterministic paper-threshold masks, straight-through gradients, identical
  masks across the two domain views, and distinct view-head outputs;
- five-kernel MMD symmetry, zero-on-identical tolerance, finite bandwidth for
  repeated/constant samples, and the expected sign of diversity objectives;
- temporal mask total variation is zero for a constant mask;
- SaDA-only and SaSSL-only parameter-update isolation;
- one joint-update batch changes SaDA before regenerating detached SaSSL views;
- one SaSSL-only batch leaves every SaDA parameter and gradient unchanged;
- EMA target has no gradients;
- EMA target values exactly match `0.99*old + 0.01*online` after one step;
- RwAM weights and 128-wide weighted sum;
- cross loss receives temporal view 1 and frequency view 1 of identical rows;
- deterministic inference after checkpoint reload; and
- projector/predictor exclusion from `encode`.

## Stage 2 — add shared Phase 6.7 encoder training

Add a method registry and walk-specific runner, preferably:

```text
src/training/phase6_7_external_encoders.py
scripts_v6/bootstrap_phase6_7_external_encoders.py
scripts_v6/validate_phase6_7_external_encoders.py
```

The SaURL config is immutable after manifest freeze. The runner must:

- validate the Phase 5 bundle and its manifest/hash;
- load only `encoder_train_sequences`;
- fit/store the walk-local five-channel input scaler;
- count complete no-drop data-loader passes as epochs; if the remainder is one,
  merge it into the preceding batch so every row appears once and MMD sees at
  least two samples;
- use seed 0, Adam with zero weight decay, SaDA learning rate `1e-2`, SaSSL
  learning rate `1e-4`, and EMA `0.99`;
- update SaDA first every two minibatches with representation parameters
  frozen; update SaSSL every minibatch with SaDA frozen and generated views
  detached; then update EMA targets;
- checkpoint at 5, 15, and 50;
- record all component/optimizer/scaler states needed for exact resume;
- record Python, NumPy, Torch CPU/CUDA, and data-loader generator RNG states so
  an interrupted trajectory resumes at the exact next shuffled batch and
  dropout state;
- record loss components, mask rates, branch embedding std/norm, time, memory, and parameter count; and
- refuse overwrite or source/config/hash drift.

### Normative minibatch implementation

Use `global_step`, not an epoch-local batch index, for the `ratio_step=2`
cadence. Step zero is a joint-update batch.

```python
for epoch in range(start_epoch, 50):
    for raw_batch in no_drop_seeded_loader:
        x = input_scaler.transform(raw_batch).float()
        sada_losses = None

        if global_step % 2 == 0:
            set_trainable(sada_modules, True)
            set_trainable(sassl_online_and_rwam, False)
            zero_grad(opt_sada_time, opt_sada_freq)
            sada_views_and_parts = model.make_views(x)
            sada_losses = model.sada_losses(sada_views_and_parts)
            assert_finite(sada_views_and_parts, sada_losses)
            sada_losses.total.backward()
            assert_only_sada_has_gradients()
            opt_sada_time.step()
            opt_sada_freq.step()

        set_trainable(sada_modules, False)
        set_trainable(sassl_online_and_rwam, True)
        with torch.no_grad():
            views = model.make_views(x).detached_views()

        opt_sassl.zero_grad(set_to_none=True)
        sassl = model.sassl_forward(views)
        assert_finite(sassl.representations, sassl.losses)
        sassl.total_loss.backward()
        assert_only_online_sassl_and_rwam_have_gradients()
        opt_sassl.step()
        model.update_targets(tau=0.99)

        log_step(global_step, sada_losses, sassl)
        global_step += 1
```

`make_views` is called again after a SaDA optimizer step; pre-update views must
not be reused for SaSSL. Mask construction is the deterministic paper
threshold conditional on the current factor logits and has no stochastic-mask
argument. The primary trajectory uses
float32 without AMP and fails on non-finite losses or gradients rather than
silently skipping an update. No gradient clipping or unplanned optimizer
scheduler is introduced.

### Checkpoint contract

At epochs 5, 15, and 50 save:

- both SaDA modules and their two Adam optimizer states;
- all three online and target encoders/projectors, three predictors, RwAM, and
  the SaSSL Adam state;
- scaler state, epoch, global step, sampler position/state, and all RNG states;
- frozen config, source/data/row hashes, code revision, device metadata, and
  loss-history hash; and
- a fixed probe input plus expected branch, fused, and attention outputs.

Resume must reproduce an uninterrupted small-run fixture exactly on CPU before
CUDA execution is admitted.

**Replay:** reload every checkpoint on CPU, reproduce a fixed probe batch, verify checkpoint hashes/config/data identity, and ensure epoch 50 passes collapse and finiteness checks. Same-device CUDA replay must be bit-exact. Cross-device replay uses relative L2 `<=5e-4` and cosine `>=0.999999`; this scale-aware criterion was frozen after the Walk 1 probe showed only `0.012%--0.021%` relative drift despite large absolute activations. A collapse is recorded as a failed run, not tuned away.

## Stage 3 — extract two SaURL master stores

Add:

```text
src/features/phase6_7_external_features.py
scripts_v6/prepare_phase6_7_external_features.py
scripts_v6/validate_phase6_7_external_features.py
```

For each walk, validate all three task bundles and extract epoch-50 SaURL features from their exact train/test raw-sequence arrays. Normalize with the frozen walk encoder scaler; feed the same real input independently through `E_T`, `E_F`, and `E_C`; global-max-pool each branch; apply RwAM; and save the 128-wide weighted sum. SaDA and every BYOL training-only module are absent from this path. Save one method/walk store with task-prefixed arrays and identity fields.

**Validation:**

- source checkpoint and input-scaler hashes match;
- every task feature matrix is `[N_task_split,128]` and finite;
- identity arrays exactly match the source task bundle in order;
- repeated extraction hashes are identical on the same device;
- CPU versus CUDA embeddings are numerically close on a frozen probe subset;
- no task target is loaded by the encoder extraction function;
- saved features equal a direct manual sum of the three saved probe-batch
  attention-weighted branch vectors; and
- changing a discarded projector, predictor, target, or SaDA parameter cannot
  change `encode(x)`.

## Stage 4 — implement native-width common probes

Add rather than modify completed Phase 6 modules:

```text
src/training/phase6_7_downstream.py
scripts_v6/bootstrap_phase6_7_downstream.py
scripts_v6/validate_phase6_7_downstream.py
```

Generalize the existing downstream recipe to a manifest-declared positive input width. Keep all task-specific behavior unchanged:

- movement: `TrendClassifier`, train-prior logit-adjusted cross-entropy;
- price: `AbsolutePriceRegressor`, MSE, sigmoid-bounded output;
- volatility: `VolatilityRegressor`, `10000 * RV`, Smooth L1, Softplus, gradient clipping;
- hidden width 128, batch 512, learning rate `1e-4`, seed 0; and
- epochs 5, 15, and 50 with epoch 50 principal.

Use a new generic feature standardizer with the existing mean/std, small-std fallback, and `[-10,10]` clipping semantics. Do not relax the completed Phase 6 `445|573` checks in place.

**Replay:** reconstruct the head and scaler on CPU; reproduce predictions and metrics from saved artifacts; assert exact row identities; and reject missing, added, or reordered rows.

## Stage 5 — freeze and execute the matrix

Before execution, `experiments/phase6_7/manifests/downstream_seed0.json` must enumerate:

- six immutable H0 references;
- six LWA-F trajectories;
- six mandatory SaURL-F trajectories; and
- every dataset, feature store, run root, hash, width, task, walk, seed, and budget.

Manifest creation and CPU smoke tests occur without `--execute`. Training begins only with explicit execution.

The implemented SaURL-only bootstrap freezes
`saurl_downstream_seed0.json` and permits `--execute` for those six SaURL
trajectories. The later complete `downstream_seed0.json` must incorporate
these immutable runs with LWA and the six `H0` references. SaURL metrics may
not motivate a change to the later LWA implementation or a SaURL rerun.

## Stage 6 — report without selection

Add:

```text
src/evaluation/phase6_7_reporting.py
scripts_v6/report_phase6_7_frozen_representations.py
```

The report must include all 5/15/50 snapshots, with epoch 50 as principal; per-walk and pooled absolute metrics; paired differences from H0; resource/parameter/native-width tables; source deviations; failure/collapse rows; and the bounded seed-0, two-walk interpretation.

Do not add or remove a model, change the extraction point, or rerun with new hyperparameters after reading downstream results.

## Stage 7 — recommended command contract

Exact filenames may be adjusted during implementation, but the behavior should remain:

```bash
# Source/CPU smoke manifest; never launches an encoder trajectory.
.venv/bin/python3 scripts_v6/audit_phase6_7_sources.py --device cpu

# Explicit CUDA/resource admission. The reviewer supplies the frozen limits;
# they are not inferred or tuned from evaluation results.
.venv/bin/python3 scripts_v6/audit_phase6_7_sources.py \
  --device cuda --admit-training \
  --max-peak-memory-gib <GIB> --max-smoke-seconds <SECONDS>

# Manifest and CPU smoke only by default.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_external_encoders.py --device cpu

# Explicit training.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_external_encoders.py --device cuda --execute
.venv/bin/python3 scripts_v6/validate_phase6_7_external_encoders.py

# Frozen epoch-50 feature extraction and replay.
.venv/bin/python3 scripts_v6/prepare_phase6_7_external_features.py --device cuda
.venv/bin/python3 scripts_v6/validate_phase6_7_external_features.py

# Manifest/CPU smoke, then explicit probe execution.
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_downstream.py --device cpu
.venv/bin/python3 scripts_v6/bootstrap_phase6_7_downstream.py --device cuda --execute
.venv/bin/python3 scripts_v6/validate_phase6_7_downstream.py
.venv/bin/python3 scripts_v6/report_phase6_7_frozen_representations.py
```

## Definition of done for SaURL-F

- source and software-licence absence are recorded, and no upstream code is reused;
- the independent implementation matches the approved Questions 4--11 contract;
- unit, gradient-isolation, alternating-order, exact-resume, CPU/CUDA smoke,
  FFT round-trip, RwAM, and extraction-boundary tests pass;
- two walk-specific trajectories have replay-valid epochs 5/15/50;
- two 128-dimensional master stores have exact task identities and hashes;
- six downstream trajectories and 18 snapshots replay independently;
- all resource and source-adaptation disclosures are present; and
- no evaluation result influenced the model, checkpoint, row set, or retry policy.
