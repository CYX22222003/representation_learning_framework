# XM-C8 endpoint-only readiness record

Date: 2026-10-09. Authority:
`docs/phase_plan/2026-10-09-phase-6-9-xlstm-endpoint-amendment.md`.

The owner-approved primary price comparator is now direct close[t+8h], one
scalar output, endpoint-only raw-probability MSE, and every original h8 price
row. This is a post-XM-MV8-result protocol correction. No endpoint metric
selected the recipe. Fresh XM-C8 weights are mandatory; no full endpoint
trajectory has been launched.

## Verified

- Source/manifest replay and original row preservation: Walk 1 36,773/29,834;
  Walk 2 56,652/13,506 train/evaluation.
- Original H0-D0/Raw-LSTM dataset and checkpoint training-source hashes,
  budgets/seed/optimizer recipe, evaluation identities/targets, and prediction
  hashes agree with the original bundles. No intersection reruns are required.
- New endpoint tests: 18 passed, including direct output shape, exact MSE,
  rejection of auxiliary paths/broadcasted labels, source reversal/RevIN,
  checkpoint architecture guards, no row selection, same-seed epoch resume,
  checkpoint/prediction/identity/target/history/metric replay, and tampering.
- Entire focused suite: 43 passed, one opt-in legacy GPU-fixture test skipped.
  The new bounded real-GPU admission is independently verified below.
- Local vanilla sLSTM on CUDA, RTX 4060 Laptop GPU: batch 512 plus one-row
  remainder, finite loss/gradients, optimizer updates, and same-backend
  checkpoint reload pass. 91,842 parameters; peak allocated 289,603,584 bytes
  (about 276 MiB); bounded synthetic GPU probe 1.076 seconds.
- Active manifest-only shell pipeline exits zero, with no --execute and no
  trajectory launch. Matrix, admission, and readiness artifacts are recorded.
- Historical fingerprinted XM-MV8 implementation is unchanged:
  `8b77c09e85cb5adeaacc3972b4aedb4910059e4745858e8ca63ce54a1e7b234a`.
  Its source/data/model/training and replay files were not replaced.

## Evidence and next action

- `manifests/training_seed0.json`: immutable two-walk endpoint freeze and
  original control provenance.
- `feasibility/cuda_admission.json`: endpoint-specific dependency/runtime/
  data/code identity and resource admission; old XM-MV8 admission is not used.
- `readiness.json`: successful CPU smoke and non-training readiness status.
- `logs/pipeline_20261009T122830_34822.log`: successful reissued GPU readiness pipeline.
- `readiness_pre_path_normalization/`: recoverable archive of the first three
  pre-training readiness records. Reissued records normalize only E:-mount
  path spelling, preserving code/data hash and recipe guards. This fixes
  false artifact drift between uppercase/lowercase WSL paths; no trajectory
  or scientific contract changed.

An explicitly requested launch can now use:

```bash
bash scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh --execute
```

It trains fresh independent 50-epoch endpoint trajectories, retains 5/15/50,
evaluates only after training, replays, and produces the endpoint-matched
H0/Raw-LSTM comparison. No XM-C8 performance result exists yet. Its inverse-
RevIN output remains unclipped and clip norm 1.0 is disclosed against the
sigmoid/no-clipping control models; matching targets and rows does not
establish architecture causality, universal superiority, or trading value.
