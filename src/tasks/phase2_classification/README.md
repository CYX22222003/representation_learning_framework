# Phase 2 probability-movement classification

This task package is isolated from the Phase 1 trend implementation. It uses hard,
split-safe `DOWN/STABLE/UP` targets and records three-class softmax scores for
every test row. The candidate imbalance protocols are:

| ID | Training treatment |
|---|---|
| `P1U` | retain both minority classes and undersample only the unique majority class |
| `P1O` | oversample every class to the largest training-class count |
| `P2` | retain all rows and use train-prior logit-adjusted cross-entropy |

`P0` is available only as an untreated natural-sampling cross-entropy
reference. Sampling and priors are derived from training labels only. Test rows
always retain their observed class distribution.

## Pipeline

Run all commands from the project root with `.venv/bin/python3`.

```bash
# 1. Create the shared h=2, tau=0.005 label bundle.
.venv/bin/python3 scripts/prepare_probability_movement_labels.py --overwrite

# 2. Create TA features and freeze the common TA-eligible row intersection.
.venv/bin/python3 scripts/prepare_phase2_ta_features.py \
  --labels-npz data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz \
  --overwrite

# 3. Bootstrap the three-model, three-protocol, three-seed command matrix.
.venv/bin/python3 scripts/bootstrap_phase2_classification.py

# Inspect commands.sh and matrix_manifest.json, then execute explicitly.
.venv/bin/python3 scripts/bootstrap_phase2_classification.py --execute --overwrite

# Add the untreated P0 reference when the comparison requires it.
.venv/bin/python3 scripts/bootstrap_phase2_classification.py \
  --include-p0-reference --execute --overwrite

# 4. Aggregate completed runs.
.venv/bin/python3 scripts/report_phase2_classification.py \
  experiments/framework/phase2/classification_relabelling/4h_h2_tau005
```

The bootstrapper runs the Raw-OHLCV MLP (`C1`), five-branch framework (`C2`),
and bundle-aware TA-MLP (`C5`) on identical TA-eligible row identities. Each
matrix also includes the full-row `C1` and `C2` primary `P2` runs unless
`--skip-full-row-primary` is passed. Reports keep full-row and TA-aligned
results in separate comparison scopes. Each run saves configuration, dataset
and sampling manifests, train-fitted scaling,
checkpoints, history, logits, probabilities, decisions, curve data, global and
per-contract metrics, and a full epoch-budget summary.

The bootstrap command only creates a manifest and replayable shell script unless
`--execute` is supplied. There is no validation split, early stopping, or
test-selected checkpoint.
