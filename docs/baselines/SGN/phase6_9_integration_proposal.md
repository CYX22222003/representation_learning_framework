# SGN-C Phase 6.9 integration proposal

**Status:** complete for the frozen two-walk seed-0 scope
**Comparison role:** supervised classification-specific complete system
**Primary metric:** macro-F1
**Primary checkpoint:** epoch 50, predeclared before evaluation

## 1. Scientific question

Can the reusable canonical target-free H0 representation remain competitive
on two-hour movement classification when compared with a recent supervised
multivariate classification architecture trained end to end for that task?

SGN-C does not test frozen representation quality. It receives labels
throughout backbone training and retains its native head, grouping
regularizer, and source-specific optimizer. A difference from H0-D0 therefore
combines representation, supervision, head, capacity, and optimization.

## 2. Fixed task and row contract

Use the accepted bundles:

```text
experiments/phase5/data_preparation/walk1/market_1h_seq64_h2.npz
experiments/phase5/data_preparation/walk2/market_1h_seq64_h2.npz
```

Input is stored `[N,64,5]` OHLCV in `open,high,low,close,volume` order. Labels
remain:

```text
delta = close[t+2h] - close[t]
DOWN   if delta < -0.001
STABLE if abs(delta) <= 0.001
UP     if delta > 0.001
```

| Walk | Train rows | Evaluation rows | Train class counts D/S/U | Evaluation class counts D/S/U |
|---:|---:|---:|---|---|
| 1 | 37,864 | 30,340 | 8,980 / 20,295 / 8,589 | 4,077 / 22,348 / 3,915 |
| 2 | 57,521 | 13,887 | 12,668 / 32,948 / 11,905 | 1,819 / 10,299 / 1,769 |

Every original identity, ordering, label, context, and evaluation row remains.
No model-specific deletion, future-path requirement, label-derived filter, or
sixth input channel is allowed.

## 3. Causal fitting boundary

For each walk independently, only classification training rows may determine:

- BDC sample membership and similarity matrix;
- K-means centroids/assignments and initial assignment logits;
- FFT/period diagnostics;
- training class priors;
- BatchNorm running statistics;
- model, optimizer, temperature, and scheduler state; and
- any resource-driven batch choice made before real training.

Walk 2 state cannot update Walk 1. Evaluation rows may be read only after the
50-epoch trajectory and all snapshot hashes are fixed. The model can convolve
and shift over the complete 64-hour input because every input timestamp is
historical at the decision time.

## 4. Training lifecycle

- two fresh trajectories: Walk 1 and Walk 2;
- seed 0 only;
- one uninterrupted or exactly resumed 50-epoch trajectory per walk;
- snapshots at epochs 5, 15, and 50;
- epoch 50 primary, with all snapshots reported;
- no validation split, early stopping, test-per-epoch logging, restart
  selection, or best-on-evaluation checkpoint;
- natural-frequency training rows; and
- existing logit-adjusted task loss plus separately recorded SGN grouping
  regularization.

The bootstrap remains audit/manifest/smoke-only unless `--execute` is supplied.

## 5. Required controls

Primary matched rows:

1. immutable canonical `H0-D0` classification;
2. immutable Raw LSTM classification; and
3. fresh SGN-C.

Raw MLP may appear as an existing identical-row contextual reference. TA-MLP
Phase 6.5C uses a feature-availability intersection and must not be mixed into
the full-row primary table. Always-STABLE and training-prior-only predictions
remain non-trained references.

Before control reuse, verify dataset path/hash, train/evaluation identity
hashes, label hash, task recipe, seed, checkpoint, prediction length/order,
and replayed metrics. Equal row counts alone are insufficient.

## 6. Evaluation and reporting

Report each walk first, then pooled out-of-future predictions:

- macro-F1 (primary);
- balanced accuracy and accuracy;
- per-class precision, recall, F1, and support;
- confusion matrix and predicted-class counts;
- collapse diagnostics;
- one-vs-rest ROC-AUC/PR-AUC, NLL, and multiclass Brier score where the
  existing task reporter supplies them;
- contract-macro and existing lifecycle/category/imputation subgroups;
- paired metric differences on identical rows;
- trainable parameters, optimizer state size, training/inference time, peak
  memory, and throughput; and
- assignment matrices, entropy, group masses, group changes from
  initialization, period evidence, and grouping-loss traces.

Pooling concatenates genuine out-of-future predictions only. Do not average
model ranks across the three heterogeneous Phase 6.9 tasks.

## 7. Claim boundary

Permitted outcomes include:

- H0 is competitive with or stronger than SGN-C on named metrics/walks;
- SGN-C's task-specific supervision improves classification on named
  metrics/walks; or
- evidence is mixed across walks, classes, or metrics.

The result cannot establish universal SOTA, representation causality,
variable-grouping causality, profitable trading, or multi-seed robustness.
Recognition benchmarks in the SGN paper and future movement prediction here
are different tasks.

## 8. Planned artifacts

```text
experiments/phase6_9/sgn_classification/
  feasibility/
  manifests/
  initialization/walk{1,2}/
  downstream/classification_h2/walk{1,2}/sgn_c/seed0/
    checkpoints/epoch_{5,15,50}.pt
    predictions/epoch_{5,15,50}.npz
    metrics/epoch_{5,15,50}.json
    history.npz
    run_manifest.json
    replay_validation.json
  reports/seed0/
```

Small manifests, metrics, predictions, summaries, and replay records may be
versioned under project policy. Large checkpoints remain governed by the
repository's established artifact rules.
