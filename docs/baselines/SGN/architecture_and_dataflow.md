# SGN-C proposed architecture and data flow

**Status:** all seven owner decisions approved
**Scope:** Phase 6.9 classification only
**Input:** saved `[B,64,5]` walk-specific OHLCV contexts
**Output:** three logits in `DOWN, STABLE, UP` order

**Implementation record:** independently authored under `src/baselines/sgn/`
with its gated lifecycle in `src/training/phase6_9_sgn.py`; 17 focused CPU
tests and real-row CUDA batch-256 admission pass. No real epoch has run.

## 1. Recommended fixed contract

| Item | Value | Basis | Approval |
|---|---:|---|---|
| groups `G` | `2` | both training-only BDC probes recover OHLC vs volume | approved |
| latent width per group `D` | `64` | paper's common best width and source main settings | approved |
| period `P` | `16` | common training-only FFT candidate; divides 64; safe four-window hierarchy | approved |
| stages | `4` | window counts `4 -> 3 -> 2 -> 1` under `P=16` | approved |
| blocks per stage | `[2,2,2,1]` | source pattern: two alternating blocks, one final block | approved |
| temporal kernels | `7` | odd kernels `1,3,5,7,9,11,13` | approved |
| expansion ratio | `2` | paper/source main setting | approved |
| dropout | `0.1` | source default | approved |
| native pooled width | `G*D = 128` | direct consequence | approved |
| head | `Linear(128,3)` | native SGN head, no shared H0 probe | approved |

The complete numerical contract is approved.

## 2. End-to-end tensor flow

```text
saved context                           [B, 64, 5]
    │
    ├─ training-only BDC/K-means prior  [5, 5] -> [5, 2]
    │
    ▼
soft/hard variable assignment M         [5, 2]
    │  assignment-weighted group sum
    ▼
two group signals                       [B, 64, 2]
    │  shared Conv1d(1,64,k=3,circular) + sinusoidal position
    ▼
group embeddings                        [B, 128, 64]
    │  partition P=16
    ▼
stage 1 input                           [B, 128, 4, 16]
    │  unshifted block -> shifted block -> overlapping merge
    ▼
stage 2 input                           [B, 128, 3, 16]
    │  unshifted block -> shifted block -> overlapping merge
    ▼
stage 3 input                           [B, 128, 2, 16]
    │  unshifted block -> shifted block -> overlapping merge
    ▼
stage 4 input                           [B, 128, 1, 16]
    │  final unshifted block; no merge
    ▼
LayerNorm(128) + mean over 16 positions [B, 128]
    │
    ▼
native linear head                      [B, 3]
```

No label, target-interval candle, evaluation statistic, H0 feature, or
additional model-specific row filter enters this path.

## 3. Training-only initialization

### 3.1 BDC sample

For each walk independently:

1. load only `train_sequences` from the accepted Phase 5 bundle;
2. select exactly 2,000 rows without labels using a deterministic,
   contract-stratified chronological-quantile sampler;
3. treat each selected length-64 channel path as one vector observation, as in
   the source helper;
4. use the accepted stored OHLCV values directly, with no additional
   time-position or channel scaler; distance correlation already normalizes
   each variable's distance variance;
5. compute pairwise Euclidean distance matrices, double-center them, and form
   normalized distance correlations;
6. run project-native deterministic K-means on the five BDC rows with `G=2`,
   fixed seed/tie rules, and multiple fixed initializations;
7. require both clusters to be nonempty; and
8. save selected identities, BDC matrix, centroids, assignments, code/config
   hashes, and replay output.

The sampler budget and stratification are project safeguards, not paper
claims. They control the quadratic cost and avoid letting pooled file order or
one long contract dominate initialization.

### 3.2 Assignment logits

Initialize a `5 x 2` trainable logits matrix from the K-means one-hot
assignment plus seed-0 Gaussian perturbation with standard deviation `0.1`,
clipped to `[0,1]`, matching the released intent. Store the realized matrix in
the manifest. Do not load any upstream `.npy` matrix.

Training draws Gumbel noise and uses a soft assignment. Evaluation uses a
deterministic hard argmax. The recommended temperature schedule is:

```text
tau(step) = max(0.1, 1.0 * exp(-0.0003 * step))
```

where `step` counts optimizer updates and is saved/restored exactly.

### 3.3 Group fusion and embedding

The recommended source-aligned fusion is:

```text
X_group[b,t,g] = sum_c X[b,t,c] * M[c,g]
```

without division by assignment mass. Apply one shared embedding module to
each group signal after reshaping groups into the batch dimension. The shared
module is a circular `Conv1d(1,D,kernel=3,padding=1)` plus fixed sinusoidal
position encoding and dropout. This follows released behavior but narrows the
paper's phrase “independent group-wise embedding”; the report must disclose
that interpretation.

The accepted pipeline already leaves OHLC in `[0,1]` probability units and
standardizes volume using walk-training mean/std, so the input channels are of
comparable numerical order. They are not identically scaled: the source-style
OHLC group is still a four-variable sum while the volume group contains one
standardized variable. The approved shared embedding and later BatchNorm can
learn around that cardinality difference; it remains a disclosed diagnostic.

## 4. Period selection

The proposed fixed `P=16` is derived before model training from a training-only
diagnostic, then shared by both walks because both produce the same candidate
set. The formal initializer should:

1. compute `rFFT` amplitude over the time axis on stored training contexts;
2. average with equal channel weight over rows and channels;
3. exclude DC and the full-length period;
4. record the top five frequency bins and rounded candidate periods; and
5. verify that `16` remains in both candidate sets before admitting the frozen
   manifest.

This is deliberately not input-local FFT selection. A changing period would
change padding and hierarchy per batch/sample and is absent from the released
forward path. If `16` fails the frozen diagnostic, stop for a new owner
decision rather than choosing from evaluation results.

## 5. Multi-scale group-window block

Each block receives `[B,G*D,N,P]`.

### 5.1 Temporal bank

Reshape windows into the batch dimension and apply two averaged depthwise
kernel banks with GELU between them:

```text
DepthwiseAvgConv(GD -> 2GD, kernels 1..13)
GELU
DepthwiseAvgConv(2GD -> GD, kernels 1..13)
```

Use group count `GD` so each latent coordinate has its own temporal filters.
Apply source-aligned `BatchNorm1d(D)` after reshaping groups into the batch
dimension, sharing normalization parameters across groups.

### 5.2 Intra-group mixing

Apply `1x1` convolutions `GD -> 2GD -> GD` with convolution groups `G`, GELU,
and dropout. Each group mixes only its own `D` coordinates.

### 5.3 Inter-group mixing

Reshape/permute to make each latent coordinate own a `G`-wide slice. Apply a
second `1x1` pair with groups `D`, then restore the original layout. Each
latent coordinate can now mix across groups. Add the original block input as
a residual.

## 6. Shift and merge semantics

Odd-numbered blocks use no shift; even-numbered blocks roll the flattened
window sequence left by `P//2=8`, repartition, process, and roll right.

The proposed source-aligned boundary rule restores the first eight positions
from the block input after the inverse roll. This avoids artificial interaction
between the oldest and newest observed timestamps but differs from the paper's
pure cyclic description.

After each of the first three stages, merge overlapping neighboring windows:

```text
4 -> 3 -> 2 -> 1
```

For each adjacent pair, concatenate along latent width (`2D`), apply
`Linear(2D,D)`, then LayerNorm. Stage 4 has no merge. Configuration validation
must simulate the complete window-count schedule and reject any plan that
would merge `N=1`.

## 7. Loss

Let `z` be the three native logits and `y` the fixed label. The task component
is the existing training-prior logit-adjusted cross-entropy:

```text
L_task = CE(z + log(training_class_prior), y)
```

with strength `1.0`. The grouping component uses the current batch's detached
channel paths, cosine similarity rescaled from `[-1,1]` to `[0,1]`, and the
soft assignment matrix:

```text
L_sim = sum_(i,j,g) S01[i,j] * (M[i,g]-M[j,g])^2 / C^2
L     = L_task + 0.1 * L_sim
```

Record `L_task`, unweighted `L_sim`, weighted grouping contribution,
temperature, assignment entropy, hard group membership, group mass, and
gradient norms per epoch. Evaluation computes no grouping loss and uses hard
assignments.

## 8. Data and comparison boundary

| Walk | Training rows | Evaluation rows | Training interval | Evaluation interval |
|---:|---:|---:|---|---|
| 1 | 37,864 | 30,340 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | 57,521 | 13,887 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Inputs preserve the accepted walk preprocessing: OHLC is unscaled probability
and volume is standardized from that walk's training interval. No second SGN
input scaler is proposed. The five model channels remain exactly
`open,high,low,close,volume`; explicit imputation metadata stays in artifacts
for validation/reporting and is not added as a sixth channel.

## 9. Replay-critical state

Every retained checkpoint must include:

- complete model and optimizer state;
- epoch, global optimizer step, and temperature;
- Python/NumPy/Torch CPU/CUDA RNG states;
- sampler permutation/generator state;
- BDC sample identities, BDC matrix, K-means state, initial logits, and period
  diagnostic hashes;
- training class counts/priors and task/grouping loss settings; and
- data, source, code, environment, and manifest hashes.

Hard assignments, logits, predictions, metrics, and resource measurements are
saved for independent replay at epochs 5, 15, and 50.
