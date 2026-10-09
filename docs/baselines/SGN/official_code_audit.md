# SGN official source and settings audit

**Audit date:** 2026-10-09
**Repository:** <https://github.com/colison/SGN>
**Branch:** `master`
**Pinned commit:** `c6d1b573dcb8c4255cde59b988f74334ea5da503`
**Commit timestamp:** `2025-05-16T13:19:49+08:00`
**Audit scope:** static behavioral reference only; no upstream code reuse

## 1. Provenance and licence

The paper names the repository above. The audited checkout contains three
commits according to GitHub and was pinned to the current `master` head. It
contains model, experiment, data-provider, and layer files but no README,
release tag, `LICENSE`, `COPYING`, `NOTICE`, package licence declaration, or
model/data licence statement.

**Licence verdict:** source inspection and factual behavioral description are
permitted for this research planning record, but copying, modifying,
vendoring, or redistributing upstream source is not admitted. SGN-C must be an
independently authored paper-guided implementation. If the owner wants source
reuse, explicit upstream permission or a published licence is required first.

## 2. Audited files and hashes

| File | SHA-256 | Role |
|---|---|---|
| `sgnmodels/SGN.py` | `7b954974387201b84ec50d3162f689ff4810280047aeea7132b16eae4a34efd2` | model and grouping behavior |
| `exp/exp_classification.py` | `f2d7710a7f807003906d82195a080b8b2d21753c6590347b365c5ba14cddf042` | BDC/K-means and training loop |
| `run.py` | `4fd95e162de2480cbd0a2bc1f906879ad9ecece9de695c87f059ef3c8b1e2ae5` | CLI and seed handling |
| `sgn.sh` | `72552b28b724d082a859380e9e9da9ad1af287385be2b8ae15efce4ef43c12ce` | published dataset settings |
| `requirements.txt` | `b55ed6ac777e0c32d5a081492c204c11cdd402933ae2a6f62dcee58907adfc8b` | dependency pins |

The four committed similarity matrices have shapes FLAAP `6x6`, PTB-XL
`12x12`, TDBRAIN `33x33`, and UCI-HAR `9x9`. Their raw-data provenance and
construction commands are not recorded.

## 3. Paper/source/settings crosswalk

| Topic | Paper | Pinned source/settings | Audit judgement |
|---|---|---|---|
| BDC input | BDC between variables; observation layout not fixed | experiment helper treats each window as one observation vector of length `L`; a commented path uses the first 2,000 windows | Ambiguous; project must freeze its own train-only bounded sampler |
| Initial groups | K-means on BDC rows | scikit-learn KMeans, `random_state=0`, then clipped Gaussian noise (`std=0.1`) added to one-hot membership | Behavioral reference, not fully reproducible because the active precomputed matrices lack provenance |
| Assignment values | paper writes probabilities/log-probabilities in Equation 2 | noisy values are stored directly as trainable logits | Mismatch |
| Group fusion | grouped according to `M`; “independent embeddings” | `einsum` sums variables into one scalar series per group; no division by group mass | Material semantic choice |
| Group embedding | independent embedding for each group | one shared `DataEmbedding(1,D)` is reused for every group | Paper/source ambiguity |
| Temperature | exponential decay, soft train/hard evaluation | starts `1.0`, floor `0.1`, decay `0.0003` per training forward; eval forces `0.1` and hard argmax | Parameters available; state/device handling is fragile |
| Similarity loss | dynamic cosine-weighted assignment distance | batch cosine, rescaled to `[0,1]`, squared assignment difference, sum divided by `C^2`; trainer multiplies by `0.1` | Source resolves reduction and sign |
| Period | FFT-derived top-K candidate; Equation 5 selects smallest candidate period | fixed CLI `--period`; model performs no FFT | Direct mismatch |
| Temporal kernels | odd multi-scale grouped convolutions, averaged | seven branches in published runs, kernels `1..13`, average aggregation | Aligned |
| Intra/inter mixing | separate pointwise group interactions | `groups=G` then group/latent permutation and `groups=D` | Source clarifies tensor implementation |
| Shift | cyclic left `P/2`, process, cyclic right | same, then overwrite the first `P//2` positions with the block input | Source adds boundary behavior absent from paper |
| Merge | non-overlap above four windows; overlapping at four or fewer | implements this rule and carries an odd last window in the non-overlap branch | Aligned in intent |
| Head | projection after hierarchy | LayerNorm, temporal mean pool, `Linear(G*D,num_class)` | Native head identified |
| Main optimization | Adam; SGN lr `1e-3`; validation early stopping | Adam, clip norm `4`, 100 epochs, early stopping; batch 32 or 256 | Project must remove validation/early stopping and use fixed 50 epochs |

## 4. Critical release defects

### 4.1 The checkout is not runnable as published

`run.py` formats and downstream code read `args.num_groups` and
`args.kernel_size`, but the parser defines neither argument. The shell script
passes `--num_groups`, which argparse would reject. `SGN.Model` reads
`configs.kernel_size`, which therefore cannot exist under the published CLI.

The Python files also import modules through `SGN4MTSC.*`, while the checkout
root contains `sgnmodels/`, `exp/`, `layers/`, and `data_provider/` directly
and provides no `SGN4MTSC` package directory or installation metadata.

These are static release defects, not project-environment failures. The
project must not claim reproduction of official training.

### 4.2 Group initialization is hard-coded to TDBRAIN

The experiment builder creates dataset-specific group files, but
`GumbelGroupEmbedding` always loads `groups_matrix_T.npy`. The committed
TDBRAIN similarity matrix has 33 variables and cannot initialize the 6-, 9-,
or 12-variable datasets as written. This also prevents using the source model
directly for five-channel OHLCV.

### 4.3 Similarity matrices are opaque artifacts

Four `.npy` matrices are committed, but there is no recorded source dataset
hash, row selection, preprocessing state, command, or random state that
reconstructs them. The active builder loads them rather than calling its BDC
helper. Phase 6.9 therefore cannot reuse any matrix; it must derive and hash a
fresh matrix from each walk's training rows only.

### 4.4 BDC computation is under-specified and quadratic

The source helper standardizes a reshaped `(N*C,L)` array by time coordinate,
then builds an `N x N` distance matrix for each variable. It is quadratic in
the number of sampled windows. The commented call slices `data[:2000]`, but
the active matrices do not prove that this was used. A bounded, deterministic,
contract-aware training-only sample must be part of the project manifest.

### 4.5 Assignment state is not checkpoint-safe enough

The source replaces the registered temperature buffer with a new CPU tensor
during forward/update. Training step is a buffer, but temperature mutation and
Gumbel RNG state are not integrated into a complete resume contract. The
project implementation should update buffers in place and save optimizer,
temperature step, sampler, and all RNG states.

### 4.6 `G=1` is not a safe active path

When `G=1`, the source bypasses group embedding, but still returns
`embeddings_for_loss`, which is undefined. Phase 6.9 should require `G>=2`
and test invalid settings explicitly.

### 4.7 Test metrics are read throughout training

The trainer evaluates validation and test loaders after every epoch and uses
validation F1 for checkpointing. This conflicts with the project's locked
evaluation and fixed-budget rules. No upstream trainer logic is reusable.

### 4.8 Metrics code contains reproducibility hazards

The source calls `softmax(preds)` without an explicit dimension and derives
the number of classes from test labels. Phase 6.9 must fix the output width to
three from the approved task contract and compute class priors only from
training labels.

## 5. Training-only project diagnostics

A read-only probe used only `train_sequences` from the two accepted Phase 5
bundles; it did not inspect labels or evaluation metrics for selection.

### Source-like BDC probe

The probe used 1,024 evenly spaced training rows per walk and the source's
window-as-observation distance-correlation layout. Both source-like reshaped
standardization and direct accepted stored values were checked and produced
the same `G=2` partition. The source-like approximate BDC matrices were:

```text
Walk 1
        O       H       L       C       V
O   1.0000  0.9996  0.9996  1.0000  0.0467
H   0.9996  1.0000  0.9985  0.9995  0.0469
L   0.9996  0.9985  1.0000  0.9996  0.0469
C   1.0000  0.9995  0.9996  1.0000  0.0469
V   0.0467  0.0469  0.0469  0.0469  1.0000

Walk 2
        O       H       L       C       V
O   1.0000  0.9995  0.9996  0.9999  0.0346
H   0.9995  1.0000  0.9986  0.9994  0.0361
L   0.9996  0.9986  1.0000  0.9997  0.0332
C   0.9999  0.9994  0.9997  1.0000  0.0344
V   0.0346  0.0361  0.0332  0.0344  1.0000
```

`G=2` produced the same partition in both walks: OHLC together and volume
alone. `G=3/4` split almost indistinguishable OHLC variables and was less
stable across walks. This supports `G=2` as a pre-evaluation recommendation.

### FFT probe

After excluding DC and the full-sequence frequency, both stored inputs and
per-window/channel standardized inputs produced frequency bins `2..6` as the
top five in both walks, corresponding approximately to periods
`32,21,16,13,11`. The recommended `P=16` is a project design choice because it
divides 64 exactly and yields four windows for a complete `4->3->2->1`
hierarchy. A literal reading of Equation 5 would instead select `P=11` and
pad to 66. This must be owner-approved.

These probes are planning evidence, not frozen initialization artifacts. The
implemented initializer must use the approved bounded sampler and save the
actual rows, matrices, clusters, spectra, and hashes.

## 6. Feasibility verdict

**Technically feasible, legally source-independent, scientifically bounded.**

The architecture can be implemented with native PyTorch/NumPy/SciPy already
present in the project. Scikit-learn is absent from the local `.venv` and is
not needed: a deterministic five-point K-means implementation can be authored
and tested locally. No additional dependency should be installed solely to
mirror the unauditable upstream clustering helper.

Static source inspection is complete. Runtime reproduction of the official
repository is neither required nor credible without repairing its release,
and repair would not resolve the missing licence. All owner decisions are now
approved; the independent implementation, focused tests, train-only
initializers, manifest, and CUDA admission are complete without upstream code
reuse.
