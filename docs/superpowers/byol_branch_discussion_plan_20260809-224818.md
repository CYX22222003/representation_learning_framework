# BYOL as an Additional Neural Encoder Branch

Timestamp: 2026-08-09 22:48:18

## Short Position

My opinion is that BYOL is a meaningful candidate addition to the current representation learning framework, but it should be added as an experimental neural branch, not as a guaranteed improvement claim.

The current framework already has three complementary representation families:

- deterministic statistical features: AR and GARCH features, currently 70 dimensions for 5 OHLCV columns;
- deterministic transformed features: FFT and Haar wavelet features, currently 55 dimensions;
- neural self-supervised features: VAE latent embeddings, currently 64 dimensions, and SimCLR-style contrastive embeddings, currently 128 dimensions.

BYOL fits the open "additional unsupervised methods" slot in the design because it is neither generative like the VAE nor negative-pair discriminative like SimCLR. It learns augmentation-invariant embeddings through an online network, a target network updated by exponential moving average, and a predictor head. In this project, that gives BYOL a clean role: it can test whether non-contrastive self-supervision captures market-sequence structure that VAE and SimCLR miss.

The strongest reason to add it is not that BYOL is fashionable. The reason is that the project is explicitly about transferable multi-branch representations. BYOL adds a genuinely different pretraining pressure while still using only unlabeled OHLCV sequences and respecting the frozen-encoder probing paradigm.

## Why BYOL Complements the Current Branches

### Compared with VAE

The VAE branch is generative. It must preserve enough sequence information to reconstruct the input under a latent bottleneck. This is useful because it encourages broad information retention, but it can also spend capacity on low-level noise, local scale patterns, or reconstruction details that are not useful for downstream price, volatility, or trend tasks.

BYOL has a different bias. It does not reconstruct the raw sequence. Instead, it learns representations that remain stable across two augmented views of the same window. If the augmentations are well chosen, the BYOL embedding should emphasize features that survive small perturbations: regime shape, relative movement structure, volatility bursts, and temporal patterns that are robust to jitter, masking, and mild scaling.

This is a meaningful complement because financial OHLCV windows contain both useful signal and high noise. A reconstruction objective can over-preserve noise; a BYOL objective can suppress some of it.

### Compared with SimCLR / NT-Xent contrastive learning

The existing contrastive branch uses positive pairs from two augmented views and negatives from other sequences in the same batch. That is a valid self-supervised objective, but in financial time series the negative-pair assumption is imperfect.

Different windows can legitimately share the same latent regime:

- two markets may both be in a volatility expansion;
- two contracts may have similar trend/momentum structure;
- two different time windows may reflect the same market response pattern.

SimCLR treats those other windows as negatives if they appear in the same batch. That may be acceptable in aggregate, but it can push apart samples that are semantically similar. BYOL avoids explicit negatives, so it may preserve shared regime clusters more naturally.

That does not mean BYOL is automatically better. BYOL can collapse without its architectural safeguards, and it depends heavily on augmentation quality. But the difference from SimCLR is real enough to justify a controlled branch experiment.

### Compared with deterministic branches

The statistical branch captures interpretable mean and volatility dynamics through AR and GARCH. The transformed branch captures frequency and multi-scale energy structure through FFT and wavelets. These are valuable but fixed.

BYOL can learn nonlinear invariances that are not hand-specified:

- "same trend shape despite small noise";
- "same volatility regime despite masked timesteps";
- "same relative pattern despite mild amplitude scaling";
- "same local structure across related OHLCV channels."

That is aligned with the project objective of combining feature families that see different parts of the market sequence.

## Expected Value

BYOL is worth adding if it helps at least one of these:

1. Improves full-framework downstream metrics when added to the existing branches.
2. Performs competitively as a single-branch ablation.
3. Receives meaningful gating weight in gated aggregation, especially on tasks where SimCLR or VAE are weak.
4. Improves transferability across timeframes or market regimes.
5. Produces embeddings that are less sensitive to batch size than the SimCLR branch.

The most likely benefits are:

- better robustness to noisy OHLCV perturbations;
- less dependence on high-quality negative samples;
- stronger regime-level embeddings for volatility and trend tasks;
- a clearer research story: generative VAE, negative-pair contrastive SimCLR, non-contrastive BYOL.

The most likely risks are:

- collapse or low-variance embeddings if the implementation is wrong or augmentations are too weak;
- redundant information if BYOL learns nearly the same invariances as the existing contrastive encoder;
- additional compute and experiment complexity;
- branch-dimension inflation in concat mode, which may make the downstream head stronger simply because it has more input features;
- data-contract friction because the current `FeatureBundle` has one optional `neural` matrix, while the architecture conceptually treats `vae`, `contrastive`, and future encoders as separate named branches.

My recommendation is to implement BYOL, but only with an ablation plan that can reject it. If it does not improve downstream tasks or show a distinct contribution, it should remain a documented negative result rather than be absorbed into the default framework.

## Fit with Current Framework

BYOL fits the current system well because:

- it trains unsupervised on the existing `[N, seq_len, 5]` OHLCV tensors;
- it can reuse the existing CNN backbone style from `src/models/contrastive.py`;
- it can reuse the same augmentations initially: jitter, scaling, and time masking;
- it produces a fixed-size embedding that can be frozen and passed into `RepresentationAggregator`;
- it follows the existing fixed-epoch, no-validation, train-only pretraining rule.

The clean conceptual branch table after adding BYOL would be:

| Branch | Module | Suggested output dim | Role |
|---|---:|---:|---|
| `statistical` | `src/features/statistical.py` | 70 | AR/GARCH interpretable dynamics |
| `transformed` | `src/features/transform.py` | 55 | FFT/wavelet structure |
| `vae` | `src/models/vae.py` | 64 | generative latent reconstruction |
| `contrastive` | `src/models/contrastive.py` | 128 | SimCLR/NT-Xent discriminative SSL |
| `byol` | `src/models/byol.py` | 128 | non-contrastive bootstrap SSL |

In concat mode, adding a 128-dimensional BYOL branch would increase the default full embedding from 317 dimensions to 445 dimensions:

```text
current: 70 + 55 + 64 + 128 = 317
with BYOL: 70 + 55 + 64 + 128 + 128 = 445
```

In gated mode, the output remains `out_dim`, for example 128, but BYOL adds one more projected branch and one more gate weight.

## Important Design Decision: Separate Branch or Packed Neural Matrix

The architecture documentation treats neural encoders as independent branches. The aggregator supports this directly because it accepts arbitrary `branch_dims: dict[str, int]`.

However, `FeatureBundle` currently stores:

- `statistical`;
- `transformed`;
- `neural`, one optional matrix.

This decision has since been resolved in favor of branch-aware storage. The two alternatives remain useful background, but new feature stores should follow Option B.

### Option A: Keep one packed `neural` matrix

Concatenate VAE, SimCLR, and BYOL embeddings into one matrix and pass it as a single `neural` branch.

Benefits:

- smallest change to `FeatureBundle`;
- easy to save in the existing `.npz` format;
- fast to implement.

Costs:

- loses separate branch identity inside the aggregator;
- gated mode cannot learn separate weights for VAE, SimCLR, and BYOL;
- weakens the multi-branch ablation story.

### Option B: Support named neural branches

Represent the full feature dictionary as separate arrays:

```python
{
    "statistical": statistical_features,
    "transformed": transformed_features,
    "vae": vae_embeddings,
    "contrastive": contrastive_embeddings,
    "byol": byol_embeddings,
}
```

Benefits:

- matches the architecture;
- enables clean single-branch ablations;
- enables gated aggregation to show whether BYOL contributes;
- scales better for future encoders.

Costs:

- requires changing feature storage/loading utilities or adding a new branch-aware feature bundle path;
- downstream scripts need to build branch dictionaries instead of assuming one `neural` matrix.

Decision: use Option B. Adding BYOL while keeping all neural encoders packed into one `neural` matrix would make the result harder to interpret. Since the research claim is about branch complementarity, separate named branches are the more defensible design. Legacy stores with a packed or empty `neural` key may still be read for compatibility, but new neural embeddings should be saved under encoder names such as `vae`, `contrastive`, and eventually `byol`.

## Implementation Plan

### Phase 1: Define the BYOL model

Add `src/models/byol.py` with:

- a reusable sequence backbone, preferably matching the current contrastive CNN backbone for fair comparison;
- an online encoder;
- an online projector;
- an online predictor;
- a target encoder;
- a target projector;
- EMA update logic for the target network;
- normalized projection outputs;
- a BYOL loss based on negative cosine similarity between online predictions and stop-gradient target projections.

Initial suggested dimensions:

- backbone hidden dim: 128;
- projection dim: 128 or 256;
- predictor hidden dim: 128;
- downstream embedding dim: 128, using the online backbone representation `h`, not the projector output.

Using the same CNN backbone as the SimCLR branch is useful for attribution. If BYOL improves results, the likely cause is the pretraining objective, not a more powerful architecture.

### Phase 2: Add BYOL training loop

Add `src/training/train_byol.py` with a `train_byol_epoch` function.

The training step should:

1. Load a batch from the train split only.
2. Create two augmented views using the existing augmentation utilities at first.
3. Pass view 1 through the online network and view 2 through the target network.
4. Pass view 2 through the online network and view 1 through the target network.
5. Compute the symmetric BYOL loss.
6. Backpropagate through the online network only.
7. Update the target network by EMA after the optimizer step.

The target network must not receive gradients. Collapse checks should be included in logged metrics, not used for early stopping.

Suggested logged diagnostics:

- train BYOL loss;
- mean embedding norm;
- per-dimension embedding standard deviation;
- average cosine similarity between paired views;
- a simple collapse warning if embedding variance falls below a fixed threshold.

### Phase 3: Add training script

Add `scripts/train_byol_encoder.py`, matching the style of:

- `scripts/train_vae_encoder.py`;
- `scripts/train_contrastive_encoder.py`.

Suggested command:

```bash
python scripts/train_byol_encoder.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name byol-4h-seq64-top50 \
  --epoch-budgets 15,20,25,50,100 \
  --seed 0 \
  --device cuda \
  --canonical-checkpoint checkpoints/byol_4h_seq64_top50.pth
```

The script should:

- load only the `train` key from the processed `.npz`;
- save experiment artifacts under `experiments/byol_encoder/<run-name>/`;
- save snapshots at each fixed epoch budget;
- copy the final epoch-budget checkpoint to `checkpoints/byol_<timeframe>_seq64_top50.pth` when requested;
- write `config.json`, `dataset_manifest.json`, `metrics.json`, `history.npz`, and `summary.md`;
- explicitly record that the test split was not used for BYOL pretraining.

### Phase 4: Add embedding extraction

Add a way to extract frozen BYOL embeddings for train and test sequences.

This can be done either as:

- a new script, for example `scripts/extract_neural_embeddings.py`, supporting `--encoder vae,contrastive,byol`; or
- BYOL-specific extraction first, then generalize later.

For research clarity, the output should preserve branch names:

```text
byol_train: [N_train, 128]
byol_test:  [N_test, 128]
```

The branch-aware feature store should contain distinct keys such as:

```text
statistical
transformed
vae
contrastive
byol
```

The companion index file should continue to store `train_size` and `test_size`.

### Phase 5: Integrate with the aggregator

When constructing `RepresentationAggregator`, register BYOL as a separate branch:

```python
branch_dims = {
    "statistical": statistical_feature_dim(n_cols=5, ar_order=5),
    "transformed": transform_feature_dim(n_cols=5),
    "vae": 64,
    "contrastive": 128,
    "byol": 128,
}

agg = RepresentationAggregator(branch_dims, mode="concat")
```

Task heads must use `agg.output_dim`, not a hard-coded dimension.

For single-branch ablations, BYOL should be evaluated alone using the same default task heads as the other branches.

### Phase 6: Add tests

Minimum useful tests:

- BYOL forward pass returns the expected shapes.
- Target network parameters have `requires_grad=False`.
- EMA update changes target parameters after an optimizer step.
- BYOL loss is finite for a normal batch.
- Frozen embedding extraction returns `[N, 128]`.
- Aggregator accepts a branch dictionary containing `byol`.
- Feature store/load path preserves a named BYOL branch if the feature store is upgraded.

### Phase 7: Update documentation only after implementation

After implementation is real, update:

- `AGENTS.md` branch table;
- `docs/design.md` representation learning section;
- `docs/training_test_data_selection.md` component table and operation order;
- `docs/Research_Ideas_Writeup.md` if the thesis framing will explicitly claim BYOL as part of the method;
- project skills under `.agents/skills/` if their architecture summaries mention neural branch dimensions.

Implementation update: the BYOL model, epoch training loop, fixed-budget pretraining script, plotting/report script, and CPU smoke tests now exist. The branch is still not a validated default until real train-split pretraining and downstream ablations show marginal value.

## Evaluation Plan

BYOL should be evaluated in a way that can prove marginal contribution.

### Pretraining setup

Use the same data and fixed-budget rules as the existing neural encoders:

- train only on the global 80% train split;
- never train BYOL on test sequences;
- no validation split;
- no early stopping;
- use epoch budgets `15,20,25,50,100`;
- report the full sweep, not the best-on-test checkpoint;
- use seed `0` for the canonical characterization sweep;
- optionally add multi-seed runs later for noise calibration.

### Downstream tasks

Evaluate on the same three downstream tasks:

- price prediction: MAE, RMSE;
- volatility prediction: MSE, Pearson correlation;
- trend classification: accuracy, macro-F1, and per-class metrics.

### Required ablations

At minimum:

| Configuration | Purpose |
|---|---|
| Statistical only | deterministic mean/volatility baseline |
| Transformed only | deterministic frequency/multi-scale baseline |
| VAE only | generative neural baseline |
| SimCLR only | negative-pair contrastive baseline |
| BYOL only | non-contrastive SSL baseline |
| Current full framework | measures existing framework without BYOL |
| Full framework + BYOL | tests marginal BYOL contribution |

If gated aggregation is available:

- compare BYOL gate weights across tasks;
- check whether BYOL receives non-trivial weight or is ignored;
- report concat and gated separately because gated mode adds supervised parameters.

### Acceptance criteria

BYOL should become a default framework branch only if at least one of these is true:

1. Full framework + BYOL improves at least two downstream tasks beyond noise.
2. BYOL-only is competitive with VAE-only or SimCLR-only on at least one task.
3. Gated aggregation assigns BYOL meaningful weight on at least one task and improves metrics.
4. BYOL improves transfer across timeframes.
5. BYOL reduces performance instability across seeds.

BYOL should remain optional if:

- it improves only one metric weakly;
- it is redundant with SimCLR;
- it increases concat dimension without clear downstream value;
- it requires fragile augmentation tuning to avoid collapse.

BYOL should be rejected or postponed if:

- embeddings collapse repeatedly under reasonable settings;
- it harms most downstream metrics;
- implementation work delays baseline completion or final evaluation;
- the feature storage contract is not ready for clean named neural branches.

## Suggested Priority

I would place BYOL after the current VAE and SimCLR pretraining/evaluation path is stable, but before adding a heavier Transformer-style encoder.

Reason:

- BYOL is architecturally close to the existing contrastive encoder, so implementation cost is moderate.
- It adds a distinct objective, unlike another minor reconstruction variant.
- It is lighter than a Transformer and easier to evaluate under the current fixed-budget experiment setup.
- It directly strengthens the framework's "complementary representation branches" story.

The right sequencing is:

1. Finish and verify current VAE + SimCLR feature extraction and downstream evaluation.
2. Upgrade or clarify the feature store so neural branches can remain separately named.
3. Implement BYOL with the same CNN backbone and augmentations as SimCLR. *(Done.)*
4. Run BYOL pretraining on one timeframe first, preferably 4h. *(Pending.)*
5. Run BYOL-only and full-with-BYOL downstream ablations.
6. Decide whether BYOL enters the default branch table based on evidence.

## Final Recommendation

BYOL is a good addition to investigate because it fills a real methodological gap between the VAE and SimCLR branches. It gives the framework a non-generative, non-negative-pair self-supervised encoder that may be especially suitable for financial time series where many different windows can share the same latent market regime.

I would implement it, but I would keep the claim conservative: BYOL is a candidate complementary branch until ablations show that it contributes unique downstream value. The implementation should prioritize clean branch identity, fair fixed-budget training, and collapse diagnostics. If those are in place, BYOL is a defensible and useful extension to the current framework.
