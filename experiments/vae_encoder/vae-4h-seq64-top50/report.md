# VAE Encoder Experiment Report

This report covers unsupervised VAE pretraining only. Test sequences are recorded in the manifest for traceability but are not used for training, early stopping, or checkpoint selection.

## Dataset

- processed npz: `data/processed/market_4h_seq64_top50.npz`
- train shape: `[109841, 64, 5]`
- test shape: `[27500, 64, 5]`

## Configuration

- seed: `0`
- batch size: `256`
- learning rate: `0.001`
- weight decay: `0.0001`
- latent dim: `64`
- hidden dim: `256`
- beta: `1.0`
- device request: `cuda`

## Epoch Budgets

| epoch | total loss | reconstruction MSE | KL divergence | elapsed seconds | checkpoint |
|---:|---:|---:|---:|---:|---|
| 15 | 0.0999306878 | 0.0699566867 | 0.0299740011 | 19.96 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/vae_encoder/vae-4h-seq64-top50/e15/checkpoint.pth` |
| 20 | 0.0865633022 | 0.0669389362 | 0.0196243661 | 26.10 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/vae_encoder/vae-4h-seq64-top50/e20/checkpoint.pth` |
| 25 | 0.0841178878 | 0.0659496934 | 0.0181681944 | 32.58 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/vae_encoder/vae-4h-seq64-top50/e25/checkpoint.pth` |
| 50 | 0.0827743671 | 0.0647486225 | 0.0180257445 | 64.44 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/vae_encoder/vae-4h-seq64-top50/e50/checkpoint.pth` |
| 100 | 0.0801516505 | 0.0625848766 | 0.0175667738 | 125.33 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/vae_encoder/vae-4h-seq64-top50/e100/checkpoint.pth` |

## Final Budget

- epoch: `100`
- final total loss: `0.0801516505`
- final reconstruction MSE: `0.0625848766`
- final KL divergence: `0.0175667738`

Generated images are saved under each `e*/images/` directory and under the run-level `images/` directory.

## Executive Summary

The 4h VAE encoder completed the canonical fixed-budget CUDA sweep on the locked training split. The run produced checkpoints, histories, metrics, summaries, and plots for epoch budgets `15`, `20`, `25`, `50`, and `100`; the final-budget checkpoint was also copied to `checkpoints/vae_4h_seq64_top50.pth` for downstream frozen-encoder use.

Training used `109841` training sequences from `data/processed/market_4h_seq64_top50.npz`. The held-out test split shape (`27500` sequences) was recorded only for traceability and was not used for training, early stopping, or checkpoint selection. This keeps the experiment aligned with the project train/test-only leakage rules.

The final 100-epoch checkpoint reached total loss `0.0801516505`, reconstruction MSE `0.0625848766`, and KL divergence `0.0175667738`. The best observed training loss across the full history was `0.0800388432` at epoch `98`, so the final checkpoint is very close to the best training point reached by the fixed recipe.

## Observations and Interpretation

The VAE optimised successfully overall, but the curve contains an early transient instability. Total loss fell from `0.0995371267` at epoch 1 to approximately `0.08094` by epoch 11, then spiked at epochs 13 and 14 (`26.4875965118` and `34.0451469421`) before recovering by epoch 15. Because the run recovered without intervention and continued improving through epoch 100, this appears to be a training instability event rather than a failed run.

The fixed-budget table should therefore be read with that context: the 15-epoch snapshot is not representative of the stable early trajectory because it was saved immediately after the spike recovery. Later budgets show a cleaner trend. Total loss improves from `0.0865633022` at epoch 20 to `0.0841178878` at epoch 25, `0.0827743671` at epoch 50, and `0.0801516505` at epoch 100.

Most of the stable post-spike improvement occurs before epoch 50, but the 50-to-100 interval still improves total loss by about 3.2% and reconstruction MSE by about 3.3%. The KL term remains small and stable after recovery, ending near `0.01757`, which suggests the latent regularisation stayed active without dominating reconstruction.

This result is meaningful as unsupervised VAE pretraining evidence. It is not yet downstream performance evidence for price prediction, volatility forecasting, or trend classification. The proper next validation step is frozen feature extraction: run the VAE encoder over train and test sequences to store a named `vae` branch, regenerate the stale deterministic feature bundles with current `70`/`55` dimensions, and train the first aggregator plus task-head probing loop on the training features.

Recommended next step: keep the 100-epoch checkpoint as the current VAE encoder candidate, then implement or run branch-aware frozen embedding extraction for `vae` and `contrastive` before the first framework task experiment.
