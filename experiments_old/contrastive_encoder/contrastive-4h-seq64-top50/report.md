# Contrastive Encoder Experiment Report

This report covers unsupervised contrastive pretraining only. Test sequences are recorded in the manifest for traceability but are not used for training, early stopping, or checkpoint selection.

## Dataset

- processed npz: `data/processed/market_4h_seq64_top50.npz`
- train shape: `[109841, 64, 5]`
- test shape: `[27500, 64, 5]`

## Configuration

- seed: `0`
- batch size: `256`
- learning rate: `0.001`
- weight decay: `0.0001`
- hidden dim: `128`
- embedding dim: `128`
- temperature: `0.2`
- device request: `cuda`

## Epoch Budgets

| epoch | train NT-Xent loss | best train loss so far | elapsed seconds | checkpoint |
|---:|---:|---:|---:|---|
| 15 | 2.5917835608 | 2.5917835608 | 214.70 | `/mnt/e/school-work-6-y3s2/fyp/representation_learning_framework/experiments/contrastive_encoder/contrastive-4h-seq64-top50/e15/checkpoint.pth` |
| 20 | 2.5495741551 | 2.5495741551 | 287.62 | `/mnt/e/school-work-6-y3s2/fyp/representation_learning_framework/experiments/contrastive_encoder/contrastive-4h-seq64-top50/e20/checkpoint.pth` |
| 25 | 2.5224684096 | 2.5224684096 | 364.87 | `/mnt/e/school-work-6-y3s2/fyp/representation_learning_framework/experiments/contrastive_encoder/contrastive-4h-seq64-top50/e25/checkpoint.pth` |
| 50 | 2.4600282077 | 2.4599065508 | 727.67 | `/mnt/e/school-work-6-y3s2/fyp/representation_learning_framework/experiments/contrastive_encoder/contrastive-4h-seq64-top50/e50/checkpoint.pth` |
| 100 | 2.4099936491 | 2.4084807370 | 1460.38 | `/mnt/e/school-work-6-y3s2/fyp/representation_learning_framework/experiments/contrastive_encoder/contrastive-4h-seq64-top50/e100/checkpoint.pth` |

## Final Budget

- epoch: `100`
- final train NT-Xent loss: `2.4099936491`
- best observed train NT-Xent loss: `2.4084807370`

Generated images are saved under each `e*/images/` directory and under the run-level `images/` directory.

## Observations and Interpretation

The contrastive pretraining run shows a healthy optimisation pattern. The NT-Xent loss falls sharply in the early epochs, from `3.7005267143` at epoch 1 to `2.5917835608` at epoch 15, then continues improving more gradually through the longer budgets. The final 100-epoch checkpoint reaches `2.4099936491`, with the best observed training loss of `2.4084807370` at epoch 98.

Across the fixed-budget sweep, the loss improves by about 7.0% from epoch 15 to epoch 100. Most of the improvement happens before or around epoch 50; the additional gain from epoch 50 to epoch 100 is only about 2.0%. This suggests that the encoder has learned the contrastive objective and is approaching a plateau, rather than diverging or failing to optimise.

The result is meaningful as an unsupervised pretraining result, but it is not yet evidence that the representation improves price prediction, volatility forecasting, or trend classification. The test split was not used for training, early stopping, or checkpoint selection, so the experiment remains consistent with the project leakage rules. The proper next validation step is downstream probing with frozen embeddings: extract contrastive features for train and test sequences, train the aggregator and task heads on the training features, and evaluate once on the locked test split.

The current evidence does not justify modifying the contrastive architecture yet. The training curve is stable, the loss decreases consistently, and all requested checkpoints were produced. Architecture changes should be considered only after downstream ablations show that the contrastive branch adds little or no value compared with deterministic features, VAE embeddings, or the full multi-branch framework.

Recommended next step: keep the 100-epoch checkpoint as the current contrastive encoder candidate, then proceed with VAE pretraining and frozen feature extraction before training the first framework task loop.
