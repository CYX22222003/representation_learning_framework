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
| 15 | 2.4686718748 | 2.4686718748 | 201.82 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/framework/phase2/encoder_refinement/pretraining/contrastive_lstm-4h-seq64-top50-seed0/e15/checkpoint.pth` |
| 50 | 2.3208122804 | 2.3073009278 | 697.01 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/framework/phase2/encoder_refinement/pretraining/contrastive_lstm-4h-seq64-top50-seed0/e50/checkpoint.pth` |
| 100 | 2.2664846646 | 2.2647794782 | 1389.11 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/framework/phase2/encoder_refinement/pretraining/contrastive_lstm-4h-seq64-top50-seed0/e100/checkpoint.pth` |

## Final Budget

- epoch: `100`
- final train NT-Xent loss: `2.2664846646`
- best observed train NT-Xent loss: `2.2647794782`

Generated images are saved under each `e*/images/` directory and under the run-level `images/` directory.
