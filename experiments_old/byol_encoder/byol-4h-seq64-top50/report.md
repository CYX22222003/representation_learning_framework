# BYOL Encoder Experiment Report

This report covers unsupervised BYOL pretraining only. Test sequences are recorded in the manifest for traceability but are not used for training, early stopping, or checkpoint selection.

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
- projection dim: `128`
- predictor hidden dim: `128`
- target decay: `0.99`
- collapse std threshold: `0.001`
- device request: `cuda`

## Epoch Budgets

| epoch | train BYOL loss | view cosine | embedding std | collapse warning | best loss so far | elapsed seconds | checkpoint |
|---:|---:|---:|---:|:---:|---:|---:|---|
| 15 | 0.0754657454 | 0.9622671515 | 0.6184719251 | False | 0.0039418837 | 191.52 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/byol_encoder/byol-4h-seq64-top50/e15/checkpoint.pth` |
| 20 | 0.0810225011 | 0.9594887715 | 0.7010175591 | False | 0.0039418837 | 250.28 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/byol_encoder/byol-4h-seq64-top50/e20/checkpoint.pth` |
| 25 | 0.0796773641 | 0.9601613389 | 0.7765034167 | False | 0.0039418837 | 315.87 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/byol_encoder/byol-4h-seq64-top50/e25/checkpoint.pth` |
| 50 | 0.0776185841 | 0.9611907314 | 0.9908381116 | False | 0.0039418837 | 635.14 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/byol_encoder/byol-4h-seq64-top50/e50/checkpoint.pth` |
| 100 | 0.0526101089 | 0.9736949717 | 1.3356986524 | False | 0.0039418837 | 1283.12 | `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework/experiments/byol_encoder/byol-4h-seq64-top50/e100/checkpoint.pth` |

## Final Budget

- epoch: `100`
- final train BYOL loss: `0.0526101089`
- final view cosine: `0.9736949717`
- final embedding std: `1.3356986524`
- final embedding norm: `13.6477851512`
- collapse warning: `False`

Generated images are saved under each `e*/images/` directory and under the run-level `images/` directory.
