# Phase 5 Canonical Encoder Pretraining Amendment

**Date:** 2026-09-21
**Status:** Executed and replay-validated
**Parent authority:** `2026-09-21-phase-5-experiment-plan.md`

## Scope

This amendment freezes the encoder-pretraining portion of Phase 5. The five
canonical representation branches are statistical, transformed, VAE,
contrastive CNN, and BYOL CNN. Statistical AR/GARCH and transformed FFT/Haar
features are deterministic and require no pretraining. The executed matrix is
therefore three neural encoders times two independently fitted calendar walks.

## Frozen matrix

| Walk | Encoder | Backbone | Downstream dimension |
|---:|---|---|---:|
| 1 | VAE | MLP encoder/decoder | 64 |
| 1 | Contrastive | CNN, NT-Xent | 128 |
| 1 | BYOL | CNN online/target | 128 |
| 2 | VAE | MLP encoder/decoder | 64 |
| 2 | Contrastive | CNN, NT-Xent | 128 |
| 2 | BYOL | CNN online/target | 128 |

Each run uses only its bundle's `encoder_train_sequences`. It is one
uninterrupted 50-epoch trajectory at seed 0 with snapshots at epochs 5, 15,
and 50. Epoch 50 is predeclared as the sole downstream feature-extraction
checkpoint. No validation split, early stopping, evaluation inference, or
downstream-metric checkpoint selection is permitted.

## Fixed recipe

- Batch size: 256, shuffled training rows, incomplete final batch dropped.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- VAE: hidden dimension 256, latent dimension 64, `beta=1.0`.
- Contrastive: CNN hidden/output dimensions 128, NT-Xent temperature `0.2`.
- BYOL: CNN hidden/output dimensions 128, target EMA decay `0.99`.
- Contrastive and BYOL reuse the existing five-channel scaling, jitter, and
  time-mask augmentations. They do not consume an explicit imputation mask.
- Canonical stored data are never rewritten by augmentation. Imputation remains
  implicit in the volume channel; explicit metadata is validation/reporting
  information only.

Walk 1 and Walk 2 use identical recipes but independently initialized and
trained weights. Phase 1--3 checkpoints are not initialization inputs.

## Artifacts and execution

Runs live under:

```text
experiments/phase5/encoder_pretraining/walk{1,2}/{vae,contrastive,byol}/seed0/
```

Each snapshot stores `checkpoint.pth`, cumulative `history.npz`, and
`metrics.json`. Each run also stores configuration, environment, dataset, and
architecture manifests plus completion and sweep summaries. The matrix is
frozen and executed through:

```bash
.venv/bin/python3 scripts_v3/launch_phase5_encoder_pretraining.py --device cuda
.venv/bin/python3 scripts_v3/launch_phase5_encoder_pretraining.py --execute
```

## Completion

All six trajectories completed on the WSL CUDA device. Every run has snapshots
at epochs 5, 15, and 50, and every checkpoint, cumulative history, dataset hash,
encoder-row identity hash, state-dict load, and epoch-50 CPU inference replay
passed. No run raised a representation-collapse warning.

| Walk | Encoder | Epoch-50 loss | Epoch-50 embedding std |
|---:|---|---:|---:|
| 1 | VAE | 0.10893485 | 0.05645689 |
| 1 | Contrastive | 2.33701080 | 1.18781137 |
| 1 | BYOL | 0.07470790 | 0.44650105 |
| 2 | VAE | 0.07066280 | 0.09024052 |
| 2 | Contrastive | 2.25663392 | 2.99989295 |
| 2 | BYOL | 0.05124977 | 0.70031059 |

These values are unsupervised training/health observations only. They are not
downstream performance measurements and did not select a checkpoint. The
consolidated report is under
`experiments/phase5/reports/encoder_pretraining_seed0/`.
