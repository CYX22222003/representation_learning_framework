# Phase 3 encoder-pretraining report

All values are training-side diagnostics from seed 0. The runs use one uninterrupted
50-epoch trajectory with snapshots at epochs 5, 15, and 50. Epoch 50 was
predeclared for downstream feature extraction; no test metric selected it.

| variant | epoch | loss | embedding std | parameters | seconds | collapse |
|---|---:|---:|---:|---:|---:|:---:|
| vae_mlp | 5 | 0.22370870 | 0.10957144 | 345536 | 2.59 | False |
| vae_mlp | 15 | 0.21336708 | 0.07104427 | 345536 | 6.47 | False |
| vae_mlp | 50 | 0.20412594 | 0.07451112 | 345536 | 19.14 | False |
| contrastive_cnn | 5 | 2.76766142 | 0.97501928 | 85632 | 16.72 | False |
| contrastive_cnn | 15 | 2.49327582 | 1.02904165 | 85632 | 49.53 | False |
| contrastive_cnn | 50 | 2.27666946 | 0.71730429 | 85632 | 171.04 | False |
| contrastive_lstm | 5 | 2.91943172 | 0.23948069 | 102144 | 17.25 | False |
| contrastive_lstm | 15 | 2.26464406 | 0.29330197 | 102144 | 46.79 | False |
| contrastive_lstm | 50 | 2.06971543 | 0.30216110 | 102144 | 151.84 | False |
| contrastive_transformer | 5 | 2.44873477 | 0.72270757 | 299008 | 24.25 | False |
| contrastive_transformer | 15 | 2.25265358 | 0.66242713 | 299008 | 75.00 | False |
| contrastive_transformer | 50 | 2.14099439 | 0.55455494 | 299008 | 237.19 | False |
| byol_cnn | 5 | 0.00505745 | 0.19732729 | 118656 | 18.09 | False |
| byol_cnn | 15 | 0.01147504 | 0.32678953 | 118656 | 50.62 | False |
| byol_cnn | 50 | 0.07148756 | 0.41898707 | 118656 | 175.68 | False |
| byol_lstm | 5 | 0.00944538 | 0.16148880 | 135168 | 17.26 | False |
| byol_lstm | 15 | 0.01149190 | 0.17803548 | 135168 | 48.51 | False |
| byol_lstm | 50 | 0.04451182 | 0.23855096 | 135168 | 158.86 | False |
| byol_transformer | 5 | 0.00635156 | 0.60722786 | 332032 | 29.19 | False |
| byol_transformer | 15 | 0.02502138 | 0.68768549 | 332032 | 90.23 | False |
| byol_transformer | 50 | 0.07440276 | 0.70379865 | 332032 | 284.97 | False |

## Visuals

- `vae_training_loss.png`
- `vae_loss_components.png`
- `contrastive_training_loss.png`
- `contrastive_embedding_std.png`
- `byol_training_loss.png`
- `byol_embedding_std.png`
