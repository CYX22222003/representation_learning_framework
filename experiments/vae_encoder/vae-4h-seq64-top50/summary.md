# VAE encoder pretraining sweep

All epoch budgets are reported as a fixed-budget characterization sweep. No checkpoint is selected using test metrics.

| epoch | total loss | reconstruction MSE | KL divergence | best total loss so far | elapsed seconds |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.0999306878 | 0.0699566867 | 0.0299740011 | 0.0809418668 | 19.96 |
| 20 | 0.0865633022 | 0.0669389362 | 0.0196243661 | 0.0809418668 | 26.10 |
| 25 | 0.0841178878 | 0.0659496934 | 0.0181681944 | 0.0809418668 | 32.58 |
| 50 | 0.0827743671 | 0.0647486225 | 0.0180257445 | 0.0809418668 | 64.44 |
| 100 | 0.0801516505 | 0.0625848766 | 0.0175667738 | 0.0800388432 | 125.33 |
