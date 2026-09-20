# BYOL encoder pretraining sweep

All epoch budgets are reported as a fixed-budget characterization sweep. No checkpoint is selected using test metrics.

| epoch | train BYOL loss | view cosine | embedding std | collapse warning | best loss so far | elapsed seconds |
|---:|---:|---:|---:|:---:|---:|---:|
| 15 | 0.0754657454 | 0.9622671515 | 0.6184719251 | False | 0.0039418837 | 191.52 |
| 20 | 0.0810225011 | 0.9594887715 | 0.7010175591 | False | 0.0039418837 | 250.28 |
| 25 | 0.0796773641 | 0.9601613389 | 0.7765034167 | False | 0.0039418837 | 315.87 |
| 50 | 0.0776185841 | 0.9611907314 | 0.9908381116 | False | 0.0039418837 | 635.14 |
| 100 | 0.0526101089 | 0.9736949717 | 1.3356986524 | False | 0.0039418837 | 1283.12 |
