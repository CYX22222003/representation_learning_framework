# Phase 2 contrastive encoder pretraining

Training-side fixed-budget diagnostics only; no downstream test result is used for selection.

| epoch | NT-Xent | embedding std | projected std | collapse | parameters | seconds |
|---:|---:|---:|---:|:---:|---:|---:|
| 15 | 2.46867187 | 0.23053039 | 0.08752129 | False | 102144 | 201.82 |
| 50 | 2.32081228 | 0.20486092 | 0.08755763 | False | 102144 | 697.01 |
| 100 | 2.26648466 | 0.18665211 | 0.08769788 | False | 102144 | 1389.11 |
