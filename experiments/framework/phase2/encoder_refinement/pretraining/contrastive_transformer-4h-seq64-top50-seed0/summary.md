# Phase 2 contrastive encoder pretraining

Training-side fixed-budget diagnostics only; no downstream test result is used for selection.

| epoch | NT-Xent | embedding std | projected std | collapse | parameters | seconds |
|---:|---:|---:|---:|:---:|---:|---:|
| 15 | 2.44306302 | 0.59455711 | 0.08761889 | False | 299008 | 340.40 |
| 50 | 2.38789970 | 0.36682164 | 0.08767137 | False | 299008 | 1145.59 |
| 100 | 2.35529483 | 0.27229752 | 0.08766470 | False | 299008 | 2275.28 |
