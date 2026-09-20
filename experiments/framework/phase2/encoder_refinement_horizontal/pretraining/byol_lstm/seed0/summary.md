# Phase 2 temporal BYOL encoder pretraining

Training-only diagnostics; no downstream test result is used for selection.

| epoch | BYOL loss | view cosine | embedding std | collapse | parameters | seconds |
|---:|---:|---:|---:|:---:|---:|---:|
| 15 | 0.06221632 | 0.96889186 | 0.17903047 | False | 135168 | 246.66 |
| 50 | 0.06982123 | 0.96508940 | 0.19576417 | False | 135168 | 807.63 |
| 100 | 0.05450169 | 0.97274918 | 0.18058631 | False | 135168 | 1684.47 |
