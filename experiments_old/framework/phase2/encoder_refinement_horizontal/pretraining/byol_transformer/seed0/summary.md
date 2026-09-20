# Phase 2 temporal BYOL encoder pretraining

Training-only diagnostics; no downstream test result is used for selection.

| epoch | BYOL loss | view cosine | embedding std | collapse | parameters | seconds |
|---:|---:|---:|---:|:---:|---:|---:|
| 15 | 0.08043442 | 0.95978281 | 0.86034750 | False | 332032 | 416.95 |
| 50 | 0.10592513 | 0.94703745 | 0.68714013 | False | 332032 | 1349.68 |
| 100 | 0.07966422 | 0.96016791 | 0.51185101 | False | 332032 | 2669.97 |
