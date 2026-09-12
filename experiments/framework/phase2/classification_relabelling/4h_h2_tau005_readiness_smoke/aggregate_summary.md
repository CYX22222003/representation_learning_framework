# Phase 2 probabilistic classification results

Candidate protocols are P1U, P1O, and P2. P0, when present, is an untreated reference only.

| scope | model | protocol | epoch | seeds | macro-F1 | balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ta_aligned | C1_raw_ohlcv_mlp | P0 | 1 | 1 | 0.2702 ± 0.0000 | 0.3333 ± 0.0000 | 0.5360 ± 0.0000 | 0.3523 ± 0.0000 |
| ta_aligned | C1_raw_ohlcv_mlp | P1O | 1 | 1 | 0.3322 ± 0.0000 | 0.3921 ± 0.0000 | 0.5722 ± 0.0000 | 0.3676 ± 0.0000 |
| ta_aligned | C1_raw_ohlcv_mlp | P1U | 1 | 1 | 0.2743 ± 0.0000 | 0.3531 ± 0.0000 | 0.5441 ± 0.0000 | 0.3581 ± 0.0000 |
| ta_aligned | C1_raw_ohlcv_mlp | P2 | 1 | 1 | 0.3685 ± 0.0000 | 0.3787 ± 0.0000 | 0.5521 ± 0.0000 | 0.3584 ± 0.0000 |
| ta_aligned | C2_framework | P0 | 1 | 1 | 0.4366 ± 0.0000 | 0.4538 ± 0.0000 | 0.7631 ± 0.0000 | 0.4978 ± 0.0000 |
| ta_aligned | C2_framework | P1O | 1 | 1 | 0.4797 ± 0.0000 | 0.4936 ± 0.0000 | 0.7645 ± 0.0000 | 0.4992 ± 0.0000 |
| ta_aligned | C2_framework | P1U | 1 | 1 | 0.4511 ± 0.0000 | 0.4863 ± 0.0000 | 0.7456 ± 0.0000 | 0.4885 ± 0.0000 |
| ta_aligned | C2_framework | P2 | 1 | 1 | 0.4676 ± 0.0000 | 0.4935 ± 0.0000 | 0.7602 ± 0.0000 | 0.4991 ± 0.0000 |
| ta_aligned | C5_ta_mlp | P0 | 1 | 1 | 0.4203 ± 0.0000 | 0.4276 ± 0.0000 | 0.6380 ± 0.0000 | 0.4214 ± 0.0000 |
| ta_aligned | C5_ta_mlp | P1O | 1 | 1 | 0.3217 ± 0.0000 | 0.4606 ± 0.0000 | 0.6266 ± 0.0000 | 0.4120 ± 0.0000 |
| ta_aligned | C5_ta_mlp | P1U | 1 | 1 | 0.3285 ± 0.0000 | 0.4519 ± 0.0000 | 0.6138 ± 0.0000 | 0.4032 ± 0.0000 |
| ta_aligned | C5_ta_mlp | P2 | 1 | 1 | 0.3629 ± 0.0000 | 0.4598 ± 0.0000 | 0.6279 ± 0.0000 | 0.4159 ± 0.0000 |
