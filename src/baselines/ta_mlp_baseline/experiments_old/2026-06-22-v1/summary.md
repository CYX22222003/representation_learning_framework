# TA-MLP Baseline — v1 Triclass Sweep Cross-Run Summary

**Date:** 2026-06-22
**Mode:** triclass (BUY / HOLD / SELL), seed=0
**Epoch budgets:** `[15, 20, 25, 50, 100]`
**Labeling params:** `b_window=5`, `f_window=2`, `hold_q=0.85`, `buy_sell_q=0.997`
**Threshold fit:** per contract, on training rows only

This document compares the five v1 runs. Per-run details are in each
`experiments/2026-06-22-v1/e<N>/summary.md`.

> **Framing — read first.** The sweep is for *characterization*, not for
> picking a "best" run. Choosing an epoch count by reading test metrics
> would convert the test set into a tuning set. All five runs are reported
> together below; do not cite a single one as the headline.

## Dataset

| Item | Value |
|---|---|
| Source | `four_hour_file_list` (same files as the LSTM baseline) |
| Train rows | 64,948 |
| Test rows | 16,258 |
| Input dim | 36 |
| Train class distribution | BUY 6,803 (10.5%) / HOLD 51,763 (79.7%) / SELL 6,382 (9.8%) |
| Test class distribution  | BUY 1,669 (10.3%) / HOLD 12,139 (74.7%) / SELL 2,450 (15.1%) |

The HOLD class dominates by design — the labeling formula tags anything
inside the `[alpha, beta]` band as HOLD, and `alpha = 0.85`-quantile of
`|pct_change|` pulls ~80% of moves into that band. Accuracy is therefore
**not** a meaningful headline metric on its own; a model that always
predicts HOLD scores ~75–80% accuracy without learning anything.

## Headline results

| Epochs | Final train CE | Test accuracy | Macro-F1 | F1 BUY | F1 HOLD | F1 SELL |
|---|---|---|---|---|---|---|
| 15  | 0.5442 | 0.7116 | 0.4546 | 0.2974 | 0.8625 | 0.2039 |
| 20  | 0.5317 | 0.7256 | 0.4524 | 0.3029 | 0.8647 | 0.1897 |
| 25  | 0.5200 | 0.7115 | 0.4651 | 0.2936 | 0.8587 | 0.2429 |
| 50  | 0.4692 | 0.7118 | 0.4570 | 0.2881 | 0.8542 | 0.2285 |
| 100 | 0.4099 | 0.7065 | 0.4595 | 0.2674 | 0.8457 | 0.2654 |

**Defensible quoted range:** accuracy 0.71–0.73, macro-F1 0.45–0.47.

## Seed determinism check

Same seed, different epoch budgets → identical train trajectory through the
minimum-shared epoch. Verified:

| Epoch | e15 | e20 | e25 | e50 | e100 |
|---|---|---|---|---|---|
| 0  | 0.6200 | 0.6200 | 0.6200 | 0.6200 | 0.6200 |
| 5  | 0.5750 | 0.5750 | 0.5750 | 0.5750 | 0.5750 |
| 10 | 0.5556 | 0.5556 | 0.5556 | 0.5556 | 0.5556 |
| 14 | 0.5442 | 0.5442 | 0.5442 | 0.5442 | 0.5442 |

Identical to 4dp at every checkpoint. Reproducibility infrastructure works
as intended.

## Observations

1. **Train CE keeps decreasing monotonically** across the full sweep
   (0.54 at e15 → 0.41 at e100). No plateau in this range, unlike the
   LSTM regression baseline where loss plateaus around epoch 10–15.
2. **Test accuracy and macro-F1 are flat in the noise band.** Despite
   train loss falling by 0.13 between e15 and e100, accuracy moves only
   0.005 and macro-F1 only 0.013. The model is fitting harder to training
   data without generalizing further — characteristic mild overfitting.
3. **Class imbalance dominates the picture.** HOLD F1 is consistently
   ~0.85–0.87 (it's 80% of the data), while BUY F1 sits at ~0.28–0.30 and
   SELL F1 at ~0.19–0.27. The headline numbers are mostly tracking HOLD
   performance.
4. **Minority-class trade-off across budgets.** A real, if small, effect:
   SELL recall climbs from 0.137 (e15) → 0.211 (e100), and SELL F1 from
   0.204 → 0.265. BUY behavior is noisier — BUY recall actually drops from
   0.43 (e15) to 0.31 (e100). Longer training appears to redistribute the
   model's error away from "predict everything as HOLD" toward catching
   more minority-class cases, but it does not improve overall macro-F1.
5. **The hardest class to recall is SELL at low budgets** (recall 0.12 at
   e20). Most test SELL examples get classified as HOLD. This is consistent
   with the labeling asymmetry: a single quantile band is symmetric in
   `|pct_change|` but `beta * (1 + f_window * 0.1)` widens the cap, which
   affects the BUY/SELL boundaries asymmetrically when forward returns are
   skewed.

## Conclusion

The TA-MLP baseline reaches ~71–73% accuracy and ~0.45–0.47 macro-F1 on the
4-hour triclass task across the full epoch sweep. None of the five training
budgets is materially better than the others on test metrics — the
differences between runs are within the noise band expected from
single-seed minority-class evaluation. The model continues to fit training
data past epoch 100 without test improvement, so longer training is not
useful for this baseline. The numbers should be reported as a range, not
as a single hand-picked run.
