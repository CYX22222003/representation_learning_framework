# Phase 4 Top-80 Walk-Forward Data Analysis

Status: exploratory data feasibility analysis; no model was trained.

## Design

The anchored and fixed-relative-length analyses use identical lifecycle evaluation fractions inside every contract. They differ only in whether the relative training start remains at zero or advances by one evaluation fraction per walk.

| walk | train_start_fraction_anchored | train_start_fraction_window | cutoff_fraction | evaluation_end_fraction | task_train_rows_anchored | task_train_rows_window | task_train_contracts_anchored | task_train_contracts_window | evaluation_rows | evaluation_contracts | task_train_row_ratio_window_to_anchored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0.5 | 0.75 | 18900 | 18900 | 80 | 80 | 12057 | 80 | 1 |
| 2 | 0 | 0.25 | 0.75 | 1 | 30957 | 18891 | 80 | 80 | 11929 | 80 | 0.610234 |

## Shared evaluation-target properties

| scheme | walk | partition | rows | contracts | identity_sha256 | probability_movement_sha256 | probability_movement_mean | probability_movement_std | zero_movement_fraction | stable_tau005_fraction | zero_baseline_mae | zero_baseline_rmse | abs_movement_q50 | abs_movement_q90 | abs_movement_q95 | abs_movement_q99 | return_defined_rows | return_undefined_zero_price_rows | return_defined_fraction | arithmetic_return_median | absolute_return_q50 | absolute_return_q90 | absolute_return_q95 | absolute_return_q99 | absolute_return_gt_1_fraction | absolute_return_gt_10_fraction | near_boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative_anchored | 1 | evaluation | 12057 | 80 | 84dee8a100d9e1b98afc59086ef3d19e6e5fd277e5a74820b841e6642591d43e | f4449c3c88b2caea78df91026fc4bb3ea1572e1b189e5b87d52e3ca07d24a145 | -5.42009e-05 | 0.0182606 | 0.469437 | 0.690139 | 0.00666379 | 0.0182607 | 0.001 | 0.02 | 0.03 | 0.069888 | 12057 | 0 | 1 | 0 | 0.00102459 | 0.2 | 0.333333 | 1 | 0.00306876 | 0 | 0.451439 |
| relative_anchored | 2 | evaluation | 11929 | 80 | 05eceecc4b856d3e9600deb3468f073fa9eefa87464538ac3cc0bad8f54d5913 | 33f554805669299de10daf6ee4c7eea90a117c65c0703039e5076aa8806d5881 | 1.80149e-05 | 0.0144585 | 0.614804 | 0.829323 | 0.00392369 | 0.0144586 | 0 | 0.01 | 0.02 | 0.06 | 11929 | 0 | 1 | 0 | 0 | 0.152605 | 0.333333 | 1 | 0.00385615 | 0 | 0.636432 |

Because the evaluation identities are hash-identical, descriptive differences between the two schemes arise from the relative training allocation rather than different evaluation targets.

## Interpretation boundaries

- Probability movement is the signed probability-point change `close[t+2] - close[t]`.
- Return is the arithmetic ratio of that movement to `close[t]`; it is undefined at zero   and can become extreme near zero.
- Labels use `DOWN < -0.005`, `STABLE` within `[-0.005, 0.005]`, and `UP > 0.005`.
- Training targets must fall strictly before each contract-relative cutoff.
- Relative positions use each contract's final observed length, which is retrospective   unless its termination boundary was known at decision time.
- Relative cutoffs map to different calendar dates across contracts. These results   characterize lifecycle structure and are not a no-leak pooled deployment backtest.

Individual reports:

- [Contract-relative anchored analysis](../relative_anchored/report.md)
- [Contract-relative fixed-length analysis](../relative_window/report.md)
