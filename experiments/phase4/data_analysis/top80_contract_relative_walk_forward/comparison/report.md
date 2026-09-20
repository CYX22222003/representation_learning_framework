# Phase 4 Top-80 Walk-Forward Data Analysis

Status: exploratory data feasibility analysis; no model was trained.

## Design

The anchored and fixed-relative-length analyses use identical lifecycle evaluation fractions inside every contract. They differ only in whether the relative training start remains at zero or advances by one evaluation fraction per walk.

| walk | train_start_fraction_anchored | train_start_fraction_window | cutoff_fraction | evaluation_end_fraction | task_train_rows_anchored | task_train_rows_window | task_train_contracts_anchored | task_train_contracts_window | evaluation_rows | evaluation_contracts | task_train_row_ratio_window_to_anchored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0.5 | 0.666667 | 18900 | 18900 | 80 | 80 | 8041 | 80 | 1 |
| 2 | 0 | 0.166667 | 0.666667 | 0.833333 | 26941 | 18890 | 80 | 80 | 8044 | 80 | 0.701162 |
| 3 | 0 | 0.333333 | 0.833333 | 1 | 34985 | 18893 | 80 | 80 | 7901 | 80 | 0.540031 |

## Shared evaluation-target properties

| scheme | walk | partition | rows | contracts | identity_sha256 | probability_movement_sha256 | probability_movement_mean | probability_movement_std | zero_movement_fraction | stable_tau005_fraction | zero_baseline_mae | zero_baseline_rmse | abs_movement_q50 | abs_movement_q90 | abs_movement_q95 | abs_movement_q99 | return_defined_rows | return_undefined_zero_price_rows | return_defined_fraction | arithmetic_return_median | absolute_return_q50 | absolute_return_q90 | absolute_return_q95 | absolute_return_q99 | absolute_return_gt_1_fraction | absolute_return_gt_10_fraction | near_boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative_anchored | 1 | evaluation | 8041 | 80 | 9a4980a566e43d426a4159fe89e304c988d53db3381c43518733c330d9745186 | f16a4897ea48853eea702222ffd924697c12b6bacc8a563c1c6f3fe360d38ba3 | -6.7616e-05 | 0.0195421 | 0.436389 | 0.669195 | 0.00730037 | 0.0195422 | 0.001 | 0.02 | 0.03 | 0.07 | 8041 | 0 | 1 | 0 | 0.00302725 | 0.21 | 0.333333 | 0.833333 | 0.00310907 | 0 | 0.424698 |
| relative_anchored | 2 | evaluation | 8044 | 80 | 6dbf2c474367223af1f8d39b85965aa2b30db586360ff8aa619dde56a1e1fd17 | 189fa4e38160e62f9a0f3f230222342737afd18b81333d127d6d8d4e848bb258 | -1.87469e-05 | 0.0139283 | 0.558304 | 0.759572 | 0.00478056 | 0.0139283 | 0 | 0.0102 | 0.02 | 0.06 | 8044 | 0 | 1 | 0 | 0 | 0.166667 | 0.333333 | 1 | 0.00298359 | 0 | 0.536176 |
| relative_anchored | 3 | evaluation | 7901 | 80 | 3421043add70dd00e10d93c414395147f3c034f8a59430aa2c5e3df19c9c4824 | 14669f00b987f1945d8f6c976151032bbbebefca1417f03e37d8ae7e2f28aece | 3.23883e-05 | 0.015437 | 0.632072 | 0.850905 | 0.00379622 | 0.0154371 | 0 | 0.01 | 0.02 | 0.06 | 7901 | 0 | 1 | 0 | 0 | 0.148148 | 0.333333 | 1 | 0.00430325 | 0 | 0.671687 |

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
