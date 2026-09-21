# Phase 5 Regression Rank IC Diagnostic (Seed 0, Epoch 50)

Global Rank IC is the Spearman correlation over all saved rows. Finance-style cross-sectional Rank IC is calculated independently at each decision timestamp with at least five active contracts, then summarized across timestamps.

| Task | Walk | Global Rank IC | Mean cross-sectional Rank IC | Median IC | Unannualized ICIR | Positive IC fraction | IC timestamps |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_delta_h2 | 1 | 0.001235 | 0.004586 | 0.004558 | 0.016594 | 0.5049 | 1820 |
| raw_delta_h2 | 2 | -0.005724 | 0.002431 | -0.007749 | 0.006591 | 0.4845 | 1034 |
| raw_delta_h8 | 1 | 0.007261 | 0.007060 | 0.016003 | 0.023842 | 0.5193 | 1814 |
| raw_delta_h8 | 2 | -0.033975 | -0.020646 | -0.036121 | -0.057475 | 0.4528 | 1007 |
| log_return_h2 | 1 | 0.005214 | 0.002252 | -0.002272 | 0.008709 | 0.4940 | 1820 |
| log_return_h2 | 2 | -0.003790 | 0.007119 | 0.000000 | 0.020640 | 0.4971 | 1034 |

## Pooled cross-sectional Rank IC

- `raw_delta_h2`: mean `0.003805`, median `0.000000`, unannualized ICIR `0.012158`, positive fraction `0.4975`, across `2854` timestamps.
- `raw_delta_h8`: mean `-0.002830`, median `-0.000759`, unannualized ICIR `-0.008836`, positive fraction `0.4956`, across `2821` timestamps.
- `log_return_h2`: mean `0.004015`, median `-0.001470`, unannualized ICIR `0.013715`, positive fraction `0.4951`, across `2854` timestamps.

All mean ICs and ICIRs are approximately zero and positive fractions are approximately 50%. The current regression outputs therefore show no stable cross-sectional ranking power. Hourly stride-one rows and targets overlap, so no independent-observation significance claim is made.
