# Phase 6 volatility horizon audit

This is the Stage A, training-period-only audit. It does not freeze a primary horizon, construct a model-ready label bundle, or authorize model training. Evaluation outcomes were not calculated or used; only evaluation row and contract capacity is reported.

Target: `phase6-future-realised-variance-raw-probability-v1` — `RV(t,H) = sum((p[t+j] - p[t+j-1])^2, j=1..H)`.

## Candidate summary

| Walk | H | Train rows | Train contracts | Eval rows | Eval contracts | Train zero rate | Largest interval share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 36,985 | 47 | 29,970 | 23 | 0.1019 | 0.0479 |
| 1 | 4 | 35,256 | 47 | 29,168 | 23 | 0.0283 | 0.0303 |
| 1 | 8 | 32,470 | 46 | 27,806 | 23 | 0.0066 | 0.0186 |
| 1 | 24 | 25,685 | 41 | 23,882 | 23 | 0.0019 | 0.0088 |
| 2 | 2 | 56,827 | 43 | 13,614 | 27 | 0.1092 | 0.0929 |
| 2 | 4 | 55,443 | 43 | 13,047 | 26 | 0.0296 | 0.0469 |
| 2 | 8 | 53,112 | 43 | 12,115 | 25 | 0.0043 | 0.0263 |
| 2 | 24 | 46,295 | 41 | 9,806 | 23 | 0.0000 | 0.0101 |

## Interpretation gate

Choose the primary horizon only after reviewing capacity in both walks, training-target degeneracy and concentration, dependence, and the reporting strata. The choice must be recorded in a dated amendment before label-bundle construction. This report intentionally does not nominate or freeze that horizon.

Drop counts in `horizon_capacity.csv` are diagnostic flags and can overlap (for example, a missing future timestamp is also a gap-segment break).
