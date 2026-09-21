# Phase 5 Fresh FinData Walk 1 and Walk 2 Exploration

**Date:** 2026-09-21  
**Status:** acquisition-to-plot exploration complete; no training launched;
Phase 5 execution gate remains closed

## 1. Scope and selection boundary

Two independent FinData searches selected 50 Polymarket conditions per walk.
Candle eligibility rows and span were probed only inside the corresponding
training interval, after which the selected conditions were downloaded through
the evaluation end at both native resolutions.

| Walk | Selection/probe interval | Download interval | Contracts |
|---:|---|---|---:|
| 1 | `[2025-12-02, 2026-04-01)` | `[2025-12-02, 2026-06-16)` | 50 |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-02-16, 2026-09-01)` | 50 |

The universes overlap on 21 conditions. Training-local candle probes prevent
evaluation coverage from qualifying a market, but FinData catalog volume,
complete market dates, and heuristic categories remain retrospective inputs.
The cohorts are therefore improved exploratory walk cohorts, not production
cutoff-local universes.

## 2. Raw-to-filled lineage

The user approved the exact forward-confirmed quarantine previews before any
clean files were written. Raw hashes remained unchanged and clean row removal
exactly replays the audit.

| Walk | Resolution | Raw rows | Quarantined | Clean rows | Isolated fills | Filled rows |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 15m | 331,366 | 110 | 331,256 | 36,014 | 367,270 |
| 1 | 1h | 113,574 | 154 | 113,420 | 5,710 | 119,130 |
| 2 | 15m | 345,280 | 17 | 345,263 | 35,896 | 381,159 |
| 2 | 1h | 119,051 | 11 | 119,040 | 5,960 | 125,000 |

Every fill is a complete isolated internal one-bar gap with flat OHLC at the
previous observed close, zero volume, and explicit observation/imputation and
time-since-observation metadata. Leading, trailing, and longer gaps remain
absent. Observed rows, output counts, hashes, and Parquet round trips passed.

Walk 1 has four hourly conditions with contract-local quarantine rates above
2%, despite a pooled hourly rate of only 0.1356%. Walk 2 has no contract above
1%. This strengthens the need for the already required affected-contract
exclusion sensitivity.

## 3. Gap evidence

| Walk | Resolution | Internal coverage | Missing slots | One-bar fills | Longer slots left absent | Maximum run |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 15m | 60.59% | 215,455 | 36,014 | 179,441 | 1,377 bars |
| 1 | 1h | 82.97% | 23,278 | 5,710 | 17,568 | 343 hours |
| 2 | 15m | 54.69% | 286,081 | 35,896 | 250,185 | 4,847 bars |
| 2 | 1h | 75.40% | 38,847 | 5,960 | 32,887 | 1,210 hours |

Walk 2 is materially more sparse. At one hour, 9/50 Walk 1 conditions and
21/50 Walk 2 conditions are below 75% internal coverage; 2/50 and 11/50 are
below 50%. Long gaps must remain sequence breaks and causal activity
eligibility is especially important in Walk 2.

Missingness is not verified inactivity. At least one native 15-minute candle
exists inside 33.30% of Walk 1 and 21.91% of Walk 2 missing hourly slots. The
native resolutions are separate sparse API views rather than interchangeable
resamplings.

## 4. Staleness effect

| Walk | Horizon | Series | Targets | Exact zero | Mean absolute movement | Filled-series paths touching fill |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1h | observed-only clean | 104,248 | 28.70% | 0.003337 | 0.00% |
| 1 | 1h | filled all | 115,668 | 32.21% | 0.003262 | 9.87% |
| 1 | 2h | observed-only clean | 98,089 | 26.13% | 0.003972 | 0.00% |
| 1 | 2h | filled all | 113,181 | 26.39% | 0.004027 | 13.33% |
| 2 | 1h | observed-only clean | 108,306 | 30.76% | 0.003067 | 0.00% |
| 2 | 1h | filled all | 120,226 | 34.41% | 0.002948 | 9.91% |
| 2 | 2h | observed-only clean | 101,538 | 27.36% | 0.003896 | 0.00% |
| 2 | 2h | filled all | 116,916 | 28.42% | 0.003872 | 13.15% |

One-bar filling restores context continuity but raises one-hour exact-zero
movement by about 3.5 percentage points in both walks. It does not add observed
information. Phase 5 must retain the mask and time-since-observation feature,
require observed decision and target endpoints, and report imputation exposure
and untouched-context sensitivity.

## 5. Plots and artifacts

Each walk has 100 replay-verified OHLCV figures: one 15-minute and one one-hour
figure for each condition. The x-axis uses that condition-resolution's actual
first and last clean candle. Isolated fills are marked and longer gaps remain
line breaks.

The Git-ignored source artifacts and detailed reports are under:

- `data_new/findata/polymarket/phase5_walk1_top50_train-2025-12-02_cutoff-2026-04-01_eval-end-2026-06-16/`
- `data_new/findata/polymarket/phase5_walk2_top50_train-2026-02-16_cutoff-2026-06-16_eval-end-2026-09-01/`
- `data_new/findata/polymarket/phase5_walk1_walk2_top50_exploration_2026-09-21.md`

## 6. Handoff judgement

The one-hour resolution remains the primary exploratory source. Walk 1 is the
stronger cohort; Walk 2 remains usable only with stricter activity eligibility,
sequence breaks, and transparent low-coverage/exclusion reporting. Native
15-minute data remains a resolution sensitivity.

This completion does not authorize model training. The next implementation
must replay quarantine decisions only after `quarantine_available_at`, freeze
walk-local selection and preprocessing, enforce target maturity and observed
endpoints, train separate walk-specific encoder/head weights, align every
baseline to identical eligible rows, and emit replayable manifests. The
condition-candle token-identity limitation remains mandatory in all claims.
