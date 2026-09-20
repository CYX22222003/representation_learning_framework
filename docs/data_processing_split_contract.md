# Raw-to-Sequence Split and Preprocessing Contract

Date: 2026-09-20
Status: Required correction; Phase 2 execution paused

Tracking issue: [#15 — Rebuild split-safe data pipeline before resuming Phase 2](https://github.com/CYX22222003/representation_learning_framework/issues/15)

## Purpose

This document records a Phase-2-wide validity defect discovered in the shared
data pipeline and defines the required replacement contract. Existing raw
artifacts, checkpoints, feature bundles, predictions, and reports must be
preserved for reproducibility. They are historical characterisation evidence,
not leakage-free confirmatory evidence.

## Legacy pipeline and defect

The current implementation in `src/data_processing/data_processing.py` applies
the following order independently to each contract:

```text
complete raw contract
-> forward-fill/interpolate the complete contract
-> fit volume mean/std on the complete contract
-> generate all stride-one length-64 windows
-> split the windows chronologically 80/20
```

This ordering has two consequences:

1. volume normalisation for training rows uses statistics influenced by the
   later test period; and
2. the last training window and first test window share 63 of 64 raw
   observations on the current length-64 setup.

The audit confirmed both properties for all 50 contracts in the active 4-hour
dataset. The first property is direct fitted-preprocessing leakage. The second
means that the documented 80/20 boundary is a boundary between generated
windows, not a clean raw-observation holdout. Whether historical context may be
carried into the first test window is a modelling choice, but it must be
explicit and must not allow test-period observations or fitted statistics to
influence training.

## Required replacement order

The corrected pipeline must predeclare a per-contract raw-time boundary before
any fitted preprocessing or sample construction:

```text
complete raw contract
-> establish chronological raw train/test boundary
-> fit every imputer/scaler on permitted training history only
-> apply the frozen preprocessing rule causally
-> construct train and test windows under an explicit context policy
-> construct task targets independently within contract and split
-> merge eligible rows across contracts
```

The test-context policy must state whether the first test prediction may use
historical observations from immediately before the boundary. If permitted,
those observations are context only: no test-period value may enter a training
input, training target, scaler, imputer, encoder update, or selection rule.

## Required provenance and tests

The replacement processed bundle must record:

- raw boundary timestamp/index for every contract;
- raw indices or timestamps covered by every generated window;
- preprocessing parameters and the exact training prefix used to fit them;
- the train/test context policy;
- source-file hashes and contract ordering; and
- processed row identities that downstream label bundles can replay.

Tests must fail when:

- a fitted statistic reads any test-period observation;
- a training window or target contains a test-period observation;
- a horizon crosses a contract or split endpoint;
- reconstruction from the manifest changes row identities; or
- framework and baseline samples differ under a claimed strict comparison.

## Artifact and execution policy

- Do not delete or overwrite existing artifacts.
- Do not continue Phase-2 training or downstream evaluation on the legacy
  processed or feature bundles.
- Rebuild processed data first, then retrain the neural encoders and regenerate
  deterministic/neural feature bundles because they inherit the old inputs.
- Regenerate price, volatility, and classification label/alignment bundles from
  the corrected identities.
- Rerun only the predeclared comparisons needed for the revised Phase-2 scope.
