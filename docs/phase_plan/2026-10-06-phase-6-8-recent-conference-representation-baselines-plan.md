# Phase 6.8 Recent Conference Representation Baseline Extension

**Date:** 2026-10-06
**Status:** Approved and frozen at the roster/comparison-contract level;
implementation and execution have not started. Method-specific source,
licence, adaptation, extraction, and compute contracts must be frozen before
coding or training.
**Predecessor:**
`2026-10-04-phase-6-7-external-representation-baseline-plan.md`
**Next planned phase:**
`2026-09-26-phase-7a-representation-ablation-plan.md`

## 1. Decision and priority

Phase 6.7 remains closed. Phase 6.8 is a separate extension that adds two
recent top-conference representation-learning comparators in response to the
research-track requirement for recent strong baselines:

1. **Di-COT-Frozen** from *Divide and Contrast: Learning Robust Temporal
   Features without Augmentation* (ICML 2026); and
2. **Monotone-VI-Frozen** from *Nonlinear Sequence Data Embedding by Monotone
   Variational Inequality* (ICLR 2025).

The project contribution is representation learning, so both methods are
evaluated as frozen representations through the established common probes.
This phase does not turn into a search for the best task-specific forecasting
architecture. Di-COT-Frozen and Monotone-VI-Frozen are both recent
representation-learning baselines. Their inclusion must not be described as
proof of empirical state-of-the-art performance.

Phase order is now:

```text
closed Phase 6.7 -> Phase 6.8 -> Phase 7A -> reassess deferred Phase 6.6
```

Phase 7A's already frozen branch-ablation matrix is unchanged; only its
execution order moves behind Phase 6.8. Phase 6.6 and Phase 7B remain deferred.

## 2. Research questions

Phase 6.8 asks:

1. How does canonical `H0` compare with two additional recent conference
   representations when downstream rows, probes, losses, budgets, and metrics
   are held fixed?
2. Are any differences consistent across movement classification, future
   price, future realised variance, and both calendar walks?
3. What representation width, fitting time, inference time, memory, and
   adaptation cost accompany each result?

The phase does not claim a comprehensive SOTA survey, an exact reproduction
of either source paper, profitable alpha, or universal superiority.

## 3. Frozen roster and comparison role

| ID | Publication | Role in this project |
|---|---|---|
| `DICOT-F` | Di-COT, ICML 2026 | Recent representation-learning baseline |
| `MVI-F` | Monotone-VI sequence embedding, ICLR 2025 | Recent representation-learning baseline |

Both methods are mandatory Phase 6.8 entries. Existing `H0`, LWA-Frozen,
SaURL-TS-Frozen, and TimeDART-Frozen results remain immutable comparison
evidence. The six `H0` task/walk results are the direct primary references;
the three completed Phase 6.7 external methods may be joined into the final
expanded table without retraining.

The source PDFs supplied for the planning decision are:

- `E:\zotero\storage\YBQ23E4V\Shamba et al. - 2026 - Divide and Contrast Learning Robust Temporal Features without Augmentation.pdf`
- `E:\zotero\storage\ZMML77B2\Zhou and Xie - 2025 - NONLINEAR SEQUENCE DATA EMBEDDING BY MONO- TONE VARIATIONAL INEQUALITY.pdf`

The source audit must record stable paper identities and official-code commits
rather than depend on these workstation paths for experiment replay.

## 4. Stage A: source, licence, and adaptation freeze

No implementation or model fitting begins until a read-only dossier exists
for each method and records:

- paper, official-source repository, stable commit, dependency versions, and
  software licence or explicit absence of one;
- whether source code can be reused, must be wrapped, or requires an
  independently authored implementation;
- source input assumptions, normalization, channel handling, sequence-length
  constraints, representation granularity, and downstream extraction path;
- exact mapping from the project's `[B,64,5]` OHLCV contexts to the method;
- target-free fitting population and proof that no downstream label enters
  representation fitting or method selection;
- exact frozen row representation, pooling/flattening rule, native output
  width, and any components discarded after fitting;
- source-derived architecture or solver settings, random seed handling,
  stopping/budget rule, and checkpoint semantics;
- CPU shape/finite-output smoke tests and a bounded CUDA resource test on
  training-only rows; and
- every deviation that makes the result an adaptation rather than an exact
  reproduction.

An absent or incompatible source licence prohibits copying the source; it does
not by itself prohibit an independently authored paper-based implementation.
The dossier must state the selected boundary. If a method cannot provide one
finite, target-free embedding for every established row, it fails admission
and requires an owner-approved amendment; it may not be silently replaced or
allowed to reduce the comparison population.

The source papers' evaluation metrics may motivate the initial adaptation,
but Phase 5/6 evaluation targets and metrics may not select an extraction
point, hyperparameter, solver setting, representation width, or retry policy.

## 5. Shared data and leakage contract

Reuse the accepted one-hour global-calendar walks:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Each method is fitted independently for each walk on the unchanged target-free
encoder population and consumes the saved 64-by-5 OHLCV contexts. Any scaler,
normalizer, partition statistic, basis, kernel setting, or other fitted
preprocessing uses only the permitted walk-training population and is frozen
before evaluation-row inference.

The downstream tasks and identities are unchanged:

| Task | Frozen target/row contract |
|---|---|
| Movement classification | two-hour `DOWN/STABLE/UP`, `tau=0.001` |
| Future price | absolute `close[t+8]`, with implied-movement diagnostics |
| Future realised variance | raw probability-change RV over `(t,t+8h]` |

Target maturity, observed endpoints, and segment continuity are applied only
when aligning embeddings to the existing supervised bundles. Every method
must cover the exact ordered rows used by `H0` within a task/walk comparison.

There is no validation split, early stopping on evaluation data, best-on-test
checkpoint selection, evaluation-driven restart, or later-walk update of an
earlier-walk representation.

## 6. Stage B: representation fitting

The mandatory fitting inventory is:

```text
2 methods x 2 walks = 4 new representation-fitting trajectories
```

Use seed `0` for the initial characterization. Each method/walk has one
predeclared uninterrupted fitting budget. The Stage A dossier must map the
source method's native optimization semantics to a fixed budget before
execution. A method that is not naturally epoch-based must not be forced into
misleading 5/15/50 encoder checkpoints; instead, its dossier freezes
scientifically meaningful intermediate/final checkpoints and one principal
representation checkpoint. This exception does not alter the downstream
5/15/50 probe contract.

Every trajectory preserves configuration, source identity, row/data hashes,
seed, fitted preprocessing, checkpoints or solver states, objective history,
parameter/state size, elapsed time, peak memory, and replay metadata.
Non-convergence, collapse, or non-finite behavior is a reportable result, not
permission for evaluation-guided retuning.

## 7. Stage C: frozen stores and common probes

The principal representation checkpoint from each trajectory produces:

```text
2 methods x 2 walks = 4 new native-width representation stores
```

Each store records ordered identities, checkpoint/solver-state hash, source
bundle hashes, output width, dtype, finiteness, extraction settings, and
content hash. The fitted representation receives no downstream gradient.

For every task and walk, reuse the canonical `H0` head family, task loss,
output transform, train-only coordinate scaling, optimizer, batch size,
learning rate, seed, epoch budget, reference predictions, and metric
implementation. No method-specific supervised bottleneck is added. Native
representation widths and the resulting probe parameter counts are reported.

The new downstream matrix is:

```text
2 representations x 3 tasks x 2 walks = 12 new trajectories
12 trajectories x 3 downstream snapshots = 36 evaluated snapshots
```

Downstream probes retain epochs 5, 15, and 50, with epoch 50 fixed as the
principal result. Six immutable `H0` task/walk trajectories form the direct
18-entry comparison table. The final report should also include the completed
LWA, SaURL, and TimeDART epoch-50 results as clearly labelled prior-phase
context, joined by exact task/walk identity rather than rerun.

## 8. Evaluation and fairness

Report walk-specific results first and pooled results second. For each
task/walk show absolute values and paired differences from immutable `H0` on
identical ordered rows.

- Classification: accuracy, macro-F1, balanced accuracy, per-class metrics,
  confusion matrix, and class-collapse diagnostics.
- Future price: MAE, RMSE/MSE, Pearson/Spearman, persistence skill, implied-
  movement correlation/sign agreement, global Rank IC, timestamp-level
  cross-sectional Rank IC, and existing subgroup diagnostics.
- Volatility: MAE, RMSE/MSE, Pearson/Spearman, persistence skill, contract-
  macro results, and existing tail/subgroup diagnostics.

For every method report native width, representation-state/parameter size,
probe parameters, fitting and extraction time, downstream time, inference
time, peak memory, implementation origin, and all documented adaptations.

Overlapping stride-one rows invalidate ordinary row-wise IID significance
tests. Uncertainty intervals require a separate predeclared contract/calendar
block procedure. The initial two-walk, seed-0 matrix is characterization
evidence only.

## 9. Implementation and execution boundary

Planned code and artifacts belong under:

```text
src/baselines/{dicot,monotone_vi}/
scripts_v7/
tests/baselines/{dicot,monotone_vi}/
experiments/phase6_8/
  feasibility/
  manifests/
  representation_fitting/{dicot_frozen,monotone_vi_frozen}/
  features/walk{1,2}/
  downstream/{classification_h2,absolute_price_h8,realised_variance}/
  reports/recent_conference_representation_seed0/
```

Reusable logic belongs under `src/`; `scripts_v7/` contains thin orchestration.
Launchers are manifest/audit-only by default and require an explicit
`--execute` flag for fitting or downstream training. Canonical Phase 5/6 and
closed Phase 6.7 artifacts are immutable.

Execution uses the persistent Lumid Sandbox rather than an ephemeral FlowMesh
workflow because the current workflow cannot reliably persist and reload the
required artifacts across stages. Workflow/Docker demonstrations remain
separate infrastructure evidence and are not the authoritative experiment
store.

## 10. Ordered gates

1. Replay the two walk bundles, two target-free encoder populations, six task
   populations, and six immutable `H0` references.
2. Complete both source/licence/adaptation dossiers without using downstream
   evaluation metrics.
3. Freeze the complete four-fit, four-store, 12-probe inventory, budgets,
   paths, hashes, and failure policy.
4. Implement both methods and pass focused CPU tests plus bounded
   training-only CUDA/resource admission.
5. Fit and replay all four walk-specific representation trajectories.
6. Extract and replay all four complete native-width stores.
7. Freeze the 18-entry direct comparison manifest: 12 new trajectories plus
   six immutable `H0` references.
8. Execute all 12 probes without result-based truncation and retain every
   5/15/50 snapshot.
9. Replay checkpoints, predictions, metrics, scalers, ordered identities,
   representation hashes, and resource records.
10. Generate the complete per-walk and pooled report, then join prior Phase
    6.7 results as immutable contextual rows.

## 11. Exit conditions and handoff

Phase 6.8 is complete only when:

- both dossiers freeze paper/source identity, licence boundary, adaptation,
  extraction, budget, and resource admission;
- all four representation-fitting trajectories and four feature stores pass
  their method-appropriate replay checks;
- all 12 new downstream trajectories and 36 snapshots pass standalone
  prediction and metric replay;
- each task/walk comparison uses the exact ordered `H0` rows and preserves the
  full predeclared matrix, including negative results;
- native widths, probe capacity, resources, and adaptation limitations
  accompany performance; and
- the report maintains the two-walk, seed-0, non-trading, adaptation, and
  non-universal claim boundaries.

After these conditions are met, Phase 7A becomes the active handoff and runs
its unchanged canonical single-branch and leave-one-branch-out matrix. The
project should reassess Phase 6.6 only after both the expanded external
representation comparison and the internal branch analysis are available.
