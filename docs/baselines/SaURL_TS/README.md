# SaURL-TS baseline dossier

**Prepared:** 2026-10-04  \
**Phase:** 6.7 recent frozen-representation comparison  \
**Current decision:** independent paper-guided model plus SaURL-specific
audit, pretraining, frozen-feature, replay, and common-probe infrastructure
implemented and CPU-tested; the unlicensed public source snapshot is audit
evidence only and must not be copied, modified, vendored, or treated as the
specification. The CUDA/resource admission and both walk-specific 50-epoch
pretraining trajectories are complete and replay-valid at 5/15/50.

This directory records the paper reading, architecture reconstruction,
official-code audit, Phase 6.7 adaptation proposal, and implementation gates
for `SaURL-TS-Frozen` (`SAURL-F`). The model is under
`src/baselines/saurl_ts/`; Stage 0--4 orchestration is under `src/training/`,
`src/features/`, and `scripts_v6/`. Both 128-dimensional master stores and all
six staged downstream trajectories are complete and replay-valid at 5/15/50.
The principal result is generally weaker than H0. Near-zero masks and large
embedding norms from pretraining remain disclosed diagnostics rather than
tuning triggers.

## Bottom line

SaURL-TS is a close conceptual baseline for the unified multi-branch framework because it learns a reusable, target-free representation from time, frequency, and cross-domain paths and combines them before lightweight downstream evaluation.

The available repository does not satisfy a direct source-reuse gate:

- the paper is a peer-reviewed 2026 Pattern Recognition article and is CC BY 4.0;
- the public repository has only one branch, no tags, and a latest commit dated 2024-10-16;
- no software licence is present;
- the public extraction path does not implement the paper's representation-wise attention mechanism and contains unresolved references;
- important paper details are not recoverable unambiguously from the paper
  alone.

The project owner therefore approved a documented independent reconstruction
on 2026-10-04. Questions 4--11 in the clarification record now define the
architecture, views, RwAM, alternating update schedule, hyperparameters, and
extraction boundary, including the Stage 1 amendment that uses the paper's
deterministic mask rather than the repository's stochastic sampler. It will
be reported as **SaURL-TS-Frozen (paper-guided
reimplementation)**, never as the official authors' implementation.

## Documents

- [Paper reading note](paper_reading_note.md): contribution, method, experiments, evidence, and limitations.
- [Architecture reconstruction](architecture_and_dataflow.md): tensor-level data flow, objectives, training, and inference boundary.
- [Official-code audit](official_code_audit.md): pinned source inspection, paper/code mismatches, and feasibility verdict.
- [Phase 6.7 integration proposal](phase6_7_integration_proposal.md): fair adaptation to the two Polymarket walks and three shared tasks.
- [Implementation plan](implementation_plan.md): ordered code, test, artifact, and execution gates.
- [Upstream clarification request](upstream_clarification_request.md): questions that would unblock a faithful implementation.
- [Machine-readable source record](source_manifest.json): paper hash, repository commit, licence state, and gate outcome.

## Primary sources

- Yusen Liu et al., “SaURL-TS: A self-adaptive framework for unsupervised time series representation learning,” *Pattern Recognition* 175 (2026), article 113129. DOI: <https://doi.org/10.1016/j.patcog.2026.113129>
- Official repository: <https://github.com/YusenL/SAURL-TS>
- Audited commit: [`96f39fe16646e866b6e6b9131e5216ac021c8950`](https://github.com/YusenL/SAURL-TS/tree/96f39fe16646e866b6e6b9131e5216ac021c8950)
- Governing project plan: [Phase 6.7 external representation baseline plan](../../phase_plan/2026-10-04-phase-6-7-external-representation-baseline-plan.md)
