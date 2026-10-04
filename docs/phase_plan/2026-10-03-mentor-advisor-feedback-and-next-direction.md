# Mentor and Advisor Feedback and Next Research Direction

**Record date:** 2026-10-03  
**Status:** Feedback record. Its proposed prioritisation is now authorized by
`2026-10-04-phase-6-7-external-representation-baseline-plan.md`, which defers
Phase 6.6, preserves Phase 7A as the subsequent branch-analysis phase, and
freezes the recent external representation comparison as the immediate next
experimental priority.

> **Later roster amendment (2026-10-04):** This document preserves the original
> candidate discussion below. The execution authority now makes both LWA and
> SaURL-TS mandatory. Independently, SISSEL and TimeDART are peer optional
> post-core extensions considered after the complete required comparison is
> reviewed with mentors.

## Purpose

This note records the main feedback received from the PhD mentor and project
advisor, explains its implications for the representation-learning claim, and
identifies the immediate research direction. The feedback dates were not
separately recorded here; the record is based on the messages supplied on
2026-10-03.

## Feedback Received

### Architecture Rationale and Branch Meaning

The PhD mentor observed that the statistical, transformed, VAE, contrastive,
and BYOL branches currently appear more like a collection of different methods
than a theoretically justified architecture. The project should explain:

- why these five branches were selected;
- which market structure or information each branch is expected to capture;
- why the branches should be complementary rather than repetitive; and
- whether each branch learns its intended information, not only whether
  downstream performance falls when it is removed.

The planned single-branch and leave-one-branch-out ablations are therefore
necessary but not sufficient by themselves. They should be accompanied by
diagnostics that test the intended role of each representation.

### Prior Art and External Baselines

Both the PhD mentor and project advisor identified the external-comparison
scope as the most important weakness. The current work contains many internal
variants and several raw-input or task-specific baselines, but the main claim
concerns reusable representations. It therefore needs direct comparison with
recent representation-learning methods under the same downstream protocol.

The advisor specifically requested strong baselines from this year or the
previous year's top conferences. Where a conference method cannot be adapted
fairly to the project's tasks and data, the advisor accepted a closely matched
method from a reputable JCR Q1 or Q2 journal. This supports relevance and
comparability over adding a nominally strong but mismatched model.

## Interpretation for the Research Claim

The immediate issue is not a lack of additional architecture variants. It is
whether the current evidence identifies the contribution of the five-branch
representation and establishes its value relative to the closest prior work.
Raw MLP, Raw LSTM, GARCH--LSTM, TA-MLP, and complete forecasting systems remain
useful task-level references, but they do not replace a direct representation-
learning comparison.

The branch rationale should be treated as a set of testable hypotheses:

| Branch | Intended information | Evidence needed beyond downstream accuracy |
|---|---|---|
| Statistical | Linear dependence, autocorrelation, shock persistence, and conditional-variance regimes through AR/GARCH summaries | Association with autocorrelation and volatility diagnostics; single-branch and leave-one-out results |
| Transformed | Global periodic or spectral structure through FFT and localized multiscale changes through Haar wavelets | Spectral-energy and scale-localization diagnostics; comparison with the statistical branch |
| VAE | Reconstructive compression that preserves broad window-level state | Reconstruction diagnostics and tests of which observable window properties remain decodable |
| Contrastive | Discriminative features stable under the declared augmented views | View-consistency tests, sensitivity to genuine market changes, and downstream probes |
| BYOL | A non-contrastive bootstrap alternative learned from the same view construction | Collapse and view-consistency diagnostics plus matched comparison with the contrastive branch |

Contrastive learning and BYOL differ in their learning objectives, but they do
not automatically encode different invariances when they use the same
augmentations. Their complementarity must therefore be demonstrated rather
than assumed.

Linear CKA and Gaussian/RBF-kernel CKA can describe similarity between branch
representations. Low similarity does not prove usefulness, while high
similarity does not prove that one branch is dispensable. CKA should be
reported together with task performance, single-branch and leave-one-out
effects, and the targeted diagnostics above.

## Candidate Prior Work

The initial literature search has identified two recent top-conference methods
that are closer to the project's core claim than another raw forecasting
architecture:

- [TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](https://proceedings.mlr.press/v267/wang25r.html), ICML 2025, is a recent self-supervised representation-learning candidate. Its training and evaluation assumptions must be audited before it is admitted to the matched protocol.
- [Learning Without Augmenting: Unsupervised Time Series Representation Learning via Frame Projections](https://openreview.net/forum?id=LwPjJHVWSn), NeurIPS 2025, is especially relevant because it learns across time, Fourier, and time-frequency views and evaluates frozen representations. Its published tasks differ from the present financial tasks, so feasibility and adaptation boundaries must be stated explicitly.
- SaURL-TS, published in *Pattern Recognition* in 2026, is the selected recent
  journal complement because its adaptive time/frequency representation can be
  frozen and tested through the common probes.
- SISSEL, published in *Information Fusion* in 2026, is a separate optional
  scale-independent representation candidate for possible post-core review.

TimeMixer++ remains useful to review as a recent complete-system forecasting
comparison, but it should not be presented as the main frozen-representation
baseline unless its encoder can be evaluated under an equivalent probing
contract. The same distinction applies to the already planned xLSTM-Mixer
candidate: it evaluates a forecasting system, not reusable representation
quality in isolation.

## Proposed Next Direction

The dated Phase 6.7 contract now adopts the following order:

1. Complete a focused prior-art table covering objective, input assumptions,
   frozen or fine-tuned evaluation, supported downstream tasks, code
   availability, and adaptation risk.
2. Freeze the required LWA/SaURL-TS roster under the same walk-specific
   training histories, task rows, targets, train-only preprocessing,
   lightweight downstream heads,
   epoch rule, and metrics. Consider SISSEL and TimeDART independently as
   optional post-core extensions after mentor review. Any unavoidable
   source-specific advantage must be disclosed.
3. Implement and run the selected external representation baselines before
   expanding the project's architecture further.
4. Execute the already planned Phase 7A single-branch and leave-one-out matrix,
   adding predeclared branch-role diagnostics and CKA as descriptive evidence.
5. Reassess the value of the unimplemented Phase 6.6 fusion and decoder studies
   after the prior-art comparison and branch evidence are available.

Phase 6 and the frozen Phase 6.5 studies are complete. Phase 6.7 is approved
but unimplemented. Phase 6.6 and Phase 7A remain unimplemented, and Phase 7B
alpha research remains deferred. This note changes no existing result or
execution status; the Phase 6.7 plan is the execution authority.
