---
name: project-design
description: Use when answering questions about this project's architecture, representation branches, embedding dimensions, aggregator modes, downstream tasks, model design, or planned extensions.
---

# Project Design

Ground design answers in the current project documents rather than assumptions.

## Read First

Read these in order:

1. `docs/Research_Ideas_Writeup.md`, especially sections 3.1 through 3.4 and 5.3 through 5.6 when downstream evaluation or alpha research is relevant.
2. The Architecture Design and Representation Learning sections of `docs/design.md`.
3. The Architecture section of `AGENTS.md`, including the branch table, aggregator modes, extension guide, and module responsibilities.

## Response Contract

Present the parts relevant to the request:

- Each current representation branch, source module, and output dimension.
- The concat and gated aggregator modes, their output dimensions, and when each is appropriate.
- Price prediction, volatility prediction, and tri-class trend classification, including their documented metrics and label/target contracts when relevant.
- The additional alpha-research capability: downstream predictions rather than latent dimensions as primitives, shallow symbolic search, and chronological OOF-only formula selection.
- Components, methods, or scope explicitly marked as open, provisional, or dependent on later work.

Prefer dimension utilities and `RepresentationAggregator.output_dim` over hard-coded assumptions. If documents disagree with current source code, call out the discrepancy and inspect the implementation before recommending a change.
