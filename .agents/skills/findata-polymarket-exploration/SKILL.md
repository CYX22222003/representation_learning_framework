---
name: findata-polymarket-exploration
description: Use when collecting, quarantining, cleaning, gap-auditing, bounded-forward-filling, plotting, or measuring staleness in native 15-minute and one-hour Polymarket condition candles from the lab FinData API. This is a data-exploration SOP, not a model-training or non-Polymarket acquisition skill.
---

# FinData Polymarket Exploration

Follow the raw-to-derived lineage and mandatory review gate in
[references/sop.md](references/sop.md). Read that reference in full before
collecting or mutating any cohort artifact.

Before changing the anomaly or filling rules, also read these evidence records
in full and reconcile the proposed change with them:

- `docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md`
- `docs/data_analysis/2026-09-21-findata-native-15m-1h-dynamics.md`
- `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`

## Current Operating Scope

- Source: FinData prediction-market endpoints for Polymarket only.
- Fixed acquisition interval: `[2025-12-01 00:00 UTC, 2026-09-01 00:00 UTC)`.
- Native resolutions: both 15 minutes and one hour.
- Required user input: desired contract count.
- Optional user input: search keywords and event-family preferences or exclusions.

The analyst may choose a reproducible candidate limit, diversity rule, and
event-family cap appropriate to the requested count. Declare those choices
before acquisition. Do not silently ignore a requested keyword or family
filter merely because the current collector lacks a corresponding flag.

If the user later changes the date range, source, venue, or resolution, confirm
the new scope and update this skill rather than assuming the current SOP still
applies unchanged.

## Non-Negotiable Boundaries

- Preserve raw downloads immutably. Never invert, repair, fill, or overwrite them.
- Preview and summarize quarantine candidates, then stop for explicit user
  approval before producing clean/pruned candles.
- Keep three distinct artifact layers: raw, post-quarantine clean, and
  clean-plus-bounded-fill derived data.
- Treat condition-candle quarantine as an exploratory anomaly filter, not proof
  of canonical YES-token identity.
- Report evidence and limitations. The user makes the later judgement about
  suitability for training or other research use.
- Run repository Python commands from the project root with `.venv/bin/python3`.

## Completion Contract

Do not call the SOP complete unless the handoff includes:

- the exact acquisition and selection configuration;
- raw, clean, and filled artifact paths with hashes and row counts;
- quarantine rates and affected-contract concentration;
- gap and bounded-fill exposure at both native resolutions;
- filled-series staleness beside clean observed-only staleness;
- one OHLCV figure per contract and resolution over its actual observed range;
- replayable manifests and commands; and
- unresolved source, script, or data-quality limitations without a premature
  training-readiness verdict.
