---
name: progress-logging-report
description: Use when creating or updating concise dated progress logging reports under progress_summary/ for supervisor submission. The report should summarize current project progress from docs/schedule.md, docs/research_plan.md, docs/Research_Ideas_Writeup.md, recent git commits, checkpoints, and experiment artifacts, while keeping the submitted document brief, action-first, and non-technical enough for busy supervisors.
---

# Progress Logging Report

Generate a short supervisor-submission report from the internal progress sources. Keep this separate from the working documentation branch:

- Internal working sources: `docs/schedule.md`, `docs/research_plan.md`, `docs/Research_Ideas_Writeup.md`, commits, checkpoints, and experiment artifacts.
- Submission output: `progress_summary/progress_logging_report_YYYY-MM-DD.md`.

## Workflow

1. Read `docs/schedule.md` first. Treat it as the current progress source of truth.
2. Read `docs/research_plan.md` as the stable roadmap.
3. Read relevant sections of `docs/Research_Ideas_Writeup.md` only if the report needs motivation or framing.
4. Inspect recent evidence:
   - `git log --date=short --pretty=format:'%h %ad %s' -12`
   - `git status --short`
   - `find checkpoints -maxdepth 2 -type f | sort`
   - relevant experiment summaries under `experiments/` and `src/baselines/*/experiments/`
5. Write or update the dated report in `progress_summary/`.

Use commit history and artifacts to verify the report, but do not include a commit-evidence table unless the user explicitly asks.

## Report Shape

Use this structure by default:

```markdown
# Progress Logging Report - YYYY-MM-DD

**Project:** ...
**Current phase:** ...
**Primary sources:** ...

## 1. Immediate Next Actions

1. ...

## 2. Current Status

Brief paragraphs only.

## 3. Completed Progress

- ...
```

Do not include a separate "Supervisor-Facing Summary" section. Do not include a long "Not Yet Completed" section. If pending work matters, mention it briefly in Current Status or Immediate Next Actions.

## Writing Rules

- Start with the action items because supervisors may only scan the first section.
- Keep the report brief but comprehensive: one page of Markdown is the target.
- Use concrete project nouns: VAE, contrastive encoder, BYOL, feature store, aggregator, baseline experiments.
- Avoid raw commit hashes, detailed metric tables, and long experiment analysis unless requested.
- State unsupported claims clearly: do not say the framework is better than baselines until downstream framework evaluation exists.
- Preserve separation of concerns: working docs guide coding and execution; `progress_summary/` is the submission-facing log.
