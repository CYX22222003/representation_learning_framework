Audit the current state of the repository against `docs/schedule.md` and update the schedule to reflect reality. Use both git history and direct file inspection to determine what is actually done, not just what the schedule claims.

---

## Step 1 — Gather git evidence

Run the following commands and note what they reveal:

```bash
git log --oneline -20
```
Summarise which areas of the codebase changed in recent commits (data pipeline, models, training scripts, evaluation, baselines, docs, etc.).

```bash
git diff HEAD~5 -- src/ scripts/ checkpoints/
```
(Adjust the commit range if fewer/more than 5 commits are relevant.) Identify which source files were added or modified.

```bash
git status
```
Note any uncommitted changes that may represent in-progress work not yet in a commit.

```bash
ls checkpoints/
```
List existing checkpoint files — each one is strong evidence that the corresponding model has been trained and saved.

---

## Step 2 — Inspect key source files

For each item in the schedule that is marked 🔄 (implemented, not trained) or ⬜ (not started), quickly check whether the file actually exists and contains substantive code. Do not re-read files whose status is already clearly ✅. Focus on:

- `src/models/vae.py` — VAE implemented?
- `src/models/contrastive.py` — contrastive encoder implemented?
- `src/training/` — do training loop scripts exist?
- `src/aggregation/aggregator.py` — aggregator implemented?
- `src/tasks/` — task head files present and non-trivial?
- `src/evaluation/` — evaluation harness present?
- `src/baselines/` — what baseline code exists?
- `scripts/` — which runnable scripts exist (prepare_sequences, prepare_features, train_framework, evaluate, etc.)?
- `data/processed/` and `data/features/` — do output .npz files exist? (evidence that pipeline scripts have been run)

---

## Step 3 — Compare against `docs/schedule.md`

Read `docs/schedule.md` in full. For each row in the achievement tables (Stages 1–4), check whether the status symbol (✅ / 🔄 / ⬜) matches the evidence gathered in Steps 1 and 2:

- **✅ Done** — code exists, and for trained models, a checkpoint exists
- **🔄 Implemented, not trained** — code exists but no checkpoint or run evidence
- **⬜ Not started** — no file, no commit, no evidence of work

Flag any mismatches (e.g. a task still marked ⬜ whose file was added in a recent commit, or a task marked 🔄 that now has a checkpoint).

Also check:
- Section 2 summary paragraph — does it still accurately describe the current blocker and immediate next step?
- Section 3 phase descriptions — does the project's actual state match the exit conditions for the current phase?

---

## Step 4 — Update `docs/schedule.md`

Apply only the changes that are supported by evidence from Steps 1–3:

- Update status symbols where the evidence is clear (⬜ → 🔄 or ✅, 🔄 → ✅)
- Update the **Last updated** date at the top to today's date
- Rewrite the Section 2 summary paragraph to reflect the current blocker and next action
- If a phase exit condition has been met, note that in the phase description

Do not invent progress. If a file exists but appears empty or skeletal, keep the status as ⬜ or 🔄 as appropriate and note the uncertainty.

---

## Step 5 — Report

After updating, produce a short report:
- Which status entries changed and why (cite the git evidence)
- Which phase the project is currently in and what remains to exit it
- The single most important next action
