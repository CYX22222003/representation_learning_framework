A design decision, new idea, or architectural change has just been discussed. Work through the checklist below to ensure all project documents — including the skill files themselves — stay in sync. This is important because the project follows an ideation → plan/design → implement → evaluate loop, and stale documentation causes confusion across all stages.

---

## Step 1 — Identify the change

If not already clear from context, ask the user to describe in one sentence what changed (e.g. "added a new neural encoder branch", "changed baseline list", "revised aggregator design", "updated data selection rule").

---

## Step 2 — Check and update each document

Work through each document below. Read the relevant sections, check whether they reflect the change, and update if needed. Do not rewrite sections that are already consistent.

### `docs/Research_Ideas_Writeup.md`
- **Section 3.2** (architecture diagram) — does the diagram show the correct branches, aggregator mode, and downstream tasks?
- **Section 3.3** (training overview) — does the training description match the current workflow?
- **Section 3.4** (innovation claims) — are the innovation bullets still accurate?
- **Section 5** (evaluation) — are the tasks, metrics, and provisional baseline list still correct?

### `docs/design.md`
- **Architecture Design section** — does the prose description of branches and aggregator match?
- **Representation Learning bullets** — do they list the right methods, with TBD markers where appropriate?
- **Training Procedure** — does it reflect the current leakage-prevention rules and training sequence?
- **Evaluation Process** — is the comparison structure (benchmark vs internal baseline, probing paradigm, decoder-controlled comparison) still accurate?

### `docs/research_plan.md`
- **Stage 2** — if a model component changed: are the subsections (2.1–2.4) current? Are completion markers accurate?
- **Stage 3** — if baselines changed: does the baseline list reflect the current provisional set?
- **Stage 4** — if evaluation tasks or metrics changed: are they current?

### `docs/schedule.md`
- **Section 1 achievement tables** — do any status entries need updating (e.g. ⬜ → 🔄 or ✅)?
- **Section 2 summary paragraph** — does it still accurately describe where the project stands?
- **Section 3 phase descriptions** — does the new change affect the scope or exit conditions of any phase?

### `CLAUDE.md`
- **Multi-branch representation table** — are branch names, source modules, and output dimensions current?
- **Aggregator section** — is the usage example (branch_dims, mode, output_dim) still correct?
- **Adding new features guide** — is the extension pattern still accurate?
- **Module responsibilities table** — does any row need updating?

### `docs/training_test_data_selection.md`
- **Data allocation table** — does the new component appear with the correct train/val/test mapping?
- **Order of operations** — does the new component slot in at the right step?
- **Rules summary** — do the six rules still cover the new setup?

### `.claude/commands/` (skill files)
- **`project-design.md`** — if the architecture changed, do the listed documents and the "After reading, present" bullets still point to the right sections and describe the right things?
- **`project-plan.md`** — if the phase structure or schedule changed, are the instructions still aligned?
- **`experiment-setup.md`** — if data rules or training order changed, does this skill still point to the right sections?
- **`project-objectives.md`** — if the motivation, tasks, or innovation claims changed, are the section pointers still correct?
- **`update-schedule.md`** — if new source directories, checkpoint paths, or schedule sections were added, does Step 2 still list the right files to inspect?
- **`sync-docs.md`** (this file) — if a new document or section was added to the project, add it to the checklist here so future syncs stay complete.

---

## Step 3 — Confirm consistency

After updating, briefly state which documents and skill files were changed and what specifically was updated. If any section marked TBD is now resolved, remove the TBD marker and fill in the current decision.
