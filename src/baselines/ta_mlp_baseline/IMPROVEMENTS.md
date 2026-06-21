# TA-MLP Baseline — Possible Methodology Improvements

Captured 2026-06-22 after the v1 triclass sweep. Pick up from here when ready
to harden the baseline before the framework comparison.

The improvements are tiered by value. Tier 1 are real gaps that limit how the
current numbers can be reported honestly. Tier 2 are worth doing if scope
allows. Tier 3 are nice-to-haves. The "Do NOT do" section is as important
as the rest — it lists tempting changes that would compromise the baseline's
role as a faithful reproduction of the upstream paper.

---

## Tier 1 — Real gaps, do these

### 1.1 Constant-predictor floor

**Why:** Right now nobody can tell whether 0.71 accuracy is good. A trivial
"always predict HOLD" classifier scores ~0.747 acc, ~0.285 macro-F1 on this
test set — meaning the TA-MLP's headline accuracy is *literally worse than
the constant predictor* on accuracy alone. The model adds value on macro-F1
(0.46 vs 0.29, +0.17) but you can't show that without the floor.

**What to build:** `src/baselines/constant_baseline/` with two modes:
- `--task triclass` → predicts always-HOLD (label 1) for every test row
- `--task regression` → predicts the previous close (naive last-value) for LSTM

No training. Reuses `build_ta_dataset` / LSTM dataset construction for
parity. Writes the same artifact shapes (predictions.npz, summary.md) so
plotting and the runs table treat it like any other baseline.

**Effort:** ~30 minutes. **ROI:** Highest — makes every other baseline
number interpretable.

### 1.2 Multi-seed runs for noise calibration

**Why:** Every macro-F1 in the v1 table is one seed. You can't tell if the
gap between e20 (0.4524) and e25 (0.4651) is real or seed noise. Without
this, you can't honestly say "the framework beats TA-MLP" unless the gap
is bigger than your noise band — and you don't know the noise band.

**What to do:** Pick one epoch budget (e25 is a good choice — middle of the
sweep, past the small early-budget effects). Run it at seed=1 and seed=2.

```
experiments/2026-06-22-v2-seeds/
  e25-s0/   ← link or re-symlink existing v1/e25 run
  e25-s1/
  e25-s2/
  summary.md  ← report mean ± std for accuracy and macro-F1
```

Report the noise band in the cross-run summary and the README runs table.

**Effort:** ~5 minutes compute, ~30 minutes writing. **ROI:** Required for
defensible cross-baseline comparison.

---

## Tier 2 — Worth considering, depends on scope

### 2.1 Held-out contracts (vs. held-out time within contracts)

**Why:** Current per-contract chronological 80/20 split tests "can the model
predict the late phase of markets it has seen the early phase of?" — not
"can it predict new markets?" For a representation-learning framework whose
pitch is *transferability*, the more honest split is leave-one-contract-out.

**Caveats:** This is a *framework-level* decision, not a baseline-level one.
If we change the TA-MLP split, we need to change the LSTM split (and every
future baseline) at the same time. Best to defer until the framework
comparison protocol is being designed.

**Effort:** ~1 hour for one baseline; multiplied across baselines. **ROI:**
High if the project's transferability claim is load-bearing in the report.

### 2.2 Floor-relative gain column in the runs table

**Why:** Even with the constant-predictor row added, readers will compare
accuracies directly. A `Δ macro-F1 vs floor` column makes the model's
contribution legible at a glance.

**What to do:** After Tier 1.1 lands, add the column to README runs table
and to the v1 cross-run summary. Format: `+0.17` (or `−0.04`).

**Effort:** ~10 minutes. **ROI:** Small but it changes how readers
interpret the table.

### 2.3 Per-contract test-F1 distribution

**Why:** Global macro-F1 is an average over wildly heterogeneous contracts.
Likely some contracts are 0.6+ macro-F1 and others are below the floor.
Reporting the *distribution* (median, IQR, worst-case) is much more useful
than the single global number — both for understanding the baseline and for
the framework comparison.

**What to do:** Post-hoc analysis script that loads `predictions.npz`,
re-associates each test row with its contract (need to track contract ids
through `build_ta_dataset` — small change), computes per-contract macro-F1,
plots a histogram or box plot.

**Effort:** ~1–2 hours (most of it in re-wiring contract-id tracking).
**ROI:** Substantial — reveals where the baseline actually wins/loses,
useful for the report's qualitative analysis.

### 2.4 Calendar-feature leakage check

**Why:** Features include `DayOfWeek`, `Month`, `Hourly`. If contracts span
limited time ranges, the chronological split lets train see early calendar
values and test see later ones — `Month` effectively becomes a
contract-time identifier. This could explain why train CE keeps dropping
without test improvement (memorization of "this contract trades on these
months").

**What to do:** Re-run e25 with `Month` and `Hourly` zeroed out at feature-
extraction time. If train CE plateaus closer to test performance →
calendar features are contributing to overfitting → consider removing them
(or making it a flag). If no change → ignore.

**Effort:** ~30 minutes. **ROI:** Diagnostic; might reveal a feature
issue or might be a clean negative result.

---

## Tier 3 — Nice-to-have, low ROI

### 3.1 ROC / PR curves

For 3-class, less informative than the confusion matrix that already exists.
Skip unless reviewers ask.

### 3.2 Inference latency / model size

Useful only if the framework explicitly claims "faster" or "smaller."
Don't add speculatively.

---

## Things to deliberately NOT do

These all look like methodology improvements but would compromise the
baseline's role as a faithful reproduction of the upstream paper. If the
framework ends up beating a tuned TA-MLP, the comparison loses meaning.

| Tempting change | Why not |
|---|---|
| **Class weighting / focal loss / oversampling** | The imbalance is a property of the labeling formula, not a bug. Fixing it turns "I ran the published baseline" into "I ran a variant I tuned." |
| **Lower `hold_q` to balance classes** | Same as above. If the *framework's* task needs balanced classes, change the label at the task level in `src/tasks/`, not at the baseline level. |
| **Hyperparameter search** (lr, batch, hidden sizes) | The architecture (36→128→64→32→3) is from the paper. Tuning it makes the baseline a moving target. |
| **K-fold cross-validation** | Overkill. Per-contract splits already give many train/test pairs implicitly; CV on top adds compute for marginal statistical gain. Multi-seed (1.2) gives the noise estimate at a fraction of the cost. |
| **Add dropout or weight decay to fix overfitting** | The published architecture doesn't have them. Adding regularization to "improve" the baseline biases the comparison. |
| **Drop the calendar features without checking 2.4 first** | If they're harmful, document and remove with evidence. Don't preemptively prune features that the paper uses. |

---

## Suggested order if you do Tier 1 + Tier 2

1. **Constant-predictor floor** (~30 min) — makes everything else
   interpretable
2. **Per-contract test-F1 distribution** (~1–2 hr) — diagnostic, no
   retraining
3. **Calendar-feature leakage check** (~30 min) — one extra run, clear
   yes/no signal
4. **Multi-seed at e25** (~5 min compute + ~30 min writing) — noise
   calibration
5. **Floor-relative gain column** (~10 min) — polish, after step 1
6. **Held-out-contract sweep** — defer until framework comparison
   protocol is being designed; apply to all baselines uniformly

Items 1–5 are scoped to the TA-MLP baseline alone. Item 6 is project-wide.

---

## What "good enough" looks like for the report

After Tier 1 (constant predictor + multi-seed), the TA-MLP section can
honestly say something like:

> The TA-MLP baseline achieves macro-F1 0.45–0.47 (mean across 3 seeds,
> ±0.0X std) on the 4-hour triclass task, compared to 0.285 for an
> always-HOLD constant predictor — a +0.17 macro-F1 improvement
> attributable to learning. Headline accuracy (0.71–0.73) is below the
> constant predictor's 0.747, reflecting the model's willingness to make
> minority-class predictions in exchange for non-trivial BUY/SELL recall.

That's a fundable, defensible characterization. Without Tier 1 you can't
write that paragraph honestly.

Tier 2 strengthens it further but isn't required to ship.
