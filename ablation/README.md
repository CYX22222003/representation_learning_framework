# Independent Ablation Workstream

This directory contains the complete ablation-study layer: design code, runner,
tests, plans, reports, and generated artifacts. It is deliberately independent
of the Phase-1 and Phase-2 experiment trees.

The workstream reads the frozen five-branch feature bundle and split-safe task
labels as immutable inputs. It writes only beneath `ablation/experiments/`.
It does not retrain the representation encoders or modify Phase-1/Phase-2
artifacts.

## Quick start

Predeclare and validate the default study without training:

```bash
.venv/bin/python3 ablation/run_ablation.py --stages plan
```

Execute the locked matrix, generate per-run plots, and build the aggregate
report:

```bash
.venv/bin/python3 ablation/run_ablation.py \
  --stages train,plot,report \
  --study-name 4h_all5_v1 \
  --device cuda
```

The default study contains all three tasks, five single-branch probes, five
leave-one-branch-out probes, a matched full-concat control, fixed epoch budgets
`15,50,100`, and seed `0`. Add `--include-gated` only in a new study name;
gated fusion is labelled separately because it adds trainable parameters.

Use a new `--study-name` to change a locked plan. `--overwrite-runs` may replace
training artifacts inside an unchanged plan, but cannot alter the plan itself.

## Directory contract

```text
ablation/
  design.py                 matrix definitions
  run_ablation.py           planning/training/plot/report entry point
  report.py                 matched result aggregation
  EXPERIMENT_PLAN.md        methodology and interpretation contract
  tests/                    focused unit tests
  experiments/<study>/      plan, runs, results, and report
```

Run the focused tests from the project root:

```bash
PYTHONPATH=src:. .venv/bin/python3 -m unittest discover -s ablation/tests -v
```
