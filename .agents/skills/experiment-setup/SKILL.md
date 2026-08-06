---
name: experiment-setup
description: Use when planning training, validation, testing, baseline fairness, data allocation, or evaluation for this project, especially when checking for leakage or the correct experiment order.
---

# Experiment Setup

Use the project documents as the authority for experiment design. Do not infer allocation rules from implementation details alone.

## Read First

Read these in order:

1. `docs/training_test_data_selection.md` in full, including the global split, validation split, allocation table, test feature extraction, operation order, aggregator modes, and rules summary.
2. The Experiment Design section of `docs/design.md`, including Data Preparation, Representation Learning, Training Procedure, and Evaluation Process.

## Response Contract

Present the relevant parts of:

- The global 80/20 train/test split and why the test side remains locked.
- The validation split carved only from training data.
- The data allocated to encoders, aggregator, task heads, and baselines for training and evaluation.
- The ordered workflow from raw data through final evaluation.
- The rules that prevent data leakage and preserve baseline fairness.

When reviewing a proposed workflow, identify any step that violates the documented leakage safeguards and explain the compliant alternative. If the documents disagree, report the conflict instead of silently choosing one.
