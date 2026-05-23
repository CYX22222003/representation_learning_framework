Read the following documents to establish the experiment design and data allocation rules, then present them clearly.

Documents to read (in order):
1. `docs/training_test_data_selection.md` — full document: global split, validation split, data allocation table, feature extraction for test set, order of operations, aggregator modes, rules summary
2. `docs/design.tex` — Experiment Design section: Data Preparation, Representation Learning, Training Procedure, Evaluation Process subsections

After reading, present:
- The global 80/20 train/test split structure and why the test side is locked
- The validation split strategy carved from training data
- The data allocation table: which data each component (encoders, aggregator, task heads, baselines) trains on and is evaluated on
- The correct order of operations from raw data to final evaluation
- The key rules that prevent data leakage

If the user asks about training procedure, baseline fairness, or evaluation correctness, use these documents as the authoritative reference. Flag any proposed workflow that would violate the leakage-prevention rules.
