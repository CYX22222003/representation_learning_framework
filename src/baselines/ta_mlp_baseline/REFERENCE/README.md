# REFERENCE — Original Labeling Algorithm

Verbatim copy of the labeling code from the upstream TA-MLP project, kept here
for provenance only. **This directory is not imported by the framework.**

The framework's ported, vectorized version lives in `../ta_labels.py`. Two
things changed during the port:

1. The feature-extraction methods (`compute_oscillators`, `find_patterns`,
   `add_timely_data`) were dropped — they duplicate `../ta_features.py`. Only
   the labeling logic (`assign_labels`, `_find_alpha_beta`) was ported.
2. `assign_labels` was rewritten without `df.apply` for speed, and `alpha`/
   `beta` are fitted **per contract on training rows only** to avoid leaking
   test-period statistics into the labels.

If the upstream project updates its labeling formula, diff against these files
to see what needs porting.
