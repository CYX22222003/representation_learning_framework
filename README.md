# A Unified Representation Learning Framework for Multi-Task Time-Series Modeling in Event Prediction Markets

A **Final Year Project (FYP)** investigating whether shared, heterogeneous
representations of prediction-market time series provide useful, complementary,
and reusable features across forecasting tasks. Using Polymarket OHLCV data, it
combines statistical features, signal transformations, and pretrained neural
features. Frozen encoders support separate heads for movement classification,
future-price prediction, and future realised-variance forecasting.

## Repository layout

- `src/` — data processing, models, training, evaluation, and analysis.
- `scripts/`, `scripts_v*/` — experiment and data-preparation entry points.
- `tests/` — automated checks.
- `docs/` — research design, experiment plans, and methodology.
- `experiments/` — experiment manifests, validation records, and reports.

See [docs/design.md](docs/design.md) for the design and
[docs/phase_plan/](docs/phase_plan/) for experiment-specific instructions.

This README is a stable overview, excluded from routine documentation sync.
Agents do not need to read it during routine work; working guidance lives in
`AGENTS.md` and the relevant repository skills.
