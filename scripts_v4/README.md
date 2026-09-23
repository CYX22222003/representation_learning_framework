# Phase 6 entry points

Phase 6 scripts are thin wrappers around reusable implementations under
`src/`. The first gate is the read-only, training-period-only volatility
horizon audit:

```bash
.venv/bin/python3 scripts_v4/audit_phase6_volatility.py
.venv/bin/python3 scripts_v4/validate_phase6_volatility_audit.py
.venv/bin/python3 scripts_v4/validate_phase6_volatility_horizon_freeze.py
```

The audit predeclares `2,4,8,24` hours by default, writes only data-analysis
artifacts, does not freeze the primary horizon, and never launches training.
Changing the candidate set is an explicit CLI action recorded in the output
manifest.

The approved primary horizon is frozen separately to eight hours by
`docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`.
The third command replays that decision against the immutable Stage A audit.
