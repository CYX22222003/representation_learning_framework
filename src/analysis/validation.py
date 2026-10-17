"""Visible, structured diagnostics; numerical warnings never imply data validity."""

from __future__ import annotations

import sys

import numpy as np

from .artifacts import write_json_atomic
from .types import ValidationObservation, ValidationResult


def compare_numbers(name, actual, expected, *, rtol, atol):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    details = {"rtol": rtol, "atol": atol, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return ValidationObservation(name, "error", "Shape mismatch or non-finite numerical values", details)
    maximum = float(np.max(np.abs(a - b), initial=0.0))
    details.update(max_abs=maximum, relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)))
    if a.size:
        details.update(actual_min=float(a.min()), actual_max=float(a.max()),
                       expected_min=float(b.min()), expected_max=float(b.max()))
    if np.array_equal(a, b):
        return ValidationObservation(name, "pass", "Exact numerical replay", details)
    accepted = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    return ValidationObservation(name, "warning" if accepted else "error",
        "Accepted numerical drift" if accepted else "Numerical drift exceeds frozen tolerance", details)


def result(subject, observations=(), checks=None):
    observations = tuple(observations)
    severity = {o.severity for o in observations}
    status = "invalid" if "error" in severity else "valid_with_warnings" if "warning" in severity else "valid"
    return ValidationResult(subject, status, checks or {}, observations)


def failure(subject, exc):
    return result(subject, (ValidationObservation("integrity", "error", str(exc), {"exception": type(exc).__name__}),))


def emit_validation(value, path=None):
    for observation in value.observations:
        if observation.severity != "pass":
            print(f"[{observation.severity.upper()}] {value.subject}: {observation.check}: "
                  f"{observation.message} {dict(observation.details)}", file=sys.stderr, flush=True)
    if path is not None:
        write_json_atomic(path, {"subject": value.subject, "status": value.status,
                                 "checks": value.checks, "observations": value.observations})
    return value


def merge_results(subject, results):
    observations = []
    checks = {}
    for value in results:
        checks[value.subject] = value.status
        observations.extend(value.observations)
        if not value.usable and not any(o.severity == "error" for o in value.observations):
            observations.append(ValidationObservation(value.subject, "error", "Required validation is not usable"))
    return result(subject, observations, checks)
