"""Read-only audits for the experiment evidence packaged in a FlowMesh image.

The functions in this module do not train models or mutate experiment results.
They summarize a deliberately bounded set of completed report artifacts and
write a JSON copy to ``FLOWMESH_OUTPUT`` when executed by a FlowMesh Python
task.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPOSITORY_ROOT = Path("/app")

TASK_PRIMARY_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "classification_h2": (
        ("macro_f1", "max"),
        ("balanced_accuracy", "max"),
    ),
    "absolute_price_h8": (
        ("price_mae", "min"),
        ("price_rmse", "min"),
    ),
    "realised_variance": (
        ("mae", "min"),
        ("rmse", "min"),
    ),
}

EXTERNAL_METHODS = {
    "saurl_frozen": "SaURL-TS-Frozen",
    "lwa_frozen": "LWA-Frozen",
    "timedart_frozen": "TimeDART-Frozen",
}


def _root(repository_root: str | Path | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(
        os.environ.get("FYP_REPOSITORY_ROOT", str(DEFAULT_REPOSITORY_ROOT))
    ).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite: {value!r}")
    return number


def _optional_number(value: Any, *, field: str) -> float | None:
    if value in (None, ""):
        return None
    return _finite_number(value, field=field)


def _parsed_metric_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": row["task"],
        "walk": int(row["walk"]),
        "configuration": row["configuration"],
    }
    if row.get("width") not in (None, ""):
        result["width"] = int(float(row["width"]))
    if row.get("role") not in (None, ""):
        result["role"] = row["role"]
    ignored = {"task", "walk", "configuration", "width", "role"}
    for key, value in row.items():
        if key in ignored or value in (None, ""):
            continue
        result[key] = _finite_number(value, field=f"{row['configuration']}.{key}")
    return result


def _write_result_artifact(
    audit_name: str,
    output: dict[str, Any],
    metrics: dict[str, float],
) -> str | None:
    raw_output_dir = os.environ.get("FLOWMESH_OUTPUT")
    if not raw_output_dir:
        return None
    output_dir = Path(raw_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{audit_name}.json"
    destination = output_dir / filename
    destination.write_text(
        json.dumps(
            {"output": output, "metrics": metrics},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return filename


def _finish(
    audit_name: str,
    output: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    finite_metrics = {
        key: _finite_number(value, field=key) for key, value in metrics.items()
    }
    artifact_file = _write_result_artifact(audit_name, output, finite_metrics)
    output["artifact_file"] = artifact_file
    return {"output": output, "metrics": finite_metrics}


def inspect_encoder_pretraining(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the six canonical Phase-5 encoder-pretraining trajectories."""

    root = _root(repository_root)
    summary_path = (
        root
        / "experiments/phase5/reports/encoder_pretraining_seed0/summary.json"
    )
    summary = _load_json(summary_path)
    raw_runs = summary.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("encoder pretraining summary has no runs list")

    expected = {
        (walk, encoder)
        for walk in (1, 2)
        for encoder in ("vae", "contrastive", "byol")
    }
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    duplicates: list[str] = []
    runs: list[dict[str, Any]] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise ValueError("encoder pretraining run must be an object")
        identity = (int(raw["walk"]), str(raw["encoder"]))
        if identity in seen:
            duplicates.append(f"walk{identity[0]}:{identity[1]}")
        seen[identity] = raw
        checkpoint_path = str(raw["epoch50_checkpoint"])
        marker = "/experiments/"
        if marker in checkpoint_path:
            checkpoint_path = "experiments/" + checkpoint_path.split(marker, 1)[1]
        runs.append(
            {
                "walk": identity[0],
                "encoder": identity[1],
                "valid": bool(raw.get("valid")),
                "collapse_warning": bool(raw.get("collapse_warning")),
                "snapshots": list(raw.get("snapshots", [])),
                "epoch50_train_loss": _finite_number(
                    raw["epoch50_train_loss"], field=f"{identity}.train_loss"
                ),
                "epoch50_embedding_std": _finite_number(
                    raw["epoch50_embedding_std"], field=f"{identity}.embedding_std"
                ),
                "epoch50_checkpoint": checkpoint_path,
                "epoch50_checkpoint_sha256": str(
                    raw["epoch50_checkpoint_sha256"]
                ),
            }
        )

    observed = set(seen)
    valid_expected = sum(
        bool(seen[identity].get("valid")) for identity in expected & observed
    )
    collapse_free_expected = sum(
        not bool(seen[identity].get("collapse_warning"))
        for identity in expected & observed
    )
    denominator = float(len(expected))
    metrics = {
        "encoder_pretraining_coverage": len(expected & observed) / denominator,
        "encoder_pretraining_valid_fraction": valid_expected / denominator,
        "encoder_pretraining_collapse_free_fraction": (
            collapse_free_expected / denominator
        ),
    }
    output = {
        "audit": "encoder_pretraining",
        "seed": 0,
        "principal_epoch": 50,
        "expected_run_count": len(expected),
        "observed_run_count": len(raw_runs),
        "missing_runs": [
            f"walk{walk}:{encoder}" for walk, encoder in sorted(expected - observed)
        ],
        "unexpected_runs": [
            f"walk{walk}:{encoder}" for walk, encoder in sorted(observed - expected)
        ],
        "duplicate_runs": sorted(duplicates),
        "runs": sorted(runs, key=lambda row: (row["walk"], row["encoder"])),
        "interpretation_note": (
            "Training losses are reported within each SSL objective and must not "
            "be ranked across VAE, contrastive, and BYOL objectives."
        ),
        "sources": [_source(summary_path, root)],
    }
    return _finish("encoder_pretraining_audit", output, metrics)


def inspect_downstream_tasks(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit canonical H0 results on the three current downstream tasks."""

    root = _root(repository_root)
    table_path = (
        root
        / "experiments/phase6/encoder_variants/reports/complete_seed0/"
        "epoch50_by_walk.csv"
    )
    all_rows = _load_csv(table_path)
    rows = [_parsed_metric_row(row) for row in all_rows if row["configuration"] == "H0"]
    expected = {(task, walk) for task in TASK_PRIMARY_METRICS for walk in (1, 2)}
    observed = {(row["task"], row["walk"]) for row in rows}

    finite_count = 0
    expected_metric_count = 0
    for row in rows:
        for metric_name, _ in TASK_PRIMARY_METRICS[row["task"]]:
            expected_metric_count += 1
            value = row.get(metric_name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                finite_count += 1

    target_metric_count = len(expected) * 2
    metrics = {
        "downstream_task_walk_coverage": len(expected & observed) / len(expected),
        "downstream_primary_metric_finite_fraction": (
            finite_count / target_metric_count
        ),
        "downstream_evidence_ready": float(
            observed == expected and finite_count == target_metric_count
        ),
    }
    output = {
        "audit": "canonical_downstream_tasks",
        "configuration": "H0",
        "seed": 0,
        "principal_epoch": 50,
        "expected_task_walk_count": len(expected),
        "observed_task_walk_count": len(observed),
        "missing_task_walks": [
            f"{task}:walk{walk}" for task, walk in sorted(expected - observed)
        ],
        "results": sorted(rows, key=lambda row: (row["task"], row["walk"])),
        "metric_directions": {
            task: {name: direction for name, direction in definitions}
            for task, definitions in TASK_PRIMARY_METRICS.items()
        },
        "sources": [_source(table_path, root)],
    }
    return _finish("downstream_tasks_audit", output, metrics)


def _best_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        task = str(row["task"])
        if task in TASK_PRIMARY_METRICS:
            grouped.setdefault((task, int(row["walk"])), []).append(row)

    best: list[dict[str, Any]] = []
    for (task, walk), candidates in sorted(grouped.items()):
        for metric_name, direction in TASK_PRIMARY_METRICS[task]:
            populated = [row for row in candidates if row.get(metric_name) is not None]
            if not populated:
                continue
            chooser = min if direction == "min" else max
            chosen = chooser(populated, key=lambda row: row[metric_name])
            best.append(
                {
                    "task": task,
                    "walk": walk,
                    "metric": metric_name,
                    "direction": direction,
                    "configuration": chosen["configuration"],
                    "value": chosen[metric_name],
                }
            )
    return best


def inspect_encoder_variants(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the completed Phase-6 encoder-variant matrix and report hashes."""

    root = _root(repository_root)
    report_dir = (
        root / "experiments/phase6/encoder_variants/reports/complete_seed0"
    )
    manifest_path = report_dir / "manifest.json"
    table_path = report_dir / "epoch50_by_walk.csv"
    manifest = _load_json(manifest_path)
    rows = [_parsed_metric_row(row) for row in _load_csv(table_path)]
    entry_count = int(manifest["entry_count"])
    unique_identities = {
        (row["task"], row["walk"], row["configuration"]) for row in rows
    }

    artifact_files = {
        "encoder_resources_csv_sha256": "encoder_resources.csv",
        "epoch50_by_walk_csv_sha256": "epoch50_by_walk.csv",
        "paired_differences_csv_sha256": "paired_differences.csv",
        "report_sha256": "report.md",
        "resources_csv_sha256": "resources.csv",
        "task_references_csv_sha256": "task_references.csv",
    }
    declared_hashes = manifest.get("artifacts", {})
    hash_checks: list[dict[str, Any]] = []
    for manifest_key, filename in artifact_files.items():
        artifact_path = report_dir / filename
        actual = _sha256(artifact_path)
        expected_hash = str(declared_hashes.get(manifest_key, ""))
        hash_checks.append(
            {
                "path": artifact_path.relative_to(root).as_posix(),
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "matches": actual == expected_hash,
            }
        )

    hash_match_count = sum(bool(row["matches"]) for row in hash_checks)
    matrix_complete = bool(manifest.get("matrix_complete"))
    metrics = {
        "encoder_variant_row_coverage": len(unique_identities) / entry_count,
        "encoder_variant_artifact_integrity": hash_match_count / len(hash_checks),
        "encoder_variant_matrix_complete": float(
            matrix_complete and len(unique_identities) == entry_count
        ),
    }
    source_paths = [manifest_path] + [
        report_dir / filename for filename in artifact_files.values()
    ]
    output = {
        "audit": "encoder_variants",
        "phase": manifest.get("phase"),
        "seed": manifest.get("seed"),
        "principal_epoch": manifest.get("principal_epoch"),
        "declared_entry_count": entry_count,
        "observed_row_count": len(rows),
        "unique_entry_count": len(unique_identities),
        "matrix_complete": matrix_complete,
        "validation_counts": manifest.get("validation_counts"),
        "best_observed_by_metric": _best_rows(rows),
        "hash_checks": hash_checks,
        "comparison_pair_count": len(manifest.get("comparison_pairs", [])),
        "paired_difference_row_count": len(
            _load_csv(report_dir / "paired_differences.csv")
        ),
        "model_selection_performed": manifest.get("model_selection_performed"),
        "universal_architecture_ranking_performed": manifest.get(
            "universal_architecture_ranking_performed"
        ),
        "interpretation_note": (
            "Best-observed rows are diagnostic metric-wise envelopes, not a "
            "preselected deployable model or a universal architecture ranking."
        ),
        "sources": [_source(path, root) for path in source_paths],
    }
    return _finish("encoder_variants_audit", output, metrics)


def _external_metrics(task: str, payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{task} result has no metrics object")
    if task == "classification_h2":
        values = metrics.get("framework")
        mapping = {"macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy"}
    elif task == "absolute_price_h8":
        price_group = metrics.get("price")
        values = price_group.get("overall") if isinstance(price_group, dict) else None
        mapping = {"price_mae": "mae", "price_rmse": "rmse"}
    elif task == "realised_variance":
        values = metrics.get("model")
        mapping = {"mae": "mae", "rmse": "rmse"}
    else:
        raise ValueError(f"unsupported downstream task: {task}")
    if not isinstance(values, dict):
        raise ValueError(f"{task} result has no expected metric group")
    return {
        target: _finite_number(values[source], field=f"{task}.{target}")
        for target, source in mapping.items()
    }


def _deduplicate_internal_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        if str(row["configuration"]).startswith("raw_ohlcv_"):
            continue
        identity = (str(row["task"]), int(row["walk"]), str(row["configuration"]))
        existing = unique.get(identity)
        if existing is None:
            unique[identity] = row
            continue
        for metric_name, _ in TASK_PRIMARY_METRICS.get(identity[0], ()):
            left = existing.get(metric_name)
            right = row.get(metric_name)
            if left is not None and right is not None and not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(
                    f"duplicate internal row disagrees for {identity}.{metric_name}"
                )
    return list(unique.values())


def inspect_baseline_comparison(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble H0, internal-envelope, and external frozen-baseline evidence."""

    root = _root(repository_root)
    internal_paths = [
        root
        / "experiments/phase6/encoder_variants/reports/complete_seed0/"
        "epoch50_by_walk.csv",
        root
        / "experiments/phase6_5/lstm_capacity/reports/complete_seed0/"
        "epoch50_by_walk.csv",
        root
        / "experiments/phase6_5/residual_cnn/reports/complete_seed0/"
        "epoch50_by_walk.csv",
    ]
    internal_rows = _deduplicate_internal_rows(
        _parsed_metric_row(row)
        for path in internal_paths
        for row in _load_csv(path)
    )

    external_root = root / "experiments/phase6_7/downstream"
    external_files = sorted(external_root.glob("*/*/walk*/seed0/e50/metrics.json"))
    external: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in external_files:
        relative = path.relative_to(external_root)
        task, method, walk_name = relative.parts[:3]
        if method not in EXTERNAL_METHODS or task not in TASK_PRIMARY_METRICS:
            continue
        walk = int(walk_name.removeprefix("walk"))
        external[(task, walk, method)] = {
            "method": EXTERNAL_METHODS[method],
            "metrics": _external_metrics(task, _load_json(path)),
            "source": _source(path, root),
        }

    expected_external = {
        (task, walk, method)
        for task in TASK_PRIMARY_METRICS
        for walk in (1, 2)
        for method in EXTERNAL_METHODS
    }
    observed_external = set(external)
    comparisons: list[dict[str, Any]] = []
    complete_groups = 0
    finite_primary = 0
    expected_primary = 0

    for task in TASK_PRIMARY_METRICS:
        for walk in (1, 2):
            candidates = [
                row
                for row in internal_rows
                if row["task"] == task and row["walk"] == walk
            ]
            h0_candidates = [row for row in candidates if row["configuration"] == "H0"]
            if len(h0_candidates) != 1:
                raise ValueError(f"expected one H0 row for {task} walk {walk}")
            h0 = h0_candidates[0]
            method_rows = [
                {
                    "method": "Canonical H0",
                    "metrics": {
                        name: h0[name] for name, _ in TASK_PRIMARY_METRICS[task]
                    },
                }
            ]
            group_complete = True
            for method in EXTERNAL_METHODS:
                value = external.get((task, walk, method))
                if value is None:
                    group_complete = False
                    continue
                method_rows.append(
                    {"method": value["method"], "metrics": value["metrics"]}
                )
            if group_complete:
                complete_groups += 1

            for row in method_rows:
                for metric_name, _ in TASK_PRIMARY_METRICS[task]:
                    expected_primary += 1
                    value = row["metrics"].get(metric_name)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        finite_primary += 1

            envelope: list[dict[str, Any]] = []
            for metric_name, direction in TASK_PRIMARY_METRICS[task]:
                populated = [row for row in candidates if row.get(metric_name) is not None]
                chooser = min if direction == "min" else max
                chosen = chooser(populated, key=lambda row: row[metric_name])
                envelope.append(
                    {
                        "metric": metric_name,
                        "direction": direction,
                        "configuration": chosen["configuration"],
                        "value": chosen[metric_name],
                    }
                )
            comparisons.append(
                {
                    "task": task,
                    "walk": walk,
                    "methods": method_rows,
                    "best_internal_metricwise_envelope": envelope,
                }
            )

    narrative_path = (
        root
        / "experiments/phase6_7/reports/frozen_representation_seed0/summary.md"
    )
    if not narrative_path.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {narrative_path}")
    group_count = len(TASK_PRIMARY_METRICS) * 2
    metrics = {
        "baseline_external_artifact_coverage": (
            len(expected_external & observed_external) / len(expected_external)
        ),
        "baseline_task_walk_coverage": complete_groups / group_count,
        "baseline_primary_metric_finite_fraction": (
            finite_primary / expected_primary if expected_primary else 0.0
        ),
        "baseline_comparison_ready": float(
            observed_external == expected_external
            and complete_groups == group_count
            and finite_primary == expected_primary
        ),
    }
    output = {
        "audit": "frozen_representation_baseline_comparison",
        "seed": 0,
        "principal_epoch": 50,
        "expected_external_artifact_count": len(expected_external),
        "observed_external_artifact_count": len(observed_external),
        "missing_external_artifacts": [
            f"{task}:{method}:walk{walk}"
            for task, walk, method in sorted(expected_external - observed_external)
        ],
        "comparisons": comparisons,
        "interpretation_note": (
            "The best-internal row is a metric-wise descriptive envelope and may "
            "select different configurations for different cells. It is not one "
            "preselected deployable model and does not establish universal SOTA."
        ),
        "sources": (
            [_source(path, root) for path in internal_paths]
            + [external[key]["source"] for key in sorted(external)]
            + [_source(narrative_path, root)]
        ),
    }
    return _finish("baseline_comparison_audit", output, metrics)


AUDITS = {
    "encoder_pretraining": inspect_encoder_pretraining,
    "downstream_tasks": inspect_downstream_tasks,
    "encoder_variants": inspect_encoder_variants,
    "baseline_comparison": inspect_baseline_comparison,
}


def run_audit(
    audit_name: str,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one named audit; useful for local validation and thin wrappers."""

    try:
        audit = AUDITS[audit_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown audit {audit_name!r}; choose from {sorted(AUDITS)}"
        ) from exc
    return audit(repository_root)
