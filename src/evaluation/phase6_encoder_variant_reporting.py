"""Complete-matrix reporting for Phase 6 temporal encoder variants."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_processing.phase5_walks import sha256_file
from evaluation.phase6_encoder_variants import validate_cka_report
from features.phase6_encoder_variant_features import (
    CONFIG_BRANCHES,
    TASKS,
    validate_variant_feature_store,
)
from training.phase5_absolute_price import validate_absolute_price_run
from training.phase5_downstream import validate_downstream_run
from training.phase5_encoder import write_json
from training.phase6_encoder_variant_downstream import (
    _cross_sectional_rank_ic,
    validate_variant_downstream,
)
from training.phase6_volatility import validate_volatility_run


COMPARISON_PAIRS = (
    ("HC-SL", "H0", "contrastive_substitution"),
    ("HC-ST", "H0", "contrastive_substitution"),
    ("HB-SL", "H0", "byol_substitution"),
    ("HB-ST", "H0", "byol_substitution"),
    ("HC-AL", "H0", "contrastive_addition_vs_base"),
    ("HC-AT", "H0", "contrastive_addition_vs_base"),
    ("HC-AL", "HC-DC", "contrastive_addition_vs_duplicate"),
    ("HC-AT", "HC-DC", "contrastive_addition_vs_duplicate"),
    ("HB-AL", "H0", "byol_addition_vs_base"),
    ("HB-AT", "H0", "byol_addition_vs_base"),
    ("HB-AL", "HB-DC", "byol_addition_vs_duplicate"),
    ("HB-AT", "HB-DC", "byol_addition_vs_duplicate"),
)

TASK_METRICS: dict[str, tuple[tuple[str, bool], ...]] = {
    "classification_h2": (("macro_f1", True), ("balanced_accuracy", True)),
    "absolute_price_h8": (
        ("price_mae", False),
        ("price_rmse", False),
        ("price_pearson", True),
        ("price_spearman", True),
        ("implied_pearson", True),
        ("implied_spearman", True),
        ("implied_sign_agreement", True),
        ("cross_sectional_rank_ic", True),
    ),
    "realised_variance": (
        ("mae", False),
        ("rmse", False),
        ("mse", False),
        ("pearson", True),
        ("spearman", True),
    ),
}


def task_dataset(repo_root: Path, task: str, walk: int) -> Path:
    if task == "classification_h2":
        return repo_root / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    if task == "absolute_price_h8":
        return repo_root / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    return repo_root / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"


def h0_run(repo_root: Path, task: str, walk: int) -> Path:
    if task == "classification_h2":
        return repo_root / "experiments" / "phase5" / "downstream" / f"walk{walk}" / "classification" / "seed0"
    if task == "absolute_price_h8":
        return repo_root / "experiments" / "phase5" / "downstream_addons" / "tasks" / "absolute_price_h8" / f"walk{walk}" / "seed0"
    return repo_root / "experiments" / "phase6" / "volatility_prediction" / "runs" / "framework_h0" / f"walk{walk}" / "seed0"


def variant_run(phase6_root: Path, task: str, configuration: str, walk: int) -> Path:
    return phase6_root / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"


def validate_h0_reference(repo_root: Path, task: str, walk: int) -> dict[str, Any]:
    dataset = task_dataset(repo_root, task, walk)
    run = h0_run(repo_root, task, walk)
    if task == "classification_h2":
        feature = repo_root / "experiments" / "phase5" / "features" / f"walk{walk}" / "five_branch_epoch50.npz"
        scaler = repo_root / "experiments" / "phase5" / "downstream" / f"walk{walk}" / "feature_standardizer.npz"
        return validate_downstream_run(dataset, feature, scaler, run)
    if task == "absolute_price_h8":
        shared = repo_root / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8"
        feature = shared / "features" / f"walk{walk}" / "five_branch_epoch50.npz"
        scaler = shared / "feature_scalers" / f"walk{walk}" / "feature_standardizer.npz"
        return validate_absolute_price_run(dataset, feature, scaler, run)
    feature = repo_root / "experiments" / "phase6" / "volatility_prediction" / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz"
    return validate_volatility_run(dataset, feature, run)


def _h0_price_rank_ic(prediction_path: Path) -> dict[str, float | int]:
    with np.load(prediction_path, allow_pickle=False) as saved:
        score = np.asarray(saved["prediction_delta_h8"], dtype=np.float64)
        target = np.asarray(saved["target_delta_h8"], dtype=np.float64)
        timestamps = np.asarray(saved["decision_date_ns"])
    return _cross_sectional_rank_ic(score, target, timestamps)


def _reference_rows(
    task: str,
    walk: int,
    metrics_payload: Mapping[str, Any],
    dataset_path: Path,
    prediction_path: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if task == "classification_h2":
        for reference in ("always_stable_reference", "repeated_training_prior_reference"):
            for metric, _ in TASK_METRICS[task]:
                result.append(
                    {
                        "task": task,
                        "walk": walk,
                        "reference": reference,
                        "metric": metric,
                        "value": float(metrics_payload[reference][metric]),
                    }
                )
        return result
    if task == "absolute_price_h8":
        persistence = metrics_payload["persistence_reference"]["overall"]
        for metric in ("mae", "rmse", "pearson", "spearman"):
            result.append(
                {
                    "task": task,
                    "walk": walk,
                    "reference": "current_price_persistence",
                    "metric": f"price_{metric}",
                    "value": float(persistence[metric]),
                }
            )
        with np.load(prediction_path, allow_pickle=False) as saved, np.load(
            dataset_path, allow_pickle=False
        ) as source:
            current = np.asarray(saved["current_close"], dtype=np.float64)
            target = np.asarray(saved["target_delta_h8"], dtype=np.float64)
            timestamps = np.asarray(saved["decision_date_ns"])
            raw = np.asarray(source["test_raw_sequences"], dtype=np.float64)
            reversal = -(raw[:, -1, 3] - raw[:, -2, 3])
            reversal_ic = _cross_sectional_rank_ic(reversal, target, timestamps)
        result.append(
            {
                "task": task,
                "walk": walk,
                "reference": "last_hour_reversal",
                "metric": "cross_sectional_rank_ic",
                "value": float(reversal_ic["mean"]),
            }
        )
        return result
    for reference, metrics in metrics_payload["references"].items():
        for metric, _ in TASK_METRICS[task]:
            result.append(
                {
                    "task": task,
                    "walk": walk,
                    "reference": reference,
                    "metric": metric,
                    "value": float(metrics[metric]),
                }
            )
    return result


def normalize_metrics(
    task: str,
    payload: Mapping[str, Any],
    *,
    price_rank_ic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    body = payload.get("metrics", payload)
    if task == "classification_h2":
        framework = body["framework"]
        return {name: float(framework[name]) for name, _ in TASK_METRICS[task]}
    if task == "absolute_price_h8":
        price = body["price"]["overall"]
        movement = body["implied_movement"]
        if "overall" in movement:
            movement = movement["overall"]
        rank_ic = body.get("implied_movement_cross_sectional_rank_ic", price_rank_ic)
        if rank_ic is None:
            raise ValueError("future-price report is missing cross-sectional Rank IC")
        return {
            "price_mae": float(price["mae"]),
            "price_rmse": float(price["rmse"]),
            "price_pearson": float(price["pearson"]),
            "price_spearman": float(price["spearman"]),
            "implied_pearson": float(movement["pearson"]),
            "implied_spearman": float(movement["spearman"]),
            "implied_sign_agreement": float(
                body.get("implied_movement_nonzero_sign_agreement", movement["sign_agreement"])
            ),
            "cross_sectional_rank_ic": float(rank_ic["mean"]),
        }
    model = body["model_metrics"] if "model_metrics" in body else body["model"]
    return {name: float(model[name]) for name, _ in TASK_METRICS[task]}


def paired_differences(
    rows: list[dict[str, Any]], task: str, walk: int
) -> list[dict[str, Any]]:
    indexed = {
        str(row["configuration"]): row
        for row in rows
        if row["task"] == task and int(row["walk"]) == walk
    }
    if set(indexed) != set(CONFIG_BRANCHES):
        raise ValueError(f"incomplete Phase 6 report rows for {task} walk {walk}")
    result = []
    for candidate, reference, comparison_type in COMPARISON_PAIRS:
        for metric, higher_is_better in TASK_METRICS[task]:
            candidate_value = float(indexed[candidate][metric])
            reference_value = float(indexed[reference][metric])
            result.append(
                {
                    "task": task,
                    "walk": walk,
                    "comparison_type": comparison_type,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "candidate_value": candidate_value,
                    "reference_value": reference_value,
                    "candidate_minus_reference": candidate_value - reference_value,
                }
            )
    return result


def _resource_fields(run: Path, metrics: Mapping[str, Any]) -> dict[str, float | int]:
    architecture = json.loads((run / "architecture_manifest.json").read_text(encoding="utf-8"))
    parameter_count = metrics.get("parameter_count", architecture["parameter_count"])
    elapsed = metrics.get("elapsed_training_seconds")
    if elapsed is None:
        with np.load(run / "e50" / "history.npz", allow_pickle=False) as history:
            elapsed = float(np.asarray(history["epoch_seconds"]).sum())
    return {
        "parameter_count": int(parameter_count),
        "elapsed_training_seconds": float(elapsed),
        "inference_seconds": float(metrics["inference_seconds"]),
        "peak_cuda_memory_bytes": int(metrics.get("peak_cuda_memory_bytes", 0)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows for key in row if key not in fieldnames
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_complete_report(repo_root: Path) -> dict[str, Any]:
    phase6_root = repo_root / "experiments" / "phase6" / "encoder_variants"
    manifest_path = phase6_root / "manifests" / "downstream_seed0.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("entry_count") != 66 or len(manifest.get("entries", [])) != 66:
        raise ValueError("Phase 6 downstream freeze is not the complete 66-entry matrix")

    validation_counts = {"encoders": 0, "features": 0, "downstream": 0, "cka": 0}
    encoder_resource_rows: list[dict[str, Any]] = []
    for walk in (1, 2):
        encoder_dataset = task_dataset(repo_root, "classification_h2", walk)
        validate_cka_report(
            phase6_root / "diagnostics" / "cka" / f"walk{walk}.json",
            encoder_dataset,
        )
        validation_counts["cka"] += 1
        for task in TASKS:
            feature_path = phase6_root / "features" / f"walk{walk}" / f"{task}.npz"
            validate_variant_feature_store(feature_path)
            validation_counts["features"] += 1
            if task == "classification_h2":
                # Feature validation above fully replays all four temporal runs.
                validation_counts["encoders"] += 4
                feature_manifest = json.loads(
                    Path(f"{feature_path}.manifest.json").read_text(encoding="utf-8")
                )
                for variant, provenance in feature_manifest["checkpoint_provenance"].items():
                    family, backbone = variant.split("_", 1)
                    run = phase6_root / "pretraining" / f"walk{walk}" / family / backbone / "seed0"
                    metrics = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
                    architecture = json.loads(
                        (run / "architecture_manifest.json").read_text(encoding="utf-8")
                    )
                    encoder_resource_rows.append(
                        {
                            "walk": walk,
                            "variant": variant,
                            "trainable_parameter_count": int(
                                architecture["trainable_parameter_count"]
                            ),
                            "total_parameter_count": int(
                                architecture["total_parameter_count"]
                            ),
                            "elapsed_training_seconds": float(
                                metrics["elapsed_training_seconds"]
                            ),
                            "train_feature_extraction_seconds": float(
                                provenance["extraction_seconds"]["train"]
                            ),
                            "test_feature_extraction_seconds": float(
                                provenance["extraction_seconds"]["test"]
                            ),
                            "peak_cuda_memory_bytes": int(
                                metrics["peak_cuda_memory_bytes"]
                            ),
                        }
                    )

    result_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for walk in (1, 2):
            dataset = task_dataset(repo_root, task, walk)
            feature = phase6_root / "features" / f"walk{walk}" / f"{task}.npz"
            for configuration in CONFIG_BRANCHES:
                if configuration == "H0":
                    validate_h0_reference(repo_root, task, walk)
                    run = h0_run(repo_root, task, walk)
                else:
                    run = variant_run(phase6_root, task, configuration, walk)
                    validate_variant_downstream(dataset, feature, run)
                validation_counts["downstream"] += 1
                metrics_path = run / "e50" / "metrics.json"
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                price_rank_ic = None
                if task == "absolute_price_h8" and configuration == "H0":
                    price_rank_ic = _h0_price_rank_ic(run / "e50" / "predictions.npz")
                if configuration == "H0":
                    reference_rows.extend(
                        _reference_rows(
                            task,
                            walk,
                            metrics_payload,
                            dataset,
                            run / "e50" / "predictions.npz",
                        )
                    )
                row = {
                    "task": task,
                    "walk": walk,
                    "configuration": configuration,
                    "width": 445 if configuration in {"H0", "HC-SL", "HC-ST", "HB-SL", "HB-ST"} else 573,
                    **normalize_metrics(task, metrics_payload, price_rank_ic=price_rank_ic),
                }
                result_rows.append(row)
                resource_rows.append(
                    {
                        "task": task,
                        "walk": walk,
                        "configuration": configuration,
                        **_resource_fields(run, metrics_payload),
                    }
                )

    if len(result_rows) != 66 or validation_counts["downstream"] != 66:
        raise ValueError("Phase 6 report did not replay all 66 entries")
    difference_rows = [
        row
        for task in TASKS
        for walk in (1, 2)
        for row in paired_differences(result_rows, task, walk)
    ]
    output = phase6_root / "reports" / "complete_seed0"
    results_csv = output / "epoch50_by_walk.csv"
    differences_csv = output / "paired_differences.csv"
    resources_csv = output / "resources.csv"
    encoder_resources_csv = output / "encoder_resources.csv"
    references_csv = output / "task_references.csv"
    _write_csv(results_csv, result_rows)
    _write_csv(differences_csv, difference_rows)
    _write_csv(resources_csv, resource_rows)
    _write_csv(encoder_resources_csv, encoder_resource_rows)
    _write_csv(references_csv, reference_rows)
    lines = [
        "# Phase 6 Temporal Encoder Variant Report",
        "",
        "Epoch 50 is the predeclared principal snapshot. All 66 task/walk/configuration entries replay before this report is written. No universal architecture ranking or model selection is performed.",
        "",
        "## Contract",
        "",
        "- Substitutions are compared with `H0` at the same 445-dimensional width.",
        "- Additions are compared with both `H0` and the matching 573-dimensional duplicate-CNN control.",
        "- CKA is descriptive only and does not select a configuration.",
        "- Results are single-seed characterisation evidence and remain separated by task, SSL family, and walk.",
        "",
        "## Artifacts",
        "",
        "Detailed epoch-50 values, paired differences, and resource measurements are stored in the adjacent CSV files. The duplicate controls increase width with perfectly collinear copied coordinates, so they are informative but imperfect capacity controls.",
    ]
    for task in TASKS:
        lines.extend(["", f"## {task}", ""])
        metric_names = [name for name, _ in TASK_METRICS[task]]
        for walk in (1, 2):
            lines.extend(
                [
                    f"### Walk {walk}",
                    "",
                    "| Configuration | Width | "
                    + " | ".join(metric_names)
                    + " |",
                    "|---|---:|" + "---:|" * len(metric_names),
                ]
            )
            for row in result_rows:
                if row["task"] == task and row["walk"] == walk:
                    values = " | ".join(
                        "nan"
                        if not math.isfinite(float(row[name]))
                        else f"{float(row[name]):.8g}"
                        for name in metric_names
                    )
                    lines.append(
                        f"| {row['configuration']} | {row['width']} | {values} |"
                    )
            lines.extend(
                [
                    "",
                    "Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.",
                    "",
                    "| Comparison | Metric | Delta |",
                    "|---|---|---:|",
                ]
            )
            for row in difference_rows:
                if row["task"] == task and row["walk"] == walk:
                    lines.append(
                        f"| {row['candidate']} - {row['reference']} | {row['metric']} | "
                        f"{float(row['candidate_minus_reference']):.8g} |"
                    )
    lines.extend(
        [
            "",
            "## Non-learned task references",
            "",
            "| Task | Walk | Reference | Metric | Value |",
            "|---|---:|---|---|---:|",
        ]
    )
    for row in reference_rows:
        lines.append(
            f"| {row['task']} | {row['walk']} | {row['reference']} | "
            f"{row['metric']} | {float(row['value']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Representation similarity",
            "",
            "| Walk | Family | Variant | Linear CKA |",
            "|---:|---|---|---:|",
        ]
    )
    for walk in (1, 2):
        cka_payload = json.loads(
            (phase6_root / "diagnostics" / "cka" / f"walk{walk}.json").read_text(
                encoding="utf-8"
            )
        )
        for row in cka_payload["rows"]:
            lines.append(
                f"| {walk} | {row['family']} | {row['variant']} | "
                f"{float(row['cka']):.8g} |"
            )
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cka_hashes = {
        f"walk{walk}": sha256_file(phase6_root / "diagnostics" / "cka" / f"walk{walk}.json")
        for walk in (1, 2)
    }
    report_manifest = {
        "phase": 6,
        "principal_epoch": 50,
        "seed": 0,
        "matrix_complete": True,
        "entry_count": 66,
        "model_selection_performed": False,
        "universal_architecture_ranking_performed": False,
        "validation_counts": validation_counts,
        "comparison_pairs": [list(row) for row in COMPARISON_PAIRS],
        "downstream_manifest_sha256": sha256_file(manifest_path),
        "cka_hashes": cka_hashes,
        "artifacts": {
            "epoch50_by_walk_csv_sha256": sha256_file(results_csv),
            "paired_differences_csv_sha256": sha256_file(differences_csv),
            "resources_csv_sha256": sha256_file(resources_csv),
            "encoder_resources_csv_sha256": sha256_file(encoder_resources_csv),
            "task_references_csv_sha256": sha256_file(references_csv),
            "report_sha256": sha256_file(report_path),
        },
    }
    write_json(output / "manifest.json", report_manifest)
    return {
        "valid": True,
        "report": str(report_path.resolve()),
        "entries": len(result_rows),
        "paired_differences": len(difference_rows),
        "validation_counts": validation_counts,
    }
