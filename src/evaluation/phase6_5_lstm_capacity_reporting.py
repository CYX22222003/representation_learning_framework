"""Complete-matrix reporting for the Phase 6.5A LSTM-capacity study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation.phase6_5_lstm_capacity import validate_lstm_capacity_cka
from evaluation.phase6_encoder_variant_reporting import (
    TASK_METRICS,
    _h0_price_rank_ic,
    h0_run,
    normalize_metrics,
    task_dataset,
    validate_h0_reference,
)
from features.phase6_5_lstm_capacity_features import (
    CONFIG_BRANCHES,
    CONFIG_DIMS,
    TASKS,
    validate_lstm_capacity_feature_store,
)
from features.phase6_encoder_variant_features import validate_variant_feature_store
from training.phase6_5_lstm_capacity import validate_lstm_capacity_encoder
from training.phase6_5_lstm_capacity_downstream import validate_lstm_capacity_downstream
from training.phase6_encoder_variant_downstream import validate_variant_downstream


REFERENCE_CONFIGURATIONS = ("H0", "HC-SL", "HB-SL", "HC-AL", "HB-AL", "HC-DC", "HB-DC")
DEEP_CONFIGURATIONS = tuple(name for name in CONFIG_BRANCHES if name != "H0")
COMPARISON_PAIRS = (
    ("HC-SL2", "HC-SL", "contrastive_depth_substitution"),
    ("HB-SL2", "HB-SL", "byol_depth_substitution"),
    ("HC-AL2", "HC-AL", "contrastive_depth_addition"),
    ("HB-AL2", "HB-AL", "byol_depth_addition"),
    ("HC-SL2", "H0", "contrastive_substitution_vs_h0"),
    ("HB-SL2", "H0", "byol_substitution_vs_h0"),
    ("HC-AL2", "H0", "contrastive_addition_vs_h0"),
    ("HB-AL2", "H0", "byol_addition_vs_h0"),
    ("HC-AL2", "HC-DC", "contrastive_addition_vs_duplicate"),
    ("HB-AL2", "HB-DC", "byol_addition_vs_duplicate"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Phase 6.5A report table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(key for row in rows for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _phase6_run(repo_root: Path, task: str, configuration: str, walk: int) -> Path:
    return (
        repo_root
        / "experiments"
        / "phase6"
        / "encoder_variants"
        / "downstream"
        / task
        / configuration.lower()
        / f"walk{walk}"
        / "seed0"
    )


def _deep_run(capacity_root: Path, task: str, configuration: str, walk: int) -> Path:
    return capacity_root / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"


def _metric_row(
    task: str,
    walk: int,
    configuration: str,
    run: Path,
) -> dict[str, Any]:
    payload = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
    price_rank_ic = None
    if task == "absolute_price_h8" and configuration == "H0":
        price_rank_ic = _h0_price_rank_ic(run / "e50" / "predictions.npz")
    width = 445 if configuration in {"H0", "HC-SL", "HB-SL", "HC-SL2", "HB-SL2"} else 573
    return {
        "task": task,
        "walk": walk,
        "configuration": configuration,
        "width": width,
        **normalize_metrics(task, payload, price_rank_ic=price_rank_ic),
    }


def build_lstm_capacity_report(repo_root: Path) -> dict[str, Any]:
    capacity_root = repo_root / "experiments" / "phase6_5" / "lstm_capacity"
    phase6_root = repo_root / "experiments" / "phase6" / "encoder_variants"
    manifest = json.loads(
        (repo_root / "experiments" / "phase6_5" / "manifests" / "lstm_capacity_downstream_seed0.json").read_text(encoding="utf-8")
    )
    if manifest.get("entry_count") != 24 or len(manifest.get("entries", [])) != 24:
        raise ValueError("Phase 6.5A downstream freeze is not the complete 24-run matrix")
    validation_counts = {"encoders": 0, "features": 0, "downstream": 0, "cka": 0}
    encoder_rows = []
    for walk in (1, 2):
        encoder_dataset = task_dataset(repo_root, "classification_h2", walk)
        validate_lstm_capacity_cka(
            capacity_root / "diagnostics" / "cka" / f"walk{walk}.json",
            encoder_dataset,
        )
        validation_counts["cka"] += 1
        for family in ("contrastive", "byol"):
            variant = f"{family}_lstm2"
            run = capacity_root / "pretraining" / f"walk{walk}" / family / "lstm2" / "seed0"
            validate_lstm_capacity_encoder(encoder_dataset, run)
            validation_counts["encoders"] += 1
            architecture = json.loads((run / "architecture_manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
            encoder_rows.append(
                {
                    "walk": walk,
                    "variant": variant,
                    "trainable_parameter_count": architecture["trainable_parameter_count"],
                    "total_parameter_count": architecture["total_parameter_count"],
                    "elapsed_training_seconds": metrics["elapsed_training_seconds"],
                    "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
                    "epoch50_loss": metrics["train_loss"],
                    "epoch50_embedding_std": metrics["embedding_std"],
                }
            )
        for task in TASKS:
            validate_lstm_capacity_feature_store(
                capacity_root / "features" / f"walk{walk}" / f"{task}.npz"
            )
            validation_counts["features"] += 1
    result_rows = []
    resource_rows = []
    for task in TASKS:
        for walk in (1, 2):
            dataset = task_dataset(repo_root, task, walk)
            deep_feature = capacity_root / "features" / f"walk{walk}" / f"{task}.npz"
            shallow_feature = phase6_root / "features" / f"walk{walk}" / f"{task}.npz"
            validate_variant_feature_store(shallow_feature)
            for configuration in REFERENCE_CONFIGURATIONS:
                if configuration == "H0":
                    validate_h0_reference(repo_root, task, walk)
                    run = h0_run(repo_root, task, walk)
                else:
                    run = _phase6_run(repo_root, task, configuration, walk)
                    validate_variant_downstream(dataset, shallow_feature, run)
                result_rows.append(_metric_row(task, walk, configuration, run))
            for configuration in DEEP_CONFIGURATIONS:
                run = _deep_run(capacity_root, task, configuration, walk)
                validate_lstm_capacity_downstream(dataset, deep_feature, run)
                validation_counts["downstream"] += 1
                result_rows.append(_metric_row(task, walk, configuration, run))
                architecture = json.loads((run / "architecture_manifest.json").read_text(encoding="utf-8"))
                metrics = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
                resource_rows.append(
                    {
                        "task": task,
                        "walk": walk,
                        "configuration": configuration,
                        "parameter_count": architecture["parameter_count"],
                        "elapsed_training_seconds": metrics["elapsed_training_seconds"],
                        "inference_seconds": metrics["inference_seconds"],
                        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
                    }
                )
    if validation_counts != {"encoders": 4, "features": 6, "downstream": 24, "cka": 2}:
        raise ValueError("Phase 6.5A report validation count mismatch")
    indexed = {
        (row["task"], int(row["walk"]), row["configuration"]): row
        for row in result_rows
    }
    difference_rows = []
    for task in TASKS:
        for walk in (1, 2):
            for candidate, reference, comparison_type in COMPARISON_PAIRS:
                left = indexed[(task, walk, candidate)]
                right = indexed[(task, walk, reference)]
                for metric, higher_is_better in TASK_METRICS[task]:
                    candidate_value = float(left[metric])
                    reference_value = float(right[metric])
                    difference_rows.append(
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
    output = capacity_root / "reports" / "complete_seed0"
    _write_csv(output / "epoch50_by_walk.csv", result_rows)
    _write_csv(output / "paired_differences.csv", difference_rows)
    _write_csv(output / "downstream_resources.csv", resource_rows)
    _write_csv(output / "encoder_resources.csv", encoder_rows)
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# Phase 6.5A LSTM Capacity Report",
                "",
                "Epoch 50 is the predeclared principal snapshot. All four encoders, six feature stores, 24 new downstream trajectories, two CKA reports, and the named Phase 6 references replay before this report is written.",
                "",
                "The primary contrasts compare each two-layer LSTM configuration with its same-family one-layer predecessor. Comparisons with H0 and the duplicate-CNN controls provide complete-system context. Results are single-seed, two-walk capacity characterisation and do not identify an optimal LSTM depth.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "valid": True,
        "validation_counts": validation_counts,
        "result_rows": len(result_rows),
        "paired_differences": len(difference_rows),
        "report_root": str(output.resolve()),
    }
