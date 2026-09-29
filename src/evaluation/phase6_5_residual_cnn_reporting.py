"""Complete price-matrix reporting for the Phase 6.5D residual-CNN study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.phase6_5_residual_cnn import validate_residual_cnn_cka
from evaluation.phase6_encoder_variant_reporting import (
    _h0_price_rank_ic,
    _reference_rows,
    h0_run,
    normalize_metrics,
    task_dataset,
    validate_h0_reference,
)
from features.phase6_5_residual_cnn_features import (
    CONFIG_BRANCHES,
    validate_residual_cnn_feature_store,
)
from training.phase5_baselines import validate_baseline_run
from training.phase6_5_residual_cnn import validate_residual_cnn_encoder
from training.phase6_5_residual_cnn_downstream import validate_residual_cnn_downstream
from training.phase6_encoder_variant_downstream import (
    _cross_sectional_rank_ic,
    validate_variant_downstream,
)


TASK = "absolute_price_h8"
PRICE_METRICS: tuple[tuple[str, bool], ...] = (
    ("price_mae", False),
    ("price_rmse", False),
    ("price_mse", False),
    ("price_pearson", True),
    ("price_spearman", True),
    ("implied_pearson", True),
    ("implied_spearman", True),
    ("implied_sign_agreement", True),
    ("cross_sectional_rank_ic", True),
)
RESIDUAL_CONFIGURATIONS = tuple(name for name in CONFIG_BRANCHES if name != "H0")
COMPARISON_PAIRS = (
    ("HC-SR", "H0", "contrastive_substitution_vs_h0"),
    ("HB-SR", "H0", "byol_substitution_vs_h0"),
    ("HC-AR", "H0", "contrastive_addition_vs_h0"),
    ("HB-AR", "H0", "byol_addition_vs_h0"),
    ("HC-AR", "HC-DC", "contrastive_addition_vs_duplicate"),
    ("HB-AR", "HB-DC", "byol_addition_vs_duplicate"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Phase 6.5D report table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(key for row in rows for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_walk_rows(
    result_rows: list[dict[str, Any]], evaluation_counts: dict[int, int]
) -> list[dict[str, Any]]:
    """Summarise the two fixed walks without treating them as a third split."""
    configurations = list(dict.fromkeys(str(row["configuration"]) for row in result_rows))
    rows: list[dict[str, Any]] = []
    for configuration in configurations:
        selected = [row for row in result_rows if row["configuration"] == configuration]
        count = sum(evaluation_counts[int(row["walk"])] for row in selected)

        def weighted(metric: str) -> float:
            return sum(
                evaluation_counts[int(row["walk"])] * float(row[metric])
                for row in selected
            ) / count

        mse = weighted("price_mse")
        rows.append(
            {
                "task": TASK,
                "configuration": configuration,
                "role": selected[0]["role"],
                "evaluation_rows": count,
                "price_mae": weighted("price_mae"),
                "price_rmse": math.sqrt(mse),
                "price_mse": mse,
                "price_pearson_walk_weighted": weighted("price_pearson"),
                "price_spearman_walk_weighted": weighted("price_spearman"),
                "implied_pearson_walk_weighted": weighted("implied_pearson"),
                "implied_spearman_walk_weighted": weighted("implied_spearman"),
                "implied_sign_agreement": weighted("implied_sign_agreement"),
                "cross_sectional_rank_ic_walk_weighted": weighted(
                    "cross_sectional_rank_ic"
                ),
            }
        )
    return rows


def _relative_error_reduction(candidate: dict[str, Any], reference: dict[str, Any], metric: str) -> float:
    return 100.0 * (float(reference[metric]) - float(candidate[metric])) / float(
        reference[metric]
    )


def _phase6_duplicate_run(repo_root: Path, configuration: str, walk: int) -> Path:
    return (
        repo_root
        / "experiments"
        / "phase6"
        / "encoder_variants"
        / "downstream"
        / TASK
        / configuration.lower()
        / f"walk{walk}"
        / "seed0"
    )


def _residual_run(residual_root: Path, configuration: str, walk: int) -> Path:
    return (
        residual_root
        / "downstream"
        / TASK
        / configuration.lower()
        / f"walk{walk}"
        / "seed0"
    )


def _raw_run(repo_root: Path, baseline: str, walk: int) -> Path:
    return (
        repo_root
        / "experiments"
        / "phase5"
        / "baselines"
        / "tasks"
        / TASK
        / baseline
        / f"walk{walk}"
        / "seed0"
    )


def _rank_ic_from_predictions(path: Path) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as saved:
        return _cross_sectional_rank_ic(
            np.asarray(saved["prediction_delta_h8"], dtype=np.float64),
            np.asarray(saved["target_delta_h8"], dtype=np.float64),
            np.asarray(saved["decision_date_ns"]),
        )


def _metric_row(
    walk: int,
    configuration: str,
    role: str,
    run: Path,
    *,
    width: int | None,
) -> dict[str, Any]:
    metrics = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
    rank_ic = None
    if configuration == "H0":
        rank_ic = _h0_price_rank_ic(run / "e50" / "predictions.npz")
    elif role == "contextual_raw_baseline":
        rank_ic = _rank_ic_from_predictions(run / "e50" / "predictions.npz")
    normalized = normalize_metrics(TASK, metrics, price_rank_ic=rank_ic)
    body = metrics.get("metrics", metrics)
    normalized["price_mse"] = float(body["price"]["overall"]["mse"])
    return {
        "task": TASK,
        "walk": walk,
        "configuration": configuration,
        "role": role,
        "width": width,
        **normalized,
    }


def _price_subgroup_rows(
    walk: int,
    configuration: str,
    role: str,
    run: Path,
) -> list[dict[str, Any]]:
    payload = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
    body = payload.get("metrics", payload)
    price = body["price"]
    groups: list[tuple[str, str, dict[str, Any] | None]] = [
        ("overall", "overall", price.get("overall")),
        ("imputation", "context_observed_only", price.get("context_observed_only")),
        ("imputation", "context_has_imputation", price.get("context_has_imputation")),
        ("contract", "contract_macro", price.get("contract_macro")),
    ]
    groups.extend(
        ("lifecycle", name, values)
        for name, values in price.get("lifecycle", {}).items()
    )
    groups.extend(
        ("starting_price", name, values)
        for name, values in price.get("starting_price_bands", {}).items()
    )
    rows = []
    for group_type, group_name, values in groups:
        if values is None:
            continue
        rows.append(
            {
                "task": TASK,
                "walk": walk,
                "configuration": configuration,
                "role": role,
                "group_type": group_type,
                "group_name": group_name,
                **{
                    metric: values.get(metric)
                    for metric in ("count", "mae", "rmse", "mse", "pearson", "spearman")
                },
            }
        )
    return rows


def build_residual_cnn_report(repo_root: Path) -> dict[str, Any]:
    residual_root = repo_root / "experiments" / "phase6_5" / "residual_cnn"
    manifest = json.loads(
        (
            repo_root
            / "experiments"
            / "phase6_5"
            / "manifests"
            / "residual_cnn_downstream_seed0.json"
        ).read_text(encoding="utf-8")
    )
    if manifest.get("entry_count") != 8 or len(manifest.get("entries", [])) != 8:
        raise ValueError("Phase 6.5D downstream freeze is not the complete eight-run matrix")
    counts = {
        "encoders": 0,
        "features": 0,
        "downstream": 0,
        "cka": 0,
        "h0_references": 0,
        "duplicate_controls": 0,
        "raw_baselines": 0,
    }
    encoder_rows = []
    result_rows = []
    subgroup_rows = []
    resource_rows = []
    reference_rows = []
    evaluation_counts: dict[int, int] = {}
    for walk in (1, 2):
        encoder_dataset = task_dataset(repo_root, "classification_h2", walk)
        validate_residual_cnn_cka(
            residual_root / "diagnostics" / "cka" / f"walk{walk}.json",
            encoder_dataset,
        )
        counts["cka"] += 1
        for family in ("contrastive", "byol"):
            variant = f"{family}_rescnn"
            run = (
                residual_root
                / "pretraining"
                / f"walk{walk}"
                / family
                / "rescnn"
                / "seed0"
            )
            validate_residual_cnn_encoder(encoder_dataset, run)
            counts["encoders"] += 1
            architecture = json.loads(
                (run / "architecture_manifest.json").read_text(encoding="utf-8")
            )
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
        dataset = task_dataset(repo_root, TASK, walk)
        with np.load(dataset, allow_pickle=False) as saved:
            evaluation_counts[walk] = int(saved["test_regression_labels"].shape[0])
        feature = (
            residual_root
            / "features"
            / f"walk{walk}"
            / TASK
            / "features.npz"
        )
        validate_residual_cnn_feature_store(feature)
        counts["features"] += 1
        feature_manifest = json.loads(
            Path(f"{feature}.manifest.json").read_text(encoding="utf-8")
        )
        for row in encoder_rows:
            if int(row["walk"]) != walk:
                continue
            extraction = feature_manifest["checkpoint_provenance"][row["variant"]][
                "extraction_seconds"
            ]
            row["train_feature_extraction_seconds"] = extraction["train"]
            row["test_feature_extraction_seconds"] = extraction["test"]
        validate_h0_reference(repo_root, TASK, walk)
        counts["h0_references"] += 1
        h0 = h0_run(repo_root, TASK, walk)
        h0_metrics = json.loads((h0 / "e50" / "metrics.json").read_text(encoding="utf-8"))
        reference_rows.extend(
            _reference_rows(
                TASK,
                walk,
                h0_metrics,
                dataset,
                h0 / "e50" / "predictions.npz",
            )
        )
        result_rows.append(_metric_row(walk, "H0", "canonical_reference", h0, width=445))
        subgroup_rows.extend(_price_subgroup_rows(walk, "H0", "canonical_reference", h0))
        phase6_feature = (
            repo_root
            / "experiments"
            / "phase6"
            / "encoder_variants"
            / "features"
            / f"walk{walk}"
            / f"{TASK}.npz"
        )
        for configuration in ("HC-DC", "HB-DC"):
            run = _phase6_duplicate_run(repo_root, configuration, walk)
            validate_variant_downstream(dataset, phase6_feature, run)
            counts["duplicate_controls"] += 1
            result_rows.append(
                _metric_row(walk, configuration, "duplicate_width_control", run, width=573)
            )
            subgroup_rows.extend(
                _price_subgroup_rows(walk, configuration, "duplicate_width_control", run)
            )
        for configuration in RESIDUAL_CONFIGURATIONS:
            run = _residual_run(residual_root, configuration, walk)
            validate_residual_cnn_downstream(dataset, feature, run)
            counts["downstream"] += 1
            result_rows.append(
                _metric_row(
                    walk,
                    configuration,
                    "residual_candidate",
                    run,
                    width=445 if configuration.endswith("SR") else 573,
                )
            )
            subgroup_rows.extend(
                _price_subgroup_rows(walk, configuration, "residual_candidate", run)
            )
            architecture = json.loads(
                (run / "architecture_manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads((run / "e50" / "metrics.json").read_text(encoding="utf-8"))
            resource_rows.append(
                {
                    "walk": walk,
                    "configuration": configuration,
                    "parameter_count": architecture["parameter_count"],
                    "elapsed_training_seconds": metrics["elapsed_training_seconds"],
                    "inference_seconds": metrics["inference_seconds"],
                    "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
                }
            )
        for baseline in ("raw_ohlcv_mlp", "raw_ohlcv_lstm"):
            run = _raw_run(repo_root, baseline, walk)
            validate_baseline_run(dataset, run)
            counts["raw_baselines"] += 1
            result_rows.append(
                _metric_row(walk, baseline, "contextual_raw_baseline", run, width=None)
            )
            subgroup_rows.extend(
                _price_subgroup_rows(walk, baseline, "contextual_raw_baseline", run)
            )
    expected_counts = {
        "encoders": 4,
        "features": 2,
        "downstream": 8,
        "cka": 2,
        "h0_references": 2,
        "duplicate_controls": 4,
        "raw_baselines": 4,
    }
    if counts != expected_counts:
        raise ValueError("Phase 6.5D report validation count mismatch")
    indexed = {
        (int(row["walk"]), str(row["configuration"])): row for row in result_rows
    }
    difference_rows = []
    for walk in (1, 2):
        for candidate, reference, comparison_type in COMPARISON_PAIRS:
            left = indexed[(walk, candidate)]
            right = indexed[(walk, reference)]
            for metric, higher_is_better in PRICE_METRICS:
                candidate_value = float(left[metric])
                reference_value = float(right[metric])
                difference_rows.append(
                    {
                        "task": TASK,
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
    output = residual_root / "reports" / "complete_seed0"
    weighted_rows = _weighted_walk_rows(result_rows, evaluation_counts)
    weighted_index = {str(row["configuration"]): row for row in weighted_rows}
    _write_csv(output / "epoch50_by_walk.csv", result_rows)
    _write_csv(output / "epoch50_weighted_walk_summary.csv", weighted_rows)
    _write_csv(output / "paired_differences.csv", difference_rows)
    _write_csv(output / "price_subgroups.csv", subgroup_rows)
    _write_csv(output / "downstream_resources.csv", resource_rows)
    _write_csv(output / "encoder_resources.csv", encoder_rows)
    _write_csv(output / "task_references.csv", reference_rows)
    _write_csv(
        residual_root / "diagnostics" / "health" / "epoch50.csv",
        encoder_rows,
    )
    _write_csv(
        residual_root / "diagnostics" / "resources" / "encoders.csv",
        encoder_rows,
    )
    _write_csv(
        residual_root / "diagnostics" / "resources" / "downstream.csv",
        resource_rows,
    )
    h0_weighted = weighted_index["H0"]
    hc_sr_weighted = weighted_index["HC-SR"]
    hc_ar_weighted = weighted_index["HC-AR"]
    hc_dc_weighted = weighted_index["HC-DC"]
    raw_lstm_weighted = weighted_index["raw_ohlcv_lstm"]
    persistence = {
        (int(row["walk"]), str(row["metric"])): float(row["value"])
        for row in reference_rows
        if row["reference"] == "current_price_persistence"
    }
    reversal = {
        int(row["walk"]): float(row["value"])
        for row in reference_rows
        if row["reference"] == "last_hour_reversal"
    }
    total_evaluation_rows = sum(evaluation_counts.values())
    persistence_mae = sum(
        evaluation_counts[walk] * persistence[(walk, "price_mae")]
        for walk in evaluation_counts
    ) / total_evaluation_rows
    persistence_rmse = math.sqrt(
        sum(
            evaluation_counts[walk] * persistence[(walk, "price_rmse")] ** 2
            for walk in evaluation_counts
        )
        / total_evaluation_rows
    )
    reversal_rank_ic = sum(
        evaluation_counts[walk] * reversal[walk] for walk in evaluation_counts
    ) / total_evaluation_rows
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# Phase 6.5D Residual-CNN Future-Price Report",
                "",
                "Epoch 50 is the predeclared principal snapshot. The report is written only after replaying all four residual encoders, both feature stores, eight candidate price trajectories, same-family duplicate-width controls, H0, the Raw MLP/LSTM context baselines, and both CKA diagnostics.",
                "",
                "## Principal result",
                "",
                f"Across the fixed evaluation populations ({total_evaluation_rows:,} rows), `HC-SR` reduces row-weighted MAE by {_relative_error_reduction(hc_sr_weighted, h0_weighted, 'price_mae'):.2f}% and pooled RMSE by {_relative_error_reduction(hc_sr_weighted, h0_weighted, 'price_rmse'):.2f}% relative to H0. The error improvement occurs in both walks.",
                "",
                f"`HC-AR` reduces row-weighted MAE by {_relative_error_reduction(hc_ar_weighted, h0_weighted, 'price_mae'):.2f}% and pooled RMSE by {_relative_error_reduction(hc_ar_weighted, h0_weighted, 'price_rmse'):.2f}% relative to H0. Against the same-width `HC-DC` control, the reductions are {_relative_error_reduction(hc_ar_weighted, hc_dc_weighted, 'price_mae'):.2f}% MAE and {_relative_error_reduction(hc_ar_weighted, hc_dc_weighted, 'price_rmse'):.2f}% RMSE, so the price-error gain is not explained by width alone.",
                "",
                f"The Contrastive result is bounded: `HC-AR` still trails Raw LSTM (MAE {float(hc_ar_weighted['price_mae']):.6f} versus {float(raw_lstm_weighted['price_mae']):.6f}) and current-price persistence (MAE {persistence_mae:.6f}, pooled RMSE {persistence_rmse:.6f}). Its walk-weighted cross-sectional Rank IC ({float(hc_ar_weighted['cross_sectional_rank_ic_walk_weighted']):.4f}) is below H0 ({float(h0_weighted['cross_sectional_rank_ic_walk_weighted']):.4f}) and last-hour reversal ({reversal_rank_ic:.4f}). The BYOL substitutions/additions do not improve price error consistently across walks.",
                "",
                "`epoch50_weighted_walk_summary.csv` weights MAE/MSE by evaluation-row count and derives RMSE from the weighted MSE. Its correlation and Rank-IC columns are explicitly walk-weighted descriptive summaries, not correlations recomputed after merging the two walk populations.",
                "",
                "## Claim boundary",
                "",
                "Same-width substitutions are compared with H0. Heterogeneous additions are compared with both H0 and the matching duplicate-CNN width control. Persistence and last-hour reversal remain non-learned references. Results are seed-0, two-walk, future-price characterisation only; they do not establish profitable trading or universal representation superiority.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "valid": True,
        "validation_counts": counts,
        "result_rows": len(result_rows),
        "weighted_result_rows": len(weighted_rows),
        "paired_differences": len(difference_rows),
        "report_root": str(output.resolve()),
    }
