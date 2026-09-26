#!/usr/bin/env python3
"""Report completed Phase 6 H=8 neural runs without selecting a model."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase6_volatility import MODELS, volatility_metrics  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402


VOLATILITY_ROOT = ROOT / "experiments" / "phase6" / "volatility_prediction"
OUTPUT_ROOT = VOLATILITY_ROOT / "reports" / "initial_neural_seed0"
DISPLAY = {
    "framework_h0": "Canonical framework H0",
    "raw_ohlcv_mlp": "Raw-OHLCV MLP",
    "raw_lstm": "Raw LSTM",
    "zero": "Exact zero",
    "training_median": "Training median",
    "historical_persistence": "Historical persistence",
}


def _row(walk: int, model: str, metrics: dict[str, object], kind: str) -> dict[str, object]:
    return {
        "walk": walk,
        "model": model,
        "display_name": DISPLAY[model],
        "kind": kind,
        **{name: metrics[name] for name in ("count", "mae", "rmse", "mse", "pearson", "spearman")},
    }


def main() -> int:
    try:
        rows = []
        pooled: dict[str, dict[str, list[np.ndarray]]] = {}
        artifact_hashes = {}
        for walk in (1, 2):
            reference_added = False
            for model in MODELS:
                snapshot = VOLATILITY_ROOT / "runs" / model / f"walk{walk}" / "seed0" / "e50"
                predictions_path = snapshot / "predictions.npz"
                metrics_path = snapshot / "metrics.json"
                if not predictions_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"missing epoch-50 run artifacts: {snapshot}")
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(_row(walk, model, metrics_payload["model_metrics"], "learned"))
                artifact_hashes[f"walk{walk}/{model}/predictions"] = sha256_file(predictions_path)
                with np.load(predictions_path, allow_pickle=False) as stored:
                    target = np.asarray(stored["target_realised_variance"], dtype=np.float64)
                    prediction = np.asarray(
                        stored["prediction_realised_variance"], dtype=np.float64
                    )
                    pooled.setdefault(model, {"prediction": [], "target": []})[
                        "prediction"
                    ].append(prediction)
                    pooled[model]["target"].append(target)
                    if not reference_added:
                        references = {
                            "zero": np.asarray(stored["zero_reference"], dtype=np.float64),
                            "training_median": np.asarray(
                                stored["training_median_reference"], dtype=np.float64
                            ),
                            "historical_persistence": np.asarray(
                                stored["historical_persistence_reference"], dtype=np.float64
                            ),
                        }
                        for name, values in references.items():
                            metrics = volatility_metrics(values, target)
                            rows.append(_row(walk, name, metrics, "non_learned_reference"))
                            pooled.setdefault(name, {"prediction": [], "target": []})[
                                "prediction"
                            ].append(values)
                            pooled[name]["target"].append(target)
                        reference_added = True
        pooled_rows = []
        for model, values in pooled.items():
            metrics = volatility_metrics(
                np.concatenate(values["prediction"]), np.concatenate(values["target"])
            )
            pooled_rows.append(
                {
                    "walk": "pooled_descriptive",
                    "model": model,
                    "display_name": DISPLAY[model],
                    "kind": "learned" if model in MODELS else "non_learned_reference",
                    **{
                        name: metrics[name]
                        for name in ("count", "mae", "rmse", "mse", "pearson", "spearman")
                    },
                }
            )
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        pooled_frame = pd.DataFrame(pooled_rows)
        frame.to_csv(OUTPUT_ROOT / "epoch50_by_walk.csv", index=False)
        pooled_frame.to_csv(OUTPUT_ROOT / "epoch50_pooled_descriptive.csv", index=False)
        lines = [
            "# Phase 6 H=8 Initial Neural Volatility Report",
            "",
            "Epoch 50 is the predeclared principal snapshot. This is an interim target-validity report, not model selection: ten temporal/control configurations remain pending mandatory entries. GARCH--LSTM is deferred to a later round and is not part of the active matrix.",
            "",
            "## By walk",
            "",
            "| Walk | Model | Kind | MAE | RMSE | Pearson | Spearman |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            def fmt(name: str) -> str:
                value = float(row[name])
                return "nan" if not math.isfinite(value) else f"{value:.8g}"

            lines.append(
                f"| {row['walk']} | {row['display_name']} | {row['kind']} | "
                f"{fmt('mae')} | {fmt('rmse')} | {fmt('pearson')} | {fmt('spearman')} |"
            )
        lines.extend(
            [
                "",
                "## Pooled descriptive view",
                "",
                "Walks are concatenated only for descriptive scale. Walk-specific rows above remain the scientific comparison.",
                "",
                "| Model | Kind | MAE | RMSE | Pearson | Spearman |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in pooled_rows:
            def pooled_fmt(name: str) -> str:
                value = float(row[name])
                return "nan" if not math.isfinite(value) else f"{value:.8g}"

            lines.append(
                f"| {row['display_name']} | {row['kind']} | {pooled_fmt('mae')} | "
                f"{pooled_fmt('rmse')} | {pooled_fmt('pearson')} | {pooled_fmt('spearman')} |"
            )
        lines.extend(
            [
                "",
                "All values are in raw future realised-variance units. Correlation alone is not evidence of usefulness; learned rows must be interpreted against historical persistence and the zero-inflated, heavy-tailed target.",
            ]
        )
        report_path = OUTPUT_ROOT / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            OUTPUT_ROOT / "manifest.json",
            {
                "phase": 6,
                "task": "future_realised_variance_h8",
                "principal_epoch": 50,
                "model_selection_performed": False,
                "matrix_complete": False,
                "pending_mandatory_entries": [
                    "HC-SL",
                    "HC-ST",
                    "HB-SL",
                    "HB-ST",
                    "HC-AL",
                    "HC-AT",
                    "HB-AL",
                    "HB-AT",
                    "HC-DC",
                    "HB-DC",
                ],
                "deferred_next_round": ["adapted_garch_lstm"],
                "prediction_artifact_hashes": artifact_hashes,
                "by_walk_csv_sha256": sha256_file(OUTPUT_ROOT / "epoch50_by_walk.csv"),
                "pooled_csv_sha256": sha256_file(
                    OUTPUT_ROOT / "epoch50_pooled_descriptive.csv"
                ),
                "report_sha256": sha256_file(report_path),
            },
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "report": str(report_path.resolve()),
                    "rows": len(rows),
                    "matrix_complete": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
