#!/usr/bin/env python3
"""Aggregate and plot the replay-validated Phase 5 seed-0 framework probes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase5")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_processing.phase5_walks import CLASS_NAMES
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from training.phase5_downstream import regression_metrics, validate_downstream_run
from training.phase5_encoder import write_json


def _run_root(walk: int, task: str) -> Path:
    return Path(f"experiments/phase5/downstream/walk{walk}/{task}/seed0")


def _validate_all() -> list[dict[str, object]]:
    results = []
    for walk in (1, 2):
        dataset = Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")
        feature = Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz")
        scaler = Path(f"experiments/phase5/downstream/walk{walk}/feature_standardizer.npz")
        for task in ("regression", "classification"):
            results.append(validate_downstream_run(dataset, feature, scaler, _run_root(walk, task)))
    return results


def _plot_training_losses(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, task in zip(axes, ("regression", "classification")):
        for walk in (1, 2):
            with np.load(_run_root(walk, task) / "e50" / "history.npz", allow_pickle=False) as history:
                axis.plot(history["epochs"], history["train_loss"], label=f"Walk {walk}")
        axis.axvline(5, color="grey", linestyle="--", linewidth=0.8)
        axis.axvline(15, color="grey", linestyle="--", linewidth=0.8)
        axis.set_title(task.title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)


def main() -> None:
    validations = _validate_all()
    output = Path("experiments/phase5/reports/framework_downstream_seed0")
    output.mkdir(parents=True, exist_ok=True)
    per_walk: dict[str, object] = {}
    regression_pred = []
    regression_target = []
    classification_logits = []
    classification_target = []
    stable_logits = []
    prior_logits = []
    regression_walk = []
    classification_walk = []
    regression_ids = []
    classification_ids = []
    for walk in (1, 2):
        per_walk[f"walk{walk}"] = {}
        for task in ("regression", "classification"):
            root = _run_root(walk, task)
            metrics = json.loads((root / "e50" / "metrics.json").read_text(encoding="utf-8"))
            per_walk[f"walk{walk}"][task] = metrics
            with np.load(root / "e50" / "predictions.npz", allow_pickle=False) as prediction:
                if task == "regression":
                    regression_pred.append(np.asarray(prediction["prediction_delta"]))
                    regression_target.append(np.asarray(prediction["target_delta"]))
                    regression_walk.append(np.full(len(prediction["target_delta"]), walk, dtype=np.int8))
                    regression_ids.append(np.asarray(prediction["condition_ids"]))
                else:
                    labels = np.asarray(prediction["targets"])
                    classification_logits.append(np.asarray(prediction["logits"]))
                    classification_target.append(labels)
                    classification_walk.append(np.full(len(labels), walk, dtype=np.int8))
                    classification_ids.append(np.asarray(prediction["condition_ids"]))
                    stable = np.full((len(labels), 3), -30.0, dtype=np.float32)
                    stable[:, 1] = 0.0
                    stable_logits.append(stable)
                    imbalance = json.loads((root / "imbalance_manifest.json").read_text(encoding="utf-8"))
                    priors = np.asarray(imbalance["training_class_priors"], dtype=np.float64)
                    prior_logits.append(np.broadcast_to(np.log(priors), (len(labels), 3)).copy())

    reg_pred = np.concatenate(regression_pred)
    reg_target = np.concatenate(regression_target)
    cls_logits = np.concatenate(classification_logits)
    cls_target = np.concatenate(classification_target)
    pooled_classification, pooled_arrays = probabilistic_classification_metrics(
        cls_logits, cls_target, CLASS_NAMES
    )
    stable_metrics, _ = probabilistic_classification_metrics(
        np.concatenate(stable_logits), cls_target, CLASS_NAMES
    )
    prior_metrics, _ = probabilistic_classification_metrics(
        np.concatenate(prior_logits), cls_target, CLASS_NAMES
    )
    pooled = {
        "regression": {
            "framework": regression_metrics(reg_pred, reg_target),
            "exact_zero_reference": regression_metrics(np.zeros_like(reg_target), reg_target),
        },
        "classification": {
            "framework": pooled_classification,
            "always_stable_reference": stable_metrics,
            "walk_local_training_prior_reference": prior_metrics,
        },
    }
    np.savez_compressed(
        output / "pooled_regression_predictions.npz",
        prediction_delta=reg_pred,
        target_delta=reg_target,
        walk=np.concatenate(regression_walk),
        condition_ids=np.concatenate(regression_ids),
    )
    np.savez_compressed(
        output / "pooled_classification_predictions.npz",
        **pooled_arrays,
        walk=np.concatenate(classification_walk),
        condition_ids=np.concatenate(classification_ids),
    )
    summary = {
        "phase": 5,
        "scope": "canonical five-branch framework seed 0; learned baselines deferred",
        "principal_epoch": 50,
        "walk_models_fitted_independently_before_pooling": True,
        "validation": validations,
        "per_walk": per_walk,
        "pooled": pooled,
    }
    write_json(output / "summary.json", summary)
    reg = pooled["regression"]
    cls = pooled["classification"]
    lines = [
        "# Phase 5 Five-Branch Framework Downstream Probe (Seed 0)",
        "",
        "Epoch 50 was predeclared. Walks were trained and evaluated independently; pooled metrics were computed only after both out-of-future prediction sets existed.",
        "",
        "## Epoch-50 results by walk",
        "",
        "| Walk | Regression MAE | Regression RMSE | Pearson | Spearman | Classification macro-F1 | Balanced accuracy | Accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for walk in (1, 2):
        walk_values = per_walk[f"walk{walk}"]
        regression = walk_values["regression"]["framework"]["overall"]
        classification = walk_values["classification"]["framework"]
        lines.append(
            f"| {walk} | {regression['mae']:.8f} | {regression['rmse']:.8f} | "
            f"{regression['pearson']:.6f} | {regression['spearman']:.6f} | "
            f"{classification['macro_f1']:.6f} | {classification['balanced_accuracy']:.6f} | "
            f"{classification['accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pooled out-of-future predictions",
            "",
            f"- Regression framework: MAE {reg['framework']['mae']:.8f}, RMSE {reg['framework']['rmse']:.8f}, Pearson {reg['framework']['pearson']:.6f}, Spearman {reg['framework']['spearman']:.6f}.",
            f"- Regression exact-zero reference: MAE {reg['exact_zero_reference']['mae']:.8f}, RMSE {reg['exact_zero_reference']['rmse']:.8f}.",
            f"- Classification framework: macro-F1 {cls['framework']['macro_f1']:.6f}, balanced accuracy {cls['framework']['balanced_accuracy']:.6f}, accuracy {cls['framework']['accuracy']:.6f}.",
            f"- Always-STABLE reference: macro-F1 {cls['always_stable_reference']['macro_f1']:.6f}, balanced accuracy {cls['always_stable_reference']['balanced_accuracy']:.6f}, accuracy {cls['always_stable_reference']['accuracy']:.6f}.",
            f"- Walk-local training-prior reference: macro-F1 {cls['walk_local_training_prior_reference']['macro_f1']:.6f}, balanced accuracy {cls['walk_local_training_prior_reference']['balanced_accuracy']:.6f}, accuracy {cls['walk_local_training_prior_reference']['accuracy']:.6f}.",
            "",
            "These are framework-only seed-0 probing results, not final comparative claims. Learned baselines and additional seeds remain deferred.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _plot_training_losses(output)
    print(json.dumps({"complete": True, "report": str(output), "pooled": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
