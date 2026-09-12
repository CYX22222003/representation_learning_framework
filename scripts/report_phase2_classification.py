from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tasks.phase2_classification.metrics import probabilistic_classification_metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Aggregate Phase 2 classification run artifacts.")
    result.add_argument("matrix_root")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.matrix_root)
    try:
        included_models = ("C1_raw_ohlcv_mlp", "C2_framework", "C5_ta_mlp")
        metric_paths = sorted(
            path
            for model in included_models
            for path in (root / model).glob("P*/seed*/e*/metrics.json")
        )
        if not metric_paths:
            raise FileNotFoundError(f"no Phase 2 metrics found under {root}")
        groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
        identity_by_scope: dict[str, tuple[str, str]] = {}
        reference_sources: dict[str, Path] = {}
        for path in metric_paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            manifest = json.loads((path.parents[1] / "dataset_manifest.json").read_text(encoding="utf-8"))
            replay_path = path.parent / "replay.json"
            if not replay_path.exists() or not json.loads(replay_path.read_text(encoding="utf-8")).get("verified"):
                raise ValueError(f"missing successful replay verification: {path.parent}")
            with np.load(path.parent / "predictions.npz", allow_pickle=False) as predictions:
                prediction_count = len(predictions["targets"])
            if prediction_count != int(manifest["test_sample_count"]):
                raise ValueError(f"prediction row count differs from manifest: {path}")
            scope = "ta_aligned" if manifest.get("alignment_manifest", {}).get("alignment_npz") else "full_rows"
            identities = (manifest["train_identity_hash"], manifest["test_identity_hash"])
            if scope in identity_by_scope and identity_by_scope[scope] != identities:
                raise ValueError(f"identity mismatch inside comparison scope {scope}: {path}")
            identity_by_scope[scope] = identities
            reference_sources.setdefault(scope, path)
            groups[(scope, str(row["model_id"]), str(row["protocol_id"]), int(row["epoch"]))].append(row)
        records = []
        metrics = ("accuracy", "macro_f1", "balanced_accuracy", "macro_roc_auc", "macro_average_precision")
        for (scope, model_id, protocol, epoch), rows in sorted(groups.items()):
            record: dict[str, object] = {
                "comparison_scope": scope, "model_id": model_id, "protocol_id": protocol,
                "epoch": epoch, "n_seeds": len(rows),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                record[f"{metric}_mean"] = float(np.nanmean(values))
                record[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            records.append(record)
        reference_metrics = {}
        for scope, metric_path in reference_sources.items():
            with np.load(metric_path.parent / "predictions.npz", allow_pickle=False) as predictions:
                targets = np.asarray(predictions["targets"], dtype=np.int64)
            sampler = json.loads(
                (metric_path.parents[1] / "sampling_manifest.json").read_text(encoding="utf-8")
            )
            priors = np.asarray(sampler["class_priors"], dtype=np.float64)
            stable_logits = np.full((len(targets), 3), -50.0, dtype=np.float64)
            stable_logits[:, 1] = 0.0
            prior_logits = np.broadcast_to(np.log(priors), (len(targets), 3)).copy()
            stable_metrics, _ = probabilistic_classification_metrics(
                stable_logits, targets, ["DOWN", "STABLE", "UP"]
            )
            prior_metrics, _ = probabilistic_classification_metrics(
                prior_logits, targets, ["DOWN", "STABLE", "UP"]
            )
            reference_metrics[scope] = {
                "always_stable": stable_metrics,
                "repeated_training_prior": prior_metrics,
            }
        (root / "reference_metrics.json").write_text(
            json.dumps(reference_metrics, indent=2),
            encoding="utf-8",
        )
        (root / "aggregate_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        lines = [
            "# Phase 2 probabilistic classification results", "",
            "Candidate protocols are P1U, P1O, and P2. P0, when present, is an untreated reference only.", "",
            "| scope | model | protocol | epoch | seeds | macro-F1 | balanced accuracy | ROC-AUC | PR-AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in records:
            lines.append(
                f"| {row['comparison_scope']} | {row['model_id']} | {row['protocol_id']} | "
                f"{row['epoch']} | {row['n_seeds']} | "
                f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
                f"{row['macro_roc_auc_mean']:.4f} ± {row['macro_roc_auc_std']:.4f} | "
                f"{row['macro_average_precision_mean']:.4f} ± {row['macro_average_precision_std']:.4f} |"
            )
        (root / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"Wrote {root / 'aggregate_metrics.json'}, {root / 'aggregate_summary.md'}, "
            f"and {root / 'reference_metrics.json'}"
        )
        return 0
    except Exception as exc:
        print(f"Phase 2 report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
