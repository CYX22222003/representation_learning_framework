from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation-learning-matplotlib")


CONTRASTS = (("D1", "D0"), ("D2", "D0"), ("D3", "D1"), ("D4", "D1"), ("D4", "D3"))


def _metric(pred: np.ndarray, target: np.ndarray, name: str) -> float:
    error = pred - target
    if name == "mae":
        return float(np.mean(np.abs(error)))
    if name == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if name == "mse":
        return float(np.mean(error**2))
    if name == "corr":
        return float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 0 and np.std(target) > 0 else float("nan")
    raise ValueError(name)


def _contract_stats(pred: np.ndarray, target: np.ndarray, contracts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(contracts)
    rows = []
    for cid in ids:
        mask = contracts == cid
        p = pred[mask].astype(np.float64)
        y = target[mask].astype(np.float64)
        error = p - y
        rows.append((len(p), np.abs(error).sum(), np.square(error).sum(), p.sum(), y.sum(),
                     np.square(p).sum(), np.square(y).sum(), (p * y).sum()))
    return ids, np.asarray(rows, np.float64)


def _metrics_from_stats(stats: np.ndarray, metric: str) -> np.ndarray:
    n, absolute, squared, sum_pred, sum_target, pred2, target2, cross = (stats[:, i] for i in range(8))
    if metric == "mae":
        return absolute / n
    if metric == "mse":
        return squared / n
    if metric == "rmse":
        return np.sqrt(squared / n)
    numerator = cross - sum_pred * sum_target / n
    denominator = np.sqrt(np.maximum(pred2 - sum_pred**2 / n, 0) * np.maximum(target2 - sum_target**2 / n, 0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def paired_contract_interval(
    candidate: Path, reference: Path, metric: str, replicates: int = 2000, seed: int = 20260914,
) -> dict[str, float]:
    with np.load(candidate, allow_pickle=False) as c, np.load(reference, allow_pickle=False) as r:
        for key in ("targets", "row_indices", "contract_ids", "window_starts"):
            if not np.array_equal(c[key], r[key]):
                raise ValueError(f"paired prediction identity mismatch: {key}")
        cp, rp, target, contracts = c["predictions"], r["predictions"], c["targets"], c["contract_ids"]
    ids, candidate_stats = _contract_stats(cp, target, contracts)
    reference_ids, reference_stats = _contract_stats(rp, target, contracts)
    if not np.array_equal(ids, reference_ids):
        raise ValueError("paired contract identities differ")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    counts = np.stack([np.bincount(row, minlength=len(ids)) for row in draws])
    differences = _metrics_from_stats(counts @ candidate_stats, metric) - _metrics_from_stats(counts @ reference_stats, metric)
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    return {"difference": _metric(cp, target, metric) - _metric(rp, target, metric),
            "ci95_low": float(low), "ci95_high": float(high)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_matrix(root: Path, paths: list[Path]) -> dict[str, object]:
    matrix = json.loads((root / "matrix_manifest.json").read_text())
    expected = {(task, decoder, seed, epoch) for task in matrix["tasks"] for decoder in matrix["decoders"]
                for seed in matrix["seeds"] for epoch in matrix["epoch_budgets"]}
    observed = set()
    source_sets: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        row = json.loads(path.read_text())
        observed.add((row["task"], row["decoder_id"], row["seed"], row["epoch"]))
        run = path.parents[1]
        manifest = json.loads((run / "dataset_manifest.json").read_text())
        replay = json.loads((path.parent / "replay.json").read_text())
        if not replay.get("verified") or not replay.get("prediction_values_verified"):
            raise ValueError(f"replay failed or is incomplete: {path}")
        for name, value in manifest.get("source_sha256", {}).items():
            source_key = f"{name}::{row['task']}" if name == "labels_npz" else name
            source_sets[source_key].add(value)
    if observed != expected:
        raise ValueError(f"matrix is incomplete: missing={len(expected-observed)}, unexpected={len(observed-expected)}")
    if any(len(values) != 1 for values in source_sets.values()):
        raise ValueError("source checksum mismatch across runs")
    first = json.loads((paths[0].parents[1] / "dataset_manifest.json").read_text())
    source_paths = {"features_npz": first.get("features_npz"),
                    "features_index_npz": f"{first.get('features_npz')}.index.npz",
                    "temporal_index_npz": first.get("temporal_index_npz")}
    for name, path_value in source_paths.items():
        if path_value and name in source_sets:
            path = Path(path_value)
            if not path.exists() or _sha(path) not in source_sets[name]:
                raise ValueError(f"current source checksum mismatch: {name}")
    return matrix


def _write_plots(root: Path, records: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    for task in sorted({str(row["task"]) for row in records}):
        rows = [row for row in records if row["task"] == task]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for decoder in sorted({str(row["decoder_id"]) for row in rows}):
            selected = sorted((row for row in rows if row["decoder_id"] == decoder), key=lambda row: int(row["epoch"]))
            epochs = [row["epoch"] for row in selected]
            for axis, metric in zip(axes, ("mae", "rmse", "corr")):
                means = np.asarray([row[f"{metric}_mean"] for row in selected])
                stds = np.asarray([row[f"{metric}_std"] for row in selected])
                axis.plot(epochs, means, marker="o", label=decoder)
                axis.fill_between(epochs, means - stds, means + stds, alpha=0.12)
                axis.set_title(metric.upper())
                axis.set_xlabel("epoch")
                axis.grid(alpha=0.25)
        axes[0].set_ylabel("test metric")
        axes[-1].legend()
        fig.suptitle(f"Phase 2 decoder refinement: {task}")
        fig.tight_layout()
        fig.savefig(root / f"{task}_decoder_metrics.png", dpi=180)
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate the Phase 2 decoder matrix.")
    parser.add_argument("matrix_root")
    args = parser.parse_args(argv)
    root = Path(args.matrix_root)
    try:
        paths = sorted(root.glob("*/*/seed*/e*/metrics.json"))
        if not paths:
            raise FileNotFoundError(f"no decoder metrics under {root}")
        matrix = _verify_matrix(root, paths)
        groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
        identities: dict[tuple[str, int, int], str] = {}
        run_index: dict[tuple[str, str, int, int], Path] = {}
        resources: dict[tuple[str, str, int], tuple[dict[str, object], int]] = {}
        for path in paths:
            row = json.loads(path.read_text())
            run = path.parents[1]
            manifest = json.loads((run / "dataset_manifest.json").read_text())
            key = (row["task"], int(row["epoch"]), int(row["seed"]))
            if key in identities and identities[key] != manifest["test_identity_hash"]:
                raise ValueError(f"identity mismatch for {key}")
            identities[key] = manifest["test_identity_hash"]
            groups[(row["task"], row["decoder_id"], int(row["epoch"]))].append(row)
            run_index[(row["task"], row["decoder_id"], int(row["seed"]), int(row["epoch"]))] = path.parent / "predictions.npz"
            resources[(row["task"], row["decoder_id"], int(row["seed"]))] = (
                json.loads((run / "timing.json").read_text()), row["parameter_count"])
        records = []
        for (task, decoder, epoch), rows in sorted(groups.items()):
            record = {"task": task, "decoder_id": decoder, "epoch": epoch, "n_seeds": len(rows)}
            for metric in ("mae", "rmse", "mse", "corr"):
                values = np.asarray([row[metric] for row in rows], float)
                record[f"{metric}_mean"] = float(np.nanmean(values))
                record[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            record["parameter_count"] = rows[0]["parameter_count"]
            records.append(record)
        intervals = []
        for candidate, reference in CONTRASTS:
            for task in matrix["tasks"]:
                for seed in matrix["seeds"]:
                    for epoch in matrix["epoch_budgets"]:
                        candidate_path = run_index[(task, candidate, seed, epoch)]
                        reference_path = run_index[(task, reference, seed, epoch)]
                        for metric in ("mae", "rmse", "mse", "corr"):
                            intervals.append({"contrast": f"{candidate}-{reference}", "task": task,
                                              "decoder_id": candidate, "reference_id": reference, "seed": seed,
                                              "epoch": epoch, "metric": metric,
                                              **paired_contract_interval(candidate_path, reference_path, metric)})
        contract_groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
        for path in paths:
            row = json.loads(path.read_text())
            per_contract = json.loads((path.parent / "per_contract_metrics.json").read_text())
            for cid, metrics in per_contract.items():
                contract_groups[(row["task"], row["decoder_id"], row["epoch"], cid)].append(metrics)
        per_contract_records = []
        for (task, decoder, epoch, cid), rows in sorted(contract_groups.items()):
            item = {"task": task, "decoder_id": decoder, "epoch": epoch, "contract_id": int(cid),
                    "n_seeds": len(rows), "count": rows[0]["count"]}
            for metric in ("mae", "rmse", "mse", "corr"):
                values = np.asarray([row[metric] for row in rows], float)
                item[f"{metric}_mean"] = float(np.nanmean(values))
                item[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            per_contract_records.append(item)
        resource_records = [{"task": task, "decoder_id": decoder, "seed": seed,
                             "parameter_count": parameters, **timing}
                            for (task, decoder, seed), (timing, parameters) in sorted(resources.items())]
        (root / "aggregate_metrics.json").write_text(json.dumps(records, indent=2, allow_nan=True))
        (root / "paired_intervals.json").write_text(json.dumps(intervals, indent=2, allow_nan=True))
        (root / "aggregate_per_contract_metrics.json").write_text(json.dumps(per_contract_records, indent=2, allow_nan=True))
        (root / "resource_metrics.json").write_text(json.dumps(resource_records, indent=2, allow_nan=True))
        lines = ["# Phase 2 decoder results", "",
                 "All fixed budgets are reported; no best-on-test decoder or epoch is selected.", "",
                 "| task | decoder | epoch | seeds | MAE | RMSE | correlation | parameters |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in records:
            lines.append(f"| {row['task']} | {row['decoder_id']} | {row['epoch']} | {row['n_seeds']} | "
                         f"{row['mae_mean']:.6f} ± {row['mae_std']:.6f} | {row['rmse_mean']:.6f} ± {row['rmse_std']:.6f} | "
                         f"{row['corr_mean']:.6f} ± {row['corr_std']:.6f} | {row['parameter_count']} |")
        lines.extend(["", "Current-test findings are characterization evidence; confirmation requires a fresh temporally later holdout."])
        (root / "aggregate_summary.md").write_text("\n".join(lines) + "\n")
        _write_plots(root, records)
        print(f"wrote replay-verified decoder reports under {root}")
        return 0
    except Exception as exc:
        print(f"decoder report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
