from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines.ta_mlp_baseline.ta_features import FEATURE_NAMES, compute_ta_features
from tasks.phase2_classification.labels import identity_hash, load_label_bundle


IDENTITY_FIELDS = ("indices", "contract_ids", "window_starts", "timestamps_ns", "labels")


def validate_ta_bundle(payload: dict[str, np.ndarray]) -> dict[str, object]:
    required = {
        "feature_names",
        *(f"{split}_{name}" for split in ("train", "test") for name in ("features", "label_positions", *IDENTITY_FIELDS)),
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"TA feature bundle missing {missing}")
    result: dict[str, object] = {"feature_count": int(len(payload["feature_names"]))}
    for split in ("train", "test"):
        features = np.asarray(payload[f"{split}_features"])
        positions = np.asarray(payload[f"{split}_label_positions"])
        if features.ndim != 2 or features.shape[1] != len(payload["feature_names"]):
            raise ValueError(f"invalid {split} TA feature shape: {features.shape}")
        if len(features) == 0 or not np.all(np.isfinite(features)):
            raise ValueError(f"{split} TA features are empty or non-finite")
        if positions.ndim != 1 or len(positions) != len(features) or np.any(np.diff(positions) <= 0):
            raise ValueError(f"invalid {split} label positions")
        for name in IDENTITY_FIELDS:
            if np.asarray(payload[f"{split}_{name}"]).shape != (len(features),):
                raise ValueError(f"invalid {split}_{name} shape")
        result[f"{split}_rows"] = int(len(features))
        result[f"{split}_identity_hash"] = identity_hash(
            payload[f"{split}_indices"], payload[f"{split}_contract_ids"], payload[f"{split}_window_starts"]
        )
    return result


def _features_for_contract(frame: pd.DataFrame, seq_len: int, train_sequence_count: int) -> pd.DataFrame:
    features = compute_ta_features(frame)
    visible_end = min(len(frame), train_sequence_count + seq_len - 1)
    volume = frame["volume"].iloc[:visible_end].astype(float)
    mean, std = float(volume.mean()), float(volume.std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    features = features.copy()
    features["zsVol"] = (frame.loc[features.index, "volume"].astype(float) - mean) / std
    return features.replace([np.inf, -np.inf], np.nan)


def build_ta_bundle(labels_path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    labels, label_manifest = load_label_bundle(labels_path)
    seq_len = int(label_manifest["seq_len"])
    data_dir = Path(label_manifest.get("data_dir", "data"))
    included = {int(row["contract_id"]): row for row in label_manifest["contracts"] if row.get("status") == "included"}
    output: dict[str, list[np.ndarray]] = {
        "train_features": [], "test_features": [], "train_label_positions": [], "test_label_positions": [],
    }
    availability = []
    for contract_id, contract in sorted(included.items()):
        frame = pd.read_feather(data_dir / str(contract["filename"])).ffill().interpolate()
        features = _features_for_contract(frame, seq_len, int(contract["train_sequence_count"]))
        row = {"contract_id": contract_id, "filename": contract["filename"]}
        for split in ("train", "test"):
            positions = np.flatnonzero(labels[f"{split}_contract_ids"] == contract_id)
            endpoints = labels[f"{split}_window_starts"][positions] + seq_len - 1
            selected = features.reindex(endpoints)[FEATURE_NAMES].to_numpy(dtype=np.float32)
            finite = np.all(np.isfinite(selected), axis=1)
            output[f"{split}_features"].append(selected[finite])
            output[f"{split}_label_positions"].append(positions[finite].astype(np.int64))
            row[f"{split}_requested"] = int(len(positions))
            row[f"{split}_eligible"] = int(finite.sum())
        availability.append(row)
    payload: dict[str, np.ndarray] = {"feature_names": np.asarray(FEATURE_NAMES)}
    for split in ("train", "test"):
        positions = np.concatenate(output[f"{split}_label_positions"]).astype(np.int64)
        values = np.concatenate(output[f"{split}_features"]).astype(np.float32)
        order = np.argsort(positions)
        positions, values = positions[order], values[order]
        payload[f"{split}_label_positions"] = positions
        payload[f"{split}_features"] = values
        for name in ("indices", "contract_ids", "window_starts", "timestamps_ns", "labels"):
            payload[f"{split}_{name}"] = np.asarray(labels[f"{split}_{name}"])[positions]
    manifest = {
        "task": "phase2_probability_movement_ta_features", "labels_npz": str(labels_path),
        "feature_count": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
        "volume_normalization": "per-contract mean/std from raw rows visible to the training sequence split only",
        "final_standardization": "performed by each runner using aligned training rows only",
        "availability": availability,
        "train_identity_hash": identity_hash(payload["train_indices"], payload["train_contract_ids"], payload["train_window_starts"]),
        "test_identity_hash": identity_hash(payload["test_indices"], payload["test_contract_ids"], payload["test_window_starts"]),
    }
    return payload, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build TA features and the strict Phase 2 eligible-row intersection.")
    result.add_argument("--labels-npz", required=True)
    result.add_argument("--out-path", default="data/features/phase2_ta_probability_movement_4h_h2_tau005.npz")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--verify", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = Path(args.out_path)
    try:
        if args.verify:
            with np.load(out, allow_pickle=False) as data:
                payload = {key: data[key].copy() for key in data.files}
            print(json.dumps(validate_ta_bundle(payload), indent=2))
            return 0
        if (out.exists() or Path(f"{out}.manifest.json").exists()) and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out}")
        payload, manifest = build_ta_bundle(Path(args.labels_npz))
        validation = validate_ta_bundle(payload)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **payload)
        Path(f"{out}.manifest.json").write_text(json.dumps({**manifest, **validation}, indent=2), encoding="utf-8")
        print(json.dumps({"wrote": str(out), **validation}, indent=2))
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"TA feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
