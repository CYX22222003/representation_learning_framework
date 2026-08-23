# Validate a saved framework FeatureBundle and its split index.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from features.feature_store import NpzFeatureStore


DEFAULT_EXPECTED_DIMS = {
    "statistical": 70,
    "transformed": 55,
    "vae": 64,
    "contrastive": 128,
}


def _load_index(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split index: {path}")
    with np.load(path) as data:
        if "train_size" not in data or "test_size" not in data:
            raise ValueError(f"{path} must contain train_size and test_size")
        return {
            "train_size": int(data["train_size"]),
            "test_size": int(data["test_size"]),
        }


def _parse_expected_dims(items: list[str]) -> dict[str, int]:
    expected = dict(DEFAULT_EXPECTED_DIMS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected dimension override shaped name=dim, got {item!r}")
        name, dim = item.split("=", 1)
        expected[name.strip()] = int(dim)
    return expected


def _processed_shapes(path: Path | None) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "train" not in data or "test" not in data:
            raise ValueError(f"{path} must contain train and test arrays")
        return tuple(data["train"].shape), tuple(data["test"].shape)


def _array_summary(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    summary = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }
    if values.size and finite.any():
        finite_values = values[finite]
        summary.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "std": float(finite_values.std()),
            }
        )
    return summary


def validate_feature_store(
    feature_npz: Path,
    processed_npz: Path | None,
    expected_dims: dict[str, int],
    require_exact_branches: bool,
) -> dict:
    if not feature_npz.exists():
        raise FileNotFoundError(feature_npz)
    index_path = Path(f"{feature_npz}.index.npz")
    manifest_path = Path(f"{feature_npz}.manifest.json")

    bundle = NpzFeatureStore(str(feature_npz)).load()
    branches = bundle.as_branch_dict()
    branch_names = list(branches)
    expected_names = list(expected_dims)

    if require_exact_branches and set(branch_names) != set(expected_names):
        raise ValueError(
            f"Branch set mismatch: expected {expected_names}, found {branch_names}"
        )
    missing = [name for name in expected_names if name not in branches]
    if missing:
        raise ValueError(f"Missing required branches: {missing}")

    index = _load_index(index_path)
    total_rows = index["train_size"] + index["test_size"]
    if index["train_size"] <= 0 or index["test_size"] <= 0:
        raise ValueError(f"Invalid split sizes: {index}")

    processed = _processed_shapes(processed_npz)
    if processed is not None:
        train_shape, test_shape = processed
        if train_shape[0] != index["train_size"] or test_shape[0] != index["test_size"]:
            raise ValueError(
                "Split index does not match processed npz: "
                f"index={index}, processed_train={train_shape}, processed_test={test_shape}"
            )

    summaries: dict[str, dict] = {}
    for name, values in branches.items():
        if values.ndim != 2:
            raise ValueError(f"Branch {name!r} must be 2D [N, dim], got {values.shape}")
        if values.shape[0] != total_rows:
            raise ValueError(
                f"Branch {name!r} row count {values.shape[0]} does not match "
                f"train_size + test_size = {total_rows}"
            )
        if name in expected_dims and values.shape[1] != expected_dims[name]:
            raise ValueError(
                f"Branch {name!r} dim {values.shape[1]} does not match expected "
                f"{expected_dims[name]}"
            )
        if values.dtype != np.float32:
            raise ValueError(f"Branch {name!r} dtype must be float32, got {values.dtype}")
        if not np.isfinite(values).all():
            summary = _array_summary(values)
            raise ValueError(f"Branch {name!r} contains non-finite values: {summary}")
        summaries[name] = _array_summary(values)

    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        combined = manifest.get("combined_branch_shapes", {})
        for name, values in branches.items():
            recorded = combined.get(name)
            if recorded is not None and list(values.shape) != list(recorded):
                raise ValueError(
                    f"Manifest shape mismatch for {name!r}: manifest={recorded}, "
                    f"actual={list(values.shape)}"
                )

    return {
        "feature_npz": str(feature_npz),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "split_index": index,
        "processed_npz": str(processed_npz) if processed_npz else None,
        "processed_shapes": {
            "train": list(processed[0]),
            "test": list(processed[1]),
        }
        if processed is not None
        else None,
        "branches": summaries,
        "manifest_checked": manifest is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a framework feature store.")
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--processed-npz", type=Path)
    parser.add_argument(
        "--expected-dim",
        action="append",
        default=[],
        help="Override or add an expected branch dimension, e.g. vae=64",
    )
    parser.add_argument(
        "--allow-extra-branches",
        action="store_true",
        help="Allow branches beyond the expected MVP set.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    result = validate_feature_store(
        feature_npz=args.features_npz,
        processed_npz=args.processed_npz,
        expected_dims=_parse_expected_dims(args.expected_dim),
        require_exact_branches=not args.allow_extra_branches,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"feature store OK: {result['feature_npz']}")
    print(f"split index: {result['split_index']}")
    if result["processed_shapes"]:
        print(f"processed shapes: {result['processed_shapes']}")
    for name, summary in result["branches"].items():
        print(
            f"{name}: shape={tuple(summary['shape'])} dtype={summary['dtype']} "
            f"finite={summary['finite']} mean={summary.get('mean', 0.0):.6g} "
            f"std={summary.get('std', 0.0):.6g}"
        )
    print(f"manifest checked: {result['manifest_checked']}")


if __name__ == "__main__":
    main()
