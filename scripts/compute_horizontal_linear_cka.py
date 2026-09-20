"""Record full-train linear CKA for a horizontal encoder feature bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.feature_store import NpzFeatureStore  # noqa: E402


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("CKA inputs must be non-empty aligned matrices")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x64 -= x64.mean(axis=0, keepdims=True)
    y64 -= y64.mean(axis=0, keepdims=True)
    cross = x64.T @ y64
    xx = x64.T @ x64
    yy = y64.T @ y64
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("CKA denominator is zero or non-finite")
    result = float(np.square(cross).sum() / denominator)
    if not np.isfinite(result):
        raise FloatingPointError("linear CKA is non-finite")
    return result


def compute_bundle_cka(
    features_npz: Path,
    branches: list[str],
    duplicate_aliases: list[tuple[str, str]],
) -> tuple[list[str], np.ndarray]:
    with np.load(Path(f"{features_npz}.index.npz"), allow_pickle=False) as index:
        train_size = int(index["train_size"])
    available = NpzFeatureStore(str(features_npz)).load().as_branch_dict()
    unknown = sorted(set(branches).difference(available))
    if unknown:
        raise ValueError(f"unknown CKA branches: {unknown}")
    values = {name: np.asarray(available[name][:train_size], dtype=np.float32) for name in branches}
    for alias, source in duplicate_aliases:
        if alias in values or source not in values:
            raise ValueError(f"invalid CKA duplicate alias {alias}={source}")
        values[alias] = values[source]
    names = list(values)
    matrix = np.empty((len(names), len(names)), dtype=np.float64)
    for i, left in enumerate(names):
        for j in range(i, len(names)):
            score = linear_cka(values[left], values[names[j]])
            matrix[i, j] = score
            matrix[j, i] = score
    return names, matrix


def _aliases(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return []
    result = []
    for item in text.split(","):
        parts = [value.strip() for value in item.split("=", 1)]
        if len(parts) != 2 or not all(parts):
            raise ValueError("aliases must use alias=source")
        result.append((parts[0], parts[1]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--branches", required=True)
    parser.add_argument("--duplicate-aliases", default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    json_path = args.out_root / "linear_cka.json"
    npz_path = args.out_root / "linear_cka.npz"
    if (json_path.exists() or npz_path.exists()) and not args.overwrite:
        parser.error("CKA output exists; pass --overwrite to replace it")
    try:
        names, matrix = compute_bundle_cka(
            args.features_npz,
            [name.strip() for name in args.branches.split(",") if name.strip()],
            _aliases(args.duplicate_aliases),
        )
        args.out_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, branch_names=np.asarray(names), linear_cka=matrix)
        json_path.write_text(json.dumps({
            "features_npz": str(args.features_npz), "split": "train",
            "accumulation_dtype": "float64", "branch_names": names,
            "linear_cka": matrix.tolist(),
        }, indent=2), encoding="utf-8")
        print(f"horizontal linear CKA recorded: {json_path}")
        return 0
    except Exception as exc:
        print(f"horizontal CKA failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
