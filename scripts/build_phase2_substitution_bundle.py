"""Build a provenance-tracked five-branch Phase-2 encoder substitution bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.feature_store import FeatureBundle, NpzFeatureStore  # noqa: E402


CANONICAL_BRANCHES = ("statistical", "transformed", "vae", "contrastive", "byol")
EXPECTED_WIDTHS = {"statistical": 70, "transformed": 55, "vae": 64, "byol": 128}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"train_size", "test_size"}:
            raise ValueError(f"unexpected split-index keys in {path}: {data.files}")
        result = {name: int(data[name]) for name in data.files}
    if min(result.values()) <= 0:
        raise ValueError(f"invalid split index: {result}")
    return result


def build_bundle(
    canonical_features: Path,
    candidate_features: Path,
    candidate_branch: str,
    output: Path,
) -> Path:
    canonical_index = Path(f"{canonical_features}.index.npz")
    candidate_index = Path(f"{candidate_features}.index.npz")
    if output.exists() or Path(f"{output}.index.npz").exists() or Path(f"{output}.manifest.json").exists():
        raise FileExistsError(f"output already exists: {output}")
    source = NpzFeatureStore(str(canonical_features)).load().as_branch_dict()
    if tuple(source) != CANONICAL_BRANCHES:
        raise ValueError(f"canonical branch order/set must be {CANONICAL_BRANCHES}, got {tuple(source)}")
    with np.load(candidate_features, allow_pickle=False) as candidate_data:
        if tuple(candidate_data.files) != (candidate_branch,):
            raise ValueError(
                f"candidate artifact must contain only {candidate_branch!r}, got {tuple(candidate_data.files)}"
            )
        candidate = {candidate_branch: np.asarray(candidate_data[candidate_branch], dtype=np.float32)}
    source_index, candidate_split = _read_index(canonical_index), _read_index(candidate_index)
    if source_index != candidate_split:
        raise ValueError(f"candidate split index {candidate_split} differs from canonical {source_index}")
    replacement = np.asarray(candidate[candidate_branch], dtype=np.float32)
    expected_rows = source_index["train_size"] + source_index["test_size"]
    if replacement.shape != (expected_rows, 128) or not np.isfinite(replacement).all():
        raise ValueError(f"candidate must be finite [N, 128], got {replacement.shape}")
    for name, width in EXPECTED_WIDTHS.items():
        values = np.asarray(source[name], dtype=np.float32)
        if values.shape != (expected_rows, width) or not np.isfinite(values).all():
            raise ValueError(f"invalid canonical branch {name}: {values.shape}")
    replacement_bundle = FeatureBundle(
        statistical=np.asarray(source["statistical"], dtype=np.float32),
        transformed=np.asarray(source["transformed"], dtype=np.float32),
        neural_branches={
            "vae": np.asarray(source["vae"], dtype=np.float32),
            candidate_branch: replacement,
            "byol": np.asarray(source["byol"], dtype=np.float32),
        },
    )
    NpzFeatureStore(str(output)).save(replacement_bundle)
    np.savez_compressed(Path(f"{output}.index.npz"), **source_index)
    manifest = {
        "artifact_type": "phase2_primary_five_branch_substitution",
        "canonical_features": str(canonical_features),
        "canonical_features_sha256": _sha256(canonical_features),
        "canonical_index_sha256": _sha256(canonical_index),
        "candidate_features": str(candidate_features),
        "candidate_features_sha256": _sha256(candidate_features),
        "candidate_index_sha256": _sha256(candidate_index),
        "replaced_branch": "contrastive",
        "inserted_branch": candidate_branch,
        "split_index": source_index,
        "combined_branch_shapes": {name: list(values.shape) for name, values in replacement_bundle.as_branch_dict().items()},
    }
    Path(f"{output}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-features", type=Path, default=Path("data/features/features_4h_seq64_top50_phase1.npz"))
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--candidate-branch", required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(build_bundle(args.canonical_features, args.candidate_features, args.candidate_branch, args.out_path))
        return 0
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Phase-2 substitution bundle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
