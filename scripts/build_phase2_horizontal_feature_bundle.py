"""Build one provenance-tracked horizontal superset feature bundle."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(path: Path) -> dict[str, int]:
    with np.load(Path(f"{path}.index.npz"), allow_pickle=False) as data:
        return {"train_size": int(data["train_size"]), "test_size": int(data["test_size"])}


def build_horizontal_bundle(
    canonical_path: Path,
    candidate_paths: list[Path],
    output_path: Path,
) -> Path:
    related = [output_path, Path(f"{output_path}.index.npz"), Path(f"{output_path}.manifest.json")]
    if any(path.exists() for path in related):
        raise FileExistsError(f"horizontal bundle output exists: {output_path}")
    canonical_bundle = NpzFeatureStore(str(canonical_path)).load()
    branches = canonical_bundle.as_branch_dict()
    expected_canonical = ("statistical", "transformed", "vae", "contrastive", "byol")
    if tuple(branches) != expected_canonical:
        raise ValueError(f"canonical branches must be {expected_canonical}, got {tuple(branches)}")
    split = _index(canonical_path)
    expected_rows = split["train_size"] + split["test_size"]
    sources: list[dict[str, object]] = []
    additions: dict[str, np.ndarray] = {}
    for path in candidate_paths:
        if _index(path) != split:
            raise ValueError(f"candidate split differs from canonical: {path}")
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError(f"candidate must contain one named branch: {path}")
            name = data.files[0]
            values = np.asarray(data[name], dtype=np.float32)
        if name in branches or name in additions:
            raise ValueError(f"duplicate branch name {name!r}")
        if values.shape != (expected_rows, 128) or not np.isfinite(values).all():
            raise ValueError(f"candidate {name!r} must be finite [{expected_rows}, 128]")
        additions[name] = values
        sources.append({
            "path": str(path), "sha256": _sha256(path),
            "index_sha256": _sha256(Path(f"{path}.index.npz")), "branch": name,
        })
    bundle = FeatureBundle(
        statistical=canonical_bundle.statistical,
        transformed=canonical_bundle.transformed,
        neural_branches={**canonical_bundle.neural_branches, **additions},
    )
    NpzFeatureStore(str(output_path)).save(bundle)
    np.savez_compressed(Path(f"{output_path}.index.npz"), **split)
    manifest = {
        "artifact_type": "phase2_horizontal_encoder_superset",
        "canonical_features": str(canonical_path),
        "canonical_sha256": _sha256(canonical_path),
        "canonical_index_sha256": _sha256(Path(f"{canonical_path}.index.npz")),
        "candidate_sources": sources,
        "split_index": split,
        "branch_order": list(bundle.as_branch_dict()),
        "branch_shapes": {name: list(values.shape) for name, values in bundle.as_branch_dict().items()},
    }
    Path(f"{output_path}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-features", type=Path, default=Path("data/features/features_4h_seq64_top50_phase1.npz"))
    parser.add_argument("--candidate-features", type=Path, action="append", required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    related = [args.out_path, Path(f"{args.out_path}.index.npz"), Path(f"{args.out_path}.manifest.json")]
    if args.overwrite:
        for path in related:
            if path.exists():
                path.unlink()
    try:
        print(build_horizontal_bundle(args.canonical_features, args.candidate_features, args.out_path))
        return 0
    except Exception as exc:
        print(f"horizontal feature bundle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
