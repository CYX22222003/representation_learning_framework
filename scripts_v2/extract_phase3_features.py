"""Extract deterministic and all seven frozen Phase 3 encoder branches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.statistical import batch_statistical_features, statistical_feature_dim  # noqa: E402
from features.transform import batch_transform_features, transform_feature_dim  # noqa: E402
from phase3_encoder_common import (  # noqa: E402
    DEFAULT_ENCODER_ROOT, DEFAULT_FEATURE_NPZ, DEFAULT_PROCESSED_NPZ,
    ENCODER_VARIANTS, PHASE3_DATA_ROOT, refuse_occupied, require_phase3_path,
    resolve_device, sha256_arrays, sha256_file, write_json,
)
from train_phase3_encoder import EncoderConfig, _build_model  # noqa: E402
from validate_phase3_encoder_data import validate_bundle  # noqa: E402


def _checkpoint(encoder_root: Path, variant: str) -> Path:
    family, backbone = variant.split("_", 1)
    return encoder_root / family / backbone / "seed0" / "e50" / "checkpoint.pth"


@torch.no_grad()
def _extract_neural(sequences, variant, checkpoint_path, batch_size, device, dataset_sha):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("phase") != 3 or checkpoint.get("variant") != variant:
        raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
    if checkpoint.get("completed_epoch") != 50 or checkpoint.get("processed_npz_sha256") != dataset_sha:
        raise ValueError(f"checkpoint epoch/data provenance mismatch: {checkpoint_path}")
    config_payload = dict(checkpoint["training_config"])
    config_payload["epoch_budgets"] = tuple(config_payload["epoch_budgets"])
    config = EncoderConfig(**config_payload)
    model = _build_model(config, int(sequences.shape[1]), int(sequences.shape[2])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    output = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start:start + batch_size]).to(device)
        if variant == "vae_mlp": embedding = model.encode(batch)[0]
        elif variant.startswith("contrastive_"): embedding = model(batch)[0]
        else: embedding = model.encode(batch)
        output.append(embedding.cpu().numpy().astype(np.float32))
    values = np.concatenate(output)
    if not np.isfinite(values).all(): raise ValueError(f"non-finite {variant} features")
    return values


def _deterministic(sequences, progress_name):
    # Each output row depends only on its own already-causal input window.
    print(f"extracting {progress_name} statistical features: {len(sequences)} rows", flush=True)
    statistical = batch_statistical_features(sequences)
    print(f"extracting {progress_name} transformation features", flush=True)
    transformed = batch_transform_features(sequences)
    return statistical, transformed


def _validate_checkpoint_provenance(manifest: dict) -> None:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ENCODER_VARIANTS):
        raise ValueError("feature checkpoint inventory mismatch")
    for variant in ENCODER_VARIANTS:
        record = checkpoints[variant]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid checkpoint provenance record: {variant}")
        checkpoint_path = Path(record["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"recorded checkpoint missing: {checkpoint_path}")
        if record["sha256"] != sha256_file(checkpoint_path):
            raise ValueError(f"checkpoint source hash mismatch: {variant}")


def validate_features(path: Path, processed_npz: Path) -> dict:
    manifest_path = Path(f"{path}.manifest.json"); index_path = Path(f"{path}.index.npz")
    if not all(item.is_file() for item in (path, manifest_path, index_path)):
        raise FileNotFoundError("feature NPZ, index, and manifest are required")
    manifest = __import__("json").loads(manifest_path.read_text())
    if manifest["processed_npz_sha256"] != sha256_file(processed_npz): raise ValueError("feature source hash mismatch")
    if manifest["feature_npz_sha256"] != sha256_file(path): raise ValueError("feature file hash mismatch")
    _validate_checkpoint_provenance(manifest)
    with np.load(processed_npz, allow_pickle=False) as source, np.load(index_path, allow_pickle=False) as index, np.load(path, allow_pickle=False) as features:
        train_size, test_size = len(source["train"]), len(source["test"])
        if int(index["train_size"]) != train_size or int(index["test_size"]) != test_size: raise ValueError("feature split mismatch")
        expected = {"statistical": 70, "transformed": 55, "vae_mlp": 64, **{v: 128 for v in ENCODER_VARIANTS if v != "vae_mlp"}}
        if set(features.files) != set(expected): raise ValueError("feature branch inventory mismatch")
        for name, dim in expected.items():
            values = features[name]
            if values.shape != (train_size + test_size, dim) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"invalid feature branch {name}: {values.shape}/{values.dtype}")
        for split in ("train", "test"):
            if str(index[f"{split}_identity_hash"].item()) != manifest["identity_hashes"][split]: raise ValueError("feature identity mismatch")
    return {"valid": True, "train_size": train_size, "test_size": test_size, "branches": expected}


def extract(processed_npz, encoder_root, out_path, batch_size, device_name):
    validation = validate_bundle(processed_npz)
    out_path = require_phase3_path(out_path, PHASE3_DATA_ROOT / "features")
    index_path, manifest_path = Path(f"{out_path}.index.npz"), Path(f"{out_path}.manifest.json")
    refuse_occupied((out_path, index_path, manifest_path))
    device = resolve_device(device_name); dataset_sha = validation["processed_npz_sha256"]
    with np.load(processed_npz, allow_pickle=False) as data:
        train, test = np.asarray(data["train"], np.float32), np.asarray(data["test"], np.float32)
    train_stat, train_trans = _deterministic(train, "train")
    test_stat, test_trans = _deterministic(test, "test")
    payload = {"statistical": np.concatenate([train_stat, test_stat]), "transformed": np.concatenate([train_trans, test_trans])}
    checkpoints = {}
    for variant in ENCODER_VARIANTS:
        path = _checkpoint(encoder_root, variant)
        if not path.is_file(): raise FileNotFoundError(path)
        print(f"extracting {variant}", flush=True)
        payload[variant] = np.concatenate([
            _extract_neural(train, variant, path, batch_size, device, dataset_sha),
            _extract_neural(test, variant, path, batch_size, device, dataset_sha),
        ])
        checkpoints[variant] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    out_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out_path, **payload)
    np.savez_compressed(index_path, train_size=len(train), test_size=len(test),
        train_identity_hash=np.asarray(validation["identity_hashes"]["train"]),
        test_identity_hash=np.asarray(validation["identity_hashes"]["test"]))
    write_json(manifest_path, {"phase": 3, "purpose": "price_and_classification_frozen_features",
        "processed_npz": str(processed_npz.resolve()), "processed_npz_sha256": dataset_sha,
        "feature_npz": str(out_path.resolve()), "feature_npz_sha256": sha256_file(out_path),
        "identity_hashes": validation["identity_hashes"], "deterministic_methods": {
            "statistical": "window-local AR(5), residual summaries, GARCH(1,1)",
            "transformed": "window-local FFT magnitudes and Haar detail energies",
            "future_usage": "none; each feature row consumes only its corresponding input window",
        }, "checkpoints": checkpoints, "branch_shapes": {k: list(v.shape) for k,v in payload.items()}})
    return validate_features(out_path, processed_npz)


def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--processed-npz",type=Path,default=DEFAULT_PROCESSED_NPZ); p.add_argument("--encoder-root",type=Path,default=DEFAULT_ENCODER_ROOT); p.add_argument("--out-path",type=Path,default=DEFAULT_FEATURE_NPZ); p.add_argument("--batch-size",type=int,default=1024); p.add_argument("--device",default="auto"); p.add_argument("--verify",action="store_true"); a=p.parse_args(argv)
    try:
        result=validate_features(a.out_path,a.processed_npz) if a.verify else extract(a.processed_npz,a.encoder_root,a.out_path,a.batch_size,a.device); print(f"Phase 3 features valid: {result}"); return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"Phase 3 feature extraction failed: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
