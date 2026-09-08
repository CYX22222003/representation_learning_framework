"""Extract frozen train/test embeddings from a Phase-2 encoder checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.encoder_variants import (  # noqa: E402
    CONTRASTIVE_VARIANTS,
    TemporalBackboneConfig,
    build_contrastive_variant,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


@torch.no_grad()
def _extract(model, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=False)
    model.eval()
    for (batch,) in loader:
        outputs.append(model.encode(batch.to(device)).cpu().numpy().astype(np.float32))
    if not outputs:
        return np.empty((0, model.hidden_dim), dtype=np.float32)
    embeddings = np.concatenate(outputs, axis=0)
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("extracted embeddings contain non-finite values")
    return embeddings


def extract_feature_artifact(
    processed_npz: Path,
    checkpoint_path: Path,
    out_path: Path,
    batch_size: int,
    device_name: str,
) -> Path:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    variant = str(checkpoint.get("variant", ""))
    if variant not in CONTRASTIVE_VARIANTS:
        raise ValueError(f"checkpoint variant must be one of {CONTRASTIVE_VARIANTS}")
    with np.load(processed_npz) as data:
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    if train.ndim != 3 or test.ndim != 3 or train.shape[1:] != test.shape[1:]:
        raise ValueError("processed train/test arrays must share [N, sequence, features] shape")
    if int(checkpoint["seq_len"]) != train.shape[1] or int(checkpoint["input_dim"]) != train.shape[2]:
        raise ValueError("checkpoint sequence contract does not match processed data")
    recorded_hash = checkpoint.get("dataset_sha256")
    processed_hash = _sha256(processed_npz)
    if recorded_hash and recorded_hash != processed_hash:
        raise ValueError("checkpoint dataset checksum does not match processed data")

    backbone_config = TemporalBackboneConfig(**checkpoint["backbone_config"])
    model = build_contrastive_variant(
        variant,
        input_dim=train.shape[2],
        embedding_dim=int(checkpoint.get("embedding_dim", 128)),
        backbone_config=backbone_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    started_at = time.perf_counter()
    train_embeddings = _extract(model, train, batch_size, device)
    test_embeddings = _extract(model, test, batch_size, device)
    elapsed_seconds = time.perf_counter() - started_at

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate([train_embeddings, test_embeddings], axis=0)
    np.savez_compressed(out_path, **{variant: combined})
    np.savez_compressed(
        Path(f"{out_path}.index.npz"),
        train_size=np.asarray(len(train_embeddings), dtype=np.int64),
        test_size=np.asarray(len(test_embeddings), dtype=np.int64),
    )
    manifest = {
        "artifact_type": "phase2_frozen_encoder_branch",
        "branch_name": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "processed_npz": str(processed_npz),
        "processed_npz_sha256": processed_hash,
        "embedding_source": "TemporalContrastiveEncoder.encode(...)[backbone_state]",
        "embedding_dim": int(combined.shape[1]),
        "train_shape": list(train_embeddings.shape),
        "test_shape": list(test_embeddings.shape),
        "combined_shape": list(combined.shape),
        "device": str(device),
        "inference_seconds": elapsed_seconds,
    }
    Path(f"{out_path}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-npz", type=Path, default=Path("data/processed/market_4h_seq64_top50.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    related = [args.out_path, Path(f"{args.out_path}.index.npz"), Path(f"{args.out_path}.manifest.json")]
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if any(path.exists() for path in related) and not args.overwrite:
        parser.error("output artifact exists; pass --overwrite to replace it")
    try:
        path = extract_feature_artifact(
            args.processed_npz, args.checkpoint, args.out_path, args.batch_size, args.device
        )
    except Exception as exc:
        print(f"Phase-2 feature extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved frozen Phase-2 encoder branch: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
