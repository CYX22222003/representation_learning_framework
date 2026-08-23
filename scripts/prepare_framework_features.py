# Build the branch-aware feature bundle used by the framework MVP.
#
# This script regenerates current deterministic features and attaches frozen
# neural embeddings from pretrained VAE and contrastive checkpoints. It writes
# one feature store with train+test rows concatenated, plus an index file that
# records the split boundary.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_processing.reader import load_processed_npz
from features.feature_store import FeatureBundle, NpzFeatureStore
from features.statistical import compute_statistical_features
from features.transform import compute_transform_features
from models.contrastive import ContrastiveEncoder
from models.vae import SequenceVAE


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain 'model_state_dict'")
    return checkpoint


def _check_sequence_contract(checkpoint: dict, sequences: np.ndarray, branch_name: str) -> None:
    _, seq_len, input_dim = sequences.shape
    if "seq_len" in checkpoint and int(checkpoint["seq_len"]) != int(seq_len):
        raise ValueError(
            f"{branch_name} checkpoint seq_len={checkpoint['seq_len']} does not match "
            f"processed sequences seq_len={seq_len}"
        )
    if "input_dim" in checkpoint and int(checkpoint["input_dim"]) != int(input_dim):
        raise ValueError(
            f"{branch_name} checkpoint input_dim={checkpoint['input_dim']} does not match "
            f"processed sequences input_dim={input_dim}"
        )


def _make_loader(sequences: np.ndarray, batch_size: int) -> DataLoader:
    tensor = torch.tensor(sequences, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)


def _compute_statistical_with_progress(
    sequences: np.ndarray,
    ar_order: int,
    split_name: str,
    progress_every: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total = len(sequences)
    for idx, sequence in enumerate(sequences, start=1):
        outputs.append(compute_statistical_features(sequence, ar_order=ar_order))
        if idx == total or idx % progress_every == 0:
            print(
                f"{split_name} statistical features: {idx}/{total} sequences",
                flush=True,
            )
    return np.stack(outputs, axis=0)


def _compute_transform_with_progress(
    sequences: np.ndarray,
    fft_top_k: int,
    wavelet_levels: int,
    split_name: str,
    progress_every: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total = len(sequences)
    for idx, sequence in enumerate(sequences, start=1):
        outputs.append(
            compute_transform_features(
                sequence,
                fft_top_k=fft_top_k,
                wavelet_levels=wavelet_levels,
            )
        )
        if idx == total or idx % progress_every == 0:
            print(
                f"{split_name} transformed features: {idx}/{total} sequences",
                flush=True,
            )
    return np.stack(outputs, axis=0)


@torch.no_grad()
def _extract_vae_embeddings(
    sequences: np.ndarray,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    _check_sequence_contract(checkpoint, sequences, "vae")

    _, seq_len, input_dim = sequences.shape
    latent_dim = int(checkpoint.get("latent_dim", 64))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    model = SequenceVAE(
        seq_len=seq_len,
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs: list[np.ndarray] = []
    for (batch,) in _make_loader(sequences, batch_size):
        mu, _ = model.encode(batch.to(device))
        outputs.append(mu.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(outputs, axis=0)
    return embeddings, {
        "checkpoint": str(checkpoint_path),
        "completed_epoch": int(checkpoint.get("completed_epoch", -1)),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "embedding_source": "SequenceVAE.encode(...)[mu]",
        "output_shape": list(embeddings.shape),
    }


@torch.no_grad()
def _extract_contrastive_embeddings(
    sequences: np.ndarray,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    _check_sequence_contract(checkpoint, sequences, "contrastive")

    _, _, input_dim = sequences.shape
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    embedding_dim = int(checkpoint.get("embedding_dim", 128))
    model = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs: list[np.ndarray] = []
    for (batch,) in _make_loader(sequences, batch_size):
        h, _ = model(batch.to(device))
        outputs.append(h.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(outputs, axis=0)
    return embeddings, {
        "checkpoint": str(checkpoint_path),
        "completed_epoch": int(checkpoint.get("completed_epoch", -1)),
        "hidden_dim": hidden_dim,
        "projector_embedding_dim": embedding_dim,
        "embedding_source": "ContrastiveEncoder.forward(...)[h]",
        "output_shape": list(embeddings.shape),
    }


def _build_split_bundle(
    sequences: np.ndarray,
    split_name: str,
    ar_order: int,
    fft_top_k: int,
    wavelet_levels: int,
    vae_checkpoint: Path,
    contrastive_checkpoint: Path,
    batch_size: int,
    device: torch.device,
    progress_every: int,
) -> tuple[FeatureBundle, dict]:
    statistical = _compute_statistical_with_progress(
        sequences,
        ar_order=ar_order,
        split_name=split_name,
        progress_every=progress_every,
    )
    transformed = _compute_transform_with_progress(
        sequences,
        fft_top_k=fft_top_k,
        wavelet_levels=wavelet_levels,
        split_name=split_name,
        progress_every=progress_every,
    )
    vae, vae_manifest = _extract_vae_embeddings(
        sequences=sequences,
        checkpoint_path=vae_checkpoint,
        batch_size=batch_size,
        device=device,
    )
    contrastive, contrastive_manifest = _extract_contrastive_embeddings(
        sequences=sequences,
        checkpoint_path=contrastive_checkpoint,
        batch_size=batch_size,
        device=device,
    )
    bundle = FeatureBundle(
        statistical=statistical,
        transformed=transformed,
        neural_branches={
            "vae": vae,
            "contrastive": contrastive,
        },
    )
    return bundle, {
        "statistical": list(statistical.shape),
        "transformed": list(transformed.shape),
        "vae": vae_manifest,
        "contrastive": contrastive_manifest,
    }


def build_and_save_framework_features(
    processed_npz: Path,
    out_path: Path,
    vae_checkpoint: Path,
    contrastive_checkpoint: Path,
    ar_order: int,
    fft_top_k: int,
    wavelet_levels: int,
    batch_size: int,
    device_name: str,
    progress_every: int,
    overwrite: bool,
) -> Path:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")
    index_path = Path(f"{out_path}.index.npz")
    manifest_path = Path(f"{out_path}.manifest.json")
    if (index_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"{index_path} or {manifest_path} already exists; pass --overwrite to replace them"
        )

    train, test = load_processed_npz(str(processed_npz))
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("Processed train/test arrays must be shaped [N, seq_len, features]")
    if train.shape[1:] != test.shape[1:]:
        raise ValueError(f"Train/test sequence shapes differ: {train.shape} vs {test.shape}")

    device = _resolve_device(device_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"building train features: sequences={train.shape}", flush=True)
    train_bundle, train_manifest = _build_split_bundle(
        sequences=train,
        split_name="train",
        ar_order=ar_order,
        fft_top_k=fft_top_k,
        wavelet_levels=wavelet_levels,
        vae_checkpoint=vae_checkpoint,
        contrastive_checkpoint=contrastive_checkpoint,
        batch_size=batch_size,
        device=device,
        progress_every=progress_every,
    )
    print(f"building test features: sequences={test.shape}", flush=True)
    test_bundle, test_manifest = _build_split_bundle(
        sequences=test,
        split_name="test",
        ar_order=ar_order,
        fft_top_k=fft_top_k,
        wavelet_levels=wavelet_levels,
        vae_checkpoint=vae_checkpoint,
        contrastive_checkpoint=contrastive_checkpoint,
        batch_size=batch_size,
        device=device,
        progress_every=progress_every,
    )

    combined = FeatureBundle(
        statistical=np.concatenate(
            [train_bundle.statistical, test_bundle.statistical],
            axis=0,
        ),
        transformed=np.concatenate(
            [train_bundle.transformed, test_bundle.transformed],
            axis=0,
        ),
        neural_branches={
            "vae": np.concatenate(
                [train_bundle.neural_branches["vae"], test_bundle.neural_branches["vae"]],
                axis=0,
            ),
            "contrastive": np.concatenate(
                [
                    train_bundle.neural_branches["contrastive"],
                    test_bundle.neural_branches["contrastive"],
                ],
                axis=0,
            ),
        },
    )
    NpzFeatureStore(str(out_path)).save(combined)
    np.savez_compressed(
        index_path,
        train_size=len(train),
        test_size=len(test),
    )
    manifest = {
        "processed_npz": str(processed_npz),
        "out_path": str(out_path),
        "index_path": str(index_path),
        "train_sequence_shape": list(train.shape),
        "test_sequence_shape": list(test.shape),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "device": str(device),
        "deterministic_config": {
            "ar_order": ar_order,
            "fft_top_k": fft_top_k,
            "wavelet_levels": wavelet_levels,
            "progress_every": progress_every,
        },
        "train_branch_shapes": train_manifest,
        "test_branch_shapes": test_manifest,
        "combined_branch_shapes": {
            name: list(values.shape)
            for name, values in combined.as_branch_dict().items()
        },
    }
    _json_write(manifest_path, manifest)
    print(f"saved feature store: {out_path}", flush=True)
    print(f"saved split index: {index_path}", flush=True)
    print(f"saved manifest: {manifest_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build framework feature bundle with deterministic and frozen SSL branches."
    )
    parser.add_argument("--processed-npz", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path("checkpoints/vae_4h_seq64_top50.pth"),
    )
    parser.add_argument(
        "--contrastive-checkpoint",
        type=Path,
        default=Path("checkpoints/contrastive_4h_seq64_top50.pth"),
    )
    parser.add_argument("--ar-order", type=int, default=5)
    parser.add_argument("--fft-top-k", type=int, default=8)
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_and_save_framework_features(
        processed_npz=args.processed_npz,
        out_path=args.out_path,
        vae_checkpoint=args.vae_checkpoint,
        contrastive_checkpoint=args.contrastive_checkpoint,
        ar_order=args.ar_order,
        fft_top_k=args.fft_top_k,
        wavelet_levels=args.wavelet_levels,
        batch_size=args.batch_size,
        device_name=args.device,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
