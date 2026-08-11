# Stage 2 of the data pipeline.
# Loads a processed .npz (from prepare_sequences.py) and extracts deterministic
# features from every sequence window:
#   statistical branch: AR(p) coefficients + residual stats + GARCH(1,1) features
#   transformed branch: FFT top-k magnitudes + Haar wavelet detail energies
# Both train and test feature arrays are concatenated and saved together in a
# single .npz (so the store file has shape [N_train+N_test, feature_dim]).
# A companion .index.npz records the train/test split sizes so downstream code
# can recover the boundary after loading.
#
# Neural embeddings (VAE, contrastive, future BYOL) are NOT computed here — they
# require pretrained models and are added later as separate named feature
# branches.
#
# Usage:
#   python scripts/prepare_features.py \
#     --processed-npz data/processed/market_4h_seq64_top50.npz \
#     --out-path data/features/features_4h_seq64_top50.npz

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from data_processing.reader import load_processed_npz
from features.feature_store import FeatureBundle, NpzFeatureStore, build_feature_bundle


def _build_features_for_split(
    sequences: np.ndarray, ar_order: int, fft_top_k: int, wavelet_levels: int
) -> FeatureBundle:
    # No neural branches are attached here because encoders have not been
    # trained yet; frozen encoder embeddings are added later by branch name.
    return build_feature_bundle(
        sequences=sequences,
        ar_order=ar_order,
        fft_top_k=fft_top_k,
        wavelet_levels=wavelet_levels,
    )


def build_and_save_features(
    processed_npz: str, out_path: str, ar_order: int, fft_top_k: int, wavelet_levels: int
) -> str:
    train, test = load_processed_npz(processed_npz)
    train_bundle = _build_features_for_split(train, ar_order, fft_top_k, wavelet_levels)
    test_bundle = _build_features_for_split(test, ar_order, fft_top_k, wavelet_levels)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    store = NpzFeatureStore(out_path)
    # Concatenate train+test so the store holds all rows in one array.
    # The index file below is the only thing that records where the boundary is.
    store.save(
        FeatureBundle(
            statistical=np.concatenate(
                [train_bundle.statistical, test_bundle.statistical], axis=0
            ),
            transformed=np.concatenate(
                [train_bundle.transformed, test_bundle.transformed], axis=0
            ),
            neural=None,
        )
    )

    # Companion index: tells downstream code how many rows belong to train vs test.
    # Load with: data = np.load(index_path); train_size = int(data["train_size"])
    index_path = f"{out_path}.index.npz"
    np.savez_compressed(
        index_path,
        train_size=len(train_bundle.statistical),
        test_size=len(test_bundle.statistical),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build statistical/transformation features from processed sequence tensors."
    )
    parser.add_argument("--processed-npz", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--ar-order", type=int, default=5)
    parser.add_argument("--fft-top-k", type=int, default=8)
    parser.add_argument("--wavelet-levels", type=int, default=3)
    args = parser.parse_args()

    out = build_and_save_features(
        processed_npz=args.processed_npz,
        out_path=args.out_path,
        ar_order=args.ar_order,
        fft_top_k=args.fft_top_k,
        wavelet_levels=args.wavelet_levels,
    )
    print(f"Saved feature store: {out}")
    print(f"Saved split index: {out}.index.npz")


if __name__ == "__main__":
    main()
