import argparse
import os

from prepare_features import build_and_save_features
from prepare_sequences import _parse_timeframes, save_sequences_for_timeframe


def run_pipeline(
    timeframes: list[str],
    seq_len: int,
    top_k: int,
    processed_dir: str,
    feature_dir: str,
    ar_order: int,
    fft_top_k: int,
    wavelet_levels: int,
) -> None:
    for timeframe in timeframes:
        processed_path = save_sequences_for_timeframe(
            timeframe=timeframe, seq_len=seq_len, top_k=top_k, out_dir=processed_dir
        )
        print(f"[{timeframe}] sequences -> {processed_path}")

        feature_path = os.path.join(
            feature_dir, f"features_{timeframe}_seq{seq_len}_top{top_k}.npz"
        )
        build_and_save_features(
            processed_npz=processed_path,
            out_path=feature_path,
            ar_order=ar_order,
            fft_top_k=fft_top_k,
            wavelet_levels=wavelet_levels,
        )
        print(f"[{timeframe}] features  -> {feature_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end data prep pipeline: raw feather -> sequences -> feature store"
    )
    parser.add_argument("--timeframes", type=str, default="1h,4h,1d")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--feature-dir", type=str, default="data/features")
    parser.add_argument("--ar-order", type=int, default=5)
    parser.add_argument("--fft-top-k", type=int, default=8)
    parser.add_argument("--wavelet-levels", type=int, default=3)
    args = parser.parse_args()

    timeframes = _parse_timeframes(args.timeframes)
    run_pipeline(
        timeframes=timeframes,
        seq_len=args.seq_len,
        top_k=args.top_k,
        processed_dir=args.processed_dir,
        feature_dir=args.feature_dir,
        ar_order=args.ar_order,
        fft_top_k=args.fft_top_k,
        wavelet_levels=args.wavelet_levels,
    )


if __name__ == "__main__":
    main()
