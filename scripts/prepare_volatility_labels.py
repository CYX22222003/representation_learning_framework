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

from data_processing.data_processing import create_sequences, preprocess_market, split_sequences
from data_processing.file_list import list_top_k
from tasks.volatility_labels import (
    TARGET_DEFINITION,
    TARGET_FORMULA,
    ContractAlignment,
    OVERLAP_NOTE,
    VolatilityLabelMetadata,
    build_aligned_volatility_labels,
    save_volatility_label_bundle,
    sha256_file,
    stable_dataset_id,
    validate_volatility_label_bundle,
)


def _load_processed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "train" not in data or "test" not in data:
            raise ValueError("processed npz must contain train and test arrays")
        return np.asarray(data["train"], dtype=np.float32), np.asarray(data["test"], dtype=np.float32)


def build_volatility_label_bundle(
    *,
    processed_npz: Path,
    timeframe: str,
    seq_len: int,
    top_k: int,
    data_dir: Path,
    train_ratio: float,
    price_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train, test = _load_processed(processed_npz)
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("processed train/test arrays must be [N, seq_len, features]")
    if train.shape[1] != seq_len or test.shape[1] != seq_len:
        raise ValueError(f"processed sequence length does not match --seq-len={seq_len}")

    file_list = list_top_k(timeframe=timeframe, top_k=top_k, data_dir=str(data_dir))
    contract_sequences: list[np.ndarray] = []
    contracts: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    train_offset = 0
    test_offset = 0
    included_id = 0

    for filename, file_size in file_list:
        path = data_dir / filename
        try:
            file_sha = sha256_file(path)
            df = pd.read_feather(path)
            features = preprocess_market(df)
            sequences = create_sequences(features, seq_len)
        except Exception as exc:
            contracts.append({"filename": filename, "status": "skipped", "reason": str(exc)})
            continue
        if len(df) == 0 or len(sequences) == 0:
            contracts.append(
                {"filename": filename, "status": "skipped", "reason": "empty or not enough rows", "raw_rows": int(len(df))}
            )
            continue

        train_seq, test_seq = split_sequences(sequences, train_ratio)
        n_train = int(train_seq.shape[0])
        n_test = int(test_seq.shape[0])
        contract_sequences.append(sequences)
        train_labels = max(0, n_train - 1)
        test_labels = max(0, n_test - 1)
        alignment = ContractAlignment(
            contract_id=included_id,
            filename=filename,
            raw_rows=int(len(df)),
            sequence_count=int(sequences.shape[0]),
            train_sequence_count=n_train,
            test_sequence_count=n_test,
            train_label_count=train_labels,
            test_label_count=test_labels,
            train_global_offset=train_offset,
            test_global_offset=test_offset,
            file_size=int(file_size),
            file_sha256=file_sha,
        )
        contracts.append(alignment.to_dict())
        source_files.append({"filename": filename, "size": int(file_size), "sha256": file_sha})
        train_offset += n_train
        test_offset += n_test
        included_id += 1

    if train_offset != train.shape[0] or test_offset != test.shape[0]:
        raise ValueError(
            "reconstructed sequence counts do not match processed npz: "
            f"reconstructed={train_offset}/{test_offset}, processed={train.shape[0]}/{test.shape[0]}"
        )

    bundle = build_aligned_volatility_labels(
        contract_sequences,
        price_index=price_index,
        horizon=1,
        train_ratio=train_ratio,
        contract_ids=range(len(contract_sequences)),
    )
    validation = validate_volatility_label_bundle(bundle, train_size=train.shape[0], test_size=test.shape[0])
    processed_sha = sha256_file(processed_npz)
    dataset_payload = {
        "task": "volatility_prediction",
        "label_mode": TARGET_DEFINITION,
        "timeframe": timeframe,
        "seq_len": seq_len,
        "top_k": top_k,
        "price_index": price_index,
        "horizon": 1,
        "train_ratio": train_ratio,
        "source_files": source_files,
        "processed_npz_sha256": processed_sha,
        "target_formula": TARGET_FORMULA,
    }
    dataset_id = stable_dataset_id(dataset_payload)
    metadata = VolatilityLabelMetadata(
        timeframe=timeframe,
        seq_len=seq_len,
        top_k=top_k,
        price_index=price_index,
        horizon=1,
        train_ratio=train_ratio,
        dataset_id=dataset_id,
        processed_npz_sha256=processed_sha,
    )
    manifest = {
        **metadata.to_dict(),
        "target_definition": TARGET_DEFINITION,
        "processed_npz": str(processed_npz),
        "processed_shapes": {"train": list(train.shape), "test": list(test.shape)},
        "data_dir": str(data_dir),
        "label_counts": {"train": validation["train_label_count"], "test": validation["test_label_count"]},
        "contract_counts": {"train": validation["train_contract_count"], "test": validation["test_contract_count"]},
        "contracts": contracts,
        "dataset_id_inputs": dataset_payload,
        "inherited_limitations": [
            "processed volume channel was z-scored per full contract before the split in the current pipeline",
            OVERLAP_NOTE,
        ],
    }
    return bundle, manifest


def verify_bundle(path: Path, processed_npz: Path | None = None) -> dict[str, object]:
    with np.load(path) as data:
        bundle = {key: data[key].copy() for key in data.files}
    manifest_path = Path(f"{path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_size = test_size = None
    if processed_npz is not None:
        train, test = _load_processed(processed_npz)
        train_size, test_size = int(train.shape[0]), int(test.shape[0])
    return validate_volatility_label_bundle(bundle, train_size=train_size, test_size=test_size, metadata=manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare shared contract-aware volatility labels.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--out-path", default="data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz")
    parser.add_argument("--timeframe", choices=("1h", "4h", "1d"), default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--price-index", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_path = Path(args.out_path)
    processed_npz = Path(args.processed_npz)
    if args.verify:
        result = verify_bundle(out_path, processed_npz)
        print(json.dumps(result, indent=2))
        return 0
    if (out_path.exists() or Path(f"{out_path}.manifest.json").exists()) and not args.overwrite:
        print(f"output exists; pass --overwrite: {out_path}", file=sys.stderr)
        return 2
    try:
        bundle, manifest = build_volatility_label_bundle(
            processed_npz=processed_npz,
            timeframe=args.timeframe,
            seq_len=args.seq_len,
            top_k=args.top_k,
            data_dir=Path(args.data_dir),
            train_ratio=args.train_ratio,
            price_index=args.price_index,
        )
        save_volatility_label_bundle(bundle, manifest, out_path)
        result = verify_bundle(out_path, processed_npz)
        print(json.dumps({"wrote": str(out_path), **result}, indent=2))
        return 0
    except Exception as exc:
        print(f"prepare_volatility_labels failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
