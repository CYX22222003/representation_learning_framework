from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from data_processing.file_list import list_top_k

from baselines.ginn_baseline.ginn_data import (
    ContractPreparationError,
    GinnDataConfig,
    load_and_validate_cache,
    prepare_contract,
    source_digest,
    write_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the framework-aligned GINN cache.")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--ar-order", type=int, default=5)
    parser.add_argument("--garch-p", type=int, default=1)
    parser.add_argument("--garch-q", type=int, default=1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-path", default="data/processed/ginn_4h_seq64_top50.npz")
    parser.add_argument(
        "--manifest-path",
        default="data/processed/ginn_4h_seq64_top50.manifest.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_ginn_cache(
    config: GinnDataConfig,
    data_dir: str | Path,
    cache_path: str | Path,
    manifest_path: str | Path,
    overwrite: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    cache_path = Path(cache_path)
    manifest_path = Path(manifest_path)
    if not overwrite and (cache_path.exists() or manifest_path.exists()):
        raise FileExistsError("cache or manifest already exists; pass --overwrite to replace")

    entries = list_top_k(config.timeframe, config.top_k, data_dir=str(data_dir))
    prepared = []
    skipped = []
    source_files = []
    for contract_id, (filename, size) in enumerate(entries):
        path = data_dir / filename
        source_files.append(
            {
                "filename": filename,
                "size": int(size),
                "sha256": source_digest(path) if path.exists() else "unavailable",
            }
        )
        try:
            frame = pd.read_feather(path)
            prepared.append(prepare_contract(frame, contract_id, filename, config))
        except ContractPreparationError as exc:
            skipped.append({"filename": filename, "reason": exc.reason, "detail": exc.detail})

    manifest = write_cache(prepared, skipped, source_files, config, cache_path, manifest_path)
    load_and_validate_cache(cache_path, manifest_path, config)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = GinnDataConfig(
        timeframe=args.timeframe,
        seq_len=args.seq_len,
        top_k=args.top_k,
        train_ratio=args.train_ratio,
        ar_order=args.ar_order,
        garch_p=args.garch_p,
        garch_q=args.garch_q,
    )
    try:
        manifest = prepare_ginn_cache(
            config,
            args.data_dir,
            args.out_path,
            args.manifest_path,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GINN cache preparation failed: {exc}", file=sys.stderr)
        return 1

    totals = manifest["totals"]
    print(f"dataset_id={manifest['dataset_id']}")
    print(
        "included={included_contracts} skipped={skipped_contracts} "
        "fallbacks={garch_fallbacks}".format(**totals)
    )
    print(f"train_shape={manifest['arrays']['X_train']['shape']}")
    print(f"test_shape={manifest['arrays']['X_test']['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
