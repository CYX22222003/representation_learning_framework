"""Build and validate the two canonical Phase 5 walk data bundles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import (  # noqa: E402
    Phase5WalkSpec,
    build_phase5_walk_bundle,
    sha256_file,
    validate_phase5_bundle_files,
)


PHASE5_ROOT = ROOT / "experiments" / "phase5"
DEFAULT_OUTPUT_ROOT = PHASE5_ROOT / "data_preparation"


@dataclass(frozen=True)
class WalkInput:
    spec: Phase5WalkSpec
    input_dir: Path


def canonical_walk_inputs() -> dict[int, WalkInput]:
    source_root = ROOT / "data_new" / "findata" / "polymarket"
    return {
        1: WalkInput(
            Phase5WalkSpec.from_values(1, "2025-12-02", "2026-04-01", "2026-06-16"),
            source_root
            / "phase5_walk1_top50_train-2025-12-02_cutoff-2026-04-01_eval-end-2026-06-16",
        ),
        2: WalkInput(
            Phase5WalkSpec.from_values(2, "2026-02-16", "2026-06-16", "2026-09-01"),
            source_root
            / "phase5_walk2_top50_train-2026-02-16_cutoff-2026-06-16_eval-end-2026-09-01",
        ),
    }


def _require_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = PHASE5_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Phase 5 outputs must be under {allowed}: {resolved}")
    return resolved


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_walk(walk_input: WalkInput, output_root: Path, *, overwrite: bool = False) -> dict[str, object]:
    output_root = _require_output_root(output_root)
    input_dir = walk_input.input_dir.resolve()
    candles_path = input_dir / "candles_1h_clean_ffill1.parquet"
    metadata_path = input_dir / "market_metadata.parquet"
    fill_manifest_path = input_dir / "fill_manifest.json"
    for path in (candles_path, metadata_path, fill_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    walk_dir = output_root / f"walk{walk_input.spec.walk}"
    npz_path = walk_dir / "market_1h_seq64_h2.npz"
    manifest_path = Path(f"{npz_path}.manifest.json")
    occupied = [path for path in (npz_path, manifest_path) if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(f"refusing to overwrite Phase 5 data artifacts: {occupied}")
    walk_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "candles_1h_clean_ffill1": {
            "path": str(candles_path),
            "sha256": sha256_file(candles_path),
        },
        "market_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "fill_manifest": {
            "path": str(fill_manifest_path),
            "sha256": sha256_file(fill_manifest_path),
        },
    }
    bundle = build_phase5_walk_bundle(
        pd.read_parquet(candles_path),
        pd.read_parquet(metadata_path),
        walk_input.spec,
        source_provenance=provenance,
    )
    temporary_npz = npz_path.with_suffix(".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **bundle.arrays)
    os.replace(temporary_npz, npz_path)
    manifest = {
        **bundle.manifest,
        "artifact": {
            "path": str(npz_path),
            "sha256": sha256_file(npz_path),
            "bytes": npz_path.stat().st_size,
        },
    }
    _write_json(manifest_path, manifest)
    return validate_phase5_bundle_files(npz_path, manifest_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--walk1-input-dir", type=Path)
    parser.add_argument("--walk2-input-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    inputs = canonical_walk_inputs()
    if args.walk1_input_dir:
        inputs[1] = WalkInput(inputs[1].spec, args.walk1_input_dir)
    if args.walk2_input_dir:
        inputs[2] = WalkInput(inputs[2].spec, args.walk2_input_dir)
    selected = (1, 2) if args.walk == "all" else (int(args.walk),)
    try:
        for walk in selected:
            result = prepare_walk(inputs[walk], args.output_root, overwrite=args.overwrite)
            print(f"Phase 5 walk {walk} data prepared and validated: {result}", flush=True)
        return 0
    except Exception as exc:
        print(f"Phase 5 data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
