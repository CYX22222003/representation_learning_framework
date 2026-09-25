#!/usr/bin/env python3
"""Build and replay-validate the two frozen Phase 6 H=8 label bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_v4"))

from audit_phase6_volatility import _phase5_identities, canonical_walk_inputs  # noqa: E402
from data_processing.phase5_walks import sha256_file  # noqa: E402
from data_processing.phase6_volatility import validate_horizon_freeze_manifest  # noqa: E402
from data_processing.phase6_volatility_labels import (  # noqa: E402
    build_volatility_label_bundle,
    validate_volatility_label_bundle_files,
    write_volatility_label_bundle,
)


PHASE6_VOLATILITY_ROOT = ROOT / "experiments" / "phase6" / "volatility_prediction"
DEFAULT_OUTPUT_ROOT = PHASE6_VOLATILITY_ROOT / "data_preparation"
DEFAULT_AUDIT_CAPACITY = PHASE6_VOLATILITY_ROOT / "data_exploration" / "horizon_capacity.csv"
DEFAULT_FREEZE_PATH = PHASE6_VOLATILITY_ROOT / "manifests" / "horizon_freeze_h8.json"


def _require_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = PHASE6_VOLATILITY_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Phase 6 volatility outputs must be under {allowed}: {resolved}")
    return resolved


def _stage_a_counts(capacity_path: Path, walk: int) -> dict[str, int]:
    if not capacity_path.is_file():
        raise FileNotFoundError(capacity_path)
    frame = pd.read_csv(capacity_path)
    selected = frame.loc[
        frame["walk"].eq(walk)
        & frame["horizon_hours"].eq(8)
        & frame["scope"].eq("aggregate")
    ]
    if set(selected["split"].astype(str)) != {"train", "evaluation"} or len(selected) != 2:
        raise ValueError(f"Stage A capacity has no unique H=8 aggregate rows for walk {walk}")
    by_split = selected.set_index("split")["eligible_rows"]
    return {"train": int(by_split["train"]), "test": int(by_split["evaluation"])}


def prepare_walk(
    walk: int,
    output_root: Path,
    *,
    capacity_path: Path,
    overwrite: bool,
) -> dict[str, object]:
    item = canonical_walk_inputs()[walk]
    candles_path = item.source_dir / "candles_1h_clean_ffill1.parquet"
    metadata_path = item.source_dir / "market_metadata.parquet"
    fill_manifest_path = item.source_dir / "fill_manifest.json"
    phase5_manifest_path = Path(f"{item.phase5_bundle}.manifest.json")
    for path in (
        candles_path,
        metadata_path,
        fill_manifest_path,
        item.phase5_bundle,
        phase5_manifest_path,
        capacity_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    supported, _ = _phase5_identities(item.phase5_bundle, walk)
    provenance = {
        "candles_1h_clean_ffill1": {
            "path": str(candles_path.resolve()),
            "sha256": sha256_file(candles_path),
        },
        "market_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256_file(metadata_path),
        },
        "fill_manifest": {
            "path": str(fill_manifest_path.resolve()),
            "sha256": sha256_file(fill_manifest_path),
        },
        "phase5_bundle": {
            "path": str(item.phase5_bundle.resolve()),
            "sha256": sha256_file(item.phase5_bundle),
        },
        "phase5_bundle_manifest": {
            "path": str(phase5_manifest_path.resolve()),
            "sha256": sha256_file(phase5_manifest_path),
        },
        "phase6_stage_a_capacity": {
            "path": str(capacity_path.resolve()),
            "sha256": sha256_file(capacity_path),
        },
    }
    bundle = build_volatility_label_bundle(
        pd.read_parquet(candles_path),
        pd.read_parquet(metadata_path),
        item.spec,
        supported_contracts=supported,
        source_provenance=provenance,
        expected_stage_a_counts=_stage_a_counts(capacity_path, walk),
    )
    npz_path = output_root / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    write_volatility_label_bundle(npz_path, bundle, overwrite=overwrite)
    return validate_volatility_label_bundle_files(npz_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--capacity-path", type=Path, default=DEFAULT_AUDIT_CAPACITY)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        freeze = validate_horizon_freeze_manifest(args.freeze_path, ROOT)
        if not freeze["label_bundle_authorized"] or freeze["primary_horizon_hours"] != 8:
            raise ValueError("the canonical horizon freeze does not authorize H=8 labels")
        output_root = _require_output_root(args.output_root)
        selected = (1, 2) if args.walk == "all" else (int(args.walk),)
        results = [
            prepare_walk(
                walk,
                output_root,
                capacity_path=args.capacity_path,
                overwrite=args.overwrite,
            )
            for walk in selected
        ]
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility label preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
