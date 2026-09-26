#!/usr/bin/env python3
"""Run the Phase 6 Stage A future-realised-variance horizon audit."""

from __future__ import annotations

import argparse
import json
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
    sha256_file,
    validate_phase5_bundle_files,
)
from data_processing.phase6_volatility import (  # noqa: E402
    DEFAULT_CANDIDATE_HORIZONS,
    audit_volatility_walk,
    combine_walk_audits,
    parse_candidate_horizons,
    validate_audit_artifacts,
    write_audit_tables,
)


PHASE6_ROOT = ROOT / "experiments" / "phase6"
DEFAULT_OUTPUT_DIR = PHASE6_ROOT / "volatility_prediction" / "data_exploration"
PHASE5_DATA_ROOT = ROOT / "experiments" / "phase5" / "data_preparation"


@dataclass(frozen=True)
class WalkInput:
    spec: Phase5WalkSpec
    source_dir: Path
    phase5_bundle: Path


def canonical_walk_inputs() -> dict[int, WalkInput]:
    source_root = ROOT / "data_new" / "findata" / "polymarket"
    return {
        1: WalkInput(
            spec=Phase5WalkSpec.from_values(1, "2025-12-02", "2026-04-01", "2026-06-16"),
            source_dir=source_root
            / "phase5_walk1_top50_train-2025-12-02_cutoff-2026-04-01_eval-end-2026-06-16",
            phase5_bundle=PHASE5_DATA_ROOT / "walk1" / "market_1h_seq64_h2.npz",
        ),
        2: WalkInput(
            spec=Phase5WalkSpec.from_values(2, "2026-02-16", "2026-06-16", "2026-09-01"),
            source_dir=source_root
            / "phase5_walk2_top50_train-2026-02-16_cutoff-2026-06-16_eval-end-2026-09-01",
            phase5_bundle=PHASE5_DATA_ROOT / "walk2" / "market_1h_seq64_h2.npz",
        ),
    }


def _require_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    allowed = PHASE6_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Phase 6 outputs must be under {allowed}: {resolved}")
    return resolved


def _phase5_identities(bundle_path: Path, walk: int) -> tuple[set[str], set[tuple[str, int, int, int]]]:
    validation = validate_phase5_bundle_files(bundle_path)
    if int(validation["walk"]) != walk:
        raise ValueError(f"Phase 5 bundle walk mismatch: expected {walk}, got {validation['walk']}")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        supported = {str(value) for value in bundle["train_condition_ids"]}
        identities = set(
            zip(
                bundle["encoder_train_condition_ids"].astype(str).tolist(),
                bundle["encoder_train_window_start_ns"].astype(np.int64).tolist(),
                bundle["encoder_train_decision_date_ns"].astype(np.int64).tolist(),
                bundle["encoder_train_decision_availability_ns"].astype(np.int64).tolist(),
                strict=True,
            )
        )
    return supported, identities


def run_audit(
    inputs: dict[int, WalkInput],
    output_dir: Path,
    *,
    candidate_horizons: tuple[int, ...],
    overwrite: bool,
) -> dict[str, object]:
    output_dir = _require_output_dir(output_dir)
    audits = []
    provenance: dict[str, object] = {}
    for walk in sorted(inputs):
        item = inputs[walk]
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
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        supported, identities = _phase5_identities(item.phase5_bundle, walk)
        audit = audit_volatility_walk(
            pd.read_parquet(candles_path),
            pd.read_parquet(metadata_path),
            item.spec,
            supported_contracts=supported,
            expected_encoder_identities=identities,
            candidate_horizons=candidate_horizons,
        )
        audits.append(audit)
        provenance[f"walk{walk}"] = {
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
                "manifest_path": str(phase5_manifest_path.resolve()),
                "manifest_sha256": sha256_file(phase5_manifest_path),
            },
        }
    frames = combine_walk_audits(audits)
    write_audit_tables(
        output_dir,
        frames,
        [audit.manifest for audit in audits],
        provenance,
        overwrite=overwrite,
    )
    return validate_audit_artifacts(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-horizons",
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_HORIZONS),
        help="Comma-separated predeclared hourly horizons; every value must exceed one.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--walk1-source-dir", type=Path)
    parser.add_argument("--walk2-source-dir", type=Path)
    parser.add_argument("--walk1-phase5-bundle", type=Path)
    parser.add_argument("--walk2-phase5-bundle", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        horizons = parse_candidate_horizons(args.candidate_horizons)
        inputs = canonical_walk_inputs()
        if args.walk1_source_dir:
            inputs[1] = WalkInput(inputs[1].spec, args.walk1_source_dir, inputs[1].phase5_bundle)
        if args.walk2_source_dir:
            inputs[2] = WalkInput(inputs[2].spec, args.walk2_source_dir, inputs[2].phase5_bundle)
        if args.walk1_phase5_bundle:
            inputs[1] = WalkInput(inputs[1].spec, inputs[1].source_dir, args.walk1_phase5_bundle)
        if args.walk2_phase5_bundle:
            inputs[2] = WalkInput(inputs[2].spec, inputs[2].source_dir, args.walk2_phase5_bundle)
        result = run_audit(
            inputs,
            args.output_dir,
            candidate_horizons=horizons,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
