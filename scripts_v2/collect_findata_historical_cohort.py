"""Collect a diverse, high-volume historical Polymarket cohort from FinData."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_acquisition.findata import (  # noqa: E402
    FinDataClient,
    aggregate_one_hour_to_four_hour,
    format_rfc3339,
    load_lumid_token,
)
from data_acquisition.polymarket_pipeline import (  # noqa: E402
    CANDLE_QUARANTINE_RULE_VERSION,
    diverse_candidate_order,
    fetch_complete_candles,
    historical_catalog,
    overlapping_candidates,
    probe_candidates,
    quarantine_condition_candles,
)


DEFAULT_START = "2025-12-01T00:00:00Z"
DEFAULT_END = "2026-09-01T00:00:00Z"
DEFAULT_OUTPUT = ROOT / "data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31"


def parse_timestamp(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data_new").resolve()
    if allowed != resolved and allowed not in resolved.parents:
        raise ValueError(f"output must be inside {allowed}: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END, help="exclusive UTC endpoint")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--max-per-family", type=int, default=2)
    parser.add_argument("--minimum-market-hours", type=float, default=336.0)
    parser.add_argument("--minimum-probe-bars", type=int, default=64)
    parser.add_argument("--minimum-probe-span-hours", type=float, default=256.0)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--pages-per-status", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-quarantine-fraction", type=float, default=0.01)
    parser.add_argument(
        "--defer-quarantine",
        action="store_true",
        help=(
            "write immutable raw candles first and defer anomaly preview/pruning to "
            "quarantine_findata_condition_candles.py"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0 or args.candidate_limit < args.top_k:
        parser.error("top-k must be positive and candidate-limit must be at least top-k")
    if args.max_per_family <= 0:
        parser.error("max-per-family must be positive")
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")

    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if not start < end:
        parser.error("start must be earlier than end")
    prepare_output(args.output_dir, args.overwrite)

    client = FinDataClient(load_lumid_token(ROOT / ".env"))
    print("Fetching paginated open and closed Polymarket catalogs...", flush=True)
    catalog = historical_catalog(
        client,
        page_size=args.page_size,
        pages_per_status=args.pages_per_status,
        workers=min(args.workers, 8),
    )
    overlapping = overlapping_candidates(
        catalog,
        start=start,
        end=end,
        minimum_overlap_hours=args.minimum_market_hours,
    )
    candidate_order = diverse_candidate_order(
        overlapping,
        candidate_limit=args.candidate_limit,
        max_per_family=args.max_per_family,
    )
    if len(candidate_order) < args.top_k:
        raise RuntimeError(
            f"only {len(candidate_order)} diverse candidates remain for top_k={args.top_k}"
        )
    print(
        f"Catalog rows: {len(catalog):,}; overlapping eligible-duration rows: "
        f"{len(overlapping):,}; probing {len(candidate_order):,} diverse candidates...",
        flush=True,
    )
    selected, audit = probe_candidates(
        client,
        candidate_order,
        start=start,
        end=end,
        minimum_bars=args.minimum_probe_bars,
        minimum_span_hours=args.minimum_probe_span_hours,
        top_k=args.top_k,
        workers=args.workers,
    )
    condition_ids = selected["condition_id"].tolist()
    print("Selected categories:")
    print(selected["category"].value_counts().sort_index().to_string(), flush=True)

    details: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(client.polymarket_detail, condition_id): condition_id
            for condition_id in condition_ids
        }
        for future in as_completed(futures):
            condition_id = futures[future]
            body = future.result().body
            if not isinstance(body, dict):
                raise ValueError(f"unexpected market-detail response for {condition_id}")
            details.append(body)

    ranges: dict[str, tuple[datetime, datetime]] = {}
    for row in selected.itertuples(index=False):
        market_start = max(start, row.start_date.to_pydatetime())
        market_end = min(end, (row.end_date + pd.Timedelta(days=1)).to_pydatetime())
        if market_start < market_end:
            ranges[row.condition_id] = (market_start, market_end)

    print("Fetching native 15-minute history over each market's overlap...", flush=True)
    candles_15m = fetch_complete_candles(
        client,
        ranges,
        interval_minutes=15,
        global_start=start,
        global_end=end,
        workers=args.workers,
    )
    print("Fetching native one-hour history over each market's overlap...", flush=True)
    candles_1h = fetch_complete_candles(
        client,
        ranges,
        interval_minutes=60,
        global_start=start,
        global_end=end,
        workers=args.workers,
    )
    candles_4h = aggregate_one_hour_to_four_hour(candles_1h)
    quarantine_15m = None
    quarantine_1h = None
    quarantine_audit = None
    candles_4h_clean = None
    if not args.defer_quarantine:
        quarantine_15m = quarantine_condition_candles(
            candles_15m,
            interval_minutes=15,
            maximum_quarantine_fraction=args.maximum_quarantine_fraction,
        )
        quarantine_1h = quarantine_condition_candles(
            candles_1h,
            interval_minutes=60,
            maximum_quarantine_fraction=args.maximum_quarantine_fraction,
        )
        quarantine_parts = []
        for resolution, result in (("15m", quarantine_15m), ("1h", quarantine_1h)):
            audit_frame = result.audit.copy()
            audit_frame.insert(0, "resolution", resolution)
            quarantine_parts.append(audit_frame)
        quarantine_audit = pd.concat(quarantine_parts, ignore_index=True)
        candles_4h_clean = aggregate_one_hour_to_four_hour(quarantine_1h.clean)

    details_frame = pd.DataFrame(details)
    metadata = selected.merge(details_frame, on="condition_id", how="left", suffixes=("_search", "_detail"))
    metadata["question_snapshot"] = metadata["title"]
    metadata["mvp_rank"] = metadata["selection_rank"]

    catalog_path = args.output_dir / "market_search_catalog.parquet"
    metadata_path = args.output_dir / "market_metadata.parquet"
    audit_path = args.output_dir / "discovery_audit.parquet"
    path_15m = args.output_dir / "candles_15min.parquet"
    path_1h = args.output_dir / "candles_1h.parquet"
    path_4h = args.output_dir / "candles_4h_derived.parquet"
    path_15m_clean = args.output_dir / "candles_15min_clean.parquet"
    path_1h_clean = args.output_dir / "candles_1h_clean.parquet"
    path_4h_clean = args.output_dir / "candles_4h_derived_clean.parquet"
    quarantine_path = args.output_dir / "candle_quarantine.parquet"
    catalog.to_parquet(catalog_path, index=False)
    metadata.to_parquet(metadata_path, index=False)
    audit.to_parquet(audit_path, index=False)
    candles_15m.to_parquet(path_15m, index=False)
    candles_1h.to_parquet(path_1h, index=False)
    candles_4h.to_parquet(path_4h, index=False)
    if not args.defer_quarantine:
        assert quarantine_15m is not None
        assert quarantine_1h is not None
        assert candles_4h_clean is not None
        assert quarantine_audit is not None
        quarantine_15m.clean.to_parquet(path_15m_clean, index=False)
        quarantine_1h.clean.to_parquet(path_1h_clean, index=False)
        candles_4h_clean.to_parquet(path_4h_clean, index=False)
        quarantine_audit.to_parquet(quarantine_path, index=False)

    artifacts = [
        catalog_path,
        metadata_path,
        audit_path,
        path_15m,
        path_1h,
        path_4h,
    ]
    if not args.defer_quarantine:
        artifacts.extend([path_15m_clean, path_1h_clean, path_4h_clean, quarantine_path])

    if args.defer_quarantine:
        quarantine_manifest = {
            "status": "deferred_pending_raw_anomaly_preview",
            "planned_rule_version": CANDLE_QUARANTINE_RULE_VERSION,
            "maximum_quarantine_fraction": args.maximum_quarantine_fraction,
            "raw_files_preserved": True,
            "prices_repaired": False,
            "next_command": (
                ".venv/bin/python3 scripts_v2/quarantine_findata_condition_candles.py "
                f"--input-dir {args.output_dir} --audit-only"
            ),
        }
    else:
        assert quarantine_15m is not None
        assert quarantine_1h is not None
        quarantine_manifest = {
            "status": "applied_during_collection",
            "rule_version": CANDLE_QUARANTINE_RULE_VERSION,
            "maximum_quarantine_fraction": args.maximum_quarantine_fraction,
            "15m": quarantine_15m.summary,
            "1h": quarantine_1h.summary,
            "raw_files_preserved": True,
            "prices_repaired": False,
            "quarantined_timestamps_remain_gaps": True,
            "downstream_windows_must_require_exact_consecutive_timestamps": True,
            "warning": (
                "Retention above 99% is an operational screen, not proof that condition-level "
                "candles have canonical YES-token identity."
            ),
        }
    manifest = {
        "schema_version": 2,
        "source": "FinData prediction-markets API",
        "base_url": client.base_url,
        "venue": "polymarket",
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_interval": {
            "start_inclusive": format_rfc3339(start),
            "end_exclusive": format_rfc3339(end),
        },
        "selection": {
            "status": "exploratory_historical_volume_category_diverse",
            "catalog_statuses": ["open", "closed"],
            "page_size": args.page_size,
            "pages_per_status": args.pages_per_status,
            "catalog_rows": len(catalog),
            "overlapping_minimum_duration_rows": len(overlapping),
            "ranking": "round-robin categories; descending total market volume within category",
            "maximum_per_heuristic_event_family": args.max_per_family,
            "candidate_limit": args.candidate_limit,
            "top_k": args.top_k,
            "minimum_overlap_hours": args.minimum_market_hours,
            "minimum_probe_1h_bars": args.minimum_probe_bars,
            "minimum_probe_span_hours": args.minimum_probe_span_hours,
            "condition_ids": condition_ids,
            "category_counts": {
                key: int(value) for key, value in selected["category"].value_counts().items()
            },
            "warning": (
                "Full-lifetime volume, complete market dates, and heuristic categories are used for "
                "exploratory retrospective analysis only. This is not a cutoff-local Phase 4 universe."
            ),
        },
        "intervals": {
            "15m": {
                "native_api_interval_minutes": 15,
                "rows": len(candles_15m),
                "clean_rows": None if args.defer_quarantine else len(quarantine_15m.clean),
            },
            "1h": {
                "native_api_interval_minutes": 60,
                "rows": len(candles_1h),
                "clean_rows": None if args.defer_quarantine else len(quarantine_1h.clean),
            },
            "4h": {
                "rows": len(candles_4h),
                "derived_from": "native 1h observed rows",
                "utc_aligned": True,
                "missing_hours_imputed": False,
                "coverage_columns": ["observed_1h_bars", "complete_1h_coverage"],
            },
            **(
                {}
                if args.defer_quarantine
                else {
                    "4h_clean": {
                        "rows": len(candles_4h_clean),
                        "derived_from": path_1h_clean.name,
                        "utc_aligned": True,
                        "missing_hours_imputed": False,
                        "coverage_columns": ["observed_1h_bars", "complete_1h_coverage"],
                    }
                }
            ),
        },
        "condition_candle_quarantine": quarantine_manifest,
        "credential_persistence": "No token or Authorization header is stored.",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote expanded historical cohort to {args.output_dir}")
    print(f"  markets: {len(condition_ids)}")
    print(f"  native 15m rows: {len(candles_15m):,}")
    print(f"  native 1h rows: {len(candles_1h):,}")
    print(f"  derived 4h rows: {len(candles_4h):,}")
    if args.defer_quarantine:
        print("  quarantine: deferred until the raw anomaly preview is reviewed")
    else:
        assert quarantine_15m is not None
        assert quarantine_1h is not None
        print(
            f"  quarantined 15m rows: {len(quarantine_15m.audit):,}; "
            f"retained {quarantine_15m.summary['retained_fraction']:.4%}"
        )
        print(
            f"  quarantined 1h rows: {len(quarantine_1h.audit):,}; "
            f"retained {quarantine_1h.summary['retained_fraction']:.4%}"
        )


if __name__ == "__main__":
    main()
