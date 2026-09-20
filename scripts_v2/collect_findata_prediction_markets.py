"""Collect a small, recent Polymarket OHLCV cohort from the lab FinData API.

The collector preserves native sparse 15-minute and one-hour responses and
derives transparent UTC-aligned four-hour bars from the observed one-hour
rows.  It does not create experiment splits, model windows, or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    build_market_candidates,
    candles_to_frame,
    format_rfc3339,
    iter_time_chunks,
    load_lumid_token,
    validate_candles,
)
from data_acquisition.polymarket_pipeline import (  # noqa: E402
    CANDLE_QUARANTINE_RULE_VERSION,
    quarantine_condition_candles,
)


DEFAULT_START = "2025-12-01T00:00:00Z"
DEFAULT_END = "2026-09-01T00:00:00Z"
DEFAULT_OUTPUT = ROOT / "data_new/findata/polymarket/mvp_2025-12-01_2026-08-31"


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


def fetch_interval(
    client: FinDataClient,
    condition_id: str,
    interval_minutes: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    response = client.polymarket_candles(
        condition_id,
        interval_minutes=interval_minutes,
        start=start,
        end=end,
    )
    return candles_to_frame(
        response.body,
        condition_id=condition_id,
        interval_minutes=interval_minutes,
    )


def fetch_complete_candles(
    client: FinDataClient,
    condition_ids: list[str],
    interval_minutes: int,
    start: datetime,
    end: datetime,
    workers: int,
) -> pd.DataFrame:
    chunk = timedelta(days=7 if interval_minutes == 15 else 30)
    jobs = [
        (condition_id, chunk_start, chunk_end)
        for condition_id in condition_ids
        for chunk_start, chunk_end in iter_time_chunks(start, end, chunk)
    ]
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_interval,
                client,
                condition_id,
                interval_minutes,
                chunk_start,
                chunk_end,
            ): (condition_id, chunk_start, chunk_end)
            for condition_id, chunk_start, chunk_end in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            condition_id, chunk_start, chunk_end = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"failed {interval_minutes}m fetch for {condition_id} "
                    f"[{format_rfc3339(chunk_start)}, {format_rfc3339(chunk_end)})"
                ) from exc
            if not frame.empty:
                frames.append(frame)
            if completed % 25 == 0 or completed == len(jobs):
                print(f"  {interval_minutes}m chunks: {completed}/{len(jobs)}", flush=True)

    if not frames:
        return candles_to_frame([], condition_id="", interval_minutes=interval_minutes).iloc[0:0]
    result = pd.concat(frames, ignore_index=True)
    result = (
        result.sort_values(["condition_id", "date"])
        .drop_duplicates(["condition_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    for condition_id, market in result.groupby("condition_id", sort=False):
        validate_candles(
            market.drop(columns=["condition_id"]).sort_values("date").reset_index(drop=True),
            start=start,
            end=end,
        )
    return result


def select_mvp_markets(
    client: FinDataClient,
    candidates: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    top_k: int,
    candidate_limit: int,
    minimum_probe_bars: int,
    minimum_span_hours: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = candidates.loc[candidates["active"]].head(candidate_limit).copy()
    if len(pool) < top_k:
        raise ValueError(f"only {len(pool)} active candidates available for top_k={top_k}")

    print(f"Probing {len(pool)} active, liquidity-ranked candidates at 1h...", flush=True)
    probe_frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_interval, client, row.market_id, 60, start, end): row.market_id
            for row in pool.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            condition_id = futures[future]
            try:
                probe_frames[condition_id] = future.result()
            except Exception as exc:
                errors[condition_id] = type(exc).__name__
            if completed % 20 == 0 or completed == len(futures):
                print(f"  probes: {completed}/{len(futures)}", flush=True)

    audits: list[dict[str, object]] = []
    for row in pool.itertuples(index=False):
        frame = probe_frames.get(row.market_id)
        count = 0 if frame is None else len(frame)
        span_hours = 0.0
        if frame is not None and len(frame) > 1:
            span_hours = float((frame["date"].max() - frame["date"].min()).total_seconds() / 3600.0)
        audits.append(
            {
                "condition_id": row.market_id,
                "snapshot_rank": int(row.snapshot_rank),
                "question": row.question,
                "liquidity_num": float(row.liquidity_num),
                "volume_24h": float(row.volume_24h),
                "probe_1h_rows": int(count),
                "probe_span_hours": span_hours,
                "probe_error": errors.get(row.market_id),
                "eligible": count >= minimum_probe_bars and span_hours >= minimum_span_hours,
            }
        )
    audit = pd.DataFrame(audits).sort_values("snapshot_rank").reset_index(drop=True)
    selected_ids = audit.loc[audit["eligible"], "condition_id"].head(top_k).tolist()
    if len(selected_ids) < top_k:
        raise RuntimeError(
            f"only {len(selected_ids)} of {top_k} requested markets passed the 1h history probe; "
            "increase --candidate-limit or reduce the declared MVP size"
        )
    selected = pool.loc[pool["market_id"].isin(selected_ids)].copy()
    order = {condition_id: index for index, condition_id in enumerate(selected_ids, start=1)}
    selected["mvp_rank"] = selected["market_id"].map(order).astype(int)
    selected = selected.sort_values("mvp_rank").reset_index(drop=True)
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END, help="exclusive UTC endpoint")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--minimum-probe-bars", type=int, default=64)
    parser.add_argument("--minimum-span-hours", type=float, default=256.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maximum-quarantine-fraction", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0 or args.candidate_limit < args.top_k:
        parser.error("top-k must be positive and candidate-limit must be at least top-k")
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")

    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if not start < end:
        parser.error("start must be earlier than end")
    prepare_output(args.output_dir, args.overwrite)

    token = load_lumid_token(ROOT / ".env")
    client = FinDataClient(token)

    print("Fetching current monitored Polymarket universe...", flush=True)
    universe_response = client.polymarket_universe(include_inactive=True)
    candidates = build_market_candidates(universe_response.body)
    selected, discovery_audit = select_mvp_markets(
        client,
        candidates,
        start=start,
        end=end,
        top_k=args.top_k,
        candidate_limit=args.candidate_limit,
        minimum_probe_bars=args.minimum_probe_bars,
        minimum_span_hours=args.minimum_span_hours,
        workers=args.workers,
    )
    condition_ids = selected["market_id"].tolist()
    print("Selected condition IDs:", *condition_ids, sep="\n  ", flush=True)

    details: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(client.polymarket_detail, condition_id): condition_id
            for condition_id in condition_ids
        }
        for future in as_completed(future_map):
            condition_id = future_map[future]
            body = future.result().body
            if not isinstance(body, dict):
                raise ValueError(f"unexpected market-detail response for {condition_id}")
            details.append(body)

    print("Fetching complete native 15m candles...", flush=True)
    candles_15m = fetch_complete_candles(client, condition_ids, 15, start, end, args.workers)
    print("Fetching complete native 1h candles...", flush=True)
    candles_1h = fetch_complete_candles(client, condition_ids, 60, start, end, args.workers)
    print("Deriving transparent UTC-aligned 4h candles from observed 1h rows...", flush=True)
    candles_4h = aggregate_one_hour_to_four_hour(candles_1h)
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

    selected.rename(columns={"market_id": "condition_id"}, inplace=True)
    details_frame = pd.DataFrame(details)
    metadata = selected.merge(details_frame, on="condition_id", how="left", suffixes=("_snapshot", "_detail"))

    universe_path = args.output_dir / "universe_snapshot.json"
    universe_path.write_text(json.dumps(universe_response.body, indent=2), encoding="utf-8")
    metadata_path = args.output_dir / "market_metadata.parquet"
    audit_path = args.output_dir / "discovery_audit.parquet"
    path_15m = args.output_dir / "candles_15min.parquet"
    path_1h = args.output_dir / "candles_1h.parquet"
    path_4h = args.output_dir / "candles_4h_derived.parquet"
    path_15m_clean = args.output_dir / "candles_15min_clean.parquet"
    path_1h_clean = args.output_dir / "candles_1h_clean.parquet"
    path_4h_clean = args.output_dir / "candles_4h_derived_clean.parquet"
    quarantine_path = args.output_dir / "candle_quarantine.parquet"
    metadata.to_parquet(metadata_path, index=False)
    discovery_audit.to_parquet(audit_path, index=False)
    candles_15m.to_parquet(path_15m, index=False)
    candles_1h.to_parquet(path_1h, index=False)
    candles_4h.to_parquet(path_4h, index=False)
    quarantine_15m.clean.to_parquet(path_15m_clean, index=False)
    quarantine_1h.clean.to_parquet(path_1h_clean, index=False)
    candles_4h_clean.to_parquet(path_4h_clean, index=False)
    quarantine_audit.to_parquet(quarantine_path, index=False)

    artifacts = [
        universe_path,
        metadata_path,
        audit_path,
        path_15m,
        path_1h,
        path_4h,
        path_15m_clean,
        path_1h_clean,
        path_4h_clean,
        quarantine_path,
    ]
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
        "universe_snapshot_as_of_ns": universe_response.body.get("as_of_ns"),
        "selection": {
            "status": "exploratory_current_snapshot_mvp",
            "ranking": ["liquidity_num desc", "volume_24h desc", "condition_id asc"],
            "active_only": True,
            "candidate_limit": args.candidate_limit,
            "top_k": args.top_k,
            "minimum_probe_1h_bars": args.minimum_probe_bars,
            "minimum_probe_span_hours": args.minimum_span_hours,
            "condition_ids": condition_ids,
            "warning": (
                "Current-snapshot selection is retrospective for earlier timestamps and is not a "
                "Phase 4 cutoff-local experiment universe."
            ),
        },
        "intervals": {
            "15m": {
                "native_api_interval_minutes": 15,
                "rows": len(candles_15m),
                "clean_rows": len(quarantine_15m.clean),
            },
            "1h": {
                "native_api_interval_minutes": 60,
                "rows": len(candles_1h),
                "clean_rows": len(quarantine_1h.clean),
            },
            "4h": {
                "rows": len(candles_4h),
                "derived_from": "native 1h observed rows",
                "utc_aligned": True,
                "missing_hours_imputed": False,
                "coverage_columns": ["observed_1h_bars", "complete_1h_coverage"],
                "api_probe_note": (
                    "The documented interval=4h alias returned HTTP 400 during the 2026-09-20 "
                    "live probe, interval=4hour was rejected, and interval=240 returned no rows "
                    "for a market with native 1h data."
                ),
            },
            "4h_clean": {
                "rows": len(candles_4h_clean),
                "derived_from": path_1h_clean.name,
                "utc_aligned": True,
                "missing_hours_imputed": False,
                "coverage_columns": ["observed_1h_bars", "complete_1h_coverage"],
            },
        },
        "condition_candle_quarantine": {
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
        },
        "credential_persistence": "No token or Authorization header is stored.",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote recent-data MVP to {args.output_dir}")
    print(f"  markets: {len(condition_ids)}")
    print(f"  native 15m rows: {len(candles_15m):,}")
    print(f"  native 1h rows: {len(candles_1h):,}")
    print(f"  derived 4h rows: {len(candles_4h):,}")
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
