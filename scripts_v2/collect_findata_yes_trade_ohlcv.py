"""Build token-consistent Polymarket OHLCV from FinData trade records."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_acquisition.findata import FinDataClient, load_lumid_token  # noqa: E402
from data_acquisition.polymarket_pipeline import (  # noqa: E402
    aggregate_token_trades,
    fetch_complete_trades,
)


DEFAULT_SOURCE = (
    ROOT / "data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31"
)
DEFAULT_OUTPUT = (
    ROOT / "data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31_yes_trades"
)


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


def canonical_yes_token(outcomes: object, token_ids: object) -> str:
    outcomes_list = [str(value).strip().lower() for value in np.asarray(outcomes).tolist()]
    token_list = [str(value) for value in np.asarray(token_ids).tolist()]
    if len(outcomes_list) != len(token_list) or outcomes_list.count("yes") != 1:
        raise ValueError(f"expected one canonical Yes outcome, got {outcomes_list}")
    return token_list[outcomes_list.index("yes")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")

    source_manifest_path = args.source_dir / "manifest.json"
    source_metadata_path = args.source_dir / "market_metadata.parquet"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(source_metadata_path).copy()
    required = {
        "condition_id",
        "outcomes",
        "clob_token_ids",
        "start_date_search",
        "end_date_search",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"source metadata is missing fields: {missing}")

    metadata["canonical_outcome"] = "Yes"
    metadata["canonical_token_id"] = [
        canonical_yes_token(outcomes, tokens)
        for outcomes, tokens in zip(metadata["outcomes"], metadata["clob_token_ids"])
    ]
    start = pd.Timestamp(source_manifest["requested_interval"]["start_inclusive"])
    end = pd.Timestamp(source_manifest["requested_interval"]["end_exclusive"])
    ranges = {
        row.condition_id: (
            max(start, row.start_date_search).to_pydatetime(),
            min(end, row.end_date_search + pd.Timedelta(days=1)).to_pydatetime(),
        )
        for row in metadata.itertuples(index=False)
    }
    ranges = {condition_id: bounds for condition_id, bounds in ranges.items() if bounds[0] < bounds[1]}
    prepare_output(args.output_dir, args.overwrite)

    client = FinDataClient(load_lumid_token(ROOT / ".env"))
    print(f"Fetching token-identified trades for {len(ranges)} conditions...", flush=True)
    trades = fetch_complete_trades(client, ranges, workers=args.workers)
    allowed_tokens = {
        str(token)
        for values in metadata["clob_token_ids"]
        for token in np.asarray(values).tolist()
    }
    unknown = sorted(set(trades["token_id"].astype(str)) - allowed_tokens)
    if unknown:
        raise ValueError(f"trade response contains unknown token IDs: {unknown[:3]}")

    yes_map = metadata.set_index("condition_id")["canonical_token_id"].astype(str)
    expected = trades["condition_id"].map(yes_map)
    yes_trades = trades.loc[trades["token_id"].astype(str).eq(expected)].copy()
    bars = {
        label: aggregate_token_trades(yes_trades, interval_minutes=minutes)
        for label, minutes in (("15m", 15), ("1h", 60), ("4h", 240))
    }

    metadata_path = args.output_dir / "market_metadata.parquet"
    trades_path = args.output_dir / "trades_all_outcomes.parquet"
    yes_trades_path = args.output_dir / "trades_yes.parquet"
    paths = {
        "metadata": metadata_path,
        "trades_all": trades_path,
        "trades_yes": yes_trades_path,
        "15m": args.output_dir / "ohlcv_yes_15m.parquet",
        "1h": args.output_dir / "ohlcv_yes_1h.parquet",
        "4h": args.output_dir / "ohlcv_yes_4h.parquet",
    }
    metadata.to_parquet(metadata_path, index=False)
    trades.to_parquet(trades_path, index=False)
    yes_trades.to_parquet(yes_trades_path, index=False)
    for label, frame in bars.items():
        frame.to_parquet(paths[label], index=False)

    artifacts = list(paths.values())
    manifest = {
        "schema_version": 1,
        "source": "FinData Polymarket token-identified trades",
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_interval": source_manifest["requested_interval"],
        "canonical_outcome": "Yes",
        "construction": {
            "raw_endpoint": "/prediction-markets/trades/polymarket/{condition_id}",
            "trade_cap_handling": "30-day requests recursively bisected whenever limit=5000 is reached",
            "boundary": "half-open per-market ranges; endpoint spillover pruned locally",
            "aggregation": "UTC-aligned trade-price OHLC; volume=sum(size); absent bins remain absent",
            "no_imputation": True,
            "warning": (
                "Exploratory retrospective cohort only. Full-lifetime selection metadata is not a "
                "cutoff-local Phase 4 universe."
            ),
        },
        "source_provenance": {
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_metadata_sha256": sha256_file(source_metadata_path),
        },
        "counts": {
            "conditions": len(metadata),
            "conditions_with_any_trade": int(trades["condition_id"].nunique()),
            "conditions_with_yes_trade": int(yes_trades["condition_id"].nunique()),
            "all_outcome_trades": len(trades),
            "yes_trades": len(yes_trades),
            "yes_15m_bars": len(bars["15m"]),
            "yes_1h_bars": len(bars["1h"]),
            "yes_4h_bars": len(bars["4h"]),
        },
        "credential_persistence": "No token or Authorization header is stored.",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote token-consistent trade OHLCV to {args.output_dir}")
    for key, value in manifest["counts"].items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
