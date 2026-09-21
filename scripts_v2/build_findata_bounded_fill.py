"""Materialize the one-bar bounded-fill layer for clean FinData candles."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31"
)
RESOLUTIONS = {
    "15m": ("candles_15min_clean.parquet", "candles_15min_clean_ffill1.parquet", 15),
    "1h": ("candles_1h_clean.parquet", "candles_1h_clean_ffill1.parquet", 60),
}
OHLC = ["open", "high", "low", "close"]
REQUIRED = {"condition_id", "interval_minutes", "date", *OHLC, "volume"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_clean(frame: pd.DataFrame, *, interval_minutes: int) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"clean candle file is missing columns: {missing}")
    result = frame.copy()
    result["condition_id"] = result["condition_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result = result.sort_values(["condition_id", "date"], kind="stable").reset_index(drop=True)
    if result.duplicated(["condition_id", "date"]).any():
        raise ValueError("clean candle file contains duplicate condition/timestamp identities")
    if not result["interval_minutes"].eq(interval_minutes).all():
        raise ValueError("clean candle file does not match its declared native resolution")
    values = result[[*OHLC, "volume"]].to_numpy(np.float64)
    if not np.isfinite(values).all() or (result["volume"] < 0).any():
        raise ValueError("clean candle file contains invalid OHLCV values")
    invalid = (
        result["high"].lt(result[["open", "close", "low"]].max(axis=1))
        | result["low"].gt(result[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("clean candle file contains inconsistent OHLC values")
    return result


def build_bounded_fill(
    clean: pd.DataFrame,
    *,
    resolution: str,
    interval_minutes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add only complete isolated internal gaps and preserve every clean row exactly."""

    source = normalize_clean(clean, interval_minutes=interval_minutes)
    observed = source.copy()
    observed["native_resolution"] = resolution
    observed["is_observed"] = True
    observed["is_imputed"] = False
    observed["original_gap_length_bars"] = 0
    observed["time_since_last_observation"] = 0

    step = pd.Timedelta(minutes=interval_minutes)
    synthetic_rows: list[pd.Series] = []
    total_internal_missing = 0
    long_gap_rows = 0
    long_gap_events = 0
    for _, contract in source.groupby("condition_id", sort=True):
        contract = contract.reset_index(drop=True)
        differences = contract["date"].diff().iloc[1:]
        ratios = differences / step
        if not ratios.apply(float.is_integer).all():
            raise ValueError("contract contains timestamps off the declared native grid")
        missing = ratios.astype(int).sub(1)
        total_internal_missing += int(missing.sum())
        long = missing.gt(1)
        long_gap_rows += int(missing.loc[long].sum())
        long_gap_events += int(long.sum())
        for next_index in np.flatnonzero(missing.to_numpy() == 1) + 1:
            prior = contract.iloc[next_index - 1]
            following = contract.iloc[next_index]
            inserted_date = prior["date"] + step
            if inserted_date + step != following["date"]:
                raise AssertionError("candidate is not a complete isolated one-bar gap")
            inserted = prior.copy()
            inserted["date"] = inserted_date
            inserted[OHLC] = float(prior["close"])
            inserted["volume"] = 0.0
            if "trades" in inserted.index:
                inserted["trades"] = 0.0
            inserted["native_resolution"] = resolution
            inserted["is_observed"] = False
            inserted["is_imputed"] = True
            inserted["original_gap_length_bars"] = 1
            inserted["time_since_last_observation"] = interval_minutes
            synthetic_rows.append(inserted)

    synthetic = pd.DataFrame(synthetic_rows, columns=observed.columns)
    filled = pd.concat([observed, synthetic], ignore_index=True).sort_values(
        ["condition_id", "date"], kind="stable"
    ).reset_index(drop=True)
    filled["is_observed"] = filled["is_observed"].astype(bool)
    filled["is_imputed"] = filled["is_imputed"].astype(bool)
    filled["original_gap_length_bars"] = filled["original_gap_length_bars"].astype("int16")
    filled["time_since_last_observation"] = filled["time_since_last_observation"].astype("int64")

    if filled.duplicated(["condition_id", "date"]).any():
        raise AssertionError("bounded fill created duplicate identities")
    replayed_observed = filled.loc[filled["is_observed"], source.columns].reset_index(drop=True)
    assert_frame_equal(replayed_observed, source, check_exact=True, check_dtype=True)
    inserted = filled.loc[filled["is_imputed"]]
    if not (
        inserted[OHLC].eq(inserted["close"], axis=0).all().all()
        and inserted["volume"].eq(0).all()
        and inserted["original_gap_length_bars"].eq(1).all()
        and inserted["time_since_last_observation"].eq(interval_minutes).all()
    ):
        raise AssertionError("an inserted row violates the frozen bounded-fill policy")
    if len(filled) != len(source) + len(synthetic_rows):
        raise AssertionError("filled row accounting failed")

    return filled, {
        "clean_rows": len(source),
        "filled_rows": len(filled),
        "inserted_rows": len(synthetic_rows),
        "total_internal_missing_rows": total_internal_missing,
        "long_gap_rows_preserved": long_gap_rows,
        "long_gap_events_preserved": long_gap_events,
        "contracts": int(source["condition_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    allowed = (ROOT / "data_new").resolve()
    resolved = output_dir.resolve()
    if allowed != resolved and allowed not in resolved.parents:
        parser.error(f"output must be inside {allowed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [output_dir / spec[1] for spec in RESOLUTIONS.values()]
    manifest_path = output_dir / "fill_manifest.json"
    occupied = [path for path in [*targets, manifest_path] if path.exists()]
    if occupied and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing fill artifacts: {occupied}")

    records: dict[str, object] = {}
    for resolution, (input_name, output_name, minutes) in RESOLUTIONS.items():
        input_path = args.input_dir / input_name
        output_path = output_dir / output_name
        clean = pd.read_parquet(input_path)
        filled, counts = build_bounded_fill(
            clean,
            resolution=resolution,
            interval_minutes=minutes,
        )
        filled.to_parquet(output_path, index=False)
        replayed = pd.read_parquet(output_path)
        assert_frame_equal(replayed, filled, check_exact=True, check_dtype=True)
        records[resolution] = {
            "input": input_name,
            "input_sha256": sha256_file(input_path),
            "output": output_name,
            "output_sha256": sha256_file(output_path),
            **counts,
        }
        print(
            f"{resolution}: {counts['clean_rows']:,} clean + {counts['inserted_rows']:,} "
            f"isolated fills = {counts['filled_rows']:,}; "
            f"{counts['long_gap_rows_preserved']:,} long-gap slots remain absent",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "policy": {
            "maximum_missing_native_bars": 1,
            "complete_internal_gaps_only": True,
            "fill_ohlc": "previous observed close",
            "fill_volume": 0.0,
            "fill_trades_when_present": 0.0,
            "leading_trailing_gaps_filled": False,
            "longer_gaps_filled": False,
            "time_since_last_observation_unit": "minutes",
            "clean_sources_modified": False,
        },
        "resolutions": records,
        "validation": {
            "observed_rows_exactly_equal_clean_sources": True,
            "inserted_rows_are_complete_isolated_one_bar_gaps": True,
            "inserted_ohlc_flat_at_previous_close": True,
            "inserted_volume_zero": True,
            "long_gaps_preserved_as_absent_timestamps": True,
            "parquet_round_trip_exact": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
