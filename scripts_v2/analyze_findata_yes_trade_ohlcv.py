"""Audit token-consistent FinData YES-trade OHLCV and target feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31_yes_trades"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, record in manifest["artifacts"].items():
        path = root / filename
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
    return manifest


def analyze_resolution(
    frame: pd.DataFrame,
    *,
    resolution: str,
    interval_minutes: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    steps = 8 * 60 // interval_minutes
    records: list[dict[str, object]] = []
    pooled: list[np.ndarray] = []
    for condition_id, market in frame.groupby("condition_id", sort=True):
        market = market.sort_values("date").reset_index(drop=True)
        eligible = market["date"].shift(-steps).sub(market["date"]).eq(pd.Timedelta(hours=8))
        indices = np.flatnonzero(eligible.fillna(False).to_numpy())
        close = market["close"].to_numpy(np.float64)
        delta = close[indices + steps] - close[indices]
        if len(delta):
            pooled.append(delta)

        seq64_h2 = 0
        if interval_minutes == 240 and len(market) >= 66:
            for target_index in range(65, len(market)):
                block = market.iloc[target_index - 65 : target_index + 1]
                if block["date"].iloc[-1] - block["date"].iloc[0] == pd.Timedelta(hours=260):
                    seq64_h2 += 1
        records.append(
            {
                "resolution": resolution,
                "condition_id": condition_id,
                "bars": len(market),
                "strict_eight_hour_targets": len(delta),
                "seq64_h2_rows_without_imputation": seq64_h2,
                "exact_zero_fraction": float((delta == 0).mean()) if len(delta) else math.nan,
                "stable_fraction_tau005": (
                    float((np.abs(delta) <= 0.005).mean()) if len(delta) else math.nan
                ),
                "mean_absolute_movement": float(np.abs(delta).mean()) if len(delta) else math.nan,
            }
        )
    by_market = pd.DataFrame(records)
    delta = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    summary = {
        "resolution": resolution,
        "bars": len(frame),
        "markets_with_bars": int(frame["condition_id"].nunique()),
        "markets_with_targets": int(by_market["strict_eight_hour_targets"].gt(0).sum()),
        "strict_eight_hour_targets": len(delta),
        "seq64_h2_rows_without_imputation": int(
            by_market["seq64_h2_rows_without_imputation"].sum()
        ),
        "seq64_h2_markets": int(by_market["seq64_h2_rows_without_imputation"].gt(0).sum()),
        "exact_zero_fraction": float((delta == 0).mean()) if len(delta) else None,
        "stable_fraction_tau005": float((np.abs(delta) <= 0.005).mean()) if len(delta) else None,
        "mean_absolute_movement": float(np.abs(delta).mean()) if len(delta) else None,
        "contract_macro_exact_zero_fraction": float(by_market["exact_zero_fraction"].mean()),
        "contract_macro_stable_fraction_tau005": float(by_market["stable_fraction_tau005"].mean()),
        "contract_macro_mean_absolute_movement": float(by_market["mean_absolute_movement"].mean()),
    }
    return by_market, summary


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    manifest = verify_manifest(args.input_dir)
    metadata = pd.read_parquet(args.input_dir / "market_metadata.parquet")
    results: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for resolution, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
        frame = pd.read_parquet(args.input_dir / f"ohlcv_yes_{resolution}.parquet")
        by_market, summary = analyze_resolution(
            frame, resolution=resolution, interval_minutes=minutes
        )
        results.append(by_market)
        summaries.append(summary)

    by_market = pd.concat(results, ignore_index=True).merge(
        metadata[["condition_id", "question_snapshot", "category"]],
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    analysis_dir = args.input_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    by_market.to_parquet(analysis_dir / "movement_by_market_and_resolution.parquet", index=False)
    summary = {
        "source_counts": manifest["counts"],
        "movement": summaries,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# FinData Token-Consistent YES-Trade OHLCV Audit",
        "",
        f"Requested interval: `{manifest['requested_interval']['start_inclusive']}` to "
        f"`{manifest['requested_interval']['end_exclusive']}` (half-open, UTC).",
        "",
        "The condition-level candle endpoint can mix complementary YES/NO observations. This audit instead "
        "uses only trades whose token ID is mapped to the market's declared `Yes` outcome. No outcome or "
        "settlement value is used to orient individual observations.",
        "",
        f"- Selected conditions: {manifest['counts']['conditions']}.",
        f"- Conditions with token-identified trades: {manifest['counts']['conditions_with_any_trade']}.",
        f"- All-outcome trades: {manifest['counts']['all_outcome_trades']:,}.",
        f"- Canonical YES trades: {manifest['counts']['yes_trades']:,}.",
        "",
        "| Resolution | Bars | Markets | Strict 8h targets | Target markets | Exact zero | Stable (0.005) | Mean abs move | Seq64+h2 rows | Seq64 markets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['resolution']} | {row['bars']:,} | {row['markets_with_bars']} | "
            f"{row['strict_eight_hour_targets']:,} | {row['markets_with_targets']} | "
            f"{percent(row['exact_zero_fraction'])} | {percent(row['stable_fraction_tau005'])} | "
            f"{number(row['mean_absolute_movement'])} | "
            f"{row['seq64_h2_rows_without_imputation']:,} | {row['seq64_h2_markets']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The token-consistent subset is more dynamic than the original three-market MVP, but it is too "
            "sparse and narrow for the current Phase 4 encoder contract. Only half of the selected conditions "
            "returned any token-identified trades. At four-hour resolution, complete 64-bar context plus a "
            "two-bar target exists only for two closely related World Cup contracts. Fifteen-minute trades "
            "never form an uninterrupted eight-hour run.",
            "",
            "Therefore this cohort supports the claim that recent active markets can contain meaningful moves, "
            "but it does not establish that all recent markets are more dynamic or supply a representative "
            "training set. A valid recent-data experiment needs token-specific midpoint history from the "
            "provider, or a separately collected token-specific order-book source, plus cutoff-local selection.",
            "",
        ]
    )
    report = "\n".join(lines)
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
