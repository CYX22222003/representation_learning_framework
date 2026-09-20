"""Create audited, gap-preserving FinData condition-candle quarantine artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_acquisition.findata import aggregate_one_hour_to_four_hour  # noqa: E402
from data_acquisition.polymarket_pipeline import (  # noqa: E402
    CANDLE_QUARANTINE_RULE_VERSION,
    quarantine_condition_candles,
)


DEFAULT_INPUT = (
    ROOT
    / "data_new/findata/polymarket/historical_diverse_top24_2025-12-01_2026-08-31"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_output(path: Path, *, audit_only: bool, overwrite: bool) -> None:
    allowed = (ROOT / "data_new").resolve()
    resolved = path.resolve()
    if allowed != resolved and allowed not in resolved.parents:
        raise ValueError(f"input directory must be inside {allowed}: {resolved}")
    outputs = (
        [
            path / "candle_quarantine_preview.parquet",
            path / "quarantine_preview.json",
            path / "quarantine_preview_report.md",
        ]
        if audit_only
        else [
            path / "candles_15min_clean.parquet",
            path / "candles_1h_clean.parquet",
            path / "candles_4h_derived_clean.parquet",
            path / "candle_quarantine.parquet",
            path / "quarantine_manifest.json",
        ]
    )
    occupied = [item for item in outputs if item.exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            "refusing to overwrite quarantine artifacts: "
            + ", ".join(str(item) for item in occupied)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--maximum-quarantine-fraction",
        type=float,
        default=0.01,
        help="fail if more than this fraction of either native resolution is flagged",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write anomaly candidates and summary without creating clean/pruned candle files",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checked_output(args.input_dir, audit_only=args.audit_only, overwrite=args.overwrite)
    raw_15m_path = args.input_dir / "candles_15min.parquet"
    raw_1h_path = args.input_dir / "candles_1h.parquet"
    candles_15m = pd.read_parquet(raw_15m_path)
    candles_1h = pd.read_parquet(raw_1h_path)

    result_15m = quarantine_condition_candles(
        candles_15m,
        interval_minutes=15,
        maximum_quarantine_fraction=args.maximum_quarantine_fraction,
        enforce_budget=not args.audit_only,
    )
    result_1h = quarantine_condition_candles(
        candles_1h,
        interval_minutes=60,
        maximum_quarantine_fraction=args.maximum_quarantine_fraction,
        enforce_budget=not args.audit_only,
    )
    audits = []
    for resolution, result in (("15m", result_15m), ("1h", result_1h)):
        audit = result.audit.copy()
        audit.insert(0, "resolution", resolution)
        audits.append(audit)
    quarantine = pd.concat(audits, ignore_index=True)

    if args.audit_only:
        audit_path = args.input_dir / "candle_quarantine_preview.parquet"
        quarantine.to_parquet(audit_path, index=False)
        metadata_path = args.input_dir / "market_metadata.parquet"
        metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else pd.DataFrame()
        title_column = next(
            (name for name in ("question_snapshot", "title", "question") if name in metadata),
            None,
        )
        title_by_condition = (
            metadata.drop_duplicates("condition_id").set_index("condition_id")[title_column]
            if title_column is not None and "condition_id" in metadata
            else pd.Series(dtype=object)
        )
        by_contract = []
        for resolution, raw, result in (
            ("15m", candles_15m, result_15m),
            ("1h", candles_1h, result_1h),
        ):
            raw_sizes = raw.groupby("condition_id").size()
            candidate_sizes = result.audit.groupby("condition_id").size()
            complement_sizes = (
                result.audit.loc[result.audit["transient_complement"]]
                .groupby("condition_id")
                .size()
            )
            wide_sizes = (
                result.audit.loc[result.audit["transient_wide_range"]]
                .groupby("condition_id")
                .size()
            )
            for condition_id, candidate_rows in candidate_sizes.items():
                input_rows = int(raw_sizes.loc[condition_id])
                by_contract.append(
                    {
                        "resolution": resolution,
                        "condition_id": condition_id,
                        "question": title_by_condition.get(condition_id),
                        "input_rows": input_rows,
                        "candidate_rows": int(candidate_rows),
                        "candidate_fraction": float(candidate_rows / input_rows),
                        "transient_complement_rows": int(
                            complement_sizes.get(condition_id, 0)
                        ),
                        "transient_wide_range_rows": int(wide_sizes.get(condition_id, 0)),
                    }
                )
        by_contract.sort(
            key=lambda row: (row["resolution"], -row["candidate_rows"], row["condition_id"])
        )
        investigated_contracts = len(
            set(candles_15m["condition_id"]) | set(candles_1h["condition_id"])
        )
        affected_contracts = len({row["condition_id"] for row in by_contract})
        preview = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "audit_only_no_rows_removed",
            "rule_version": CANDLE_QUARANTINE_RULE_VERSION,
            "maximum_quarantine_fraction": args.maximum_quarantine_fraction,
            "raw_inputs": {
                raw_15m_path.name: {
                    "sha256": sha256_file(raw_15m_path),
                    "rows": len(candles_15m),
                },
                raw_1h_path.name: {
                    "sha256": sha256_file(raw_1h_path),
                    "rows": len(candles_1h),
                },
            },
            "resolutions": {"15m": result_15m.summary, "1h": result_1h.summary},
            "contracts_investigated": investigated_contracts,
            "affected_contracts_union": affected_contracts,
            "affected_contracts": by_contract,
            "candidate_artifact": {
                "filename": audit_path.name,
                "sha256": sha256_file(audit_path),
                "rows": len(quarantine),
            },
            "contract": {
                "raw_files_immutable": True,
                "prices_repaired": False,
                "clean_files_written": False,
                "forward_confirmation_used": True,
                "walk_forward_decisions_require_quarantine_available_at_before_cutoff": True,
            },
        }
        preview_path = args.input_dir / "quarantine_preview.json"
        preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        report_lines = [
            "# FinData Forward-Confirmed Quarantine Preview",
            "",
            f"Rule: `{CANDLE_QUARANTINE_RULE_VERSION}`. This preview removed no rows.",
            "",
            f"Investigated {investigated_contracts} contracts; {affected_contracts} have at least "
            "one candidate at either resolution.",
            "",
            "| Resolution | Raw rows | Candidates | Candidate share | Persistent large jumps retained |",
            "|---|---:|---:|---:|---:|",
        ]
        for resolution, result in (("15m", result_15m), ("1h", result_1h)):
            summary = result.summary
            report_lines.append(
                f"| {resolution} | {summary['input_rows']:,} | "
                f"{summary['quarantined_rows']:,} | {summary['quarantined_fraction']:.4%} | "
                f"{summary['confirmed_persistent_large_jump_rows']:,} |"
            )
        report_lines.extend(
            [
                "",
                "Persistent large moves are retained when at least two of the next four observed bars "
                "exist within the confirmation span and at least two thirds remain closer to the new "
                "regime. Candidate rows are transient reversions or unsupported wide candles.",
                "",
                "| Resolution | Candidate rows | Contract share | Complement | Wide range | Contract |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in by_contract:
            question = row["question"] or row["condition_id"]
            report_lines.append(
                f"| {row['resolution']} | {row['candidate_rows']:,} | "
                f"{row['candidate_fraction']:.4%} | {row['transient_complement_rows']:,} | "
                f"{row['transient_wide_range_rows']:,} | {question} |"
            )
        report_lines.extend(
            [
                "",
                "The rule is intentionally forward-confirmed. For walk-forward use, a decision is "
                "available only after its recorded `quarantine_available_at`; applying it earlier "
                "would leak future information. Condition-level candles remain noncanonical without "
                "outcome-token identity even after quarantine.",
                "",
            ]
        )
        report_path = args.input_dir / "quarantine_preview_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        preview["report_artifact"] = {
            "filename": report_path.name,
            "sha256": sha256_file(report_path),
        }
        preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(f"Wrote anomaly preview under {args.input_dir}; no rows were removed")
        for resolution, summary in preview["resolutions"].items():
            print(
                f"  {resolution}: candidates {summary['quarantined_rows']:,}/"
                f"{summary['input_rows']:,} ({summary['quarantined_fraction']:.4%}); "
                f"budget exceeded: {summary['budget_exceeded']}"
            )
        return

    clean_4h = aggregate_one_hour_to_four_hour(result_1h.clean)

    path_15m = args.input_dir / "candles_15min_clean.parquet"
    path_1h = args.input_dir / "candles_1h_clean.parquet"
    path_4h = args.input_dir / "candles_4h_derived_clean.parquet"
    audit_path = args.input_dir / "candle_quarantine.parquet"
    result_15m.clean.to_parquet(path_15m, index=False)
    result_1h.clean.to_parquet(path_1h, index=False)
    clean_4h.to_parquet(path_4h, index=False)
    quarantine.to_parquet(audit_path, index=False)

    outputs = [path_15m, path_1h, path_4h, audit_path]
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule_version": CANDLE_QUARANTINE_RULE_VERSION,
        "maximum_quarantine_fraction": args.maximum_quarantine_fraction,
        "raw_inputs": {
            raw_15m_path.name: {"sha256": sha256_file(raw_15m_path), "rows": len(candles_15m)},
            raw_1h_path.name: {"sha256": sha256_file(raw_1h_path), "rows": len(candles_1h)},
        },
        "resolutions": {"15m": result_15m.summary, "1h": result_1h.summary},
        "derived_4h": {
            "rows": len(clean_4h),
            "derived_from": path_1h.name,
            "missing_quarantined_hours_imputed": False,
            "required_downstream_columns": ["observed_1h_bars", "complete_1h_coverage"],
        },
        "contract": {
            "raw_files_immutable": True,
            "prices_repaired": False,
            "quarantined_timestamps_remain_gaps": True,
            "downstream_windows_must_require_exact_consecutive_timestamps": True,
            "condition_candles_remain_noncanonical_without_outcome_token_identity": True,
            "forward_confirmation_used": True,
            "walk_forward_decisions_require_quarantine_available_at_before_cutoff": True,
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in outputs
        },
    }
    manifest_path = args.input_dir / "quarantine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote quarantine artifacts under {args.input_dir}")
    for resolution, summary in manifest["resolutions"].items():
        print(
            f"  {resolution}: quarantined {summary['quarantined_rows']:,}/"
            f"{summary['input_rows']:,} ({summary['quarantined_fraction']:.4%}); "
            f"retained {summary['retained_fraction']:.4%}"
        )
    print(f"  clean derived 4h rows: {len(clean_4h):,}")


if __name__ == "__main__":
    main()
