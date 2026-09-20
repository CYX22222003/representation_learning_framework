"""Reusable discovery and download pipeline for recent Polymarket candles."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .findata import (
    FinDataClient,
    candles_to_frame,
    format_rfc3339,
    iter_time_chunks,
    trades_to_frame,
    validate_candles,
)


_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Avoid bare ticker substrings such as ``eth``: they incorrectly classify
    # titles containing words such as "Ethiopia" as crypto markets.
    ("crypto", ("bitcoin", "ethereum", "solana", "crypto", "xrp", "dogecoin")),
    ("geopolitics", ("iran", "israel", "ukraine", "russia", "war", "peace deal", "ceasefire", "nato")),
    ("politics", ("election", "president", "presidential", "nomination", "congress", "senate", "governor", "prime minister", "party")),
    ("economy_business", ("fed ", "interest rate", "inflation", "gdp", "ipo", "stock", "company", "tariff", "recession", "earnings")),
    ("sports", (" vs", "vs.", "win on 2026", "world cup", "super bowl", "championship", "finals", "league", "ufc", "spread:", "o/u", "nba", "nfl", "mlb", "nhl")),
    ("entertainment_culture", ("oscar", "movie", "film", "album", "grammy", "box office", "celebrity", "tweet", "followers")),
    ("science_weather", ("temperature", "weather", "hurricane", "earthquake", "space", "nasa", "ai model")),
)

CANDLE_QUARANTINE_RULE_VERSION = "condition-orientation-v2-forward-confirmed"


@dataclass(frozen=True)
class CandleQuarantineResult:
    """Raw-preserving result of the condition-candle quarantine screen."""

    clean: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, object]


def classify_market(title: str) -> str:
    lowered = f" {title.lower()} "
    for category, needles in _CATEGORY_PATTERNS:
        if any(needle in lowered for needle in needles):
            return category
    return "other"


def event_family(title: str) -> str:
    lowered = title.lower()
    declared = (
        ("2026_fifa_world_cup", "2026 fifa world cup"),
        ("2028_democratic_nomination", "2028 democratic presidential nomination"),
        ("2028_republican_nomination", "2028 republican presidential nomination"),
        ("fed_chair", "fed chair"),
    )
    for family, phrase in declared:
        if phrase in lowered:
            return family
    normalized = re.sub(r"\b(will|be|the|a|an|by|before|after|in|on|of|to)\b", " ", lowered)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "#", normalized)
    normalized = re.sub(r"[^a-z#]+", " ", normalized)
    tokens = normalized.split()
    return "_".join(tokens[:6]) or "other"


def historical_catalog(
    client: FinDataClient,
    *,
    statuses: Iterable[str] = ("open", "closed"),
    page_size: int = 500,
    pages_per_status: int = 4,
    workers: int = 4,
) -> pd.DataFrame:
    jobs = [
        (status, page * page_size)
        for status in statuses
        for page in range(pages_per_status)
    ]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                client.polymarket_search,
                status=status,
                limit=page_size,
                offset=offset,
            ): (status, offset)
            for status, offset in jobs
        }
        for future in as_completed(futures):
            status, offset = futures[future]
            body = future.result().body
            if not isinstance(body, list):
                raise ValueError(f"unexpected search response for {status=} {offset=}")
            for record in body:
                if not isinstance(record, dict):
                    raise ValueError("market search response contains a non-object row")
                copied = dict(record)
                copied["search_status"] = status
                copied["search_offset"] = offset
                rows.append(copied)
    if not rows:
        raise RuntimeError("historical market search returned no rows")
    frame = pd.DataFrame(rows)
    required = {"condition_id", "title", "volume", "start_date", "end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market search rows are missing fields: {missing}")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["start_date"] = pd.to_datetime(frame["start_date"], utc=True, errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], utc=True, errors="coerce")
    frame = frame.loc[
        frame["condition_id"].notna() & frame["start_date"].notna() & frame["end_date"].notna()
    ].copy()
    frame = frame.sort_values(
        ["condition_id", "volume", "search_status", "search_offset"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates("condition_id", keep="first")
    frame["category"] = frame["title"].map(classify_market)
    frame["event_family"] = frame["title"].map(event_family)
    return frame.sort_values(["volume", "condition_id"], ascending=[False, True]).reset_index(drop=True)


def overlapping_candidates(
    catalog: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    minimum_overlap_hours: float,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = catalog.loc[(catalog["start_date"] < end_ts) & (catalog["end_date"] > start_ts)].copy()
    frame["requested_overlap_start"] = frame["start_date"].clip(lower=start_ts)
    frame["requested_overlap_end"] = frame["end_date"].clip(upper=end_ts)
    frame["overlap_hours"] = (
        frame["requested_overlap_end"] - frame["requested_overlap_start"]
    ).dt.total_seconds() / 3600.0
    frame = frame.loc[frame["overlap_hours"] >= minimum_overlap_hours].copy()
    return frame.sort_values(["volume", "condition_id"], ascending=[False, True]).reset_index(drop=True)


def diverse_candidate_order(
    candidates: pd.DataFrame,
    *,
    candidate_limit: int,
    max_per_family: int,
) -> pd.DataFrame:
    buckets = {
        category: group.sort_values(["volume", "condition_id"], ascending=[False, True]).to_dict("records")
        for category, group in candidates.groupby("category", sort=True)
    }
    ordered: list[dict[str, object]] = []
    family_counts: dict[str, int] = {}
    while len(ordered) < candidate_limit and any(buckets.values()):
        made_progress = False
        for category in sorted(buckets):
            while buckets[category]:
                row = buckets[category].pop(0)
                family = str(row["event_family"])
                if family_counts.get(family, 0) >= max_per_family:
                    continue
                family_counts[family] = family_counts.get(family, 0) + 1
                ordered.append(row)
                made_progress = True
                break
            if len(ordered) >= candidate_limit:
                break
        if not made_progress:
            break
    result = pd.DataFrame(ordered)
    if result.empty:
        return result
    result.insert(0, "candidate_rank", range(1, len(result) + 1))
    return result


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
    return candles_to_frame(response.body, condition_id=condition_id, interval_minutes=interval_minutes)


def probe_candidates(
    client: FinDataClient,
    candidates: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    minimum_bars: int,
    minimum_span_hours: float,
    top_k: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_interval, client, row.condition_id, 60, start, end): row.condition_id
            for row in candidates.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            condition_id = futures[future]
            try:
                frames[condition_id] = future.result()
            except Exception as exc:
                errors[condition_id] = type(exc).__name__
            if completed % 20 == 0 or completed == len(futures):
                print(f"  historical probes: {completed}/{len(futures)}", flush=True)

    audit = candidates.copy()
    counts: list[int] = []
    spans: list[float] = []
    for row in audit.itertuples(index=False):
        frame = frames.get(row.condition_id)
        counts.append(0 if frame is None else len(frame))
        spans.append(
            0.0
            if frame is None or len(frame) < 2
            else float((frame["date"].max() - frame["date"].min()).total_seconds() / 3600.0)
        )
    audit["probe_1h_rows"] = counts
    audit["probe_span_hours"] = spans
    audit["probe_error"] = audit["condition_id"].map(errors)
    audit["eligible"] = (
        audit["probe_1h_rows"].ge(minimum_bars)
        & audit["probe_span_hours"].ge(minimum_span_hours)
    )
    selected = audit.loc[audit["eligible"]].head(top_k).copy()
    if len(selected) < top_k:
        raise RuntimeError(
            f"only {len(selected)} of {top_k} requested historical markets passed the hourly probe"
        )
    selected["selection_rank"] = range(1, len(selected) + 1)
    return selected.reset_index(drop=True), audit.reset_index(drop=True)


def fetch_complete_candles(
    client: FinDataClient,
    ranges: Mapping[str, tuple[datetime, datetime]],
    *,
    interval_minutes: int,
    global_start: datetime,
    global_end: datetime,
    workers: int,
) -> pd.DataFrame:
    chunk = timedelta(days=7 if interval_minutes == 15 else 30)
    jobs = [
        (condition_id, chunk_start, chunk_end)
        for condition_id, (market_start, market_end) in ranges.items()
        for chunk_start, chunk_end in iter_time_chunks(market_start, market_end, chunk)
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
    in_requested_range = pd.Series(False, index=result.index)
    for condition_id, (market_start, market_end) in ranges.items():
        in_requested_range |= (
            result["condition_id"].eq(condition_id)
            & result["date"].ge(pd.Timestamp(market_start))
            & result["date"].lt(pd.Timestamp(market_end))
        )
    result = result.loc[in_requested_range].reset_index(drop=True)
    for _, market in result.groupby("condition_id", sort=False):
        validate_candles(
            market.drop(columns=["condition_id"]).sort_values("date").reset_index(drop=True),
            start=global_start,
            end=global_end,
        )
    return result


def quarantine_condition_candles(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    maximum_quarantine_fraction: float = 0.01,
    jump_threshold: float = 0.5,
    complement_tolerance: float = 0.02,
    wide_range_threshold: float = 0.5,
    confirmation_bars: int = 4,
    minimum_confirmation_bars: int = 2,
    confirmation_fraction: float = 2.0 / 3.0,
    enforce_budget: bool = True,
) -> CandleQuarantineResult:
    """Quarantine likely mixed-outcome candles without repairing their prices.

    The screen looks ahead over a short, bounded confirmation window.  A large
    complementary transition is quarantined only when the future closes move
    back toward the previous regime.  A wide candle is quarantined only when
    the future fails to support its closing price as a persistent new regime.
    A drastic move whose future closes remain near the new level is retained.

    The original frame is never modified.  Removing a row deliberately leaves
    a timestamp gap so downstream aggregation and window builders can reject
    any sample that crosses it.  The audit records when the future confirmation
    became available; a walk-forward consumer must not apply the decision at
    an earlier cutoff.  The batch fails closed if more than the declared
    fraction would be quarantined.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if not 0.0 <= maximum_quarantine_fraction < 1.0:
        raise ValueError("maximum_quarantine_fraction must be in [0, 1)")
    if confirmation_bars <= 0:
        raise ValueError("confirmation_bars must be positive")
    if not 1 <= minimum_confirmation_bars <= confirmation_bars:
        raise ValueError("minimum_confirmation_bars must be between 1 and confirmation_bars")
    if not 0.5 <= confirmation_fraction <= 1.0:
        raise ValueError("confirmation_fraction must be in [0.5, 1.0]")
    required = {
        "condition_id",
        "interval_minutes",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candle quarantine input is missing columns: {missing}")

    ordered = frame.sort_values(["condition_id", "date"]).reset_index(drop=True).copy()
    if ordered.empty:
        return CandleQuarantineResult(
            clean=ordered,
            audit=pd.DataFrame(),
            summary={
                "rule_version": CANDLE_QUARANTINE_RULE_VERSION,
                "input_rows": 0,
                "quarantined_rows": 0,
                "retained_rows": 0,
                "quarantined_fraction": 0.0,
                "retained_fraction": 1.0,
                "affected_contracts": 0,
                "raw_large_jump_rows": 0,
                "confirmed_persistent_large_jump_rows": 0,
                "confirmed_reverting_large_jump_rows": 0,
                "unconfirmed_large_jump_rows": 0,
                "raw_complement_transition_rows": 0,
                "retained_persistent_complement_rows": 0,
                "retained_unconfirmed_complement_rows": 0,
                "transient_complement_rows": 0,
                "raw_wide_range_rows": 0,
                "retained_persistent_wide_range_rows": 0,
                "retained_unconfirmed_wide_range_rows": 0,
                "transient_wide_range_rows": 0,
                "maximum_contract_quarantined_fraction": 0.0,
                "maximum_quarantine_fraction": maximum_quarantine_fraction,
                "budget_exceeded": False,
                "confirmation_bars": confirmation_bars,
                "minimum_confirmation_bars": minimum_confirmation_bars,
                "confirmation_fraction": confirmation_fraction,
                "forward_looking": True,
            },
        )
    observed_intervals = set(ordered["interval_minutes"].dropna().astype(int).unique())
    if observed_intervals != {interval_minutes}:
        raise ValueError(
            f"expected only interval_minutes={interval_minutes}, observed {sorted(observed_intervals)}"
        )

    grouped = ordered.groupby("condition_id", sort=False)
    previous_close = grouped["close"].shift()
    current_close = ordered["close"].to_numpy(np.float64)
    previous_close_values = previous_close.to_numpy(np.float64)
    confirmation_span = pd.Timedelta(minutes=interval_minutes * confirmation_bars)

    future_close_columns: list[np.ndarray] = []
    future_valid_columns: list[np.ndarray] = []
    future_date_columns: list[pd.Series] = []
    for offset in range(1, confirmation_bars + 1):
        future_close = grouped["close"].shift(-offset)
        future_date = grouped["date"].shift(-offset)
        future_gap = future_date - ordered["date"]
        valid = future_gap.gt(pd.Timedelta(0)) & future_gap.le(confirmation_span)
        future_close_columns.append(future_close.to_numpy(np.float64))
        future_valid_columns.append(valid.to_numpy(bool))
        future_date_columns.append(future_date.where(valid))

    future_closes = np.column_stack(future_close_columns)
    future_valid = np.column_stack(future_valid_columns)
    confirmation_rows = future_valid.sum(axis=1)
    enough_confirmation = confirmation_rows >= minimum_confirmation_bars
    distance_to_previous = np.abs(future_closes - previous_close_values[:, None])
    distance_to_current = np.abs(future_closes - current_close[:, None])
    closer_to_previous = future_valid & (distance_to_previous < distance_to_current)
    closer_to_current = future_valid & (distance_to_current < distance_to_previous)
    reversion_fraction = np.divide(
        closer_to_previous.sum(axis=1),
        confirmation_rows,
        out=np.zeros(len(ordered), dtype=np.float64),
        where=confirmation_rows > 0,
    )
    persistence_fraction = np.divide(
        closer_to_current.sum(axis=1),
        confirmation_rows,
        out=np.zeros(len(ordered), dtype=np.float64),
        where=confirmation_rows > 0,
    )
    confirmation_end_date = pd.concat(future_date_columns, axis=1).max(axis=1)
    quarantine_available_at = confirmation_end_date + pd.Timedelta(minutes=interval_minutes)

    large_jump = (
        previous_close.notna()
        & ordered["close"].sub(previous_close).abs().gt(jump_threshold)
    )
    previous_complement = (
        large_jump
        & ordered["close"].add(previous_close).sub(1.0).abs().le(complement_tolerance)
    )
    future_reverts = enough_confirmation & (reversion_fraction >= confirmation_fraction)
    future_supports_new_regime = (
        previous_close.notna().to_numpy()
        & (np.abs(current_close - previous_close_values) > jump_threshold)
        & enough_confirmation
        & (persistence_fraction >= confirmation_fraction)
    )
    transient_complement = previous_complement.to_numpy() & future_reverts
    wide_range = ordered["high"].sub(ordered["low"]).gt(wide_range_threshold)
    base_transient_wide_range = (
        wide_range.to_numpy()
        & previous_close.notna().to_numpy()
        & enough_confirmation
        & ~future_supports_new_regime
    )
    previous_initial_quarantine = (
        pd.Series(transient_complement | base_transient_wide_range, index=ordered.index)
        .groupby(ordered["condition_id"], sort=False)
        .shift(fill_value=False)
        .to_numpy(bool)
    )
    transient_wide_range = base_transient_wide_range | (
        wide_range.to_numpy()
        & previous_close.notna().to_numpy()
        & enough_confirmation
        & previous_initial_quarantine
    )
    quarantined = transient_complement | transient_wide_range
    confirmed_reverting_large_jump = large_jump.to_numpy() & future_reverts
    confirmed_persistent_large_jump = large_jump.to_numpy() & future_supports_new_regime
    unconfirmed_large_jump = large_jump.to_numpy() & ~(
        confirmed_reverting_large_jump | confirmed_persistent_large_jump
    )
    retained_persistent_complement = (
        previous_complement.to_numpy() & ~quarantined & future_supports_new_regime
    )
    retained_unconfirmed_complement = (
        previous_complement.to_numpy() & ~quarantined & ~future_supports_new_regime
    )
    retained_wide_range = wide_range.to_numpy() & ~quarantined
    retained_persistent_wide_range = retained_wide_range & future_supports_new_regime
    retained_unconfirmed_wide_range = retained_wide_range & ~future_supports_new_regime

    audit = ordered.loc[quarantined].copy()
    audit["quarantine_rule_version"] = CANDLE_QUARANTINE_RULE_VERSION
    audit["transient_complement"] = transient_complement[quarantined]
    audit["wide_range"] = wide_range.loc[quarantined].to_numpy(bool)
    audit["transient_wide_range"] = transient_wide_range[quarantined]
    audit["quarantine_reason"] = np.select(
        [
            audit["transient_complement"] & audit["transient_wide_range"],
            audit["transient_complement"],
            audit["transient_wide_range"],
        ],
        ["transient_complement+wide_range", "transient_complement", "transient_wide_range"],
        default="unknown",
    )
    audit["previous_close"] = previous_close.loc[quarantined].to_numpy()
    audit["confirmation_rows"] = confirmation_rows[quarantined]
    audit["future_reversion_fraction"] = reversion_fraction[quarantined]
    audit["future_persistence_fraction"] = persistence_fraction[quarantined]
    audit["future_supports_new_regime"] = future_supports_new_regime[quarantined]
    audit["confirmation_end_date"] = confirmation_end_date.loc[quarantined].to_numpy()
    audit["quarantine_available_at"] = quarantine_available_at.loc[quarantined].to_numpy()

    quarantined_fraction = float(quarantined.mean())
    budget_exceeded = quarantined_fraction > maximum_quarantine_fraction
    if enforce_budget and budget_exceeded:
        raise RuntimeError(
            f"condition-candle quarantine would remove {quarantined.sum()}/{len(ordered)} rows "
            f"({quarantined_fraction:.2%}), exceeding the "
            f"{maximum_quarantine_fraction:.2%} safety budget"
        )

    contract_sizes = ordered.groupby("condition_id").size()
    contract_quarantined = audit.groupby("condition_id").size()
    contract_fractions = contract_quarantined.div(contract_sizes, fill_value=0.0)
    clean = ordered.loc[~quarantined].reset_index(drop=True)
    summary = {
        "rule_version": CANDLE_QUARANTINE_RULE_VERSION,
        "input_rows": len(ordered),
        "quarantined_rows": len(audit),
        "retained_rows": len(clean),
        "quarantined_fraction": quarantined_fraction,
        "retained_fraction": float(1.0 - quarantined_fraction),
        "affected_contracts": int(audit["condition_id"].nunique()),
        "raw_large_jump_rows": int(large_jump.sum()),
        "confirmed_persistent_large_jump_rows": int(confirmed_persistent_large_jump.sum()),
        "confirmed_reverting_large_jump_rows": int(confirmed_reverting_large_jump.sum()),
        "unconfirmed_large_jump_rows": int(unconfirmed_large_jump.sum()),
        "raw_complement_transition_rows": int(previous_complement.sum()),
        "retained_persistent_complement_rows": int(retained_persistent_complement.sum()),
        "retained_unconfirmed_complement_rows": int(retained_unconfirmed_complement.sum()),
        "transient_complement_rows": int(transient_complement.sum()),
        "transient_wide_range_rows": int(transient_wide_range.sum()),
        "raw_wide_range_rows": int(wide_range.sum()),
        "retained_persistent_wide_range_rows": int(retained_persistent_wide_range.sum()),
        "retained_unconfirmed_wide_range_rows": int(retained_unconfirmed_wide_range.sum()),
        "maximum_contract_quarantined_fraction": (
            float(contract_fractions.max()) if len(contract_fractions) else 0.0
        ),
        "maximum_quarantine_fraction": maximum_quarantine_fraction,
        "budget_exceeded": budget_exceeded,
        "confirmation_bars": confirmation_bars,
        "minimum_confirmation_bars": minimum_confirmation_bars,
        "confirmation_fraction": confirmation_fraction,
        "forward_looking": True,
        "walk_forward_rule": (
            "A quarantine decision may affect a cutoff only when quarantine_available_at "
            "is strictly before that cutoff."
        ),
        "prices_repaired": False,
        "gaps_preserved": True,
    }
    return CandleQuarantineResult(clean=clean, audit=audit.reset_index(drop=True), summary=summary)


def fetch_trade_range(
    client: FinDataClient,
    condition_id: str,
    start: datetime,
    end: datetime,
    *,
    limit: int = 5000,
) -> pd.DataFrame:
    """Fetch a bounded trade range, bisecting responses that hit the API cap."""

    response = client.polymarket_trades(
        condition_id,
        start=start,
        end=end,
        limit=limit,
    )
    frame = trades_to_frame(response.body, condition_id=condition_id)
    if len(frame) < limit:
        return frame
    if end - start <= timedelta(minutes=1):
        raise RuntimeError(
            f"trade response still hits limit={limit} at one-minute resolution for {condition_id}"
        )
    midpoint = start + (end - start) / 2
    midpoint = midpoint.replace(microsecond=0)
    if midpoint <= start or midpoint >= end:
        raise RuntimeError(f"cannot bisect saturated trade interval for {condition_id}")
    left = fetch_trade_range(client, condition_id, start, midpoint, limit=limit)
    right = fetch_trade_range(client, condition_id, midpoint, end, limit=limit)
    return pd.concat([left, right], ignore_index=True)


def fetch_complete_trades(
    client: FinDataClient,
    ranges: Mapping[str, tuple[datetime, datetime]],
    *,
    workers: int,
) -> pd.DataFrame:
    jobs = [
        (condition_id, chunk_start, chunk_end)
        for condition_id, (market_start, market_end) in ranges.items()
        for chunk_start, chunk_end in iter_time_chunks(market_start, market_end, timedelta(days=30))
    ]
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_trade_range, client, condition_id, chunk_start, chunk_end): (
                condition_id,
                chunk_start,
                chunk_end,
            )
            for condition_id, chunk_start, chunk_end in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            condition_id, chunk_start, chunk_end = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"failed trade fetch for {condition_id} "
                    f"[{format_rfc3339(chunk_start)}, {format_rfc3339(chunk_end)})"
                ) from exc
            if not frame.empty:
                frames.append(frame)
            if completed % 25 == 0 or completed == len(jobs):
                print(f"  trade chunks: {completed}/{len(jobs)}", flush=True)
    if not frames:
        return trades_to_frame([], condition_id="")
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["condition_id", "date", "trade_id"])
    result = result.drop_duplicates(["condition_id", "trade_id"], keep="last").reset_index(drop=True)
    in_requested_range = pd.Series(False, index=result.index)
    for condition_id, (market_start, market_end) in ranges.items():
        in_requested_range |= (
            result["condition_id"].eq(condition_id)
            & result["date"].ge(pd.Timestamp(market_start))
            & result["date"].lt(pd.Timestamp(market_end))
        )
    result = result.loc[in_requested_range].reset_index(drop=True)
    numeric = result[["price", "size"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not ((numeric[:, 0] >= 0) & (numeric[:, 0] <= 1)).all():
        raise ValueError("trade prices or sizes are invalid")
    if (result["size"] < 0).any():
        raise ValueError("trade sizes must be non-negative")
    return result


def aggregate_token_trades(
    trades: pd.DataFrame,
    *,
    interval_minutes: int,
) -> pd.DataFrame:
    if interval_minutes not in {15, 60, 240}:
        raise ValueError("trade aggregation interval must be 15, 60, or 240 minutes")
    columns = [
        "condition_id",
        "token_id",
        "interval_minutes",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    frame = trades.sort_values(["condition_id", "date", "trade_id"]).copy()
    frame["bucket"] = frame["date"].dt.floor(f"{interval_minutes}min")
    grouped = frame.groupby(["condition_id", "token_id", "bucket"], sort=True).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trades=("trade_id", "nunique"),
    )
    result = grouped.reset_index().rename(columns={"bucket": "date"})
    result.insert(2, "interval_minutes", interval_minutes)
    return result.loc[:, columns]
