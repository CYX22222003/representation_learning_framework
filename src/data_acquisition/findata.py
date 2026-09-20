"""Small authenticated client for the lab FinData prediction-market API.

Only read-only endpoints are implemented.  Authentication is loaded at run
time and is never included in returned data, manifests, or exception text.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_BASE_URL = "https://lum.id/findata"
_CANDLE_COLUMNS = ("bucket_ts", "open", "high", "low", "close", "volume", "trades")
_TRADE_COLUMNS = ("trade_id", "token_id", "side", "price", "size", "taker", "ts")


def load_lumid_token(env_path: Path | None = None) -> str:
    """Load the FinData token without adding a dotenv dependency."""

    value = os.environ.get("LUMID_API_KEYS", "").strip()
    if value:
        return value.strip("\"'")

    if env_path is None:
        raise RuntimeError("LUMID_API_KEYS is not set and no .env path was provided")
    if not env_path.exists():
        raise FileNotFoundError(f"FinData credential file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == "LUMID_API_KEYS":
            token = raw_value.strip().strip("\"'")
            if token:
                return token
    raise RuntimeError(f"LUMID_API_KEYS is missing or empty in {env_path}")


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True)
class FinDataResponse:
    body: Any
    url: str
    etag: str | None
    rate_limit_remaining: str | None
    rate_limit_reset: str | None


class FinDataClient:
    """Read-only FinData HTTP client with bounded retry/backoff behavior."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_attempts: int = 5,
    ) -> None:
        if not token:
            raise ValueError("a non-empty FinData token is required")
        self._token = token
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def get(self, path: str, params: Mapping[str, object] | None = None) -> FinDataResponse:
        query = urlencode(
            [(key, str(value)) for key, value in (params or {}).items() if value is not None]
        )
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/json",
                    "User-Agent": "representation-learning-framework/findata-collector",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read()
                    return FinDataResponse(
                        body=json.loads(payload.decode("utf-8")),
                        url=url,
                        etag=response.headers.get("ETag"),
                        rate_limit_remaining=response.headers.get("x-ratelimit-remaining"),
                        rate_limit_reset=response.headers.get("x-ratelimit-reset"),
                    )
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_attempts:
                    detail = exc.read(512).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"FinData GET {path} failed with HTTP {exc.code}: {detail}"
                    ) from exc
                delay = _retry_after_seconds(exc.headers.get("Retry-After"))
                if delay is None:
                    delay = min(30.0, 2.0 ** (attempt - 1))
                time.sleep(delay + random.uniform(0.0, 0.25))
                last_error = exc
            except (TimeoutError, URLError) as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(f"FinData GET {path} failed after retries") from exc
                time.sleep(min(30.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.25))
                last_error = exc
        raise RuntimeError(f"FinData GET {path} failed") from last_error

    def polymarket_universe(self, *, include_inactive: bool = True) -> FinDataResponse:
        return self.get(
            "/api/v1/lqt/universe",
            {"venue": "polymarket", "include_inactive": int(include_inactive)},
        )

    def polymarket_detail(self, condition_id: str) -> FinDataResponse:
        return self.get(f"/prediction-markets/markets/polymarket/{condition_id}")

    def polymarket_search(
        self,
        *,
        status: str,
        limit: int = 500,
        offset: int = 0,
        query: str = "",
    ) -> FinDataResponse:
        """Search the historical Polymarket catalog.

        The live API currently requires the ``q`` field even though its
        OpenAPI schema does not mark it as required, so an explicit empty
        query is sent for catalog-wide pagination.
        """

        if status not in {"open", "closed", "all"}:
            raise ValueError("status must be open, closed, or all")
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset non-negative")
        return self.get(
            "/prediction-markets/markets/search",
            {
                "q": query,
                "venue": "polymarket",
                "status": status,
                "limit": limit,
                "offset": offset,
            },
        )

    def polymarket_candles(
        self,
        condition_id: str,
        *,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        limit: int = 5000,
    ) -> FinDataResponse:
        if interval_minutes not in {1, 5, 15, 30, 60, 1440}:
            raise ValueError("FinData native candle interval must be 1, 5, 15, 30, 60, or 1440")
        if not 1 <= limit <= 5000:
            raise ValueError("FinData candle limit must be between 1 and 5000")
        return self.get(
            f"/prediction-markets/candles/polymarket/{condition_id}",
            {
                "interval": interval_minutes,
                "from": format_rfc3339(start),
                "to": format_rfc3339(end),
                "limit": limit,
            },
        )

    def polymarket_trades(
        self,
        condition_id: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 5000,
    ) -> FinDataResponse:
        if not 1 <= limit <= 5000:
            raise ValueError("FinData trade limit must be between 1 and 5000")
        return self.get(
            f"/prediction-markets/trades/polymarket/{condition_id}",
            {
                "from": format_rfc3339(start),
                "to": format_rfc3339(end),
                "limit": limit,
            },
        )


def format_rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_time_chunks(start: datetime, end: datetime, chunk: timedelta) -> Iterable[tuple[datetime, datetime]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("chunk boundaries must be timezone-aware")
    if not start < end:
        raise ValueError("start must be earlier than end")
    if chunk <= timedelta(0):
        raise ValueError("chunk duration must be positive")
    cursor = start
    while cursor < end:
        next_cursor = min(end, cursor + chunk)
        yield cursor, next_cursor
        cursor = next_cursor


def build_market_candidates(universe_body: Mapping[str, Any]) -> pd.DataFrame:
    """Deduplicate the LQT universe's two outcome-token rows per condition."""

    markets = universe_body.get("markets")
    if not isinstance(markets, list):
        raise ValueError("FinData universe response is missing a markets list")
    frame = pd.DataFrame(markets)
    required = {"market_id", "question", "liquidity_num", "volume_24h", "active"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FinData universe rows are missing fields: {missing}")
    frame = frame.loc[frame["market_id"].notna()].copy()
    frame["liquidity_num"] = pd.to_numeric(frame["liquidity_num"], errors="coerce").fillna(0.0)
    frame["volume_24h"] = pd.to_numeric(frame["volume_24h"], errors="coerce").fillna(0.0)
    frame["active"] = frame["active"].fillna(False).astype(bool)
    frame = frame.sort_values(
        ["market_id", "liquidity_num", "volume_24h", "instrument_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    frame = frame.drop_duplicates("market_id", keep="first")
    frame = frame.sort_values(
        ["liquidity_num", "volume_24h", "market_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "snapshot_rank", np.arange(1, len(frame) + 1, dtype=np.int64))
    return frame


def candles_to_frame(body: Any, *, condition_id: str, interval_minutes: int) -> pd.DataFrame:
    if not isinstance(body, list):
        raise ValueError("FinData candle response must be a JSON array")
    if not body:
        return pd.DataFrame(
            columns=["condition_id", "interval_minutes", "date", "open", "high", "low", "close", "volume", "trades"]
        )
    frame = pd.DataFrame(body)
    missing = sorted(set(_CANDLE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"FinData candle rows are missing fields: {missing}")
    frame = frame.loc[:, list(_CANDLE_COLUMNS)].copy()
    frame.rename(columns={"bucket_ts": "date"}, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume", "trades"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.insert(0, "interval_minutes", int(interval_minutes))
    frame.insert(0, "condition_id", condition_id)
    return frame


def trades_to_frame(body: Any, *, condition_id: str) -> pd.DataFrame:
    if not isinstance(body, list):
        raise ValueError("FinData trade response must be a JSON array")
    columns = ["condition_id", "trade_id", "token_id", "side", "price", "size", "taker", "date"]
    if not body:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(body)
    missing = sorted(set(_TRADE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"FinData trade rows are missing fields: {missing}")
    frame = frame.loc[:, list(_TRADE_COLUMNS)].copy()
    frame.rename(columns={"ts": "date"}, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame.insert(0, "condition_id", condition_id)
    return frame.loc[:, columns]


def validate_candles(frame: pd.DataFrame, *, start: datetime, end: datetime) -> None:
    if frame.empty:
        return
    if frame["date"].duplicated().any():
        raise ValueError("duplicate candle timestamps remain after collection")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("candle timestamps are not increasing")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("OHLCV contains non-finite values")
    if (frame["volume"] < 0).any():
        raise ValueError("candle volume contains negative values")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("candle low exceeds an OHLC component")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("candle high is below an OHLC component")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if (frame["date"] < start_ts).any() or (frame["date"] >= end_ts).any():
        raise ValueError("candle timestamp falls outside the requested half-open interval")


def aggregate_one_hour_to_four_hour(one_hour: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed 1h rows into UTC-aligned 4h OHLCV bins.

    Missing source hours are not imputed here.  ``observed_1h_bars`` and
    ``complete_1h_coverage`` make partial four-hour bins explicit so later
    preprocessing can choose a causal missing-data policy.
    """

    output_columns = [
        "condition_id",
        "interval_minutes",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
        "observed_1h_bars",
        "complete_1h_coverage",
        "source_interval_minutes",
        "derived",
    ]
    if one_hour.empty:
        return pd.DataFrame(columns=output_columns)

    parts: list[pd.DataFrame] = []
    for condition_id, market in one_hour.groupby("condition_id", sort=True):
        market = market.sort_values("date").copy()
        market["four_hour_start"] = market["date"].dt.floor("4h")
        grouped = market.groupby("four_hour_start", sort=True)
        result = grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trades=("trades", lambda values: values.sum(min_count=1)),
            observed_1h_bars=("date", "nunique"),
        ).reset_index(names="date")
        result.insert(0, "interval_minutes", 240)
        result.insert(0, "condition_id", condition_id)
        result["complete_1h_coverage"] = result["observed_1h_bars"].eq(4)
        result["source_interval_minutes"] = 60
        result["derived"] = True
        parts.append(result)
    combined = pd.concat(parts, ignore_index=True)
    return combined.loc[:, output_columns].sort_values(["condition_id", "date"]).reset_index(drop=True)
