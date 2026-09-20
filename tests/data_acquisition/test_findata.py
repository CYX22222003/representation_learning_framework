from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from src.data_acquisition.findata import (
    FinDataClient,
    aggregate_one_hour_to_four_hour,
    build_market_candidates,
    iter_time_chunks,
    trades_to_frame,
)
from src.data_acquisition.polymarket_pipeline import (
    aggregate_token_trades,
    classify_market,
    diverse_candidate_order,
    event_family,
    fetch_complete_candles,
    quarantine_condition_candles,
)


class FindataUtilitiesTest(unittest.TestCase):
    def test_candle_requests_send_explicit_safe_limit(self) -> None:
        client = FinDataClient("secret")
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        with patch.object(client, "get", return_value=object()) as get:
            response = client.polymarket_candles(
                "condition",
                interval_minutes=15,
                start=start,
                end=end,
            )
        self.assertIsNotNone(response)
        params = get.call_args.args[1]
        self.assertEqual(params["limit"], 5000)
        with self.assertRaisesRegex(ValueError, "between 1 and 5000"):
            client.polymarket_candles(
                "condition",
                interval_minutes=15,
                start=start,
                end=end,
                limit=5001,
            )

    def test_universe_deduplicates_condition_rows_and_ranks_deterministically(self) -> None:
        body = {
            "markets": [
                {
                    "market_id": "b",
                    "instrument_id": "2",
                    "question": "B",
                    "liquidity_num": 10,
                    "volume_24h": 5,
                    "active": True,
                },
                {
                    "market_id": "a",
                    "instrument_id": "3",
                    "question": "A",
                    "liquidity_num": 20,
                    "volume_24h": 1,
                    "active": True,
                },
                {
                    "market_id": "a",
                    "instrument_id": "1",
                    "question": "A",
                    "liquidity_num": 20,
                    "volume_24h": 1,
                    "active": True,
                },
            ]
        }
        result = build_market_candidates(body)
        self.assertEqual(result["market_id"].tolist(), ["a", "b"])
        self.assertEqual(result["snapshot_rank"].tolist(), [1, 2])
        self.assertEqual(result.loc[0, "instrument_id"], "1")

    def test_chunks_cover_half_open_interval_without_gaps(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 4, tzinfo=timezone.utc)
        chunks = list(iter_time_chunks(start, end, timedelta(days=2)))
        self.assertEqual(chunks, [(start, start + timedelta(days=2)), (start + timedelta(days=2), end)])

    def test_four_hour_aggregation_marks_partial_coverage(self) -> None:
        frame = pd.DataFrame(
            {
                "condition_id": ["m"] * 5,
                "interval_minutes": [60] * 5,
                "date": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T01:00:00Z",
                        "2026-01-01T02:00:00Z",
                        "2026-01-01T03:00:00Z",
                        "2026-01-01T04:00:00Z",
                    ],
                    utc=True,
                ),
                "open": [0.1, 0.2, 0.15, 0.3, 0.4],
                "high": [0.2, 0.25, 0.2, 0.35, 0.45],
                "low": [0.05, 0.1, 0.1, 0.2, 0.35],
                "close": [0.15, 0.2, 0.18, 0.3, 0.42],
                "volume": [1, 2, 3, 4, 5],
                "trades": [1, 1, 1, 1, 1],
            }
        )
        result = aggregate_one_hour_to_four_hour(frame)
        self.assertEqual(result["observed_1h_bars"].tolist(), [4, 1])
        self.assertEqual(result["complete_1h_coverage"].tolist(), [True, False])
        self.assertAlmostEqual(result.loc[0, "open"], 0.1)
        self.assertAlmostEqual(result.loc[0, "close"], 0.3)
        self.assertAlmostEqual(result.loc[0, "volume"], 10.0)

    def test_historical_categories_and_families_support_diverse_selection(self) -> None:
        self.assertEqual(classify_market("Will Bitcoin exceed $100k?"), "crypto")
        self.assertEqual(classify_market("Who will be Prime Minister of Ethiopia?"), "politics")
        self.assertEqual(classify_market("Will Spain win the 2026 FIFA World Cup?"), "sports")
        self.assertEqual(
            event_family("Will Argentina win the 2026 FIFA World Cup?"),
            "2026_fifa_world_cup",
        )
        candidates = pd.DataFrame(
            [
                {"condition_id": "s1", "category": "sports", "event_family": "cup", "volume": 10},
                {"condition_id": "s2", "category": "sports", "event_family": "cup", "volume": 9},
                {"condition_id": "p1", "category": "politics", "event_family": "vote", "volume": 8},
            ]
        )
        result = diverse_candidate_order(candidates, candidate_limit=3, max_per_family=1)
        self.assertEqual(set(result["condition_id"]), {"s1", "p1"})

    def test_historical_fetch_prunes_endpoint_spillover(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)
        returned = pd.DataFrame(
            {
                "condition_id": ["m"] * 4,
                "interval_minutes": [60] * 4,
                "date": pd.to_datetime(
                    [start - timedelta(hours=1), start, start + timedelta(hours=1), end],
                    utc=True,
                ),
                "open": [0.1] * 4,
                "high": [0.1] * 4,
                "low": [0.1] * 4,
                "close": [0.1] * 4,
                "volume": [1.0] * 4,
                "trades": [1] * 4,
            }
        )
        with patch(
            "src.data_acquisition.polymarket_pipeline.fetch_interval",
            return_value=returned,
        ):
            result = fetch_complete_candles(
                object(),
                {"m": (start, end)},
                interval_minutes=60,
                global_start=start,
                global_end=end,
                workers=1,
            )
        self.assertEqual(result["date"].tolist(), [pd.Timestamp(start), pd.Timestamp(start + timedelta(hours=1))])

    def test_condition_candle_quarantine_preserves_raw_and_leaves_gaps(self) -> None:
        contract_ids = ["mixed"] * 4 + ["repriced"] * 4 + ["crash"] * 4 + ["wide"] * 4
        dates = pd.concat(
            [pd.Series(pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"))] * 4,
            ignore_index=True,
        )
        frame = pd.DataFrame(
            {
                "condition_id": contract_ids,
                "interval_minutes": [60] * 16,
                "date": dates,
                "open": [
                    0.006, 0.993, 0.007, 0.006,
                    0.238, 0.756, 0.74, 0.72,
                    0.9, 0.9, 0.12, 0.08,
                    0.1, 0.1, 0.18, 0.15,
                ],
                "high": [
                    0.007, 0.994, 0.008, 0.007,
                    0.24, 0.76, 0.75, 0.73,
                    0.91, 0.9, 0.13, 0.09,
                    0.11, 0.9, 0.19, 0.16,
                ],
                "low": [
                    0.005, 0.993, 0.006, 0.005,
                    0.23, 0.75, 0.73, 0.71,
                    0.89, 0.1, 0.11, 0.07,
                    0.09, 0.1, 0.17, 0.14,
                ],
                "close": [
                    0.006, 0.994, 0.007, 0.006,
                    0.238, 0.756, 0.74, 0.72,
                    0.9, 0.1, 0.12, 0.08,
                    0.1, 0.2, 0.18, 0.15,
                ],
                "volume": [1.0] * 16,
                "trades": [1] * 16,
            }
        )
        original = frame.copy(deep=True)
        result = quarantine_condition_candles(
            frame,
            interval_minutes=60,
            maximum_quarantine_fraction=0.5,
        )

        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(len(result.audit), 2)
        self.assertEqual(
            set(result.audit["quarantine_reason"]),
            {"transient_complement", "transient_wide_range"},
        )
        mixed_clean = result.clean.loc[result.clean["condition_id"].eq("mixed")]
        self.assertEqual(
            mixed_clean["date"].tolist(),
            [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T02:00:00Z"),
                pd.Timestamp("2026-01-01T03:00:00Z"),
            ],
        )
        self.assertEqual(len(result.clean.loc[result.clean["condition_id"].eq("repriced")]), 4)
        self.assertEqual(len(result.clean.loc[result.clean["condition_id"].eq("crash")]), 4)
        mixed_audit = result.audit.loc[result.audit["condition_id"].eq("mixed")].iloc[0]
        self.assertEqual(mixed_audit["confirmation_rows"], 2)
        self.assertEqual(
            mixed_audit["quarantine_available_at"],
            pd.Timestamp("2026-01-01T04:00:00Z"),
        )
        self.assertFalse(result.summary["prices_repaired"])
        self.assertTrue(result.summary["gaps_preserved"])
        self.assertTrue(result.summary["forward_looking"])
        self.assertGreaterEqual(result.summary["confirmed_persistent_large_jump_rows"], 2)
        self.assertGreaterEqual(result.summary["retained_persistent_wide_range_rows"], 1)

    def test_condition_candle_quarantine_fails_when_one_percent_budget_is_exceeded(self) -> None:
        frame = pd.DataFrame(
            {
                "condition_id": ["m"] * 4,
                "interval_minutes": [60] * 4,
                "date": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
                "open": [0.006, 0.993, 0.007, 0.006],
                "high": [0.007, 0.994, 0.008, 0.007],
                "low": [0.005, 0.993, 0.006, 0.005],
                "close": [0.006, 0.994, 0.007, 0.006],
                "volume": [1.0] * 4,
                "trades": [1] * 4,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "safety budget"):
            quarantine_condition_candles(frame, interval_minutes=60)

    def test_wide_candle_contaminated_by_prior_flag_is_also_quarantined(self) -> None:
        frame = pd.DataFrame(
            {
                "condition_id": ["m"] * 5,
                "interval_minutes": [60] * 5,
                "date": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
                "open": [0.1, 0.1, 0.9, 0.1, 0.1],
                "high": [0.1, 0.9, 0.9, 0.1, 0.1],
                "low": [0.1, 0.1, 0.1, 0.1, 0.1],
                "close": [0.1, 0.9, 0.1, 0.1, 0.1],
                "volume": [1.0] * 5,
                "trades": [1] * 5,
            }
        )
        result = quarantine_condition_candles(
            frame,
            interval_minutes=60,
            maximum_quarantine_fraction=0.5,
        )
        self.assertEqual(
            result.audit["date"].tolist(),
            [
                pd.Timestamp("2026-01-01T01:00:00Z"),
                pd.Timestamp("2026-01-01T02:00:00Z"),
            ],
        )
        self.assertTrue(result.audit["transient_wide_range"].all())

    def test_token_trades_aggregate_without_mixing_outcomes(self) -> None:
        rows = [
            {
                "trade_id": "yes-1",
                "token_id": "yes",
                "side": "BUY",
                "price": 0.2,
                "size": 10,
                "taker": "a",
                "ts": "2026-01-01T00:01:00Z",
            },
            {
                "trade_id": "yes-2",
                "token_id": "yes",
                "side": "SELL",
                "price": 0.3,
                "size": 20,
                "taker": "b",
                "ts": "2026-01-01T00:14:00Z",
            },
            {
                "trade_id": "no-1",
                "token_id": "no",
                "side": "BUY",
                "price": 0.8,
                "size": 5,
                "taker": "c",
                "ts": "2026-01-01T00:05:00Z",
            },
        ]
        trades = trades_to_frame(rows, condition_id="m")
        yes = trades.loc[trades["token_id"].eq("yes")]
        result = aggregate_token_trades(yes, interval_minutes=15)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "open"], 0.2)
        self.assertAlmostEqual(result.loc[0, "close"], 0.3)
        self.assertAlmostEqual(result.loc[0, "volume"], 30.0)
        self.assertEqual(result.loc[0, "trades"], 2)


if __name__ == "__main__":
    unittest.main()
