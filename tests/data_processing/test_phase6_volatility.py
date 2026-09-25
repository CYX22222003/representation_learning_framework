from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.phase5_walks import Phase5WalkSpec
from data_processing.phase6_volatility import (
    audit_volatility_walk,
    combine_walk_audits,
    future_realised_variance,
    validate_audit_artifacts,
    write_audit_tables,
)
from data_processing.phase6_volatility_labels import (
    build_volatility_label_bundle,
    validate_volatility_label_arrays,
)


def _candles(*, imputed_position: int | None = None) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    close = 0.20 + np.arange(100, dtype=np.float64) * 0.001
    observed = np.ones(100, dtype=bool)
    imputed = np.zeros(100, dtype=bool)
    volume = np.ones(100, dtype=np.float64)
    gap = np.zeros(100, dtype=np.int16)
    since = np.zeros(100, dtype=np.int16)
    if imputed_position is not None:
        close[imputed_position] = close[imputed_position - 1]
        observed[imputed_position] = False
        imputed[imputed_position] = True
        volume[imputed_position] = 0.0
        gap[imputed_position] = 1
        since[imputed_position] = 60
    return pd.DataFrame(
        {
            "condition_id": "contract-a",
            "interval_minutes": 60,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "native_resolution": "1h",
            "is_observed": observed,
            "is_imputed": imputed,
            "original_gap_length_bars": gap,
            "time_since_last_observation": since,
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": ["contract-a"],
            "category": ["test"],
            "event_family": ["test-family"],
            "selection_rank": [1],
            "start_date_search": [pd.Timestamp("2025-12-01", tz="UTC")],
            "end_date_search": [pd.Timestamp("2026-02-01", tz="UTC")],
        }
    )


def _midnight_end_metadata() -> pd.DataFrame:
    metadata = _metadata()
    metadata["end_date_search"] = pd.Timestamp("2026-01-03 00:00:00+00:00")
    return metadata


def _spec() -> Phase5WalkSpec:
    return Phase5WalkSpec.from_values(
        1,
        "2026-01-01 00:00:00+00:00",
        "2026-01-04 08:00:00+00:00",  # hour 80
        "2026-01-05 04:00:00+00:00",  # hour 100
    )


class FutureRealisedVarianceTests(unittest.TestCase):
    def test_formula_uses_raw_probability_changes(self) -> None:
        value, increments = future_realised_variance(np.asarray([0.10, 0.20, 0.15]))
        np.testing.assert_allclose(increments, [0.01, 0.0025])
        self.assertAlmostEqual(value, 0.0125)

    def test_cutoff_and_evaluation_end_maturity_are_separate(self) -> None:
        audit = audit_volatility_walk(
            _candles(),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
            candidate_horizons=(2,),
        )
        aggregate = audit.capacity.loc[audit.capacity["scope"].eq("aggregate")]
        train = aggregate.loc[aggregate["split"].eq("train")].iloc[0]
        evaluation = aggregate.loc[aggregate["split"].eq("evaluation")].iloc[0]
        self.assertEqual(int(train["candidate_rows"]), 16)
        self.assertEqual(int(train["eligible_rows"]), 14)
        self.assertEqual(int(train["drop_cutoff_maturity"]), 2)
        self.assertEqual(int(evaluation["candidate_rows"]), 20)
        self.assertEqual(int(evaluation["eligible_rows"]), 19)
        self.assertEqual(int(evaluation["drop_evaluation_end_maturity"]), 1)

    def test_imputed_future_candle_invalidates_target_but_context_may_contain_it(self) -> None:
        audit = audit_volatility_walk(
            _candles(imputed_position=70),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
            candidate_horizons=(2,),
        )
        train = audit.capacity.loc[
            audit.capacity["scope"].eq("aggregate") & audit.capacity["split"].eq("train")
        ].iloc[0]
        self.assertEqual(int(train["drop_imputed_target_candle"]), 2)
        self.assertGreater(float(audit.distribution.iloc[0]["context_imputation_exposure_rate"]), 0.0)

    def test_retrospective_end_metadata_is_reporting_only(self) -> None:
        audit = audit_volatility_walk(
            _candles(),
            _midnight_end_metadata(),
            _spec(),
            supported_contracts={"contract-a"},
            candidate_horizons=(2,),
        )
        train = audit.capacity.loc[
            audit.capacity["scope"].eq("aggregate") & audit.capacity["split"].eq("train")
        ].iloc[0]
        self.assertEqual(int(train["eligible_rows"]), 14)
        self.assertEqual(int(train["drop_contract_boundary"]), 0)

    def test_evaluation_target_values_are_absent_from_every_diagnostic_table(self) -> None:
        audit = audit_volatility_walk(
            _candles(),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
            candidate_horizons=(2,),
        )
        for frame in (audit.distribution, audit.dependence, audit.strata):
            self.assertEqual(set(frame["split"]), {"train"})
        evaluation = audit.capacity.loc[audit.capacity["split"].eq("evaluation")]
        self.assertFalse({"target", "realised_variance", "rv_value"} & set(evaluation.columns))
        self.assertFalse(audit.manifest["evaluation_outcome_policy"]["target_values_computed_or_emitted"])

    def test_written_artifacts_hash_and_validate(self) -> None:
        audit = audit_volatility_walk(
            _candles(),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
            candidate_horizons=(2,),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            write_audit_tables(
                output,
                combine_walk_audits([audit]),
                [audit.manifest],
                {"synthetic": True},
            )
            result = validate_audit_artifacts(output)
            self.assertEqual(result["candidate_horizons_hours"], [2])
            self.assertFalse(result["evaluation_outcomes_used"])
            self.assertTrue((output / "report.md").is_file())
            self.assertEqual(len(list((output / "plots").glob("*.png"))), 2)


class VolatilityLabelBundleTests(unittest.TestCase):
    def test_h8_bundle_stores_replayable_paths_and_shared_rows(self) -> None:
        bundle = build_volatility_label_bundle(
            _candles(),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
        )
        result = validate_volatility_label_arrays(bundle.arrays, bundle.manifest)
        self.assertTrue(result["valid"])
        self.assertEqual(result["row_counts"], {"train": 8, "test": 13})
        self.assertEqual(bundle.arrays["train_target_close_path"].shape[1], 9)
        self.assertEqual(bundle.arrays["train_target_squared_changes"].shape[1], 8)
        np.testing.assert_allclose(
            bundle.arrays["train_realised_variance"],
            bundle.arrays["train_target_squared_changes"].sum(axis=1),
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            bundle.arrays["test_shared_row_indices"],
            np.arange(len(bundle.arrays["test_realised_variance"])),
        )

    def test_h8_bundle_rejects_an_imputed_future_candle(self) -> None:
        bundle = build_volatility_label_bundle(
            _candles(imputed_position=75),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
        )
        decision_dates = bundle.arrays["train_decision_date_ns"]
        imputed_date = int(pd.Timestamp("2026-01-04 03:00:00+00:00").value)
        self.assertFalse(
            np.any(
                (decision_dates < imputed_date)
                & (bundle.arrays["train_target_end_ns"] >= imputed_date)
            )
        )

    def test_validator_detects_target_component_tampering(self) -> None:
        bundle = build_volatility_label_bundle(
            _candles(),
            _metadata(),
            _spec(),
            supported_contracts={"contract-a"},
        )
        arrays = {name: value.copy() for name, value in bundle.arrays.items()}
        arrays["train_target_squared_changes"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "component squared changes"):
            validate_volatility_label_arrays(arrays, bundle.manifest)


if __name__ == "__main__":
    unittest.main()
