from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import (
    Phase5WalkSpec,
    build_phase5_walk_bundle,
    sha256_file,
    validate_phase5_arrays,
    validate_phase5_bundle_files,
)


def candles(rows: int = 216) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = 0.25 + (np.arange(rows, dtype=np.float64) % 40) * 0.001
    result = pd.DataFrame(
        {
            "condition_id": ["condition-a"] * rows,
            "interval_minutes": [60] * rows,
            "date": dates,
            "open": close,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": np.arange(rows, dtype=np.float64) + 1.0,
            "native_resolution": ["1h"] * rows,
            "is_observed": [True] * rows,
            "is_imputed": [False] * rows,
            "original_gap_length_bars": np.zeros(rows, dtype=np.int16),
            "time_since_last_observation": np.zeros(rows, dtype=np.int64),
        }
    )
    # One approved context-only isolated-gap fill.
    prior_close = float(result.loc[69, "close"])
    result.loc[70, ["open", "high", "low", "close"]] = prior_close
    result.loc[70, "volume"] = 0.0
    result.loc[70, "is_observed"] = False
    result.loc[70, "is_imputed"] = True
    result.loc[70, "original_gap_length_bars"] = 1
    result.loc[70, "time_since_last_observation"] = 60
    return result


def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": ["condition-a"],
            "category": ["test"],
            "event_family": ["fixture"],
            "selection_rank": [1],
            "start_date_search": [pd.Timestamp("2025-12-01", tz="UTC")],
            "end_date_search": [pd.Timestamp("2026-03-01", tz="UTC")],
        }
    )


def spec() -> Phase5WalkSpec:
    return Phase5WalkSpec.from_values(1, "2026-01-01", "2026-01-06", "2026-01-10")


def mark_imputed(source: pd.DataFrame, index: int, *, value: float | None = None) -> None:
    preceding_close = float(source.loc[index - 1, "close"]) if index > 0 else float(source.loc[index, "close"])
    fill_value = preceding_close if value is None else float(value)
    source.loc[index, ["open", "high", "low", "close"]] = fill_value
    source.loc[index, "volume"] = 0.0
    source.loc[index, "is_observed"] = False
    source.loc[index, "is_imputed"] = True
    source.loc[index, "original_gap_length_bars"] = 1
    source.loc[index, "time_since_last_observation"] = 60


class Phase5WalkPreparationTests(unittest.TestCase):
    def test_builds_shared_horizon_rows_with_strict_calendar_boundaries(self) -> None:
        bundle = build_phase5_walk_bundle(candles(), metadata(), spec())
        result = validate_phase5_arrays(bundle.arrays, bundle.manifest)
        self.assertTrue(result["valid"])
        self.assertGreater(len(bundle.arrays["train_sequences"]), 0)
        self.assertGreater(len(bundle.arrays["test_sequences"]), 0)
        self.assertGreaterEqual(
            len(bundle.arrays["encoder_train_sequences"]),
            len(bundle.arrays["train_sequences"]),
        )
        cutoff = spec().cutoff.value
        self.assertTrue((bundle.arrays["train_target_availability_ns"] < cutoff).all())
        self.assertTrue((bundle.arrays["test_decision_availability_ns"] >= cutoff).all())
        self.assertTrue(
            set(bundle.arrays["test_condition_ids"]).issubset(
                set(bundle.arrays["train_condition_ids"])
            )
        )
        np.testing.assert_array_equal(
            bundle.arrays["train_target_date_ns"],
            bundle.arrays["train_decision_date_ns"] + 2 * 60 * 60 * 1_000_000_000,
        )
        replay_class = np.where(
            bundle.arrays["train_regression_labels"] < -0.001,
            0,
            np.where(np.abs(bundle.arrays["train_regression_labels"]) <= 0.001, 1, 2),
        )
        np.testing.assert_array_equal(replay_class, bundle.arrays["train_classification_labels"])

    def test_builds_eight_hour_sensitivity_without_changing_encoder_population(self) -> None:
        primary = build_phase5_walk_bundle(candles(), metadata(), spec())
        sensitivity = build_phase5_walk_bundle(
            candles(),
            metadata(),
            spec(),
            horizon=8,
            task_role="exploratory_raw_delta_h8",
        )
        self.assertTrue(validate_phase5_arrays(sensitivity.arrays, sensitivity.manifest)["valid"])
        self.assertEqual(sensitivity.manifest["horizon_hours"], 8)
        self.assertEqual(sensitivity.manifest["task_role"], "exploratory_raw_delta_h8")
        np.testing.assert_array_equal(
            primary.arrays["encoder_train_sequences"],
            sensitivity.arrays["encoder_train_sequences"],
        )
        np.testing.assert_array_equal(
            sensitivity.arrays["train_target_date_ns"],
            sensitivity.arrays["train_decision_date_ns"] + 8 * 60 * 60 * 1_000_000_000,
        )

    def test_evaluation_changes_do_not_change_training_scaler_or_rows(self) -> None:
        original = candles()
        changed = original.copy()
        evaluation = changed["date"].add(pd.Timedelta(hours=1)).ge(spec().cutoff)
        changed.loc[evaluation, "volume"] = 1_000_000_000.0
        changed.loc[evaluation, ["open", "high", "low", "close"]] = 0.75
        first = build_phase5_walk_bundle(original, metadata(), spec())
        second = build_phase5_walk_bundle(changed, metadata(), spec())
        np.testing.assert_array_equal(first.arrays["encoder_train_sequences"], second.arrays["encoder_train_sequences"])
        np.testing.assert_array_equal(first.arrays["train_sequences"], second.arrays["train_sequences"])
        self.assertEqual(first.manifest["preprocessing"], second.manifest["preprocessing"])
        self.assertEqual(first.manifest["identity_hashes"]["train"], second.manifest["identity_hashes"]["train"])

    def test_evaluation_target_status_does_not_change_encoder_population(self) -> None:
        original = candles()
        changed = original.copy()
        target_date = spec().cutoff - pd.Timedelta(hours=1)
        target_index = int(changed.index[changed["date"].eq(target_date)][0])
        mark_imputed(changed, target_index)

        first = build_phase5_walk_bundle(original, metadata(), spec())
        long_gap = original.loc[
            ~original["date"].isin(
                [target_date, target_date + pd.Timedelta(hours=1)]
            )
        ].reset_index(drop=True)
        for perturbed in (changed, long_gap):
            second = build_phase5_walk_bundle(perturbed, metadata(), spec())
            for name in (
                "encoder_train_condition_ids",
                "encoder_train_window_start_ns",
                "encoder_train_decision_date_ns",
                "encoder_train_decision_availability_ns",
                "encoder_train_raw_sequences",
                "encoder_train_sequences",
            ):
                np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
            self.assertEqual(
                first.manifest["identity_hashes"]["encoder_train"],
                second.manifest["identity_hashes"]["encoder_train"],
            )

    def test_long_gap_is_a_sequence_break(self) -> None:
        source = candles().drop(index=[90, 91]).reset_index(drop=True)
        bundle = build_phase5_walk_bundle(source, metadata(), spec())
        for split in ("encoder_train", "train", "test"):
            np.testing.assert_array_equal(
                bundle.arrays[f"{split}_decision_date_ns"]
                - bundle.arrays[f"{split}_window_start_ns"],
                np.full(len(bundle.arrays[f"{split}_sequences"]), 63 * 60 * 60 * 1_000_000_000),
            )

    def test_imputation_metadata_and_scaled_sentinel_replay(self) -> None:
        bundle = build_phase5_walk_bundle(candles(), metadata(), spec())
        mask = bundle.arrays["train_context_is_imputed"]
        sentinel = bundle.manifest["preprocessing"]["volume"]["imputed_scaled_value"]
        self.assertTrue(mask.any())
        np.testing.assert_allclose(bundle.arrays["train_sequences"][:, :, 4][mask], sentinel)
        self.assertFalse(mask[:, -1].any())

    def test_invalid_raw_volume_mask_invariant_fails_closed(self) -> None:
        source = candles()
        source.loc[70, "volume"] = 3.0
        with self.assertRaisesRegex(ValueError, "is_imputed"):
            build_phase5_walk_bundle(source, metadata(), spec())

    def test_imputed_rows_require_observed_hourly_neighbors_and_preceding_close(self) -> None:
        wrong_close = candles()
        mark_imputed(wrong_close, 70, value=0.123)
        with self.assertRaisesRegex(ValueError, "preceding observed close"):
            build_phase5_walk_bundle(wrong_close, metadata(), spec())

        boundary = candles()
        mark_imputed(boundary, 0)
        with self.assertRaisesRegex(ValueError, "contract boundary"):
            build_phase5_walk_bundle(boundary, metadata(), spec())

        adjacent = candles()
        mark_imputed(adjacent, 71)
        with self.assertRaisesRegex(ValueError, "not isolated"):
            build_phase5_walk_bundle(adjacent, metadata(), spec())

    def test_saved_bundle_replays_source_and_artifact_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(b"source")
            source = {"fixture": {"path": str(source_path), "sha256": sha256_file(source_path)}}
            bundle = build_phase5_walk_bundle(candles(), metadata(), spec(), source_provenance=source)
            npz_path = root / "bundle.npz"
            np.savez_compressed(npz_path, **bundle.arrays)
            manifest = {
                **bundle.manifest,
                "artifact": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
            }
            manifest_path = Path(f"{npz_path}.manifest.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(validate_phase5_bundle_files(npz_path)["valid"])
            source_path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                validate_phase5_bundle_files(npz_path)


if __name__ == "__main__":
    unittest.main()
