from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from baselines.ginn_baseline.ginn_data import (
    ContractPreparationError,
    GinnDataConfig,
    apply_ar_channels,
    compute_split_indices,
    fit_ar_channels,
    fit_garch11,
    garch_volatility_target,
    load_and_validate_cache,
    prepare_contract,
    realized_volatility_target,
    transform_ohlcv,
    write_cache,
)
from tests.baselines.ginn_baseline.helpers import synthetic_contract


class DataPrimitiveTests(unittest.TestCase):
    def test_split_indices_match_framework_boundary(self):
        split = compute_split_indices(n_rows=100, seq_len=64, train_ratio=0.8, ar_order=5)
        self.assertEqual(split.n_samples, 36)
        self.assertEqual(split.n_train, 28)
        self.assertEqual(split.fit_end, 92)
        np.testing.assert_array_equal(split.train_starts, np.arange(5, 28))
        np.testing.assert_array_equal(split.test_starts, np.arange(28, 36))

    def test_transform_preserves_row_alignment(self):
        frame = synthetic_contract(80)
        transformed = transform_ohlcv(frame)
        self.assertEqual(transformed.shape, (80, 5))
        np.testing.assert_array_equal(transformed[0], np.zeros(5))
        expected_close = np.diff(np.log(frame["close"].to_numpy()))
        np.testing.assert_allclose(transformed[1:, 3], expected_close)
        expected_volume = np.diff(np.log1p(frame["volume"].to_numpy()))
        np.testing.assert_allclose(transformed[1:, 4], expected_volume)

    def test_ar_residuals_are_length_aligned(self):
        transformed = transform_ohlcv(synthetic_contract(100))
        coeffs = fit_ar_channels(transformed, fit_end=92, ar_order=5)
        residuals = apply_ar_channels(transformed, coeffs)
        self.assertEqual(coeffs.shape, (5, 5))
        self.assertEqual(residuals.shape, (100, 5))
        np.testing.assert_array_equal(residuals[:5], np.zeros((5, 5)))

    def test_targets_are_volatility_not_variance(self):
        close = np.exp(np.arange(70, dtype=np.float64) * 0.1)
        sigma2 = np.full(70, 0.04, dtype=np.float64)
        self.assertAlmostEqual(realized_volatility_target(close, start=0, seq_len=64), 0.1)
        self.assertAlmostEqual(garch_volatility_target(sigma2, start=0, seq_len=64), 0.2)


class ContractPreparationTests(unittest.TestCase):
    def test_test_period_changes_do_not_change_fits_or_training_arrays(self):
        config = GinnDataConfig(seq_len=64, ar_order=5)
        base = synthetic_contract(140)
        split = compute_split_indices(140, 64, 0.8, 5)
        changed = base.copy()
        changed.loc[split.fit_end:, ["open", "high", "low", "close"]] *= 1.2
        changed.loc[split.fit_end:, "volume"] *= 2.0

        first = prepare_contract(base, contract_id=3, filename="base.feather", config=config)
        second = prepare_contract(changed, contract_id=3, filename="base.feather", config=config)

        np.testing.assert_allclose(first.ar_coefficients, second.ar_coefficients)
        np.testing.assert_allclose(first.garch.params, second.garch.params)
        np.testing.assert_allclose(first.X_train, second.X_train)
        np.testing.assert_allclose(first.y_gt_train, second.y_gt_train)
        np.testing.assert_allclose(first.y_garch_train, second.y_garch_train)

    def test_samples_use_declared_starts_and_contract_id(self):
        prepared = prepare_contract(
            synthetic_contract(140), 7, "contract.feather", GinnDataConfig()
        )
        self.assertEqual(prepared.X_train.shape[1:], (64, 5))
        self.assertEqual(prepared.y_gt_train.shape[1:], (1,))
        self.assertTrue(np.all(prepared.contract_id_train == 7))
        self.assertTrue(np.all(prepared.contract_id_test == 7))
        self.assertEqual(prepared.train_starts[0], 5)
        self.assertEqual(prepared.test_starts[0], int(np.floor(0.8 * (140 - 64))))

    def test_garch_failure_uses_recorded_deterministic_fallback(self):
        residuals = np.linspace(-0.1, 0.1, 100)
        with patch("baselines.ginn_baseline.ginn_data.minimize", side_effect=RuntimeError("boom")):
            fit = fit_garch11(residuals, fit_end=80)
        self.assertTrue(fit.used_fallback)
        self.assertFalse(fit.converged)
        self.assertEqual(fit.status, "fallback: optimizer exception")
        self.assertTrue(np.isfinite(fit.params).all())

    def test_missing_column_is_attributed_to_contract(self):
        frame = synthetic_contract(140).drop(columns=["volume"])
        with self.assertRaisesRegex(ContractPreparationError, "missing.feather.*missing_columns"):
            prepare_contract(frame, 0, "missing.feather", GinnDataConfig())

    def test_short_contract_has_explicit_reason(self):
        with self.assertRaisesRegex(ContractPreparationError, "short.feather.*insufficient_length"):
            prepare_contract(synthetic_contract(64), 0, "short.feather", GinnDataConfig())


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_records_shapes_and_rejects_manifest_mismatch(self):
        prepared = [
            prepare_contract(synthetic_contract(140, phase=0.0), 0, "a.feather", GinnDataConfig()),
            prepare_contract(synthetic_contract(145, phase=1.0), 1, "b.feather", GinnDataConfig()),
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            manifest = write_cache(
                prepared,
                skipped_contracts=[],
                source_files=[
                    {"filename": "a.feather", "size": 10, "sha256": "a"},
                    {"filename": "b.feather", "size": 20, "sha256": "b"},
                ],
                config=GinnDataConfig(),
                cache_path=cache_path,
                manifest_path=manifest_path,
            )
            loaded = load_and_validate_cache(cache_path, manifest_path, GinnDataConfig())
            self.assertEqual(loaded.manifest["dataset_id"], manifest["dataset_id"])
            self.assertEqual(loaded.X_train.dtype, np.float32)
            self.assertEqual(loaded.contract_id_train.dtype, np.int32)

            bad = dict(manifest)
            bad["config"] = dict(manifest["config"], seq_len=32)
            manifest_path.write_text(__import__("json").dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest config mismatch"):
                load_and_validate_cache(cache_path, manifest_path, GinnDataConfig())


if __name__ == "__main__":
    unittest.main()
