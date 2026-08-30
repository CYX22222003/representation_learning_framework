from __future__ import annotations

import unittest

import numpy as np

from baselines.garch_lstm_stacking.meta import StackingMetaModel, build_meta_features, clipped_predictions, clipping_diagnostics


class MetaTests(unittest.TestCase):
    def test_feature_order_and_interaction(self) -> None:
        z = build_meta_features(np.array([2.0, 3.0]), np.array([5.0, 7.0]))
        np.testing.assert_allclose(z, np.array([[2.0, 5.0, 10.0], [3.0, 7.0, 21.0]]))

    def test_rejects_nonfinite_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_meta_features(np.array([1.0, np.nan]), np.array([2.0, 3.0]))

    def test_fit_replay_and_test_features_do_not_change_scaler(self) -> None:
        g = np.linspace(0.01, 0.05, 20)
        l = np.linspace(0.02, 0.06, 20)
        y = 0.3 * g + 0.7 * l + 0.1 * g * l
        model = StackingMetaModel.fit(g, l, y)
        payload = {**model.scaler_dict(), **model.model_dict()}
        replay = StackingMetaModel.from_dict(payload)
        np.testing.assert_allclose(model.predict_raw(g, l), replay.predict_raw(g, l), rtol=1e-12, atol=1e-12)
        center_before = model.center.copy()
        _ = model.predict_raw(np.array([999.0]), np.array([888.0]))
        np.testing.assert_array_equal(model.center, center_before)

    def test_clipping_diagnostics(self) -> None:
        raw = np.array([-1.0, 0.5])
        clipped = clipped_predictions(raw)
        np.testing.assert_allclose(clipped, np.array([0.0, 0.5]))
        diag = clipping_diagnostics(raw, clipped, np.array([0.2, 0.4]))
        self.assertEqual(diag["clipped_row_count"], 1)
        self.assertEqual(diag["raw_negative_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
