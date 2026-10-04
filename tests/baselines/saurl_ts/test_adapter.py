from __future__ import annotations

import unittest

import torch

from baselines.saurl_ts import SaURLAdapter, SaURLInputScaler, build_saurl
from tests.baselines.saurl_ts.helpers import small_config


class SaURLAdapterTests(unittest.TestCase):
    def test_scaler_fits_all_rows_and_timestamps_with_constant_fallback(self) -> None:
        sequences = torch.randn(6, 8, 5)
        sequences[..., 4] = 3.0
        scaler = SaURLInputScaler.fit(sequences)
        normalized = scaler.transform(sequences)
        torch.testing.assert_close(normalized.mean(dim=(0, 1)), torch.zeros(5), atol=1e-6, rtol=0)
        self.assertEqual(float(scaler.scale[4]), 1.0)
        self.assertTrue(torch.isfinite(normalized).all())
        replay = SaURLInputScaler.from_state_dict(scaler.state_dict())
        torch.testing.assert_close(replay.transform(sequences), normalized)

    def test_adapter_returns_native_width_finite_embedding(self) -> None:
        config = small_config()
        raw = torch.randn(6, 8, 5)
        adapter = SaURLAdapter(build_saurl(config), SaURLInputScaler.fit(raw))
        embedding = adapter.encode(raw[:2])
        self.assertEqual(embedding.shape, (2, 16))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertFalse(embedding.requires_grad)

    def test_encode_excludes_training_only_modules(self) -> None:
        config = small_config()
        raw = torch.randn(6, 8, 5)
        model = build_saurl(config).eval()
        normalized = SaURLInputScaler.fit(raw).transform(raw[:2])
        before = model.encode(normalized).detach().clone()
        with torch.no_grad():
            for module in (
                model.temporal_sada,
                model.frequency_sada,
                model.online_projectors,
                model.online_predictors,
                model.target_encoders,
                model.target_projectors,
            ):
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter))
        after = model.encode(normalized)
        torch.testing.assert_close(after, before)

    def test_model_rejects_wrong_project_shape(self) -> None:
        model = build_saurl(small_config())
        with self.assertRaises(ValueError):
            model.encode(torch.randn(2, 7, 5))


if __name__ == "__main__":
    unittest.main()
