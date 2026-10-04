from __future__ import annotations

import unittest

import torch

from baselines.saurl_ts import build_saurl
from tests.baselines.saurl_ts.helpers import (
    any_parameter_changed,
    clone_parameters,
    set_trainable,
    small_config,
)


class SaURLAlternatingUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(30)
        self.model = build_saurl(small_config())
        self.batch = torch.randn(3, 8, 5)

    def test_sada_only_update_is_isolated(self) -> None:
        sada_parameters = list(self.model.sada_parameters())
        sassl_parameters = list(self.model.sassl_parameters())
        set_trainable(sada_parameters, True)
        set_trainable(sassl_parameters, False)
        sada_before = clone_parameters(sada_parameters)
        sassl_before = clone_parameters(sassl_parameters)
        optimizer = torch.optim.Adam(sada_parameters, lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        made = self.model.make_views(self.batch)
        self.model.sada_losses(made).total.backward()
        optimizer.step()
        self.assertTrue(any_parameter_changed(sada_before, sada_parameters))
        self.assertFalse(any_parameter_changed(sassl_before, sassl_parameters))

    def test_sassl_only_update_leaves_sada_unchanged_and_without_gradients(self) -> None:
        sada_parameters = list(self.model.sada_parameters())
        sassl_parameters = list(self.model.sassl_parameters())
        set_trainable(sada_parameters, False)
        set_trainable(sassl_parameters, True)
        sada_before = clone_parameters(sada_parameters)
        sassl_before = clone_parameters(sassl_parameters)
        with torch.no_grad():
            views = self.model.make_views(self.batch).detached_views()
        optimizer = torch.optim.Adam(sassl_parameters, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        outputs = self.model.sassl_forward(views)
        outputs.total_loss.backward()
        optimizer.step()
        self.assertFalse(any_parameter_changed(sada_before, sada_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in sada_parameters))
        self.assertTrue(any_parameter_changed(sassl_before, sassl_parameters))

    def test_target_parameters_have_no_gradients_and_ema_is_exact(self) -> None:
        target_parameters = list(self.model.target_encoders.parameters()) + list(
            self.model.target_projectors.parameters()
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in target_parameters))
        online_parameter = next(self.model.online_encoders.parameters())
        target_parameter = next(self.model.target_encoders.parameters())
        old_target = target_parameter.detach().clone()
        with torch.no_grad():
            online_parameter.add_(1.0)
        expected = 0.99 * old_target + 0.01 * online_parameter.detach()
        self.model.update_targets(tau=0.99)
        torch.testing.assert_close(target_parameter, expected)

    def test_cross_branch_uses_temporal_one_and_frequency_one(self) -> None:
        views = self.model.make_views(self.batch).detached_views()
        bundle_a, bundle_b = self.model._bundle_inputs(views)
        self.assertIs(bundle_a["cross"], views.temporal_view1)
        self.assertIs(bundle_b["cross"], views.frequency_view1)


if __name__ == "__main__":
    unittest.main()
