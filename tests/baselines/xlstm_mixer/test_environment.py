"""Regression tests for index-free and explicit local GPU requests."""

from unittest import TestCase
from unittest.mock import patch

import torch

from platform_integration.xlstm_mixer_environment import synthetic_gpu_smoke


class EnvironmentDeviceTests(TestCase):
    def check_device(self, request: str, expected_index: int) -> None:
        # Stop immediately after device selection; no GPU or backend required.
        with (
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.set_device") as select,
            patch(
                "platform_integration.xlstm_mixer_environment.set_seed",
                side_effect=RuntimeError("stop after device selection"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after device selection"):
                synthetic_gpu_smoke(torch.device(request), 512, "vanilla")
            select.assert_called_once_with(expected_index)

    def test_default_cuda_uses_current_device_index(self) -> None:
        self.check_device("cuda", 0)

    def test_explicit_cuda_preserves_requested_index(self) -> None:
        self.check_device("cuda:2", 2)
