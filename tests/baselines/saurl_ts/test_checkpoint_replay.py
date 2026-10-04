from __future__ import annotations

import io
import unittest

import torch

from baselines.saurl_ts import build_saurl
from tests.baselines.saurl_ts.helpers import small_config


class SaURLCheckpointReplayTests(unittest.TestCase):
    def test_state_dict_reload_replays_deterministic_inference(self) -> None:
        torch.manual_seed(40)
        model = build_saurl(small_config()).eval()
        probe = torch.randn(3, 8, 5)
        expected = model.encode_parts(probe)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        reloaded = build_saurl(small_config()).eval()
        reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        actual = reloaded.encode_parts(probe)
        torch.testing.assert_close(actual.branches, expected.branches)
        torch.testing.assert_close(actual.attention_weights, expected.attention_weights)
        torch.testing.assert_close(actual.fused, expected.fused)


if __name__ == "__main__":
    unittest.main()
