from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from baselines.xlstm_mixer.endpoint import XLSTMMixerEndpoint, architecture_manifest
from training.phase6_9_xlstm_endpoint import (
    EndpointTrainingConfig, evaluate_endpoint, load_endpoint_data,
    phase_root, provenance_path, run_endpoint_training, run_root, validate_endpoint_training,
)


class RecordingStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.1)
        self.calls = []

    def forward(self, tokens):
        self.calls.append(tokens.detach().clone())
        return self.dropout(self.projection(tokens + tokens.mean(dim=1, keepdim=True)))


@pytest.fixture
def endpoint():
    with patch("baselines.xlstm_mixer.endpoint.build_xlstm_stack", side_effect=lambda *_: RecordingStack()):
        yield XLSTMMixerEndpoint()


def test_endpoint_is_one_forecast_at_eight_hours(endpoint):
    spec = architecture_manifest()
    assert (spec["forecast_steps"], spec["forecast_horizon_hours"]) == (1, 8)
    assert spec["target_channel"] == "close"
    assert not spec["auxiliary_supervision"]
    assert endpoint.time_projection.weight.shape == (1, 64)
    assert endpoint.up_projection.weight.shape == (128, 1)
    assert endpoint.output_projection.weight.shape == (1, 256)
    assert endpoint(torch.rand(3, 64, 5)).shape == (3, 1)


def test_endpoint_reversal_keeps_source_axis_and_call_order(endpoint):
    endpoint(torch.rand(2, 64, 5))
    reverse, forward = endpoint.slstm_stack.calls
    assert forward.shape == (2, 6, 128)
    torch.testing.assert_close(reverse, torch.flip(forward, dims=(-1,)))


def test_endpoint_mse_is_exact_single_target(endpoint):
    prediction = torch.tensor([[0.2], [0.4]], requires_grad=True)
    target = torch.tensor([[0.3], [0.6]])
    loss = endpoint.endpoint_mse_loss(prediction, target)
    torch.testing.assert_close(loss, ((prediction - target) ** 2).mean())
    loss.backward()
    torch.testing.assert_close(prediction.grad, prediction.detach() - target)


@pytest.mark.parametrize("target", [torch.rand(2, 8, 5), torch.rand(2), torch.full((2, 1), float("nan"))])
def test_endpoint_rejects_paths_broadcasting_and_nonfinite_targets(endpoint, target):
    with pytest.raises(ValueError):
        endpoint.endpoint_mse_loss(torch.rand(2, 1), target)


@pytest.mark.parametrize("values", [torch.rand(2, 8, 5), torch.rand(2, 64, 4), torch.full((2, 64, 5), float("nan"))])
def test_endpoint_rejects_invalid_contexts(endpoint, values):
    with pytest.raises(ValueError):
        endpoint(values)


def test_endpoint_backward_and_checkpoint_guard(endpoint):
    contexts = torch.rand(2, 64, 5)
    loss = endpoint.endpoint_mse_loss(endpoint(contexts), torch.rand(2, 1))
    loss.backward()
    assert endpoint.initial_token.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in endpoint.parameters() if p.grad is not None)
    endpoint.eval()
    with patch("baselines.xlstm_mixer.endpoint.build_xlstm_stack", side_effect=lambda *_: RecordingStack()):
        replay = XLSTMMixerEndpoint().eval()
    state = copy.deepcopy(endpoint.state_dict())
    replay.load_state_dict(state)
    torch.testing.assert_close(replay(contexts), endpoint(contexts), rtol=0, atol=0)
    state["_extra_state"]["architecture"]["forecast_horizon_hours"] = 1
    with pytest.raises(RuntimeError, match="architecture/source"):
        replay.load_state_dict(state)


def test_endpoint_inverse_revin_uses_close_statistics_without_clipping(endpoint):
    with torch.no_grad():
        endpoint.output_projection.weight.zero_()
        endpoint.output_projection.bias.zero_()
    contexts = torch.rand(2, 64, 5)
    torch.testing.assert_close(endpoint(contexts), contexts[:, :, 3].mean(dim=1, keepdim=True))
    with torch.no_grad():
        endpoint.output_projection.bias.fill_(100)
    assert (endpoint(contexts) > 1).all()  # No silent sigmoid/clipping policy.


def _fixture_data():
    generator = np.random.default_rng(0)
    data = {}
    for split, count in (("train", 9), ("test", 6)):
        x = generator.random((count, 64, 5), dtype=np.float32)
        current = x[:, -1, 3].astype(np.float64)
        target = np.minimum(1, current + 0.01)
        data[f"X_{split}"] = x
        data[f"y_{split}"] = target
        data[f"{split}_metadata"] = {
            "condition_ids": np.asarray([f"condition-{i}" for i in range(count)]),
            "window_start_ns": np.zeros(count, dtype=np.int64),
            "decision_date_ns": np.zeros(count, dtype=np.int64),
            "decision_availability_ns": np.full(count, 3_600_000_000_000),
            "target_date_ns": np.full(count, 8 * 3_600_000_000_000),
            "current_close": current, "target_close": target,
            "context_imputed_rows": np.zeros(count, dtype=np.int32),
            "lifecycle_stage": np.zeros(count, dtype=np.int32),
        }
    return data, {"method_id": "XM-C8", "walk": 1, "dataset_sha256": "fixture"}


def test_endpoint_metrics_keep_original_target_and_minimum_contract_count():
    data, _ = _fixture_data()
    metrics, arrays = evaluate_endpoint(data["y_test"].reshape(-1, 1), data)
    assert metrics["price"]["overall"]["mae"] == 0
    assert metrics["timestamp_cross_sectional_rank_ic"]["minimum_contract_count"] == 5
    assert "target_full_path" not in arrays and "prediction_full_path" not in arrays
    np.testing.assert_array_equal(arrays["target_future_price"], data["y_test"])


def test_endpoint_loader_preserves_rows_without_future_path_requirement(tmp_path):
    data, _ = _fixture_data()
    path = tmp_path / "original.npz"
    manifest = {"walk": 1, "horizon_bars": 8, "identity_hashes": {"train": "t", "test": "e"}}
    path.write_bytes(b"fixture")
    Path(f"{path}.manifest.json").write_text(json.dumps(manifest))
    with patch("training.phase6_9_xlstm_endpoint.original_data_path", return_value=path), \
         patch("training.phase6_9_xlstm_endpoint.load_baseline_data", return_value=data), \
         patch.dict("training.phase6_9_xlstm_endpoint.EXPECTED_ROWS", {1: (9, 6)}):
        loaded, audit = load_endpoint_data(tmp_path, 1)
    assert loaded is data  # No selection, copy, transform, or future-path join.
    assert not audit["row_filtering"] and not audit["auxiliary_supervision"]


def test_endpoint_recipe_rejects_test_selected_budget_or_batch():
    for kwargs in ({"epochs": 15}, {"batch_size": 128}, {"seed": 1}, {"learning_rate": 1e-3}):
        with pytest.raises(ValueError):
            EndpointTrainingConfig(1, **kwargs)


def test_wsl_provenance_normalizes_only_the_case_insensitive_e_mount():
    assert provenance_path(Path("/mnt/e/School-Work/Model.npz")) == "/mnt/e/school-work/model.npz"
    assert provenance_path(Path("/tmp/Model.npz")) == "/tmp/Model.npz"


def test_endpoint_cpu_fixture_resume_replay_and_tamper(tmp_path):
    data, manifest = _fixture_data()
    config = EndpointTrainingConfig(1, device="cpu")
    admission = phase_root(tmp_path) / "feasibility/cuda_admission.json"
    admission.parent.mkdir(parents=True)
    admission.write_text(json.dumps({"fixture": True}))
    with patch("training.phase6_9_xlstm_endpoint.load_endpoint_data", return_value=(data, manifest)), \
         patch("baselines.xlstm_mixer.endpoint.build_xlstm_stack", side_effect=lambda *_: RecordingStack()):
        stopped = run_endpoint_training(tmp_path, config, stop_after_epoch=2, _fixture=True)
        assert stopped == {"complete": False, "completed_epoch": 2}
        location = run_root(tmp_path, 1)
        assert not (location / "training_complete.json").exists()
        assert not (location / "e5/predictions.npz").exists()
        result = run_endpoint_training(tmp_path, config, _fixture=True)
        assert result["valid"]
        assert validate_endpoint_training(tmp_path, config)["valid"]
        checkpoint = torch.load(location / "e50/checkpoint.pth", weights_only=True)
        assert all(row["seen_rows"] == 9 for row in checkpoint["history"])
        assert all(row["last_batch_size"] == 9 for row in checkpoint["history"])
        with np.load(location / "e50/predictions.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["target_future_price"] = arrays["target_future_price"] + 0.01
        np.savez_compressed(location / "e50/predictions.npz", **arrays)
        with pytest.raises(ValueError, match="hash mismatch"):
            validate_endpoint_training(tmp_path, config)


def test_endpoint_resume_is_bit_exact_for_controlled_interruption(tmp_path):
    data, manifest = _fixture_data()
    config = EndpointTrainingConfig(1, device="cpu")
    roots = [tmp_path / name for name in ("interrupted", "continuous")]
    for root in roots:
        admission = phase_root(root) / "feasibility/cuda_admission.json"
        admission.parent.mkdir(parents=True)
        admission.write_text(json.dumps({"fixture": True}))
    with patch("training.phase6_9_xlstm_endpoint.load_endpoint_data", return_value=(data, manifest)), \
         patch("baselines.xlstm_mixer.endpoint.build_xlstm_stack", side_effect=lambda *_: RecordingStack()):
        run_endpoint_training(roots[0], config, stop_after_epoch=1, _fixture=True)
        run_endpoint_training(roots[0], config, stop_after_epoch=2, _fixture=True)
        run_endpoint_training(roots[1], config, stop_after_epoch=2, _fixture=True)
    states = [torch.load(run_root(root, 1) / "resume.pth", weights_only=True) for root in roots]
    for key in states[0]["model_state_dict"]:
        if isinstance(states[0]["model_state_dict"][key], torch.Tensor):
            torch.testing.assert_close(states[0]["model_state_dict"][key], states[1]["model_state_dict"][key], rtol=0, atol=0)
    assert states[0]["history"][-1]["train_loss"] == states[1]["history"][-1]["train_loss"]


def test_endpoint_full_training_refuses_cpu_without_explicit_fixture(tmp_path):
    with pytest.raises(ValueError, match="CUDA admission"):
        run_endpoint_training(tmp_path, EndpointTrainingConfig(1, device="cpu"))
