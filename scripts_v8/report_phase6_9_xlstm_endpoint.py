#!/usr/bin/env python3
"""Replay completed XM-C8 runs and report controls with their native metric schemas.

Reporting is deliberately separate from the fingerprinted, frozen training code.
No training, provenance migration, checkpoint rewriting, or metric retuning occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.xlstm_mixer.endpoint import METHOD_ID, architecture_manifest
from training.phase5_encoder import resolve_device, write_json
from training.phase6_9_xlstm_endpoint import (
    EndpointTrainingConfig, audit_existing_controls, dependency_identity,
    implementation_fingerprint, load_endpoint_data, phase_root, run_root,
    validate_endpoint_training,
)
from training.phase6_9_xlstm_mixer import runtime_environment_payload


def principal_row(walk: int, method: str, metrics: dict) -> dict:
    price = metrics["price"]["overall"]
    movement = metrics["implied_movement"]
    # H0 saves subgroup breakdowns; Raw LSTM and XM-C8 save overall directly.
    movement = movement.get("overall", movement)
    return {"walk": walk, "method": method, "count": price["count"],
            "mae": price["mae"], "rmse": price["rmse"],
            "implied_movement_spearman": movement["spearman"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("vanilla", "cuda"), default="vanilla")
    args = parser.parse_args()
    root = phase_root(ROOT)
    matrix = json.loads((root / "manifests/training_seed0.json").read_text())
    admission = json.loads((root / "feasibility/cuda_admission.json").read_text())
    if (matrix["method_id"] != METHOD_ID
            or matrix["implementation_sha256"] != implementation_fingerprint()
            or matrix["architecture"] != architecture_manifest()
            or matrix["principal_epoch"] != 50
            or admission["implementation_sha256"] != implementation_fingerprint()
            or admission["admitted"] is not True
            or admission["backend"] != args.backend
            or admission["environment"] != runtime_environment_payload(resolve_device(args.device))
            or admission["dependency"] != dependency_identity()):
        raise ValueError("completed-run report admission/freeze mismatch")
    rows, replay, controls, diagnostics, snapshots = [], [], {}, [], []
    for walk in (1, 2):
        _, manifest = load_endpoint_data(ROOT, walk)
        controls[str(walk)] = audit_existing_controls(ROOT, walk, manifest)
        config = EndpointTrainingConfig(walk, device=args.device, backend=args.backend)
        if (matrix["data"][str(walk)] != manifest
                or matrix["controls"][str(walk)] != controls[str(walk)]
                or matrix["entries"][walk - 1] != config.to_dict()
                or admission["data"][str(walk)] != manifest):
            raise ValueError("completed-run report data/control/recipe mismatch")
        location = run_root(ROOT, walk)
        complete = json.loads((location / "training_complete.json").read_text())
        if complete != {"completed_epoch": 50, "method_id": METHOD_ID,
                        "principal_epoch": 50, "replay_valid": True}:
            raise ValueError("XM-C8 trajectory is not replay-valid and complete")
        replay.append(validate_endpoint_training(ROOT, config))
        locations = {METHOD_ID: location,
                     **{name: Path(row["run_root"]) for name, row in controls[str(walk)].items()}}
        for method, control_location in locations.items():
            metrics = json.loads((control_location / "e50/metrics.json").read_text())
            rows.append(principal_row(walk, method, metrics))
        metrics = json.loads((location / "e50/metrics.json").read_text())
        checkpoint = torch.load(location / "e50/checkpoint.pth", map_location="cpu", weights_only=True)
        with np.load(location / "history.npz", allow_pickle=False) as history:
            seconds = float(history["seconds"].sum())
        diagnostics.append({"walk": walk, "train_seconds": seconds,
                            "peak_cuda_memory_bytes": checkpoint["peak_cuda_memory_bytes"],
                            "persistence_reference": metrics["persistence_reference"]["overall"],
                            "persistence_relative_skill": metrics["persistence_relative_skill"],
                            "timestamp_cross_sectional_rank_ic": metrics["timestamp_cross_sectional_rank_ic"],
                            "outside_probability_range_fraction": metrics["outside_probability_range_fraction"]})
        del checkpoint
        for epoch in (5, 15, 50):
            snapshot = json.loads((location / f"e{epoch}/metrics.json").read_text())
            snapshots.append({"epoch": epoch, **principal_row(walk, METHOD_ID, snapshot)})
    payload = {"principal_epoch": 50, "seed": 0, "endpoint_matched_comparison": True,
               "rows": rows, "controls": controls, "replay": replay,
               "diagnostics": diagnostics, "all_retained_snapshots": snapshots,
               "training_implementation_sha256": implementation_fingerprint(),
               "report_schema_correction": "H0 implied_movement.overall; Raw LSTM/XM-C8 implied_movement"}
    report = "# Endpoint-only xLSTM-Mixer comparison\n\nSeed 0, fixed epoch 50; original h8 price rows and endpoint-only MSE. Both trajectories and all six snapshots independently replay-valid.\n\n"
    report += "| Walk | Method | Rows | MAE | RMSE | Movement Spearman |\n|---:|---|---:|---:|---:|---:|\n"
    report += "".join(f"| {r['walk']} | {r['method']} | {r['count']} | {r['mae']:.9f} | {r['rmse']:.9f} | {r['implied_movement_spearman']:.6f} |\n" for r in rows)
    report += "\nXM-C8 improves MAE/RMSE over H0-D0 and Raw LSTM in both walks. Persistence remains stronger on MAE in both walks; XM-C8 improves persistence RMSE only in Walk 1. These are endpoint-matched complete-system results, not isolated representation or architecture gains.\n\n"
    report += "| Walk | Persistence MAE | Persistence RMSE | Timestamp Rank IC | Outside [0,1] | Training seconds | Peak allocated MiB |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    report += "".join(f"| {d['walk']} | {d['persistence_reference']['mae']:.9f} | {d['persistence_reference']['rmse']:.9f} | {d['timestamp_cross_sectional_rank_ic']['mean']:.6f} | {d['outside_probability_range_fraction']:.6f} | {d['train_seconds']:.2f} | {d['peak_cuda_memory_bytes']/1024**2:.2f} |\n" for d in diagnostics)
    report += "\nThe inverse-RevIN output is unclipped, unlike the sigmoid controls; XM-C8 also retains clip norm 1.0. This is an endpoint adaptation, not the paper's full-path reproduction. The protocol correction followed review of XM-MV8 results; fresh holdouts or separately approved replications are required for strong confirmatory claims. No checkpoint was selected by evaluation. Historical XM-MV8 artifacts are unchanged. No universal, multi-seed, significance, or trading claim follows.\n\n"
    report += "The original launcher completed training and replay but its final writer encountered `KeyError: spearman` because H0 stores subgroup metrics. This standalone report reads the correct existing field and independently replays all artifacts without changing fingerprinted training code, checkpoints, predictions, or metrics.\n"
    write_json(root / "reports/seed0/summary.json", payload)
    (root / "reports/seed0/summary.md").write_text(report)
    readiness_path = root / "readiness.json"
    readiness = json.loads(readiness_path.read_text())
    readiness.update({"real_data_trajectories_complete": True, "results": replay,
                      "independent_report_replay_valid": True,
                      "comparison_report": "reports/seed0/summary.md"})
    write_json(readiness_path, readiness)
    print(json.dumps({"valid": True, "rows": rows, "replay": replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
