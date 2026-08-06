from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_contract(n_rows: int = 140, phase: float = 0.0) -> pd.DataFrame:
    t = np.arange(n_rows, dtype=np.float64)
    close = 0.45 + 0.08 * np.sin(t / 7.0 + phase) + 0.0005 * t
    return pd.DataFrame(
        {
            "open": close * (1.0 + 0.002 * np.sin(t / 5.0)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 10.0 + t + 0.5 * np.cos(t / 3.0),
        }
    )


def write_synthetic_cache(cache_path: Path, manifest_path: Path) -> dict:
    rng = np.random.default_rng(123)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    X_train = rng.normal(size=(12, 64, 5)).astype(np.float32)
    X_test = rng.normal(size=(6, 64, 5)).astype(np.float32)
    y_gt_train = np.abs(rng.normal(0.08, 0.01, size=(12, 1))).astype(np.float32)
    y_gt_test = np.abs(rng.normal(0.08, 0.01, size=(6, 1))).astype(np.float32)
    y_garch_train = np.abs(rng.normal(0.07, 0.01, size=(12, 1))).astype(np.float32)
    y_garch_test = np.abs(rng.normal(0.07, 0.01, size=(6, 1))).astype(np.float32)
    contract_id_train = np.repeat(np.arange(3, dtype=np.int32), 4)
    contract_id_test = np.repeat(np.arange(3, dtype=np.int32), 2)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_gt_train=y_gt_train,
        y_garch_train=y_garch_train,
        contract_id_train=contract_id_train,
        X_test=X_test,
        y_gt_test=y_gt_test,
        y_garch_test=y_garch_test,
        contract_id_test=contract_id_test,
    )
    manifest = {
        "dataset_id": "synthetic-cache",
        "config": {
            "timeframe": "4h",
            "seq_len": 64,
            "top_k": 50,
            "train_ratio": 0.8,
            "ar_order": 5,
            "garch_p": 1,
            "garch_q": 1,
        },
        "totals": {
            "train_samples": 12,
            "test_samples": 6,
            "included_contracts": 3,
            "skipped_contracts": 0,
            "garch_fallbacks": 0,
        },
        "arrays": {
            "X_train": {"shape": [12, 64, 5], "dtype": "float32"},
            "y_gt_train": {"shape": [12, 1], "dtype": "float32"},
            "y_garch_train": {"shape": [12, 1], "dtype": "float32"},
            "contract_id_train": {"shape": [12], "dtype": "int32"},
            "X_test": {"shape": [6, 64, 5], "dtype": "float32"},
            "y_gt_test": {"shape": [6, 1], "dtype": "float32"},
            "y_garch_test": {"shape": [6, 1], "dtype": "float32"},
            "contract_id_test": {"shape": [6], "dtype": "int32"},
        },
        "skipped_contracts": [],
        "contracts": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def make_two_epoch_synthetic_run(root: Path) -> Path:
    from baselines.ginn_baseline.ginn_data import GinnDataConfig, load_and_validate_cache
    from baselines.ginn_baseline.run_experiment import ExperimentConfig, run_loaded_experiment

    cache_path = root / "cache.npz"
    manifest_path = root / "manifest.json"
    run_root = root / "run"
    write_synthetic_cache(cache_path, manifest_path)
    cache = load_and_validate_cache(cache_path, manifest_path, GinnDataConfig())
    run_loaded_experiment(
        cache,
        run_root,
        ExperimentConfig(
            epoch_budgets=(1, 2),
            seed=0,
            batch_size=4,
            learning_rate=1e-3,
            lambda_garch=0.3,
            input_size=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            device="cpu",
        ),
        cache_path=cache_path,
        manifest_path=manifest_path,
    )
    return run_root
