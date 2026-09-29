from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.garch_lstm_stacking.garch import RawChangeGarchHorizonForecaster
from baselines.garch_lstm_stacking.phase6_5 import (
    HOUR_NS,
    Phase65GarchLSTMConfig,
    _garch_evaluation_predictions,
    make_calendar_expanding_oof_plan,
    run_phase65_garch_lstm,
    smoke_test_phase65_garch_lstm,
)


class Phase65GarchLSTMTests(unittest.TestCase):
    def test_raw_change_forecast_is_eight_step_nonnegative_and_frozen(self) -> None:
        prices = 0.4 + np.cumsum(np.asarray([0.0, 0.01, -0.004, 0.007, -0.002] * 8))
        forecaster = RawChangeGarchHorizonForecaster()
        state = forecaster.fit(prices)
        first = forecaster.forecast(state, horizon=8)
        second = forecaster.forecast(state, horizon=8)
        self.assertEqual(first.hourly_variance_guarded.shape, (8,))
        self.assertGreaterEqual(first.prediction_guarded, 0.0)
        self.assertEqual(first.prediction_guarded, second.prediction_guarded)
        self.assertEqual(state.fit_change_count, len(prices) - 1)

    def test_oof_plan_is_calendar_ordered_and_target_mature(self) -> None:
        decisions = np.repeat(np.arange(36, dtype=np.int64) * HOUR_NS, 2)
        targets = decisions + 2 * HOUR_NS
        contracts = np.asarray(["a", "b"] * 36)
        plan = make_calendar_expanding_oof_plan(decisions, targets, contracts)
        self.assertEqual(len(plan.folds), 5)
        self.assertEqual(len(np.unique(plan.prediction_positions)), len(plan.prediction_positions))
        for fold in plan.folds:
            self.assertLess(
                int(targets[fold.training_positions].max()),
                int(decisions[fold.prediction_positions].min()),
            )

    def test_oof_plan_discards_contract_rows_without_fold_start_history(self) -> None:
        decisions = np.repeat(np.arange(36, dtype=np.int64) * HOUR_NS, 2)
        targets = decisions + 2 * HOUR_NS
        contracts = np.asarray(["old", "new"] * 36)
        decision_dates = decisions.copy()
        decision_dates[contracts == "new"] += 200 * HOUR_NS
        plan = make_calendar_expanding_oof_plan(
            decisions,
            targets,
            contracts,
            decision_date_ns=decision_dates,
        )
        predicted_contracts = contracts[plan.prediction_positions]
        self.assertTrue(np.all(predicted_contracts == "old"))
        self.assertGreater(len(plan.burn_in_positions), len(np.unique(decisions)) // 6)

    def test_frozen_config_and_smoke(self) -> None:
        self.assertTrue(smoke_test_phase65_garch_lstm()["valid"])
        with self.assertRaises(ValueError):
            Phase65GarchLSTMConfig(walk=1, folds=4)
        with self.assertRaises(ValueError):
            Phase65GarchLSTMConfig(walk=1, horizon_hours=4)

    def test_unseen_evaluation_contract_uses_training_only_pooled_fallback(self) -> None:
        sequences = np.zeros((2, 64, 5), dtype=np.float32)
        sequences[0, :, 3] = np.linspace(0.2, 0.4, 64)
        sequences[1, :, 3] = np.linspace(0.5, 0.45, 64)
        predictions, diagnostics = _garch_evaluation_predictions(
            {
                "train_raw_sequences": sequences,
                "train_condition_ids": np.asarray(["seen-a", "seen-b"]),
                "train_decision_date_ns": np.asarray([100, 100], dtype=np.int64) * HOUR_NS,
                "test_condition_ids": np.asarray(["unseen", "unseen"]),
            }
        )
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertEqual(diagnostics[0]["fallback_reason"], "unseen_evaluation_contract")
        self.assertEqual(diagnostics[0]["pooled_training_contract_count"], 2)

    def test_occupied_run_is_refused_before_loading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                run_phase65_garch_lstm(
                    root / "missing-labels.npz",
                    root / "missing-run",
                    root,
                    Phase65GarchLSTMConfig(walk=1, device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
