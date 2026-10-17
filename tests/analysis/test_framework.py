from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.ablation import assert_paired_predictions, removal_contribution, single_branch_skill
from analysis.artifacts import (hash_array, identity_hash, load_prediction, read_json,
                               save_prediction, subset_identity, write_arrays_atomic, write_json_atomic)
from analysis.cli import main
from analysis.conditional import assign_groups, evaluate_groups, fit_conditions, pool_walks, summarize_contracts
from analysis.contracts import ABLATIONS, BRANCHES, TASKS, build_matrix, validate_contracts
from analysis.features import apply_standardizer, fit_standardizer, save_standardizer, select_features, validate_scaler_parity
from analysis.runtime import resolve_recorded_path
from analysis.similarity import compute_cka, sample_classification_splits
from analysis.tasks import build_head, build_loss, evaluate_predictions, rank_ic_values, reference_records
from analysis.types import *
from analysis.validation import compare_numbers, result

ROOT = Path(__file__).resolve().parents[2]
HOUR = 3600 * 10**9


def dataset_fixture(root, task=AnalysisTask.PRICE, walk=1):
    rng = np.random.default_rng(3)
    source = TaskWalkSource(task, walk, root / "data.npz", root / "features.npz",
        root / "data.npz.manifest.json", root / "features.npz.manifest.json", root / "h0", root / "scaler.npz")
    splits = []
    for name, n, base in ((Split.TRAIN, 15, 1000), (Split.TEST, 10, 2000 + walk * 1000)):
        times = (base + np.arange(n) // 5) * HOUR
        identity = RowIdentity(task, walk, np.asarray([f"c{i%5}" for i in range(n)]),
            times - 63 * HOUR, times, times + HOUR,
            {"target_date_ns": times + (2 if task == AnalysisTask.CLASSIFICATION else 8) * HOUR})
        targets = np.arange(n) % 3 if task == AnalysisTask.CLASSIFICATION else np.linspace(.45, .55, n)
        if task == AnalysisTask.VARIANCE:
            targets = np.linspace(0, .01, n)
        branches = {b.key: rng.normal(size=(n, b.width)).astype(np.float32) for b in BRANCHES}
        branches["statistical"][:, 0] = 1.
        metadata = {"current_close": np.full(n, .5), "last_hour_reversal": np.linspace(.1, -.1, n),
                    "historical_rv": np.full(n, .001)}
        splits.append(TaskSplit(name, identity, targets, branches, metadata))
    return TaskDataset(source, *splits, {"dataset": "fixture-data", "features": "fixture-features"})


def record_fixture(dataset, outputs=None, config="H0"):
    task = dataset.source.task
    if outputs is None:
        outputs = np.eye(3)[dataset.test.targets.astype(int)] if task == AnalysisTask.CLASSIFICATION else dataset.test.targets + .001
    return PredictionRecord(task, dataset.source.walk, config, 0, 50, dataset.test.identity,
        dataset.test.targets, np.asarray(outputs), {AnalysisTask.CLASSIFICATION:"class_logits",
        AnalysisTask.PRICE:"raw_probability", AnalysisTask.VARIANCE:"raw_realised_variance"}[task],
        dataset.source_hashes, dataset.test.metadata)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = AnalysisContext(ROOT, self.root / "output")

    def test_matrix_and_width_contracts(self):
        validate_contracts()
        matrix = build_matrix(self.context)
        self.assertEqual((len(matrix.runs), len(matrix.references)), (60, 6))
        self.assertEqual(len({s.output_path for s in matrix.runs}), 60)
        self.assertEqual(len({s.ablation.width for s in matrix.runs}), 8)
        with self.assertRaises(ValueError):
            replace(matrix.runs[0], recipe=TrainingRecipe(epochs=2))

    def test_source_paths_relocate_explicitly(self):
        context = replace(self.context, path_mappings=(PathMapping("/old/root", ROOT),))
        self.assertEqual(resolve_recorded_path("/old/root/experiments/a.npz", context), ROOT / "experiments/a.npz")
        with self.assertRaises(ValueError):
            resolve_recorded_path("/other/a.npz", context)
        with self.assertRaises(ValueError):
            resolve_recorded_path("/old/root/../../escape", context)

    def test_scaling_training_only_and_constant_coordinates(self):
        data = dataset_fixture(self.root)
        full = np.concatenate([data.train.branches[b.key] for b in BRANCHES], axis=1)
        canonical = fit_standardizer(full, data.train.identity)
        for ablation in ABLATIONS:
            raw = select_features(data.train, ablation)
            scaler = fit_standardizer(raw, data.train.identity)
            validate_scaler_parity(scaler, canonical, ablation)
            transformed = apply_standardizer(raw, scaler)
            self.assertTrue(np.isfinite(transformed).all())
            self.assertEqual(transformed.shape, (15, ablation.width))
        self.assertEqual(canonical.scale[0], 1)
        self.assertTrue(np.all(apply_standardizer(full + 1e6, canonical) <= 10))
        np.testing.assert_array_equal(canonical.mean, fit_standardizer(full, data.train.identity).mean)

    def test_pairing_rejects_reordered_targets_and_identities(self):
        record = record_fixture(dataset_fixture(self.root))
        assert_paired_predictions(record, record)
        with self.assertRaises(ValueError):
            assert_paired_predictions(record, replace(record, targets=record.targets[::-1]))
        with self.assertRaises(ValueError):
            assert_paired_predictions(record, replace(record, identity=subset_identity(record.identity, np.arange(9,-1,-1))))

    def test_metric_and_removal_signs_and_denominator(self):
        data = dataset_fixture(self.root)
        h0 = record_fixture(data, data.test.targets + .01)
        removed = replace(h0, configuration="A-noB", outputs=data.test.targets + .02)
        values = {v.metric:v for v in removal_contribution(h0, removed)}
        self.assertAlmostEqual(values["mae"].benefit_positive_difference, .01)
        self.assertAlmostEqual(values["mae"].relative_error_change, 1.)
        skill = {v.metric:v for v in single_branch_skill(h0, removed)}
        self.assertAlmostEqual(skill["mae"].relative_error_change, .5)

    def test_rank_ic_requires_five_distinct_conditions_and_common_coverage(self):
        data = dataset_fixture(self.root)
        record = record_fixture(data, data.test.targets)
        self.assertEqual(len(rank_ic_values(record)), 2)
        self.assertAlmostEqual(next(iter(rank_ic_values(record).values())), 1.)
        constant = replace(record, outputs=np.full(10, .5), configuration="A-C")
        values = {v.metric:v for v in single_branch_skill(record, constant)}
        self.assertIsNone(values["rank_ic"].benefit_positive_difference)
        mask = np.arange(10) % 5 != 0
        from analysis.artifacts import subset_record
        self.assertEqual(rank_ic_values(subset_record(record, mask)), {})

    def test_metric_nulls_are_not_zero_and_prediction_roundtrip(self):
        data = dataset_fixture(self.root, AnalysisTask.VARIANCE)
        record = record_fixture(data, np.zeros(10))
        values = {m.name:m for m in evaluate_predictions(record)}
        self.assertIsNone(values["spearman"].value)
        path = self.root / "predictions.npz"
        save_prediction(path, record)
        restored = load_prediction(path)
        self.assertEqual(identity_hash(record.identity), identity_hash(restored.identity))
        np.testing.assert_array_equal(record.outputs, restored.outputs)
        write_json_atomic(self.root / "metrics.json", {"metrics": tuple(values.values())})
        self.assertNotIn("NaN", (self.root / "metrics.json").read_text())

    def test_rv_thresholds_training_only_and_disjoint_bins(self):
        data = dataset_fixture(self.root, AnalysisTask.VARIANCE)
        spec = fit_conditions(data)
        shifted_test = replace(data, test=replace(data.test, targets=data.test.targets * 100))
        self.assertEqual(spec, fit_conditions(shifted_test))
        record = record_fixture(data)
        groups = assign_groups(record, spec)
        bins = [g.mask for g in groups if g.dimension == "rv"]
        np.testing.assert_array_equal(sum(bins), np.ones(10))
        tail = [r for r in evaluate_groups(record, groups) if r.group == "rv_tail:q90"][0]
        self.assertIn("tail_sse_share", {m.name for m in tail.metrics})
        with self.assertRaises(ValueError):
            assign_groups(record, replace(spec, dimensions=("imputation",)))

    def test_collapsed_rv_quantiles_have_explicit_empty_bin(self):
        data = dataset_fixture(self.root, AnalysisTask.VARIANCE)
        data = replace(data, train=replace(data.train, targets=np.zeros(15)))
        groups = assign_groups(record_fixture(data), fit_conditions(data))
        middle = next(g for g in groups if g.name == "q90_to_q99")
        self.assertFalse(middle.mask.any())
        self.assertTrue(all(m.value is None for m in evaluate_groups(record_fixture(data), (middle,))[0].metrics))

    def test_contract_macro_and_pooling_recompute_metrics(self):
        data = dataset_fixture(self.root)
        first = record_fixture(data)
        second = record_fixture(dataset_fixture(self.root, walk=2))
        grouped = evaluate_groups(first, assign_groups(first, fit_conditions(data)))
        self.assertTrue(summarize_contracts(grouped))
        pooled = pool_walks((first, second))
        self.assertTrue(any(r.group == "pooled" for r in pooled))
        with self.assertRaises(ValueError):
            pool_walks((first, first))

    def test_cka_samples_and_known_geometry(self):
        data = [dataset_fixture(self.root, AnalysisTask.CLASSIFICATION, w) for w in (1,2)]
        samples = sample_classification_splits(data)
        repeated = sample_classification_splits(data)
        self.assertEqual([s.identity_hash for s in samples], [s.identity_hash for s in repeated])
        for d in data:
            scaler = fit_standardizer(np.concatenate([d.train.branches[b.key] for b in BRANCHES],axis=1), d.train.identity)
            sample = next(s for s in samples if s.walk == d.source.walk and s.split == Split.TRAIN)
            cka = compute_cka(d.train, scaler, sample)
            np.testing.assert_allclose(cka.scores, cka.scores.T)
            np.testing.assert_allclose(np.diag(cka.scores), np.ones(5))

    def test_numerical_warnings_and_material_failure(self):
        warning = compare_numbers("outputs", [1.000001], [1.], rtol=1e-5, atol=1e-7)
        self.assertEqual(result("example", (warning,)).status, "valid_with_warnings")
        self.assertTrue(result("example", (warning,)).usable)
        self.assertFalse(result("example", (compare_numbers("outputs", [2.], [1.], rtol=1e-5, atol=1e-7),)).usable)
        self.assertFalse(result("example", (compare_numbers("outputs", [np.nan], [1.], rtol=1e-5, atol=1e-7),)).usable)

    def test_heads_losses_all_widths_and_tasks(self):
        import torch
        torch.set_num_threads(1)
        matrix = build_matrix(self.context)
        seen = set()
        for spec in matrix.runs:
            key = (spec.source.task, spec.ablation.width)
            if key in seen:
                continue
            seen.add(key)
            head = build_head(spec)
            y = np.asarray([0,1,2]) if spec.source.task == AnalysisTask.CLASSIFICATION else np.asarray([.1,.2,.3])
            loss = build_loss(spec, y)
            outputs = head(torch.zeros(3, spec.ablation.width))
            target = torch.tensor(y, dtype=torch.long) if spec.source.task == AnalysisTask.CLASSIFICATION else torch.tensor(y, dtype=torch.float32).reshape(-1,1)
            value = loss(outputs, target)
            value.backward()
            self.assertTrue(torch.isfinite(value))
        self.assertEqual(len(seen), 24)

    def test_plot_and_table_exports(self):
        from analysis.ablation import summarize_predictions
        from analysis.reporting import plot_contributions, plot_similarity, write_metric_tables
        data = dataset_fixture(self.root)
        full = record_fixture(data)
        removed = replace(full, configuration="A-noB", outputs=full.outputs + .001)
        comparisons = removal_contribution(full, removed)
        files = write_metric_tables(summarize_predictions((full, removed)), self.root / "tables")
        self.assertTrue(all(p.is_file() and p.stat().st_size > 0 for p in files))
        figures = plot_contributions(comparisons, self.root / "figures")
        self.assertEqual(len(figures), 6)
        data = [dataset_fixture(self.root, AnalysisTask.CLASSIFICATION, w) for w in (1,2)]
        samples = sample_classification_splits(data)
        results = []
        for dataset in data:
            scaler = fit_standardizer(np.concatenate([dataset.train.branches[b.key] for b in BRANCHES],axis=1), dataset.train.identity)
            for sample in (s for s in samples if s.walk == dataset.source.walk):
                results.append(compute_cka(dataset.train if sample.split == Split.TRAIN else dataset.test, scaler, sample))
        self.assertEqual(len(plot_similarity(results, self.root / "figures")), 8)

    def test_missing_snapshot_diagnostics_do_not_occupy_a_training_path(self):
        from analysis.replay import replay_snapshot
        data = dataset_fixture(self.root)
        spec = ProbeRunSpec(data.source, ABLATIONS[0], self.context.output_root / "missing-run")
        with contextlib.redirect_stderr(io.StringIO()):
            value = replay_snapshot(spec, data, 50, context=self.context)
        self.assertFalse(value.usable)
        self.assertFalse(spec.output_path.exists())

    def test_metric_table_replay_accepts_small_drift_but_rejects_shape_changes(self):
        from analysis.artifacts import json_value
        from analysis.ablation import summarize_predictions
        from analysis.replay import compare_metric_rows
        original = json_value(summarize_predictions((record_fixture(dataset_fixture(self.root)),)))
        import copy
        changed = copy.deepcopy(original)
        changed[0]["metrics"][0]["value"] += 1e-9
        self.assertTrue(result("metrics", compare_metric_rows(original, changed, ReplayPolicy())).usable)
        changed[0]["configuration"] = "different"
        with self.assertRaises(ValueError):
            compare_metric_rows(original, changed, ReplayPolicy())

    def test_cli_execute_requires_verified_lumid(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["--repo-root", str(ROOT), "--output-root", str(self.root / "out"), "launch", "--execute"])
        self.assertEqual(code, 1)
        self.assertFalse(list((self.root / "out").glob("downstream/**/checkpoint.pth")))

    def test_preparation_similarity_and_partial_report_connect(self):
        from analysis.pipeline import AnalysisPipeline
        from analysis.replay import validate_preparation
        from analysis.reporting import build_report
        from analysis.validation import result
        data = {(task, walk): dataset_fixture(self.root / task.value / str(walk), task, walk)
                for task in TASKS for walk in (1,2)}
        for dataset in data.values():
            full = np.concatenate([dataset.train.branches[b.key] for b in BRANCHES], axis=1)
            save_standardizer(dataset.source.h0_scaler_path, fit_standardizer(full, dataset.train.identity))

        def locate(context, task, walk):
            return data[(task, walk)].source

        def load(source, context):
            return data[(source.task, source.walk)]

        def reference(source, epoch, context, dataset=None):
            return replace(record_fixture(load(source, context)), epoch=epoch)

        with patch("analysis.sources.locate_source", side_effect=locate), patch("analysis.sources.load_dataset", side_effect=load), patch("analysis.replay.load_dataset", side_effect=load):
            pipeline = AnalysisPipeline(self.context)
            with contextlib.redirect_stderr(io.StringIO()):
                pipeline.launch(execute=False)
            matrix = build_matrix(self.context)
            self.assertTrue(validate_preparation(self.context, matrix).usable)
            self.assertEqual(len(list((self.context.output_root / "features").rglob("branch_view.json"))), 60)
            self.assertFalse((self.context.output_root / "downstream").exists())
            pipeline.similarity()
            self.assertEqual(len(list((self.context.output_root / "analysis/cka").rglob("scores.npz"))), 4)
            for source in matrix.references:
                write_json_atomic(self.context.output_root / "analysis/reference_replay" / source.task.value / f"walk{source.walk}.json", {"status":"valid"})
            with patch("analysis.replay.validate_complete", return_value=ValidationResult("matrix", "invalid")), patch("analysis.sources.audit_source", return_value=result("fixture")), patch("analysis.sources.load_h0_predictions", side_effect=reference):
                with self.assertRaisesRegex(ValueError, "missing/invalid"):
                    build_report(self.context, matrix)
                report = build_report(self.context, matrix, allow_partial=True)
            self.assertFalse(report.complete)
            self.assertIn("PARTIAL / INCOMPLETE", (report.output_path / "summary.md").read_text())
            self.assertTrue((report.output_path / "artifact_inventory.json").is_file())


class TrainingLifecycleTests(unittest.TestCase):
    def test_failed_attempt_restarts_fresh_and_saved_predictions_replay(self):
        import torch
        from analysis.runtime import collect_environment
        from analysis.training import ProbeRunner
        from analysis.replay import replay_snapshot
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = AnalysisContext(ROOT, root / "outputs")
            data = dataset_fixture(root, AnalysisTask.CLASSIFICATION)
            full = np.concatenate([data.train.branches[b.key] for b in BRANCHES],axis=1)
            save_standardizer(data.source.h0_scaler_path, fit_standardizer(full, data.train.identity))
            spec = ProbeRunSpec(data.source, ABLATIONS[0], context.output_root / "run", device="cpu")
            runner = ProbeRunner(context, collect_environment(context))
            # Only synthetic rows run locally; production admission stays enforced.
            with patch("analysis.training.require_admission"), patch("analysis.training.build_loss", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    runner.train(spec, data)
            self.assertEqual(read_json(spec.output_path / "state.json")["state"], "failed")
            with patch("analysis.training.require_admission"), contextlib.redirect_stdout(io.StringIO()):
                value = runner.train(spec, data)
            self.assertEqual(value.state, RunState.EVALUATED)
            self.assertEqual(len(list((context.output_root / "failed_attempts").iterdir())), 1)
            self.assertFalse((spec.output_path / "resume.pth").exists())
            for epoch in (5,15,50):
                self.assertTrue(replay_snapshot(spec, data, epoch, context=context).usable)
            changed = replace(data, source_hashes={"dataset":"different"})
            with patch("analysis.training.require_admission"):
                with self.assertRaisesRegex(ValueError, "conflicting provenance"):
                    runner.train(spec, changed)


if __name__ == "__main__":
    unittest.main()
