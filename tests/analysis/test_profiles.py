"""Selected-framework tests use synthetic rows only; no production launch."""

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from .test_framework import ROOT, dataset_fixture
except ImportError:
    from test_framework import ROOT, dataset_fixture

from analysis.artifacts import (IDENTITY_FIELDS, hash_array, hash_file, identity_arrays,
                               read_json, write_arrays_atomic, write_json_atomic)
from analysis.contracts import ABLATIONS, BRANCHES, build_matrix, make_ablations, validate_contracts
from analysis.features import apply_standardizer, fit_standardizer, save_standardizer, select_features, validate_scaler_parity
from analysis.profiles import (SELECTED_PROFILES_PATH, load_profiles, profile_catalog, reference_name, similarity_sources,
                              snapshot_count, source_branches, validate_profile)
from analysis.similarity import compute_cka, load_saved_similarity, sample_splits
from analysis.sources import locate_source
from analysis.types import AnalysisContext, AnalysisProfile, AnalysisTask, ProbeRunSpec, RunState


def selected_fixture(root, task=AnalysisTask.PRICE, walk=1, family="lstm_capacity", configuration="HC-AL2"):
    data = dataset_fixture(root, task, walk)
    source = replace(data.source, profile=AnalysisProfile(task, family, configuration, "fixture selection"))
    branches = source_branches(source)
    rng = np.random.default_rng(19)
    def adapt(split):
        return replace(split, branches={b.key: split.branches[b.key] if b.key in split.branches
                       else rng.normal(size=(len(split.targets), b.width)).astype(np.float32) for b in branches})
    return replace(data, source=source, train=adapt(data.train), test=adapt(data.test))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = AnalysisContext(ROOT, self.root / "out")

    def test_canonical_matrix_and_labels_are_unchanged(self):
        self.assertEqual(make_ablations(), ABLATIONS)
        matrix = build_matrix(self.context)
        self.assertEqual((len(matrix.runs), len(matrix.references), snapshot_count(matrix)), (60, 6, 198))
        self.assertEqual(len(similarity_sources(self.context, matrix)), 2)
        self.assertTrue(all(s.profile is None for s in matrix.references))

    def test_task_assignments_are_not_cross_product_and_additions_expand(self):
        profiles = (AnalysisProfile(AnalysisTask.CLASSIFICATION, "temporal", "HC-SL", "macro-F1"),
                    AnalysisProfile(AnalysisTask.PRICE, "residual_cnn", "HC-AR", "MAE"),
                    AnalysisProfile(AnalysisTask.VARIANCE, "lstm_capacity", "HB-SL2", "MAE"))
        context = replace(self.context, profiles=profiles)
        matrix = build_matrix(context)
        self.assertEqual((len(matrix.runs), len(matrix.references), snapshot_count(matrix)), (64, 6, 210))
        self.assertEqual(len(similarity_sources(context, matrix)), 6)
        for source in matrix.references:
            self.assertEqual(source.profile, next(p for p in profiles if p.task == source.task))
            branches = source_branches(source)
            ablations = make_ablations(branches)
            validate_contracts(branches, ablations)
            self.assertEqual(len(ablations), 2 * len(branches))
            self.assertEqual(branches[-1].h0_interval[1], 573 if source.task == AnalysisTask.PRICE else 445)

    def test_same_variant_can_be_reused_for_all_tasks(self):
        profiles = tuple(AnalysisProfile(t, "temporal", "HC-SL", "recorded metric") for t in AnalysisTask)
        matrix = build_matrix(replace(self.context, profiles=profiles))
        self.assertEqual((len(matrix.runs), len(matrix.references)), (60, 6))
        with self.assertRaisesRegex(ValueError, "one analysis profile"):
            replace(self.context, profiles=(profiles[0], profiles[0]))

    def test_approved_phase7c_roster_and_counts(self):
        profiles = load_profiles(SELECTED_PROFILES_PATH)
        expected = {AnalysisTask.CLASSIFICATION: ("HC-ST", 5, 445),
                    AnalysisTask.PRICE: ("HB-AL", 6, 573),
                    AnalysisTask.VARIANCE: ("HC-AL", 6, 573)}
        context = replace(self.context, profiles=profiles)
        matrix = build_matrix(context)
        self.assertEqual((len(matrix.runs), len(matrix.references), snapshot_count(matrix)), (68, 6, 222))
        for source in matrix.references:
            config, count, width = expected[source.task]
            self.assertEqual((source.profile.family, source.profile.configuration), ("temporal", config))
            branches = source_branches(source)
            self.assertEqual((len(branches), sum(b.width for b in branches)), (count, width))
            self.assertEqual(sum(r.source.task == source.task for r in matrix.runs), 4 * count)
        variance = next(p for p in profiles if p.task == AnalysisTask.VARIANCE)
        self.assertIn("H0 has lower mean MAE", variance.selection_reason)

    def test_cli_phase7c_shortcut_is_explicit_and_nonexecuting(self):
        from analysis.cli import build_parser, main
        self.assertIsNone(build_parser().parse_args(["launch"]).profiles)
        self.assertEqual(build_parser().parse_args(["--phase7c", "launch"]).profiles, SELECTED_PROFILES_PATH)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--phase7c", "--profiles", "custom.json", "launch"])
        with patch("analysis.pipeline.AnalysisPipeline.launch", autospec=True, return_value=None) as launch:
            self.assertEqual(main(["--repo-root", str(self.root), "--phase7c", "launch"]), 0)
        context = launch.call_args.args[0].context
        self.assertEqual(context.profiles, load_profiles(SELECTED_PROFILES_PATH))
        self.assertEqual(context.output_root, self.root / "experiments/phase7/selected_variants")
        self.assertFalse(launch.call_args.kwargs["execute"])

    def test_profile_file_rejects_incomplete_unknown_or_unsupported_selection(self):
        path = self.root / "profiles.json"
        payload = {"profiles": {t.value: {"family": "lstm_capacity", "configuration": "HC-SL2",
                                          "selection_reason": "previous task metric"} for t in AnalysisTask}}
        write_json_atomic(path, payload)
        self.assertEqual(len(load_profiles(path)), 3)
        del payload["profiles"][AnalysisTask.PRICE.value]
        write_json_atomic(path, payload)
        with self.assertRaisesRegex(ValueError, "three tasks"):
            load_profiles(path)
        for family, config, task in (("unknown", "HC-SL", AnalysisTask.PRICE),
                                     ("temporal", "../escape", AnalysisTask.PRICE),
                                     ("residual_cnn", "HC-SR", AnalysisTask.CLASSIFICATION)):
            with self.assertRaises(ValueError):
                validate_profile(AnalysisProfile(task, family, config, "fixture"))
        with self.assertRaises(ValueError):
            validate_profile(AnalysisProfile(AnalysisTask.PRICE, "temporal", "HC-SL", ""))

    def test_adapter_paths_use_existing_native_layouts(self):
        for family, config in (("temporal", "HC-SL"), ("lstm_capacity", "HC-AL2"), ("residual_cnn", "HC-SR")):
            profile = AnalysisProfile(AnalysisTask.PRICE, family, config, "fixture")
            source = locate_source(replace(self.context, profiles=(profile,)), profile.task, 2)
            self.assertEqual(reference_name(source), config)
            self.assertIn(f"/{config.lower()}/walk2/seed0", str(source.h0_run_path))
            self.assertEqual(source.h0_scaler_path.parent, source.h0_run_path)
            self.assertTrue(str(source.feature_path).endswith("features.npz" if family == "residual_cnn" else "absolute_price_h8.npz"))

    def test_dynamic_scalers_and_cka_cover_all_three_populations(self):
        for task in AnalysisTask:
            data = [selected_fixture(self.root / task.value / str(w), task, w) for w in (1, 2)]
            samples = sample_splits(data)
            for dataset in data:
                branches = source_branches(dataset.source)
                full = np.concatenate([dataset.train.branches[b.key] for b in branches], axis=1)
                canonical = fit_standardizer(full, dataset.train.identity)
                for ablation in make_ablations(branches):
                    scaler = fit_standardizer(select_features(dataset.train, ablation, branches), dataset.train.identity)
                    validate_scaler_parity(scaler, canonical, ablation, branches)
                for sample in (s for s in samples if s.walk == dataset.source.walk):
                    split = dataset.train if sample.split.value == "train" else dataset.test
                    result = compute_cka(split, canonical, sample, branches)
                    self.assertEqual(result.scores.shape, (6, 6))
                    np.testing.assert_allclose(result.scores, result.scores.T)
                    np.testing.assert_allclose(result.scores.diagonal(), 1.)
        with self.assertRaises(ValueError):
            sample_splits([data[0], selected_fixture(self.root / "other", AnalysisTask.PRICE, 2)])

    def test_profile_freeze_cannot_be_changed_or_omitted(self):
        from analysis.pipeline import AnalysisPipeline
        profile = AnalysisProfile(AnalysisTask.PRICE, "lstm_capacity", "HC-AL2", "fixture")
        context = replace(self.context, profiles=(profile,))
        AnalysisPipeline(context)._freeze_policy()
        with self.assertRaises(ValueError):
            AnalysisPipeline(replace(context, profiles=(replace(profile, configuration="HC-SL2"),)))._freeze_policy()
        with self.assertRaises(ValueError):
            AnalysisPipeline(self.context)._freeze_policy()

    def test_cli_routes_profiles_to_separate_output_without_execution(self):
        from analysis.cli import main
        path = self.root / "profiles.json"
        write_json_atomic(path, {"profiles": {task.value: {"family": "temporal", "configuration": "HC-SL",
            "selection_reason": "fixture metric"} for task in AnalysisTask}})
        with patch("analysis.pipeline.AnalysisPipeline.launch", autospec=True, return_value=None) as launch:
            self.assertEqual(main(["--repo-root", str(self.root), "--profiles", str(path), "launch"]), 0)
        pipeline = launch.call_args.args[0]
        self.assertEqual(len(pipeline.context.profiles), 3)
        self.assertEqual(pipeline.context.output_root, self.root / "experiments/phase7/selected_variants")
        self.assertFalse(launch.call_args.kwargs["execute"])
        import subprocess
        self.assertEqual(subprocess.run(["git", "check-ignore", "--no-index", "-q",
            "experiments/phase7/selected_variants/manifests/preflight.json"], cwd=ROOT).returncode, 0)

    def test_selected_preparation_similarity_and_partial_report(self):
        from analysis.pipeline import AnalysisPipeline
        from analysis.replay import validate_preparation, result
        from analysis.reporting import build_report
        from analysis.types import ValidationResult
        profiles = tuple(AnalysisProfile(t, "lstm_capacity", "HC-AL2", "fixture") for t in AnalysisTask)
        context = replace(self.context, profiles=profiles)
        datasets = {(t, w): selected_fixture(self.root / t.value / str(w), t, w)
                    for t in AnalysisTask for w in (1, 2)}
        for data in datasets.values():
            full = np.concatenate(list(data.train.branches.values()), axis=1)
            save_standardizer(data.source.h0_scaler_path, fit_standardizer(full, data.train.identity))
        def locate(ctx, task, walk):
            return datasets[task, walk].source
        def load(source, ctx):
            return datasets[source.task, source.walk]
        with patch("analysis.sources.locate_source", side_effect=locate), patch("analysis.sources.load_dataset", side_effect=load), patch("analysis.replay.load_dataset", side_effect=load):
            pipeline = AnalysisPipeline(context)
            with contextlib.redirect_stderr(io.StringIO()):
                pipeline.launch(execute=False)
                self.assertTrue(validate_preparation(context, build_matrix(context)).usable)
                pipeline.similarity()
            self.assertEqual(len(list(context.output_root.rglob("branch_view.json"))), 72)
            self.assertEqual(len(list((context.output_root / "analysis/cka").rglob("scores.npz"))), 12)
            self.assertEqual(read_json(context.output_root / "manifests/matrix_seed0.json")["expected_snapshots"], 234)
            self.assertFalse((context.output_root / "downstream").exists())
            for data in datasets.values():
                for split in (data.train, data.test):
                    self.assertEqual(load_saved_similarity(context, data, split.split).scores.shape, (6, 6))
                write_json_atomic(context.output_root / "analysis/reference_replay" / data.source.task.value / f"walk{data.source.walk}.json", {"status": "invalid"})
            with patch("analysis.replay.validate_complete", return_value=ValidationResult("fixture", "invalid")), patch("analysis.sources.audit_source", return_value=result("fixture")):
                report = build_report(context, build_matrix(context), allow_partial=True)
            self.assertFalse(report.complete)
            self.assertIn("/234", (report.output_path / "summary.md").read_text())
            self.assertTrue((report.output_path / "figures/cka_absolute_price_h8_walk1_test.png").is_file())

    def test_dynamic_head_trains_and_replays_without_changing_recipe(self):
        import torch
        from analysis.runtime import collect_environment
        from analysis.training import ProbeRunner
        from analysis.replay import replay_snapshot
        torch.set_num_threads(1)
        for task in AnalysisTask:
            data = selected_fixture(self.root / task.value, task)
            context = replace(self.context, profiles=(data.source.profile,))
            full = np.concatenate(list(data.train.branches.values()), axis=1)
            save_standardizer(data.source.h0_scaler_path, fit_standardizer(full, data.train.identity))
            ablation = make_ablations(source_branches(data.source))[-3]
            spec = ProbeRunSpec(data.source, ablation, context.output_root / task.value, device="cpu")
            runner = ProbeRunner(context, collect_environment(context))
            with patch("analysis.training.require_admission"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.train(spec, data).state, RunState.EVALUATED)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(replay_snapshot(spec, data, 50, context=context).usable)
            self.assertEqual(read_json(spec.output_path / "config.json")["epochs"], 50)

    def test_completion_requires_all_task_cka_but_not_conditional_or_shap(self):
        from analysis.pipeline import AnalysisPipeline
        from analysis.replay import validate_complete
        from analysis.validation import result
        profiles = tuple(AnalysisProfile(t, "lstm_capacity", "HC-SL2", "fixture") for t in AnalysisTask)
        context = replace(self.context, profiles=profiles)
        data = {(t, w): selected_fixture(self.root / t.value / str(w), t, w, configuration="HC-SL2")
                for t in AnalysisTask for w in (1, 2)}
        def load(source, ctx):
            return data[source.task, source.walk]
        with patch("analysis.sources.locate_source", side_effect=lambda ctx, t, w: data[t, w].source), patch("analysis.sources.load_dataset", side_effect=load), patch("analysis.replay.load_dataset", side_effect=load):
            pipeline = AnalysisPipeline(context)
            pipeline.similarity()
            matrix = build_matrix(context)
            for spec in matrix.runs:
                write_json_atomic(spec.output_path / "training_complete.json", {"complete": True,
                    "snapshots": [5, 15, 50], "principal_epoch": 50})
            with patch("analysis.sources.audit_source", return_value=result("source")), patch("analysis.replay.replay_h0", return_value=result("reference")), patch("analysis.replay.replay_snapshot", return_value=result("snapshot")), contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(validate_complete(context, matrix).usable)
                self.assertFalse((context.output_root / "analysis/conditional").exists())
                path = context.output_root / "analysis/cka/absolute_price_h8/walk1/test/result.json"
                saved = read_json(path)
                saved["branches"][0] = "wrong"
                write_json_atomic(path, saved)
                self.assertFalse(validate_complete(context, matrix).usable)


class NativeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = AnalysisContext(self.root, self.root / "out")
        self.canonical = dataset_fixture(self.root / "canonical")
        source = self.canonical.source
        write_arrays_atomic(source.dataset_path, {"fixture": np.ones(1)})
        write_arrays_atomic(source.feature_path, {"fixture": np.zeros(1)})
        write_json_atomic(source.dataset_manifest_path, {"fixture": True})
        hashes = {"dataset": hash_file(source.dataset_path), "features": hash_file(source.feature_path),
                  "dataset_manifest": hash_file(source.dataset_manifest_path), "feature_manifest": "fixture"}
        self.canonical = replace(self.canonical, source_hashes=hashes)
        profile = AnalysisProfile(source.task, "lstm_capacity", "HC-AL2", "fixture")
        self.source = replace(source, feature_path=self.root / "native.npz",
                              feature_manifest_path=self.root / "native.npz.manifest.json", profile=profile)
        configurations, widths = profile_catalog(profile.family)
        rng = np.random.default_rng(23)
        self.arrays, branch_hashes = {}, {}
        for split in (self.canonical.train, self.canonical.test):
            prefix = split.split.value + "_"
            self.arrays.update({prefix + name: getattr(split.identity, name) for name in IDENTITY_FIELDS})
            branch_hashes[split.split.value] = {}
            for name, width in widths.items():
                value = split.branches.get(name)
                if value is None:
                    value = rng.normal(size=(len(split.targets), width)).astype(np.float32)
                self.arrays[prefix + name] = value.copy()
                branch_hashes[split.split.value][name] = hash_array(value)
        provenance = {}
        for name in set(widths) - set(self.canonical.train.branches):
            checkpoint = self.root / (name + ".npz")
            write_arrays_atomic(checkpoint, {"fixture": np.ones(1)})
            provenance[name] = {"path": str(checkpoint), "sha256": hash_file(checkpoint),
                                "validation": {"valid": True, "target_independent": True, "walk": 1}}
        self.manifest = {"task": source.task.value, "walk": 1, "epoch": 50,
            "target_independent_extraction": True, "checkpoint_provenance": provenance,
            "configuration_branches": {k: list(v) for k, v in configurations.items()},
            "master_branch_dims": widths,
            "configuration_dims": {k: sum(widths[b] for b in v) for k, v in configurations.items()},
            "dataset_path": str(source.dataset_path), "dataset_sha256": hashes["dataset"],
            "dataset_manifest_sha256": hashes["dataset_manifest"],
            "h0_feature_path": str(source.feature_path), "h0_feature_sha256": hashes["features"],
            "row_counts": {"train": 15, "test": 10}, "branch_hashes": branch_hashes}
        self.save_store()

    def save_store(self):
        write_arrays_atomic(self.source.feature_path, self.arrays)
        self.manifest["feature_store_sha256"] = hash_file(self.source.feature_path)
        write_json_atomic(self.source.feature_manifest_path, self.manifest)

    def load_store(self):
        from analysis.profile_sources import load_selected_dataset
        # Canonical validation has separate coverage; test only the native adapter.
        with patch("analysis.sources.locate_source", return_value=self.canonical.source), patch("analysis.sources.load_dataset", return_value=self.canonical):
            return load_selected_dataset(self.source, self.context)

    def test_native_store_joins_exact_rows_and_retains_selected_branch_order(self):
        dataset = self.load_store()
        self.assertEqual(tuple(dataset.train.branches), tuple(b.key for b in source_branches(self.source)))
        np.testing.assert_array_equal(dataset.test.targets, self.canonical.test.targets)
        self.assertEqual(dataset.source_hashes["features"], self.manifest["feature_store_sha256"])
        self.assertEqual(dataset.source_hashes["canonical_features"], self.canonical.source_hashes["features"])

    def test_native_store_rejects_identity_drift_even_with_new_file_hash(self):
        self.arrays["test_condition_ids"] = self.arrays["test_condition_ids"][::-1]
        self.save_store()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load_store()

    def test_native_store_rejects_changed_canonical_block_even_with_new_array_hash(self):
        self.arrays["test_statistical"][0, 1] += 1
        self.manifest["branch_hashes"]["test"]["statistical"] = hash_array(self.arrays["test_statistical"])
        self.save_store()
        with self.assertRaisesRegex(ValueError, "canonical feature changed"):
            self.load_store()

    def test_native_store_rejects_learned_block_hash_and_target_dependent_provenance(self):
        self.arrays["test_contrastive_lstm2"][0, 0] += 1
        self.save_store()
        with self.assertRaisesRegex(ValueError, "array hash"):
            self.load_store()
        self.manifest["target_independent_extraction"] = False
        self.save_store()
        with self.assertRaisesRegex(ValueError, "target-independent"):
            self.load_store()

    def test_native_store_rejects_encoder_checkpoint_drift(self):
        entry = self.manifest["checkpoint_provenance"]["contrastive_lstm2"]
        write_arrays_atomic(Path(entry["path"]), {"changed": np.zeros(1)})
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.load_store()


class ReferenceReplayTests(unittest.TestCase):
    def test_native_full_reference_replays_all_tasks_and_rejects_wrong_selection(self):
        import torch
        from analysis.replay import replay_h0
        from analysis.tasks import build_head
        from analysis.training import atomic_checkpoint, predict_output
        from analysis.types import AblationKind, AblationSpec
        from data_processing.phase5_walks import sha256_arrays
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in AnalysisTask:
                data = selected_fixture(root / task.value, task)
                context = AnalysisContext(ROOT, root / "outputs", profiles=(data.source.profile,))
                branches = source_branches(data.source)
                full = AblationSpec(reference_name(data.source), AblationKind.SINGLE,
                                    tuple(b.key for b in branches), sum(b.width for b in branches))
                spec = ProbeRunSpec(data.source, full, data.source.h0_run_path, device="cpu")
                train = np.concatenate(list(data.train.branches.values()), axis=1)
                scaler = fit_standardizer(train, data.train.identity)
                save_standardizer(data.source.h0_scaler_path, scaler)
                write_json_atomic(spec.output_path / "feature_standardizer.manifest.json",
                                  {"sha256": hash_file(data.source.h0_scaler_path)})
                manifest = {"dataset_sha256": data.source_hashes["dataset"],
                            "feature_sha256": data.source_hashes["features"],
                            "configuration": full.configuration, "branch_order": full.branches, "input_dim": full.width}
                for split in (data.train, data.test):
                    key = split.split.value
                    manifest[key + "_identity_hash"] = sha256_arrays(*(getattr(split.identity, f) for f in IDENTITY_FIELDS))
                    manifest[key + "_target_hash"] = hash_array(split.targets)
                write_json_atomic(spec.output_path / "dataset_manifest.json", manifest)
                config = {"configuration": full.configuration, "task": task, "walk": 1,
                          "seed": 0, "epochs": 50, "batch_size": 512, "learning_rate": 1e-4,
                          "weight_decay": 0., "hidden_dim": 128, "snapshot_epochs": [5, 15, 50]}
                write_json_atomic(spec.output_path / "config.json", config)
                write_json_atomic(spec.output_path / "architecture_manifest.json", {"branches": full.branches, "input_dim": full.width})
                write_json_atomic(spec.output_path / "training_complete.json", {"complete": True, "snapshots": [5, 15, 50], "principal_epoch": 50})
                model = build_head(spec)
                test = apply_standardizer(np.concatenate(list(data.test.branches.values()), axis=1), scaler)
                multiplier = 10000. if task == AnalysisTask.VARIANCE else 1.
                output = predict_output(model, test, torch.device("cpu"), multiplier)
                if task != AnalysisTask.CLASSIFICATION:
                    output = output.reshape(-1).astype(np.float64)
                keys = {AnalysisTask.CLASSIFICATION: ("logits", "targets"),
                        AnalysisTask.PRICE: ("prediction_future_price", "target_future_price"),
                        AnalysisTask.VARIANCE: ("prediction_realised_variance", "target_realised_variance")}[task]
                for epoch in (5, 15, 50):
                    folder = spec.output_path / f"e{epoch}"
                    atomic_checkpoint(folder / "checkpoint.pth", {"completed_epoch": epoch,
                        "dataset_sha256": data.source_hashes["dataset"], "feature_sha256": data.source_hashes["features"],
                        "model_state_dict": model.state_dict()})
                    write_arrays_atomic(folder / "predictions.npz", {keys[0]: output, keys[1]: data.test.targets,
                        **{name: getattr(data.test.identity, name) for name in IDENTITY_FIELDS}})
                    write_json_atomic(folder / "metrics.json", {"checkpoint_sha256": hash_file(folder / "checkpoint.pth"),
                        "predictions_sha256": hash_file(folder / "predictions.npz")})
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertTrue(replay_h0(data.source, context, data).usable)
                    write_json_atomic(spec.output_path / "config.json", {**config, "configuration": "wrong"})
                    self.assertFalse(replay_h0(data.source, context, data).usable)
