"""Evidence corruption, scope isolation, graph handoff, and report contract tests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from platform_integration.research_audit import (
    AUDIT_DEPTH, DEPENDENCIES, SCHEMA, canonical_bytes, load_manifest,
    comparisons, merge_rows, node_metrics, run_node, scientific_metrics, sha256, unwrap_output,
    validate_rows, verify_files,
)
from platform_integration.research_audit_sources import DATA_PATHS, check_run_descriptor, collect_node
from run_research_audit import validate_workflow
from prepare_research_audit_bundle import restore_bundle


def example_row(protocol="original_rows", epoch=50):
    scope = {"task": "classification_h2", "walk": 1, "dataset_sha256": "a" * 64,
             "identity_hashes": {"train": "b" * 64, "test": "c" * 64},
             "train_rows": 10, "test_rows": 5, "protocol": protocol}
    return {"task": "classification_h2", "walk": 1, "method": "H0", "epoch": epoch,
            "protocol": protocol, "scope": scope,
            "scores": {"macro_f1": .45, "balanced_accuracy": .5}}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        paths = []
        for task, pattern in DATA_PATHS.items():
            for walk in (1, 2):
                relative = "sources/" + pattern.format(walk=walk)
                path = self.bundle / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"walk": walk, "preprocessing": {"volume": {"uses_evaluation_rows": False}},
                                           "artifact": {"sha256": "a" * 64},
                                           "identity_hashes": {"train": "b" * 64, "test": "c" * 64},
                                           "row_counts": {"train": 10, "test": 5}}))
                paths.append(relative)
        self.manifest = {
            "schema_version": SCHEMA, "seed": 0, "principal_epoch": 50, "audit_depth": AUDIT_DEPTH,
            "adapter_hashes": {name: sha256(ROOT / "src/platform_integration" / name)
                               for name in ("research_audit.py", "research_audit_sources.py")},
            "files": {p: {"sha256": sha256(self.bundle / p), "bytes": (self.bundle / p).stat().st_size} for p in paths},
            "nodes": {n: {"sources": sorted(paths), "depends_on": list(deps),
                          "emits": [n + "_audit_pass"]} for n, deps in DEPENDENCIES.items()},
        }
        payload, _ = collect_node("evidence_inventory", self.bundle / "sources")
        self.manifest["nodes"]["evidence_inventory"]["emits"] = sorted(node_metrics("evidence_inventory", payload))
        self.write_manifest()

    def write_manifest(self):
        self.manifest.pop("bundle_id", None)
        self.manifest["bundle_id"] = hashlib.sha256(canonical_bytes(self.manifest)).hexdigest()
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))

    def test_inventory_roundtrip_persists_artifact(self):
        destination = self.bundle / "output"
        result = run_node("evidence_inventory", bundle_dir=self.bundle, output_dir=destination)
        saved = json.loads((destination / "evidence_inventory.json").read_text())
        self.assertEqual(saved, result)
        self.assertFalse(result["output"]["audit_depth"]["training_executed"])

    def test_changed_source_hash_is_fatal(self):
        path = self.bundle / next(iter(self.manifest["files"]))
        path.write_text(path.read_text().replace('"walk": 1', '"walk": 9'))
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            run_node("evidence_inventory", bundle_dir=self.bundle)

    def test_missing_source_is_fatal(self):
        (self.bundle / next(iter(self.manifest["files"]))).unlink()
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            run_node("evidence_inventory", bundle_dir=self.bundle)

    def test_manifest_modification_without_new_identity_is_fatal(self):
        self.manifest["seed"] = 1
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_manifest(self.bundle)

    def test_workflow_pins_bundle_identity(self):
        with self.assertRaisesRegex(ValueError, "workflow/image"):
            run_node("evidence_inventory", bundle_dir=self.bundle, expected_bundle_id="wrong")

    def test_wrong_adapter_hash_is_fatal(self):
        self.manifest["adapter_hashes"]["research_audit.py"] = "f" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "adapter implementation"):
            load_manifest(self.bundle)

    def test_path_traversal_is_fatal(self):
        self.manifest["files"]["../outside.json"] = {"sha256": "a" * 64, "bytes": 1}
        with self.assertRaisesRegex(ValueError, "unsafe evidence path"):
            verify_files(self.bundle, self.manifest, ["../outside.json"])

    def test_missing_upstream_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "dependency roster"):
            run_node("data_contract", bundle_dir=self.bundle)

    def test_changed_upstream_output_is_fatal(self):
        prior = run_node("evidence_inventory", bundle_dir=self.bundle)
        prior["output"]["scopes"]["classification_h2:walk1"]["test_rows"] = 99
        with self.assertRaisesRegex(ValueError, "upstream output integrity"):
            run_node("data_contract", {"evidence_inventory": prior}, self.bundle)

    def test_mixed_bundle_upstream_is_fatal(self):
        prior = run_node("evidence_inventory", bundle_dir=self.bundle)
        prior["output"]["bundle_id"] = "other"
        with self.assertRaisesRegex(ValueError, "mismatched upstream"):
            run_node("data_contract", {"evidence_inventory": prior}, self.bundle)

    def test_evaluation_fitted_preprocessing_is_fatal(self):
        path = self.bundle / next(iter(self.manifest["files"]))
        value = json.loads(path.read_text())
        value["preprocessing"]["volume"]["uses_evaluation_rows"] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "evaluation-fitted"):
            collect_node("evidence_inventory", self.bundle / "sources")

    def test_locked_bundle_reconstructs_without_current_git_revision(self):
        destination = self.bundle / "restored"
        result = restore_bundle(self.bundle / "sources", destination, self.bundle / "manifest.json")
        self.assertEqual(result["bundle_id"], self.manifest["bundle_id"])
        verify_files(destination, result, list(result["files"]))

    def test_lock_rejects_changed_repository_evidence(self):
        path = self.bundle / next(iter(self.manifest["files"]))
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "differs from lock"):
            restore_bundle(self.bundle / "sources", self.bundle / "restored", self.bundle / "manifest.json")

    def test_existing_audit_bundle_is_not_replaced(self):
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            restore_bundle(self.bundle / "sources", self.bundle, self.bundle / "manifest.json")


class ComparisonTests(unittest.TestCase):
    def test_ta_intersection_stays_separate(self):
        merged = merge_rows([[example_row()], [example_row("ta_p2")]])
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(scientific_metrics(merged)), 4)

    def test_duplicate_result_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "duplicate result"):
            validate_rows([example_row(), example_row()])

    def test_conflicting_score_is_fatal(self):
        other = example_row()
        other["scores"]["macro_f1"] += .01
        with self.assertRaisesRegex(ValueError, "conflicting scientific"):
            merge_rows([[example_row()], [other]])

    def test_conflicting_population_is_fatal(self):
        other = example_row()
        other["scope"]["test_rows"] += 1
        with self.assertRaisesRegex(ValueError, "conflicting comparison scope"):
            merge_rows([[example_row()], [other]])

    def test_nonfinite_primary_score_is_fatal(self):
        other = example_row()
        other["scores"]["macro_f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite mandatory"):
            validate_rows([other])

    def test_earlier_snapshot_is_not_primary_metric(self):
        self.assertEqual(scientific_metrics([example_row(epoch=5)]), {})

    def test_run_identity_mismatch_is_fatal(self):
        scope = example_row()["scope"]
        manifest = {"dataset_sha256": "a" * 64, "test_identity_hash": "d" * 64}
        with self.assertRaisesRegex(ValueError, "population identity"):
            check_run_descriptor(manifest, scope)

    def test_worker_input_encodings(self):
        output = {"node": "evidence_inventory"}
        for value in (output, {"output": output}, {"output": {"output": output}},
                      SimpleNamespace(output=output)):
            self.assertEqual(unwrap_output(value), output)

    def test_comparison_keeps_ta_intersection_and_skips_training_sensitivity(self):
        h0 = example_row("ta_p2")
        candidate = copy.deepcopy(h0)
        candidate["method"] = "ta_mlp"
        candidate["scores"]["macro_f1"] += .03
        sensitivity = copy.deepcopy(candidate)
        sensitivity["protocol"] = "ta_p1u"
        pairs = comparisons([example_row(), h0, candidate, sensitivity])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["protocol"], "ta_p2")
        self.assertAlmostEqual(pairs[0]["deltas"]["macro_f1"]["candidate_minus_reference"], .03)

    def test_comparison_refuses_different_population(self):
        candidate = example_row()
        candidate["method"] = "saurl_frozen"
        candidate["scope"]["test_rows"] += 1
        with self.assertRaisesRegex(ValueError, "comparison population"):
            comparisons([example_row(), candidate])

    def test_undefined_constant_reference_is_not_fabricated(self):
        value = example_row()
        value.update(task="realised_variance", method="zero")
        value["scope"]["task"] = value["task"]
        value["scores"] = {"mae": .001, "rmse": .003, "spearman": None}
        validate_rows([value])
        self.assertNotIn("zero_original_rows_realised_variance_walk1_spearman", scientific_metrics([value]))

    def test_yaml_rejects_broken_edge(self):
        import yaml
        document = yaml.safe_load((ROOT / "workflows/flowmesh_research_audit.yaml").read_text())
        manifest = {"nodes": {n["name"]: {"emits": n["spec"]["emits"]}
                              for n in document["spec"]["graph"]["nodes"]},
                    "bundle_id": json.loads((ROOT / "workflows/research_audit_evidence_manifest.json").read_text())["bundle_id"]}
        self.assertEqual(len(validate_workflow(document, manifest)), 19)
        document = copy.deepcopy(document)
        document["spec"]["graph"]["nodes"][1]["dependsOn"] = []
        with self.assertRaisesRegex(ValueError, "workflow dependency"):
            validate_workflow(document, manifest)


if __name__ == "__main__":
    unittest.main()
