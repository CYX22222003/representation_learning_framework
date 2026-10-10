#!/usr/bin/env python3
"""Execute the exact YAML Python wrappers locally against a frozen bundle."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
from platform_integration.research_audit import DEPENDENCIES, require


def validate_workflow(document: dict, bundle_manifest: dict) -> list[dict]:
    require(document.get("apiVersion") == "flowmesh/v1" and document.get("kind") == "Task",
            "unexpected workflow envelope")
    nodes = document["spec"]["graph"]["nodes"]
    names = [node["name"] for node in nodes]
    require(len(names) == len(set(names)) and set(names) == set(DEPENDENCIES),
            "workflow node roster mismatch")
    seen, metrics = set(), set()
    for node in nodes:
        name, spec = node["name"], node["spec"]
        deps = node.get("dependsOn", [])
        require(deps == list(DEPENDENCIES[name]), f"{name}: workflow dependency mismatch")
        require(set(deps) <= seen, "workflow contains a cycle or is not topologically ordered")
        require(spec["taskType"] == "python" and spec["network"] == "none", "unexpected execution mode")
        require(set(spec["emits"]) == set(bundle_manifest["nodes"][name]["emits"]),
                f"{name}: workflow/bundle metrics mismatch")
        require(not metrics.intersection(spec["emits"]), "metric name collision across nodes")
        metrics.update(spec["emits"])
        compile(spec["code"], f"workflow:{name}", "exec")
        calls = [c for c in ast.walk(ast.parse(spec["code"])) if isinstance(c, ast.Call) and
                 isinstance(c.func, ast.Name) and c.func.id == "run_node"]
        require(len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == name,
                f"{name}: workflow wrapper node mismatch")
        pins = [k.value for k in calls[0].keywords if k.arg == "expected_bundle_id"]
        require(len(pins) == 1 and ast.literal_eval(pins[0]) == bundle_manifest["bundle_id"],
                f"{name}: workflow evidence pin mismatch")
        seen.add(name)
    return nodes


def run_local(workflow: Path, bundle: Path, output: Path) -> dict:
    require(not output.exists(), f"refusing to replace audit run: {output}")
    document = yaml.safe_load(workflow.read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    nodes = validate_workflow(document, manifest)
    output.mkdir(parents=True)
    prior_bundle, prior_output = os.environ.get("FYP_AUDIT_BUNDLE"), os.environ.get("FLOWMESH_OUTPUT")
    results = {}
    try:
        os.environ["FYP_AUDIT_BUNDLE"] = str(bundle.resolve())
        for node in nodes:
            name, spec = node["name"], node["spec"]
            os.environ["FLOWMESH_OUTPUT"] = str((output / name).resolve())
            namespace = {}
            exec(compile(spec["code"], f"workflow:{name}", "exec"), namespace)
            result = namespace[spec["entrypoint"]](
                **{dep: results[dep] for dep in node.get("dependsOn", [])})
            require(set(result["metrics"]) == set(spec["emits"]), f"{name}: missing declared metric")
            require(len(json.dumps(result, allow_nan=False).encode()) <= spec["pythonOutput"]["maxBytes"],
                    f"{name}: result exceeds FlowMesh return limit")
            results[name] = result
            print(f"{name}: pass ({len(result['metrics'])} metrics)", flush=True)
    finally:
        for key, value in (("FYP_AUDIT_BUNDLE", prior_bundle), ("FLOWMESH_OUTPUT", prior_output)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return results["research_synthesis"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=ROOT / "workflows/flowmesh_research_audit.yaml")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "build/flowmesh-research-evidence")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_local(args.workflow, args.bundle_dir, args.output_dir)
    print(json.dumps({"status": "pass", "nodes": 19, "bundle_id": result["output"]["bundle_id"],
                      "report": str(args.output_dir / "research_synthesis/research_report.md")}))


if __name__ == "__main__":
    main()
