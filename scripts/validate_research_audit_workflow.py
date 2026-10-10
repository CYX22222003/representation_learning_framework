#!/usr/bin/env python3
"""Validate locally; optionally request a platform dry-run (never dispatch)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
from platform_integration.research_audit import load_manifest, require, verify_files
from run_research_audit import validate_workflow


def platform_validate(workflow: str, site: str, token: str) -> dict:
    require(site in ("home", "office", "cloud"), "unknown Lumid site")
    request = urllib.request.Request(
        f"https://lum.id/fm/{site}/api/v1/workflows/validate",
        data=workflow.encode(), method="POST",
        headers={"Content-Type": "text/plain", "Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Do not print request headers or credentials.
        detail = exc.read(4096).decode("utf-8", errors="replace").replace(token, "[redacted]")
        raise RuntimeError(f"Lumid validation refused: HTTP {exc.code}: {detail}") from None
    require(isinstance(result, dict) and (result.get("valid") is True or result.get("ok") is True),
            f"Lumid did not confirm validity: {result}")
    if "tasks" in result:
        nodes = yaml.safe_load(workflow)["spec"]["graph"]["nodes"]
        expected = {n["name"]: n.get("dependsOn", []) for n in nodes}
        tasks = result["tasks"]
        names = [t["graph_node_name"] for t in tasks]
        require(result["count"] == len(expected) and len(names) == len(set(names)) and
                set(names) == set(expected), "platform compiled a different node roster")
        identifiers = {t["graph_node_name"]: t["task_id"] for t in tasks}
        for task in tasks:
            require(set(task["depends_on"]) == {identifiers[n] for n in expected[task["graph_node_name"]]},
                    "platform compiled different dependencies")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=ROOT / "workflows/flowmesh_research_audit.yaml")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "build/flowmesh-research-evidence")
    parser.add_argument("--smoke", action="store_true", help="Validate the two-node handoff smoke definition")
    parser.add_argument("--platform", action="store_true", help="Call the validation endpoint; does not submit jobs")
    parser.add_argument("--site", choices=("home", "office", "cloud"), default="home")
    args = parser.parse_args()
    content = args.workflow.read_text()
    document = yaml.safe_load(content)
    if args.smoke:
        require(document["apiVersion"] == "flowmesh/v1", "unexpected smoke dialect")
        nodes = document["spec"]["graph"]["nodes"]
        require([n["name"] for n in nodes] == ["prepare", "verify"] and
                nodes[1]["dependsOn"] == ["prepare"], "smoke graph topology mismatch")
        for node in nodes:
            compile(node["spec"]["code"], node["name"], "exec")
    else:
        manifest, bundle_id = load_manifest(args.bundle_dir)
        verify_files(args.bundle_dir, manifest, list(manifest["files"]))
        nodes = validate_workflow(document, manifest)
    result = {"local_validation": "pass", "nodes": len(nodes), "platform_validation": "not_requested"}
    if args.platform:
        token = os.environ.get("LUMID_PAT")
        require(bool(token), "LUMID_PAT is required for platform validation")
        result["platform_validation"] = platform_validate(content, args.site, token)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
