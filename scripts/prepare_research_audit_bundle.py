#!/usr/bin/env python3
"""Freeze a bounded report-only bundle; never train or modify experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from platform_integration.research_audit import (
    AUDIT_DEPTH, DEPENDENCIES, SCHEMA, canonical_bytes, merge_rows,
    node_metrics, require, safe_path, sha256, validate_manifest, validate_rows,
)
from platform_integration.research_audit_sources import collect_node


def prepare_bundle(repository: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace frozen bundle: {destination}")
    payloads, node_sources = {}, {}
    for node in DEPENDENCIES:
        payload, sources = collect_node(node, repository)
        validate_rows(payload["rows"])
        payloads[node], node_sources[node] = payload, sources
    combined = merge_rows([p["rows"] for p in payloads.values()])
    validate_rows(combined)
    manifest = {
        "schema_version": SCHEMA, "seed": 0, "principal_epoch": 50,
        "audit_depth": AUDIT_DEPTH,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "source_worktree_modified": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True).strip()),
        "adapter_hashes": {name: sha256(repository / "src/platform_integration" / name)
                           for name in ("research_audit.py", "research_audit_sources.py")},
        "nodes": {}, "files": {},
    }
    for node, payload in payloads.items():
        metric_payload = {**payload, "rows": combined} if node == "research_synthesis" else payload
        manifest["nodes"][node] = {
            "depends_on": list(DEPENDENCIES[node]),
            "sources": ["sources/" + p for p in node_sources[node]],
            "emits": sorted(node_metrics(node, metric_payload)),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".research-audit-", dir=destination.parent))
    try:
        for relative in sorted(set(p for ps in node_sources.values() for p in ps)):
            source = repository / relative
            target = temporary / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["files"]["sources/" + relative] = {
                "sha256": sha256(target), "bytes": target.stat().st_size,
            }
        if sum(v["bytes"] for v in manifest["files"].values()) > 32 * 1024 * 1024:
            raise ValueError("Evidence exceeds the 32 MiB report-only bundle limit")
        manifest["bundle_id"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def restore_bundle(repository: Path, destination: Path, frozen_manifest: Path) -> dict:
    """Materialize the exact locked report bundle, even after HEAD advances."""
    require(not destination.exists(), f"refusing to replace frozen bundle: {destination}")
    manifest = json.loads(frozen_manifest.read_text())
    validate_manifest(manifest)
    require(sum(v["bytes"] for v in manifest["files"].values()) <= 32 * 1024 * 1024,
            "frozen evidence exceeds bundle size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".research-audit-", dir=destination.parent))
    try:
        for relative, info in manifest["files"].items():
            require(relative.startswith("sources/"), "invalid frozen source prefix")
            source = safe_path(repository, relative[len("sources/"):])
            require(source.is_file() and source.stat().st_size == info["bytes"] and sha256(source) == info["sha256"],
                    f"repository evidence differs from lock: {relative}")
            target = safe_path(temporary, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/flowmesh-research-evidence")
    parser.add_argument("--frozen-manifest", type=Path, help="Restore an existing locked evidence version")
    parser.add_argument("--manifest-output", type=Path, help="Save a new reviewable evidence lock alongside the YAML")
    args = parser.parse_args()
    if args.manifest_output and args.manifest_output.exists():
        parser.error("Refusing to replace an existing evidence lock; choose a new filename")
    if args.frozen_manifest:
        manifest = restore_bundle(args.repository_root.resolve(), args.output_dir.resolve(), args.frozen_manifest)
    else:
        manifest = prepare_bundle(args.repository_root.resolve(), args.output_dir.resolve())
    if args.manifest_output:
        args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bundle_id": manifest["bundle_id"], "nodes": len(manifest["nodes"]),
                      "files": len(manifest["files"]),
                      "bytes": sum(x["bytes"] for x in manifest["files"].values()),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
