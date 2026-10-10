"""Offline, CPU-only research evidence DAG. No model fitting or inference.

Workers verify packaged evidence and manifest descriptors. Checkpoint/source
replay records describe earlier validation, never fresh numerical replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "research-audit-v1"
TASKS = ("classification_h2", "absolute_price_h8", "realised_variance")
DEPENDENCIES = {
    "evidence_inventory": (),
    "data_contract": ("evidence_inventory",),
    "canonical_encoders": ("data_contract",),
    "canonical_features": ("canonical_encoders", "data_contract"),
    "classification": ("canonical_features", "data_contract"),
    "future_price": ("canonical_features", "data_contract"),
    "realised_variance": ("canonical_features", "data_contract"),
    "temporal_variants": ("classification", "future_price", "realised_variance"),
    "lstm_capacity": ("temporal_variants",),
    "residual_cnn": ("future_price", "temporal_variants"),
    "saurl_frozen": ("classification", "future_price", "realised_variance"),
    "lwa_frozen": ("classification", "future_price", "realised_variance"),
    "timedart_frozen": ("classification", "future_price", "realised_variance"),
    "raw_baselines": ("classification", "future_price", "realised_variance"),
    "ta_mlp": ("classification", "raw_baselines"),
    "sgn_c": ("classification", "raw_baselines"),
    "xm_c8": ("future_price", "raw_baselines"),
    "garch_lstm": ("realised_variance", "raw_baselines"),
    "research_synthesis": (
        "temporal_variants", "lstm_capacity", "residual_cnn", "saurl_frozen",
        "lwa_frozen", "timedart_frozen", "raw_baselines", "ta_mlp", "sgn_c",
        "xm_c8", "garch_lstm",
    ),
}
PRIMARY = {
    "classification_h2": {"macro_f1": "max", "balanced_accuracy": "max"},
    "absolute_price_h8": {"mae": "min", "rmse": "min", "rank_ic": "max"},
    "realised_variance": {"mae": "min", "rmse": "min", "spearman": "max"},
}
AUDIT_DEPTH = {
    "executed": ["packaged file hashes", "schema and expected coverage",
                 "finite required metrics", "manifest descriptor consistency",
                 "upstream bundle and output consistency"],
    "recorded_only": ["raw source replay", "feature-array replay",
                      "checkpoint/prediction replay"],
    "training_executed": False,
    "inference_executed": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "empty evidence path")
    part = Path(relative)
    require(not part.is_absolute() and ".." not in part.parts, "unsafe evidence path")
    result = (root / part).resolve()
    require(result.is_relative_to(root.resolve()), "evidence path escapes bundle")
    return result


def finite(value: Any, field: str) -> float:
    require(not isinstance(value, bool), f"boolean scientific metric: {field}")
    result = float(value)
    require(math.isfinite(result), f"non-finite mandatory metric: {field}")
    return result


def portable(value: Any) -> Any:
    """Keep undefined diagnostics as null, never fabricated zero scores."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_name(*parts: Any) -> str:
    return "_".join(re.sub(r"[^a-z0-9]+", "_", str(p).lower()).strip("_")
                    for p in parts)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen = set()
    for row in rows:
        key = (row["task"], row["walk"], row["method"], row["protocol"], row["epoch"])
        require(key not in seen, f"duplicate result identity: {key}")
        seen.add(key)
        require(row["walk"] in (1, 2) and row["epoch"] in (5, 15, 50),
                f"unexpected walk or snapshot: {key}")
        require(row["scope"]["task"] == row["task"] and
                row["scope"]["walk"] == row["walk"], "row scope mismatch")
        for name in PRIMARY[row["task"]]:
            # Constant references have undefined correlations; older price
            # snapshots may lack the subsequently added Rank IC diagnostic.
            if name == "rank_ic" and name not in row["scores"]:
                continue
            if name == "spearman" and row["method"] in ("zero", "training_median") and row["scores"].get(name) is None:
                continue
            finite(row["scores"][name], f"{key}.{name}")


def scientific_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for row in rows:
        if row["epoch"] != 50:
            continue
        for name, value in row["scores"].items():
            if value is None:
                continue
            key = metric_name(row["method"], row["protocol"], row["task"],
                              f"walk{row['walk']}", name)
            require(key not in result, f"duplicate emitted metric: {key}")
            result[key] = finite(value, key)
    return result


def node_metrics(node: str, payload: dict) -> dict[str, float]:
    """Declare finite principal scores plus recorded health/resource diagnostics."""
    result = {"audit_pass": 1.0, "result_rows": float(len(payload["rows"])),
              **scientific_metrics(payload["rows"])}
    diagnostics = payload["diagnostics"]

    def add(key: str, value: Any) -> None:
        require(key not in result, f"duplicate diagnostic metric: {key}")
        result[key] = finite(value, key)

    if node in ("evidence_inventory", "data_contract"):
        for scope in payload["scopes"].values():
            for split in ("train", "test"):
                add(metric_name(scope["task"], f"walk{scope['walk']}", f"{split}_rows"), scope[f"{split}_rows"])
    if node == "canonical_encoders":
        for item in diagnostics["runs"]:
            prefix = metric_name(item["encoder"], f"walk{item['walk']}")
            for key in ("epoch50_train_loss", "epoch50_embedding_std"):
                add(prefix + "_" + key, item[key])
            add(prefix + "_collapse_warning", int(item["collapse_warning"]))
    if node == "canonical_features":
        for walk in (1, 2):
            add(f"walk{walk}_concat_dim", diagnostics[f"walk{walk}"]["concat_dim"])
    for diagnostic in diagnostics.get("cka", []):
        for item in diagnostic["rows"]:
            add(metric_name(item.get("variant", item.get("candidate")), item.get("reference", "family_cnn"),
                            f"walk{diagnostic['walk']}", "linear_cka"), item["cka"])
    resource_fields = ("parameter_count", "trainable_parameter_count", "elapsed_training_seconds",
                       "inference_seconds", "peak_cuda_memory_bytes", "train_feature_extraction_seconds",
                       "test_feature_extraction_seconds")
    for kind in ("encoder_resources", "downstream_resources"):
        for item in diagnostics.get(kind, []):
            prefix = metric_name(kind, item.get("task", "encoder"),
                                 item.get("configuration", item.get("variant")), f"walk{item['walk']}")
            for key in resource_fields:
                if item.get(key) not in (None, ""):
                    add(prefix + "_" + key, item[key])
    for walk in (1, 2):
        feature = diagnostics.get(f"features_walk{walk}")
        if feature:
            add(f"walk{walk}_native_feature_dim", feature["output_dim"])
        replay = diagnostics.get(f"encoder_replay_walk{walk}")
        if replay:
            add(f"walk{walk}_recorded_replay_warnings", replay["warning_count"])
    return {f"{node}_{k}": v for k, v in result.items()}


def comparisons(rows: list[dict]) -> list[dict]:
    """Epoch-50 deltas within one task, walk and exact saved population."""
    principal = [r for r in rows if r["epoch"] == 50]
    index = {(r["task"], r["walk"], r["method"], r["protocol"]): r for r in principal}
    nonlearned = {"always_stable_reference", "repeated_training_prior_reference",
                  "persistence", "zero", "training_median", "historical_persistence"}
    predecessors = {"HC-SL2": "HC-SL", "HB-SL2": "HB-SL", "HC-AL2": "HC-AL", "HB-AL2": "HB-AL"}
    result = []
    for candidate in principal:
        method = candidate["method"]
        if method == "H0" or method in nonlearned or candidate["protocol"] == "ta_p1u":
            continue
        references = ["H0"]
        if method in predecessors:
            references.append(predecessors[method])
        if method.startswith(("HC-A", "HB-A")):
            references.append(method[:2] + "-DC")
        if method in ("sgn_c", "xm_c8", "garch_lstm"):
            references += ["raw_ohlcv_mlp", "raw_ohlcv_lstm"]
        if method == "xm_c8":
            references.append("persistence")
        for name in references:
            reference = index.get((candidate["task"], candidate["walk"], name, candidate["protocol"]))
            require(reference is not None, f"missing matched comparison reference: {method}/{name}")
            require(reference["scope"] == candidate["scope"], "comparison population mismatch")
            deltas = {}
            for metric, direction in PRIMARY[candidate["task"]].items():
                a, b = candidate["scores"].get(metric), reference["scores"].get(metric)
                if a is None or b is None:
                    continue
                delta = finite(a, metric) - finite(b, metric)
                deltas[metric] = {"candidate_minus_reference": delta, "better_direction": direction}
            result.append({"task": candidate["task"], "walk": candidate["walk"],
                           "protocol": candidate["protocol"], "candidate": method,
                           "reference": name, "deltas": deltas})
    return result


def merge_rows(groups: list[list[dict]]) -> list[dict]:
    result = {}
    for group in groups:
        for row in group:
            key = (row["task"], row["walk"], row["method"], row["protocol"], row["epoch"])
            if key in result:
                prior = result[key]
                require(prior["scope"] == row["scope"], f"conflicting comparison scope: {key}")
                for metric in prior["scores"].keys() & row["scores"].keys():
                    if prior["scores"][metric] is None or row["scores"][metric] is None:
                        require(prior["scores"][metric] is row["scores"][metric], f"conflicting undefined score: {key}.{metric}")
                        continue
                    require(math.isclose(prior["scores"][metric], row["scores"][metric],
                                         abs_tol=1e-12, rel_tol=0), f"conflicting scientific score: {key}.{metric}")
                prior["scores"].update(row["scores"])
            else:
                result[key] = {**row, "scores": dict(row["scores"])}
    return [result[key] for key in sorted(result)]


def validate_manifest(manifest: dict) -> str:
    require(manifest["schema_version"] == SCHEMA, "unsupported bundle schema")
    require(set(manifest["nodes"]) == set(DEPENDENCIES), "bundle node roster mismatch")
    claimed = manifest["bundle_id"]
    identity = {k: v for k, v in manifest.items() if k != "bundle_id"}
    require(hashlib.sha256(canonical_bytes(identity)).hexdigest() == claimed,
            "bundle manifest identity mismatch")
    require(manifest["seed"] == 0 and manifest["principal_epoch"] == 50,
            "unexpected experiment lifecycle")
    require(manifest["audit_depth"] == AUDIT_DEPTH, "audit-depth contract mismatch")
    require(set(manifest["adapter_hashes"]) == {"research_audit.py", "research_audit_sources.py"},
            "incomplete adapter identity contract")
    for filename, digest in manifest["adapter_hashes"].items():
        require(filename in ("research_audit.py", "research_audit_sources.py"), "unknown adapter")
        require(sha256(Path(__file__).with_name(filename)) == digest,
                f"adapter implementation differs from frozen bundle: {filename}")
    return claimed


def load_manifest(bundle: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    return manifest, validate_manifest(manifest)


def verify_files(bundle: Path, manifest: dict, paths: list[str]) -> None:
    require(len(paths) == len(set(paths)), "duplicate evidence paths")
    for relative in paths:
        info = manifest["files"][relative]
        path = safe_path(bundle, relative)
        require(path.is_file() and path.stat().st_size == info["bytes"] and
                sha256(path) == info["sha256"], f"evidence integrity failure: {relative}")


def unwrap_output(value: Any) -> dict:
    for _ in range(4):
        if isinstance(value, dict) and "node" in value:
            return value
        if isinstance(value, dict) and "output" in value:
            value = value["output"]
        elif hasattr(value, "output"):
            value = value.output
        else:
            break
    raise ValueError("unrecognized connected node output")


def result_digest(output: dict) -> str:
    return hashlib.sha256(canonical_bytes(
        {k: v for k, v in output.items() if k not in ("result_sha256", "artifact_files")}
    )).hexdigest()


def run_node(node: str, inputs: dict | None = None,
             bundle_dir: str | Path | None = None,
             output_dir: str | Path | None = None,
             expected_bundle_id: str | None = None) -> dict:
    require(node in DEPENDENCIES, f"unknown audit node: {node}")
    bundle = Path(bundle_dir or os.environ.get("FYP_AUDIT_BUNDLE", "/app/evidence"))
    manifest, bundle_id = load_manifest(bundle)
    if expected_bundle_id is not None:
        require(bundle_id == expected_bundle_id, "workflow/image evidence bundle mismatch")
    expected_inputs = set(DEPENDENCIES[node])
    require(set(inputs or {}) == expected_inputs, f"{node}: dependency roster mismatch")
    upstream = {k: unwrap_output(v) for k, v in (inputs or {}).items()}
    for name, value in upstream.items():
        require(value["node"] == name and value["bundle_id"] == bundle_id and
                value["status"] == "pass", f"{node}: failed or mismatched upstream {name}")
        require(value["result_sha256"] == result_digest(value),
                f"{node}: upstream output integrity failure: {name}")

    descriptor = manifest["nodes"][node]
    require(descriptor["depends_on"] == list(DEPENDENCIES[node]), "frozen dependency roster mismatch")
    paths = list(manifest["files"]) if node == "evidence_inventory" else descriptor["sources"]
    verify_files(bundle, manifest, paths)
    from .research_audit_sources import collect_node
    payload, actual_sources = collect_node(node, bundle / "sources")
    require(sorted("sources/" + p for p in actual_sources) == sorted(descriptor["sources"]),
            f"{node}: source inventory changed")
    validate_rows(payload.get("rows", []))
    merge_rows([payload.get("rows", [])] + [v["results"].get("rows", []) for v in upstream.values()])
    for value in upstream.values():
        require(value["scopes"] == payload["scopes"], f"{node}: upstream population mismatch")

    output = {
        "schema_version": SCHEMA, "node": node, "status": "pass",
        "bundle_id": bundle_id, "seed": 0, "principal_epoch": 50,
        "audit_depth": AUDIT_DEPTH, "scopes": payload["scopes"],
        "results": payload, "sources": descriptor["sources"],
        "upstream_receipts": {k: v["result_sha256"] for k, v in upstream.items()},
    }
    lineage = {}
    for name, value in upstream.items():
        for ancestor, digest in {**value["lineage"], name: value["result_sha256"]}.items():
            require(ancestor not in lineage or lineage[ancestor] == digest, "inconsistent audit lineage")
            lineage[ancestor] = digest
    output["lineage"] = lineage
    # Preserve canonical snapshots and non-learned references through joins,
    # without pretending every node performed its ancestors' checks again.
    output["context_rows"] = merge_rows([payload["rows"]] +
                                        [v["context_rows"] for v in upstream.values()])
    report = None
    if node == "research_synthesis":
        require(set(lineage) == set(DEPENDENCIES) - {node}, "incomplete final audit lineage")
        output["results"]["rows"] = output["context_rows"]
        validate_rows(output["results"]["rows"])
        output["results"]["comparisons"] = comparisons(output["results"]["rows"])
        report = render_report(output)
    metrics = node_metrics(node, output["results"])
    require(set(metrics) == set(descriptor["emits"]), f"{node}: emitted metric roster mismatch")
    output["result_sha256"] = result_digest(output)
    destination = output_dir or os.environ.get("FLOWMESH_OUTPUT")
    output["artifact_files"] = []
    if destination:
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        output["artifact_files"].append(f"{node}.json")
        if report is not None:
            output["artifact_files"].append("research_report.md")
            (directory / "research_report.md").write_text(report, encoding="utf-8")
        (directory / f"{node}.json").write_text(
            json.dumps({"output": output, "metrics": metrics}, indent=2,
                       sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"output": output, "metrics": metrics}


def render_report(output: dict) -> str:
    lines = ["# Research evidence audit", "", f"Bundle: `{output['bundle_id']}`", "",
             "Seed 0; epoch 50 primary; both global-calendar walks. No training or inference ran.",
             "Packaged hashes and manifest descriptors were checked. Source, feature-array, and",
             "checkpoint replay are previously recorded evidence, not fresh numerical replay.",
             "Undefined diagnostic correlations are null. Repeated audit runs add no independent evidence.", ""]
    for task in TASKS:
        lines += [f"## {task}", "", "| Walk | Method | Protocol | Evaluation rows | Scores |",
                  "|---|---|---|---:|---|"]
        for row in output["results"]["rows"]:
            if row["task"] == task and row["epoch"] == 50:
                scores = "; ".join(f"{k}={v:.8g}" for k, v in row["scores"].items() if v is not None)
                lines.append(f"| {row['walk']} | {row['method']} | {row['protocol']} | "
                             f"{row['scope']['test_rows']} | {scores} |")
        lines.append("")
    lines += ["## Matched principal differences", "",
              "Candidate minus reference: negative MAE/RMSE is better; positive F1/accuracy/Rank IC/Spearman is better.",
              "These are descriptive seed-0 differences, without significance claims or cross-task ranking.", "",
              "| Task | Walk | Protocol | Candidate | Reference | Differences |",
              "|---|---|---|---|---|---|"]
    for pair in output["results"]["comparisons"]:
        delta = "; ".join(f"{k}={v['candidate_minus_reference']:+.8g}" for k, v in pair["deltas"].items())
        lines.append(f"| {pair['task']} | {pair['walk']} | {pair['protocol']} | {pair['candidate']} | {pair['reference']} | {delta} |")
    lines.append("")
    lines += ["## Interpretation boundaries", "",
              "Frozen-representation and task-specific complete-system comparisons are distinct.",
              "TA P2 uses its own common intersection; P1U is a training-only sensitivity.",
              "Compare variants with their precommitted H0/predecessor/duplicate controls.",
              "Do not rank across tasks, select checkpoints from evaluation, or infer profitable alpha.",
              "Optional/deferred phases are outside this audit's executed scope.", "",
              "## Audit sources", ""]
    for name in sorted(output["upstream_receipts"]):
        lines.append(f"- `{name}`: `{output['upstream_receipts'][name]}`")
    return "\n".join(lines) + "\n"
