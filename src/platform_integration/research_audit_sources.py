"""Normalize existing JSON/CSV reports using only the Python standard library."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .research_audit import PRIMARY, TASKS, finite, portable, require, safe_path, sha256

DATA_PATHS = {
    "classification_h2": "experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz.manifest.json",
    "absolute_price_h8": "experiments/phase5/downstream_addons/shared/h8/data/walk{walk}/market_1h_seq64_h8.npz.manifest.json",
    "realised_variance": "experiments/phase6/volatility_prediction/data_preparation/walk{walk}/volatility_1h_seq64_h8.npz.manifest.json",
}
REPORTS = {
    "temporal_variants": "experiments/phase6/encoder_variants/reports/complete_seed0",
    "lstm_capacity": "experiments/phase6_5/lstm_capacity/reports/complete_seed0",
    "residual_cnn": "experiments/phase6_5/residual_cnn/reports/complete_seed0",
}
CONFIGURATIONS = {
    "temporal_variants": ("H0", "HC-SL", "HC-ST", "HB-SL", "HB-ST", "HC-AL",
                          "HC-AT", "HB-AL", "HB-AT", "HC-DC", "HB-DC"),
    "lstm_capacity": ("H0", "HC-SL", "HB-SL", "HC-AL", "HB-AL", "HC-DC", "HB-DC",
                      "HC-SL2", "HB-SL2", "HC-AL2", "HB-AL2"),
    "residual_cnn": ("H0", "HC-DC", "HB-DC", "HC-SR", "HB-SR", "HC-AR", "HB-AR",
                     "raw_ohlcv_lstm", "raw_ohlcv_mlp"),
}
BRANCH_DIMS = {"statistical": 70, "transformed": 55, "vae": 64,
               "contrastive": 128, "byol": 128}


class Reader:
    def __init__(self, root: Path):
        self.root = root
        self.sources: set[str] = set()

    def path(self, relative: str) -> Path:
        result = safe_path(self.root, relative)
        require(result.is_file(), f"missing required evidence: {relative}")
        require(result.suffix in (".json", ".csv", ".md"), "non-report evidence is forbidden")
        require(result.stat().st_size <= 16 * 1024 * 1024, "unbounded evidence file")
        self.sources.add(relative)
        return result

    def json(self, relative: str) -> dict:
        value = json.loads(self.path(relative).read_text(encoding="utf-8"))
        require(isinstance(value, dict), f"expected object: {relative}")
        return value

    def csv(self, relative: str) -> list[dict]:
        with self.path(relative).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def report(self, relative: str) -> str:
        return self.path(relative).read_text(encoding="utf-8")


def scopes(reader: Reader) -> tuple[dict, dict]:
    result, manifests = {}, {}
    for task in TASKS:
        for walk in (1, 2):
            m = reader.json(DATA_PATHS[task].format(walk=walk))
            require(m["walk"] == walk, "dataset manifest walk mismatch")
            require(m["preprocessing"]["volume"]["uses_evaluation_rows"] is False,
                    "evaluation-fitted volume preprocessing")
            result[f"{task}:walk{walk}"] = {
                "task": task, "walk": walk, "protocol": "original_rows",
                "dataset_sha256": m["artifact"]["sha256"],
                "identity_hashes": {k: m["identity_hashes"][k] for k in ("train", "test")},
                "train_rows": m["row_counts"]["train"], "test_rows": m["row_counts"]["test"],
                "target_definition": {"classification_h2": "h2_tau0.001_DOWN_STABLE_UP",
                                      "absolute_price_h8": "close[t+8h]",
                                      "realised_variance": "sum_j1..8_squared_raw_probability_changes"}[task],
            }
            manifests[f"{task}:walk{walk}"] = m
    return result, manifests


def check_run_descriptor(m: dict, scope: dict) -> None:
    """Check declared provenance only; does not read absent prediction arrays."""
    require(m.get("walk", scope["walk"]) == scope["walk"], "run walk mismatch")
    for key in ("evaluation_used_for_selection", "evaluation_used_for_fitting_or_selection",
                "evaluation_resampled", "evaluation_updates_garch_state"):
        require(m.get(key, False) is False, f"invalid evaluation policy: {key}")
    hashes = [m[k] for k in ("dataset_sha256", "label_sha256", "store_sha256") if k in m]
    require(bool(hashes) and all(h == scope["dataset_sha256"] for h in hashes),
            "run dataset provenance mismatch")
    identity = m.get("identity_hashes", {})
    for split in ("train", "test"):
        aliases = [f"{split}_identity_hash", f"{split}_identity_sha256"]
        if split == "test":
            aliases += ["evaluation_identity_sha256"]
        values = [m[k] for k in aliases if k in m]
        if split in identity:
            values.append(identity[split])
        require(all(x == scope["identity_hashes"][split] for x in values),
                f"run {split} population identity mismatch")
        count = m.get(f"{split}_rows", m.get("evaluation_rows") if split == "test" else None)
        if count is not None:
            require(count == scope[f"{split}_rows"], f"run {split} count mismatch")


def check_completion(reader: Reader, run: str) -> dict:
    marker = reader.json(run + "/training_complete.json")
    principal = marker.get("principal_epoch")
    if principal is None:
        require("/e50/" in marker.get("principal_checkpoint", ""),
                f"wrong principal checkpoint: {run}")
        principal = 50
    require(principal == 50, f"wrong principal epoch: {run}")
    require(marker.get("complete", marker.get("replay_valid")) is True,
            f"incomplete recorded trajectory: {run}")
    require(marker.get("snapshots", [5, 15, 50]) == [5, 15, 50], "snapshot contract mismatch")
    return marker


def score_group(task: str, payload: dict) -> dict:
    if task == "classification_h2":
        groups = payload.get("metrics", payload)
        values = next((groups[k] for k in ("framework", "baseline", "sgn_c", "classification")
                       if isinstance(groups.get(k), dict)), groups)
        return {k: finite(values[k], k) for k in ("macro_f1", "balanced_accuracy")}
    if task == "absolute_price_h8":
        groups = payload.get("metrics", payload)
        value = groups.get("price", groups)
        values = value.get("overall", value)
        result = {k: finite(values[k], k) for k in ("mae", "rmse")}
        rank = groups.get("timestamp_cross_sectional_rank_ic",
                          groups.get("implied_movement_cross_sectional_rank_ic"))
        if rank and rank.get("mean") is not None:
            result["rank_ic"] = finite(rank["mean"], "rank_ic")
        movement = groups.get("implied_movement", {})
        movement = movement.get("overall", movement)
        if movement.get("spearman") is not None:
            result["movement_spearman"] = finite(movement["spearman"], "movement_spearman")
        return result
    groups = payload.get("metrics", payload)
    values = next((groups[k] for k in ("model_metrics", "model", "stack")
                   if isinstance(groups.get(k), dict)), groups)
    return {k: finite(values[k], k) for k in ("mae", "rmse", "spearman")}


def row(task: str, walk: int, method: str, scope: dict, scores: dict,
        epoch: int = 50, protocol: str = "original_rows") -> dict:
    return {"task": task, "walk": walk, "method": method, "protocol": protocol,
            "epoch": epoch, "scope": scope, "scores": scores}


def table_rows(reader: Reader, node: str, all_scopes: dict) -> list[dict]:
    values = reader.csv(REPORTS[node] + "/epoch50_by_walk.csv")
    tasks = ("absolute_price_h8",) if node == "residual_cnn" else TASKS
    expected = {(t, w, c) for t in tasks for w in (1, 2) for c in CONFIGURATIONS[node]}
    observed = [(v["task"], int(v["walk"]), v["configuration"]) for v in values]
    require(len(observed) == len(set(observed)) and set(observed) == expected,
            f"{node}: incomplete, duplicated or unexpected matrix")
    result = []
    for value in values:
        task, walk = value["task"], int(value["walk"])
        mapping = {"macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy"}
        if task == "absolute_price_h8":
            mapping = {"mae": "price_mae", "rmse": "price_rmse", "rank_ic": "cross_sectional_rank_ic",
                       "movement_spearman": "implied_spearman"}
        elif task == "realised_variance":
            mapping = {"mae": "mae", "rmse": "rmse", "spearman": "spearman"}
        scores = {k: finite(value[v], v) for k, v in mapping.items()}
        result.append(row(task, walk, value["configuration"],
                          all_scopes[f"{task}:walk{walk}"], scores))
    return result


def run_evidence(reader: Reader, run: str, task: str, walk: int, method: str,
                 scope: dict, protocol: str = "original_rows") -> tuple[list[dict], dict]:
    check_run_descriptor(reader.json(run + "/dataset_manifest.json"), scope)
    marker = check_completion(reader, run)
    snapshots, detail = [], {"completion_record": marker, "snapshots": {}}
    if any(fragment in run for fragment in ("/encoder_variants/downstream/", "/lstm_capacity/downstream/",
                                             "/residual_cnn/downstream/", "/phase6_7/downstream/",
                                             "/ta_mlp/downstream/h0/", "/ta_mlp/downstream/ta_mlp/")):
        scaler = reader.json(run + "/feature_standardizer.manifest.json")
        flags = [scaler[k] for k in ("evaluation_used", "evaluation_used_for_fit") if k in scaler]
        require(bool(flags) and all(flag is False for flag in flags), "evaluation-fitted feature scaler")
        detail["recorded_train_only_scaler"] = portable(scaler)
    for epoch in (5, 15, 50):
        payload = reader.json(f"{run}/e{epoch}/metrics.json")
        require(payload.get("epoch", payload.get("completed_epoch", epoch)) == epoch,
                "snapshot epoch mismatch")
        snapshots.append(row(task, walk, method, scope, score_group(task, payload), epoch, protocol))
        fields = ("checkpoint_sha256", "predictions_sha256", "elapsed_training_seconds",
                  "inference_seconds", "parameter_count", "peak_cuda_memory_bytes",
                  "hard_group_ids", "temperature", "outside_probability_range_fraction",
                  "persistence_relative_skill", "clipping", "meta_model", "unit_contract")
        detail["snapshots"][str(epoch)] = portable({k: payload[k] for k in fields if k in payload})
        detail["snapshots"][str(epoch)]["scores"] = snapshots[-1]["scores"]
        if epoch == 50:
            # Rich classifications and non-learned reference tables survive in
            # the audit artifact, without returning every per-contract prediction.
            for k in ("framework", "baseline", "sgn_c", "classification", "references",
                      "always_stable_reference", "persistence_reference"):
                if k in payload:
                    detail[k] = portable(payload[k])
    return snapshots, detail


def reference_rows(reader: Reader, run: str, task: str, walk: int, scope: dict) -> list[dict]:
    """Non-learned references have no selectable training trajectory."""
    payload = reader.json(run + "/e50/metrics.json")
    result = []
    if task == "classification_h2":
        for key in ("always_stable_reference", "repeated_training_prior_reference"):
            result.append(row(task, walk, key, scope, score_group(task, payload[key])))
    elif task == "absolute_price_h8":
        values = payload["persistence_reference"]["overall"]
        result.append(row(task, walk, "persistence", scope,
                          {k: finite(values[k], k) for k in ("mae", "rmse")}))
    else:
        for name, values in payload["references"].items():
            scores = {k: finite(values[k], k) for k in ("mae", "rmse")}
            scores["spearman"] = portable(values.get("spearman"))
            result.append(row(task, walk, name, scope, scores))
    return result


def variants(reader: Reader, node: str, all_scopes: dict) -> dict:
    rows = table_rows(reader, node, all_scopes)
    directory = REPORTS[node]
    diagnostics = {"paired_differences": reader.csv(directory + "/paired_differences.csv"),
                   "encoder_resources": reader.csv(directory + "/encoder_resources.csv"),
                   "downstream_resources": reader.csv(directory + (
                       "/resources.csv" if node == "temporal_variants" else "/downstream_resources.csv")),
                   "cka": [], "feature_manifests": [], "completion_records": []}
    reader.report(directory + ("/report.md" if node == "temporal_variants" else "/summary.md"))
    phase_root = directory.split("/reports/")[0]
    for walk in (1, 2):
        cka = reader.json(f"{phase_root}/diagnostics/cka/walk{walk}.json")
        require(cka["walk"] == walk and cka["selection_or_checkpoint_choice_performed"] is False,
                "invalid CKA selection policy")
        for item in cka["rows"]:
            finite(item["cka"], "cka")
        diagnostics["cka"].append(portable(cka))
    for item in rows:
        method, task, walk = item["method"], item["task"], item["walk"]
        # Reference configurations are checked against connected canonical and
        # predecessor outputs. Fresh run manifests exist only for new candidates.
        candidates = {"temporal_variants": set(CONFIGURATIONS[node]) - {"H0"},
                      "lstm_capacity": {"HC-SL2", "HB-SL2", "HC-AL2", "HB-AL2"},
                      "residual_cnn": {"HC-SR", "HB-SR", "HC-AR", "HB-AR"}}[node]
        if method in candidates:
            run = f"{phase_root}/downstream/{task}/{method.lower()}/walk{walk}/seed0"
            snapshots, detail = run_evidence(reader, run, task, walk, method, item["scope"])
            require(all(abs(snapshots[-1]["scores"][k] - v) < 1e-12
                        for k, v in item["scores"].items() if k in snapshots[-1]["scores"]),
                    "matrix scores disagree with original snapshot")
            diagnostics["completion_records"].append({"method": method, "task": task,
                                                        "walk": walk, **detail})
    tasks = ("absolute_price_h8",) if node == "residual_cnn" else TASKS
    for walk in (1, 2):
        for task in tasks:
            suffix = f"features/walk{walk}/{task}/features.npz.manifest.json" if node == "residual_cnn" else f"features/walk{walk}/{task}.npz.manifest.json"
            m = reader.json(f"{phase_root}/{suffix}")
            require(m["walk"] == walk, "variant feature walk mismatch")
            # These are stored array-replay records, not a new array comparison.
            for split, hashes in m.get("branch_hashes", {}).items():
                for family in ("contrastive", "byol"):
                    if family + "_duplicate" in hashes:
                        require(hashes[family] == hashes[family + "_duplicate"],
                                f"duplicate-control recorded hash mismatch: {split}")
            diagnostics["feature_manifests"].append(portable(m))
    if node == "temporal_variants":
        manifest = reader.json(directory + "/manifest.json")
        require(manifest["matrix_complete"] is True and manifest["entry_count"] == 66 and
                manifest["seed"] == 0 and manifest["principal_epoch"] == 50,
                "temporal report lifecycle mismatch")
        for key, filename in {
            "encoder_resources_csv_sha256": "encoder_resources.csv",
            "epoch50_by_walk_csv_sha256": "epoch50_by_walk.csv",
            "paired_differences_csv_sha256": "paired_differences.csv",
            "report_sha256": "report.md", "resources_csv_sha256": "resources.csv",
            "task_references_csv_sha256": "task_references.csv",
        }.items():
            require(sha256(reader.path(directory + "/" + filename)) == manifest["artifacts"][key],
                    f"temporal report declared hash mismatch: {filename}")
        diagnostics["recorded_validation_counts"] = manifest["validation_counts"]
    return {"rows": rows, "diagnostics": portable(diagnostics)}


def collect_node(node: str, root: Path) -> tuple[dict, list[str]]:
    reader = Reader(root)
    all_scopes, manifests = scopes(reader)
    result: dict[str, Any] = {"scopes": all_scopes, "rows": [], "diagnostics": {}}
    if node == "evidence_inventory":
        result["diagnostics"] = {"tasks": list(TASKS), "walks": [1, 2], "snapshots": [5, 15, 50]}
    elif node == "data_contract":
        for key, m in manifests.items():
            task = key.split(":")[0]
            if task == "classification_h2":
                require(m["seq_len"] == 64 and m["horizon_hours"] == 2 and
                        m["classification"]["tau"] == 0.001, "classification contract mismatch")
                require(m["row_population_rules"]["encoder_train"]["target_dependency"] == "none",
                        "target-dependent encoder population")
                require(m["gap_policy"]["longer_gaps"] == "sequence_break", "long-gap contract mismatch")
            elif task == "absolute_price_h8":
                require(m["seq_len"] == 64 and m["horizon_hours"] == 8, "price contract mismatch")
            else:
                require(m["sequence_length"] == 64 and m["horizon_hours"] == 8 and
                        m["future_candles_required_observed"] is True, "volatility contract mismatch")
            result["diagnostics"][key] = portable(m)
        reader.report("docs/phase_plan/2026-09-24-phase-6-volatility-horizon-freeze-amendment.md")
    elif node == "canonical_encoders":
        m = reader.json("experiments/phase5/reports/encoder_pretraining_seed0/summary.json")
        expected = {(w, e) for w in (1, 2) for e in ("vae", "contrastive", "byol")}
        found = [(r["walk"], r["encoder"]) for r in m["runs"]]
        require(len(found) == 6 and set(found) == expected, "canonical encoder coverage mismatch")
        for run in m["runs"]:
            require(run["valid"] is True and run["snapshots"] == [5, 15, 50],
                    "invalid recorded canonical encoder replay")
            finite(run["epoch50_train_loss"], "encoder loss")
        result["diagnostics"] = portable(m)
    elif node == "canonical_features":
        for walk in (1, 2):
            m = reader.json(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz.manifest.json")
            require(m["branch_dims"] == BRANCH_DIMS and m["concat_dim"] == 445,
                    "canonical feature width mismatch")
            require(m["fitted_on_evaluation"] is False and m["alignment_audit"]["valid"] is True,
                    "invalid recorded feature alignment")
            require(m["source_identity_hashes"] == all_scopes[f"classification_h2:walk{walk}"]["identity_hashes"],
                    "canonical feature population mismatch")
            result["diagnostics"][f"walk{walk}"] = portable(m)
            for task, path in {
                "classification_h2": f"experiments/phase5/downstream/walk{walk}/feature_standardizer.npz.manifest.json",
                "absolute_price_h8": f"experiments/phase5/downstream_addons/shared/h8/feature_scalers/walk{walk}/feature_standardizer.npz.manifest.json",
                "realised_variance": f"experiments/phase6/volatility_prediction/runs/framework_h0/walk{walk}/seed0/feature_standardizer.manifest.json",
            }.items():
                scaler = reader.json(path)
                key = "evaluation_used_for_fit" if task == "realised_variance" else "evaluation_values_used_for_fit"
                require(scaler[key] is False, "canonical feature scaler uses evaluation")
                result["diagnostics"][f"scaler:{task}:walk{walk}"] = portable(scaler)
    elif node in ("classification", "future_price", "realised_variance"):
        task = {"classification": TASKS[0], "future_price": TASKS[1], "realised_variance": TASKS[2]}[node]
        values = table_rows(reader, "temporal_variants", all_scopes)
        result["rows"] = [r for r in values if r["method"] == "H0" and r["task"] == task]
        for walk in (1, 2):
            run = {"classification_h2": f"experiments/phase5/downstream/walk{walk}/classification/seed0",
                   "absolute_price_h8": f"experiments/phase5/downstream_addons/tasks/absolute_price_h8/walk{walk}/seed0",
                   "realised_variance": f"experiments/phase6/volatility_prediction/runs/framework_h0/walk{walk}/seed0"}[task]
            snapshots, detail = run_evidence(reader, run, task, walk, "H0", all_scopes[f"{task}:walk{walk}"])
            principal = next(r for r in result["rows"] if r["walk"] == walk)
            require(all(abs(principal["scores"][k] - v) < 1e-12 for k, v in snapshots[-1]["scores"].items()),
                    "H0 table disagrees with snapshot")
            result["rows"] += snapshots[:-1]
            result["rows"] += reference_rows(reader, run, task, walk, all_scopes[f"{task}:walk{walk}"])
            result["diagnostics"][f"walk{walk}"] = detail
    elif node in REPORTS:
        result.update(variants(reader, node, all_scopes))
    elif node in ("saurl_frozen", "lwa_frozen", "timedart_frozen"):
        dossier = {"saurl_frozen": "SaURL_TS", "lwa_frozen": "LWA", "timedart_frozen": "TimeDART"}[node]
        result["diagnostics"]["source_dossier"] = portable(reader.json(f"docs/baselines/{dossier}/source_manifest.json"))
        for walk in (1, 2):
            feature = reader.json(f"experiments/phase6_7/features/{node}/walk{walk}/representations.npz.manifest.json")
            width = {"saurl_frozen": 128, "lwa_frozen": 384, "timedart_frozen": 170}[node]
            require(feature["output_dim"] == width and feature["targets_loaded_during_extraction"] is False,
                    "external feature contract mismatch")
            require(feature["source_checkpoint_validation"]["valid"] is True,
                    "external recorded encoder validation failed")
            result["diagnostics"][f"features_walk{walk}"] = portable(feature)
            if node == "lwa_frozen":
                replay = reader.json(f"experiments/phase6_7/encoder_pretraining/lwa_frozen/walk{walk}/seed0/replay_validation.json")
                require(replay["valid"] is True and replay["critical_count"] == 0,
                        "LWA recorded encoder replay failed")
                result["diagnostics"][f"encoder_replay_walk{walk}"] = portable(replay)
            for task in TASKS:
                run = f"experiments/phase6_7/downstream/{task}/{node}/walk{walk}/seed0"
                snapshots, detail = run_evidence(reader, run, task, walk, node, all_scopes[f"{task}:walk{walk}"])
                result["rows"] += snapshots
                result["diagnostics"][f"{task}:walk{walk}"] = detail
        reader.report("experiments/phase6_7/reports/frozen_representation_seed0/summary.md")
    elif node == "raw_baselines":
        price_refs = table_rows(reader, "residual_cnn", all_scopes)
        for walk in (1, 2):
            for task in TASKS:
                for method in ("raw_ohlcv_mlp", "raw_ohlcv_lstm"):
                    if task == "realised_variance":
                        model = "raw_lstm" if method == "raw_ohlcv_lstm" else method
                        run = f"experiments/phase6/volatility_prediction/runs/{model}/walk{walk}/seed0"
                    else:
                        run = f"experiments/phase5/baselines/tasks/{task}/{method}/walk{walk}/seed0"
                    snapshots, detail = run_evidence(reader, run, task, walk, method, all_scopes[f"{task}:walk{walk}"])
                    if task == "absolute_price_h8":
                        ref = next(r for r in price_refs if r["walk"] == walk and r["method"] == method)
                        require(all(abs(ref["scores"][k] - v) < 1e-12 for k, v in snapshots[-1]["scores"].items()),
                                "raw price table disagrees with snapshot")
                        snapshots[-1] = ref
                    result["rows"] += snapshots
                    result["diagnostics"][f"{method}:{task}:walk{walk}"] = detail
    elif node == "ta_mlp":
        task = "classification_h2"
        for walk in (1, 2):
            m = reader.json(f"experiments/phase6_5/ta_mlp/data_preparation/walk{walk}/classification_h2_ta36.npz.manifest.json")
            require(m["intersection_uses_labels"] is False and m["feature_contract"]["feature_count"] == 36,
                    "invalid TA intersection contract")
            scope = {**all_scopes[f"{task}:walk{walk}"], "protocol": "ta_intersection",
                     "dataset_sha256": m["artifact"]["sha256"],
                     "identity_hashes": {s: m["intersection"][s]["identity_sha256"] for s in ("train", "test")},
                     **{f"{s}_rows": m["intersection"][s]["included_rows"] for s in ("train", "test")}}
            for model, protocol in [("h0", "p2"), ("raw_mlp", "p2"),
                                    ("raw_lstm", "p2"), ("ta_mlp", "p2"), ("ta_mlp", "p1u")]:
                run = f"experiments/phase6_5/ta_mlp/downstream/{model}/walk{walk}/{protocol}/seed0"
                method = {"h0": "H0", "raw_lstm": "raw_ohlcv_lstm", "raw_mlp": "raw_ohlcv_mlp"}.get(model, model)
                snapshots, detail = run_evidence(reader, run, task, walk, method, scope, "ta_" + protocol)
                result["rows"] += snapshots
                result["diagnostics"][f"{model}:{protocol}:walk{walk}"] = detail
    elif node == "sgn_c":
        result["diagnostics"]["source_dossier"] = portable(reader.json("docs/baselines/SGN/source_manifest.json"))
        task = "classification_h2"
        replay = reader.json("experiments/phase6_9/sgn_classification/replay_validation.json")
        require(replay["status"] == "pass" and replay["same_backend"] is True,
                "SGN recorded replay failed")
        for walk in (1, 2):
            run = f"experiments/phase6_9/sgn_classification/downstream/classification_h2/walk{walk}/sgn_c/seed0"
            snapshots, detail = run_evidence(reader, run, task, walk, node, all_scopes[f"{task}:walk{walk}"])
            result["rows"] += snapshots
            result["diagnostics"][f"walk{walk}"] = detail
        result["diagnostics"]["recorded_same_backend_replay"] = replay
        reader.json("experiments/phase6_9/sgn_classification/reports/seed0/matched_comparison.json")
    elif node == "xm_c8":
        result["diagnostics"]["source_dossier"] = portable(reader.json("docs/baselines/xLSTM-Mixer/source_manifest.json"))
        task = "absolute_price_h8"
        summary = reader.json("experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/summary.json")
        require(summary["endpoint_matched_comparison"] is True and summary["principal_epoch"] == 50,
                "XM endpoint comparison mismatch")
        for walk in (1, 2):
            require(all(v["valid"] is True for v in summary["controls"][str(walk)].values()),
                    "invalid recorded XM control audit")
            run = f"experiments/phase6_9/xlstm_mixer_endpoint/downstream/absolute_price_h8/walk{walk}/xm_c8/seed0"
            snapshots, detail = run_evidence(reader, run, task, walk, node, all_scopes[f"{task}:walk{walk}"])
            replay = reader.json(run + "/replay_validation.json")
            require(replay["valid"] is True and replay["snapshots"] == [5, 15, 50], "XM recorded replay failed")
            result["rows"] += snapshots
            result["diagnostics"][f"walk{walk}"] = {**detail, "recorded_replay": replay}
        result["diagnostics"]["matched_summary"] = portable(summary)
    elif node == "garch_lstm":
        task = "realised_variance"
        for walk in (1, 2):
            run = f"experiments/phase6_5/garch_lstm/walk{walk}/seed0"
            snapshots, detail = run_evidence(reader, run, task, walk, node, all_scopes[f"{task}:walk{walk}"])
            result["rows"] += snapshots
            result["diagnostics"][f"walk{walk}"] = detail
            crossfit = reader.json(run + "/crossfit_manifest.json")
            require(crossfit["fold_count"] == 5 and len(crossfit["folds"]) == 5,
                    "GARCH chronological OOF coverage mismatch")
            for fold in crossfit["folds"]:
                require(fold["training_target_cutoff_ns"] <= fold["prediction_start_ns"],
                        "GARCH OOF chronology mismatch")
            detail["chronological_oof_record"] = portable(crossfit)
        reader.report("experiments/phase6_5/reports/b_c_seed0/principal_results.md")
    elif node == "research_synthesis":
        summary = reader.json("experiments/phase6_9/reports/task_specific_competitiveness_seed0/summary.json")
        require(summary["status"] == "complete" and summary["rank_aggregation_across_tasks"] is False,
                "incomplete or invalid phase synthesis")
        result["diagnostics"]["recorded_phase6_9_synthesis"] = summary
        reader.report("experiments/phase6_9/reports/task_specific_competitiveness_seed0/summary.md")
    else:
        raise ValueError(f"unknown node: {node}")
    return portable(result), sorted(reader.sources)
