"""Complete descriptive tables and standalone report figures."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import (atomic_bytes, hash_file, json_value, load_prediction, read_json,
                        write_json_atomic)
from .types import (AnalysisContext, AnalysisTask, ComparisonResult, ExperimentMatrix,
                    MetricDirection, MetricResult, MetricRow, ReportResult, SimilarityResult)
from .profiles import reference_name, similarity_sources, snapshot_count, source_branches


def row_from_json(payload):
    return MetricRow(AnalysisTask(payload["task"]), payload["walk"], payload["configuration"],
        payload["epoch"], payload["group"],
        tuple(MetricResult(m["name"], m["value"], MetricDirection(m["direction"]), m["count"],
                           m["availability"], m["reason"]) for m in payload["metrics"]),
        payload.get("diagnostics", {}))


def write_metric_tables(rows: Sequence[MetricRow], output_path: Path) -> tuple[Path, ...]:
    output_path.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    fields = ("task", "walk", "configuration", "epoch", "group", "metric", "value",
              "direction", "count", "availability", "reason")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        for m in row.metrics:
            writer.writerow(dict(task=row.task.value, walk=row.walk, configuration=row.configuration,
                epoch=row.epoch, group=row.group, metric=m.name, value=m.value, direction=m.direction.value,
                count=m.count, availability=m.availability, reason=m.reason))
    numeric, text = output_path / "metrics.json", output_path / "metrics.csv"
    write_json_atomic(numeric, {"rows": rows})
    atomic_bytes(text, stream.getvalue().encode())
    return numeric, text


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, root):
    root.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(root.with_suffix(suffix) for suffix in (".png", ".svg"))
    for path in paths:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=path.suffix[1:], bbox_inches="tight", dpi=160)
        atomic_bytes(path, buffer.getvalue())
    return paths


def plot_contributions(comparisons: Sequence[ComparisonResult], output_path: Path) -> tuple[Path, ...]:
    from .tasks import PRIMARY_METRICS
    plt = _plotting()
    artifacts = []
    for task in AnalysisTask:
        for metric in PRIMARY_METRICS[task]:
            rows = [v for v in comparisons if v.candidate.task == task and v.metric == metric
                    and v.candidate.epoch == 50
                    and v.reference.configuration.startswith("A-no")]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(7, 3))
            names = tuple(dict.fromkeys(r.reference.configuration for r in rows))
            for walk, offset in ((1, -.18), (2, .18)):
                lookup = {r.reference.configuration: r.benefit_positive_difference for r in rows if r.candidate.walk == walk}
                values = [lookup.get(k) if lookup.get(k) is not None else np.nan for k in names]
                ax.bar(np.arange(len(names)) + offset, values, width=.35, label=f"Walk {walk}")
            ax.set_xticks(np.arange(len(names)), [name.removeprefix("A-no") for name in names])
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_ylabel(f"{metric}: benefit-positive full-model contribution")
            ax.set_xlabel("Removed branch")
            ax.legend()
            artifacts.extend(_save_figure(fig, output_path / f"{task.value}_{metric}_removal"))
            plt.close(fig)
    return tuple(artifacts)


def plot_similarity(results: Sequence[SimilarityResult], output_path: Path) -> tuple[Path, ...]:
    plt = _plotting()
    paths = []
    for result in results:
        fig, ax = plt.subplots(figsize=(5, 4))
        scores = np.ma.masked_invalid(result.scores)
        picture = ax.imshow(scores, vmin=0, vmax=1, cmap="viridis")
        from .contracts import BRANCHES
        symbols = {b.key: b.symbol for b in BRANCHES}
        labels = tuple(symbols.get(name, name) for name in result.branches)
        count = len(labels)
        ax.set_xticks(range(count), labels, rotation=45, ha="right")
        ax.set_yticks(range(count), labels)
        for i in range(count):
            for j in range(count):
                ax.text(j, i, f"{result.scores[i,j]:.2f}" if np.isfinite(result.scores[i,j]) else "N/A",
                        ha="center", va="center", color="white" if result.scores[i,j] < .5 else "black")
        ax.set_title(f"Walk {result.sample.walk} — {result.sample.split.value}")
        fig.colorbar(picture, ax=ax, label="Linear CKA")
        task = result.sample.identity.task
        prefix = "cka" if task == AnalysisTask.CLASSIFICATION else f"cka_{task.value}"
        paths.extend(_save_figure(fig, output_path / f"{prefix}_walk{result.sample.walk}_{result.sample.split.value}"))
        plt.close(fig)
    return tuple(paths)


def build_report(context: AnalysisContext, matrix: ExperimentMatrix, *, allow_partial: bool = False) -> ReportResult:
    from .ablation import compare, removal_contribution, single_branch_skill, summarize_predictions
    from .replay import validate_complete, replay_similarity
    from .sources import audit_source, load_dataset, load_h0_predictions
    from .tasks import PRIMARY_METRICS, reference_records
    from .similarity import load_saved_similarity

    validation = validate_complete(context, matrix)
    if not validation.usable and not allow_partial:
        raise ValueError("Required results are missing/invalid; use --allow-partial for a labelled partial report")
    destination = context.output_root / ("reports/selected_variants_seed0" if context.profiles
                                          else "reports/canonical_five_branch_seed0")
    rows, comparisons, similarities, omissions, available_entries = [], [], [], [], set()
    for source in matrix.references:
        try:
            if not audit_source(source, context).usable:
                raise ValueError("Source lineage failed audit; excluding its dependent results")
            data = load_dataset(source, context)
            h0_validation = read_json(context.output_root / "analysis/reference_replay" / source.task.value / f"walk{source.walk}.json")
            h0_valid = h0_validation["status"] in ("valid", "valid_with_warnings")
            for epoch in (5, 15, 50):
                h0 = load_h0_predictions(source, epoch, context, data) if h0_valid else None
                if h0 is not None:
                    rows.extend(summarize_predictions((h0,)))
                    available_entries.add((source.task.value, source.walk, reference_name(source), epoch))
                for spec in (s for s in matrix.runs if s.source == source):
                    folder = spec.output_path / f"e{epoch}"
                    try:
                        replay = read_json(folder / "replay_validation.json")
                        if replay["status"] not in ("valid", "valid_with_warnings"):
                            raise ValueError("Snapshot is invalid")
                        record = load_prediction(folder / "predictions.npz")
                        rows.extend(summarize_predictions((record,)))
                        available_entries.add((source.task.value, source.walk, spec.ablation.configuration, epoch))
                        if h0 is not None:
                            comparisons.extend(removal_contribution(h0, record) if spec.ablation.removed_branch else compare(record, h0))
                        if not spec.ablation.removed_branch:
                            for reference in reference_records(data, epoch):
                                comparisons.extend(single_branch_skill(record, reference))
                    except (ValueError, KeyError, OSError) as exc:
                        omissions.append(f"{source.task.value}/walk{source.walk}/{spec.ablation.configuration}/e{epoch}: {exc}")
            path = context.output_root / "analysis/conditional" / source.task.value / f"walk{source.walk}/results.json"
            if not context.profiles and path.is_file():
                rows.extend(row_from_json(row) for row in read_json(path)["rows"])
            if source in similarity_sources(context, matrix):
                for split in (data.train, data.test):
                    try:
                        similarity = load_saved_similarity(context, data, split.split)
                        if not replay_similarity(similarity, data).usable:
                            raise ValueError("CKA replay is invalid")
                        similarities.append(similarity)
                    except (ValueError, KeyError, OSError) as exc:
                        omissions.append(f"CKA walk{source.walk}/{split.split.value}: {exc}")
        except (ValueError, KeyError, OSError) as exc:
            omissions.append(f"{source.task.value}/walk{source.walk}: {exc}")
    pooled_path = context.output_root / "analysis/conditional/pooled.json"
    if not context.profiles and pooled_path.is_file():
        rows.extend(row_from_json(row) for row in read_json(pooled_path)["rows"])
    expected_snapshots = snapshot_count(matrix)
    complete = validation.usable and len(available_entries) == expected_snapshots and len(similarities) == 2 * len(similarity_sources(context, matrix))
    tables = write_metric_tables(rows, destination / "tables")
    write_json_atomic(destination / "tables/comparisons.json", {"comparisons": comparisons})
    figures = plot_contributions(comparisons, destination / "figures") + plot_similarity(similarities, destination / "figures")
    title = "Selected-variant branch analysis" if context.profiles else "Canonical five-branch analysis"
    lines = [f"# {title} — seed 0", "",
             f"Status: {'COMPLETE' if complete else 'PARTIAL / INCOMPLETE'}",
             f"Validated snapshots included: {len(available_entries)}/{expected_snapshots}.", "",
             "Epoch 50 is primary; epochs 5/15 are supporting snapshots.", "",
             "| Task | Walk | Configuration | Metric | Epoch-50 value |", "|---|---:|---|---|---:|"]
    for row in rows:
        if row.epoch != 50 or row.group != "overall":
            continue
        for m in row.metrics:
            if m.name in PRIMARY_METRICS[row.task]:
                lines.append(f"| {row.task.value} | {row.walk} | {row.configuration} | {m.name} | {m.value if m.value is not None else 'unavailable'} |")
    lines += ["", "## Interpretation", "",
              "1. Standalone usefulness: compare individual branches with their full-model and fixed references.",
              "2. Incremental contribution: positive removal effects mean retaining that branch helped the observed metric.",
              "3. Similarity: CKA describes representation geometry, not a percentage of shared task information.",
              ("4. Selected variants: post-hoc characterization, not independent confirmation; no conditional or SHAP completion gate."
               if context.profiles else "4. Conditions: inspect walk, equal-contract and RV-tail tables; sparse groups are case-specific."), "",
              "Effects are descriptive. Input width/head capacity differ across configurations. There are no statistical-significance, equivalence, universal-ranking or trading-profit claims.",
              "Encoders and heads use seed 0; two walks do not establish seed robustness. Cohorts are retrospective and approved pruning is assumed correct offline cleaning.",
              "Undefined metrics and sparse groups remain visible in the numeric tables. Validation diagnostics, including accepted cross-environment drift, are retained under analysis/ and manifests/.",
              "", "## Missing or invalid outputs", ""]
    lines += [f"- {item}" for item in omissions] or ["None."]
    if context.profiles:
        lines += ["", "## Selected frameworks", ""]
        for profile in context.profiles:
            source = next(s for s in matrix.references if s.task == profile.task)
            names = ", ".join(f"{b.symbol}={b.key}" for b in source_branches(source))
            lines.append(f"- {profile.task.value}: {profile.family}/{profile.configuration}; {profile.selection_reason}. Branches: {names}.")
    summary = destination / "summary.md"
    atomic_bytes(summary, ("\n".join(lines) + "\n").encode())
    artifacts = (*tables, destination / "tables/comparisons.json", *figures, summary)
    write_json_atomic(destination / "artifact_inventory.json", {"complete": complete, "files": [
        {"path": str(p.relative_to(context.output_root)), "bytes": p.stat().st_size, "sha256": hash_file(p)} for p in artifacts]})
    return ReportResult(destination, complete, artifacts)
