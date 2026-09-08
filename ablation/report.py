"""Aggregate matched ablation artifacts into machine- and human-readable tables."""

from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_metrics(task: str) -> tuple[tuple[str, str], ...]:
    if task == "trend_classification":
        return (("accuracy", "higher"), ("macro_f1", "higher"))
    if task == "price_prediction":
        return (("mae", "lower"), ("rmse", "lower"))
    if task == "volatility_prediction":
        return (("mse", "lower"), ("corr", "higher"))
    raise ValueError(f"Unsupported task: {task}")


def build_report(study_root: str | Path, seed: int = 0) -> tuple[Path, Path]:
    root = Path(study_root)
    plan = _read_json(root / "plan.json")
    if not isinstance(plan, dict):
        raise ValueError("plan.json must contain an object")
    rows: list[dict[str, object]] = []
    for task in plan["tasks"]:
        control_path = root / "runs" / task / "full_concat" / f"seed_{seed}" / "sweep_metrics.json"
        if not control_path.exists():
            raise FileNotFoundError(f"Missing matched full-concat control: {control_path}")
        controls = {int(row["epoch"]): row for row in _read_json(control_path)}
        for variant in plan["variants"]:
            sweep_path = root / "runs" / task / variant["name"] / f"seed_{seed}" / "sweep_metrics.json"
            if not sweep_path.exists():
                raise FileNotFoundError(f"Missing ablation sweep: {sweep_path}")
            for metric_row in _read_json(sweep_path):
                epoch = int(metric_row["epoch"])
                control = controls[epoch]
                row: dict[str, object] = {
                    "task": task,
                    "variant": variant["name"],
                    "family": variant["family"],
                    "removed_branch": (
                        variant["name"].removeprefix("without_")
                        if variant["family"] == "leave_one_out" else None
                    ),
                    "branches": variant["branches"],
                    "mode": variant["mode"],
                    "epoch": epoch,
                    "model_input_dim": int(metric_row.get("model_input_dim", sum(metric_row["branch_dims"].values()))),
                    "trainable_parameter_count": int(metric_row.get("trainable_parameter_count", 0)),
                }
                for metric, direction in _primary_metrics(task):
                    value = float(metric_row[metric])
                    reference = float(control[metric])
                    # Positive means the variant beats the full-concat control.
                    delta = reference - value if direction == "lower" else value - reference
                    row[metric] = value
                    row[f"{metric}_delta_vs_full_concat"] = delta
                rows.append(row)

    json_path = root / "ablation_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Ablation study results",
        "",
        f"Seed: `{seed}`.",
        "",
        "Every row is a predeclared fixed-budget result. Positive deltas mean the variant outperformed the matched full-concat control; no budget is selected from test performance.",
        "",
    ]
    for task in plan["tasks"]:
        metrics = [name for name, _ in _primary_metrics(task)]
        lines += [
            f"## {task.replace('_', ' ').title()}",
            "",
            "| variant | family | epoch | input dim | trainable params | " + " | ".join(metrics) + " | "
            + " | ".join(f"delta {metric}" for metric in metrics) + " |",
            "|---|---|---:|---:|---:|" + "---:|" * (len(metrics) * 2),
        ]
        for row in (item for item in rows if item["task"] == task):
            values = [f"{float(row[metric]):.8f}" for metric in metrics]
            deltas = [f"{float(row[f'{metric}_delta_vs_full_concat']):+.8f}" for metric in metrics]
            lines.append(
                f"| {row['variant']} | {row['family']} | {row['epoch']} | "
                f"{row['model_input_dim']} | {row['trainable_parameter_count']} | "
                + " | ".join(values + deltas) + " |"
            )
        lines.append("")
    markdown_path = root / "REPORT.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
