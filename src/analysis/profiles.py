"""Explicit selected-variant adapters; no ranking or automatic model selection."""

from dataclasses import replace
from pathlib import Path

from .artifacts import read_json
from .types import AnalysisProfile, AnalysisTask, BranchSpec


# Explicit opt-in roster; canonical Phase 7A remains the CLI default.
SELECTED_PROFILES_PATH = Path(__file__).with_name("configs") / "phase7c_selected_profiles.json"


def profile_catalog(family):
    if family == "canonical":
        from .contracts import BRANCHES
        return {"H0": tuple(b.key for b in BRANCHES)}, {b.key: b.width for b in BRANCHES}
    if family == "temporal":
        from features.phase6_encoder_variant_features import CONFIG_BRANCHES, MASTER_BRANCH_DIMS
    elif family == "lstm_capacity":
        from features.phase6_5_lstm_capacity_features import CONFIG_BRANCHES, MASTER_BRANCH_DIMS
    elif family == "residual_cnn":
        from features.phase6_5_residual_cnn_features import CONFIG_BRANCHES, MASTER_BRANCH_DIMS
    else:
        raise ValueError(f"Unknown analysis profile family: {family}")
    return CONFIG_BRANCHES, MASTER_BRANCH_DIMS


def validate_profile(profile):
    configurations, _ = profile_catalog(profile.family)
    if not isinstance(profile.task, AnalysisTask) or profile.configuration not in configurations:
        raise ValueError("Unknown profile task/configuration")
    if profile.family != "canonical" and profile.configuration == "H0":
        raise ValueError("Use the canonical family for H0")
    if profile.family == "residual_cnn" and profile.task != AnalysisTask.PRICE:
        raise ValueError("Residual-CNN artifacts cover future price only")
    if not profile.selection_reason.strip():
        raise ValueError("Record the profile selection metric/rule")


def load_profiles(path: Path) -> tuple[AnalysisProfile, ...]:
    payload = read_json(path)
    if set(payload) != {"profiles"} or not isinstance(payload["profiles"], dict):
        raise ValueError("Profiles JSON requires a task-keyed profiles object")
    if set(payload["profiles"]) != {t.value for t in AnalysisTask}:
        raise ValueError("Select exactly one framework for each of the three tasks")
    profiles = tuple(AnalysisProfile(AnalysisTask(task), **entry)
                     for task, entry in payload["profiles"].items())
    for profile in profiles:
        validate_profile(profile)
    return tuple(sorted(profiles, key=lambda p: list(AnalysisTask).index(p.task)))


def source_branches(source) -> tuple[BranchSpec, ...]:
    from .contracts import BRANCHES
    if source.profile is None or source.profile.family == "canonical":
        return BRANCHES
    validate_profile(source.profile)
    configurations, widths = profile_catalog(source.profile.family)
    names = configurations[source.profile.configuration]
    symbols = {b.key: b.symbol for b in BRANCHES}
    result, offset = [], 0
    for index, name in enumerate(names):
        # Slots C/B preserve familiar labels for substitutions; additions get X6.
        symbol = symbols.get(name, BRANCHES[index].symbol if index < 5 else f"X{index + 1}")
        width = widths[name]
        result.append(BranchSpec(name, symbol, width, (offset, offset + width)))
        offset += width
    return tuple(result)


def reference_name(source):
    return source.profile.configuration if source.profile else "H0"


def adapt_source(canonical, context):
    profile = next((p for p in context.profiles if p.task == canonical.task), None)
    if profile is None:
        return canonical
    validate_profile(profile)
    if profile.family == "canonical":
        return replace(canonical, profile=profile)
    relative = {"temporal": "phase6/encoder_variants", "lstm_capacity": "phase6_5/lstm_capacity",
                "residual_cnn": "phase6_5/residual_cnn"}[profile.family]
    base = context.repo_root / "experiments" / relative
    if profile.family == "residual_cnn":
        feature = base / f"features/walk{canonical.walk}/{canonical.task.value}/features.npz"
    else:
        feature = base / f"features/walk{canonical.walk}/{canonical.task.value}.npz"
    run = base / "downstream" / canonical.task.value / profile.configuration.lower() / f"walk{canonical.walk}/seed0"
    return replace(canonical, feature_path=feature, feature_manifest_path=Path(f"{feature}.manifest.json"),
                   h0_run_path=run, h0_scaler_path=run / "feature_standardizer.npz", profile=profile)


def similarity_sources(context, matrix):
    # Canonical 7A retains classification-only CKA; selected variants use each task.
    return tuple(s for s in matrix.references if context.profiles or s.task == AnalysisTask.CLASSIFICATION)


def snapshot_count(matrix):
    return 3 * (len(matrix.runs) + len(matrix.references))
