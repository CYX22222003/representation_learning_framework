"""Canonical branches, matrix and fixed task recipes."""

from .types import (
    AblationKind, AblationSpec, AnalysisContext, BranchSpec, ExperimentMatrix,
    AnalysisTask, TaskRecipe, TrainingRecipe,
)

BRANCHES = (
    BranchSpec("statistical", "S", 70, (0, 70)),
    BranchSpec("transformed", "T", 55, (70, 125)),
    BranchSpec("vae", "V", 64, (125, 189)),
    BranchSpec("contrastive", "C", 128, (189, 317)),
    BranchSpec("byol", "B", 128, (317, 445)),
)

ABLATIONS = (
    AblationSpec("A-S", AblationKind.SINGLE, ("statistical",), 70),
    AblationSpec("A-T", AblationKind.SINGLE, ("transformed",), 55),
    AblationSpec("A-V", AblationKind.SINGLE, ("vae",), 64),
    AblationSpec("A-C", AblationKind.SINGLE, ("contrastive",), 128),
    AblationSpec("A-B", AblationKind.SINGLE, ("byol",), 128),
    AblationSpec("A-noS", AblationKind.LEAVE_ONE_OUT,
                 ("transformed", "vae", "contrastive", "byol"), 375, "statistical"),
    AblationSpec("A-noT", AblationKind.LEAVE_ONE_OUT,
                 ("statistical", "vae", "contrastive", "byol"), 390, "transformed"),
    AblationSpec("A-noV", AblationKind.LEAVE_ONE_OUT,
                 ("statistical", "transformed", "contrastive", "byol"), 381, "vae"),
    AblationSpec("A-noC", AblationKind.LEAVE_ONE_OUT,
                 ("statistical", "transformed", "vae", "byol"), 317, "contrastive"),
    AblationSpec("A-noB", AblationKind.LEAVE_ONE_OUT,
                 ("statistical", "transformed", "vae", "contrastive"), 317, "byol"),
)

TASKS = (AnalysisTask.CLASSIFICATION, AnalysisTask.PRICE, AnalysisTask.VARIANCE)
WALKS = (1, 2)
RECIPE = TrainingRecipe()
EXPECTED_NEW_RUNS = 60
EXPECTED_REFERENCES = 6
CKA_SAMPLE_SIZE = 4096
CKA_SEED = 170012


def make_ablations(branches=BRANCHES) -> tuple[AblationSpec, ...]:
    singles = tuple(AblationSpec(f"A-{b.symbol}", AblationKind.SINGLE, (b.key,), b.width) for b in branches)
    removals = tuple(AblationSpec(f"A-no{b.symbol}", AblationKind.LEAVE_ONE_OUT,
        tuple(other.key for other in branches if other.key != b.key),
        sum(other.width for other in branches if other.key != b.key), b.key) for b in branches)
    return singles + removals


def validate_contracts(branches=BRANCHES, ablations=ABLATIONS) -> None:
    """Check branch maps, matrix membership and frozen recipe settings."""
    widths = {b.key: b.width for b in branches}
    if len(widths) != len(branches) or len({b.symbol for b in branches}) != len(branches):
        raise ValueError("Duplicate branch keys/symbols")
    offset = 0
    for branch in branches:
        if branch.width <= 0 or branch.h0_interval != (offset, offset + branch.width):
            raise ValueError("Invalid branch coordinate map")
        offset += branch.width
    for spec in ablations:
        if spec.width != sum(widths[b] for b in spec.branches):
            raise ValueError(f"Invalid branch width: {spec.configuration}")
        expected = tuple(b.key for b in branches if b.key in spec.branches)
        if expected != spec.branches or len(set(spec.branches)) != len(spec.branches):
            raise ValueError("Branch order/uniqueness mismatch")
        if spec.kind == AblationKind.SINGLE and len(spec.branches) != 1:
            raise ValueError("Single requires one branch")
        if spec.kind == AblationKind.LEAVE_ONE_OUT and (
            len(spec.branches) != len(branches) - 1 or spec.removed_branch in spec.branches
            or spec.removed_branch not in widths
        ):
            raise ValueError("Removal must retain all other branches")


def task_recipe(task: AnalysisTask) -> TaskRecipe:
    """Resolve the existing task's loss, transform and regularization."""
    task = AnalysisTask(task)
    if task == AnalysisTask.CLASSIFICATION:
        return TaskRecipe(task, 3, "logits", "logit_adjusted_ce", 0.2,
                          logit_adjustment_strength=1.0)
    if task == AnalysisTask.PRICE:
        return TaskRecipe(task, 1, "sigmoid", "mse", 0.1)
    return TaskRecipe(task, 1, "softplus", "smooth_l1", 0.0, 10000.0, 5.0, 1.0)


def build_matrix(context: AnalysisContext) -> ExperimentMatrix:
    """Expand task-assigned branches across both walks, retaining full references."""
    from .sources import locate_source
    from .types import ProbeRunSpec
    from .profiles import source_branches
    validate_contracts()
    tasks = tuple(t for t in TASKS if not context.profiles or any(p.task == t for p in context.profiles))
    sources = tuple(locate_source(context, task, walk) for task in tasks for walk in WALKS)
    runs = []
    for source in sources:
        branches = source_branches(source)
        ablations = make_ablations(branches)
        validate_contracts(branches, ablations)
        runs.extend(ProbeRunSpec(source, ablation,
                 context.output_root / "downstream" / source.task.value /
                 ablation.configuration / f"walk{source.walk}" / "seed0")
                 for ablation in ablations)
    runs = tuple(runs)
    if not sources or (not context.profiles and len(runs) != EXPECTED_NEW_RUNS) or len({r.output_path for r in runs}) != len(runs):
        raise ValueError("Experiment matrix inventory mismatch")
    return ExperimentMatrix(runs, sources)
