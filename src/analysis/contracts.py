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


def validate_contracts() -> None:
    """Check branch maps, matrix membership and frozen recipe settings."""
    widths = {b.key: b.width for b in BRANCHES}
    for spec in ABLATIONS:
        if spec.width != sum(widths[b] for b in spec.branches):
            raise ValueError(f"Invalid branch width: {spec.configuration}")
        expected = tuple(b.key for b in BRANCHES if b.key in spec.branches)
        if expected != spec.branches or len(set(spec.branches)) != len(spec.branches):
            raise ValueError("Branch order/uniqueness mismatch")
        if spec.kind == AblationKind.SINGLE and len(spec.branches) != 1:
            raise ValueError("Single requires one branch")
        if spec.kind == AblationKind.LEAVE_ONE_OUT and (
            len(spec.branches) != 4 or spec.removed_branch in spec.branches
            or spec.removed_branch not in widths
        ):
            raise ValueError("Removal requires exactly four remaining branches")


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
    """Expand ten ablations across three tasks/two walks, plus H0 references."""
    from .sources import locate_source
    from .types import ProbeRunSpec
    validate_contracts()
    sources = tuple(locate_source(context, task, walk) for task in TASKS for walk in WALKS)
    runs = tuple(ProbeRunSpec(source, ablation,
                 context.output_root / "downstream" / source.task.value /
                 ablation.configuration / f"walk{source.walk}" / "seed0")
                 for source in sources for ablation in ABLATIONS)
    if len(runs) != EXPECTED_NEW_RUNS or len({r.output_path for r in runs}) != len(runs):
        raise ValueError("Experiment matrix inventory mismatch")
    return ExperimentMatrix(runs, sources)
