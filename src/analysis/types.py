"""Shared records for the fixed Phase 7A experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
SnapshotEpoch = Literal[5, 15, 50]
Availability = Literal["available", "unavailable"]


class AnalysisTask(str, Enum):
    CLASSIFICATION = "classification_h2"
    PRICE = "absolute_price_h8"
    VARIANCE = "realised_variance"


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


class AblationKind(str, Enum):
    SINGLE = "single"
    LEAVE_ONE_OUT = "leave_one_out"


class RunState(str, Enum):
    PLANNED = "planned"
    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    REPLAY_VALID = "replay_valid"
    FAILED = "failed"


class MetricDirection(str, Enum):
    HIGHER = "higher_is_better"
    LOWER = "lower_is_better"


@dataclass(frozen=True)
class BranchSpec:
    key: str
    symbol: str
    width: int
    h0_interval: tuple[int, int]  # Zero-based, half-open coordinates.


@dataclass(frozen=True)
class AblationSpec:
    configuration: str
    kind: AblationKind
    branches: tuple[str, ...]
    width: int
    removed_branch: str | None = None


@dataclass(frozen=True)
class TrainingRecipe:
    seed: int = 0
    epochs: int = 50
    snapshots: tuple[SnapshotEpoch, ...] = (5, 15, 50)
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, int] = (128, 64)


@dataclass(frozen=True)
class TaskRecipe:
    task: AnalysisTask
    output_dim: int
    output_transform: str
    loss: str
    dropout: float
    target_multiplier: float = 1.0
    gradient_clip_norm: float | None = None
    smooth_l1_beta: float | None = None
    logit_adjustment_strength: float | None = None


@dataclass(frozen=True)
class PathMapping:
    recorded_root: str  # Preserve the legacy path spelling for provenance.
    resolved_root: Path


@dataclass(frozen=True)
class ReplayPolicy:
    rtol: float = 2e-5
    logit_atol: float = 2e-5
    probability_atol: float = 2e-6
    variance_atol: float = 2e-8
    metric_atol: float = 1e-7
    classification_metric_atol: float = 1e-4
    rank_metric_atol: float = 1e-4
    cka_atol: float = 1e-8

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("Replay tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class AnalysisProfile:
    """One explicitly selected framework per task, shared by both walks."""

    task: AnalysisTask
    family: str
    configuration: str
    selection_reason: str  # Record the metric/rule; never select from this analysis.

    def __post_init__(self) -> None:
        if not isinstance(self.task, AnalysisTask) or any(not isinstance(v, str) or not v.strip()
                for v in (self.family, self.configuration, self.selection_reason)):
            raise ValueError("Profile requires a known task and explicit family/configuration/selection reason")


@dataclass(frozen=True)
class AnalysisContext:
    repo_root: Path
    output_root: Path
    path_mappings: tuple[PathMapping, ...] = ()
    replay_policy: ReplayPolicy = field(default_factory=ReplayPolicy)
    sandbox_name: str | None = None
    site: str = "unspecified"
    profiles: tuple[AnalysisProfile, ...] = ()  # Empty preserves canonical Phase 7A.

    def __post_init__(self) -> None:
        if any(not isinstance(p, AnalysisProfile) for p in self.profiles):
            raise ValueError("Profiles must be AnalysisProfile records")
        if len({p.task for p in self.profiles}) != len(self.profiles):
            raise ValueError("Select at most one analysis profile per task")


@dataclass(frozen=True)
class TaskWalkSource:
    task: AnalysisTask
    walk: int
    dataset_path: Path
    feature_path: Path
    dataset_manifest_path: Path
    feature_manifest_path: Path
    h0_run_path: Path
    h0_scaler_path: Path
    profile: AnalysisProfile | None = None


@dataclass(frozen=True)
class ExecutionEnvironment:
    site: str
    sandbox: str
    hostname: str
    repo_root: Path
    commit_sha: str
    python_executable: Path
    dependency_versions: Mapping[str, str]
    device: str
    backend: str
    log_root: Path
    output_root: Path


@dataclass(frozen=True)
class DependencyFile:
    relative_path: Path
    role: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class RowIdentity:
    """Identity columns for an ordered batch of examples, not just one row."""

    task: AnalysisTask
    walk: int
    condition_ids: Array
    window_start_ns: NDArray[np.int64]
    decision_date_ns: NDArray[np.int64]
    decision_availability_ns: NDArray[np.int64]
    target_fields: Mapping[str, Array]  # Task-specific endpoint/maturity columns.


@dataclass(frozen=True)
class TaskSplit:
    split: Split
    identity: RowIdentity
    targets: Array
    branches: Mapping[str, NDArray[np.float32]]
    metadata: Mapping[str, Array] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskDataset:
    source: TaskWalkSource
    train: TaskSplit
    test: TaskSplit
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: NDArray[np.float64]
    raw_std: NDArray[np.float64]
    scale: NDArray[np.float64]
    training_identity_hash: str
    clip_limits: tuple[float, float] = (-10.0, 10.0)


@dataclass(frozen=True)
class ProbeRunSpec:
    source: TaskWalkSource
    ablation: AblationSpec
    output_path: Path
    recipe: TrainingRecipe = field(default_factory=TrainingRecipe)
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.recipe != TrainingRecipe():
            raise ValueError("Phase 7A training recipe is frozen")
        if self.source.walk not in (1, 2) or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid walk/device")


@dataclass(frozen=True)
class ExperimentMatrix:
    runs: tuple[ProbeRunSpec, ...]
    references: tuple[TaskWalkSource, ...]  # H0 references never enter training.


@dataclass(frozen=True)
class PredictionRecord:
    task: AnalysisTask
    walk: int
    configuration: str
    seed: int
    epoch: SnapshotEpoch
    identity: RowIdentity
    targets: Array
    outputs: Array  # Classification logits or regression predictions.
    units: str
    hashes: Mapping[str, str]
    metadata: Mapping[str, Array] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionSpec:
    task: AnalysisTask
    walk: int
    dimensions: tuple[str, ...] = ("calendar", "contract")
    rv_q90: float | None = None
    rv_q99: float | None = None
    training_target_hash: str | None = None


@dataclass(frozen=True)
class ConditionGroup:
    dimension: str
    name: str
    mask: NDArray[np.bool_]


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    direction: MetricDirection
    count: int
    availability: Availability
    reason: str | None = None


@dataclass(frozen=True)
class MetricRow:
    task: AnalysisTask
    walk: int | None  # None denotes an explicitly pooled result.
    configuration: str
    epoch: SnapshotEpoch
    group: str
    metrics: tuple[MetricResult, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    candidate: MetricRow
    reference: MetricRow
    metric: str
    benefit_positive_difference: float | None
    relative_error_change: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StructuralSample:
    walk: int
    split: Split
    positions: NDArray[np.int64]
    identity: RowIdentity
    identity_hash: str
    seed: int = 170012


@dataclass(frozen=True)
class SimilarityResult:
    sample: StructuralSample
    branches: tuple[str, ...]
    scores: NDArray[np.float64]
    unavailable_reasons: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationObservation:
    check: str
    severity: Literal["pass", "warning", "error"]
    message: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    subject: str
    status: Literal["pending", "valid", "valid_with_warnings", "invalid"] = "pending"
    checks: Mapping[str, str] = field(default_factory=dict)
    observations: tuple[ValidationObservation, ...] = ()

    @property
    def usable(self) -> bool:
        return self.status in ("valid", "valid_with_warnings")


@dataclass(frozen=True)
class RunResult:
    spec: ProbeRunSpec
    state: RunState
    snapshots: tuple[Path, ...] = ()
    validation: ValidationResult | None = None


@dataclass(frozen=True)
class ReportResult:
    output_path: Path
    complete: bool
    artifacts: tuple[Path, ...]


# Source loaders freeze array buffers; metadata records alone cannot do so.
