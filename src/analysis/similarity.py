"""Deterministic condition-balanced CKA on classification feature populations."""

from typing import Sequence

import numpy as np

from .artifacts import identity_hash, subset_identity
from .contracts import BRANCHES, CKA_SAMPLE_SIZE, CKA_SEED
from .features import apply_standardizer
from .types import (AnalysisTask, FeatureStandardizer, SimilarityResult, StructuralSample,
                    TaskDataset, TaskSplit)


def sample_classification_splits(datasets: Sequence[TaskDataset]) -> tuple[StructuralSample, ...]:
    if len(datasets) != 2 or {d.source.walk for d in datasets} != {1, 2}:
        raise ValueError("CKA requires classification datasets for both walks")
    rng = np.random.Generator(np.random.PCG64(CKA_SEED))
    result = []
    for dataset in sorted(datasets, key=lambda d: d.source.walk):
        if dataset.source.task != AnalysisTask.CLASSIFICATION:
            raise ValueError("CKA population must be classification")
        for split in (dataset.train, dataset.test):
            identity = split.identity
            queues = []
            for condition in np.unique(identity.condition_ids):
                positions = np.flatnonzero(identity.condition_ids == condition)
                order = np.lexsort((identity.decision_availability_ns[positions],
                                    identity.decision_date_ns[positions], identity.window_start_ns[positions]))
                queues.append(rng.permutation(positions[order]).tolist())
            chosen = []
            for depth in range(max(map(len, queues), default=0)):
                for queue in queues:
                    if depth < len(queue):
                        chosen.append(queue[depth])
                        if len(chosen) == min(CKA_SAMPLE_SIZE, len(split.targets)):
                            break
                if len(chosen) == min(CKA_SAMPLE_SIZE, len(split.targets)):
                    break
            positions = np.asarray(chosen, dtype=np.int64)
            sampled = subset_identity(identity, positions)
            result.append(StructuralSample(identity.walk, split.split, positions, sampled, identity_hash(sampled)))
    return tuple(result)


def compute_cka(split: TaskSplit, scaler: FeatureStandardizer, sample: StructuralSample) -> SimilarityResult:
    from evaluation.phase6_encoder_variants import centered_linear_cka
    if split.identity.task != AnalysisTask.CLASSIFICATION or (sample.walk, sample.split) != (split.identity.walk, split.split):
        raise ValueError("CKA sample task/walk/split mismatch")
    if identity_hash(subset_identity(split.identity, sample.positions)) != sample.identity_hash:
        raise ValueError("CKA sample identities changed")
    features = np.concatenate([split.branches[b.key] for b in BRANCHES], axis=1)
    scaled = apply_standardizer(features[sample.positions], scaler)
    values, unavailable = [], {}
    for b in BRANCHES:
        keep = scaler.raw_std[slice(*b.h0_interval)] >= 1e-8
        values.append(scaled[:, slice(*b.h0_interval)][:, keep])
    scores = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i, 5):
            key = f"{BRANCHES[i].key}:{BRANCHES[j].key}"
            if values[i].shape[1] == 0 or values[j].shape[1] == 0:
                unavailable[key] = "training_constant_branch"
                continue
            try:
                scores[i, j] = scores[j, i] = centered_linear_cka(values[i], values[j])
            except ValueError as exc:
                unavailable[key] = str(exc)
    return SimilarityResult(sample, tuple(b.key for b in BRANCHES), scores, unavailable)


def load_saved_similarity(context, dataset, split):
    from .artifacts import hash_array, identity_arrays, read_json
    base = context.output_root / f"analysis/cka/classification_h2/walk{dataset.source.walk}/{split.value}"
    metadata = read_json(base / "result.json")
    population = dataset.train if split.value == "train" else dataset.test
    if metadata["source_hashes"] != dict(dataset.source_hashes):
        raise ValueError("CKA source provenance changed")
    with np.load(base / "sample.npz", allow_pickle=False) as arrays:
        positions = arrays["positions"]
        if hash_array(positions) != metadata["positions_sha256"]:
            raise ValueError("CKA sample positions changed")
        identity = subset_identity(population.identity, positions)
        for name, value in identity_arrays(identity).items():
            if not np.array_equal(value, arrays[name]):
                raise ValueError("CKA saved identity columns changed")
    if identity_hash(identity) != metadata["sample_identity_hash"]:
        raise ValueError("CKA sample identity changed")
    with np.load(base / "scores.npz", allow_pickle=False) as arrays:
        scores = arrays["scores"]
    if scores.shape != (5, 5) or hash_array(scores) != metadata["scores_sha256"]:
        raise ValueError("CKA matrix shape/hash mismatch")
    sample = StructuralSample(dataset.source.walk, split, positions, identity, identity_hash(identity))
    return SimilarityResult(sample, tuple(metadata["branches"]), scores, metadata["unavailable_reasons"])
