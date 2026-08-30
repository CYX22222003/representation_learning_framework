from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FoldRecord:
    contract_id: int
    processed_row_index: int
    window_start: int
    fold_id: int
    max_training_start: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CrossfitPlan:
    records: tuple[FoldRecord, ...]
    burn_in_row_indices: np.ndarray
    skipped_contracts: tuple[dict[str, object], ...]
    n_folds: int

    def prediction_row_indices(self) -> np.ndarray:
        return np.asarray([row.processed_row_index for row in self.records], dtype=np.int64)

    def contract_ids(self) -> np.ndarray:
        return np.asarray([row.contract_id for row in self.records], dtype=np.int32)

    def window_starts(self) -> np.ndarray:
        return np.asarray([row.window_start for row in self.records], dtype=np.int64)

    def fold_ids(self) -> np.ndarray:
        return np.asarray([row.fold_id for row in self.records], dtype=np.int32)

    def max_training_starts(self) -> np.ndarray:
        return np.asarray([row.max_training_start for row in self.records], dtype=np.int64)

    def manifest(self) -> dict[str, object]:
        by_fold: dict[int, int] = {}
        for row in self.records:
            by_fold[row.fold_id] = by_fold.get(row.fold_id, 0) + 1
        return {
            "crossfit_type": "contract_aware_expanding",
            "n_folds": self.n_folds,
            "prediction_row_count": len(self.records),
            "meta_excluded_burn_in_count": int(self.burn_in_row_indices.size),
            "fold_prediction_counts": {str(k): int(v) for k, v in sorted(by_fold.items())},
            "skipped_contracts": list(self.skipped_contracts),
            "protocol_note": "OOF folds create stacking training features; they are not validation or model selection.",
        }


def make_expanding_folds(
    contract_ids: np.ndarray,
    processed_row_indices: np.ndarray,
    window_starts: np.ndarray,
    *,
    n_folds: int = 5,
) -> CrossfitPlan:
    contracts = np.asarray(contract_ids, dtype=np.int32).reshape(-1)
    rows = np.asarray(processed_row_indices, dtype=np.int64).reshape(-1)
    starts = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    if not (contracts.size == rows.size == starts.size):
        raise ValueError("contract_ids, processed_row_indices, and window_starts must have identical lengths")
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    records: list[FoldRecord] = []
    burn_in: list[np.ndarray] = []
    skipped: list[dict[str, object]] = []
    for contract_id in np.unique(contracts):
        mask = contracts == contract_id
        order = np.argsort(starts[mask], kind="stable")
        c_rows = rows[mask][order]
        c_starts = starts[mask][order]
        if c_rows.size < n_folds + 1:
            skipped.append(
                {
                    "contract_id": int(contract_id),
                    "reason": "requires burn-in plus one non-empty block per fold",
                    "row_count": int(c_rows.size),
                }
            )
            continue
        blocks = np.array_split(np.arange(c_rows.size), n_folds + 1)
        if any(block.size == 0 for block in blocks):
            skipped.append({"contract_id": int(contract_id), "reason": "empty crossfit block", "row_count": int(c_rows.size)})
            continue
        burn_in.append(c_rows[blocks[0]])
        for fold_offset, block in enumerate(blocks[1:], start=1):
            max_train_start = int(c_starts[block[0] - 1])
            for local in block:
                records.append(
                    FoldRecord(
                        contract_id=int(contract_id),
                        processed_row_index=int(c_rows[local]),
                        window_start=int(c_starts[local]),
                        fold_id=int(fold_offset),
                        max_training_start=max_train_start,
                    )
                )
    records.sort(key=lambda item: (item.fold_id, item.contract_id, item.window_start, item.processed_row_index))
    burn_arr = np.concatenate(burn_in).astype(np.int64, copy=False) if burn_in else np.asarray([], dtype=np.int64)
    return CrossfitPlan(tuple(records), burn_arr, tuple(skipped), n_folds)


def assert_oof_integrity(plan: CrossfitPlan) -> None:
    keys = [(r.contract_id, r.processed_row_index) for r in plan.records]
    if len(keys) != len(set(keys)):
        raise ValueError("an OOF row is predicted more than once")
    for row in plan.records:
        if row.max_training_start >= row.window_start:
            raise ValueError("OOF row can be trained on itself or a later within-contract row")


def assert_same_row_identity(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    for key in ("targets", "contract_ids", "processed_row_indices", "window_starts"):
        if key not in reference or key not in candidate:
            raise ValueError(f"missing row identity key: {key}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"row identity mismatch for {key}")
