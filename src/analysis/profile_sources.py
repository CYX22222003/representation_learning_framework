"""Read-only native variant stores, anchored to validated canonical task rows."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from .artifacts import IDENTITY_FIELDS, hash_array, hash_file, read_json
from .profiles import profile_catalog, source_branches
from .runtime import resolve_recorded_path


def load_selected_dataset(source, context):
    from .sources import load_dataset, locate_source, readonly
    canonical_context = replace(context, profiles=())
    canonical_source = locate_source(canonical_context, source.task, source.walk)
    canonical = load_dataset(canonical_source, canonical_context)
    manifest = read_json(source.feature_manifest_path)
    configurations, master_widths = profile_catalog(source.profile.family)
    branches = source_branches(source)
    if (manifest["task"], manifest["walk"], manifest["epoch"]) != (source.task.value, source.walk, 50):
        raise ValueError("Selected feature task/walk/epoch mismatch")
    if manifest.get("target_independent_extraction") is not True:
        raise ValueError("Selected features lack target-independent extraction provenance")
    if manifest["configuration_branches"] != {k: list(v) for k, v in configurations.items()}:
        raise ValueError("Selected configuration branch map changed")
    if manifest["master_branch_dims"] != master_widths or manifest["configuration_dims"] != {
            name: sum(master_widths[b] for b in names) for name, names in configurations.items()}:
        raise ValueError("Selected configuration dimensions changed")
    for key, expected in (("dataset", source.dataset_path), ("h0_feature", canonical_source.feature_path)):
        actual = resolve_recorded_path(manifest[key + "_path"], context)
        if actual.resolve() != expected.resolve() or manifest[key + "_sha256"] != hash_file(expected):
            raise ValueError(f"Selected {key} lineage mismatch")
    if manifest["dataset_manifest_sha256"] != hash_file(source.dataset_manifest_path):
        raise ValueError("Selected dataset manifest hash mismatch")
    learned = set(master_widths) - set(canonical.train.branches)
    learned = {name for name in learned if not name.endswith("_duplicate")}
    if set(manifest["checkpoint_provenance"]) != learned:
        raise ValueError("Selected encoder provenance inventory mismatch")
    for name, provenance in manifest["checkpoint_provenance"].items():
        validation = provenance["validation"]
        if validation.get("valid") is not True or validation.get("target_independent") is not True or validation.get("walk") != source.walk:
            raise ValueError("Selected encoder lacks valid target-free walk provenance")
        checkpoint = resolve_recorded_path(provenance["path"], context)
        if hash_file(checkpoint) != provenance["sha256"]:
            raise ValueError(f"Selected encoder checkpoint hash mismatch: {name}")
    feature_hash = hash_file(source.feature_path)
    if manifest["feature_store_sha256"] != feature_hash:
        raise ValueError("Selected feature store hash mismatch")
    splits = []
    with np.load(source.feature_path, allow_pickle=False) as stored:
        expected_fields = {f"{split}_{name}" for split in ("train", "test")
                           for name in (*IDENTITY_FIELDS, *master_widths)}
        if set(stored.files) != expected_fields:
            raise ValueError("Selected feature array inventory mismatch")
        for split in (canonical.train, canonical.test):
            prefix = split.split.value + "_"
            if manifest["row_counts"][split.split.value] != len(split.targets):
                raise ValueError("Selected feature row count mismatch")
            for field in IDENTITY_FIELDS:
                if not np.array_equal(stored[prefix + field], getattr(split.identity, field)):
                    raise ValueError("Selected feature identity mismatch")
            values = {}
            # Validate every block, including retained canonical blocks and controls.
            for name, width in master_widths.items():
                value = readonly(stored[prefix + name])
                if value.shape != (len(split.targets), width) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid selected feature block: {name}")
                if hash_array(value) != manifest["branch_hashes"][split.split.value][name]:
                    raise ValueError(f"Selected feature array hash mismatch: {name}")
                if name in split.branches and not np.array_equal(value, split.branches[name]):
                    raise ValueError(f"Selected canonical feature changed: {name}")
                if name.endswith("_duplicate") and not np.array_equal(value, stored[prefix + name.removesuffix("_duplicate")]):
                    raise ValueError("Selected duplicate control is not exact")
                if name in {b.key for b in branches}:
                    values[name] = value
            splits.append(replace(split, branches=MappingProxyType({b.key: values[b.key] for b in branches})))
    hashes = {**canonical.source_hashes, "canonical_features": canonical.source_hashes["features"],
              "canonical_feature_manifest": canonical.source_hashes["feature_manifest"],
              "features": feature_hash, "feature_manifest": hash_file(source.feature_manifest_path)}
    return replace(canonical, source=source, train=splits[0], test=splits[1], source_hashes=MappingProxyType(hashes))
