from __future__ import annotations

import unittest

from ablation.design import CANONICAL_BRANCHES, build_variants, validate_available_branches


class AblationDesignTests(unittest.TestCase):
    def test_default_matrix_has_five_single_five_loo_and_control(self) -> None:
        variants = build_variants()
        self.assertEqual(len(variants), 11)
        self.assertEqual(sum(item.family == "single" for item in variants), 5)
        self.assertEqual(sum(item.family == "leave_one_out" for item in variants), 5)
        self.assertEqual(variants[-1].name, "full_concat")

    def test_leave_one_out_removes_exactly_named_branch(self) -> None:
        variants = {item.name: item for item in build_variants()}
        for branch in CANONICAL_BRANCHES:
            variant = variants[f"without_{branch}"]
            self.assertNotIn(branch, variant.branches)
            self.assertEqual(len(variant.branches), len(CANONICAL_BRANCHES) - 1)

    def test_gated_is_explicit_and_separate(self) -> None:
        default_names = {item.name for item in build_variants()}
        gated = build_variants(include_gated=True)
        self.assertNotIn("full_gated", default_names)
        self.assertEqual(gated[-1].family, "fusion")
        self.assertEqual(gated[-1].mode, "gated")

    def test_gated_requires_matched_full_control(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched concat control"):
            build_variants(families=("single",), include_gated=True)

    def test_missing_branch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "byol"):
            validate_available_branches(CANONICAL_BRANCHES[:-1])


if __name__ == "__main__":
    unittest.main()
