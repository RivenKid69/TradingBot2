"""
Code review test for PPO target clipping fix.

This test verifies that the actual code changes are correct by checking:
1. The right variables are used in the right places
2. Comments explain the fix
3. No regressions were introduced
"""

import pytest
import re
from pathlib import Path


class TestPPOTargetFixCodeReview:
    """Verify that the code fix is correctly implemented."""

    @pytest.fixture
    def ppo_code(self):
        """Load the distributional_ppo.py file."""
        ppo_file = Path(__file__).parent.parent / "distributional_ppo.py"
        assert ppo_file.exists(), f"File not found: {ppo_file}"
        return ppo_file.read_text()

    def test_training_quantile_uses_unclipped_target(self, ppo_code):
        """
        Verify that training section uses target_returns_norm_raw_selected.

        Line ~8368 should have:
            targets_norm_for_loss = target_returns_norm_raw_selected.reshape(-1, 1)

        NOT:
            targets_norm_for_loss = target_returns_norm_selected.reshape(-1, 1)
        """
        # Search for the correct usage
        pattern = r"targets_norm_for_loss\s*=\s*target_returns_norm_raw_selected\.reshape"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, (
            "Missing correct usage of target_returns_norm_raw_selected for "
            "targets_norm_for_loss in training section"
        )

        # Ensure we don't use the wrong (clipped) version
        wrong_pattern = r"targets_norm_for_loss\s*=\s*target_returns_norm_selected\.reshape"
        wrong_matches = re.findall(wrong_pattern, ppo_code)
        assert len(wrong_matches) == 0, (
            "Found incorrect usage of target_returns_norm_selected "
            "(clipped) for targets_norm_for_loss"
        )

    def test_training_distributional_uses_unclipped_target(self, ppo_code):
        """
        Verify that distributional (C51) path uses unclipped targets.

        Line ~8198 should have:
            clamped_targets = target_returns_norm_raw.clamp(...)

        NOT:
            clamped_targets = target_returns_norm.clamp(...)
        """
        # Search for correct usage in distributional projection
        pattern = r"clamped_targets\s*=\s*target_returns_norm_raw\.clamp"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, (
            "Missing correct usage of target_returns_norm_raw in " "distributional projection"
        )

    def test_eval_uses_unclipped_target(self, ppo_code):
        """
        Verify that eval section uses target_returns_norm_unclipped.

        Line ~7158 should have:
            target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        NOT:
            target_norm_col = target_returns_norm.reshape(-1, 1)
        """
        # Search for correct usage
        pattern = r"target_norm_col\s*=\s*target_returns_norm_unclipped\.reshape"
        matches = re.findall(pattern, ppo_code)
        assert (
            len(matches) >= 1
        ), "Missing correct usage of target_returns_norm_unclipped in eval section"

    def test_explained_variance_batches_use_unclipped(self, ppo_code):
        """
        Verify that value_target_batches_norm stores unclipped targets.

        Line ~8258 should have:
            target_returns_norm_raw_selected.reshape(-1, 1)

        NOT:
            target_returns_norm_selected.reshape(-1, 1)
        """
        # Find the value_target_batches_norm.append section
        pattern = (
            r"value_target_batches_norm\.append\s*\(\s*target_returns_norm_raw_selected\.reshape"
        )
        matches = re.findall(pattern, ppo_code, re.DOTALL)
        assert len(matches) >= 1, (
            "value_target_batches_norm should append target_returns_norm_raw_selected "
            "(unclipped), not target_returns_norm_selected (clipped)"
        )

    def test_critical_fix_comments_present(self, ppo_code):
        """
        Verify that critical fix comments are present to explain the change.
        """
        # Check for key comments
        comments = [
            "CRITICAL FIX: Use UNCLIPPED target",
            "Do NOT clip targets",
            "V_targ must remain unchanged",
            "PPO formula",
        ]

        for comment in comments:
            assert comment in ppo_code, f"Missing important comment: {comment}"

    def test_no_double_clipping_in_distributional(self, ppo_code):
        """
        Verify that distributional projection doesn't double-clip.

        Should use target_returns_norm_raw (unclipped by normalization),
        then clamp to [v_min, v_max] for C51 algorithm.
        """
        # Look for the section that builds target distribution
        pattern = r"clamped_targets\s*=\s*target_returns_norm_raw\.clamp\s*\(\s*self\.policy\.v_min"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, (
            "Distributional projection should clamp target_returns_norm_raw "
            "to [v_min, v_max], not target_returns_norm"
        )

    def test_unclipped_variables_exist(self, ppo_code):
        """
        Verify that unclipped target variables are created.
        """
        # Check that target_returns_norm_raw is created in training section
        pattern = r"target_returns_norm_raw\s*="
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 2, (
            "target_returns_norm_raw should be created in multiple places "
            "(eval and training sections)"
        )

        # Check that target_returns_norm_unclipped exists in eval section
        pattern = r"target_returns_norm_unclipped\s*="
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 2, "target_returns_norm_unclipped should be created in eval section"

    def test_clipped_and_unclipped_both_exist(self, ppo_code):
        """
        Verify that both clipped and unclipped versions exist.

        We need clipped for logging/statistics, unclipped for loss.
        """
        # Both should exist
        assert "target_returns_norm_raw" in ppo_code
        assert "target_returns_norm" in ppo_code
        assert "target_returns_norm_selected" in ppo_code
        assert "target_returns_norm_raw_selected" in ppo_code

    def test_weight_tensor_uses_correct_numel(self, ppo_code):
        """
        Verify that weight tensor creation uses unclipped target for consistency.

        Line ~8272 should use target_returns_norm_raw_selected.numel()
        """
        # Find weight_tensor creation with numel()
        pattern = r"target_returns_norm_raw_selected\.numel\(\)"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, (
            "weight_tensor should use target_returns_norm_raw_selected.numel() " "for consistency"
        )

    def test_expected_group_len_uses_correct_shape(self, ppo_code):
        """
        Verify that expected_group_len uses unclipped target for consistency.
        """
        pattern = r"expected_group_len\s*=.*target_returns_norm_raw_selected"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, (
            "expected_group_len should use target_returns_norm_raw_selected " "for consistency"
        )

    def test_quantile_huber_loss_receives_unclipped_target(self, ppo_code):
        """
        Verify that _quantile_huber_loss receives unclipped target.

        Both critic_loss_unclipped and critic_loss_clipped should use
        targets_norm_for_loss (which should be unclipped).
        """
        # Find calls to _quantile_huber_loss
        pattern = r"self\._quantile_huber_loss\s*\([^)]*targets_norm_for_loss"
        matches = re.findall(pattern, ppo_code, re.DOTALL)
        assert len(matches) >= 2, (
            "Both unclipped and clipped losses should use targets_norm_for_loss "
            "(which should be the unclipped target)"
        )

    def test_ppo_vf_clipping_comment_accurate(self, ppo_code):
        """
        Verify that PPO VF clipping formula is documented correctly.
        """
        # Check for the correct formula in comments
        formula_patterns = [
            r"L\^CLIP_VF",
            r"max\(\s*\(V.*V_targ\)",
            r"clip\(V.*V_old",
        ]

        for pattern in formula_patterns:
            matches = re.findall(pattern, ppo_code, re.IGNORECASE)
            assert (
                len(matches) >= 1
            ), f"Missing or incorrect PPO VF clipping formula documentation: {pattern}"


class TestPPOTargetFixRegression:
    """Verify that the fix doesn't break anything else."""

    @pytest.fixture
    def ppo_code(self):
        """Load the distributional_ppo.py file."""
        ppo_file = Path(__file__).parent.parent / "distributional_ppo.py"
        return ppo_file.read_text()

    def test_old_values_still_clipped(self, ppo_code):
        """
        Verify that old value predictions are still used for clipping.

        The fix should NOT change how predictions are clipped.
        Only targets should remain unclipped.
        """
        # Check that old_values_raw_tensor is still used
        pattern = r"old_values_raw_tensor"
        matches = re.findall(pattern, ppo_code)
        assert (
            len(matches) >= 5
        ), "old_values_raw_tensor should still be used for prediction clipping"

        # Check for value prediction clipping
        pattern = r"value_pred_raw_clipped\s*=\s*torch\.clamp"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 1, "Value predictions should still be clipped"

    def test_clip_range_vf_still_functional(self, ppo_code):
        """
        Verify that clip_range_vf is still checked and used.
        """
        pattern = r"if\s+clip_range_vf"
        matches = re.findall(pattern, ppo_code)
        assert len(matches) >= 2, "clip_range_vf should still be checked in multiple places"

    def test_statistics_logging_preserved(self, ppo_code):
        """
        Verify that statistics and logging are still present.
        """
        patterns = [
            r"_record_value_debug_stats",
            r"train_target_norm",
            r"ev_target_norm",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, ppo_code)
            assert len(matches) >= 1, f"Statistics/logging missing: {pattern}"

    def test_both_quantile_and_distributional_paths_exist(self, ppo_code):
        """
        Verify that both quantile and distributional paths still exist.
        """
        # Check for quantile path
        assert "_use_quantile_value" in ppo_code
        assert "_quantile_huber_loss" in ppo_code

        # Check for distributional path
        assert "target_distribution" in ppo_code
        assert "num_atoms" in ppo_code

    def test_no_syntax_errors_introduced(self, ppo_code):
        """
        Basic check: code should be valid Python.
        """
        try:
            compile(ppo_code, "distributional_ppo.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in distributional_ppo.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
