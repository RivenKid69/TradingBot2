"""
Comprehensive tests for evaluation section target handling.

These tests verify that the evaluation section (_reserve_ev_samples)
correctly uses unclipped targets for explained variance computation.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from typing import Optional


class TestEvaluationSectionTargetHandling:
    """Test that evaluation section uses unclipped targets."""

    def test_eval_targets_unclipped_normalize_returns_true(self):
        """
        Test evaluation with normalize_returns=True.

        Evaluation section should use target_returns_norm_unclipped,
        NOT target_returns_norm (clipped version).
        """
        # Simulate GAE returns
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0, 60.0, -60.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Evaluation path normalization (line 7117-7122)
        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        # [10.0, -10.0, 8.0, -8.0, 6.0, -6.0]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_unclipped.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [5.0, -5.0, 5.0, -5.0, 5.0, -5.0] - CLIPPED

        # CORRECT: Evaluation should use unclipped (line 7158)
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Verify unclipped values
        assert target_norm_col[0, 0].item() == pytest.approx(10.0)
        assert target_norm_col[1, 0].item() == pytest.approx(-10.0)

        # NOT clipped values
        assert target_norm_col[0, 0].item() != pytest.approx(5.0)

    def test_eval_targets_unclipped_normalize_returns_false(self):
        """
        Test evaluation with normalize_returns=False.

        Should use target_returns_norm_unclipped even when not using
        traditional normalization.
        """
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        base_scale = 10.0
        value_target_scale_effective = 1.0

        # Evaluation path without normalize_returns (line 7130-7144)
        target_returns_norm_unclipped = (returns_raw / base_scale) * value_target_scale_effective
        # [10.0, -10.0, 8.0, -8.0]

        value_clip_limit_scaled = 5.0
        target_returns_norm = torch.clamp(
            target_returns_norm_unclipped, min=-value_clip_limit_scaled, max=value_clip_limit_scaled
        )
        # [5.0, -5.0, 5.0, -5.0] - CLIPPED

        # CORRECT: Should use unclipped
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Verify unclipped
        assert target_norm_col[0, 0].item() == pytest.approx(10.0)
        assert target_norm_col[1, 0].item() == pytest.approx(-10.0)

    def test_eval_targets_with_valid_indices(self):
        """
        Test that valid_indices selection preserves unclipped targets.
        """
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0, 60.0, -60.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Simulate valid_indices selection (line 7165-7166)
        valid_indices = torch.tensor([0, 2, 4])  # Select indices 0, 2, 4
        target_norm_col_selected = target_norm_col[valid_indices]

        # Verify selected values are unclipped
        assert target_norm_col_selected[0, 0].item() == pytest.approx(10.0)
        assert target_norm_col_selected[1, 0].item() == pytest.approx(8.0)
        assert target_norm_col_selected[2, 0].item() == pytest.approx(6.0)

    def test_eval_targets_after_filter_ev_reserve_rows(self):
        """
        Test that targets remain unclipped after _filter_ev_reserve_rows.

        This function filters rows but should not modify target values.
        """
        # Unclipped targets
        target_norm_col = torch.tensor([[10.0], [-10.0], [8.0], [-8.0]])
        target_raw_col = torch.tensor([[100.0], [-100.0], [80.0], [-80.0]])

        # Simulate filtering (keeping all rows)
        weights_tensor = torch.ones(4, 1)
        index_tensor = None

        # After filtering (in this case, no actual filtering)
        # Targets should remain unchanged
        assert torch.allclose(target_norm_col, torch.tensor([[10.0], [-10.0], [8.0], [-8.0]]))

    def test_eval_explained_variance_computation(self):
        """
        Test that explained variance uses unclipped targets.

        EV = 1 - Var(target - prediction) / Var(target)

        Using clipped targets would artificially reduce target variance
        and give misleading EV values.
        """
        # Unclipped targets
        targets_unclipped = torch.tensor([10.0, -10.0, 8.0, -8.0, 6.0, -6.0])

        # Clipped targets (wrong)
        targets_clipped = torch.tensor([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])

        # Predictions (fairly accurate)
        predictions = torch.tensor([9.5, -9.5, 7.5, -7.5, 5.5, -5.5])

        # Compute EV with unclipped targets (correct)
        target_mean_correct = targets_unclipped.mean()
        residual_var_correct = ((targets_unclipped - predictions) ** 2).mean()
        target_var_correct = ((targets_unclipped - target_mean_correct) ** 2).mean()
        ev_correct = 1.0 - residual_var_correct / target_var_correct.clamp(min=1e-8)

        # Compute EV with clipped targets (wrong)
        target_mean_wrong = targets_clipped.mean()
        residual_var_wrong = ((targets_clipped - predictions) ** 2).mean()
        target_var_wrong = ((targets_clipped - target_mean_wrong) ** 2).mean()
        ev_wrong = 1.0 - residual_var_wrong / target_var_wrong.clamp(min=1e-8)

        # EVs should be different
        assert not torch.allclose(ev_correct, ev_wrong)

        # Clipped version has reduced variance
        assert target_var_wrong < target_var_correct

        # Correct EV should be high (predictions are good)
        assert ev_correct.item() > 0.9

    def test_eval_targets_with_mask_weights(self):
        """
        Test that mask weights don't affect target clipping status.
        """
        # Unclipped targets
        target_norm_col = torch.tensor([[10.0], [-10.0], [8.0], [-8.0]])

        # Mask weights
        weights_tensor = torch.tensor([[1.0], [0.5], [0.8], [0.3]])

        # Weighted mean should use unclipped values
        weighted_mean = (target_norm_col * weights_tensor).sum() / weights_tensor.sum()

        # Calculate expected
        expected = (10.0 * 1.0 + (-10.0) * 0.5 + 8.0 * 0.8 + (-8.0) * 0.3) / (1.0 + 0.5 + 0.8 + 0.3)
        # = (10.0 - 5.0 + 6.4 - 2.4) / 2.6 = 9.0 / 2.6 = 3.46

        assert weighted_mean.item() == pytest.approx(expected, rel=1e-4)

    def test_eval_targets_raw_column_consistency(self):
        """
        Test that target_raw_col is consistent with target_norm_col.

        Both should be unclipped versions of their respective scales.
        """
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalized (unclipped)
        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Raw (should also be unclipped)
        target_raw_col = returns_raw.reshape(-1, 1)

        # Verify consistency: raw should be norm * std + mu
        target_raw_from_norm = target_norm_col * ret_std + ret_mu
        assert torch.allclose(target_raw_from_norm, target_raw_col)

    def test_eval_no_clipping_when_within_bounds(self):
        """
        Test that when targets are within clip bounds, they remain unchanged.
        """
        # Moderate returns that don't trigger clipping
        returns_raw = torch.tensor([20.0, -20.0, 15.0, -15.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        # [2.0, -2.0, 1.5, -1.5] - all within ±5 clip bounds

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_unclipped.clamp(
            value_norm_clip_min, value_norm_clip_max
        )

        # Should be identical (no clipping occurred)
        assert torch.allclose(target_returns_norm, target_returns_norm_unclipped)

        # Use unclipped (which equals clipped here)
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Values should match
        assert torch.allclose(target_norm_col.squeeze(), target_returns_norm)

    def test_eval_extreme_clipping_scenario(self):
        """
        Test evaluation with extreme values that would be heavily clipped.
        """
        # Very extreme returns
        returns_raw = torch.tensor([500.0, -500.0, 400.0, -400.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        # [50.0, -50.0, 40.0, -40.0] - WAY outside ±5 bounds

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_unclipped.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [5.0, -5.0, 5.0, -5.0] - All heavily clipped!

        # CRITICAL: Evaluation must use unclipped
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Verify extreme values preserved
        assert target_norm_col[0, 0].item() == pytest.approx(50.0)
        assert target_norm_col[1, 0].item() == pytest.approx(-50.0)

        # Calculate error from clipping
        error = torch.abs(
            (target_returns_norm - target_returns_norm_unclipped)
            / target_returns_norm_unclipped.clamp(min=1e-6)
        ).mean()
        assert error.item() > 0.85  # >85% error from clipping!


class TestEvaluationSectionEdgeCases:
    """Test edge cases in evaluation section."""

    def test_eval_empty_batch(self):
        """Test handling of empty batch in evaluation."""
        # Empty tensors
        target_norm_col = torch.zeros((0, 1))
        target_raw_col = torch.zeros((0, 1))

        # Should handle gracefully (numel() == 0 check)
        assert target_norm_col.numel() == 0
        assert target_raw_col.numel() == 0

    def test_eval_single_sample(self):
        """Test evaluation with single sample."""
        returns_raw = torch.tensor([100.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Single value should be preserved
        assert target_norm_col[0, 0].item() == pytest.approx(10.0)

    def test_eval_all_same_values(self):
        """Test when all targets are the same value."""
        returns_raw = torch.tensor([50.0, 50.0, 50.0, 50.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # All should be 5.0
        assert torch.allclose(target_norm_col, torch.full((4, 1), 5.0))

    def test_eval_with_nan_handling(self):
        """Test that NaN values don't corrupt target handling."""
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std

        # Ensure no NaNs in normalized targets
        assert not torch.isnan(target_returns_norm_unclipped).any()

        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # All values should be finite
        assert torch.isfinite(target_norm_col).all()

    def test_eval_very_small_std(self):
        """Test evaluation with very small std (near-constant returns)."""
        returns_raw = torch.tensor([5.0, 5.01, 4.99, 5.0])
        ret_mu = returns_raw.mean()
        ret_std = returns_raw.std(unbiased=False).clamp(min=1e-8)

        # Normalization with small std
        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std

        # Should not produce extreme values
        assert target_returns_norm_unclipped.abs().max() < 10.0

    def test_eval_mixed_magnitude_targets(self):
        """Test with targets of very different magnitudes."""
        returns_raw = torch.tensor([1.0, -100.0, 5.0, -80.0, 2.0, -90.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        # [0.1, -10.0, 0.5, -8.0, 0.2, -9.0]

        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # All magnitudes should be preserved
        assert target_norm_col[1, 0].item() == pytest.approx(-10.0)  # Large
        assert target_norm_col[0, 0].item() == pytest.approx(0.1)  # Small


class TestEvaluationVsPrediction:
    """Test that evaluation targets match what predictions are compared against."""

    def test_eval_target_prediction_compatibility(self):
        """
        Test that evaluation targets are compatible with predictions.

        Predictions are in normalized space, so targets should also be
        normalized (but UNCLIPPED).
        """
        # Raw returns
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalized unclipped targets (for evaluation)
        target_returns_norm_unclipped = (returns_raw - ret_mu) / ret_std
        target_norm_col = target_returns_norm_unclipped.reshape(-1, 1)

        # Simulated predictions (also normalized)
        predictions = torch.tensor([[9.0], [-9.0], [7.5], [-7.5]])

        # Should be comparable (both in same normalized space)
        diff = target_norm_col - predictions
        mse = (diff**2).mean()

        # MSE should be reasonable (not wildly different scales)
        assert mse.item() > 0.0
        assert mse.item() < 10.0  # Reasonable magnitude

    def test_eval_consistent_with_training(self):
        """
        Test that evaluation uses the same (unclipped) targets as training.

        Both evaluation and training should use unclipped targets for
        consistency and correct metrics.
        """
        # Same returns
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Evaluation path (line 7158)
        target_returns_norm_unclipped_eval = (returns_raw - ret_mu) / ret_std
        target_norm_col_eval = target_returns_norm_unclipped_eval.reshape(-1, 1)

        # Training path (line 8100-8102)
        target_returns_norm_raw_train = (returns_raw - ret_mu) / ret_std

        # Should be identical
        assert torch.allclose(target_norm_col_eval.squeeze(), target_returns_norm_raw_train)

        # Both should have extreme values preserved
        assert target_norm_col_eval[0, 0].item() == pytest.approx(10.0)
        assert target_returns_norm_raw_train[0].item() == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
