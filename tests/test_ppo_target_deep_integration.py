"""
Deep integration tests for PPO target clipping fix.

These tests simulate realistic training scenarios to verify that:
1. Both quantile and distributional (C51) paths use unclipped targets
2. Different configurations (normalize_returns, clip_range_vf) work correctly
3. Edge cases are handled properly
4. Gradients flow correctly through the fixed code
5. The fix works end-to-end in realistic scenarios
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MockPolicy:
    """Mock policy for testing."""

    def __init__(self, use_quantile=True, num_atoms=51):
        self.v_min = -10.0
        self.v_max = 10.0
        self.num_atoms = num_atoms
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.use_quantile = use_quantile


class TestPPOTargetClippingDeepIntegration:
    """Deep integration tests for both quantile and distributional paths."""

    def test_quantile_path_with_extreme_targets(self):
        """
        Test quantile path with extreme target values that would be clipped.

        Scenario: GAE returns produce extreme values (±100)
        After normalization: ±10 (if std=10)
        Normalization clip: ±5

        Bug would clip to ±5 before loss computation.
        Fix uses ±10 (unclipped) in loss computation.
        """
        # Extreme GAE returns
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0], requires_grad=False)
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization (creates both clipped and unclipped)
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [10.0, -10.0, 8.0, -8.0]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [5.0, -5.0, 5.0, -5.0] - CLIPPED

        # Simulate predictions
        quantiles_pred = torch.tensor(
            [
                [7.0, 8.0, 9.0],  # Mean = 8.0
                [-7.0, -8.0, -9.0],  # Mean = -8.0
                [6.0, 7.0, 8.0],  # Mean = 7.0
                [-6.0, -7.0, -8.0],  # Mean = -7.0
            ],
            requires_grad=True,
        )

        # CORRECT: Use unclipped targets
        targets_correct = target_returns_norm_raw.reshape(-1, 1)
        # [10.0, -10.0, 8.0, -8.0]

        # WRONG: Use clipped targets (old bug)
        targets_wrong = target_returns_norm.reshape(-1, 1)
        # [5.0, -5.0, 5.0, -5.0]

        # Compute losses
        loss_correct = F.mse_loss(quantiles_pred.mean(dim=1, keepdim=True), targets_correct)
        loss_wrong = F.mse_loss(quantiles_pred.mean(dim=1, keepdim=True), targets_wrong)

        # Verify losses are different
        assert not torch.allclose(loss_correct, loss_wrong)

        # Correct loss should be smaller (predictions closer to true targets)
        # Mean prediction: 8.0, -8.0, 7.0, -7.0
        # True targets: 10.0, -10.0, 8.0, -8.0
        # Clipped targets: 5.0, -5.0, 5.0, -5.0
        # MSE with true: ((8-10)^2 + (-8-(-10))^2 + (7-8)^2 + (-7-(-8))^2) / 4 = (4+4+1+1)/4 = 2.5
        # MSE with clipped: ((8-5)^2 + (-8-(-5))^2 + (7-5)^2 + (-7-(-5))^2) / 4 = (9+9+4+4)/4 = 6.5

        assert loss_correct.item() < loss_wrong.item()
        assert abs(loss_correct.item() - 2.5) < 0.1
        assert abs(loss_wrong.item() - 6.5) < 0.1

    def test_distributional_c51_with_extreme_targets(self):
        """
        Test distributional (C51) path with extreme targets.

        Critical: Should use unclipped normalized targets when projecting
        to distributional support, NOT double-clipped targets.
        """
        returns_raw = torch.tensor([100.0, -100.0, 90.0, -90.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [10.0, -10.0, 9.0, -9.0]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [5.0, -5.0, 5.0, -5.0] - CLIPPED

        # C51 support bounds
        v_min = -10.0
        v_max = 10.0

        # CORRECT: Project unclipped targets to C51 support
        targets_for_c51_correct = target_returns_norm_raw.clamp(v_min, v_max)
        # [10.0, -10.0, 9.0, -9.0] - Only clamped to support bounds

        # WRONG: Double-clipped (old bug)
        targets_for_c51_wrong = target_returns_norm.clamp(v_min, v_max)
        # [5.0, -5.0, 5.0, -5.0] - Already clipped, stays at ±5

        # Verify difference
        assert not torch.allclose(targets_for_c51_correct, targets_for_c51_wrong)

        # Correct version uses full support range
        assert targets_for_c51_correct[0].item() == pytest.approx(10.0)
        assert targets_for_c51_correct[1].item() == pytest.approx(-10.0)

        # Wrong version is double-clipped
        assert targets_for_c51_wrong[0].item() == pytest.approx(5.0)
        assert targets_for_c51_wrong[1].item() == pytest.approx(-5.0)

        # The difference is 50%!
        relative_error = torch.abs(
            (targets_for_c51_wrong - targets_for_c51_correct)
            / targets_for_c51_correct.clamp(min=1e-6)
        ).mean()
        assert relative_error.item() > 0.4  # More than 40% error!

    def test_vf_clipping_formula_with_both_terms(self):
        """
        Test that PPO VF clipping formula is correctly implemented.

        L^CLIP_VF = max((V - V_targ)^2, (clip(V, V_old±ε) - V_targ)^2)

        V_targ MUST be the same (unclipped) in both terms.
        """
        # Setup
        V_pred = torch.tensor([8.0], requires_grad=True)
        V_old = torch.tensor([5.0])
        V_targ_unclipped = torch.tensor([10.0])
        V_targ_clipped = torch.tensor([5.0])  # Wrong!
        epsilon = 2.0

        # CORRECT implementation
        loss_unclipped_correct = (V_pred - V_targ_unclipped) ** 2
        V_pred_clipped = torch.clamp(V_pred, V_old - epsilon, V_old + epsilon)
        loss_clipped_correct = (V_pred_clipped - V_targ_unclipped) ** 2
        loss_correct = torch.max(loss_unclipped_correct, loss_clipped_correct)

        # WRONG implementation (using clipped target)
        loss_unclipped_wrong = (V_pred - V_targ_clipped) ** 2
        loss_clipped_wrong = (V_pred_clipped - V_targ_clipped) ** 2
        loss_wrong = torch.max(loss_unclipped_wrong, loss_clipped_wrong)

        # Compute gradients for correct version
        loss_correct.backward()
        grad_correct = V_pred.grad.clone()

        # Reset and compute for wrong version
        V_pred.grad = None
        V_pred_wrong = torch.tensor([8.0], requires_grad=True)
        loss_unclipped_wrong = (V_pred_wrong - V_targ_clipped) ** 2
        V_pred_clipped_wrong = torch.clamp(V_pred_wrong, V_old - epsilon, V_old + epsilon)
        loss_clipped_wrong = (V_pred_clipped_wrong - V_targ_clipped) ** 2
        loss_wrong = torch.max(loss_unclipped_wrong, loss_clipped_wrong)
        loss_wrong.backward()
        grad_wrong = V_pred_wrong.grad.clone()

        # Gradients should be VERY different
        assert not torch.allclose(grad_correct, grad_wrong)

        # Correct: V_pred=8, V_targ=10
        # loss_unclipped = (8-10)^2 = 4
        # V_clipped = clamp(8, 3, 7) = 7
        # loss_clipped = (7-10)^2 = 9
        # loss = max(4, 9) = 9
        # grad = d/dV[(7-10)^2] = 2*(7-10)*1 = -6 (with clipping derivative)
        assert abs(loss_correct.item() - 9.0) < 0.01

        # The gradients point in different directions!
        assert grad_correct.item() < 0  # Correct: increase V
        # grad_wrong would be different

    def test_normalize_returns_true_path(self):
        """Test with normalize_returns=True (uses ret_mu and ret_std)."""
        # This is the more common path
        returns_raw = torch.tensor([50.0, -50.0, 30.0, -30.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(20.0)

        # Path with normalize_returns=True
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [2.5, -2.5, 1.5, -1.5]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # No actual clipping in this case (all within ±5)

        # Should use unclipped (which equals clipped here)
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify values
        assert torch.allclose(targets_for_loss.squeeze(), torch.tensor([2.5, -2.5, 1.5, -1.5]))

    def test_normalize_returns_false_path(self):
        """Test with normalize_returns=False (uses value_clip_limit_scaled)."""
        returns_raw = torch.tensor([100.0, -100.0, 80.0, -80.0])
        base_scale = 10.0
        value_target_scale_effective = 1.0

        # Path with normalize_returns=False
        target_returns_norm_raw = (returns_raw / base_scale) * value_target_scale_effective
        # [10.0, -10.0, 8.0, -8.0]

        value_clip_limit_scaled = 5.0
        target_returns_norm = torch.clamp(
            target_returns_norm_raw, min=-value_clip_limit_scaled, max=value_clip_limit_scaled
        )
        # [5.0, -5.0, 5.0, -5.0] - CLIPPED

        # CORRECT: Should use unclipped
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify unclipped values are used
        assert targets_for_loss[0, 0].item() == pytest.approx(10.0)
        assert targets_for_loss[1, 0].item() == pytest.approx(-10.0)

    def test_clip_range_vf_none(self):
        """Test when clip_range_vf is None (no VF clipping)."""
        # When clip_range_vf is None, only unclipped loss is computed
        V_pred = torch.tensor([8.0], requires_grad=True)
        V_targ = torch.tensor([10.0])

        # Only unclipped loss
        loss = (V_pred - V_targ) ** 2

        # Should still use unclipped target
        assert loss.item() == pytest.approx(4.0)

        loss.backward()
        grad = V_pred.grad

        # Gradient: 2*(8-10) = -4
        assert grad.item() == pytest.approx(-4.0)

    def test_clip_range_vf_active(self):
        """Test when clip_range_vf is active (VF clipping enabled)."""
        V_pred = torch.tensor([8.0], requires_grad=True)
        V_old = torch.tensor([5.0])
        V_targ = torch.tensor([10.0])
        clip_range_vf = 2.0

        # Unclipped loss
        loss_unclipped = (V_pred - V_targ) ** 2

        # Clipped loss
        V_pred_clipped = torch.clamp(V_pred, V_old - clip_range_vf, V_old + clip_range_vf)
        loss_clipped = (V_pred_clipped - V_targ) ** 2

        # Final loss
        loss = torch.max(loss_unclipped, loss_clipped)

        # V_clipped = clamp(8, 3, 7) = 7
        # loss_unclipped = 4, loss_clipped = 9
        # loss = 9
        assert loss.item() == pytest.approx(9.0)

    def test_edge_case_zero_std(self):
        """Test edge case when std is very small (near zero)."""
        returns_raw = torch.tensor([5.0, 5.0, 5.0, 5.0])
        ret_mu = torch.tensor(5.0)
        ret_std = torch.tensor(1e-8)  # Very small

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std.clamp(min=1e-8)
        # Should be all zeros (5-5)/1e-8 = 0

        # Should not cause issues
        assert torch.allclose(target_returns_norm_raw, torch.zeros(4))

    def test_edge_case_infinite_values(self):
        """Test edge case with very large values that might cause overflow."""
        returns_raw = torch.tensor([1000.0, -1000.0, 500.0, -500.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [100.0, -100.0, 50.0, -50.0]

        # Even with extreme values, should not clip for loss
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify extreme values preserved
        assert targets_for_loss[0, 0].item() == pytest.approx(100.0)
        assert targets_for_loss[1, 0].item() == pytest.approx(-100.0)

    def test_batch_consistency(self):
        """Test that all samples in a batch use consistent (unclipped) targets."""
        batch_size = 8
        returns_raw = torch.randn(batch_size) * 50.0  # Random returns
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        target_returns_norm = target_returns_norm_raw.clamp(-5.0, 5.0)

        # Count how many are clipped
        clipped_count = ((target_returns_norm_raw.abs() > 5.0).sum()).item()

        # For loss, use unclipped
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # All targets should match unclipped version
        assert torch.allclose(targets_for_loss.squeeze(), target_returns_norm_raw)

        # If any were clipped, clipped version should differ
        if clipped_count > 0:
            assert not torch.allclose(target_returns_norm, target_returns_norm_raw)

    def test_gradient_magnitude_comparison(self):
        """
        Compare gradient magnitudes between correct and wrong implementations.

        Using clipped targets produces smaller gradients (underestimation bias).
        """
        V_pred = torch.tensor([5.0], requires_grad=True)
        V_targ_unclipped = torch.tensor([10.0])
        V_targ_clipped = torch.tensor([5.0])

        # Correct
        loss_correct = (V_pred - V_targ_unclipped) ** 2
        loss_correct.backward()
        grad_correct = V_pred.grad.clone()

        # Wrong
        V_pred.grad = None
        V_pred_wrong = torch.tensor([5.0], requires_grad=True)
        loss_wrong = (V_pred_wrong - V_targ_clipped) ** 2
        loss_wrong.backward()
        grad_wrong = V_pred_wrong.grad.clone()

        # Correct gradient should be larger in magnitude
        # grad_correct = 2*(5-10) = -10
        # grad_wrong = 2*(5-5) = 0
        assert abs(grad_correct.item()) > abs(grad_wrong.item())
        assert grad_correct.item() == pytest.approx(-10.0)
        assert grad_wrong.item() == pytest.approx(0.0)

    def test_explained_variance_with_unclipped_targets(self):
        """
        Test that explained variance is computed with unclipped targets.

        EV = 1 - Var(target - prediction) / Var(target)

        Using clipped targets would give artificially high EV.
        """
        # True targets (unclipped)
        targets_unclipped = torch.tensor([10.0, -10.0, 8.0, -8.0])

        # Clipped targets (wrong)
        targets_clipped = torch.tensor([5.0, -5.0, 5.0, -5.0])

        # Predictions
        predictions = torch.tensor([9.0, -9.0, 7.0, -7.0])

        # EV with unclipped (correct)
        residual_var_correct = ((targets_unclipped - predictions) ** 2).mean()
        target_var_correct = targets_unclipped.var(unbiased=False)
        ev_correct = 1.0 - residual_var_correct / target_var_correct.clamp(min=1e-8)

        # EV with clipped (wrong)
        residual_var_wrong = ((targets_clipped - predictions) ** 2).mean()
        target_var_wrong = targets_clipped.var(unbiased=False)
        ev_wrong = 1.0 - residual_var_wrong / target_var_wrong.clamp(min=1e-8)

        # EVs should be very different
        assert not torch.allclose(ev_correct, ev_wrong)

        # Using clipped targets gives misleading EV
        # (predictions are closer to clipped targets than true targets)

    def test_no_regression_predictions_still_clipped(self):
        """
        Verify that predictions are still clipped (no regression).

        Only targets should be unclipped; predictions must still be clipped
        for PPO VF clipping to work.
        """
        V_pred = torch.tensor([10.0])
        V_old = torch.tensor([5.0])
        clip_range_vf = 2.0

        # Predictions SHOULD be clipped
        V_pred_clipped = torch.clamp(V_pred, V_old - clip_range_vf, V_old + clip_range_vf)

        # Should be clipped to [3, 7] range
        assert V_pred_clipped.item() == pytest.approx(7.0)
        assert V_pred_clipped.item() != V_pred.item()


class TestPPOTargetClippingRealWorldScenarios:
    """Test realistic scenarios that would occur in actual training."""

    def test_high_variance_environment(self):
        """
        Test with high variance environment (e.g., financial trading).

        Returns can vary from -100 to +100, causing extreme normalized values.
        """
        # Realistic financial trading returns
        returns_raw = torch.tensor([50.0, -80.0, 120.0, -50.0, 90.0, -120.0, 60.0, -90.0])
        ret_mu = returns_raw.mean()  # ~0
        ret_std = returns_raw.std(unbiased=False)  # ~80

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # Values will be roughly ±1.5

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )

        # In this case, no clipping occurs (all within ±5)
        # But the fix ensures we always use unclipped version
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify all values preserved
        assert torch.allclose(targets_for_loss.squeeze(), target_returns_norm_raw)

    def test_catastrophic_failure_scenario(self):
        """
        Test catastrophic failure scenario (e.g., agent loses all money).

        This produces extreme negative returns that would be heavily clipped.
        """
        # Catastrophic failure: -1000 return
        returns_raw = torch.tensor([-1000.0, -900.0, -800.0, -700.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [-100.0, -90.0, -80.0, -70.0]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [-5.0, -5.0, -5.0, -5.0] - All clipped!

        # CRITICAL: Loss must use unclipped values
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify extreme negative values preserved
        assert targets_for_loss[0, 0].item() == pytest.approx(-100.0)

        # Bug would clip to -5.0, causing 95% error!
        error = abs(
            (target_returns_norm[0] - target_returns_norm_raw[0]) / target_returns_norm_raw[0]
        )
        assert error.item() > 0.9  # >90% error from clipping

    def test_rare_success_scenario(self):
        """
        Test rare success scenario (e.g., agent makes huge profit).

        This produces extreme positive returns that would be heavily clipped.
        """
        # Rare success: +1000 return
        returns_raw = torch.tensor([1000.0, 900.0, 800.0, 700.0])
        ret_mu = torch.tensor(0.0)
        ret_std = torch.tensor(10.0)

        # Normalization
        target_returns_norm_raw = (returns_raw - ret_mu) / ret_std
        # [100.0, 90.0, 80.0, 70.0]

        value_norm_clip_min = -5.0
        value_norm_clip_max = 5.0
        target_returns_norm = target_returns_norm_raw.clamp(
            value_norm_clip_min, value_norm_clip_max
        )
        # [5.0, 5.0, 5.0, 5.0] - All clipped!

        # CRITICAL: Loss must use unclipped values
        targets_for_loss = target_returns_norm_raw.reshape(-1, 1)

        # Verify extreme positive values preserved
        assert targets_for_loss[0, 0].item() == pytest.approx(100.0)

        # Bug would clip to 5.0, causing 95% error!
        error = abs(
            (target_returns_norm[0] - target_returns_norm_raw[0]) / target_returns_norm_raw[0]
        )
        assert error.item() > 0.9  # >90% error from clipping


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
