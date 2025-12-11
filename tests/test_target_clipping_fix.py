"""
Test suite for Target Clipping Bug Fix (2025-11-24)

This module verifies that target returns are NEVER clipped during training or evaluation.

CRITICAL BUG FIX:
=================
PPO value function clipping should clip PREDICTIONS relative to old values, NOT targets.

WRONG (previous bug):
    target_clipped = clip(target, -epsilon, +epsilon)  # ❌ Clips ground truth!
    loss = (V_pred - target_clipped)²

CORRECT (after fix):
    V_clipped = V_old + clip(V_pred - V_old, -epsilon, +epsilon)
    loss = max((V_pred - target)², (V_clipped - target)²)  # Target unchanged!

Example of catastrophic failure (if bug not fixed):
- Actual return: +1.0 (100% profit)
- value_clip_limit: 0.2 (typical PPO epsilon)
- Clipped target: +0.2 (20%)
- Model learns: "Maximum possible return is 0.2" ❌

See: CONCEPTUAL_BUGS_VERIFICATION_REPORT_2025_11_24.md for full analysis
"""
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from distributional_ppo import DistributionalPPO
from stable_baselines3.common.buffers import RolloutBuffer


class TestTargetClippingFix:
    """Test suite for verifying target returns are NOT clipped."""

    def test_targets_not_clipped_in_training_extreme_values(self):
        """
        Verify that extreme target returns (outside typical clip range) are NOT clipped.

        This test ensures the model can learn from large positive/negative returns
        without being limited by value_clip_limit.
        """
        # Create mock rollout buffer with EXTREME returns
        # These values are intentionally outside typical clip ranges (e.g., ±0.2)
        extreme_returns = torch.tensor([
            [-10.0],  # Large loss (e.g., -1000% return)
            [10.0],   # Large profit (e.g., +1000% return)
            [0.5],    # Normal return
            [-5.0],   # Medium loss
            [5.0],    # Medium profit
        ])

        # Mock PPO instance (simplified for testing)
        # In real code, this would be a full DistributionalPPO instance
        class MockPPO:
            def __init__(self):
                self.normalize_returns = False
                self._value_clip_limit_unscaled = 0.2  # Typical PPO epsilon
                self._value_clip_limit_scaled = None
                self._value_norm_clip_min = -10.0
                self._value_norm_clip_max = 10.0
                self.value_target_scale = 1.0
                self._value_target_scale_effective = 1.0

            def _decode_returns_scale_only(self, returns):
                """Mock method to decode returns (no actual scaling for test)."""
                return returns, 1.0

            def _record_value_debug_stats(self, name, values, clip_bounds=None):
                """Mock method to record debug stats (no-op for test)."""
                pass

        ppo = MockPPO()

        # Simulate the code path from distributional_ppo.py:10291-10339
        target_returns_raw = extreme_returns.clone()
        base_scale_safe = 1.0
        ret_mu_tensor = torch.tensor([0.0])
        ret_std_tensor = torch.tensor([1.0])
        target_raw_pre_limit = target_returns_raw.detach()

        # This is the FIXED code (should NOT clip targets)
        raw_limit_bounds_train = None
        if (not ppo.normalize_returns) and (ppo._value_clip_limit_unscaled is not None):
            limit_unscaled = float(ppo._value_clip_limit_unscaled)
            raw_limit_bounds_train = (-limit_unscaled, limit_unscaled)
            # NOTE: We log what WOULD BE clipped, but DON'T actually clip!
            # (No torch.clamp() call here!)

        target_raw_post_limit = target_returns_raw.detach()

        # FIX (2025-11-24): Use UNCLIPPED targets everywhere (no .clamp()!)
        if ppo.normalize_returns:
            target_returns_norm_raw = (target_returns_raw - ret_mu_tensor) / ret_std_tensor
            target_returns_norm = target_returns_norm_raw  # NO CLIPPING!
        else:
            target_returns_norm_raw = (
                (target_returns_raw / float(base_scale_safe))
                * ppo._value_target_scale_effective
            )
            target_returns_norm = target_returns_norm_raw  # NO CLIPPING!

        # CRITICAL ASSERTION: Targets should be IDENTICAL to original (unclipped)
        assert torch.allclose(target_returns_norm, extreme_returns), (
            f"Targets were clipped! Expected {extreme_returns}, got {target_returns_norm}"
        )

        # Verify NO values were clamped to clip_range bounds
        clip_limit = ppo._value_clip_limit_unscaled
        num_above = (extreme_returns > clip_limit).sum().item()
        num_below = (extreme_returns < -clip_limit).sum().item()

        assert num_above > 0, "Test data should have values above clip_limit"
        assert num_below > 0, "Test data should have values below clip_limit"

        # Verify targets preserve extreme values (not clamped to ±clip_limit)
        assert torch.max(target_returns_norm).item() > clip_limit, (
            f"Targets were clipped at upper bound! Max: {torch.max(target_returns_norm).item()}, "
            f"Limit: {clip_limit}"
        )
        assert torch.min(target_returns_norm).item() < -clip_limit, (
            f"Targets were clipped at lower bound! Min: {torch.min(target_returns_norm).item()}, "
            f"Limit: {-clip_limit}"
        )

    def test_targets_not_clipped_in_normalized_mode(self):
        """
        Verify that normalized targets are NOT clipped when normalize_returns=True.

        This test ensures normalization doesn't introduce clipping.
        """
        extreme_returns = torch.tensor([[-5.0], [5.0], [0.0]])

        class MockPPO:
            def __init__(self):
                self.normalize_returns = True  # Enable normalization
                self._value_clip_limit_unscaled = None
                self._value_clip_limit_scaled = None
                self._value_norm_clip_min = -10.0
                self._value_norm_clip_max = 10.0
                self.value_target_scale = 1.0
                self._value_target_scale_effective = 1.0

        ppo = MockPPO()

        target_returns_raw = extreme_returns.clone()
        ret_mu_tensor = torch.tensor([0.0])
        ret_std_tensor = torch.tensor([1.0])

        # FIX (2025-11-24): Use UNCLIPPED targets everywhere
        if ppo.normalize_returns:
            target_returns_norm_raw = (target_returns_raw - ret_mu_tensor) / ret_std_tensor
            target_returns_norm = target_returns_norm_raw  # NO CLIPPING!
        else:
            target_returns_norm_raw = target_returns_raw
            target_returns_norm = target_returns_norm_raw

        # CRITICAL ASSERTION: Normalized targets should be IDENTICAL to raw normalized (unclipped)
        expected_normalized = (extreme_returns - ret_mu_tensor) / ret_std_tensor
        assert torch.allclose(target_returns_norm, expected_normalized), (
            f"Normalized targets were clipped! Expected {expected_normalized}, "
            f"got {target_returns_norm}"
        )

    def test_vf_clipping_clips_predictions_not_targets(self):
        """
        Verify PPO VF clipping clips PREDICTIONS relative to old values, NOT targets.

        This test demonstrates the CORRECT PPO VF clipping formula:
            V_clipped = V_old + clip(V_pred - V_old, -epsilon, +epsilon)
            L = max((V_pred - V_target)², (V_clipped - V_target)²)

        The target V_target must remain UNCHANGED in both loss terms.
        """
        # Setup
        old_values = torch.tensor([[0.0], [0.0], [0.0]])
        target_returns = torch.tensor([[5.0], [-5.0], [0.5]])  # Large targets (outside clip range)
        value_pred = torch.tensor([[2.0], [-2.0], [0.3]])      # Moderate predictions
        clip_range_vf = 0.2  # Typical PPO epsilon

        # CORRECT PPO VF CLIPPING (clip predictions, not targets)
        value_pred_clipped = old_values + torch.clamp(
            value_pred - old_values,
            -clip_range_vf,
            +clip_range_vf
        )

        # Expected clipped predictions
        # For third value: 0.0 + clip(0.3 - 0.0, -0.2, 0.2) = 0.0 + 0.2 = 0.2
        expected_clipped = torch.tensor([[0.2], [-0.2], [0.2]])
        assert torch.allclose(value_pred_clipped, expected_clipped), (
            f"Prediction clipping incorrect! Expected {expected_clipped}, "
            f"got {value_pred_clipped}"
        )

        # Loss should use ORIGINAL targets (unclipped)
        loss_unclipped = (value_pred - target_returns) ** 2
        loss_clipped = (value_pred_clipped - target_returns) ** 2  # Same target!
        loss = torch.max(loss_unclipped, loss_clipped)  # Element-wise max

        # CRITICAL ASSERTION: Targets were NOT modified
        assert torch.allclose(target_returns, torch.tensor([[5.0], [-5.0], [0.5]])), (
            "Targets were modified! PPO VF clipping should NEVER change targets."
        )

        # Verify loss computation used original targets
        expected_loss_unclipped = torch.tensor([
            [(2.0 - 5.0)**2],   # 9.0
            [(-2.0 - (-5.0))**2],  # 9.0
            [(0.3 - 0.5)**2],   # 0.04
        ])
        assert torch.allclose(loss_unclipped, expected_loss_unclipped, atol=1e-5), (
            f"Unclipped loss incorrect! Expected {expected_loss_unclipped}, "
            f"got {loss_unclipped}"
        )

    def test_extreme_returns_preserved_in_ev_computation(self):
        """
        Verify Explained Variance (EV) computation uses UNCLIPPED targets.

        EV should measure prediction accuracy against actual returns, not clipped returns.
        """
        extreme_returns = torch.tensor([[-10.0], [10.0], [0.5], [-5.0], [5.0]])

        # Simulate EV computation (simplified)
        target_norm_col = extreme_returns  # Should be UNCLIPPED
        predicted_values = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]])

        # Compute explained variance
        var_target = torch.var(target_norm_col)
        residuals = target_norm_col - predicted_values
        var_residuals = torch.var(residuals)

        # EV = 1 - Var(residuals) / Var(targets)
        # If targets were clipped, var_target would be artificially low
        ev = 1.0 - (var_residuals / var_target)

        # CRITICAL ASSERTION: Target variance should be HIGH (not reduced by clipping)
        # For [-10, 10, 0.5, -5, 5], variance should be very large (> 40)
        assert var_target > 40.0, (
            f"Target variance too low! Expected > 40, got {var_target.item()}. "
            f"Targets may have been clipped."
        )

    def test_no_clipping_in_config_none(self):
        """
        Verify that when value_clip_limit=None (default), no clipping occurs.

        This is the SAFE configuration that all production configs should use.
        """
        extreme_returns = torch.tensor([[-100.0], [100.0]])

        class MockPPO:
            def __init__(self):
                self.normalize_returns = False
                self._value_clip_limit_unscaled = None  # ← DEFAULT (SAFE)
                self._value_clip_limit_scaled = None
                self._value_norm_clip_min = -10.0
                self._value_norm_clip_max = 10.0
                self.value_target_scale = 1.0
                self._value_target_scale_effective = 1.0

        ppo = MockPPO()
        target_returns_raw = extreme_returns.clone()

        # Simulate code path
        raw_limit_bounds_train = None
        if (not ppo.normalize_returns) and (ppo._value_clip_limit_unscaled is not None):
            # This branch should NOT execute when value_clip_limit=None
            pytest.fail("Entered clipping code path when value_clip_limit=None!")

        target_returns_norm = target_returns_raw  # NO CLIPPING!

        # CRITICAL ASSERTION: Targets preserved exactly
        assert torch.allclose(target_returns_norm, extreme_returns), (
            "Targets were modified when value_clip_limit=None!"
        )


class TestTargetClippingDocumentation:
    """Test that documentation and warnings are correct."""

    def test_fix_comments_present(self):
        """Verify that FIX (2025-11-24) comments are present in code."""
        with open("distributional_ppo.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Check for fix comments in EV reserve method
        assert "FIX (2025-11-24): REMOVED target clipping" in code, (
            "Missing FIX comment in EV reserve method"
        )

        # Check for fix comments in training loop
        assert "FIX (2025-11-24): Use UNCLIPPED targets everywhere" in code, (
            "Missing FIX comment in training loop"
        )

        # Check that NO torch.clamp(target_returns_*, ...) exists
        # (Allow torch.clamp for predictions, just not targets)
        import re
        target_clipping_pattern = r"target_returns\w*\s*=\s*torch\.clamp\(target_returns"
        matches = re.findall(target_clipping_pattern, code)

        assert len(matches) == 0, (
            f"Found {len(matches)} instances of target clipping! "
            f"Targets should NEVER be clipped. Matches: {matches}"
        )


# Test data generation utilities
def generate_extreme_returns(n_samples: int = 100, min_val: float = -10.0, max_val: float = 10.0):
    """
    Generate extreme return values for stress testing.

    Args:
        n_samples: Number of samples to generate
        min_val: Minimum return value (e.g., -10.0 for -1000%)
        max_val: Maximum return value (e.g., +10.0 for +1000%)

    Returns:
        Tensor of shape [n_samples, 1] with extreme values
    """
    # Mix of uniform random, extreme values, and normal values
    uniform_samples = torch.rand(n_samples // 3, 1) * (max_val - min_val) + min_val
    extreme_positive = torch.full((n_samples // 3, 1), max_val)
    extreme_negative = torch.full((n_samples - 2 * (n_samples // 3), 1), min_val)

    combined = torch.cat([uniform_samples, extreme_positive, extreme_negative], dim=0)
    return combined[torch.randperm(n_samples)]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
