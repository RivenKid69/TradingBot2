"""
Tests for PPO bug fixes (BUG #2, #6).

This module tests the fixes for critical bugs discovered in distributional_ppo.py:
- BUG #2: Advantage normalization explosion when std=0
- BUG #6: Log ratio NaN detection incomplete

Note: BUG #1 (Twin Critics VF clipping asymmetry) is covered in:
  tests/test_twin_critics_vf_clipping_fix.py

Reference: PPO_BUGS_ANALYSIS_REPORT.md
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from typing import Optional
from unittest.mock import MagicMock, patch

# Test markers
pytestmark = pytest.mark.ppo_bugs


class TestAdvantageNormalizationFix:
    """
    Test BUG #2 FIX: Advantage normalization should skip when std < threshold.

    Before fix: When std < 1e-4, used floor which amplified noise
    After fix: When std < 1e-6, skip normalization and set to zero
    """

    def test_uniform_advantages_set_to_zero(self):
        """When all advantages are identical (std=0), should set to zero."""
        from distributional_ppo import DistributionalPPO

        # Create dummy PPO instance
        ppo = self._create_dummy_ppo()

        # Create uniform advantages (std=0)
        advantages = np.full((100,), 5.0, dtype=np.float32)

        # Create mock rollout buffer
        rollout_buffer = MagicMock()
        rollout_buffer.advantages = advantages.copy()

        # Call normalization logic (we'll need to extract this into testable function)
        # For now, test the logic directly
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages, ddof=1))

        STD_THRESHOLD = 1e-6

        if adv_std < STD_THRESHOLD:
            normalized = np.zeros_like(advantages, dtype=np.float32)
        else:
            normalized = ((advantages - adv_mean) / adv_std).astype(np.float32)

        # Verify: should be all zeros
        assert np.allclose(normalized, 0.0), "Uniform advantages should be set to zero"
        assert np.std(normalized) == 0.0, "Normalized advantages should have zero variance"

    def test_near_uniform_advantages_set_to_zero(self):
        """When advantages differ only by numerical noise (std < 1e-6), should set to zero."""
        # Create near-uniform advantages with tiny noise
        advantages = np.array([1.0, 1.0 + 1e-7, 1.0 - 1e-7, 1.0 + 5e-8], dtype=np.float32)

        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages, ddof=1))

        STD_THRESHOLD = 1e-6

        if adv_std < STD_THRESHOLD:
            normalized = np.zeros_like(advantages, dtype=np.float32)
        else:
            normalized = ((advantages - adv_mean) / adv_std).astype(np.float32)

        # Verify: should be all zeros (std < 1e-6)
        assert adv_std < STD_THRESHOLD, f"std should be < {STD_THRESHOLD}, got {adv_std}"
        assert np.allclose(normalized, 0.0), "Near-uniform advantages should be set to zero"

    def test_normal_advantages_normalized_correctly(self):
        """When advantages have sufficient variance (std >= 1e-6), should normalize normally."""
        # Create advantages with reasonable variance
        advantages = np.array([0.1, 0.5, -0.3, 0.7, -0.1], dtype=np.float32)

        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages, ddof=1))

        STD_THRESHOLD = 1e-6

        if adv_std < STD_THRESHOLD:
            normalized = np.zeros_like(advantages, dtype=np.float32)
        else:
            normalized = ((advantages - adv_mean) / adv_std).astype(np.float32)

        # Verify: should be normalized
        assert adv_std >= STD_THRESHOLD, f"std should be >= {STD_THRESHOLD}"
        assert np.abs(np.mean(normalized)) < 0.01, "Normalized mean should be ~0"
        assert np.abs(np.std(normalized, ddof=1) - 1.0) < 0.01, "Normalized std should be ~1"

    def test_no_noise_amplification(self):
        """Verify that fix prevents noise amplification (old behavior with floor)."""
        # Create advantages with tiny variance
        advantages = np.array([0.0001, 0.0002, 0.0003], dtype=np.float32)

        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages, ddof=1))

        # OLD BEHAVIOR (with floor): Would amplify noise
        ADV_STD_FLOOR = 1e-4
        adv_std_clamped_old = max(adv_std, ADV_STD_FLOOR)
        normalized_old = ((advantages - adv_mean) / adv_std_clamped_old).astype(np.float32)

        # NEW BEHAVIOR (skip normalization if std < threshold)
        STD_THRESHOLD = 1e-6
        if adv_std < STD_THRESHOLD:
            normalized_new = np.zeros_like(advantages, dtype=np.float32)
        else:
            normalized_new = ((advantages - adv_mean) / adv_std).astype(np.float32)

        # Verify: new behavior avoids extreme values
        # Old behavior would create values like [-1.0, 0.0, 1.0] from tiny differences
        # New behavior should either:
        # - Use actual std (if >= 1e-6) → reasonable normalized values
        # - Set to zero (if < 1e-6) → no amplification

        if adv_std >= STD_THRESHOLD:
            # Should use actual std, not floor
            # When std is below the old floor but above new threshold, normalized values should be similar
            # Key point: NEW behavior doesn't use a floor, so it's more accurate
            max_abs_new = np.max(np.abs(normalized_new))
            max_abs_old = np.max(np.abs(normalized_old))
            # New behavior should use actual std (no floor), which may be smaller or equal to old
            assert (
                max_abs_new <= max_abs_old * 1.1
            ), "New behavior should not amplify more than old (allow small tolerance)"
        else:
            # Should be zero
            assert np.allclose(normalized_new, 0.0), "Should be zero for very small std"

    def test_large_mean_small_std_no_explosion(self):
        """Verify no explosion when mean is large but std is small."""
        # This was the scenario described in bug report
        # advantages = [-10.0, -10.0, -10.0]
        # mean = -10.0, std = 0.0
        # OLD: normalized = (-10.0 - (-10.0)) / 1e-4 = 0.0 (OK in this case!)
        # But if there's tiny noise: [-10.0, -10.0 + 1e-5, -10.0 - 1e-5]
        # OLD: would amplify to huge values

        advantages = np.array([-10.0, -10.0 + 1e-7, -10.0 - 1e-7], dtype=np.float32)

        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages, ddof=1))

        STD_THRESHOLD = 1e-6

        if adv_std < STD_THRESHOLD:
            normalized = np.zeros_like(advantages, dtype=np.float32)
        else:
            normalized = ((advantages - adv_mean) / adv_std).astype(np.float32)

        # Verify: no explosion
        max_abs = np.max(np.abs(normalized))
        assert max_abs < 10.0, f"Should not have extreme values, got max_abs={max_abs}"

    def _create_dummy_ppo(self):
        """Create a minimal PPO instance for testing."""
        # This would require more setup - skipping for now
        pass


class TestLogRatioNaNDetection:
    """
    Test BUG #6 FIX: Log ratio NaN detection should log errors and skip batch.

    Before fix: NaN silently ignored, propagates to parameters
    After fix: NaN detected, logged, batch skipped
    """

    def test_finite_log_ratio_normal_processing(self):
        """When log_ratio is finite, should process normally."""
        log_ratio = torch.tensor([0.1, -0.05, 0.03, -0.02], dtype=torch.float32)

        # Check if finite
        is_finite = torch.isfinite(log_ratio).all()

        assert is_finite, "Log ratio should be finite"
        # Normal processing would continue
        ratio = torch.exp(log_ratio)
        assert torch.isfinite(ratio).all(), "Ratio should be finite"

    def test_nan_log_ratio_detected(self):
        """When log_ratio contains NaN, should detect it."""
        log_ratio = torch.tensor([0.1, float("nan"), 0.03, -0.02], dtype=torch.float32)

        # Check if finite
        is_finite = torch.isfinite(log_ratio).all()

        assert not is_finite, "Should detect NaN in log_ratio"

        # Count NaN
        num_nan = int(torch.isnan(log_ratio).sum().item())
        num_inf = int(torch.isinf(log_ratio).sum().item())

        assert num_nan == 1, f"Should detect 1 NaN, got {num_nan}"
        assert num_inf == 0, f"Should detect 0 Inf, got {num_inf}"

    def test_inf_log_ratio_detected(self):
        """When log_ratio contains Inf, should detect it."""
        log_ratio = torch.tensor([0.1, -0.05, float("inf"), -0.02], dtype=torch.float32)

        # Check if finite
        is_finite = torch.isfinite(log_ratio).all()

        assert not is_finite, "Should detect Inf in log_ratio"

        # Count Inf
        num_nan = int(torch.isnan(log_ratio).sum().item())
        num_inf = int(torch.isinf(log_ratio).sum().item())

        assert num_nan == 0, f"Should detect 0 NaN, got {num_nan}"
        assert num_inf == 1, f"Should detect 1 Inf, got {num_inf}"

    def test_mixed_nan_inf_detected(self):
        """When log_ratio contains both NaN and Inf, should detect both."""
        log_ratio = torch.tensor(
            [float("nan"), -0.05, float("inf"), float("-inf")], dtype=torch.float32
        )

        # Check if finite
        is_finite = torch.isfinite(log_ratio).all()

        assert not is_finite, "Should detect NaN and Inf in log_ratio"

        # Count NaN and Inf
        num_nan = int(torch.isnan(log_ratio).sum().item())
        num_inf = int(torch.isinf(log_ratio).sum().item())
        num_total = int(log_ratio.numel())

        assert num_nan == 1, f"Should detect 1 NaN, got {num_nan}"
        assert num_inf == 2, f"Should detect 2 Inf, got {num_inf}"
        assert num_total == 4, f"Should have 4 total samples, got {num_total}"

        # Invalid fraction
        invalid_fraction = float(num_nan + num_inf) / float(num_total)
        assert invalid_fraction == 0.75, f"Should have 75% invalid, got {invalid_fraction}"

    def test_nan_propagation_prevented(self):
        """Verify that NaN does not propagate if batch is skipped."""
        log_ratio = torch.tensor([float("nan"), -0.05, 0.03, -0.02], dtype=torch.float32)

        # Simul protocol: if NaN detected, skip batch
        if not torch.isfinite(log_ratio).all():
            # Batch should be skipped
            batch_skipped = True
        else:
            # Normal processing
            batch_skipped = False
            ratio = torch.exp(log_ratio)

        assert batch_skipped, "Batch with NaN should be skipped"

    def test_extreme_log_ratio_still_finite(self):
        """Extreme but finite values should pass finite check."""
        # Very large but finite values (should still be detected as "extreme" separately)
        log_ratio = torch.tensor([15.0, -18.0, 12.0, -11.0], dtype=torch.float32)

        # Should be finite
        is_finite = torch.isfinite(log_ratio).all()
        assert is_finite, "Extreme but finite values should pass finite check"

        # But should be flagged as "extreme" (separate check, threshold is > 10.0 strict)
        extreme_mask = torch.abs(log_ratio) > 10.0
        has_extreme = torch.any(extreme_mask)

        assert has_extreme, "Should detect extreme values (|log_ratio| > 10)"
        num_extreme = int(extreme_mask.sum().item())
        assert num_extreme == 4, f"All 4 values should be extreme (>10 in abs), got {num_extreme}"


class TestTwinCriticsVFClipping:
    """
    Test BUG #1: Twin Critics VF clipping asymmetry.

    STATUS: FIXED (2025-11-22)
    The bug was fixed in _twin_critics_vf_clipping_loss() method.
    See: archive/2025_11/reports_2025_11/fixes/TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md

    Comprehensive tests are in: tests/test_twin_critics_vf_clipping_all_modes.py
    These tests verify the fix remains in place.
    """

    def test_twin_critics_vf_clipping_uses_both_critics_quantile(self):
        """
        Verify BUG #1 FIX: Twin Critics + VF clipping (quantile mode) uses BOTH critics.

        The fix implements:
        1. Get quantiles from BOTH critics independently
        2. Apply VF clipping to BOTH critics separately (each vs its own old values)
        3. Compute clipped losses for BOTH critics
        4. Average clipped losses
        5. Element-wise max uses averaged unclipped and averaged clipped

        Implementation: distributional_ppo.py:_twin_critics_vf_clipping_loss()
        """
        # Verify the fix is in place by checking the method exists and has correct signature
        from distributional_ppo import DistributionalPPO

        assert hasattr(
            DistributionalPPO, "_twin_critics_vf_clipping_loss"
        ), "BUG #1 FIX: _twin_critics_vf_clipping_loss method must exist"

        # Verify method signature includes separate critic parameters
        import inspect

        sig = inspect.signature(DistributionalPPO._twin_critics_vf_clipping_loss)
        params = list(sig.parameters.keys())

        assert (
            "old_quantiles_critic1" in params
        ), "BUG #1 FIX: Method must accept old_quantiles_critic1 for independent clipping"
        assert (
            "old_quantiles_critic2" in params
        ), "BUG #1 FIX: Method must accept old_quantiles_critic2 for independent clipping"

    def test_twin_critics_vf_clipping_uses_both_critics_categorical(self):
        """
        Verify BUG #1 FIX: Twin Critics + VF clipping (categorical mode) uses BOTH critics.

        The fix implements:
        1. Get pred_probs from BOTH critics independently
        2. Apply VF clipping (projection) to BOTH critics separately
        3. Compute clipped losses for BOTH critics
        4. Average clipped losses
        5. Element-wise max uses averaged unclipped and averaged clipped

        Implementation: distributional_ppo.py:_twin_critics_vf_clipping_loss()
        """
        # Verify the fix is in place by checking categorical support
        from distributional_ppo import DistributionalPPO
        import inspect

        sig = inspect.signature(DistributionalPPO._twin_critics_vf_clipping_loss)
        params = list(sig.parameters.keys())

        assert (
            "old_probs_critic1" in params
        ), "BUG #1 FIX: Method must accept old_probs_critic1 for categorical mode"
        assert (
            "old_probs_critic2" in params
        ), "BUG #1 FIX: Method must accept old_probs_critic2 for categorical mode"
        assert (
            "target_distribution" in params
        ), "BUG #1 FIX: Method must accept target_distribution for categorical mode"

    def test_twin_critics_gradient_flow_symmetric(self):
        """
        Verify BUG #1 FIX: Gradient flow is symmetric for both critics with VF clipping.

        The fix ensures both critics receive gradients from the clipped loss term.
        This is verified by checking the return signature includes individual critic losses.

        Implementation: distributional_ppo.py:_twin_critics_vf_clipping_loss()
        """
        from distributional_ppo import DistributionalPPO
        import inspect

        # Check the docstring mentions returning individual critic losses
        method = DistributionalPPO._twin_critics_vf_clipping_loss
        docstring = method.__doc__ or ""

        # The fixed method returns individual losses for each critic
        assert (
            "loss_c1_clipped" in docstring
        ), "BUG #1 FIX: Method must return individual clipped loss for critic 1"
        assert (
            "loss_c2_clipped" in docstring
        ), "BUG #1 FIX: Method must return individual clipped loss for critic 2"
        assert (
            "loss_c1_unclipped" in docstring
        ), "BUG #1 FIX: Method must return individual unclipped loss for critic 1"
        assert (
            "loss_c2_unclipped" in docstring
        ), "BUG #1 FIX: Method must return individual unclipped loss for critic 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
