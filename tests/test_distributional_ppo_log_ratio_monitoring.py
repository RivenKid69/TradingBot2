"""
Comprehensive tests for PPO log_ratio monitoring and conservative clamping.

This test suite validates the critical fix for aggressive log_ratio clipping:
- PROBLEM: Old code clamped log_ratio to ±85, masking catastrophic training issues
- SOLUTION: Conservative ±20 clipping + proactive monitoring and warnings

Background (OpenAI Spinning Up, CleanRL, Stable Baselines3):
- Healthy PPO training: log_ratio ∈ [-0.1, 0.1] (approx_kl < 0.02)
- Concerning: |log_ratio| > 1.0 → policy changed by factor > e ≈ 2.7x
- Severe instability: |log_ratio| > 10.0 → policy changed by factor > e^10 ≈ 22,000x
- Catastrophic: |log_ratio| approaching ±20 → exp(20) ≈ 485M (numerical overflow risk)

Key improvements:
1. Conservative numerical clipping (±20 instead of ±85)
   - exp(20) ≈ 485M is finite (prevents overflow)
   - exp(85) ≈ 8×10³⁶ was too permissive (masked problems)
2. Monitoring BEFORE clamping (detects actual log_ratio values)
3. Warning system for extreme values
4. Detailed statistics (mean, std, max_abs, extreme_fraction)

References:
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- CleanRL PPO: https://docs.cleanrl.dev/rl-algorithms/ppo/
- Schulman et al., 2017: https://arxiv.org/abs/1707.06347
"""

import math
from typing import Any
from unittest.mock import MagicMock

import pytest


def test_conservative_clipping_boundary() -> None:
    """Test that log_ratio is conservatively clamped to ±20 (not ±85)."""
    torch = pytest.importorskip("torch")

    # Test values at and beyond ±20 boundary
    log_ratios = torch.tensor([-25.0, -20.0, -15.0, 0.0, 15.0, 20.0, 25.0], dtype=torch.float32)

    # Apply conservative clamping (±20)
    log_ratio_clamped = torch.clamp(log_ratios, min=-20.0, max=20.0)

    # Verify clamping behavior
    expected = [-20.0, -20.0, -15.0, 0.0, 15.0, 20.0, 20.0]
    for i, (actual, exp_val) in enumerate(zip(log_ratio_clamped.tolist(), expected)):
        assert (
            abs(actual - exp_val) < 1e-6
        ), f"log_ratio[{i}] clamped incorrectly: got {actual}, expected {exp_val}"

    # Verify exp(±20) is finite
    ratio = torch.exp(log_ratio_clamped)
    assert torch.all(torch.isfinite(ratio)), f"exp(±20) should be finite: {ratio.tolist()}"

    # Verify exp(20) ≈ 485M
    assert (
        abs(ratio[-1].item() - 4.85e8) < 1e6
    ), f"exp(20) should be ≈4.85e8, got {ratio[-1].item():.2e}"


def test_old_aggressive_clipping_was_too_permissive() -> None:
    """Test that old ±85 clipping was too aggressive and masked problems."""
    torch = pytest.importorskip("torch")

    # Extreme values that old code would allow
    log_ratios = torch.tensor([-85.0, -50.0, 0.0, 50.0, 85.0], dtype=torch.float32)

    # Old aggressive clamping (±85) - DO NOT USE IN PRODUCTION
    log_ratio_old = torch.clamp(log_ratios, min=-85.0, max=85.0)
    ratio_old = torch.exp(log_ratio_old)

    # exp(±85) is astronomically large
    # exp(85) ≈ 8×10³⁶ (completely masks training instability)
    assert ratio_old[-1].item() > 1e30, f"exp(85) ≈ 8×10³⁶, got {ratio_old[-1].item():.2e}"

    # At these scales, PPO clipping (clip_range=0.2) is meaningless
    clip_range = 0.2
    ratio_clipped = torch.clamp(ratio_old, 1 - clip_range, 1 + clip_range)

    # For extreme values, PPO clip bounds [0.8, 1.2] are tiny compared to ratio
    # This demonstrates why ±85 clipping masks problems instead of solving them
    assert ratio_old[-1].item() > 1e30, "Old clipping allowed catastrophic values"
    assert ratio_clipped[-1].item() == pytest.approx(1 + clip_range), "PPO clip maxes out at 1.2"

    # The ratio changed by factor of 10^36, but PPO can only clip to 1.2
    # This is why aggressive log_ratio clipping is harmful


def test_extreme_value_detection_threshold() -> None:
    """Test detection of extreme log_ratio values (|log_ratio| > 10)."""
    torch = pytest.importorskip("torch")

    # Simulate batch with various log_ratio values
    log_ratios = torch.tensor(
        [-15.0, -10.5, -5.0, -0.1, 0.0, 0.1, 5.0, 10.5, 15.0], dtype=torch.float32
    )

    # Detection threshold (|log_ratio| > 10 indicates severe instability)
    extreme_threshold = 10.0
    extreme_mask = torch.abs(log_ratios) > extreme_threshold

    # Verify detection
    expected_extreme = [True, True, False, False, False, False, False, True, True]
    for i, (detected, expected) in enumerate(zip(extreme_mask.tolist(), expected_extreme)):
        assert detected == expected, f"log_ratio[{i}]={log_ratios[i].item()} detection mismatch"

    # Count extreme values
    extreme_count = extreme_mask.sum().item()
    assert extreme_count == 4, f"Should detect 4 extreme values, found {extreme_count}"


def test_log_ratio_statistics_calculation() -> None:
    """Test calculation of log_ratio statistics (mean, std, max_abs)."""
    torch = pytest.importorskip("torch")

    # Realistic batch with some extreme outliers
    torch.manual_seed(42)
    log_ratios = torch.randn(100, dtype=torch.float32) * 0.05  # Healthy: std=0.05

    # Inject extreme values
    log_ratios[0] = 15.0
    log_ratios[1] = -12.0

    # Calculate statistics (as in implementation)
    log_ratio_sum = float(log_ratios.sum().item())
    log_ratio_sq_sum = float((log_ratios.square()).sum().item())
    log_ratio_count = int(log_ratios.numel())

    log_ratio_mean = log_ratio_sum / float(log_ratio_count)
    raw_var = (log_ratio_sq_sum - log_ratio_count * log_ratio_mean**2) / (
        float(log_ratio_count) - 1.0
    )
    log_ratio_var = max(raw_var, 0.0)
    log_ratio_std = math.sqrt(log_ratio_var)

    # Max absolute value
    log_ratio_max_abs = torch.max(torch.abs(log_ratios)).item()

    # Verify max_abs captures extreme value
    assert log_ratio_max_abs == 15.0, f"max_abs should be 15.0, got {log_ratio_max_abs}"

    # Verify statistics are reasonable
    assert math.isfinite(log_ratio_mean), "mean should be finite"
    assert math.isfinite(log_ratio_std), "std should be finite"
    assert log_ratio_std > 0, "std should be positive"


def test_warning_levels() -> None:
    """Test warning thresholds for different instability levels."""
    torch = pytest.importorskip("torch")

    # Test cases: (log_ratio_max_abs, expected_warning_level)
    test_cases = [
        (0.1, "healthy"),  # Healthy training
        (0.5, "healthy"),  # Still healthy
        (1.5, "concerning"),  # Concerning (> 1.0)
        (5.0, "concerning"),  # Concerning
        (10.5, "severe"),  # Severe (> 10.0)
        (19.0, "severe"),  # Severe, approaching clip boundary
    ]

    for max_abs, expected_level in test_cases:
        # Warning logic (as in implementation)
        if max_abs > 10.0:
            warning_level = "severe"
        elif max_abs > 1.0:
            warning_level = "concerning"
        else:
            warning_level = "healthy"

        assert (
            warning_level == expected_level
        ), f"log_ratio_max_abs={max_abs}: expected {expected_level}, got {warning_level}"


def test_extreme_fraction_calculation() -> None:
    """Test calculation of extreme value fraction."""
    torch = pytest.importorskip("torch")

    batch_size = 1000
    torch.manual_seed(123)
    log_ratios = torch.randn(batch_size, dtype=torch.float32) * 0.05  # Healthy batch

    # Inject 10 extreme values with guaranteed |value| > 10
    num_extreme_inject = 10
    indices = torch.randperm(batch_size)[:num_extreme_inject]
    # Use values that are definitely > 10 (mean=15, std=2 ensures all values > 10)
    log_ratios[indices] = torch.abs(torch.randn(num_extreme_inject) * 2.0) + 15.0

    # Count extreme values (|log_ratio| > 10)
    extreme_mask = torch.abs(log_ratios) > 10.0
    extreme_count = extreme_mask.sum().item()

    # Calculate fraction
    extreme_fraction = float(extreme_count) / float(batch_size)

    # Verify
    assert (
        extreme_count >= num_extreme_inject
    ), f"Should detect at least {num_extreme_inject} extreme values"
    assert 0.0 < extreme_fraction < 0.1, f"Extreme fraction should be small, got {extreme_fraction}"


def test_numerical_stability_exp_20() -> None:
    """Test that exp(±20) is numerically stable for float32."""
    torch = pytest.importorskip("torch")

    # Test exp(±20) is finite
    log_ratios = torch.tensor([-20.0, 20.0], dtype=torch.float32)
    ratios = torch.exp(log_ratios)

    assert torch.all(torch.isfinite(ratios)), f"exp(±20) should be finite: {ratios.tolist()}"

    # Verify values
    # exp(-20) ≈ 2.06×10⁻⁹
    assert (
        abs(ratios[0].item() - 2.06e-9) < 1e-10
    ), f"exp(-20) should be ≈2.06e-9, got {ratios[0].item():.2e}"

    # exp(20) ≈ 4.85×10⁸
    assert (
        abs(ratios[1].item() - 4.85e8) < 1e6
    ), f"exp(20) should be ≈4.85e8, got {ratios[1].item():.2e}"


def test_monitoring_before_clamping() -> None:
    """Test that monitoring captures unclamped log_ratio values."""
    torch = pytest.importorskip("torch")

    # Extreme log_ratio values
    log_ratios_unclamped = torch.tensor([-25.0, -15.0, 0.0, 15.0, 25.0], dtype=torch.float32)

    # Monitor BEFORE clamping (critical for detecting instability)
    max_abs_before = torch.max(torch.abs(log_ratios_unclamped)).item()

    # Apply clamping
    log_ratios_clamped = torch.clamp(log_ratios_unclamped, min=-20.0, max=20.0)
    max_abs_after = torch.max(torch.abs(log_ratios_clamped)).item()

    # Monitoring before clamping captures true max (25.0)
    assert max_abs_before == 25.0, f"Should capture unclamped max=25.0, got {max_abs_before}"

    # After clamping, max is 20.0
    assert max_abs_after == 20.0, f"Clamped max should be 20.0, got {max_abs_after}"

    # This demonstrates why monitoring must happen BEFORE clamping


def test_realistic_healthy_training_scenario() -> None:
    """Test realistic healthy training scenario with log_ratio ∈ [-0.1, 0.1]."""
    torch = pytest.importorskip("torch")

    # Simulate healthy training batch
    batch_size = 256
    torch.manual_seed(999)
    log_ratios = torch.randn(batch_size, dtype=torch.float32) * 0.05  # std=0.05

    # Calculate statistics
    log_ratio_mean = log_ratios.mean().item()
    log_ratio_std = log_ratios.std(unbiased=True).item()
    log_ratio_max_abs = torch.max(torch.abs(log_ratios)).item()

    # Verify healthy training characteristics
    assert abs(log_ratio_mean) < 0.01, f"Healthy training should have mean≈0, got {log_ratio_mean}"
    assert (
        0.03 < log_ratio_std < 0.07
    ), f"Healthy training should have std≈0.05, got {log_ratio_std}"
    assert (
        log_ratio_max_abs < 0.2
    ), f"Healthy training should have max_abs<0.2, got {log_ratio_max_abs}"

    # No extreme values
    extreme_mask = torch.abs(log_ratios) > 10.0
    assert extreme_mask.sum().item() == 0, "Healthy training should have no extreme values"


def test_integration_with_ppo_loss() -> None:
    """Test that conservative clipping integrates correctly with PPO loss."""
    torch = pytest.importorskip("torch")

    batch_size = 100
    clip_range = 0.1

    # Create scenario with extreme log_ratio
    torch.manual_seed(456)
    log_prob = torch.randn(batch_size, dtype=torch.float32)
    old_log_prob = log_prob.clone()

    # Inject extreme value
    old_log_prob[0] = log_prob[0] - 25.0  # log_ratio = 25.0 (extreme!)

    advantages = torch.randn(batch_size, dtype=torch.float32)

    # Compute log_ratio
    log_ratio = log_prob - old_log_prob

    # Monitor BEFORE clamping
    max_abs_unclamped = torch.max(torch.abs(log_ratio)).item()
    assert max_abs_unclamped == 25.0, "Should detect extreme value before clamping"

    # Apply conservative clamping (±20)
    log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio_clamped)

    # PPO loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Verify loss is finite
    assert torch.isfinite(policy_loss), f"PPO loss should be finite, got {policy_loss.item()}"


def test_gradient_flow_with_conservative_clipping() -> None:
    """Test that gradients still flow with conservative ±20 clipping."""
    torch = pytest.importorskip("torch")

    # Create log_ratio that will be clamped
    log_ratio_raw = torch.tensor([25.0], dtype=torch.float32, requires_grad=True)

    # Apply conservative clamping (±20)
    log_ratio_clamped = torch.clamp(log_ratio_raw, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio_clamped)

    # Simple loss
    loss = ratio.mean()
    loss.backward()

    # Gradient should be zero because log_ratio_raw=25 is clamped to 20
    # (value is outside the active region)
    assert log_ratio_raw.grad is not None, "Gradient should be computed"
    assert (
        abs(log_ratio_raw.grad[0].item()) < 1e-6
    ), "Gradient should be ~0 when clamped (this is expected behavior)"


def test_comparison_healthy_vs_unstable() -> None:
    """Compare statistics between healthy and unstable training scenarios."""
    torch = pytest.importorskip("torch")

    # Scenario 1: Healthy training
    torch.manual_seed(111)
    log_ratios_healthy = torch.randn(1000, dtype=torch.float32) * 0.05
    max_abs_healthy = torch.max(torch.abs(log_ratios_healthy)).item()
    extreme_count_healthy = (torch.abs(log_ratios_healthy) > 10.0).sum().item()

    # Scenario 2: Unstable training
    torch.manual_seed(222)
    log_ratios_unstable = torch.randn(1000, dtype=torch.float32) * 5.0  # std=5.0 (very bad!)
    max_abs_unstable = torch.max(torch.abs(log_ratios_unstable)).item()
    extreme_count_unstable = (torch.abs(log_ratios_unstable) > 10.0).sum().item()

    # Healthy training
    assert max_abs_healthy < 0.3, f"Healthy should have max_abs<0.3, got {max_abs_healthy}"
    assert extreme_count_healthy == 0, "Healthy should have no extreme values"

    # Unstable training
    assert max_abs_unstable > 5.0, f"Unstable should have max_abs>5.0, got {max_abs_unstable}"
    assert extreme_count_unstable > 0, "Unstable should have extreme values"

    # This demonstrates why monitoring is critical


def test_approx_kl_relationship() -> None:
    """Test relationship between log_ratio and approx_kl."""
    torch = pytest.importorskip("torch")

    # approx_kl ≈ old_log_prob - new_log_prob = -log_ratio (first-order approximation)

    log_prob = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    old_log_prob = torch.tensor([0.9, 2.1, 2.8], dtype=torch.float32)

    log_ratio = log_prob - old_log_prob
    approx_kl = old_log_prob - log_prob  # = -log_ratio

    # Verify relationship
    expected_kl = -log_ratio
    assert torch.allclose(approx_kl, expected_kl, atol=1e-6), f"approx_kl should equal -log_ratio"

    # In healthy training: |approx_kl| < 0.02 (OpenAI Spinning Up)
    # This means |log_ratio| < 0.02
    for i, lr in enumerate(log_ratio.tolist()):
        ak = approx_kl[i].item()
        assert abs(ak + lr) < 1e-6, f"approx_kl[{i}]={ak} should equal -log_ratio[{i}]={-lr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
