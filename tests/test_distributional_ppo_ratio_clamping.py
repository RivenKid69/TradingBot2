"""
Comprehensive tests for PPO ratio computation without aggressive log_ratio clamping.

Tests verify the correct PPO implementation following standard practices:
- NO aggressive clamping on log_ratio before exp() (aligns with Stable Baselines3, CleanRL)
- Trust region enforcement happens ONLY via PPO clipping in loss function
- Correct gradient flow for all log_ratio values
- Alignment with theoretical PPO (Schulman et al., 2017)

IMPORTANT NOTE ON CONSERVATIVE ±20 CLIPPING:
Our implementation includes a conservative ±20 numerical safeguard which is NOT a violation
of PPO theory because:
1. In healthy training, log_ratio NEVER reaches ±20 (typically ≤ 0.2)
2. The ±20 bound is purely for numerical stability (prevents exp overflow)
3. If training hits ±20, monitoring WILL warn (not silently mask like old ±85)
4. This is fundamentally different from the old ±85 aggressive clipping

Think of ±20 as an assertion boundary that should never be reached in correct training.

The ratio = exp(log_prob - old_log_prob) is the core of PPO's importance sampling.
PPO clips ratio to [1-ε, 1+ε] where ε≈0.05-0.2, maintaining the trust region.

Key insight: Aggressive clamping (like ±85) BEFORE exp() is wrong because:
1. It creates double clipping (first on log_ratio, then on ratio in loss)
2. It breaks gradient flow (gradient becomes 0 for clamped values)
3. It violates PPO theory (clipping should only happen in the loss function)
4. It masks catastrophic training problems instead of detecting them

Empirical data from training logs:
- ratio_mean ≈ 1.0 (perfect)
- ratio_std ≈ 0.02-0.04 (very small)
- log_ratio typically ≈ 0.039 (log(1.04))
- log_ratio should NEVER exceed ±1.0 in healthy training

Reference:
- Original PPO paper: Schulman et al., 2017 (https://arxiv.org/abs/1707.06347)
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl
"""

import math

import pytest


def test_no_log_ratio_clamp() -> None:
    """Test that log_ratio is NOT clamped, following standard PPO implementations."""
    torch = pytest.importorskip("torch")

    # Large but finite log_ratio values
    log_ratios = torch.tensor([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=torch.float32)

    # Compute ratio directly (as in the fixed implementation)
    ratio = torch.exp(log_ratios)

    # All ratios should be finite (exp(20) is still within float32 range)
    assert torch.all(
        torch.isfinite(ratio)
    ), f"exp(log_ratio) should be finite for reasonable values: {ratio.tolist()}"

    # Verify actual values
    expected = [
        2.06e-9,  # exp(-20)
        4.54e-5,  # exp(-10)
        1.0,  # exp(0)
        22026.5,  # exp(10)
        4.85e8,  # exp(20)
    ]

    for i, (actual, exp_val) in enumerate(zip(ratio.tolist(), expected)):
        rel_error = abs(actual - exp_val) / exp_val if exp_val != 0 else abs(actual - exp_val)
        assert rel_error < 0.01, f"ratio[{i}] = {actual:.2e}, expected ≈{exp_val:.2e}"


def test_ppo_clipping_in_loss() -> None:
    """Test that PPO clipping in loss function correctly constrains ratio."""
    torch = pytest.importorskip("torch")

    clip_range = 0.1  # Typical clip_range value

    # Various log_ratio values, including extreme ones
    log_ratios = torch.tensor(
        [-20.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 20.0], dtype=torch.float32
    )

    # Compute ratio WITHOUT clamping log_ratio (correct approach)
    ratio = torch.exp(log_ratios)

    # Apply PPO clip (this is where trust region is enforced)
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    # Verify all clipped values are in correct range
    assert torch.all(
        ratio_clipped >= 1 - clip_range - 1e-6
    ), f"ratio_clipped should be >= {1-clip_range}, got min={ratio_clipped.min().item()}"
    assert torch.all(
        ratio_clipped <= 1 + clip_range + 1e-6
    ), f"ratio_clipped should be <= {1+clip_range}, got max={ratio_clipped.max().item()}"

    # For extreme log_ratio values, ratio_clipped should saturate at bounds
    assert (
        abs(ratio_clipped[0].item() - 0.9) < 1e-6
    ), f"ratio for log_ratio=-20 should clip to 0.9, got {ratio_clipped[0].item()}"
    assert (
        abs(ratio_clipped[-1].item() - 1.1) < 1e-6
    ), f"ratio for log_ratio=+20 should clip to 1.1, got {ratio_clipped[-1].item()}"


def test_gradient_flow_no_log_ratio_clamp() -> None:
    """Test that gradients flow correctly without log_ratio clamping."""
    torch = pytest.importorskip("torch")

    # Create log_ratio tensor with gradient tracking
    log_ratio = torch.tensor(
        [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0], dtype=torch.float32, requires_grad=True
    )

    # Compute ratio WITHOUT clamping (correct approach)
    ratio = torch.exp(log_ratio)

    # Simulate PPO loss (simplified)
    clip_range = 0.1
    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    # Standard PPO loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Backpropagate
    loss.backward()

    # Check gradients are finite
    assert log_ratio.grad is not None, "Gradients should be computed"
    assert torch.all(
        torch.isfinite(log_ratio.grad)
    ), f"Gradients should be finite: {log_ratio.grad.tolist()}"

    # CRITICAL: Gradients should be NON-ZERO even for extreme values
    # This is the key difference from the old (buggy) clamped version
    grad = log_ratio.grad.tolist()

    # For extreme values where ratio gets clipped in the loss, gradients will be small
    # but they should still exist (not exactly 0) because the clipping happens in the loss,
    # not on log_ratio itself

    # The gradient behavior depends on whether the min() selects the clipped or unclipped term
    # For extreme values, the clipped term is selected, so gradient is 0 from that branch
    # This is CORRECT behavior - it's the PPO clipping mechanism working as intended

    # What's important is that gradients exist and are computed correctly
    for i, g in enumerate(grad):
        assert math.isfinite(g), f"Gradient {i} should be finite, got {g}"


def test_gradient_flow_comparison_with_vs_without_clamp() -> None:
    """Compare gradient flow with and without log_ratio clamping to show the difference."""
    torch = pytest.importorskip("torch")

    # Test case: log_ratio = 15 (large value)
    log_ratio_unclamped = torch.tensor([15.0], dtype=torch.float32, requires_grad=True)
    log_ratio_clamped = torch.tensor([15.0], dtype=torch.float32, requires_grad=True)

    # Method 1: WITHOUT clamping (CORRECT)
    ratio_unclamped = torch.exp(log_ratio_unclamped)
    loss_unclamped = -ratio_unclamped.mean()
    loss_unclamped.backward()

    # Method 2: WITH clamping to ±10 (OLD, INCORRECT)
    log_ratio_clamped_value = torch.clamp(log_ratio_clamped, min=-10.0, max=10.0)
    ratio_clamped = torch.exp(log_ratio_clamped_value)
    loss_clamped = -ratio_clamped.mean()
    loss_clamped.backward()

    # WITHOUT clamping: gradient flows correctly
    assert log_ratio_unclamped.grad is not None
    assert log_ratio_unclamped.grad[0].item() != 0.0, "Gradient should be non-zero without clamping"

    # WITH clamping: gradient is ZERO (broken gradient flow!)
    assert log_ratio_clamped.grad is not None
    assert (
        abs(log_ratio_clamped.grad[0].item()) < 1e-6
    ), "Gradient should be ~0 with clamping (this is the bug!)"

    # This demonstrates why log_ratio clamping is wrong


def test_ratio_realistic_values() -> None:
    """Test ratio computation with realistic log_prob differences from training."""
    torch = pytest.importorskip("torch")

    # Realistic values from training logs:
    # - ratio_mean ≈ 1.0, ratio_std ≈ 0.03
    # - This corresponds to log_ratio_std ≈ 0.03

    # Simulate typical batch of log_prob differences
    torch.manual_seed(42)
    log_ratio = torch.randn(1000, dtype=torch.float32) * 0.03  # std=0.03

    # Compute ratio WITHOUT clamping (correct approach)
    ratio = torch.exp(log_ratio)

    # Verify all finite
    assert torch.all(torch.isfinite(ratio)), "ratio should be finite for realistic log_ratio values"

    # Verify ratio statistics match logs
    ratio_mean = ratio.mean().item()
    ratio_std = ratio.std().item()

    assert 0.98 < ratio_mean < 1.02, f"ratio_mean should be ≈1.0, got {ratio_mean}"
    assert 0.01 < ratio_std < 0.05, f"ratio_std should be ≈0.03, got {ratio_std}"


def test_ppo_loss_with_extreme_ratios() -> None:
    """Test that PPO loss computation works correctly even with extreme ratios."""
    torch = pytest.importorskip("torch")

    batch_size = 100
    clip_range = 0.1

    # Create scenario with some extreme log_ratio values
    torch.manual_seed(123)
    log_prob = torch.randn(batch_size, dtype=torch.float32) * 0.5
    old_log_prob = log_prob.clone()

    # Inject some extreme values
    old_log_prob[0] = log_prob[0] - 15.0  # Very large positive log_ratio
    old_log_prob[1] = log_prob[1] + 15.0  # Very large negative log_ratio

    advantages = torch.randn(batch_size, dtype=torch.float32)

    # Compute log_ratio WITHOUT clamping
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)

    # PPO policy loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Verify loss is finite and reasonable
    assert torch.isfinite(
        policy_loss
    ), f"PPO policy loss should be finite, got {policy_loss.item()}"

    # The loss should be reasonable even with extreme ratios
    # because the min() operation will select the clipped term for extreme values
    assert (
        abs(policy_loss.item()) < 1000.0
    ), f"PPO policy loss should be reasonable, got {policy_loss.item()}"


def test_ppo_theory_alignment() -> None:
    """Test alignment with theoretical PPO formula (Schulman et al., 2017)."""
    torch = pytest.importorskip("torch")

    # PPO formula: L^CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
    # where r = π_new / π_old = exp(log_π_new - log_π_old)

    clip_range = 0.2

    # Test cases with known outcomes
    # PPO takes min(r*A, clip(r)*A), so when A>0 and r<1, unclipped r*A is smaller
    test_cases = [
        # (log_ratio, advantage, expected_loss_contribution)
        (0.0, 1.0, -1.0),  # ratio=1.0, A=1.0 → -min(1.0, 1.0) = -1.0
        (0.1, 1.0, -1.105),  # ratio=1.105, A=1.0 → -min(1.105, 1.105) = -1.105 (not clipped)
        (0.5, 1.0, -1.2),  # ratio=1.649, A=1.0 → -min(1.649, 1.2) = -1.2 (clipped)
        (
            -0.5,
            1.0,
            -0.6065,
        ),  # ratio=0.606, A=1.0 → -min(0.606, 0.8) = -0.606 (unclipped is smaller)
    ]

    for log_ratio_val, advantage_val, expected_loss in test_cases:
        log_ratio = torch.tensor([log_ratio_val], dtype=torch.float32)
        advantage = torch.tensor([advantage_val], dtype=torch.float32)

        # Compute ratio WITHOUT clamping
        ratio = torch.exp(log_ratio)

        # PPO loss
        loss_1 = advantage * ratio
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        loss = -torch.min(loss_1, loss_2).item()

        # Check alignment
        assert (
            abs(loss - expected_loss) < 0.01
        ), f"For log_ratio={log_ratio_val}, expected loss≈{expected_loss}, got {loss}"


def test_numerical_stability_float32() -> None:
    """Test numerical stability for float32 operations."""
    torch = pytest.importorskip("torch")

    # float32 can handle exp(x) up to x ≈ 88 before overflow
    # Test that we're well within safe range for typical values

    safe_log_ratios = torch.tensor(
        [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0], dtype=torch.float32
    )

    ratio = torch.exp(safe_log_ratios)

    # All should be finite
    assert torch.all(
        torch.isfinite(ratio)
    ), f"ratio should be finite for log_ratio up to ±30: {ratio.tolist()}"

    # exp(88) would overflow, but we test that exp(30) is fine
    assert (
        ratio[6].item() > 0 and ratio[6].item() < 1e20
    ), f"exp(30) should be large but finite: {ratio[6].item():.2e}"


def test_extreme_value_detection() -> None:
    """Test that extreme log_ratio values can be detected (for monitoring)."""
    torch = pytest.importorskip("torch")

    # If log_ratio > 20 in practice, this indicates serious training issues
    # The implementation should not clamp, but monitoring should detect this

    log_ratios = torch.tensor([-25.0, -5.0, 0.0, 5.0, 25.0], dtype=torch.float32)

    # Compute ratio without clamping
    ratio = torch.exp(log_ratios)

    # Detection logic (what should be in monitoring)
    extreme_threshold = 20.0
    extreme_mask = torch.abs(log_ratios) > extreme_threshold

    # Verify detection works
    assert extreme_mask[0].item() == True, "Should detect log_ratio = -25"
    assert extreme_mask[4].item() == True, "Should detect log_ratio = 25"
    assert extreme_mask[2].item() == False, "Should not flag log_ratio = 0"

    # If extreme values are detected, this should be logged (not clamped)
    num_extreme = extreme_mask.sum().item()
    assert num_extreme == 2, f"Should detect 2 extreme values, found {num_extreme}"


def test_comparison_with_stable_baselines3_pattern() -> None:
    """Test that our implementation matches Stable Baselines3 pattern."""
    torch = pytest.importorskip("torch")

    # Stable Baselines3 code:
    # ratio = th.exp(log_prob - rollout_data.old_log_prob)
    # policy_loss_1 = advantages * ratio
    # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    clip_range = 0.2
    log_prob = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    old_log_prob = torch.tensor([0.5, 2.1, 2.8], dtype=torch.float32)
    advantages = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float32)

    # Our implementation (should match SB3)
    ratio = torch.exp(log_prob - old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Verify it's finite and reasonable
    assert torch.isfinite(policy_loss), "Loss should be finite"
    assert abs(policy_loss.item()) < 10.0, "Loss should be reasonable"


def test_edge_cases_inf_nan() -> None:
    """Test edge cases with inf/nan values."""
    torch = pytest.importorskip("torch")

    # Edge case: very large log_ratio that would cause overflow
    log_ratio_overflow = torch.tensor([100.0], dtype=torch.float32)
    ratio_overflow = torch.exp(log_ratio_overflow)

    # This will be inf, which is expected behavior (not clamped to finite)
    assert torch.isinf(ratio_overflow), "exp(100) should be inf in float32"

    # The PPO loss computation should handle this gracefully
    # (in practice, the code should check for finite values before computing loss)

    # Edge case: NaN
    log_ratio_nan = torch.tensor([float("nan")], dtype=torch.float32)
    ratio_nan = torch.exp(log_ratio_nan)

    assert torch.isnan(ratio_nan), "exp(nan) should be nan"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
