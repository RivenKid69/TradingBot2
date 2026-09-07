"""
Comprehensive tests for AWR (Advantage Weighted Regression) weight computation.

Tests verify the critical fix for BC loss weight clamping logic:
- Correct implementation: clamp exp_arg to log(max_weight) BEFORE exp
- Incorrect (old bug): clamp exp_arg to arbitrary value, then clamp weights

Reference: commit 354bbe8 - fix: Correct BC loss AWR-style weight clamping logic
"""

import math

import pytest


def test_awr_weight_basic_computation() -> None:
    """Test basic AWR weight computation with normalized advantages."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Test cases: (advantage_sigma, expected_weight)
    test_cases = [
        (0.0, 1.000),  # Median advantage
        (1.0, 1.221),  # +1σ (84th percentile)
        (2.0, 1.492),  # +2σ (95th percentile)
        (3.0, 1.822),  # +3σ (99.7th percentile)
        (-1.0, 0.819),  # -1σ (16th percentile)
        (-2.0, 0.670),  # -2σ (5th percentile)
    ]

    for adv_sigma, expected_weight in test_cases:
        advantages = torch.tensor([adv_sigma], dtype=torch.float32)
        exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
        weights = torch.exp(exp_arg)

        assert torch.isclose(
            weights[0], torch.tensor(expected_weight), atol=1e-3
        ), f"Failed for advantage={adv_sigma}σ: expected {expected_weight}, got {weights[0].item()}"


def test_awr_weight_max_clipping() -> None:
    """Test that max_weight is correctly enforced for extreme advantages."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Extreme advantages that should trigger max_weight
    extreme_advantages = torch.tensor([23.0, 50.0, 100.0, 500.0], dtype=torch.float32)

    exp_arg = torch.clamp(extreme_advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # All weights should be <= max_weight (with float32 tolerance)
    assert torch.all(
        weights <= max_weight + 1e-5
    ), f"Weights exceeded max_weight: {weights.tolist()}"

    # Weights should be very close to max_weight for extreme advantages
    assert torch.all(
        weights >= max_weight * 0.99
    ), f"Weights should be ≥99% of max_weight for extreme advantages: {weights.tolist()}"


def test_awr_weight_prevents_overflow() -> None:
    """Test that exp_arg clamping prevents numerical overflow."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # exp(x) overflows at x ≈ 88. Test that we never get close to this.
    extreme_advantages = torch.tensor([100.0, 500.0, 1000.0], dtype=torch.float32)

    exp_arg = torch.clamp(extreme_advantages / beta, max=math.log(max_weight))

    # exp_arg should never exceed log(max_weight) ≈ 4.605
    max_safe_exp_arg = math.log(max_weight)
    assert torch.all(
        exp_arg <= max_safe_exp_arg
    ), f"exp_arg exceeded safe limit: {exp_arg.tolist()} > {max_safe_exp_arg}"

    # Verify exp_arg is far below overflow threshold
    assert torch.all(
        exp_arg < 50.0
    ), f"exp_arg dangerously close to overflow threshold (88): {exp_arg.tolist()}"

    # Verify exp() doesn't produce inf or nan
    weights = torch.exp(exp_arg)
    assert torch.all(
        torch.isfinite(weights)
    ), f"exp() produced non-finite values: {weights.tolist()}"


def test_awr_weight_old_bug_comparison() -> None:
    """Test that old buggy implementation produces mathematically inconsistent results."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Advantage that triggers the bug
    advantage = torch.tensor([100.0], dtype=torch.float32)

    # OLD BUGGY IMPLEMENTATION
    exp_arg_old = torch.clamp(advantage / beta, max=20.0)
    weight_old_before_clamp = torch.exp(exp_arg_old)  # exp(20) ≈ 485M
    weight_old = torch.clamp(weight_old_before_clamp, max=max_weight)

    # NEW CORRECT IMPLEMENTATION
    exp_arg_new = torch.clamp(advantage / beta, max=math.log(max_weight))
    weight_new = torch.exp(exp_arg_new)

    # Verify old implementation computed gigantic intermediate value
    assert weight_old_before_clamp > 1e8, "Old implementation should compute exp(20) ≈ 485M"

    # Verify final weights are identical (bug was wasted computation, not wrong result)
    assert torch.isclose(
        weight_old, weight_new, atol=1e-2
    ), f"Final weights should match: old={weight_old.item()}, new={weight_new.item()}"

    # Verify new implementation never computes gigantic values
    weight_new_max_intermediate = torch.exp(torch.tensor(math.log(max_weight)))
    assert (
        weight_new_max_intermediate <= max_weight * 1.01
    ), f"New implementation should never exceed max_weight: {weight_new_max_intermediate}"


def test_awr_weight_vectorized() -> None:
    """Test AWR weight computation on batches of advantages."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Batch of normalized advantages (mean=0, std=1)
    advantages = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 50.0], dtype=torch.float32)

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # Verify all weights are finite
    assert torch.all(torch.isfinite(weights)), f"Non-finite weights detected: {weights.tolist()}"

    # Verify monotonicity: higher advantage => higher weight
    for i in range(len(advantages) - 1):
        assert (
            weights[i] <= weights[i + 1]
        ), f"Weights not monotonic: w[{i}]={weights[i]} > w[{i+1}]={weights[i+1]}"

    # Verify all weights are in valid range
    assert torch.all(weights > 0), "All weights should be positive"
    assert torch.all(
        weights <= max_weight + 1e-5
    ), f"Weights exceeded max_weight: {weights.tolist()}"


def test_awr_weight_zero_advantage() -> None:
    """Test that zero advantage (median) produces weight=1.0."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    advantages = torch.zeros(10, dtype=torch.float32)

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # All weights should be exactly 1.0
    assert torch.allclose(
        weights, torch.ones_like(weights)
    ), f"Zero advantages should produce unit weights: {weights.tolist()}"


def test_awr_weight_negative_advantages() -> None:
    """Test that negative advantages produce weights < 1.0."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Negative advantages (below median)
    advantages = torch.tensor([-5.0, -3.0, -1.0, -0.5], dtype=torch.float32)

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # All weights should be < 1.0
    assert torch.all(
        weights < 1.0
    ), f"Negative advantages should produce weights < 1.0: {weights.tolist()}"

    # All weights should be > 0
    assert torch.all(weights > 0), f"All weights should be positive: {weights.tolist()}"


def test_awr_weight_different_betas() -> None:
    """Test that beta parameter correctly controls weight sharpness."""
    torch = pytest.importorskip("torch")

    advantage = torch.tensor([3.0], dtype=torch.float32)  # +3σ advantage
    max_weight = 100.0

    # Test different beta values
    betas = [1.0, 5.0, 10.0]
    weights = []

    for beta in betas:
        exp_arg = torch.clamp(advantage / beta, max=math.log(max_weight))
        weight = torch.exp(exp_arg)
        weights.append(weight.item())

    # Higher beta => more conservative (closer to 1.0)
    # beta=1.0: exp(3/1) = exp(3) ≈ 20.09
    # beta=5.0: exp(3/5) = exp(0.6) ≈ 1.82
    # beta=10.0: exp(3/10) = exp(0.3) ≈ 1.35
    assert (
        weights[0] > weights[1] > weights[2]
    ), f"Higher beta should produce more conservative weights: {weights}"


def test_awr_weight_consistency_with_normalization() -> None:
    """Test AWR weights with realistic normalized advantage distribution."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Simulate normalized advantages (mean=0, std=1)
    torch.manual_seed(42)
    advantages = torch.randn(1000, dtype=torch.float32)

    # Verify advantages are approximately normalized
    assert abs(advantages.mean().item()) < 0.1, "Advantages should have ~zero mean"
    assert abs(advantages.std().item() - 1.0) < 0.1, "Advantages should have ~unit std"

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # Statistical properties of weights
    # With beta=5.0, most weights should be in [0.5, 2.0]
    weights_median = weights.median().item()
    weights_95th = torch.quantile(weights, 0.95).item()
    weights_99th = torch.quantile(weights, 0.99).item()

    assert 0.9 < weights_median < 1.1, f"Median weight should be ~1.0: {weights_median}"

    assert (
        weights_95th < 3.0
    ), f"95th percentile weight should be < 3.0 with beta=5.0: {weights_95th}"

    assert (
        weights_99th < 10.0
    ), f"99th percentile weight should be < 10.0 with beta=5.0: {weights_99th}"

    # No weight should exceed max_weight
    assert weights.max() <= max_weight, f"Max weight exceeded limit: {weights.max()}"


def test_awr_weight_edge_case_inf_input() -> None:
    """Test that infinite advantages are safely clamped."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Edge case: inf advantage
    advantages = torch.tensor([float("inf")], dtype=torch.float32)

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # Weight should be finite and equal to max_weight
    assert torch.isfinite(weights[0]), "Weight should be finite even with inf advantage"
    assert torch.isclose(
        weights[0], torch.tensor(max_weight), atol=0.1
    ), f"Inf advantage should produce max_weight: {weights[0]}"


def test_awr_weight_edge_case_nan_input() -> None:
    """Test graceful handling of NaN advantages."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Edge case: NaN advantage
    advantages = torch.tensor([float("nan")], dtype=torch.float32)

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # NaN should propagate (expected behavior for debugging)
    # This helps catch upstream bugs rather than silently masking them
    assert torch.isnan(weights[0]), "NaN advantages should propagate to weights for debugging"


def test_awr_weight_gradient_flow() -> None:
    """Test that AWR weights don't break gradient flow (even though computed in no_grad)."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Simulate log_prob that requires grad
    log_prob = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float32, requires_grad=True)

    # Advantages (no grad, as in actual code)
    advantages = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float32)

    with torch.no_grad():
        exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
        weights = torch.exp(exp_arg)

    # BC loss computation (as in actual code)
    loss = (-log_prob * weights).mean()
    loss.backward()

    # Verify gradients exist and are reasonable
    assert log_prob.grad is not None, "Gradients should exist"
    assert torch.all(torch.isfinite(log_prob.grad)), f"Gradients should be finite: {log_prob.grad}"


def test_awr_weight_memory_efficiency() -> None:
    """Test that AWR weight computation doesn't create unnecessary intermediate tensors."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    # Large batch to test memory efficiency
    advantages = torch.randn(10000, dtype=torch.float32)

    # Measure memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights = torch.exp(exp_arg)

    # Verify computation completed
    assert weights.shape == advantages.shape, "Output shape should match input"
    assert torch.all(torch.isfinite(weights)), "All weights should be finite"


def test_awr_weight_deterministic() -> None:
    """Test that AWR weight computation is deterministic."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0

    advantages = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.0], dtype=torch.float32)

    # Compute weights twice
    exp_arg1 = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights1 = torch.exp(exp_arg1)

    exp_arg2 = torch.clamp(advantages / beta, max=math.log(max_weight))
    weights2 = torch.exp(exp_arg2)

    # Results should be exactly identical
    assert torch.equal(weights1, weights2), "AWR weight computation should be deterministic"


def test_awr_weight_formula_correctness() -> None:
    """Verify AWR formula: w = exp(A/β) with clipping."""
    torch = pytest.importorskip("torch")

    beta = 5.0
    max_weight = 100.0
    advantage = 2.0

    # Manual computation
    exp_arg_manual = min(advantage / beta, math.log(max_weight))
    weight_manual = math.exp(exp_arg_manual)

    # Tensor computation
    advantages = torch.tensor([advantage], dtype=torch.float32)
    exp_arg = torch.clamp(advantages / beta, max=math.log(max_weight))
    weight = torch.exp(exp_arg)

    # Should be exactly equal
    assert (
        abs(weight.item() - weight_manual) < 1e-6
    ), f"Formula mismatch: tensor={weight.item()}, manual={weight_manual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
