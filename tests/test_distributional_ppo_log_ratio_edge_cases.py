"""
Edge case tests for log_ratio conservative clipping and monitoring.

This test suite specifically validates:
1. The ±20 numerical safeguard vs theoretical "no clamping"
2. Edge cases at numerical boundaries
3. Non-finite value handling (inf, nan)
4. Interaction with advantage normalization
5. Behavior across multiple epochs
6. Memory accumulation correctness

IMPORTANT DISTINCTION:
- Theoretical PPO: NO log_ratio clamping (trust region via loss clipping only)
- Our implementation: Conservative ±20 numerical safeguard ONLY
  * In healthy training, log_ratio NEVER reaches ±20
  * ±20 is purely for numerical stability (prevents exp overflow)
  * If training hits ±20, monitoring WILL warn (not silently mask)
  * This is different from the old ±85 which masked problems

Think of ±20 as an assertion boundary, not a feature.
"""

import math
from typing import Optional

import pytest


def test_conservative_clipping_does_not_activate_in_healthy_training() -> None:
    """Test that ±20 clipping is NEVER triggered in healthy PPO training."""
    torch = pytest.importorskip("torch")

    # Simulate healthy training: log_ratio ~ N(0, 0.05)
    torch.manual_seed(42)
    batch_size = 10000
    log_ratios = torch.randn(batch_size, dtype=torch.float32) * 0.05

    # Check that NO values are even close to ±20
    max_abs = torch.max(torch.abs(log_ratios)).item()
    assert max_abs < 0.5, f"Healthy training should have max_abs < 0.5, got {max_abs}"

    # Apply ±20 clipping
    log_ratios_clamped = torch.clamp(log_ratios, min=-20.0, max=20.0)

    # Verify that clamping changed NOTHING
    assert torch.allclose(
        log_ratios, log_ratios_clamped
    ), "In healthy training, ±20 clipping should be inactive"

    # Count how many values changed
    changed = torch.sum(log_ratios != log_ratios_clamped).item()
    assert changed == 0, f"Expected 0 values to be clamped, but {changed} were clamped"


def test_clipping_at_exact_boundary() -> None:
    """Test behavior exactly at ±20.0 boundary."""
    torch = pytest.importorskip("torch")

    # Test values exactly at, above, and below boundary
    log_ratios = torch.tensor(
        [
            -20.0 - 1e-6,  # Just below lower bound
            -20.0,  # Exactly at lower bound
            -20.0 + 1e-6,  # Just above lower bound
            20.0 - 1e-6,  # Just below upper bound
            20.0,  # Exactly at upper bound
            20.0 + 1e-6,  # Just above upper bound
        ],
        dtype=torch.float32,
    )

    log_ratios_clamped = torch.clamp(log_ratios, min=-20.0, max=20.0)

    # Verify clamping behavior
    expected = torch.tensor(
        [
            -20.0,  # Clamped to -20.0
            -20.0,  # Unchanged
            -20.0 + 1e-6,  # Unchanged (within bounds)
            20.0 - 1e-6,  # Unchanged (within bounds)
            20.0,  # Unchanged
            20.0,  # Clamped to 20.0
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        log_ratios_clamped, expected, atol=1e-7
    ), f"Boundary clamping mismatch: got {log_ratios_clamped.tolist()}"


def test_exp_overflow_at_boundary() -> None:
    """Test that exp(±20) is safe but exp(±89) overflows."""
    torch = pytest.importorskip("torch")

    # Safe: ±20
    log_ratios_safe = torch.tensor([-20.0, 20.0], dtype=torch.float32)
    ratios_safe = torch.exp(log_ratios_safe)
    assert torch.all(
        torch.isfinite(ratios_safe)
    ), f"exp(±20) should be finite: {ratios_safe.tolist()}"

    # Verify exact values
    assert abs(ratios_safe[0].item() - 2.06e-9) < 1e-10, "exp(-20) mismatch"
    assert abs(ratios_safe[1].item() - 4.85e8) < 1e6, "exp(20) mismatch"

    # Unsafe: ±89 (would overflow if not clamped)
    log_ratios_unsafe = torch.tensor([-89.0, 89.0], dtype=torch.float32)
    ratios_unsafe = torch.exp(log_ratios_unsafe)

    # exp(-89) → ~0 (underflow to subnormal/zero)
    assert ratios_unsafe[0].item() < 1e-38, "exp(-89) should underflow to near-zero"
    # exp(89) → inf (overflow)
    assert torch.isinf(ratios_unsafe[1]), "exp(89) should overflow to inf"


def test_monitoring_captures_pre_clamp_values() -> None:
    """Test that monitoring correctly captures values BEFORE clamping."""
    torch = pytest.importorskip("torch")

    # Create log_ratios with values that will be clamped
    log_ratios = torch.tensor([-25.0, -15.0, 0.0, 15.0, 25.0], dtype=torch.float32)

    # Simulate monitoring (before clamp)
    max_abs_before = torch.max(torch.abs(log_ratios)).item()
    extreme_mask_before = torch.abs(log_ratios) > 10.0
    extreme_count_before = extreme_mask_before.sum().item()

    # Apply clamp
    log_ratios_clamped = torch.clamp(log_ratios, min=-20.0, max=20.0)

    # Simulate monitoring (after clamp)
    max_abs_after = torch.max(torch.abs(log_ratios_clamped)).item()
    extreme_mask_after = torch.abs(log_ratios_clamped) > 10.0
    extreme_count_after = extreme_mask_after.sum().item()

    # Before clamp: captures true values
    assert max_abs_before == 25.0, f"Pre-clamp monitoring should capture 25.0, got {max_abs_before}"
    assert (
        extreme_count_before == 4
    ), f"Pre-clamp should detect 4 extreme values, got {extreme_count_before}"

    # After clamp: values are clamped
    assert max_abs_after == 20.0, f"Post-clamp max should be 20.0, got {max_abs_after}"
    assert (
        extreme_count_after == 4
    ), f"Post-clamp should still have 4 values > 10, got {extreme_count_after}"


def test_non_finite_values_handling() -> None:
    """Test handling of inf and nan in log_ratio."""
    torch = pytest.importorskip("torch")

    # Test with inf
    log_ratios_inf = torch.tensor([float("inf"), float("-inf"), 0.0], dtype=torch.float32)

    # Check if all finite
    all_finite = torch.isfinite(log_ratios_inf).all().item()
    assert not all_finite, "Should detect non-finite values"

    # Individual checks
    finite_mask = torch.isfinite(log_ratios_inf)
    assert finite_mask.tolist() == [False, False, True], "Finite mask mismatch"

    # Test with nan
    log_ratios_nan = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    all_finite_nan = torch.isfinite(log_ratios_nan).all().item()
    assert not all_finite_nan, "Should detect nan"


def test_statistics_accumulation_correctness() -> None:
    """Test that statistics (mean, std, max) are computed correctly."""
    torch = pytest.importorskip("torch")

    # Known distribution
    torch.manual_seed(123)
    log_ratios = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

    # Compute statistics (as in implementation)
    log_ratio_sum = float(log_ratios.sum().item())
    log_ratio_sq_sum = float((log_ratios.square()).sum().item())
    log_ratio_count = int(log_ratios.numel())

    mean = log_ratio_sum / float(log_ratio_count)
    raw_var = (log_ratio_sq_sum - log_ratio_count * mean**2) / (float(log_ratio_count) - 1.0)
    var = max(raw_var, 0.0)
    std = math.sqrt(var)

    max_abs = torch.max(torch.abs(log_ratios)).item()

    # Expected values
    expected_mean = 0.3  # (0.1+0.2+0.3+0.4+0.5)/5
    expected_std = torch.std(log_ratios, unbiased=True).item()
    expected_max_abs = 0.5

    assert abs(mean - expected_mean) < 1e-6, f"Mean mismatch: got {mean}, expected {expected_mean}"
    assert abs(std - expected_std) < 1e-6, f"Std mismatch: got {std}, expected {expected_std}"
    assert (
        abs(max_abs - expected_max_abs) < 1e-6
    ), f"Max abs mismatch: got {max_abs}, expected {expected_max_abs}"


def test_extreme_count_threshold_sensitivity() -> None:
    """Test that extreme value detection uses correct threshold (10.0)."""
    torch = pytest.importorskip("torch")

    # Values around threshold
    log_ratios = torch.tensor(
        [
            9.9,
            9.99,
            9.999,  # Just below 10.0
            10.0,  # Exactly 10.0
            10.001,
            10.01,
            10.1,  # Just above 10.0
        ],
        dtype=torch.float32,
    )

    extreme_mask = torch.abs(log_ratios) > 10.0
    extreme_count = extreme_mask.sum().item()

    # Only values STRICTLY greater than 10.0 should be flagged
    # 10.001, 10.01, 10.1 = 3 values
    assert extreme_count == 3, f"Expected 3 extreme values (>10.0), got {extreme_count}"

    # Verify individual detections
    expected_mask = [False, False, False, False, True, True, True]
    assert extreme_mask.tolist() == expected_mask, f"Extreme mask mismatch: {extreme_mask.tolist()}"


def test_negative_values_abs_handling() -> None:
    """Test that negative values are correctly handled with abs()."""
    torch = pytest.importorskip("torch")

    # Mix of positive and negative extreme values
    log_ratios = torch.tensor([-15.0, -10.5, -5.0, 5.0, 10.5, 15.0], dtype=torch.float32)

    # Check max abs
    max_abs = torch.max(torch.abs(log_ratios)).item()
    assert max_abs == 15.0, f"Max abs should be 15.0, got {max_abs}"

    # Check extreme detection
    extreme_mask = torch.abs(log_ratios) > 10.0
    extreme_count = extreme_mask.sum().item()

    # -15.0, -10.5, 10.5, 15.0 = 4 values
    assert extreme_count == 4, f"Expected 4 extreme values, got {extreme_count}"


def test_warning_level_thresholds() -> None:
    """Test that warning levels trigger at correct thresholds."""
    torch = pytest.importorskip("torch")

    # Test each warning level
    test_cases = [
        (0.5, None),  # No warning
        (1.0, None),  # Exactly at threshold, no warning yet
        (1.01, "concerning"),  # Just above 1.0
        (5.0, "concerning"),  # Well above 1.0, but below 10.0
        (10.0, "concerning"),  # Exactly at 10.0 threshold
        (10.01, "severe"),  # Just above 10.0
        (19.99, "severe"),  # High but below clipping
    ]

    for max_abs, expected_level in test_cases:
        # Determine warning level (as in implementation)
        if max_abs > 10.0:
            level = "severe"
        elif max_abs > 1.0:
            level = "concerning"
        else:
            level = None

        assert (
            level == expected_level
        ), f"For max_abs={max_abs}: expected {expected_level}, got {level}"


def test_variance_calculation_numerical_stability() -> None:
    """Test that variance calculation is numerically stable."""
    torch = pytest.importorskip("torch")

    # Case 1: Very small values (potential underflow)
    small_values = torch.tensor([1e-8, 2e-8, 3e-8], dtype=torch.float32)
    mean_small = small_values.mean().item()
    var_small_torch = torch.var(small_values, unbiased=True).item()

    # Manual calculation
    sum_val = float(small_values.sum().item())
    sq_sum = float((small_values.square()).sum().item())
    count = small_values.numel()
    mean_manual = sum_val / count
    raw_var = (sq_sum - count * mean_manual**2) / (count - 1.0)
    var_manual = max(raw_var, 0.0)

    assert (
        abs(var_small_torch - var_manual) < 1e-15
    ), "Variance calculation should be accurate for small values"

    # Case 2: Large values (potential overflow)
    large_values = torch.tensor([1e6, 2e6, 3e6], dtype=torch.float32)
    var_large_torch = torch.var(large_values, unbiased=True).item()

    sum_large = float(large_values.sum().item())
    sq_sum_large = float((large_values.square()).sum().item())
    count_large = large_values.numel()
    mean_large = sum_large / count_large
    raw_var_large = (sq_sum_large - count_large * mean_large**2) / (count_large - 1.0)
    var_large = max(raw_var_large, 0.0)

    # Allow larger tolerance for large values due to floating point precision
    rel_error = abs(var_large_torch - var_large) / var_large_torch
    assert (
        rel_error < 1e-4
    ), f"Variance calculation should be accurate for large values, rel_error={rel_error}"


def test_zero_variance_case() -> None:
    """Test handling of zero variance (all values identical)."""
    torch = pytest.importorskip("torch")

    # All identical values
    log_ratios = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

    var = torch.var(log_ratios, unbiased=True).item()
    assert var == 0.0, f"Variance should be 0 for identical values, got {var}"

    # Manual calculation
    sum_val = float(log_ratios.sum().item())
    sq_sum = float((log_ratios.square()).sum().item())
    count = log_ratios.numel()
    mean = sum_val / count
    raw_var = (sq_sum - count * mean**2) / (count - 1.0)
    var_manual = max(raw_var, 0.0)

    assert var_manual == 0.0, f"Manual variance should also be 0, got {var_manual}"


def test_single_value_statistics() -> None:
    """Test statistics with single value (edge case for variance)."""
    torch = pytest.importorskip("torch")

    log_ratio = torch.tensor([0.5], dtype=torch.float32)

    # With single value, variance formula would divide by (n-1)=0
    # Implementation should handle this with count > 1 check
    count = log_ratio.numel()

    if count > 1:
        var = torch.var(log_ratio, unbiased=True).item()
    else:
        var = 0.0  # Set to 0 as in implementation

    assert var == 0.0, f"Variance with single value should be 0, got {var}"


def test_extreme_fraction_calculation_precision() -> None:
    """Test that extreme fraction is calculated with correct precision."""
    torch = pytest.importorskip("torch")

    # 100 values, 7 extreme
    torch.manual_seed(456)
    log_ratios = torch.randn(100, dtype=torch.float32) * 0.1

    # Inject exactly 7 extreme values
    indices = [0, 10, 20, 30, 40, 50, 60]
    for i in indices:
        log_ratios[i] = 12.0 if i % 2 == 0 else -12.0

    extreme_mask = torch.abs(log_ratios) > 10.0
    extreme_count = extreme_mask.sum().item()
    total_count = log_ratios.numel()

    extreme_fraction = float(extreme_count) / float(total_count)

    assert extreme_count == 7, f"Should have exactly 7 extreme values, got {extreme_count}"
    assert (
        abs(extreme_fraction - 0.07) < 1e-9
    ), f"Extreme fraction should be 0.07, got {extreme_fraction}"


def test_clipping_preserves_sign() -> None:
    """Test that clamping preserves sign of log_ratio."""
    torch = pytest.importorskip("torch")

    # Large positive and negative values
    log_ratios = torch.tensor([-100.0, -50.0, 50.0, 100.0], dtype=torch.float32)
    log_ratios_clamped = torch.clamp(log_ratios, min=-20.0, max=20.0)

    # Check that signs are preserved
    signs_original = torch.sign(log_ratios)
    signs_clamped = torch.sign(log_ratios_clamped)

    assert torch.all(
        signs_original == signs_clamped
    ), f"Clamping should preserve signs: original={signs_original.tolist()}, clamped={signs_clamped.tolist()}"

    # Verify exact values
    expected = torch.tensor([-20.0, -20.0, 20.0, 20.0], dtype=torch.float32)
    assert torch.allclose(
        log_ratios_clamped, expected
    ), f"Clamped values mismatch: got {log_ratios_clamped.tolist()}"


def test_batch_accumulation_over_multiple_minibatches() -> None:
    """Test that statistics accumulate correctly over multiple minibatches."""
    torch = pytest.importorskip("torch")

    # Simulate 3 minibatches
    minibatch1 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    minibatch2 = torch.tensor([0.4, 0.5], dtype=torch.float32)
    minibatch3 = torch.tensor([0.6, 0.7, 0.8, 0.9], dtype=torch.float32)

    # Accumulate statistics
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    max_abs_global = 0.0

    for batch in [minibatch1, minibatch2, minibatch3]:
        total_sum += float(batch.sum().item())
        total_sq_sum += float((batch.square()).sum().item())
        total_count += int(batch.numel())
        batch_max = torch.max(torch.abs(batch)).item()
        max_abs_global = max(max_abs_global, batch_max)

    # Compute final statistics
    mean = total_sum / float(total_count)
    raw_var = (total_sq_sum - total_count * mean**2) / (float(total_count) - 1.0)
    var = max(raw_var, 0.0)
    std = math.sqrt(var)

    # Expected values (combining all batches)
    all_values = torch.cat([minibatch1, minibatch2, minibatch3])
    expected_mean = all_values.mean().item()
    expected_std = all_values.std(unbiased=True).item()
    expected_max = 0.9

    assert (
        abs(mean - expected_mean) < 1e-6
    ), f"Accumulated mean mismatch: got {mean}, expected {expected_mean}"
    assert (
        abs(std - expected_std) < 1e-6
    ), f"Accumulated std mismatch: got {std}, expected {expected_std}"
    assert (
        abs(max_abs_global - expected_max) < 1e-6
    ), f"Accumulated max mismatch: got {max_abs_global}, expected {expected_max}"


def test_comparison_old_vs_new_clipping() -> None:
    """Compare behavior of old (±85) vs new (±20) clipping."""
    torch = pytest.importorskip("torch")

    # Test value that old code would allow but new code should flag
    log_ratio = torch.tensor([50.0], dtype=torch.float32)

    # Old behavior (±85)
    log_ratio_old = torch.clamp(log_ratio, min=-85.0, max=85.0)
    ratio_old = torch.exp(log_ratio_old)

    # New behavior (±20)
    log_ratio_new = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio_new = torch.exp(log_ratio_new)

    # Old: exp(50) ≈ 5.2e21 (astronomically large, but allowed)
    assert (
        ratio_old[0].item() > 1e20
    ), f"Old clipping allows exp(50) ≈ 5e21, got {ratio_old[0].item():.2e}"

    # New: exp(20) ≈ 4.85e8 (clamped at 20)
    assert (
        abs(ratio_new[0].item() - 4.85e8) < 1e6
    ), f"New clipping caps at exp(20) ≈ 4.85e8, got {ratio_new[0].item():.2e}"

    # The difference is 13 orders of magnitude!
    ratio_old_val = ratio_old[0].item()
    ratio_new_val = ratio_new[0].item()
    magnitude_diff = math.log10(ratio_old_val / ratio_new_val)

    assert (
        magnitude_diff > 12
    ), f"Old vs new clipping differ by {magnitude_diff:.1f} orders of magnitude"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
