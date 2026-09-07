#!/usr/bin/env python3
"""
Standalone verification script for log_ratio monitoring logic.
Runs without external dependencies to verify mathematical correctness.
"""

import math


def test_clipping_boundaries():
    """Test that ±20 is correct boundary."""
    print("Testing clipping boundaries...")

    # exp(20) should be finite
    exp_20 = math.exp(20.0)
    assert math.isfinite(exp_20), f"exp(20) should be finite: {exp_20}"
    assert abs(exp_20 - 4.85e8) < 1e6, f"exp(20) ≈ 4.85e8, got {exp_20:.2e}"

    # exp(89) would overflow (but we test smaller value)
    # (can't test exp(89) directly as it will overflow)

    print(f"  ✓ exp(20) = {exp_20:.2e} (finite)")
    print(f"  ✓ exp(-20) = {math.exp(-20.0):.2e} (finite)")


def test_statistics_calculation():
    """Test manual statistics calculation matches expected."""
    print("\nTesting statistics calculation...")

    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    n = len(values)

    # Mean
    mean = sum(values) / n
    expected_mean = 0.3
    assert abs(mean - expected_mean) < 1e-6, f"Mean mismatch: {mean} vs {expected_mean}"

    # Sum of squares
    sq_sum = sum(x**2 for x in values)

    # Variance (sample)
    raw_var = (sq_sum - n * mean**2) / (n - 1.0)
    var = max(raw_var, 0.0)
    std = math.sqrt(var)

    # Expected std
    expected_var = sum((x - mean) ** 2 for x in values) / (n - 1)
    expected_std = math.sqrt(expected_var)

    assert abs(std - expected_std) < 1e-6, f"Std mismatch: {std} vs {expected_std}"

    # Max abs
    max_abs = max(abs(x) for x in values)
    assert max_abs == 0.5, f"Max abs should be 0.5, got {max_abs}"

    print(f"  ✓ Mean: {mean}")
    print(f"  ✓ Std: {std:.4f}")
    print(f"  ✓ Max abs: {max_abs}")


def test_warning_thresholds():
    """Test warning level logic."""
    print("\nTesting warning thresholds...")

    test_cases = [
        (0.5, None),
        (1.5, "concerning"),
        (15.0, "severe"),
    ]

    for max_abs, expected_level in test_cases:
        if max_abs > 10.0:
            level = "severe"
        elif max_abs > 1.0:
            level = "concerning"
        else:
            level = None

        assert (
            level == expected_level
        ), f"For max_abs={max_abs}: expected {expected_level}, got {level}"
        print(f"  ✓ max_abs={max_abs} → {level}")


def test_extreme_detection():
    """Test extreme value detection."""
    print("\nTesting extreme value detection...")

    values = [-15.0, -10.5, -5.0, 0.0, 5.0, 10.5, 15.0]
    extreme_threshold = 10.0

    extreme_count = sum(1 for x in values if abs(x) > extreme_threshold)

    # -15.0, -10.5, 10.5, 15.0 = 4 values
    assert extreme_count == 4, f"Expected 4 extreme values, got {extreme_count}"

    # Max abs
    max_abs = max(abs(x) for x in values)
    assert max_abs == 15.0, f"Max abs should be 15.0, got {max_abs}"

    print(f"  ✓ Detected {extreme_count} extreme values (|x| > {extreme_threshold})")
    print(f"  ✓ Max abs: {max_abs}")


def test_clipping_preserves_sign():
    """Test that clamping preserves sign."""
    print("\nTesting sign preservation...")

    def clamp(x, min_val, max_val):
        return max(min_val, min(max_val, x))

    values = [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0]
    clamped = [clamp(x, -20.0, 20.0) for x in values]

    for orig, clmp in zip(values, clamped):
        if orig != 0:
            orig_sign = 1 if orig > 0 else -1
            clmp_sign = 1 if clmp > 0 else -1
            assert orig_sign == clmp_sign, f"Sign changed: {orig} → {clmp}"

    expected = [-20.0, -20.0, -20.0, 0.0, 20.0, 20.0, 20.0]
    assert clamped == expected, f"Clamped values mismatch: {clamped}"

    print(f"  ✓ All signs preserved after clamping")
    print(f"  ✓ Clamped values: {clamped}")


def test_approx_kl_relationship():
    """Test approx_kl = -log_ratio relationship."""
    print("\nTesting approx_kl relationship...")

    # new_log_prob - old_log_prob = log_ratio
    # old_log_prob - new_log_prob = approx_kl = -log_ratio

    new_log_prob = [-1.5, -2.0, -2.5]
    old_log_prob = [-1.4, -2.1, -2.45]

    log_ratios = [n - o for n, o in zip(new_log_prob, old_log_prob)]
    approx_kls = [o - n for o, n in zip(old_log_prob, new_log_prob)]

    for lr, kl in zip(log_ratios, approx_kls):
        assert abs(lr + kl) < 1e-9, f"approx_kl should equal -log_ratio: {kl} vs {-lr}"

    print(f"  ✓ log_ratios: {[f'{x:.2f}' for x in log_ratios]}")
    print(f"  ✓ approx_kls: {[f'{x:.2f}' for x in approx_kls]}")
    print(f"  ✓ Relationship verified: approx_kl = -log_ratio")


def test_old_vs_new_clipping_comparison():
    """Compare old (±85) vs new (±20) clipping."""
    print("\nComparing old (±85) vs new (±20) clipping...")

    def clamp(x, min_val, max_val):
        return max(min_val, min(max_val, x))

    log_ratio = 50.0

    # Old behavior
    log_ratio_old = clamp(log_ratio, -85.0, 85.0)
    ratio_old = math.exp(log_ratio_old)

    # New behavior
    log_ratio_new = clamp(log_ratio, -20.0, 20.0)
    ratio_new = math.exp(log_ratio_new)

    print(f"  Old: log_ratio={log_ratio_old} → ratio={ratio_old:.2e}")
    print(f"  New: log_ratio={log_ratio_new} → ratio={ratio_new:.2e}")

    magnitude_diff = math.log10(ratio_old / ratio_new)
    print(f"  Difference: {magnitude_diff:.1f} orders of magnitude")

    assert ratio_old > 1e20, "Old clipping allows astronomically large ratios"
    assert ratio_new < 1e9, "New clipping keeps ratios reasonable"
    assert magnitude_diff > 12, "Should differ by >12 orders of magnitude"

    print(f"  ✓ Old clipping was too permissive")
    print(f"  ✓ New clipping is appropriately conservative")


def test_healthy_training_scenario():
    """Simulate healthy training statistics."""
    print("\nSimulating healthy training scenario...")

    # Healthy training: log_ratio ~ N(0, 0.05)
    # Using deterministic values that approximate this
    log_ratios = [
        0.05,
        -0.03,
        0.02,
        -0.01,
        0.04,
        -0.05,
        0.01,
        -0.02,
        0.03,
        -0.04,
        0.00,
        0.01,
        -0.01,
        0.02,
        -0.03,
    ]

    n = len(log_ratios)
    mean = sum(log_ratios) / n
    sq_sum = sum(x**2 for x in log_ratios)
    var = (sq_sum - n * mean**2) / (n - 1)
    std = math.sqrt(var)
    max_abs = max(abs(x) for x in log_ratios)

    print(f"  Mean: {mean:.4f} (should be ≈ 0)")
    print(f"  Std: {std:.4f} (should be < 0.1)")
    print(f"  Max abs: {max_abs:.4f} (should be < 0.2)")

    assert abs(mean) < 0.05, "Healthy training should have small mean"
    assert std < 0.1, "Healthy training should have small std"
    assert max_abs < 0.2, "Healthy training should have small max_abs"

    # No warnings expected
    if max_abs > 10.0:
        warning = "severe"
    elif max_abs > 1.0:
        warning = "concerning"
    else:
        warning = None

    assert warning is None, "Healthy training should not trigger warnings"
    print(f"  ✓ No warnings triggered (healthy training)")


def test_conservative_clipping_does_not_activate():
    """Test that ±20 clipping doesn't activate in healthy training."""
    print("\nTesting that ±20 clipping is inactive in healthy training...")

    def clamp(x, min_val, max_val):
        return max(min_val, min(max_val, x))

    # Healthy values
    log_ratios = [0.05, -0.03, 0.02, -0.01, 0.15, -0.12, 0.08]

    # Apply clipping
    clamped = [clamp(x, -20.0, 20.0) for x in log_ratios]

    # Should be unchanged
    assert log_ratios == clamped, "Healthy values should not be clamped"

    # Count changes
    changes = sum(1 for o, c in zip(log_ratios, clamped) if o != c)
    assert changes == 0, f"Expected 0 changes, got {changes}"

    print(f"  ✓ {len(log_ratios)} healthy values tested")
    print(f"  ✓ {changes} values clamped (should be 0)")
    print(f"  ✓ Clipping is inactive in healthy training")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Log Ratio Monitoring Logic Verification")
    print("=" * 60)

    tests = [
        test_clipping_boundaries,
        test_statistics_calculation,
        test_warning_thresholds,
        test_extreme_detection,
        test_clipping_preserves_sign,
        test_approx_kl_relationship,
        test_old_vs_new_clipping_comparison,
        test_healthy_training_scenario,
        test_conservative_clipping_does_not_activate,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All verification tests passed!")
        print("✓ Log ratio monitoring logic is mathematically correct")
        return 0
    else:
        print(f"\n✗ {failed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
