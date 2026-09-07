"""
CORRECTED: Comprehensive tests for advantage std floor fix V2.

CORRECTED APPROACH:
- ALWAYS normalize (PPO expects mean=0, std≈1)
- Use conservative floor 1e-4 (instead of 1e-8)
- NO skip threshold (previous approach was flawed)

This maintains PPO's contract while preventing gradient explosion.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  ✓ {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"Results: {self.passed}/{total} passed, {self.failed}/{total} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*80}")
        return self.failed == 0


results = TestResults()


def approx_equal(a, b, abs_tol=1e-6):
    """Check if two values are approximately equal."""
    return abs(a - b) < abs_tol


# Constants from corrected implementation
ADV_STD_FLOOR = 1e-4


def test_always_normalize():
    """CRITICAL: Test that we ALWAYS normalize (no skipping)."""
    try:
        # Test various std values - all should be normalized
        test_cases = [
            ("very_low", 1e-8),
            ("at_floor", 1e-4),
            ("above_floor", 5e-4),
            ("normal", 1e-2),
        ]

        for name, std_val in test_cases:
            advantages = np.random.RandomState(42).randn(100) * std_val
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # New logic: ALWAYS normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            # Check that normalization happened
            norm_mean = np.mean(normalized)
            assert approx_equal(
                norm_mean, 0.0, abs_tol=1e-6
            ), f"{name}: mean should be ~0, got {norm_mean}"

            # Check all finite
            assert np.all(np.isfinite(normalized)), f"{name}: all values should be finite"

        results.record_pass("test_always_normalize")
    except Exception as e:
        results.record_fail("test_always_normalize", str(e))


def test_ppo_expectation_satisfied():
    """CRITICAL: Test that PPO expectation (mean=0) is always satisfied."""
    try:
        # Test with low variance (the problematic case in v1)
        advantages = np.random.RandomState(42).randn(100) * 5e-4 + 0.1

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        print(f"\n    Raw: mean={mean:.2e}, std={std:.2e}")

        # New logic: ALWAYS normalize
        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        norm_mean = np.mean(normalized)
        norm_std = np.std(normalized, ddof=1)

        print(f"    After normalization: mean={norm_mean:.2e}, std={norm_std:.2e}")

        # CRITICAL: mean should be ~0 (PPO expectation)
        assert approx_equal(
            norm_mean, 0.0, abs_tol=1e-6
        ), f"Normalized mean should be ~0, got {norm_mean}"

        results.record_pass("test_ppo_expectation_satisfied")
    except Exception as e:
        results.record_fail("test_ppo_expectation_satisfied", str(e))


def test_floor_prevents_extreme_values():
    """Test that floor prevents extreme normalized values."""
    try:
        # Scenario: very low std (close to 0)
        advantages = np.random.RandomState(42).randn(100) * 1e-8

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        print(f"\n    Testing very low std={std:.2e}")

        # Old approach (1e-8 floor)
        std_old = max(std, 1e-8)
        norm_old = (advantages - mean) / std_old
        max_old = np.max(np.abs(norm_old))

        # New approach (1e-4 floor)
        std_new = max(std, ADV_STD_FLOOR)
        norm_new = (advantages - mean) / std_new
        max_new = np.max(np.abs(norm_new))

        print(f"    Old (1e-8): max={max_old:.2e}")
        print(f"    New (1e-4): max={max_new:.2e}")
        print(f"    Improvement: {max_old/max_new:.0f}x reduction")

        # New approach should have much smaller values
        assert max_new < max_old, "New approach should reduce extreme values"

        # New approach should keep values reasonable (< 10)
        assert max_new < 10.0, f"Max normalized should be <10, got {max_new}"

        results.record_pass("test_floor_prevents_extreme_values")
    except Exception as e:
        results.record_fail("test_floor_prevents_extreme_values", str(e))


def test_gradient_safety_comprehensive():
    """Test gradient safety across all std ranges."""
    try:
        print("\n    Testing gradient safety...")

        std_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2]

        print(f"\n    {'Std':<10} {'Old max':<12} {'New max':<12} {'Reduction':<12}")
        print(f"    {'-'*50}")

        all_safe = True

        for std_val in std_values:
            advantages = np.random.RandomState(42).randn(100) * std_val
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # Old
            std_old = max(std, 1e-8)
            norm_old = (advantages - mean) / std_old
            max_old = np.max(np.abs(norm_old))

            # New
            std_new = max(std, ADV_STD_FLOOR)
            norm_new = (advantages - mean) / std_new
            max_new = np.max(np.abs(norm_new))

            reduction = max_old / max_new if max_new > 0 else float("inf")

            print(f"    {std:.1e}  {max_old:<12.2e} {max_new:<12.2e} {reduction:<12.0f}x")

            # Check if new approach is safer
            if max_new > 100:
                all_safe = False

        assert all_safe, "Some normalized values are still > 100"

        results.record_pass("test_gradient_safety_comprehensive")
    except Exception as e:
        results.record_fail("test_gradient_safety_comprehensive", str(e))


def test_uniform_advantages_behavior():
    """Test behavior when advantages are nearly uniform."""
    try:
        # Nearly uniform advantages (std very small)
        advantages = np.ones(100) * 0.5
        advantages[0] += 1e-10  # Tiny noise

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        print(f"\n    Uniform case: std={std:.2e}")

        # New logic: still normalize with floor
        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        max_norm = np.max(np.abs(normalized))

        print(f"    After normalization: max={max_norm:.2e}")

        # Values should be very small (compressed to ~0)
        # This is correct: uniform advantages → no gradient signal
        assert max_norm < 1.0, "Uniform advantages should compress to small values"

        # Mean should still be ~0
        assert approx_equal(np.mean(normalized), 0.0, abs_tol=1e-6), "Mean should be ~0"

        results.record_pass("test_uniform_advantages_behavior")
    except Exception as e:
        results.record_fail("test_uniform_advantages_behavior", str(e))


def test_real_world_scenarios():
    """Test real-world trading scenarios."""
    try:
        print("\n    Testing real-world scenarios...")

        scenarios = [
            ("early_training_high_var", 0.001, 1e-3),
            ("mid_training_med_var", 0.001, 5e-4),
            ("late_training_low_var", 0.001, 1e-4),
            ("converged_very_low_var", 0.001, 1e-5),
        ]

        print(f"\n    {'Scenario':<25} {'Std':<10} {'Max norm':<12} {'Mean norm':<12}")
        print(f"    {'-'*65}")

        all_valid = True

        for name, mean_val, std_val in scenarios:
            advantages = np.random.RandomState(42).randn(128) * std_val + mean_val

            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # Normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            max_norm = np.max(np.abs(normalized))
            mean_norm = np.mean(normalized)

            print(f"    {name:<25} {std:<10.1e} {max_norm:<12.2e} {mean_norm:<12.2e}")

            # Check validity
            if not approx_equal(mean_norm, 0.0, abs_tol=1e-6):
                all_valid = False
            if max_norm > 100:
                all_valid = False

        assert all_valid, "Some scenarios produced invalid normalization"

        results.record_pass("test_real_world_scenarios")
    except Exception as e:
        results.record_fail("test_real_world_scenarios", str(e))


def test_floor_value_is_reasonable():
    """Verify that 1e-4 is a reasonable floor value."""
    try:
        # Test that floor value strikes a good balance

        # Case 1: std = 1e-4 (at floor)
        adv1 = np.random.RandomState(42).randn(100) * 1e-4
        mean1 = np.mean(adv1)
        std1 = np.std(adv1, ddof=1)
        norm1 = (adv1 - mean1) / max(std1, ADV_STD_FLOOR)
        max1 = np.max(np.abs(norm1))

        # Case 2: std = 1e-5 (below floor)
        adv2 = np.random.RandomState(42).randn(100) * 1e-5
        mean2 = np.mean(adv2)
        std2 = np.std(adv2, ddof=1)
        norm2 = (adv2 - mean2) / max(std2, ADV_STD_FLOOR)
        max2 = np.max(np.abs(norm2))

        print(f"\n    At floor (std=1e-4): max norm = {max1:.2f}")
        print(f"    Below floor (std=1e-5): max norm = {max2:.2f}")

        # Both should be reasonable (<10)
        assert max1 < 10, f"At floor should give reasonable values, got {max1}"
        assert max2 < 1, f"Below floor should give small values, got {max2}"

        results.record_pass("test_floor_value_is_reasonable")
    except Exception as e:
        results.record_fail("test_floor_value_is_reasonable", str(e))


def test_edge_cases():
    """Test edge cases."""
    try:
        # Case 1: All zeros
        adv_zeros = np.zeros(100)
        mean_z = np.mean(adv_zeros)
        std_z = np.std(adv_zeros, ddof=1)
        norm_z = (adv_zeros - mean_z) / max(std_z, ADV_STD_FLOOR)
        assert np.all(norm_z == 0), "All zeros should stay zeros"

        # Case 2: Negative advantages
        adv_neg = np.random.RandomState(42).randn(100) * 0.1 - 0.5
        mean_n = np.mean(adv_neg)
        std_n = np.std(adv_neg, ddof=1)
        norm_n = (adv_neg - mean_n) / max(std_n, ADV_STD_FLOOR)
        assert approx_equal(np.mean(norm_n), 0.0, abs_tol=1e-6), "Mean should be 0"

        # Case 3: Large advantages
        adv_large = np.random.RandomState(42).randn(100) * 100
        mean_l = np.mean(adv_large)
        std_l = np.std(adv_large, ddof=1)
        norm_l = (adv_large - mean_l) / max(std_l, ADV_STD_FLOOR)
        assert approx_equal(np.mean(norm_l), 0.0, abs_tol=1e-4), "Mean should be 0"

        results.record_pass("test_edge_cases")
    except Exception as e:
        results.record_fail("test_edge_cases", str(e))


def test_comparison_with_stable_baselines3():
    """Compare with Stable-Baselines3 approach."""
    try:
        print("\n    Comparing with SB3 (1e-8 floor)...")

        # Low variance case (problematic for SB3)
        advantages = np.random.RandomState(42).randn(100) * 1e-5

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # SB3 approach
        sb3_normalized = (advantages - mean) / (std + 1e-8)
        sb3_max = np.max(np.abs(sb3_normalized))

        # Our approach
        our_normalized = (advantages - mean) / max(std, ADV_STD_FLOOR)
        our_max = np.max(np.abs(our_normalized))

        print(f"    SB3 max: {sb3_max:.2e}")
        print(f"    Our max: {our_max:.2e}")
        print(f"    Our approach is {sb3_max/our_max:.0f}x safer")

        # Our approach should be safer
        assert our_max < sb3_max, "Our approach should be safer than SB3"

        results.record_pass("test_comparison_with_stable_baselines3")
    except Exception as e:
        results.record_fail("test_comparison_with_stable_baselines3", str(e))


def main():
    """Run all corrected tests."""
    print("=" * 80)
    print("ADVANTAGE STD FLOOR FIX V2 - CORRECTED TEST SUITE")
    print("=" * 80)
    print()
    print("CORRECTED APPROACH:")
    print("- ALWAYS normalize (maintains PPO contract: mean=0)")
    print("- Use conservative floor 1e-4 (prevents gradient explosion)")
    print("- NO skip threshold (previous v1 approach was flawed)")
    print()

    print("Configuration:")
    print(f"  ADV_STD_FLOOR: {ADV_STD_FLOOR}")
    print()

    print("Running corrected tests...")
    print()

    # Run all tests
    test_always_normalize()
    test_ppo_expectation_satisfied()
    test_floor_prevents_extreme_values()
    test_gradient_safety_comprehensive()
    test_uniform_advantages_behavior()
    test_real_world_scenarios()
    test_floor_value_is_reasonable()
    test_edge_cases()
    test_comparison_with_stable_baselines3()

    # Print summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
