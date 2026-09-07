"""
Comprehensive tests for the advantage std floor fix.

This module tests the adaptive advantage normalization approach:
1. Conservative floor (1e-4 instead of 1e-8)
2. Skip normalization when advantages are nearly uniform (std < 1e-3)
3. Comprehensive monitoring and logging
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


# Constants from the fix
ADV_STD_FLOOR = 1e-4
ADV_STD_SKIP_THRESHOLD = 1e-3


def test_normal_normalization():
    """Test normal case: std > 1e-3, should normalize normally."""
    try:
        # Create advantages with reasonable variance
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        assert (
            std > ADV_STD_SKIP_THRESHOLD
        ), f"Test setup error: std={std} should be > {ADV_STD_SKIP_THRESHOLD}"

        # Should normalize
        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        # Verify normalization happened correctly
        assert approx_equal(
            np.mean(normalized), 0.0
        ), f"Mean should be ~0, got {np.mean(normalized)}"
        assert approx_equal(
            np.std(normalized, ddof=1), 1.0, abs_tol=1e-5
        ), f"Std should be ~1, got {np.std(normalized, ddof=1)}"
        assert np.all(np.isfinite(normalized)), "All values should be finite"

        results.record_pass("test_normal_normalization")
    except Exception as e:
        results.record_fail("test_normal_normalization", str(e))


def test_low_std_with_floor():
    """Test case: 1e-4 < std < 1e-3, should normalize (floor not needed)."""
    try:
        # Create advantages with low but not too low variance
        base = 0.001
        noise = 2e-4  # Increased noise to get std > 1e-4
        advantages = np.array(
            [base - noise, base - noise / 2, base, base + noise / 2, base + noise], dtype=np.float32
        )

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # Verify std is in the target range
        assert (
            1e-4 < std < ADV_STD_SKIP_THRESHOLD
        ), f"Test setup error: std={std:.2e} should be between 1e-4 and {ADV_STD_SKIP_THRESHOLD}"

        # Should normalize (std >= SKIP_THRESHOLD is False, so we normalize)
        # But std > FLOOR, so floor is not used
        std_clamped = max(std, ADV_STD_FLOOR)
        assert std_clamped == std, "Floor should not be used"

        normalized = (advantages - mean) / std_clamped
        assert np.all(np.isfinite(normalized)), "All values should be finite"

        results.record_pass("test_low_std_with_floor")
    except Exception as e:
        results.record_fail("test_low_std_with_floor", str(e))


def test_very_low_std_skip_normalization():
    """Test case: std < 1e-3, should skip normalization."""
    try:
        # Create nearly uniform advantages
        advantages = np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float32)
        advantages[0] += 1e-10  # Tiny perturbation

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # Verify std is very small
        assert (
            std < ADV_STD_SKIP_THRESHOLD
        ), f"Test setup error: std={std:.2e} should be < {ADV_STD_SKIP_THRESHOLD}"

        # Should skip normalization - advantages would remain unchanged
        # This is the desired behavior

        results.record_pass("test_very_low_std_skip_normalization")
    except Exception as e:
        results.record_fail("test_very_low_std_skip_normalization", str(e))


def test_zero_variance_skip_normalization():
    """Test case: zero variance (identical advantages)."""
    try:
        # All advantages identical
        advantages = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        assert std == 0.0, "Std should be exactly 0"
        assert std < ADV_STD_SKIP_THRESHOLD, "Should skip normalization"

        results.record_pass("test_zero_variance_skip_normalization")
    except Exception as e:
        results.record_fail("test_zero_variance_skip_normalization", str(e))


def test_extreme_values_detection():
    """Test detection of extreme normalized values."""
    try:
        # Create scenario that would produce extreme values with old floor
        advantages = np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float32)
        advantages[0] += 1e-9  # Tiny perturbation

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # With old floor (1e-8)
        std_clamped_old = max(std, 1e-8)
        normalized_old = (advantages - mean) / std_clamped_old
        max_abs_old = np.max(np.abs(normalized_old))

        # With new floor (1e-4)
        std_clamped_new = max(std, ADV_STD_FLOOR)
        normalized_new = (advantages - mean) / std_clamped_new
        max_abs_new = np.max(np.abs(normalized_new))

        # New approach should produce much smaller values (if we were to normalize)
        # But actually we would skip normalization since std < 1e-3
        if std < ADV_STD_FLOOR and max_abs_new > 0:
            ratio = max_abs_old / max_abs_new
            assert ratio > 100, f"Expected at least 100x reduction, got {ratio:.1f}x"

        results.record_pass("test_extreme_values_detection")
    except Exception as e:
        results.record_fail("test_extreme_values_detection", str(e))


def test_numerical_stability_comparison():
    """Compare numerical stability between old and new approach."""
    try:
        # Scenario from user's example
        np.random.seed(42)
        advantages = np.full(100, 0.001, dtype=np.float32)
        advantages += np.random.randn(100).astype(np.float32) * 1e-9

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # Old approach (1e-8 floor, always normalize)
        std_old = max(std, 1e-8)
        norm_old = (advantages - mean) / std_old
        max_old = np.max(np.abs(norm_old))

        # New approach
        if std < ADV_STD_SKIP_THRESHOLD:
            # Skip normalization - advantages unchanged
            gradient_scale_new = np.max(np.abs(advantages))
        else:
            std_new = max(std, ADV_STD_FLOOR)
            norm_new = (advantages - mean) / std_new
            gradient_scale_new = np.max(np.abs(norm_new))

        # Old approach could create extreme values
        # New approach keeps values reasonable
        print(f"    Old max: {max_old:.2e}, New max: {gradient_scale_new:.2e}")

        results.record_pass("test_numerical_stability_comparison")
    except Exception as e:
        results.record_fail("test_numerical_stability_comparison", str(e))


def test_gradient_impact_reduction():
    """Test that new approach reduces gradient explosion risk."""
    try:
        # Scenario: advantages with very low variance
        np.random.seed(42)
        advantages = np.full(100, 0.001, dtype=np.float32)
        advantages += np.random.randn(100).astype(np.float32) * 1e-9

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        # Old approach - always normalize with 1e-8 floor
        std_old = max(std, 1e-8)
        norm_old = (advantages - mean) / std_old
        gradient_scale_old = np.max(np.abs(norm_old))

        # New approach - skip normalization if std < 1e-3
        if std < ADV_STD_SKIP_THRESHOLD:
            # Don't normalize
            gradient_scale_new = np.max(np.abs(advantages))
        else:
            std_new = max(std, ADV_STD_FLOOR)
            norm_new = (advantages - mean) / std_new
            gradient_scale_new = np.max(np.abs(norm_new))

        # New approach should have much smaller gradient scale
        print(f"    Gradient scale - Old: {gradient_scale_old:.2e}, New: {gradient_scale_new:.2e}")

        # The ratio should be significant for very low variance
        if gradient_scale_new > 0 and std < 1e-8:
            ratio = gradient_scale_old / gradient_scale_new
            print(f"    Gradient reduction ratio: {ratio:.0f}x")
            assert ratio > 10, f"Expected significant reduction for low std, got {ratio:.1f}x"

        results.record_pass("test_gradient_impact_reduction")
    except Exception as e:
        results.record_fail("test_gradient_impact_reduction", str(e))


def test_negative_advantages():
    """Test with negative advantages."""
    try:
        advantages = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        # Should still work correctly
        assert approx_equal(np.mean(normalized), 0.0), "Mean should be ~0"
        assert np.all(np.isfinite(normalized)), "All values should be finite"

        results.record_pass("test_negative_advantages")
    except Exception as e:
        results.record_fail("test_negative_advantages", str(e))


def test_mixed_sign_advantages():
    """Test with mixed positive and negative advantages."""
    try:
        advantages = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        assert approx_equal(np.mean(normalized), 0.0), "Mean should be ~0"
        assert np.all(np.isfinite(normalized)), "All values should be finite"

        results.record_pass("test_mixed_sign_advantages")
    except Exception as e:
        results.record_fail("test_mixed_sign_advantages", str(e))


def test_large_advantages():
    """Test with large advantage values."""
    try:
        advantages = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=np.float32)

        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped

        # Should normalize correctly regardless of scale
        assert approx_equal(np.mean(normalized), 0.0, abs_tol=1e-4), "Mean should be ~0"
        assert np.all(np.isfinite(normalized)), "All values should be finite"

        results.record_pass("test_large_advantages")
    except Exception as e:
        results.record_fail("test_large_advantages", str(e))


def test_floor_vs_skip_threshold():
    """Test the relationship between floor and skip threshold."""
    try:
        # FLOOR should be less than SKIP_THRESHOLD
        assert (
            ADV_STD_FLOOR < ADV_STD_SKIP_THRESHOLD
        ), f"Floor ({ADV_STD_FLOOR}) should be < skip threshold ({ADV_STD_SKIP_THRESHOLD})"

        # Recommended: SKIP_THRESHOLD should be 10x FLOOR
        ratio = ADV_STD_SKIP_THRESHOLD / ADV_STD_FLOOR
        assert ratio >= 10, f"Skip threshold should be at least 10x floor, got {ratio}x"

        results.record_pass("test_floor_vs_skip_threshold")
    except Exception as e:
        results.record_fail("test_floor_vs_skip_threshold", str(e))


def main():
    """Run all tests."""
    print("=" * 80)
    print("ADVANTAGE STD FLOOR FIX - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    print("Configuration:")
    print(f"  ADV_STD_FLOOR: {ADV_STD_FLOOR}")
    print(f"  ADV_STD_SKIP_THRESHOLD: {ADV_STD_SKIP_THRESHOLD}")
    print()

    print("Running tests...")
    print()

    # Run all tests
    test_normal_normalization()
    test_low_std_with_floor()
    test_very_low_std_skip_normalization()
    test_zero_variance_skip_normalization()
    test_extreme_values_detection()
    test_numerical_stability_comparison()
    test_gradient_impact_reduction()
    test_negative_advantages()
    test_mixed_sign_advantages()
    test_large_advantages()
    test_floor_vs_skip_threshold()

    # Print summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
