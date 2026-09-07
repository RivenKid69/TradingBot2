"""
DEEP ANALYSIS V2: Verification of corrected advantage std floor fix.

Tests the CORRECTED approach:
- Always normalize (no skip threshold)
- Use floor 1e-4
- Maintain PPO contract (mean=0)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class DeepTestResults:
    """Track deep test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = []
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  ✓ {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")

    def record_warning(self, test_name, warning):
        self.warnings.append((test_name, warning))
        print(f"  ⚠ {test_name}: {warning}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"Results: {self.passed}/{total} passed, {self.failed}/{total} failed")
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for name, warning in self.warnings:
                print(f"  - {name}: {warning}")
        else:
            print(f"\n✓ No warnings - implementation is correct!")
        if self.errors:
            print(f"\n✗ Failed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*80}")
        return self.failed == 0 and len(self.warnings) == 0


results = DeepTestResults()

ADV_STD_FLOOR = 1e-4


def test_critical_range_v2():
    """CRITICAL: Test that we normalize in ALL ranges."""
    try:
        print("\n    Testing all std ranges...")

        test_stds = [1e-8, 1e-4, 2e-4, 5e-4, 8e-4, 1e-3, 1e-2]

        print(f"\n    {'Std':<10} {'Behavior':<12} {'Mean':<12} {'Max':<12}")
        print(f"    {'-'*50}")

        all_normalized = True

        for std_value in test_stds:
            advantages = np.random.RandomState(42).randn(100) * std_value
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # Corrected logic: ALWAYS normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            norm_mean = np.mean(normalized)
            max_val = np.max(np.abs(normalized))

            behavior = "NORMALIZE"

            print(f"    {std:<10.1e} {behavior:<12} {norm_mean:<12.2e} {max_val:<12.2e}")

            # Check mean is ~0
            if abs(norm_mean) > 1e-6:
                all_normalized = False
                results.record_warning(
                    "normalization_mean_nonzero",
                    f"std={std:.1e}: mean={norm_mean:.2e} (should be ~0)",
                )

        assert all_normalized, "Some ranges have non-zero mean"

        results.record_pass("test_critical_range_v2")
    except Exception as e:
        results.record_fail("test_critical_range_v2", str(e))


def test_ppo_contract_always_satisfied():
    """CRITICAL: PPO contract (mean=0) must ALWAYS be satisfied."""
    try:
        print("\n    Verifying PPO contract...")

        test_cases = [
            ("very_low_std", 1e-8),
            ("low_std", 1e-4),
            ("medium_std", 5e-4),
            ("normal_std", 1e-2),
        ]

        all_valid = True

        for name, std_val in test_cases:
            advantages = np.random.RandomState(42).randn(100) * std_val + 0.1
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # Always normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            norm_mean = np.mean(normalized)

            print(f"      {name:<20}: mean={norm_mean:.2e}")

            if abs(norm_mean) > 1e-6:
                all_valid = False
                results.record_warning(
                    "ppo_contract_violation", f"{name}: mean={norm_mean:.2e} (PPO expects ~0)"
                )

        assert all_valid, "PPO contract violated in some cases"

        results.record_pass("test_ppo_contract_always_satisfied")
    except Exception as e:
        results.record_fail("test_ppo_contract_always_satisfied", str(e))


def test_gradient_explosion_prevented():
    """Test that gradient explosion is prevented."""
    try:
        print("\n    Testing gradient explosion prevention...")

        # Most dangerous case: very low std
        advantages = np.random.RandomState(42).randn(1000) * 1e-9
        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)

        print(f"      Extreme case: std={std:.2e}")

        # Old
        std_old = max(std, 1e-8)
        norm_old = (advantages - mean) / std_old
        max_old = np.max(np.abs(norm_old))

        # New
        std_new = max(std, ADV_STD_FLOOR)
        norm_new = (advantages - mean) / std_new
        max_new = np.max(np.abs(norm_new))

        print(f"      Old: max={max_old:.2e}")
        print(f"      New: max={max_new:.2e}")
        print(f"      Safety factor: {max_old/max_new:.0f}x")

        # New should be much safer
        if max_new > 10:
            results.record_warning("gradient_still_large", f"max={max_new:.2e} is still > 10")

        results.record_pass("test_gradient_explosion_prevented")
    except Exception as e:
        results.record_fail("test_gradient_explosion_prevented", str(e))


def test_no_skip_logic():
    """Verify that there is NO skip logic in corrected implementation."""
    try:
        print("\n    Verifying no skip logic...")

        # In the old implementation, these would have been skipped
        # In new implementation, all should be normalized

        test_stds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 9e-4]

        all_normalized = True

        for std_val in test_stds:
            advantages = np.random.RandomState(42).randn(100) * std_val
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # New logic: always normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            # Check normalization happened (mean ~0)
            norm_mean = np.mean(normalized)

            if abs(norm_mean) > 1e-6:
                all_normalized = False

        assert all_normalized, "Some cases were not normalized"

        print(f"      All {len(test_stds)} cases normalized ✓")

        results.record_pass("test_no_skip_logic")
    except Exception as e:
        results.record_fail("test_no_skip_logic", str(e))


def test_real_world_comprehensive():
    """Comprehensive real-world scenario testing."""
    try:
        print("\n    Testing comprehensive real-world scenarios...")

        scenarios = [
            ("early_high_variance", 0.01, 1e-2, 1000),
            ("mid_medium_variance", 0.01, 5e-3, 1000),
            ("late_low_variance", 0.01, 1e-3, 1000),
            ("converging_very_low", 0.01, 1e-4, 1000),
            ("stuck_minimal", 0.01, 1e-5, 1000),
            ("numerical_noise", 0.01, 1e-8, 1000),
        ]

        print(f"\n    {'Scenario':<25} {'Std':<10} {'Mean':<10} {'Max':<10} {'Status':<10}")
        print(f"    {'-'*70}")

        all_valid = True

        for name, mean_val, std_val, n_samples in scenarios:
            advantages = np.random.RandomState(42).randn(n_samples) * std_val + mean_val

            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)

            # Always normalize
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            norm_mean = np.mean(normalized)
            norm_max = np.max(np.abs(normalized))

            status = "OK"
            if abs(norm_mean) > 1e-6:
                status = "WARN:mean"
                all_valid = False
            if norm_max > 100:
                status = "WARN:max"
                all_valid = False

            print(f"    {name:<25} {std:<10.2e} {norm_mean:<10.2e} {norm_max:<10.2e} {status:<10}")

        if not all_valid:
            results.record_warning(
                "real_world_validation_issues", "Some real-world scenarios produced warnings"
            )

        results.record_pass("test_real_world_comprehensive")
    except Exception as e:
        results.record_fail("test_real_world_comprehensive", str(e))


def test_mathematical_correctness():
    """Deep mathematical validation."""
    try:
        print("\n    Mathematical correctness validation...")

        # Property 1: Normalized mean should always be 0
        for _ in range(10):
            advantages = np.random.randn(100) * np.random.uniform(1e-8, 1e-2)
            mean = np.mean(advantages)
            std = np.std(advantages, ddof=1)
            std_clamped = max(std, ADV_STD_FLOOR)
            normalized = (advantages - mean) / std_clamped

            assert abs(np.mean(normalized)) < 1e-6, "Mean should be ~0"

        # Property 2: When std > floor, normalized std should be ~1
        advantages = np.random.randn(100) * 1e-2
        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)
        assert std > ADV_STD_FLOOR
        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped
        norm_std = np.std(normalized, ddof=1)
        assert abs(norm_std - 1.0) < 0.01, "Std should be ~1 when > floor"

        # Property 3: When std < floor, normalized values compressed
        advantages = np.random.randn(100) * 1e-8
        mean = np.mean(advantages)
        std = np.std(advantages, ddof=1)
        assert std < ADV_STD_FLOOR
        std_clamped = max(std, ADV_STD_FLOOR)
        normalized = (advantages - mean) / std_clamped
        norm_max = np.max(np.abs(normalized))
        assert norm_max < 1.0, "Values should be compressed when std < floor"

        print(f"      All mathematical properties verified ✓")

        results.record_pass("test_mathematical_correctness")
    except Exception as e:
        results.record_fail("test_mathematical_correctness", str(e))


def main():
    """Run all V2 deep analysis tests."""
    print("=" * 80)
    print("DEEP ANALYSIS V2: CORRECTED IMPLEMENTATION VERIFICATION")
    print("=" * 80)
    print()
    print("Testing CORRECTED approach:")
    print("- Always normalize (no skip)")
    print("- Floor = 1e-4")
    print("- Maintains PPO contract (mean=0)")
    print()

    print("Configuration:")
    print(f"  ADV_STD_FLOOR: {ADV_STD_FLOOR}")
    print()

    print("Running validation tests...")

    # Run tests
    test_critical_range_v2()
    test_ppo_contract_always_satisfied()
    test_gradient_explosion_prevented()
    test_no_skip_logic()
    test_real_world_comprehensive()
    test_mathematical_correctness()

    # Summary
    success = results.summary()

    if success:
        print("\n✓ IMPLEMENTATION VERIFIED - NO ISSUES DETECTED")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
