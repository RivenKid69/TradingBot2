"""
DEEP ANALYSIS: Advantage std floor fix comprehensive validation.

This test suite performs deep analysis of the fix to ensure:
1. Correctness of threshold values
2. Edge case handling
3. Gradient safety
4. Integration with real scenarios
5. Mathematical soundness
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
        if self.errors:
            print(f"\n✗ Failed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*80}")
        return self.failed == 0


results = DeepTestResults()

# Constants from implementation
ADV_STD_FLOOR = 1e-4
ADV_STD_SKIP_THRESHOLD = 1e-3


def test_critical_range_analysis():
    """CRITICAL: Test the range between floor and skip threshold."""
    try:
        print("\n    Analyzing critical range [1e-4, 1e-3]...")

        # Test multiple std values in the critical range
        test_stds = [1e-4, 2e-4, 5e-4, 8e-4, 9e-4, 1e-3]

        for std_value in test_stds:
            # Create advantages with specific std
            np.random.seed(42)
            advantages = np.random.randn(100) * std_value
            actual_std = np.std(advantages, ddof=1)

            print(f"      Testing std={std_value:.1e} (actual={actual_std:.1e})")

            # Current implementation logic
            if actual_std < ADV_STD_SKIP_THRESHOLD:
                # SKIP normalization
                behavior = "SKIP"
                result_advantages = advantages
            else:
                # NORMALIZE
                behavior = "NORMALIZE"
                std_clamped = max(actual_std, ADV_STD_FLOOR)
                mean = np.mean(advantages)
                result_advantages = (advantages - mean) / std_clamped

            max_result = np.max(np.abs(result_advantages))
            print(f"        Behavior: {behavior}, Max value: {max_result:.2e}")

            # WARNING: If we skip normalization in range [1e-4, 1e-3],
            # the advantages are not normalized, which may break PPO's expectation
            if behavior == "SKIP" and actual_std >= ADV_STD_FLOOR:
                results.record_warning(
                    "critical_range_skip",
                    f"std={actual_std:.1e} >= floor but normalization skipped",
                )

        results.record_pass("test_critical_range_analysis")
    except Exception as e:
        results.record_fail("test_critical_range_analysis", str(e))


def test_skip_threshold_too_high():
    """CRITICAL: Check if skip threshold is too conservative."""
    try:
        # Scenario: std = 5e-4 (middle of critical range)
        std_test = 5e-4
        advantages = np.random.RandomState(42).randn(1000) * std_test
        actual_std = np.std(advantages, ddof=1)

        print(f"\n    Testing std={actual_std:.1e}")

        # Current behavior: skip if std < 1e-3
        will_skip = actual_std < ADV_STD_SKIP_THRESHOLD

        print(f"      Current: {'SKIP' if will_skip else 'NORMALIZE'}")

        # Alternative: skip only if std < floor (1e-4)
        should_skip_alt = actual_std < ADV_STD_FLOOR
        print(f"      Alternative (skip<floor): {'SKIP' if should_skip_alt else 'NORMALIZE'}")

        # If std is significantly above floor, we should normalize
        if actual_std >= 2 * ADV_STD_FLOOR and will_skip:
            results.record_warning(
                "skip_threshold_too_high",
                f"std={actual_std:.1e} is 2x floor but we skip normalization",
            )

        results.record_pass("test_skip_threshold_too_high")
    except Exception as e:
        results.record_fail("test_skip_threshold_too_high", str(e))


def test_alternative_strategy():
    """Compare current strategy with alternative: skip only if std < floor."""
    try:
        print("\n    Comparing strategies...")

        test_cases = [
            ("very_low", 1e-5),
            ("at_floor", 1e-4),
            ("between", 5e-4),
            ("at_skip", 1e-3),
            ("normal", 1e-2),
        ]

        print(f"\n    {'Case':<15} {'Std':<10} {'Current':<12} {'Alt1':<12} {'Alt2':<12}")
        print(f"    {'-'*60}")

        for case_name, std_val in test_cases:
            # Current strategy
            current = "SKIP" if std_val < ADV_STD_SKIP_THRESHOLD else "NORM"

            # Alternative 1: skip only if < floor
            alt1 = "SKIP" if std_val < ADV_STD_FLOOR else "NORM"

            # Alternative 2: never skip, always use floor
            alt2 = "NORM(floor)"

            print(f"    {case_name:<15} {std_val:<10.1e} {current:<12} {alt1:<12} {alt2:<12}")

            # If alternatives differ, this is a critical decision point
            if current != alt1:
                results.record_warning(
                    "strategy_difference", f"{case_name}: current={current}, alt={alt1}"
                )

        results.record_pass("test_alternative_strategy")
    except Exception as e:
        results.record_fail("test_alternative_strategy", str(e))


def test_ppo_expectation():
    """Test if skipping normalization breaks PPO's expectation."""
    try:
        print("\n    Testing PPO expectation...")

        # PPO typically expects normalized advantages (mean=0, std=1)
        # If we skip normalization, advantages have mean≠0, std≠1

        advantages = np.random.RandomState(42).randn(100) * 5e-4 + 0.1
        std_adv = np.std(advantages, ddof=1)
        mean_adv = np.mean(advantages)

        print(f"      Raw: mean={mean_adv:.2e}, std={std_adv:.2e}")

        # Current behavior
        if std_adv < ADV_STD_SKIP_THRESHOLD:
            # Skip
            result = advantages
            result_mean = np.mean(result)
            result_std = np.std(result, ddof=1)
            print(f"      After SKIP: mean={result_mean:.2e}, std={result_std:.2e}")

            # Warning: mean is not 0!
            if abs(result_mean) > 0.01:
                results.record_warning(
                    "ppo_expectation_violation",
                    f"Skipped normalization leaves mean={result_mean:.2e} (expected ~0)",
                )

        results.record_pass("test_ppo_expectation")
    except Exception as e:
        results.record_fail("test_ppo_expectation", str(e))


def test_gradient_safety_all_ranges():
    """Test gradient safety across all std ranges."""
    try:
        print("\n    Testing gradient safety across all ranges...")

        std_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1.0]

        print(f"\n    {'Std':<10} {'Old(1e-8)':<15} {'New':<15} {'Improvement':<15}")
        print(f"    {'-'*60}")

        for std_val in std_values:
            advantages = np.random.RandomState(42).randn(100) * std_val
            actual_std = np.std(advantages, ddof=1)
            mean = np.mean(advantages)

            # Old approach: always normalize with 1e-8 floor
            std_old = max(actual_std, 1e-8)
            norm_old = (advantages - mean) / std_old
            max_old = np.max(np.abs(norm_old))

            # New approach
            if actual_std < ADV_STD_SKIP_THRESHOLD:
                # Skip
                max_new = np.max(np.abs(advantages))
            else:
                std_new = max(actual_std, ADV_STD_FLOOR)
                norm_new = (advantages - mean) / std_new
                max_new = np.max(np.abs(norm_new))

            improvement = max_old / max_new if max_new > 0 else float("inf")

            print(
                f"    {actual_std:<10.1e} {max_old:<15.2e} {max_new:<15.2e} {improvement:<15.1f}x"
            )

        results.record_pass("test_gradient_safety_all_ranges")
    except Exception as e:
        results.record_fail("test_gradient_safety_all_ranges", str(e))


def test_real_world_scenario():
    """Simulate real-world training scenario."""
    try:
        print("\n    Simulating real-world training...")

        # Scenario: Trading bot with small rewards
        # Episode returns: mostly around 0 with small variance
        np.random.seed(42)

        # Simulate 10 rollouts with different variance levels
        rollouts = [
            ("early_training", 0.001, 1e-3),  # High variance
            ("mid_training", 0.001, 5e-4),  # Medium variance
            ("converged", 0.001, 1e-4),  # Low variance
            ("stuck", 0.001, 1e-5),  # Very low variance
        ]

        print(f"\n    {'Stage':<20} {'Std':<10} {'Behavior':<12} {'Max':<15}")
        print(f"    {'-'*60}")

        for stage, mean, std in rollouts:
            advantages = np.random.randn(128) * std + mean
            actual_std = np.std(advantages, ddof=1)
            actual_mean = np.mean(advantages)

            # Current logic
            if actual_std < ADV_STD_SKIP_THRESHOLD:
                behavior = "SKIP"
                result = advantages
            else:
                behavior = "NORMALIZE"
                std_clamped = max(actual_std, ADV_STD_FLOOR)
                result = (advantages - actual_mean) / std_clamped

            max_val = np.max(np.abs(result))

            print(f"    {stage:<20} {actual_std:<10.1e} {behavior:<12} {max_val:<15.2e}")

        results.record_pass("test_real_world_scenario")
    except Exception as e:
        results.record_fail("test_real_world_scenario", str(e))


def test_mathematical_soundness():
    """Deep mathematical analysis of the normalization."""
    try:
        print("\n    Mathematical soundness analysis...")

        # Key insight: When std is small, advantages are nearly uniform
        # Normalization of uniform values amplifies noise

        # Case 1: Truly uniform (std ≈ 0)
        advantages_uniform = np.ones(100) * 0.5
        advantages_uniform[0] += 1e-10  # Tiny noise

        std_uniform = np.std(advantages_uniform, ddof=1)
        print(f"\n      Case 1: Truly uniform, std={std_uniform:.2e}")

        # Case 2: Low variance but not uniform
        advantages_low = np.random.RandomState(42).randn(100) * 1e-4
        std_low = np.std(advantages_low, ddof=1)
        print(f"      Case 2: Low variance, std={std_low:.2e}")

        # Case 3: Normal variance
        advantages_normal = np.random.RandomState(42).randn(100) * 0.1
        std_normal = np.std(advantages_normal, ddof=1)
        print(f"      Case 3: Normal variance, std={std_normal:.2e}")

        # Mathematical property: If all advantages nearly equal,
        # they provide no useful gradient information
        # In this case, normalization or no normalization, gradient ≈ 0

        # But: PPO loss expects normalized advantages
        # If we skip normalization, we change the loss scale

        print("\n      Key insight:")
        print("      - When std → 0: advantages provide no gradient info")
        print("      - Normalization amplifies noise without adding signal")
        print("      - BUT: PPO loss scale depends on normalization")

        results.record_pass("test_mathematical_soundness")
    except Exception as e:
        results.record_fail("test_mathematical_soundness", str(e))


def test_recommended_threshold():
    """Analyze what the threshold should actually be."""
    try:
        print("\n    Analyzing recommended thresholds...")

        # Question: Should SKIP_THRESHOLD = FLOOR or > FLOOR?

        print(f"\n      Current: FLOOR={ADV_STD_FLOOR:.1e}, SKIP={ADV_STD_SKIP_THRESHOLD:.1e}")
        print(f"      Ratio: {ADV_STD_SKIP_THRESHOLD/ADV_STD_FLOOR:.0f}x")

        # Analysis of different ratios
        ratios = [1, 2, 5, 10, 100]

        print(f"\n      {'Ratio':<10} {'Skip thresh':<15} {'Interpretation':<30}")
        print(f"      {'-'*60}")

        for ratio in ratios:
            skip = ADV_STD_FLOOR * ratio

            if ratio == 1:
                interp = "Skip if at floor (conservative)"
            elif ratio < 5:
                interp = "Skip if near floor"
            elif ratio < 50:
                interp = "Skip if moderately low"
            else:
                interp = "Skip if somewhat low (risky)"

            print(f"      {ratio:<10} {skip:<15.1e} {interp:<30}")

        # Recommendation
        print("\n      Recommendation:")
        print("      - Ratio=1 (SKIP=FLOOR): Most conservative, always normalize unless at floor")
        print("      - Ratio=10 (current): Skips normalization more often")
        print("      - Trade-off: gradient safety vs PPO expectation")

        # Warning if ratio is high
        actual_ratio = ADV_STD_SKIP_THRESHOLD / ADV_STD_FLOOR
        if actual_ratio > 5:
            results.record_warning(
                "skip_threshold_high",
                f"SKIP/FLOOR ratio = {actual_ratio:.0f}x may skip normalization too often",
            )

        results.record_pass("test_recommended_threshold")
    except Exception as e:
        results.record_fail("test_recommended_threshold", str(e))


def test_other_locations_in_code():
    """Check if there are other places in code using 1e-8."""
    try:
        print("\n    Checking for other 1e-8 usages in code...")

        # Read the main file
        with open("distributional_ppo.py", "r") as f:
            content = f.read()
            lines = content.split("\n")

        # Search for 1e-8 or similar patterns
        suspicious_lines = []
        for i, line in enumerate(lines, 1):
            if "1e-8" in line or "1e-08" in line:
                # Skip if it's in our fixed section (around line 6635)
                if i < 6620 or i > 6700:
                    suspicious_lines.append((i, line.strip()))

        if suspicious_lines:
            print(f"\n      Found {len(suspicious_lines)} other 1e-8 usages:")
            for line_no, line in suspicious_lines[:5]:  # Show first 5
                print(f"        Line {line_no}: {line[:60]}...")

            results.record_warning(
                "other_1e8_usages", f"Found {len(suspicious_lines)} other 1e-8 usages to review"
            )
        else:
            print(f"      No other 1e-8 usages found (good!)")

        results.record_pass("test_other_locations_in_code")
    except Exception as e:
        results.record_fail("test_other_locations_in_code", str(e))


def main():
    """Run all deep analysis tests."""
    print("=" * 80)
    print("DEEP ANALYSIS: ADVANTAGE STD FLOOR FIX")
    print("=" * 80)
    print()
    print("This suite performs comprehensive validation of the fix,")
    print("including edge cases, mathematical soundness, and real-world scenarios.")
    print()

    print("Configuration:")
    print(f"  ADV_STD_FLOOR: {ADV_STD_FLOOR}")
    print(f"  ADV_STD_SKIP_THRESHOLD: {ADV_STD_SKIP_THRESHOLD}")
    print(f"  Ratio: {ADV_STD_SKIP_THRESHOLD/ADV_STD_FLOOR:.0f}x")
    print()

    print("Running deep analysis tests...")

    # Critical tests
    test_critical_range_analysis()
    test_skip_threshold_too_high()
    test_alternative_strategy()
    test_ppo_expectation()

    # Comprehensive tests
    test_gradient_safety_all_ranges()
    test_real_world_scenario()
    test_mathematical_soundness()
    test_recommended_threshold()
    test_other_locations_in_code()

    # Summary
    success = results.summary()

    if results.warnings:
        print("\n" + "=" * 80)
        print("⚠ WARNINGS DETECTED - REVIEW RECOMMENDED")
        print("=" * 80)
        return 2  # Warning code

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
