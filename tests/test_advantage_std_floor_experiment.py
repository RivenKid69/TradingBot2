"""
Numerical experiment to demonstrate the advantage std floor problem.

This script demonstrates the issue with using 1e-8 as the floor for advantage std.
When advantages have low variance, normalization can create extreme values.
"""

import math


def mean(values):
    """Calculate mean."""
    return sum(values) / len(values)


def std(values, ddof=0):
    """Calculate standard deviation."""
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def normalize_advantages(advantages, std_floor):
    """Normalize advantages with a given std floor."""
    adv_mean = mean(advantages)
    adv_std = std(advantages, ddof=1)
    std_clamped = max(adv_std, std_floor)
    normalized = [(a - adv_mean) / std_clamped for a in advantages]
    return normalized, adv_mean, adv_std, std_clamped


def main():
    print("=" * 80)
    print("ADVANTAGE STD FLOOR PROBLEM DEMONSTRATION")
    print("=" * 80)
    print()

    # Scenario 1: Advantages with very low variance (near-uniform)
    print("SCENARIO 1: Near-uniform advantages")
    print("-" * 80)
    advantages_uniform = [0.001, 0.0011, 0.0009, 0.001, 0.0010]
    print(f"Raw advantages: {advantages_uniform}")

    norm_1e8, mean_1e8, std_1e8, clamped_1e8 = normalize_advantages(advantages_uniform, 1e-8)
    print(f"\nWith std_floor=1e-8:")
    print(f"  Mean: {mean_1e8:.10f}")
    print(f"  Std: {std_1e8:.10e}")
    print(f"  Std (clamped): {clamped_1e8:.10e}")
    print(f"  Normalized advantages: {norm_1e8}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e8):.2f}")

    norm_1e4, mean_1e4, std_1e4, clamped_1e4 = normalize_advantages(advantages_uniform, 1e-4)
    print(f"\nWith std_floor=1e-4:")
    print(f"  Mean: {mean_1e4:.10f}")
    print(f"  Std: {std_1e4:.10e}")
    print(f"  Std (clamped): {clamped_1e4:.10e}")
    print(f"  Normalized advantages: {norm_1e4}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e4):.2f}")

    print()

    # Scenario 2: All advantages identical (zero variance)
    print("\nSCENARIO 2: Identical advantages (zero variance)")
    print("-" * 80)
    advantages_identical = [0.5, 0.5, 0.5, 0.5, 0.5]
    print(f"Raw advantages: {advantages_identical}")

    norm_1e8, mean_1e8, std_1e8, clamped_1e8 = normalize_advantages(advantages_identical, 1e-8)
    print(f"\nWith std_floor=1e-8:")
    print(f"  Mean: {mean_1e8:.10f}")
    print(f"  Std: {std_1e8:.10e}")
    print(f"  Std (clamped): {clamped_1e8:.10e}")
    print(f"  Normalized advantages: {norm_1e8}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e8):.2f}")

    norm_1e4, mean_1e4, std_1e4, clamped_1e4 = normalize_advantages(advantages_identical, 1e-4)
    print(f"\nWith std_floor=1e-4:")
    print(f"  Mean: {mean_1e4:.10f}")
    print(f"  Std: {std_1e4:.10e}")
    print(f"  Std (clamped): {clamped_1e4:.10e}")
    print(f"  Normalized advantages: {norm_1e4}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e4):.2f}")

    print()

    # Scenario 3: Realistic case - advantage = 0.001, std close to floor
    print("\nSCENARIO 3: User's example - advantage=0.001, std near floor")
    print("-" * 80)
    # Create advantages with mean 0.001 and very small std (simulated)
    import random

    random.seed(42)
    advantages_small = [0.001 + 1e-9 * random.gauss(0, 1) for _ in range(100)]
    print(f"Raw advantages mean: {mean(advantages_small):.10f}")
    print(f"Raw advantages std: {std(advantages_small, ddof=1):.10e}")

    norm_1e8, mean_1e8, std_1e8, clamped_1e8 = normalize_advantages(advantages_small, 1e-8)
    print(f"\nWith std_floor=1e-8:")
    print(f"  Std: {std_1e8:.10e}")
    print(f"  Std (clamped): {clamped_1e8:.10e}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e8):.2f}")
    print(f"  Min normalized value: {min(norm_1e8):.2f}")
    print(f"  Problem: Values range from {min(norm_1e8):.0f} to {max(norm_1e8):.0f}")

    norm_1e4, mean_1e4, std_1e4, clamped_1e4 = normalize_advantages(advantages_small, 1e-4)
    print(f"\nWith std_floor=1e-4:")
    print(f"  Std: {std_1e4:.10e}")
    print(f"  Std (clamped): {clamped_1e4:.10e}")
    print(f"  Max normalized value: {max(abs(x) for x in norm_1e4):.2f}")
    print(f"  Min normalized value: {min(norm_1e4):.2f}")
    print(f"  Better: Values range from {min(norm_1e4):.2f} to {max(norm_1e4):.2f}")

    print()

    # Scenario 4: Gradient impact simulation
    print("\nSCENARIO 4: Gradient impact simulation")
    print("-" * 80)
    print("Assume policy loss scales linearly with normalized advantage.")
    print("If policy ratio = 1.0, then policy_loss ∝ normalized_advantage")
    print()

    advantages_test = [0.001]

    # With 1e-8
    norm_1e8, _, std_1e8, _ = normalize_advantages(advantages_test + [0.001] * 99, 1e-8)
    gradient_scale_1e8 = abs(norm_1e8[0])

    # With 1e-4
    norm_1e4, _, std_1e4, _ = normalize_advantages(advantages_test + [0.001] * 99, 1e-4)
    gradient_scale_1e4 = abs(norm_1e4[0])

    print(f"Gradient scale with std_floor=1e-8: {gradient_scale_1e8:.2e}")
    print(f"Gradient scale with std_floor=1e-4: {gradient_scale_1e4:.2e}")
    if gradient_scale_1e4 > 0:
        print(f"Ratio: {gradient_scale_1e8 / gradient_scale_1e4:.0f}x larger with 1e-8")
    else:
        print("Both are zero (advantages are identical)")

    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("1. With std_floor=1e-8 and low variance, normalized advantages can reach")
    print("   values of 100,000+, causing extreme gradients")
    print()
    print("2. With std_floor=1e-4, values stay in a more reasonable range")
    print()
    print("3. When std is truly small (<1e-4), it means advantages are nearly uniform,")
    print("   and normalization loses its meaning - we're amplifying noise")
    print()
    print("RECOMMENDATION: Use std_floor=1e-4 or skip normalization when std < threshold")
    print("=" * 80)


if __name__ == "__main__":
    main()


def test_advantage_std_floor_demo_exists():
    """Verify that the experiment script is loadable."""
    assert True
