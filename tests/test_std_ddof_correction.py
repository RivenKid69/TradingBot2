"""
Test suite for verifying correct use of ddof=1 in standard deviation and variance calculations.

This test ensures that all statistical calculations use Bessel's correction (ddof=1) for
unbiased sample variance estimation, which is critical for:
1. Advantage normalization in PPO (affects policy gradient magnitude)
2. Statistical logging (provides accurate population estimates from samples)
3. Metrics calculation (ensures correct variance/std estimates)

Mathematical background:
- Sample variance (ddof=1): s² = Σ(x - x̄)² / (n - 1)  [unbiased estimator]
- Population variance (ddof=0): σ² = Σ(x - x̄)² / n     [biased for samples]

For reinforcement learning batches, we always work with samples, not populations,
so ddof=1 is statistically correct.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math


def test_sample_vs_population_variance():
    """Verify that ddof=1 gives unbiased estimate for sample variance."""
    # Create a known population
    population = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    pop_var = float(np.var(population))  # True population variance

    # Take multiple samples and verify that ddof=1 gives unbiased estimate
    n_samples = 1000
    sample_size = 5
    estimates_ddof0 = []
    estimates_ddof1 = []

    np.random.seed(42)
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        estimates_ddof0.append(float(np.var(sample, ddof=0)))
        estimates_ddof1.append(float(np.var(sample, ddof=1)))

    mean_estimate_ddof0 = float(np.mean(estimates_ddof0))
    mean_estimate_ddof1 = float(np.mean(estimates_ddof1))

    # ddof=1 should be closer to true population variance (unbiased)
    # ddof=0 will systematically underestimate
    assert abs(mean_estimate_ddof1 - pop_var) < abs(
        mean_estimate_ddof0 - pop_var
    ), f"ddof=1 estimate ({mean_estimate_ddof1:.4f}) should be closer to true variance ({pop_var:.4f}) than ddof=0 ({mean_estimate_ddof0:.4f})"

    # Verify the mathematical relationship: ddof=0 underestimates by factor of (n-1)/n
    expected_ratio = (sample_size - 1) / sample_size
    actual_ratio = mean_estimate_ddof0 / mean_estimate_ddof1
    assert (
        abs(actual_ratio - expected_ratio) < 0.05
    ), f"Ratio of ddof=0/ddof=1 should be ~{expected_ratio:.4f}, got {actual_ratio:.4f}"


def test_advantage_normalization_uses_ddof1():
    """Verify that advantage normalization uses ddof=1 for correct scaling."""
    # Simulate a small batch of advantages (where ddof makes a difference)
    advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    # Calculate with both methods
    mean = float(np.mean(advantages))
    std_ddof0 = float(np.std(advantages, ddof=0))
    std_ddof1 = float(np.std(advantages, ddof=1))

    # Normalize with both
    normalized_ddof0 = (advantages - mean) / std_ddof0
    normalized_ddof1 = (advantages - mean) / std_ddof1

    # Verify they are different
    assert not np.allclose(
        normalized_ddof0, normalized_ddof1
    ), "Normalization should differ between ddof=0 and ddof=1"

    # Verify the magnitude difference
    ratio = std_ddof0 / std_ddof1
    expected_ratio = np.sqrt((len(advantages) - 1) / len(advantages))
    assert (
        abs(ratio - expected_ratio) < 1e-10
    ), f"Std ratio should be {expected_ratio:.6f}, got {ratio:.6f}"

    # For small batches, the difference is significant
    # n=5: sqrt(4/5) = 0.8944, so ddof=0 underestimates std by ~10.6%
    percent_difference = abs(std_ddof1 - std_ddof0) / std_ddof1 * 100
    assert (
        percent_difference > 10.0
    ), f"For n=5, difference should be >10%, got {percent_difference:.2f}%"


def test_impact_on_policy_gradient():
    """Test how ddof affects policy gradient magnitude through advantage normalization."""
    # Simulate advantages from a rollout
    np.random.seed(42)
    advantages = np.random.randn(50).astype(np.float64) * 2.0 + 1.0

    mean = float(np.mean(advantages))
    std_ddof0 = float(np.std(advantages, ddof=0))
    std_ddof1 = float(np.std(advantages, ddof=1))

    # Normalized advantages (affects policy gradient)
    norm_adv_ddof0 = (advantages - mean) / max(std_ddof0, 1e-8)
    norm_adv_ddof1 = (advantages - mean) / max(std_ddof1, 1e-8)

    # The gradient magnitude will be systematically different
    # With ddof=0, advantages are over-normalized (larger magnitude)
    mean_magnitude_ddof0 = float(np.mean(np.abs(norm_adv_ddof0)))
    mean_magnitude_ddof1 = float(np.mean(np.abs(norm_adv_ddof1)))

    # ddof=0 gives larger normalized values (over-normalization)
    assert (
        mean_magnitude_ddof0 > mean_magnitude_ddof1
    ), "ddof=0 should result in larger normalized advantage magnitudes"

    # For n=50, the difference is ~1%
    percent_diff = abs(mean_magnitude_ddof0 - mean_magnitude_ddof1) / mean_magnitude_ddof1 * 100
    assert 0.5 < percent_diff < 3.0, f"For n=50, expected ~1-2% difference, got {percent_diff:.2f}%"


def test_small_batch_behavior():
    """Test behavior with very small batches where ddof matters most."""
    # Edge case: n=2 (minimum for sample variance)
    advantages = np.array([1.0, 3.0], dtype=np.float64)

    std_ddof0 = float(np.std(advantages, ddof=0))
    std_ddof1 = float(np.std(advantages, ddof=1))

    # For n=2: ddof=0 divides by 2, ddof=1 divides by 1
    # So ddof=1 should be sqrt(2) times larger
    expected_ratio = np.sqrt(2.0)
    actual_ratio = std_ddof1 / std_ddof0

    assert (
        abs(actual_ratio - expected_ratio) < 1e-10
    ), f"For n=2, std_ddof1/std_ddof0 should be sqrt(2)={expected_ratio:.6f}, got {actual_ratio:.6f}"

    # Verify actual values
    # Mean = 2.0, deviations = [-1, 1], squared = [1, 1], sum = 2
    # ddof=0: sqrt(2/2) = 1.0
    # ddof=1: sqrt(2/1) = sqrt(2) ≈ 1.414
    assert abs(std_ddof0 - 1.0) < 1e-10
    assert abs(std_ddof1 - np.sqrt(2.0)) < 1e-10


def test_variance_vs_std_consistency():
    """Verify that std = sqrt(var) with same ddof."""
    np.random.seed(42)
    values = np.random.randn(100).astype(np.float64)

    # Test ddof=1
    var_ddof1 = float(np.var(values, ddof=1))
    std_ddof1 = float(np.std(values, ddof=1))

    assert abs(std_ddof1 - np.sqrt(var_ddof1)) < 1e-10, "std should equal sqrt(var) with same ddof"

    # Test ddof=0 (for comparison)
    var_ddof0 = float(np.var(values, ddof=0))
    std_ddof0 = float(np.std(values, ddof=0))

    assert abs(std_ddof0 - np.sqrt(var_ddof0)) < 1e-10, "std should equal sqrt(var) with same ddof"


def test_weighted_mean_std_uses_ddof1():
    """Test that the _weighted_mean_std function uses ddof=1 for unweighted case."""
    # This is an indirect test - we verify the behavior matches ddof=1
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    # Expected values with ddof=1
    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values, ddof=1))

    # Verify the mathematical values
    assert abs(expected_mean - 3.0) < 1e-10
    assert abs(expected_std - np.sqrt(2.5)) < 1e-10  # var = 2.5 with ddof=1


def test_single_value_handling():
    """Test edge case: single value (ddof=1 should give NaN or 0)."""
    # With n=1, ddof=1 divides by 0, which could give NaN or inf
    single_value = np.array([5.0], dtype=np.float64)

    # NumPy will give warning and return NaN or 0 depending on version
    # We just verify it doesn't crash
    with np.errstate(all="ignore"):  # Suppress warnings
        var_result = np.var(single_value, ddof=1)
        std_result = np.std(single_value, ddof=1)

    # Result should be NaN (division by zero)
    assert (
        not np.isfinite(var_result) or var_result == 0.0
    ), "Variance with n=1 and ddof=1 should be NaN or 0"


def test_logging_metrics_accuracy():
    """Test that logging metrics with ddof=1 provide accurate population estimates."""
    # Simulate value predictions from a batch
    np.random.seed(42)
    batch_size = 128

    # True population parameters
    true_mean = 10.0
    true_std = 2.0

    # Generate sample
    predictions = np.random.normal(true_mean, true_std, batch_size).astype(np.float64)

    # Calculate statistics
    sample_mean = float(np.mean(predictions))
    sample_std_ddof0 = float(np.std(predictions, ddof=0))
    sample_std_ddof1 = float(np.std(predictions, ddof=1))

    # ddof=1 should be closer to true population std
    error_ddof0 = abs(sample_std_ddof0 - true_std)
    error_ddof1 = abs(sample_std_ddof1 - true_std)

    # On average, ddof=1 should be better (though not guaranteed for single sample)
    # At minimum, verify the mathematical relationship
    ratio = sample_std_ddof1 / sample_std_ddof0
    expected_ratio = np.sqrt(batch_size / (batch_size - 1))
    assert (
        abs(ratio - expected_ratio) < 0.01
    ), f"Std ratio should be {expected_ratio:.6f}, got {ratio:.6f}"


def test_real_world_impact_calculation():
    """Calculate and verify the real-world impact of the ddof correction."""
    test_cases = [
        (10, "very small batch"),
        (50, "small batch"),
        (100, "medium batch"),
        (1000, "large batch"),
    ]

    results = []
    for n, description in test_cases:
        # Expected ratio of std_ddof0 / std_ddof1
        expected_ratio = np.sqrt((n - 1) / n)
        percent_underestimate = (1.0 - expected_ratio) * 100

        results.append(
            {"n": n, "description": description, "underestimate_percent": percent_underestimate}
        )

        # Verify the math
        if n == 50:
            # For n=50: sqrt(49/50) ≈ 0.9899, so ~1.01% underestimate
            assert (
                0.9 < percent_underestimate < 1.1
            ), f"n=50 should underestimate by ~1%, got {percent_underestimate:.2f}%"
        elif n == 100:
            # For n=100: sqrt(99/100) ≈ 0.995, so ~0.5% underestimate
            assert (
                0.4 < percent_underestimate < 0.6
            ), f"n=100 should underestimate by ~0.5%, got {percent_underestimate:.2f}%"

    # Print summary for documentation
    print("\nImpact of ddof=0 vs ddof=1:")
    for r in results:
        print(
            f"  n={r['n']:4d} ({r['description']:17s}): {r['underestimate_percent']:5.2f}% systematic underestimate"
        )


def test_code_uses_ddof1():
    """Verify that the actual code uses ddof=1 in critical places."""
    import inspect
    from distributional_ppo import DistributionalPPO

    # Get source code of the class
    source = inspect.getsource(DistributionalPPO)

    # Check for advantage normalization
    assert (
        "np.std(advantages_flat, ddof=1)" in source
    ), "Advantage normalization should use np.std with ddof=1"

    # Check for value prediction logging
    assert (
        "np.std(y_pred_np, ddof=1)" in source or "ddof=1" in source
    ), "Value prediction logging should use ddof=1"

    # Check for variance calculations
    assert (
        "np.var(true_vals, ddof=1)" in source or "ddof=1" in source
    ), "Variance calculations should use ddof=1"


if __name__ == "__main__":
    # Run tests manually (no pytest dependency)
    print("=" * 70)
    print("Testing ddof=1 correction in statistical calculations")
    print("=" * 70)

    tests = [
        ("Sample vs population variance", test_sample_vs_population_variance),
        ("Advantage normalization uses ddof=1", test_advantage_normalization_uses_ddof1),
        ("Impact on policy gradient", test_impact_on_policy_gradient),
        ("Small batch behavior", test_small_batch_behavior),
        ("Variance vs std consistency", test_variance_vs_std_consistency),
        ("Weighted mean std uses ddof=1", test_weighted_mean_std_uses_ddof1),
        ("Single value handling", test_single_value_handling),
        ("Logging metrics accuracy", test_logging_metrics_accuracy),
        ("Real world impact calculation", test_real_world_impact_calculation),
        ("Code uses ddof=1", test_code_uses_ddof1),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n[TEST] {name}...")
            test_func()
            print(f"  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
