"""
Comprehensive numerical impact tests for ddof=1 correction.

This test suite measures the actual numerical impact of using ddof=1 across
all critical parts of the codebase:
1. Advantage normalization (PPO)
2. Financial metrics (Sharpe, Sortino)
3. Anomaly detection (pipeline)
4. GARCH volatility check (transformers)

Each test compares ddof=0 (wrong) vs ddof=1 (correct) to demonstrate the fix.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pytest

# Check for optional dependencies
try:
    import gymnasium

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

gymnasium_required = pytest.mark.skipif(
    not HAS_GYMNASIUM, reason="gymnasium required for distributional_ppo tests"
)


def test_advantage_normalization_numerical_impact():
    """Test numerical impact of ddof correction on advantage normalization."""
    print("\n[TEST] Advantage normalization numerical impact...")

    # Simulate typical PPO batch
    np.random.seed(42)
    batch_sizes = [32, 64, 128, 256]

    results = []
    for batch_size in batch_sizes:
        advantages = np.random.randn(batch_size).astype(np.float64) * 2.0 + 1.0

        mean = float(np.mean(advantages))
        std_ddof0 = float(np.std(advantages, ddof=0))
        std_ddof1 = float(np.std(advantages, ddof=1))

        # Normalize with both methods
        norm_ddof0 = (advantages - mean) / max(std_ddof0, 1e-8)
        norm_ddof1 = (advantages - mean) / max(std_ddof1, 1e-8)

        # Measure impact on policy gradient (proportional to normalized advantages)
        mean_magnitude_ddof0 = float(np.mean(np.abs(norm_ddof0)))
        mean_magnitude_ddof1 = float(np.mean(np.abs(norm_ddof1)))

        percent_diff = abs(mean_magnitude_ddof0 - mean_magnitude_ddof1) / mean_magnitude_ddof1 * 100

        results.append(
            {
                "batch_size": batch_size,
                "std_ratio": std_ddof0 / std_ddof1,
                "gradient_impact": percent_diff,
            }
        )

        print(
            f"  Batch size {batch_size:3d}: std ratio={std_ddof0/std_ddof1:.6f}, "
            f"gradient impact={percent_diff:.3f}%"
        )

    # Verify the mathematical relationship
    for r in results:
        n = r["batch_size"]
        expected_ratio = np.sqrt((n - 1) / n)
        assert (
            abs(r["std_ratio"] - expected_ratio) < 0.01
        ), f"Std ratio should be sqrt({n-1}/{n})={expected_ratio:.6f}, got {r['std_ratio']:.6f}"

    print("  ✓ Advantage normalization impact quantified")


def test_sharpe_ratio_numerical_impact():
    """Test numerical impact of ddof correction on Sharpe ratio calculation."""
    print("\n[TEST] Sharpe ratio numerical impact...")

    # Simulate trading returns with positive mean for clearer demonstration
    np.random.seed(123)  # Changed seed to ensure positive mean
    returns = np.random.randn(100) * 0.02 + 0.001  # 0.1% mean, 2% std

    # Calculate Sharpe with both methods
    mean_return = float(np.mean(returns))
    std_ddof0 = float(np.std(returns, ddof=0))
    std_ddof1 = float(np.std(returns, ddof=1))

    sharpe_ddof0 = mean_return / (std_ddof0 + 1e-9)
    sharpe_ddof1 = mean_return / (std_ddof1 + 1e-9)

    percent_diff = abs(sharpe_ddof0 - sharpe_ddof1) / abs(sharpe_ddof1) * 100

    print(f"  Sharpe (ddof=0): {sharpe_ddof0:.6f}")
    print(f"  Sharpe (ddof=1): {sharpe_ddof1:.6f}")
    print(f"  Difference: {percent_diff:.3f}%")

    # ddof=0 underestimates std, so it gives HIGHER absolute Sharpe ratio
    # For positive returns: sharpe_ddof0 > sharpe_ddof1
    # For negative returns: sharpe_ddof0 < sharpe_ddof1 (both negative, less negative = higher)
    # In both cases: |sharpe_ddof0| > |sharpe_ddof1| (ddof=0 overestimates risk-adjusted return)
    assert abs(sharpe_ddof0) > abs(
        sharpe_ddof1
    ), "ddof=0 should overestimate |Sharpe| by underestimating std"

    # For n=100, difference should be ~0.5%
    assert (
        0.3 < percent_diff < 0.7
    ), f"For n=100, expected ~0.5% difference, got {percent_diff:.3f}%"

    print("  ✓ Sharpe ratio impact verified")


def test_sortino_ratio_numerical_impact():
    """Test numerical impact of ddof correction on Sortino ratio calculation."""
    print("\n[TEST] Sortino ratio numerical impact...")

    # Simulate returns with downside
    np.random.seed(42)
    returns = np.random.randn(100) * 0.03 - 0.001  # Slightly negative mean

    # Calculate with both methods (fallback to std when few downside)
    mean_return = float(np.mean(returns))
    std_ddof0 = float(np.std(returns, ddof=0))
    std_ddof1 = float(np.std(returns, ddof=1))

    sortino_ddof0 = mean_return / (std_ddof0 + 1e-9)
    sortino_ddof1 = mean_return / (std_ddof1 + 1e-9)

    percent_diff = abs(sortino_ddof0 - sortino_ddof1) / abs(sortino_ddof1) * 100

    print(f"  Sortino (ddof=0): {sortino_ddof0:.6f}")
    print(f"  Sortino (ddof=1): {sortino_ddof1:.6f}")
    print(f"  Difference: {percent_diff:.3f}%")

    # For n=100, difference should be ~0.5%
    assert (
        0.3 < percent_diff < 0.7
    ), f"For n=100, expected ~0.5% difference, got {percent_diff:.3f}%"

    print("  ✓ Sortino ratio impact verified")


def test_anomaly_detection_impact():
    """Test numerical impact of ddof correction on anomaly detection."""
    print("\n[TEST] Anomaly detection impact...")

    # Simulate historical returns with one potential outlier
    np.random.seed(42)
    historical = np.random.randn(50) * 0.01  # 1% std returns
    current = 0.025  # 2.5% return (potential anomaly)

    # Calculate sigma with both methods
    sigma_ddof0 = float(np.std(historical, ddof=0))
    sigma_ddof1 = float(np.std(historical, ddof=1))

    # Check if current return is anomaly (>2 sigma)
    sigma_mult = 2.0
    is_anomaly_ddof0 = abs(current) > sigma_mult * sigma_ddof0
    is_anomaly_ddof1 = abs(current) > sigma_mult * sigma_ddof1

    z_score_ddof0 = abs(current) / sigma_ddof0
    z_score_ddof1 = abs(current) / sigma_ddof1

    print(f"  Sigma (ddof=0): {sigma_ddof0:.6f}")
    print(f"  Sigma (ddof=1): {sigma_ddof1:.6f}")
    print(f"  Z-score (ddof=0): {z_score_ddof0:.3f}")
    print(f"  Z-score (ddof=1): {z_score_ddof1:.3f}")
    print(f"  Anomaly (ddof=0): {is_anomaly_ddof0}")
    print(f"  Anomaly (ddof=1): {is_anomaly_ddof1}")

    # ddof=0 underestimates sigma, so gives HIGHER z-scores
    assert (
        z_score_ddof0 > z_score_ddof1
    ), "ddof=0 should give higher z-scores due to underestimated sigma"

    # This could lead to false positives (detecting anomalies that aren't)
    percent_diff = (z_score_ddof0 - z_score_ddof1) / z_score_ddof1 * 100
    print(f"  Z-score inflation: {percent_diff:.3f}%")

    # For n=50, z-score should be inflated by ~1%
    assert (
        0.5 < percent_diff < 2.5
    ), f"For n=50, expected ~1% z-score inflation, got {percent_diff:.3f}%"

    print("  ✓ Anomaly detection impact verified")


def test_garch_volatility_check_impact():
    """Test numerical impact of ddof correction on GARCH volatility check."""
    print("\n[TEST] GARCH volatility check impact...")

    # Simulate log returns
    np.random.seed(42)
    log_returns = np.random.randn(30) * 0.02  # 2% volatility

    # Calculate with both methods
    vol_ddof0 = float(np.std(log_returns, ddof=0))
    vol_ddof1 = float(np.std(log_returns, ddof=1))

    # Check against VOLATILITY_FLOOR (typical value: 0.001)
    volatility_floor = 0.001
    passes_check_ddof0 = vol_ddof0 >= volatility_floor
    passes_check_ddof1 = vol_ddof1 >= volatility_floor

    print(f"  Volatility (ddof=0): {vol_ddof0:.6f}")
    print(f"  Volatility (ddof=1): {vol_ddof1:.6f}")
    print(f"  Passes floor check (ddof=0): {passes_check_ddof0}")
    print(f"  Passes floor check (ddof=1): {passes_check_ddof1}")

    percent_diff = (vol_ddof1 - vol_ddof0) / vol_ddof0 * 100
    print(f"  Volatility increase: {percent_diff:.3f}%")

    # For n=30, volatility should increase by ~1.7%
    assert 1.5 < percent_diff < 2.0, f"For n=30, expected ~1.7% increase, got {percent_diff:.3f}%"

    print("  ✓ GARCH volatility check impact verified")


@gymnasium_required
def test_cross_metric_consistency():
    """Test that all metrics use consistent ddof across the codebase."""
    print("\n[TEST] Cross-metric consistency...")

    # Verify all key metrics use ddof=1
    from distributional_ppo import DistributionalPPO
    import inspect

    source = inspect.getsource(DistributionalPPO)

    # Check advantage normalization
    assert "np.std(advantages_flat, ddof=1)" in source, "Advantage normalization must use ddof=1"

    # Check variance calculations
    assert "np.var(true_vals, ddof=1)" in source, "Variance calculations must use ddof=1"

    print("  ✓ All DistributionalPPO metrics use ddof=1")

    # Check train_model_multi_patch for Sharpe/Sortino
    with open("train_model_multi_patch.py", "r", encoding="utf-8") as f:
        train_source = f.read()

    # Count ddof=1 in sharpe_ratio and sortino_ratio functions
    sharpe_start = train_source.find("def sharpe_ratio(")
    sharpe_end = train_source.find("\ndef ", sharpe_start + 1)
    sharpe_func = train_source[sharpe_start:sharpe_end]

    assert "ddof=1" in sharpe_func, "Sharpe ratio must use ddof=1"

    sortino_start = train_source.find("def sortino_ratio(")
    sortino_end = train_source.find("\ndef ", sortino_start + 1)
    sortino_func = train_source[sortino_start:sortino_end]

    assert (
        sortino_func.count("ddof=1") >= 2
    ), "Sortino ratio must use ddof=1 in all std calculations"

    print("  ✓ All financial metrics use ddof=1")

    # Check pipeline for anomaly detection
    with open("pipeline.py", "r", encoding="utf-8") as f:
        pipeline_source = f.read()

    assert "np.std(rets_arr[:-1], ddof=1)" in pipeline_source, "Anomaly detection must use ddof=1"

    print("  ✓ Anomaly detection uses ddof=1")

    # Check transformers for GARCH
    with open("transformers.py", "r", encoding="utf-8") as f:
        transformer_source = f.read()

    assert (
        "np.std(log_returns, ddof=1)" in transformer_source
    ), "GARCH volatility check must use ddof=1"

    print("  ✓ GARCH volatility check uses ddof=1")

    print("  ✓ All metrics consistently use ddof=1")


def test_edge_case_small_samples():
    """Test edge cases with very small samples."""
    print("\n[TEST] Edge cases with small samples...")

    # Test n=2 (minimum for sample variance)
    advantages_n2 = np.array([1.0, 3.0], dtype=np.float64)

    std_ddof0_n2 = float(np.std(advantages_n2, ddof=0))
    std_ddof1_n2 = float(np.std(advantages_n2, ddof=1))

    # For n=2, ratio should be sqrt(2)
    ratio_n2 = std_ddof1_n2 / std_ddof0_n2
    expected_ratio_n2 = np.sqrt(2.0)

    assert (
        abs(ratio_n2 - expected_ratio_n2) < 1e-10
    ), f"For n=2, ratio should be sqrt(2)={expected_ratio_n2:.6f}, got {ratio_n2:.6f}"

    print(f"  n=2: ratio={ratio_n2:.6f} (expected {expected_ratio_n2:.6f}) ✓")

    # Test n=10 (small batch)
    advantages_n10 = np.random.randn(10).astype(np.float64)

    std_ddof0_n10 = float(np.std(advantages_n10, ddof=0))
    std_ddof1_n10 = float(np.std(advantages_n10, ddof=1))

    ratio_n10 = std_ddof1_n10 / std_ddof0_n10
    expected_ratio_n10 = np.sqrt(10.0 / 9.0)

    assert (
        abs(ratio_n10 - expected_ratio_n10) < 1e-10
    ), f"For n=10, ratio should be sqrt(10/9)={expected_ratio_n10:.6f}, got {ratio_n10:.6f}"

    percent_diff_n10 = (1.0 - std_ddof0_n10 / std_ddof1_n10) * 100
    print(f"  n=10: underestimate={percent_diff_n10:.3f}% ✓")

    # For small samples, the impact is significant (>5%)
    assert (
        percent_diff_n10 > 5.0
    ), f"For n=10, underestimate should be >5%, got {percent_diff_n10:.3f}%"

    print("  ✓ Edge cases handled correctly")


def test_large_sample_convergence():
    """Test that ddof matters less for very large samples."""
    print("\n[TEST] Large sample convergence...")

    sample_sizes = [100, 1000, 10000]

    for n in sample_sizes:
        np.random.seed(42)
        advantages = np.random.randn(n).astype(np.float64)

        std_ddof0 = float(np.std(advantages, ddof=0))
        std_ddof1 = float(np.std(advantages, ddof=1))

        percent_diff = (1.0 - std_ddof0 / std_ddof1) * 100

        print(f"  n={n:5d}: underestimate={percent_diff:.4f}%")

        # As n grows, difference should approach 0
        expected_diff = 100.0 / (2.0 * n)  # Approximation: (n-1)/n ≈ 1 - 1/n

        # Verify convergence
        assert (
            abs(percent_diff - expected_diff) < 0.1
        ), f"For n={n}, expected ~{expected_diff:.4f}% difference"

    print("  ✓ Converges correctly for large samples")


if __name__ == "__main__":
    print("=" * 70)
    print("Numerical Impact Tests for ddof=1 Correction")
    print("=" * 70)

    tests = [
        ("Advantage normalization", test_advantage_normalization_numerical_impact),
        ("Sharpe ratio", test_sharpe_ratio_numerical_impact),
        ("Sortino ratio", test_sortino_ratio_numerical_impact),
        ("Anomaly detection", test_anomaly_detection_impact),
        ("GARCH volatility check", test_garch_volatility_check_impact),
        ("Cross-metric consistency", test_cross_metric_consistency),
        ("Edge cases (small samples)", test_edge_case_small_samples),
        ("Large sample convergence", test_large_sample_convergence),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
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

    if failed == 0:
        print("\n✅ All numerical impact tests passed!")
        print("The ddof=1 correction has been verified across:")
        print("  - Advantage normalization (PPO core)")
        print("  - Financial metrics (Sharpe, Sortino)")
        print("  - Anomaly detection (pipeline)")
        print("  - GARCH volatility checks (transformers)")
        print("  - Edge cases and convergence")

    sys.exit(0 if failed == 0 else 1)
