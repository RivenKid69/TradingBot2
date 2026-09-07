"""
Test per_quantile logic using pure Python/numpy (no torch dependency).

This verifies the mathematical correctness of the per_quantile implementation.
"""

import numpy as np


def per_quantile_clip_logic(new_quantiles, old_value, clip_delta):
    """
    Simulate per_quantile clipping logic.

    Formula: quantile_clipped = old_value + clip(quantile - old_value, -delta, +delta)
    """
    return old_value + np.clip(new_quantiles - old_value, -clip_delta, clip_delta)


def test_basic_clipping():
    """Test basic per_quantile clipping."""
    print("\n" + "=" * 80)
    print("TEST: Basic per_quantile clipping")
    print("=" * 80)

    old_value = 10.0
    clip_delta = 5.0

    # Example from problem description
    new_quantiles = np.array([5.0, 20.0, 35.0])

    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)

    expected = np.array([5.0, 15.0, 15.0])

    print(f"Old value: {old_value}")
    print(f"Clip delta: {clip_delta}")
    print(f"Bounds: [{old_value - clip_delta}, {old_value + clip_delta}]")
    print(f"New quantiles: {new_quantiles}")
    print(f"Clipped quantiles: {quantiles_clipped}")
    print(f"Expected: {expected}")

    assert np.allclose(quantiles_clipped, expected), f"Expected {expected}, got {quantiles_clipped}"

    # Verify bounds
    assert np.all(
        quantiles_clipped >= old_value - clip_delta
    ), "All quantiles should be >= lower bound"
    assert np.all(
        quantiles_clipped <= old_value + clip_delta
    ), "All quantiles should be <= upper bound"

    print("✓ Basic clipping works correctly")


def test_problem_case_from_description():
    """Test the exact problem case from the issue description."""
    print("\n" + "=" * 80)
    print("TEST: Problem case from issue description")
    print("=" * 80)

    old_value = 10.0
    clip_delta = 5.0
    clip_min = old_value - clip_delta  # 5.0
    clip_max = old_value + clip_delta  # 15.0

    # Original problem: New quantiles [5, 20, 35], mean = 20
    new_quantiles = np.array([5.0, 20.0, 35.0])
    new_mean = new_quantiles.mean()

    print(f"\nOriginal problem:")
    print(f"  old_value = {old_value}, clip_delta = {clip_delta}")
    print(f"  Bounds: [{clip_min}, {clip_max}]")
    print(f"  New quantiles: {new_quantiles}, mean = {new_mean}")

    # mean_only mode (problematic)
    clipped_mean_only = np.clip(new_mean, clip_min, clip_max)  # 15.0
    delta = clipped_mean_only - new_mean  # -5.0
    quantiles_mean_only = new_quantiles + delta  # [0, 15, 30]

    print(f"\nmean_only mode:")
    print(f"  Clipped mean: {clipped_mean_only}")
    print(f"  Delta (shift): {delta}")
    print(f"  Clipped quantiles: {quantiles_mean_only}")

    # Check violations in mean_only
    violation_low = np.any(quantiles_mean_only < clip_min)
    violation_high = np.any(quantiles_mean_only > clip_max)

    if violation_low or violation_high:
        print(f"  ❌ VIOLATION DETECTED in mean_only:")
        if violation_low:
            print(f"     Min quantile {quantiles_mean_only.min():.1f} < {clip_min}")
        if violation_high:
            print(f"     Max quantile {quantiles_mean_only.max():.1f} > {clip_max}")

    assert violation_low or violation_high, "mean_only should allow violations for this case"

    # per_quantile mode (solution)
    quantiles_per_quantile = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)

    print(f"\nper_quantile mode:")
    print(f"  Clipped quantiles: {quantiles_per_quantile}")
    print(f"  Mean after clipping: {quantiles_per_quantile.mean():.1f}")

    # Check no violations in per_quantile
    violation_low_pq = np.any(quantiles_per_quantile < clip_min)
    violation_high_pq = np.any(quantiles_per_quantile > clip_max)

    if not violation_low_pq and not violation_high_pq:
        print(f"  ✓ NO VIOLATIONS in per_quantile mode!")
        print(f"     All quantiles within [{clip_min}, {clip_max}]")

    assert not violation_low_pq and not violation_high_pq, "per_quantile must NOT allow violations"

    print("\n✓ Problem case verified: per_quantile solves the issue!")


def test_batch_specific_clipping():
    """Test that different samples get different clipping bounds."""
    print("\n" + "=" * 80)
    print("TEST: Batch-specific clipping")
    print("=" * 80)

    clip_delta = 5.0

    # Different old_values for each sample
    old_values = np.array([10.0, 20.0, 30.0])

    # Same new quantiles for all samples (for simplicity)
    new_quantiles = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 50.0],  # Sample 1
            [0.0, 10.0, 20.0, 30.0, 50.0],  # Sample 2
            [0.0, 10.0, 20.0, 30.0, 50.0],  # Sample 3
        ]
    )

    print(f"Clip delta: {clip_delta}")
    print(f"Old values: {old_values}")
    print(f"New quantiles (same for all samples): {new_quantiles[0]}")

    # Apply per-sample clipping
    quantiles_clipped = np.zeros_like(new_quantiles)
    for i, old_val in enumerate(old_values):
        quantiles_clipped[i] = per_quantile_clip_logic(new_quantiles[i], old_val, clip_delta)

    print(f"\nClipped quantiles:")
    for i, old_val in enumerate(old_values):
        clip_min = old_val - clip_delta
        clip_max = old_val + clip_delta
        print(f"  Sample {i} (old_value={old_val:.1f}, bounds=[{clip_min:.1f}, {clip_max:.1f}]):")
        print(f"    {quantiles_clipped[i]}")

        # Verify bounds
        assert np.all(quantiles_clipped[i] >= clip_min - 1e-6), f"Sample {i}: quantiles below bound"
        assert np.all(quantiles_clipped[i] <= clip_max + 1e-6), f"Sample {i}: quantiles above bound"

    # Verify samples have different clipped values
    assert not np.allclose(
        quantiles_clipped[0], quantiles_clipped[1]
    ), "Different samples should have different clipped quantiles"
    assert not np.allclose(
        quantiles_clipped[1], quantiles_clipped[2]
    ), "Different samples should have different clipped quantiles"

    print("\n✓ Batch-specific clipping works correctly!")


def test_edge_cases():
    """Test various edge cases."""
    print("\n" + "=" * 80)
    print("TEST: Edge cases")
    print("=" * 80)

    clip_delta = 5.0

    # Test 1: All quantiles below bound
    print("\n1. All quantiles below bound:")
    old_value = 10.0
    new_quantiles = np.array([-50.0, -20.0, -10.0, -5.0, 0.0])
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    expected = np.full_like(new_quantiles, old_value - clip_delta)
    print(f"   New: {new_quantiles} -> Clipped: {quantiles_clipped}")
    assert np.allclose(quantiles_clipped, expected), "All should be clipped to lower bound"
    print("   ✓ All clipped to lower bound")

    # Test 2: All quantiles above bound
    print("\n2. All quantiles above bound:")
    new_quantiles = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    expected = np.full_like(new_quantiles, old_value + clip_delta)
    print(f"   New: {new_quantiles} -> Clipped: {quantiles_clipped}")
    assert np.allclose(quantiles_clipped, expected), "All should be clipped to upper bound"
    print("   ✓ All clipped to upper bound")

    # Test 3: All quantiles within bounds
    print("\n3. All quantiles within bounds:")
    new_quantiles = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    print(f"   New: {new_quantiles} -> Clipped: {quantiles_clipped}")
    assert np.allclose(quantiles_clipped, new_quantiles), "Should remain unchanged"
    print("   ✓ Unchanged when within bounds")

    # Test 4: Zero old_value
    print("\n4. Zero old_value:")
    old_value = 0.0
    new_quantiles = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    expected = np.array([-5.0, -5.0, 0.0, 5.0, 5.0])
    print(f"   New: {new_quantiles} -> Clipped: {quantiles_clipped}")
    assert np.allclose(quantiles_clipped, expected)
    print("   ✓ Zero old_value handled correctly")

    # Test 5: Negative old_value
    print("\n5. Negative old_value:")
    old_value = -10.0
    new_quantiles = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    expected = np.array([-15.0, -10.0, -5.0, -5.0, -5.0])
    print(
        f"   Old value: {old_value}, bounds: [{old_value - clip_delta}, {old_value + clip_delta}]"
    )
    print(f"   New: {new_quantiles} -> Clipped: {quantiles_clipped}")
    assert np.allclose(quantiles_clipped, expected)
    print("   ✓ Negative old_value handled correctly")

    print("\n✓ All edge cases passed!")


def test_cvar_preservation():
    """Test that per_quantile preserves CVaR bounds."""
    print("\n" + "=" * 80)
    print("TEST: CVaR preservation")
    print("=" * 80)

    old_value = 10.0
    clip_delta = 5.0
    clip_min = old_value - clip_delta

    # Distribution with very negative tail (high risk)
    # Quantiles at τ = [0.1, 0.3, 0.5, 0.7, 0.9]
    new_quantiles = np.array([-100.0, -50.0, 10.0, 50.0, 100.0])

    print(f"Old value: {old_value}, clip_delta: {clip_delta}")
    print(f"Bounds: [{clip_min}, {old_value + clip_delta}]")
    print(f"New quantiles (τ=[0.1, 0.3, 0.5, 0.7, 0.9]): {new_quantiles}")

    # Original CVaR at α=0.3 (tail quantiles 0.1, 0.3)
    tail_original = new_quantiles[:2]
    cvar_original = tail_original.mean()
    print(f"\nOriginal:")
    print(f"  Tail quantiles: {tail_original}")
    print(f"  CVaR (α=0.3): {cvar_original:.2f}")

    # After per_quantile clipping
    quantiles_clipped = per_quantile_clip_logic(new_quantiles, old_value, clip_delta)
    tail_clipped = quantiles_clipped[:2]
    cvar_clipped = tail_clipped.mean()

    print(f"\nAfter per_quantile clipping:")
    print(f"  Clipped quantiles: {quantiles_clipped}")
    print(f"  Tail quantiles: {tail_clipped}")
    print(f"  CVaR (α=0.3): {cvar_clipped:.2f}")

    # Verify tail quantiles are bounded
    assert np.all(tail_clipped >= clip_min), f"Tail quantiles must be >= {clip_min}"

    # CVaR must be bounded
    assert cvar_clipped >= clip_min - 1e-6, f"CVaR must be >= {clip_min}"

    # CVaR should be more conservative after clipping
    assert cvar_clipped >= cvar_original, "Clipped CVaR should be >= original (less extreme risk)"

    print(f"\n✓ CVaR properly bounded:")
    print(f"   Original CVaR: {cvar_original:.2f} (extreme risk)")
    print(f"   Clipped CVaR: {cvar_clipped:.2f} (bounded risk >= {clip_min})")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PER_QUANTILE VF CLIPPING: LOGIC VERIFICATION (No Torch)")
    print("=" * 80)

    tests = [
        test_basic_clipping,
        test_problem_case_from_description,
        test_batch_specific_clipping,
        test_edge_cases,
        test_cvar_preservation,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("\n" + "=" * 80)
    print("✓ ALL LOGIC TESTS PASSED!")
    print("=" * 80)
    print("\nConclusion:")
    print("  - per_quantile mode correctly implements element-wise clipping")
    print("  - All quantiles are guaranteed to stay within bounds")
    print("  - CVaR (tail risk) is properly constrained")
    print("  - Batch-specific clipping works correctly")
    print("  - Edge cases are handled properly")
    print("\n" + "=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
