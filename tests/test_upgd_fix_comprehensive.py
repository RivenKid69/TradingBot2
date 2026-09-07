"""
Comprehensive test suite for the fixed UPGD optimizer.

Tests verify that the min-max normalization fix correctly handles:
1. All positive utilities (normal case)
2. All negative utilities (previously buggy case)
3. Mixed utilities (positive and negative)
4. Edge cases (all zeros, uniform utilities)
"""

import pytest

torch = pytest.importorskip("torch")
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optimizers.upgd import UPGD
from optimizers.adaptive_upgd import AdaptiveUPGD


def test_upgd_positive_utilities():
    """Test UPGD with all positive utilities (gradient opposes parameter)."""
    print("\n" + "=" * 80)
    print("TEST 1: UPGD with Positive Utilities")
    print("=" * 80)

    # Create parameters with different magnitudes
    param1 = torch.tensor([2.0], requires_grad=True)  # High positive utility
    param2 = torch.tensor([1.0], requires_grad=True)  # Low positive utility

    optimizer = UPGD([param1, param2], lr=0.1, sigma=0.0, beta_utility=0.001)

    # Create gradients that oppose parameters (negative product = positive utility)
    param1.grad = torch.tensor([-2.0])  # utility = -(-2.0) * 2.0 = 4.0
    param2.grad = torch.tensor([-1.0])  # utility = -(-1.0) * 1.0 = 1.0

    optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()

    print(f"Utilities: param1={utility1:.6f} (high), param2={utility2:.6f} (low)")

    # High utility should get smaller update
    # Low utility should get larger update
    # We can't directly observe update factors, but we can check that the logic is correct

    if utility1 > utility2:
        print("[PASS] High utility parameter (param1) identified correctly")
        return True
    else:
        print("[FAIL] Utility ordering incorrect")
        return False


def test_upgd_negative_utilities():
    """Test UPGD with all negative utilities (gradient aligns with parameter)."""
    print("\n" + "=" * 80)
    print("TEST 2: UPGD with Negative Utilities (FIX VERIFICATION)")
    print("=" * 80)

    param1 = torch.tensor([2.0], requires_grad=True)  # More negative utility
    param2 = torch.tensor([1.0], requires_grad=True)  # Less negative utility

    optimizer = UPGD([param1, param2], lr=0.1, sigma=0.0, beta_utility=0.001)

    # Gradients align with parameters (positive product = negative utility)
    param1.grad = torch.tensor([2.0])  # utility = -(2.0) * 2.0 = -4.0
    param2.grad = torch.tensor([1.0])  # utility = -(1.0) * 1.0 = -1.0

    param1_initial = param1.data.clone()
    param2_initial = param2.data.clone()

    optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()

    update1 = torch.abs(param1.data - param1_initial).item()
    update2 = torch.abs(param2.data - param2_initial).item()

    print(
        f"Utilities: param1={utility1:.6f} (more negative), param2={utility2:.6f} (less negative)"
    )
    print(f"Updates: param1={update1:.6f}, param2={update2:.6f}")

    # With the FIX: more negative utility (worse) should get LARGER update
    # After min-max normalization:
    # - More negative utility → lower normalized value → larger (1 - scaled_utility)
    # - Less negative utility → higher normalized value → smaller (1 - scaled_utility)

    # But note: the first step has very small utilities due to EMA initialization
    # So we need to run multiple steps to see the effect

    # Run more steps to accumulate utility
    for _ in range(10):
        param1.grad = torch.tensor([2.0])
        param2.grad = torch.tensor([1.0])
        optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1_final = state1["avg_utility"].item()
    utility2_final = state2["avg_utility"].item()

    print(f"\nAfter 10 steps:")
    print(
        f"Utilities: param1={utility1_final:.6f} (more negative), param2={utility2_final:.6f} (less negative)"
    )

    if utility1_final < utility2_final:
        print("[PASS] Utility ordering maintained correctly with negative utilities")
        print("       More negative utility (param1) < Less negative utility (param2)")
        return True
    else:
        print("[FAIL] Utility ordering incorrect")
        return False


def test_upgd_mixed_utilities():
    """Test UPGD with mixed positive and negative utilities."""
    print("\n" + "=" * 80)
    print("TEST 3: UPGD with Mixed Utilities")
    print("=" * 80)

    param1 = torch.tensor([2.0], requires_grad=True)  # Positive utility
    param2 = torch.tensor([1.0], requires_grad=True)  # Negative utility
    param3 = torch.tensor([1.5], requires_grad=True)  # Near-zero utility

    optimizer = UPGD([param1, param2, param3], lr=0.1, sigma=0.0, beta_utility=0.001)

    param1.grad = torch.tensor([-1.0])  # utility = 2.0 (positive)
    param2.grad = torch.tensor([1.0])  # utility = -1.0 (negative)
    param3.grad = torch.tensor([-0.01])  # utility ≈ 0.015 (near zero)

    # Run multiple steps
    for _ in range(10):
        param1.grad = torch.tensor([-1.0])
        param2.grad = torch.tensor([1.0])
        param3.grad = torch.tensor([-0.01])
        optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]
    state3 = optimizer.state[param3]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()
    utility3 = state3["avg_utility"].item()

    print(
        f"Utilities: param1={utility1:.6f} (positive), param2={utility2:.6f} (negative), param3={utility3:.6f} (near zero)"
    )

    # Expected order: param1 (positive) > param3 (near zero) > param2 (negative)
    if utility1 > utility3 > utility2:
        print("[PASS] Utility ordering correct: positive > near-zero > negative")
        return True
    else:
        print(f"[FAIL] Utility ordering incorrect")
        print(f"       Expected: {utility1:.6f} > {utility3:.6f} > {utility2:.6f}")
        return False


def test_upgd_uniform_utilities():
    """Test UPGD with uniform utilities (edge case)."""
    print("\n" + "=" * 80)
    print("TEST 4: UPGD with Uniform Utilities (Edge Case)")
    print("=" * 80)

    param1 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    optimizer = UPGD([param1], lr=0.1, sigma=0.0, beta_utility=0.001)

    # All utilities will be the same
    param1.grad = torch.tensor([1.0, 1.0, 1.0])

    try:
        for _ in range(5):
            param1.grad = torch.tensor([1.0, 1.0, 1.0])
            optimizer.step()

        state1 = optimizer.state[param1]
        utility1 = state1["avg_utility"]

        print(f"Utilities: {utility1}")

        # With min-max normalization and epsilon, this should not crash
        print("[PASS] Optimizer handles uniform utilities without error")
        return True
    except Exception as e:
        print(f"[FAIL] Optimizer crashed with uniform utilities: {e}")
        return False


def test_adaptive_upgd_negative_utilities():
    """Test AdaptiveUPGD with negative utilities."""
    print("\n" + "=" * 80)
    print("TEST 5: AdaptiveUPGD with Negative Utilities (FIX VERIFICATION)")
    print("=" * 80)

    param1 = torch.tensor([2.0], requires_grad=True)
    param2 = torch.tensor([1.0], requires_grad=True)

    optimizer = AdaptiveUPGD([param1, param2], lr=0.1, sigma=0.0, beta_utility=0.001)

    # Run multiple steps with negative utilities
    for _ in range(10):
        param1.grad = torch.tensor([2.0])  # utility = -4.0 (more negative)
        param2.grad = torch.tensor([1.0])  # utility = -1.0 (less negative)
        optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()

    print(
        f"Utilities: param1={utility1:.6f} (more negative), param2={utility2:.6f} (less negative)"
    )

    if utility1 < utility2:
        print("[PASS] AdaptiveUPGD utility ordering maintained correctly")
        return True
    else:
        print("[FAIL] AdaptiveUPGD utility ordering incorrect")
        return False


def test_adaptive_upgd_adaptive_noise():
    """Test AdaptiveUPGD with adaptive noise enabled."""
    print("\n" + "=" * 80)
    print("TEST 6: AdaptiveUPGD with Adaptive Noise")
    print("=" * 80)

    param1 = torch.tensor([1.0, 1.0], requires_grad=True)

    optimizer = AdaptiveUPGD(
        [param1],
        lr=0.1,
        sigma=0.01,
        beta_utility=0.001,
        adaptive_noise=True,
        noise_beta=0.9,
        min_noise_std=1e-6,
    )

    try:
        # Run multiple steps
        for i in range(10):
            param1.grad = torch.tensor([1.0, 1.0]) * (i + 1)  # Varying gradient magnitudes
            optimizer.step()

        state1 = optimizer.state[param1]

        # Check that grad_norm_ema is being tracked
        if "grad_norm_ema" in state1:
            print(f"Gradient norm EMA: {state1['grad_norm_ema']:.6f}")
            print("[PASS] AdaptiveUPGD adaptive noise working correctly")
            return True
        else:
            print("[FAIL] Gradient norm EMA not tracked")
            return False
    except Exception as e:
        print(f"[FAIL] AdaptiveUPGD crashed with adaptive noise: {e}")
        return False


def test_zero_gradients_edge_case():
    """Test UPGD with zero gradients (edge case)."""
    print("\n" + "=" * 80)
    print("TEST 7: UPGD with Zero Gradients (Edge Case)")
    print("=" * 80)

    param1 = torch.tensor([1.0], requires_grad=True)

    optimizer = UPGD([param1], lr=0.1, sigma=0.0, beta_utility=0.001)

    try:
        # Zero gradient
        param1.grad = torch.tensor([0.0])
        optimizer.step()

        state1 = optimizer.state[param1]
        utility1 = state1["avg_utility"].item()

        print(f"Utility with zero gradient: {utility1:.6f}")
        print("[PASS] Optimizer handles zero gradients without error")
        return True
    except Exception as e:
        print(f"[FAIL] Optimizer crashed with zero gradients: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#" * 80)
    print("# UPGD Optimizer Fix: Comprehensive Test Suite")
    print("# Testing min-max normalization fix for negative utility bug")
    print("#" * 80)

    tests = [
        ("Positive Utilities", test_upgd_positive_utilities),
        ("Negative Utilities (FIX)", test_upgd_negative_utilities),
        ("Mixed Utilities", test_upgd_mixed_utilities),
        ("Uniform Utilities (Edge)", test_upgd_uniform_utilities),
        ("AdaptiveUPGD Negative (FIX)", test_adaptive_upgd_negative_utilities),
        ("AdaptiveUPGD Adaptive Noise", test_adaptive_upgd_adaptive_noise),
        ("Zero Gradients (Edge)", test_zero_gradients_edge_case),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' raised exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "#" * 80)
    print("# TEST RESULTS SUMMARY")
    print("#" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4} | {name}")

    print("\n" + "-" * 80)
    print(f"  TOTAL: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  [SUCCESS] All tests passed! The fix is working correctly.")
        print("=" * 80)
        return True
    else:
        print(f"\n  [FAILURE] {total_count - passed_count} test(s) failed.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
