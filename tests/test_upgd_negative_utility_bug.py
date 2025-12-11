"""
Test to verify the negative utility scaling bug in UPGD optimizer.

This test demonstrates that when global_max_util becomes negative,
the scaling logic inverts: parameters with lower utility (more negative)
receive smaller updates, which is opposite to the intended behavior.
"""

import pytest
torch = pytest.importorskip("torch")
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optimizers.upgd import UPGD


def test_negative_utility_inversion():
    """
    Test that demonstrates the inversion bug when all utilities are negative.

    Expected behavior:
    - High utility parameters → small updates (protection)
    - Low utility parameters → large updates (exploration)

    Bug: When global_max_util < 0, this relationship inverts.
    """
    print("\n" + "="*80)
    print("TEST: Negative Utility Inversion Bug")
    print("="*80)

    # Create two simple parameters with different magnitudes
    # We'll engineer the gradients to create negative utilities
    param1 = torch.tensor([2.0], requires_grad=True)  # Will have less negative utility
    param2 = torch.tensor([1.0], requires_grad=True)  # Will have more negative utility

    optimizer = UPGD([param1, param2], lr=0.1, sigma=0.0)  # No noise for clarity

    # Create gradients such that grad * param > 0 (negative utility)
    # utility = -grad * param
    # For param1: utility = -2.0 * 2.0 = -4.0 (more negative)
    # For param2: utility = -1.0 * 1.0 = -1.0 (less negative, "better")
    param1.grad = torch.tensor([2.0])  # Same sign as param1
    param2.grad = torch.tensor([1.0])  # Same sign as param2

    # Store initial values
    param1_initial = param1.data.clone()
    param2_initial = param2.data.clone()

    print(f"\nInitial state:")
    print(f"  param1 = {param1_initial.item():.4f}, grad1 = {param1.grad.item():.4f}")
    print(f"  param2 = {param2_initial.item():.4f}, grad2 = {param2.grad.item():.4f}")
    print(f"\nExpected utilities (utility = -grad * param):")
    print(f"  utility1 = -{param1.grad.item():.4f} * {param1_initial.item():.4f} = {(-param1.grad * param1_initial).item():.4f}")
    print(f"  utility2 = -{param2.grad.item():.4f} * {param2_initial.item():.4f} = {(-param2.grad * param2_initial).item():.4f}")

    # First step: utilities will be initialized
    optimizer.step()

    # Check utilities after first step
    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()

    print(f"\nUtilities after first step:")
    print(f"  avg_utility1 = {utility1:.4f} (more negative, 'worse')")
    print(f"  avg_utility2 = {utility2:.4f} (less negative, 'better')")

    # Compute what the scaling should have been
    global_max = max(utility1, utility2)
    print(f"\nGlobal max utility = {global_max:.4f} (NEGATIVE!)")

    # Compute scaled utilities
    epsilon = 1e-8
    scaled_utility1 = torch.sigmoid(torch.tensor(utility1 / (global_max + epsilon)))
    scaled_utility2 = torch.sigmoid(torch.tensor(utility2 / (global_max + epsilon)))

    update_factor1 = 1 - scaled_utility1
    update_factor2 = 1 - scaled_utility2

    print(f"\nScaling computation (CURRENT BUGGY LOGIC):")
    print(f"  param1: utility/max = {utility1:.4f}/{global_max:.4f} = {utility1/(global_max+epsilon):.4f}")
    print(f"          sigmoid({utility1/(global_max+epsilon):.4f}) = {scaled_utility1.item():.4f}")
    print(f"          update_factor = 1 - {scaled_utility1.item():.4f} = {update_factor1.item():.4f}")
    print(f"\n  param2: utility/max = {utility2:.4f}/{global_max:.4f} = {utility2/(global_max+epsilon):.4f}")
    print(f"          sigmoid({utility2/(global_max+epsilon):.4f}) = {scaled_utility2.item():.4f}")
    print(f"          update_factor = 1 - {scaled_utility2.item():.4f} = {update_factor2.item():.4f}")

    # Compute actual updates
    update1 = param1_initial - param1.data
    update2 = param2_initial - param2.data

    print(f"\nActual parameter updates:")
    print(f"  param1 change = {update1.item():.6f}")
    print(f"  param2 change = {update2.item():.6f}")

    # Analysis
    print(f"\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)

    if update_factor1 < update_factor2:
        print("[FAIL] BUG CONFIRMED!")
        print(f"\n  param1 (MORE negative utility = 'worse') got SMALLER update factor ({update_factor1.item():.4f})")
        print(f"  param2 (LESS negative utility = 'better') got LARGER update factor ({update_factor2.item():.4f})")
        print("\n  Expected behavior: 'worse' parameters should get LARGER updates (exploration)")
        print("  Actual behavior: 'worse' parameters got SMALLER updates (inverted logic!)")

        print("\n  ROOT CAUSE:")
        print("  When global_max < 0, dividing by negative number inverts the relationship:")
        print(f"    More negative utility ({utility1:.4f}) / negative max ({global_max:.4f}) = {utility1/global_max:.4f} (LARGER)")
        print(f"    Less negative utility ({utility2:.4f}) / negative max ({global_max:.4f}) = {utility2/global_max:.4f} (SMALLER)")
        print("  This causes sigmoid to produce inverted outputs!")

        return False
    else:
        print("[PASS] No inversion detected (may need different test case)")
        return True


def test_positive_utility_normal_case():
    """
    Test that the logic works correctly when utilities are positive (normal case).
    """
    print("\n" + "="*80)
    print("TEST: Positive Utility (Normal Case)")
    print("="*80)

    # Create two simple parameters
    param1 = torch.tensor([2.0], requires_grad=True)  # Will have high positive utility
    param2 = torch.tensor([1.0], requires_grad=True)  # Will have low positive utility

    optimizer = UPGD([param1, param2], lr=0.1, sigma=0.0)

    # Create gradients such that grad * param < 0 (positive utility)
    # utility = -grad * param
    # For param1: utility = -(-2.0) * 2.0 = 4.0 (high positive)
    # For param2: utility = -(-1.0) * 1.0 = 1.0 (low positive)
    param1.grad = torch.tensor([-2.0])  # Opposite sign to param1
    param2.grad = torch.tensor([-1.0])  # Opposite sign to param2

    print(f"\nInitial state:")
    print(f"  param1 = {param1.data.item():.4f}, grad1 = {param1.grad.item():.4f}")
    print(f"  param2 = {param2.data.item():.4f}, grad2 = {param2.grad.item():.4f}")
    print(f"\nExpected utilities:")
    print(f"  utility1 = -{param1.grad.item():.4f} * {param1.data.item():.4f} = {(-param1.grad * param1.data).item():.4f} (high)")
    print(f"  utility2 = -{param2.grad.item():.4f} * {param2.data.item():.4f} = {(-param2.grad * param2.data).item():.4f} (low)")

    # First step
    optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()

    print(f"\nUtilities after first step:")
    print(f"  avg_utility1 = {utility1:.4f} (higher)")
    print(f"  avg_utility2 = {utility2:.4f} (lower)")

    global_max = max(utility1, utility2)
    print(f"\nGlobal max utility = {global_max:.4f} (POSITIVE)")

    epsilon = 1e-8
    scaled_utility1 = torch.sigmoid(torch.tensor(utility1 / (global_max + epsilon)))
    scaled_utility2 = torch.sigmoid(torch.tensor(utility2 / (global_max + epsilon)))

    update_factor1 = 1 - scaled_utility1
    update_factor2 = 1 - scaled_utility2

    print(f"\nUpdate factors:")
    print(f"  param1 (high utility): update_factor = {update_factor1.item():.4f} (should be SMALLER)")
    print(f"  param2 (low utility): update_factor = {update_factor2.item():.4f} (should be LARGER)")

    print(f"\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)

    if update_factor1 < update_factor2:
        print("[PASS] CORRECT BEHAVIOR!")
        print(f"\n  param1 (HIGH utility) got SMALLER update factor ({update_factor1.item():.4f}) - PROTECTED")
        print(f"  param2 (LOW utility) got LARGER update factor ({update_factor2.item():.4f}) - EXPLORED")
        return True
    else:
        print("[FAIL] UNEXPECTED: Logic is inverted even for positive utilities!")
        return False


def test_mixed_utility_case():
    """
    Test behavior when utilities are mixed (some positive, some negative).
    """
    print("\n" + "="*80)
    print("TEST: Mixed Utility (Positive and Negative)")
    print("="*80)

    param1 = torch.tensor([2.0], requires_grad=True)   # Will have positive utility
    param2 = torch.tensor([1.0], requires_grad=True)   # Will have negative utility
    param3 = torch.tensor([1.5], requires_grad=True)   # Will have near-zero utility

    optimizer = UPGD([param1, param2, param3], lr=0.1, sigma=0.0)

    # utility1 = -(-1.0) * 2.0 = 2.0 (positive)
    # utility2 = -(1.0) * 1.0 = -1.0 (negative)
    # utility3 = -(-0.01) * 1.5 = 0.015 (near zero)
    param1.grad = torch.tensor([-1.0])
    param2.grad = torch.tensor([1.0])
    param3.grad = torch.tensor([-0.01])

    print(f"\nExpected utilities:")
    print(f"  param1: utility = {(-param1.grad * param1.data).item():.4f} (positive)")
    print(f"  param2: utility = {(-param2.grad * param2.data).item():.4f} (negative)")
    print(f"  param3: utility = {(-param3.grad * param3.data).item():.4f} (near zero)")

    optimizer.step()

    state1 = optimizer.state[param1]
    state2 = optimizer.state[param2]
    state3 = optimizer.state[param3]

    utility1 = state1["avg_utility"].item()
    utility2 = state2["avg_utility"].item()
    utility3 = state3["avg_utility"].item()

    global_max = max(utility1, utility2, utility3)

    print(f"\nActual utilities:")
    print(f"  avg_utility1 = {utility1:.4f}")
    print(f"  avg_utility2 = {utility2:.4f}")
    print(f"  avg_utility3 = {utility3:.4f}")
    print(f"\nGlobal max = {global_max:.4f}")

    epsilon = 1e-8
    update_factor1 = 1 - torch.sigmoid(torch.tensor(utility1 / (global_max + epsilon)))
    update_factor2 = 1 - torch.sigmoid(torch.tensor(utility2 / (global_max + epsilon)))
    update_factor3 = 1 - torch.sigmoid(torch.tensor(utility3 / (global_max + epsilon)))

    print(f"\nUpdate factors:")
    print(f"  param1 (positive utility): {update_factor1.item():.4f}")
    print(f"  param2 (negative utility): {update_factor2.item():.4f}")
    print(f"  param3 (near-zero utility): {update_factor3.item():.4f}")

    print(f"\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)
    print(f"  When global_max is positive, negative utilities get very small sigmoid inputs")
    print(f"  This gives them larger update factors, which may be unexpected.")
    print(f"  Question: Should negative utility parameters be treated differently?")

    return True


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# UPGD Optimizer: Negative Utility Scaling Bug Verification")
    print("#"*80)

    test1_pass = test_positive_utility_normal_case()
    test2_pass = test_negative_utility_inversion()
    test3_pass = test_mixed_utility_case()

    print("\n" + "#"*80)
    print("# SUMMARY")
    print("#"*80)
    print(f"\n  Test 1 (Positive Utility): {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (Negative Utility): {'BUG CONFIRMED' if not test2_pass else 'PASS'}")
    print(f"  Test 3 (Mixed Utility): {'PASS' if test3_pass else 'FAIL'}")

    if not test2_pass:
        print("\n" + "="*80)
        print("CONCLUSION: BUG CONFIRMED!")
        print("="*80)
        print("\nThe UPGD optimizer has inverted scaling logic when all utilities are negative.")
        print("This occurs when gradients and parameters are co-directional (grad * param > 0).")
        print("\nRecommended fix: Use min-max normalization instead of division by global_max.")
        print("="*80)
