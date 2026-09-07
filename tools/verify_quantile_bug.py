#!/usr/bin/env python3
"""
Simplified test to verify the quantile regression loss bug.

This test uses only standard Python (no numpy/torch) to demonstrate
the mathematical error in the current implementation.
"""


def test_quantile_asymmetry():
    """
    Test quantile loss asymmetry without external dependencies.

    According to Dabney et al. 2018, quantile regression loss should satisfy:
    - For τ = 0.25 (25th percentile):
      * Overestimation penalty / Underestimation penalty = (1-τ)/τ = 3.0
    - For τ = 0.75 (75th percentile):
      * Overestimation penalty / Underestimation penalty = (1-τ)/τ = 0.333
    """
    print("=" * 80)
    print("QUANTILE REGRESSION LOSS BUG VERIFICATION")
    print("=" * 80)
    print()
    print("Testing mathematical correctness per Dabney et al. 2018")
    print("Formula: ρ_τ(u) = |u| · |τ - I{u < 0}|, where u = target - predicted")
    print()

    # Test case 1: τ = 0.25 (25th percentile)
    tau = 0.25
    target = 0.0
    predicted_under = -1.0  # Underestimation (predicted < target)
    predicted_over = 1.0  # Overestimation (predicted > target)

    print(f"Test 1: τ = {tau} (25th percentile)")
    print(f"  Target = {target}")
    print()

    # CORRECT implementation
    print("  CORRECT implementation (delta = target - predicted):")
    delta_under_correct = target - predicted_under  # 0 - (-1) = 1
    delta_over_correct = target - predicted_over  # 0 - 1 = -1

    indicator_under_correct = 1.0 if delta_under_correct < 0 else 0.0  # 0 (since 1 >= 0)
    indicator_over_correct = 1.0 if delta_over_correct < 0 else 0.0  # 1 (since -1 < 0)

    coef_under_correct = abs(tau - indicator_under_correct)  # |0.25 - 0| = 0.25
    coef_over_correct = abs(tau - indicator_over_correct)  # |0.25 - 1| = 0.75

    print(f"    Underestimation (pred={predicted_under}, tgt={target}):")
    print(f"      delta = {delta_under_correct}")
    print(f"      indicator = {indicator_under_correct}")
    print(f"      coefficient = {coef_under_correct}")
    print()
    print(f"    Overestimation (pred={predicted_over}, tgt={target}):")
    print(f"      delta = {delta_over_correct}")
    print(f"      indicator = {indicator_over_correct}")
    print(f"      coefficient = {coef_over_correct}")
    print()
    print(f"    Ratio (over/under) = {coef_over_correct/coef_under_correct:.4f}")
    print(f"    Expected ratio = (1-τ)/τ = {(1-tau)/tau:.4f}")

    correct_ratio = coef_over_correct / coef_under_correct
    expected_ratio = (1 - tau) / tau

    if abs(correct_ratio - expected_ratio) < 1e-6:
        print(f"    ✓ CORRECT: Ratio matches expected!")
    else:
        print(f"    ✗ WRONG: Ratio doesn't match!")

    print()
    print("-" * 80)
    print()

    # CURRENT (buggy) implementation
    print("  CURRENT implementation (delta = predicted - target):")
    delta_under_current = predicted_under - target  # -1 - 0 = -1
    delta_over_current = predicted_over - target  # 1 - 0 = 1

    indicator_under_current = 1.0 if delta_under_current < 0 else 0.0  # 1 (since -1 < 0)
    indicator_over_current = 1.0 if delta_over_current < 0 else 0.0  # 0 (since 1 >= 0)

    coef_under_current = abs(tau - indicator_under_current)  # |0.25 - 1| = 0.75
    coef_over_current = abs(tau - indicator_over_current)  # |0.25 - 0| = 0.25

    print(f"    Underestimation (pred={predicted_under}, tgt={target}):")
    print(f"      delta = {delta_under_current}")
    print(f"      indicator = {indicator_under_current}")
    print(f"      coefficient = {coef_under_current}")
    print()
    print(f"    Overestimation (pred={predicted_over}, tgt={target}):")
    print(f"      delta = {delta_over_current}")
    print(f"      indicator = {indicator_over_current}")
    print(f"      coefficient = {coef_over_current}")
    print()
    print(f"    Ratio (over/under) = {coef_over_current/coef_under_current:.4f}")
    print(f"    Expected ratio = (1-τ)/τ = {expected_ratio:.4f}")

    current_ratio = coef_over_current / coef_under_current

    if abs(current_ratio - expected_ratio) < 1e-6:
        print(f"    ✓ CORRECT: Ratio matches expected!")
        bug_exists = False
    else:
        print(f"    ✗ WRONG: Ratio is INVERTED!")
        print(f"    ✗ Actual ratio is 1/3 instead of 3!")
        bug_exists = True

    print()
    print("=" * 80)

    if bug_exists:
        print()
        print("CONCLUSION: BUG CONFIRMED!")
        print()
        print("The current implementation uses:")
        print("  delta = predicted_quantiles - targets")
        print()
        print("This causes the asymmetry coefficients to be INVERTED:")
        print(f"  - Underestimation gets coefficient (1-τ) instead of τ")
        print(f"  - Overestimation gets coefficient τ instead of (1-τ)")
        print()
        print("The fix is simple: change to")
        print("  delta = targets - predicted_quantiles")
        print()
        print("=" * 80)
        return False
    else:
        print()
        print("CONCLUSION: No bug found (implementation is correct)")
        print("=" * 80)
        return True


def test_multiple_tau_values():
    """
    Test across multiple tau values to confirm the pattern.
    """
    print()
    print("=" * 80)
    print("TESTING MULTIPLE TAU VALUES")
    print("=" * 80)
    print()

    tau_values = [0.1, 0.25, 0.5, 0.75, 0.9]

    print("For each τ, the ratio of (overestimation / underestimation) should be (1-τ)/τ")
    print()

    all_bugs_confirmed = True

    for tau in tau_values:
        target = 0.0
        predicted_under = -1.0
        predicted_over = 1.0

        # CURRENT (buggy) implementation
        delta_under = predicted_under - target
        delta_over = predicted_over - target

        indicator_under = 1.0 if delta_under < 0 else 0.0
        indicator_over = 1.0 if delta_over < 0 else 0.0

        coef_under = abs(tau - indicator_under)
        coef_over = abs(tau - indicator_over)

        current_ratio = coef_over / coef_under
        expected_ratio = (1 - tau) / tau

        bug_for_tau = abs(current_ratio - expected_ratio) > 1e-6

        status = "✗ WRONG" if bug_for_tau else "✓ OK"
        print(
            f"  τ = {tau:4.2f}: ratio = {current_ratio:6.4f}, expected = {expected_ratio:6.4f}  {status}"
        )

        all_bugs_confirmed = all_bugs_confirmed and bug_for_tau

    print()
    if all_bugs_confirmed:
        print("All tau values show INVERTED coefficients!")
        print("This confirms the bug is systematic, not a special case.")
    print("=" * 80)


if __name__ == "__main__":
    is_correct = test_quantile_asymmetry()
    test_multiple_tau_values()

    if not is_correct:
        exit(1)  # Indicate failure
    else:
        exit(0)  # Indicate success
