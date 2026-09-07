"""
Test mathematical correctness of quantile regression loss.

This test verifies that the quantile loss implementation follows the correct formula
from Dabney et al. 2018: "Distributional Reinforcement Learning with Quantile Regression"

The correct formula is:
    ρ_τ(u) = u · (τ - I{u < 0})

where u = target - predicted

This translates to:
- When predicted < target (underestimation): coefficient = τ
- When predicted ≥ target (overestimation): coefficient = (1 - τ)
"""

import numpy as np


def quantile_loss_reference(predicted, target, tau):
    """
    Reference implementation of quantile loss (without Huber).

    Based on Dabney et al. 2018, equation (9):
    ρ_τ(u) = u · (τ - I{u < 0}), where u = target - predicted

    Args:
        predicted: Predicted quantile value
        target: Target value
        tau: Quantile level

    Returns:
        Quantile loss value
    """
    u = target - predicted
    indicator = float(u < 0.0)
    loss = u * (tau - indicator)
    return loss


def quantile_loss_with_huber_reference(predicted, target, tau, kappa=1.0):
    """
    Reference implementation of quantile Huber loss.

    ρ_τ^κ(u) = |τ - I{u < 0}| · L_κ(u)

    where:
    - u = target - predicted
    - L_κ(u) is the Huber loss

    Args:
        predicted: Predicted quantile value
        target: Target value
        tau: Quantile level
        kappa: Huber loss threshold

    Returns:
        Quantile Huber loss value
    """
    u = target - predicted
    indicator = float(u < 0.0)

    # Huber loss
    abs_u = abs(u)
    if abs_u <= kappa:
        huber = 0.5 * u**2
    else:
        huber = kappa * (abs_u - 0.5 * kappa)

    # Quantile coefficient
    coefficient = abs(tau - indicator)

    return coefficient * huber


def test_quantile_loss_asymmetry():
    """
    Test that quantile loss has correct asymmetry.

    For τ < 0.5: should penalize overestimation more
    For τ > 0.5: should penalize underestimation more
    For τ = 0.5: should penalize both equally (median)
    """
    print("Testing quantile loss asymmetry...")

    # Test with τ = 0.25 (25th percentile)
    tau = 0.25
    target = 0.0

    # Underestimation: predicted < target
    predicted_under = -1.0
    loss_under = quantile_loss_reference(predicted_under, target, tau)

    # Overestimation: predicted > target
    predicted_over = 1.0
    loss_over = quantile_loss_reference(predicted_over, target, tau)

    print(f"\nτ = {tau} (25th percentile):")
    print(f"  Underestimation (pred=-1, tgt=0): loss = {loss_under:.4f}")
    print(f"  Overestimation (pred=+1, tgt=0): loss = {loss_over:.4f}")
    print(f"  Ratio (over/under) = {loss_over/loss_under:.4f}")
    print(f"  Expected ratio = (1-τ)/τ = {(1-tau)/tau:.4f}")

    # For τ = 0.25, we want MORE penalty for overestimation
    # loss_over should be 3x loss_under
    expected_ratio = (1 - tau) / tau
    actual_ratio = loss_over / loss_under

    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6), \
        f"Asymmetry incorrect for τ={tau}: expected {expected_ratio}, got {actual_ratio}"

    # Test with τ = 0.75 (75th percentile)
    tau = 0.75

    loss_under = quantile_loss_reference(predicted_under, target, tau)
    loss_over = quantile_loss_reference(predicted_over, target, tau)

    print(f"\nτ = {tau} (75th percentile):")
    print(f"  Underestimation (pred=-1, tgt=0): loss = {loss_under:.4f}")
    print(f"  Overestimation (pred=+1, tgt=0): loss = {loss_over:.4f}")
    print(f"  Ratio (over/under) = {loss_over/loss_under:.4f}")
    print(f"  Expected ratio = (1-τ)/τ = {(1-tau)/tau:.4f}")

    # For τ = 0.75, we want MORE penalty for underestimation
    # loss_under should be 3x loss_over
    expected_ratio = (1 - tau) / tau
    actual_ratio = loss_over / loss_under

    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6), \
        f"Asymmetry incorrect for τ={tau}: expected {expected_ratio}, got {actual_ratio}"

    # Test with τ = 0.5 (median)
    tau = 0.5

    loss_under = quantile_loss_reference(predicted_under, target, tau)
    loss_over = quantile_loss_reference(predicted_over, target, tau)

    print(f"\nτ = {tau} (median):")
    print(f"  Underestimation (pred=-1, tgt=0): loss = {loss_under:.4f}")
    print(f"  Overestimation (pred=+1, tgt=0): loss = {loss_over:.4f}")
    print(f"  Ratio (over/under) = {loss_over/loss_under:.4f}")

    # For τ = 0.5, losses should be equal
    assert np.isclose(loss_under, loss_over, rtol=1e-6), \
        f"Losses should be equal for τ=0.5: {loss_under} vs {loss_over}"

    print("\n✓ Asymmetry test passed!")


def test_current_implementation():
    """
    Test the CURRENT (potentially buggy) implementation.
    """
    print("\n" + "="*80)
    print("Testing CURRENT implementation (delta = predicted - targets)")
    print("="*80)

    # Simulate current implementation
    def current_quantile_huber_loss(predicted, target, tau, kappa=1.0):
        # CURRENT (possibly wrong) implementation
        delta = predicted - target  # Q - T (potentially inverted!)
        abs_delta = abs(delta)

        if abs_delta <= kappa:
            huber = 0.5 * delta**2
        else:
            huber = kappa * (abs_delta - 0.5 * kappa)

        indicator = float(delta < 0.0)  # I{Q < T}
        coefficient = abs(tau - indicator)

        return coefficient * huber

    tau = 0.25
    target = 0.0
    predicted_under = -1.0  # Underestimation
    predicted_over = 1.0   # Overestimation

    loss_under_current = current_quantile_huber_loss(predicted_under, target, tau)
    loss_over_current = current_quantile_huber_loss(predicted_over, target, tau)

    print(f"\nτ = {tau} (25th percentile):")
    print(f"  Underestimation (pred=-1, tgt=0): loss = {loss_under_current:.4f}")
    print(f"  Overestimation (pred=+1, tgt=0): loss = {loss_over_current:.4f}")
    print(f"  Ratio (over/under) = {loss_over_current/loss_under_current:.4f}")
    print(f"  Expected ratio = {(1-tau)/tau:.4f}")

    # Check if it's wrong
    expected_ratio = (1 - tau) / tau
    actual_ratio = loss_over_current / loss_under_current

    if not np.isclose(actual_ratio, expected_ratio, rtol=1e-6):
        print(f"\n❌ CURRENT IMPLEMENTATION IS WRONG!")
        print(f"   Expected ratio: {expected_ratio:.4f}")
        print(f"   Actual ratio: {actual_ratio:.4f}")
        print(f"   The coefficients are INVERTED!")
        return False
    else:
        print(f"\n✓ Current implementation is correct")
        return True


def test_fixed_implementation():
    """
    Test the FIXED implementation.
    """
    print("\n" + "="*80)
    print("Testing FIXED implementation (delta = targets - predicted)")
    print("="*80)

    # Simulate fixed implementation
    def fixed_quantile_huber_loss(predicted, target, tau, kappa=1.0):
        # FIXED implementation
        delta = target - predicted  # T - Q (correct!)
        abs_delta = abs(delta)

        if abs_delta <= kappa:
            huber = 0.5 * delta**2
        else:
            huber = kappa * (abs_delta - 0.5 * kappa)

        indicator = float(delta < 0.0)  # I{T < Q}
        coefficient = abs(tau - indicator)

        return coefficient * huber

    tau = 0.25
    target = 0.0
    predicted_under = -1.0  # Underestimation
    predicted_over = 1.0   # Overestimation

    loss_under_fixed = fixed_quantile_huber_loss(predicted_under, target, tau)
    loss_over_fixed = fixed_quantile_huber_loss(predicted_over, target, tau)

    print(f"\nτ = {tau} (25th percentile):")
    print(f"  Underestimation (pred=-1, tgt=0): loss = {loss_under_fixed:.4f}")
    print(f"  Overestimation (pred=+1, tgt=0): loss = {loss_over_fixed:.4f}")
    print(f"  Ratio (over/under) = {loss_over_fixed/loss_under_fixed:.4f}")
    print(f"  Expected ratio = {(1-tau)/tau:.4f}")

    # Check if it's correct
    expected_ratio = (1 - tau) / tau
    actual_ratio = loss_over_fixed / loss_under_fixed

    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6), \
        f"Fixed implementation still wrong: expected {expected_ratio}, got {actual_ratio}"

    print(f"\n✓ Fixed implementation is CORRECT!")
    return True


def test_coefficient_values():
    """
    Test that coefficients have correct values in different scenarios.
    """
    print("\n" + "="*80)
    print("Testing coefficient values for different quantile levels")
    print("="*80)

    # Test various tau values
    tau_values = [0.1, 0.25, 0.5, 0.75, 0.9]

    for tau in tau_values:
        print(f"\nτ = {tau}:")

        target = 0.0
        predicted_under = -1.0
        predicted_over = 1.0

        # Reference implementation
        loss_under = quantile_loss_with_huber_reference(predicted_under, target, tau)
        loss_over = quantile_loss_with_huber_reference(predicted_over, target, tau)

        # For τ-quantile:
        # - Underestimation should have coefficient τ
        # - Overestimation should have coefficient (1 - τ)

        # Since |error| = 1.0 for both, and Huber(±1) is same, we can extract coefficients
        # Huber(1) with kappa=1.0: 0.5 * 1^2 = 0.5
        huber_value = 0.5

        coef_under = loss_under / huber_value
        coef_over = loss_over / huber_value

        print(f"  Underestimation coefficient: {coef_under:.4f} (expected: {tau:.4f})")
        print(f"  Overestimation coefficient: {coef_over:.4f} (expected: {1-tau:.4f})")

        assert np.isclose(coef_under, tau, rtol=1e-6), \
            f"Underestimation coefficient wrong for τ={tau}"
        assert np.isclose(coef_over, 1-tau, rtol=1e-6), \
            f"Overestimation coefficient wrong for τ={tau}"

    print("\n✓ Coefficient values test passed!")


if __name__ == "__main__":
    print("QUANTILE REGRESSION LOSS CORRECTNESS TEST")
    print("=" * 80)
    print("\nThis test verifies the mathematical correctness of quantile loss")
    print("according to Dabney et al. 2018.\n")

    # Test reference implementation
    test_quantile_loss_asymmetry()
    test_coefficient_values()

    # Test current implementation (should fail if bug exists)
    current_is_correct = test_current_implementation()

    # Test fixed implementation
    test_fixed_implementation()

    if not current_is_correct:
        print("\n" + "="*80)
        print("CONCLUSION: The current implementation HAS A BUG!")
        print("The delta sign is inverted, causing incorrect asymmetry.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("CONCLUSION: The current implementation is correct!")
        print("="*80)
