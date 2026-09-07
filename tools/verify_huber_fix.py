#!/usr/bin/env python3
"""
Verification script for quantile Huber loss kappa normalization fix.

This script verifies that the division by kappa has been removed and the
implementation now matches the standard QR-DQN formula.
"""

import sys
import torch

# Import the implementation
sys.path.insert(0, "/home/user/ai-quant-platform")
from distributional_ppo import DistributionalPPO


def test_quadratic_region_independence():
    """
    Test that in quadratic region, loss is independent of kappa.
    This is the key test that verifies division by kappa was removed.
    """
    print("Test 1: Quadratic region independence from kappa")
    print("-" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Error in quadratic region
    error = 0.3
    predicted = torch.tensor([[0.0]], dtype=torch.float32)
    targets = torch.tensor([[error]], dtype=torch.float32)

    # Test with different kappa values
    losses = {}
    for kappa in [0.5, 1.0, 2.0, 5.0]:
        algo._quantile_huber_kappa = kappa
        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
        losses[kappa] = loss.item()
        print(f"  κ = {kappa:3.1f}: loss = {loss.item():.8f}")

    # All losses should be the same
    base_loss = losses[1.0]
    all_equal = all(abs(loss - base_loss) < 1e-6 for loss in losses.values())

    if all_equal:
        print("  ✓ PASS: Loss is independent of kappa in quadratic region")
        return True
    else:
        print("  ✗ FAIL: Loss varies with kappa (division by kappa still present?)")
        return False


def test_linear_region_scaling():
    """
    Test that in linear region, loss scales proportionally with kappa.
    """
    print("\nTest 2: Linear region scaling with kappa")
    print("-" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test with different kappa values
    losses = {}
    for kappa in [1.0, 2.0]:
        algo._quantile_huber_kappa = kappa

        # Error in linear region (3 * kappa)
        error = kappa * 3.0
        predicted = torch.tensor([[0.0]], dtype=torch.float32)
        targets = torch.tensor([[error]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
        losses[kappa] = loss.item()
        print(f"  κ = {kappa:3.1f}, error = {error:4.1f}: loss = {loss.item():.8f}")

    # Calculate expected values
    # For κ=1, error=3: L_H = 1 * (3 - 0.5) = 2.5, with |τ-I|=0.5: loss = 1.25
    # For κ=2, error=6: L_H = 2 * (6 - 1.0) = 10.0, with |τ-I|=0.5: loss = 5.0
    expected_kappa1 = 0.5 * 1.0 * (3.0 - 0.5 * 1.0)  # 1.25
    expected_kappa2 = 0.5 * 2.0 * (6.0 - 0.5 * 2.0)  # 5.0

    print(f"\n  Expected for κ=1.0: {expected_kappa1:.8f}")
    print(f"  Expected for κ=2.0: {expected_kappa2:.8f}")

    match_kappa1 = abs(losses[1.0] - expected_kappa1) < 1e-6
    match_kappa2 = abs(losses[2.0] - expected_kappa2) < 1e-6

    if match_kappa1 and match_kappa2:
        print("  ✓ PASS: Loss scales correctly in linear region")
        return True
    else:
        print("  ✗ FAIL: Loss does not match expected values")
        return False


def test_transition_continuity():
    """
    Test that loss is continuous at the transition point |error| = κ.
    """
    print("\nTest 3: Continuity at transition point |error| = κ")
    print("-" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    all_pass = True
    for kappa in [0.5, 1.0, 2.0]:
        algo._quantile_huber_kappa = kappa

        # Just below transition
        predicted_below = torch.tensor([[0.0]], dtype=torch.float32)
        targets_below = torch.tensor([[kappa - 1e-4]], dtype=torch.float32)
        loss_below = DistributionalPPO._quantile_huber_loss(algo, predicted_below, targets_below)

        # Just above transition
        predicted_above = torch.tensor([[0.0]], dtype=torch.float32)
        targets_above = torch.tensor([[kappa + 1e-4]], dtype=torch.float32)
        loss_above = DistributionalPPO._quantile_huber_loss(algo, predicted_above, targets_above)

        diff = abs(loss_below.item() - loss_above.item())
        rel_diff = diff / max(loss_below.item(), 1e-8)

        print(
            f"  κ = {kappa:3.1f}: below = {loss_below.item():.8f}, above = {loss_above.item():.8f}, diff = {diff:.2e}"
        )

        if rel_diff < 1e-3:
            print(f"    ✓ Continuous")
        else:
            print(f"    ✗ Discontinuous (relative diff: {rel_diff:.2e})")
            all_pass = False

    if all_pass:
        print("  ✓ PASS: Loss is continuous at all transition points")
        return True
    else:
        print("  ✗ FAIL: Loss has discontinuities")
        return False


def test_asymmetric_weighting():
    """
    Test that quantile weighting creates asymmetric loss.
    """
    print("\nTest 4: Asymmetric quantile weighting")
    print("-" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1], dtype=torch.float32)

    algo.policy = PolicyStub()

    error = 0.5  # Quadratic region

    # Overestimation
    predicted_over = torch.tensor([[error]], dtype=torch.float32)
    targets_over = torch.tensor([[0.0]], dtype=torch.float32)
    loss_over = DistributionalPPO._quantile_huber_loss(algo, predicted_over, targets_over)

    # Underestimation
    predicted_under = torch.tensor([[0.0]], dtype=torch.float32)
    targets_under = torch.tensor([[error]], dtype=torch.float32)
    loss_under = DistributionalPPO._quantile_huber_loss(algo, predicted_under, targets_under)

    ratio = loss_under.item() / loss_over.item()
    expected_ratio = 0.9 / 0.1  # 9.0

    print(f"  Overestimation loss:  {loss_over.item():.8f}")
    print(f"  Underestimation loss: {loss_under.item():.8f}")
    print(f"  Ratio: {ratio:.4f} (expected: {expected_ratio:.1f})")

    if abs(ratio - expected_ratio) < 0.1:
        print("  ✓ PASS: Asymmetric weighting works correctly")
        return True
    else:
        print("  ✗ FAIL: Asymmetric weighting is incorrect")
        return False


def main():
    print("=" * 60)
    print("Quantile Huber Loss Kappa Normalization Fix Verification")
    print("=" * 60)
    print()

    results = []

    try:
        results.append(("Quadratic independence", test_quadratic_region_independence()))
        results.append(("Linear scaling", test_linear_region_scaling()))
        results.append(("Transition continuity", test_transition_continuity()))
        results.append(("Asymmetric weighting", test_asymmetric_weighting()))
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe fix successfully removed division by kappa.")
        print("Implementation now matches standard QR-DQN formula.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
