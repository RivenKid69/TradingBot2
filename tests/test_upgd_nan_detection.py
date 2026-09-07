"""
Comprehensive test for UPGD NaN/Inf detection

This test properly detects when UPGD produces NaN or Inf values in parameters
due to division by zero in scaled_utility computation.

Root cause: When global_max_util = 0, division produces Inf, leading to NaN parameters.
"""

import pytest
import sys

torch = pytest.importorskip("torch")
import torch.nn as nn
from optimizers.upgd import UPGD


def test_upgd_nan_detection():
    """Test if UPGD produces NaN parameters when global_max_util = 0."""

    print("=" * 80)
    print("UPGD NaN/Inf Detection Test")
    print("=" * 80)
    print()

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    # Create UPGD optimizer
    optimizer = UPGD(model.parameters(), lr=0.01)

    print("Scenario: All parameters initialized to zero")
    print("Expected: This may cause global_max_util = 0 -> division by zero")
    print()

    # Zero out all parameters to force avg_utility = 0
    for param in model.parameters():
        param.data.zero_()

    # Create dummy input and target
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Forward pass
    output = model(x)
    loss = ((output - target) ** 2).mean()

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients before step
    print("\nGradients before step:")
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  Param {i}: grad_norm = {grad_norm:.6f}")

    # Save parameters before step
    params_before = [p.data.clone() for p in model.parameters()]

    # Optimizer step
    print("\nRunning optimizer.step()...")
    try:
        optimizer.step()
        print("  [OK] Step completed without crash")
    except Exception as e:
        print(f"  [FAIL] Optimizer crashed: {e}")
        return True

    # Check for NaN/Inf in parameters
    print("\nChecking for NaN/Inf in parameters...")
    has_nan = False
    has_inf = False

    for i, param in enumerate(model.parameters()):
        param_has_nan = torch.isnan(param).any().item()
        param_has_inf = torch.isinf(param).any().item()

        if param_has_nan or param_has_inf:
            print(f"  Param {i}:")
            if param_has_nan:
                print(f"    [FAIL] Contains NaN!")
                has_nan = True
            if param_has_inf:
                print(f"    [FAIL] Contains Inf!")
                has_inf = True
        else:
            max_val = param.abs().max().item()
            print(f"  Param {i}: [OK] max_abs_value = {max_val:.6e}")

    print()

    if has_nan or has_inf:
        print("=" * 80)
        print("RESULT: BUG CONFIRMED - UPGD produces NaN/Inf values")
        print("=" * 80)
        print()
        print("Root cause: Division by zero in scaled_utility computation")
        print("Location: optimizers/upgd.py:143")
        print()
        print("Fix required: Add epsilon protection to prevent division by zero")
        return True
    else:
        print("=" * 80)
        print("RESULT: BUG NOT FOUND - UPGD handles zero parameters correctly")
        print("=" * 80)
        return False


def test_upgd_multiple_scenarios():
    """Test UPGD in various edge case scenarios."""

    print("\n" + "=" * 80)
    print("Additional Edge Case Tests")
    print("=" * 80)
    print()

    scenarios = [
        ("Small positive parameters", 1e-8),
        ("Small negative parameters", -1e-8),
        ("Mixed signs near zero", None),
    ]

    bugs_found = []

    for scenario_name, init_value in scenarios:
        print(f"Scenario: {scenario_name}")
        print("-" * 40)

        model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 1))
        optimizer = UPGD(model.parameters(), lr=0.01)

        # Initialize parameters
        for param in model.parameters():
            if init_value is not None:
                param.data.fill_(init_value)
            else:
                # Mixed signs near zero
                param.data = torch.randn_like(param.data) * 1e-8

        # Run optimization step
        x = torch.randn(3, 5)
        target = torch.randn(3, 1)

        output = model(x)
        loss = ((output - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for NaN/Inf
        has_issue = False
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_issue = True
                break

        if has_issue:
            print(f"  [FAIL] NaN/Inf detected!")
            bugs_found.append(scenario_name)
        else:
            print(f"  [OK] No NaN/Inf")

        print()

    if bugs_found:
        print("Failed scenarios:", bugs_found)
        return True
    else:
        print("All scenarios passed")
        return False


if __name__ == "__main__":
    print("\n")

    bug1 = test_upgd_nan_detection()
    bug2 = test_upgd_multiple_scenarios()

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if bug1 or bug2:
        print("BUG CONFIRMED [FAIL]")
        print()
        print("Severity: HIGH - NaN/Inf in parameters breaks training")
        print()
        print("Impact:")
        print("- Parameters become NaN/Inf during training")
        print("- Model becomes unusable after few steps")
        print("- Numerical instability in edge cases")
        print()
        print("Fix: Add epsilon protection in UPGD division (line 143)")
        sys.exit(1)
    else:
        print("NO BUGS FOUND [OK]")
        print("UPGD handles all edge cases correctly")
        sys.exit(0)
