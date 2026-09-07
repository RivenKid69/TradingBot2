"""
Critical Bug #5 Verification: UPGD деление на ноль когда global_max_util = 0

Hypothesis: UPGD optimizer делит avg_utility на global_max_util без защиты от нуля.
Если все avg_utility равны нулю или отрицательны, то global_max_util = 0,
что приводит к делению на ноль: util / 0.0 = inf, sigmoid(inf) = 1.0,
и градиенты полностью обнуляются: (1 - scaled_utility) = 0.

Expected: Обучение блокируется, параметры не обновляются
Location: optimizers/upgd.py:143
"""

import sys
import torch
import torch.nn as nn
from optimizers.upgd import UPGD


def test_upgd_division_by_zero():
    """Test UPGD behavior when global_max_util is zero."""

    print("=" * 80)
    print("CRITICAL BUG #5: UPGD деление на ноль когда global_max_util = 0")
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

    print("1. Setting up scenario where global_max_util = 0...")
    print("   (Force all parameters to zero)")
    print()

    # Zero out all parameters to force avg_utility = 0
    for param in model.parameters():
        param.data.zero_()

    print("2. Running optimization step with zero parameters...")

    # Create dummy input and target
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Forward pass
    output = model(x)
    loss = ((output - target) ** 2).mean()

    print(f"   Loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"   Gradient norms before step: {[f'{g:.6f}' for g in grad_norms]}")

    # Save parameters before step
    params_before = [p.data.clone() for p in model.parameters()]

    # Optimizer step (this should trigger division by zero)
    try:
        optimizer.step()
        print(f"   [OK] Optimizer step completed")
    except Exception as e:
        print(f"   [FAIL] Optimizer crashed: {e}")
        print()
        print("RESULT: BUG CONFIRMED - UPGD crashes on zero global_max_util")
        return True

    print()
    print("3. Checking parameter updates...")

    # Check if parameters changed
    params_after = [p.data for p in model.parameters()]

    param_changes = []
    for i, (before, after) in enumerate(zip(params_before, params_after)):
        change = (after - before).abs().max().item()
        param_changes.append(change)
        print(f"   Parameter {i}: max change = {change:.10f}")

    max_change = max(param_changes)

    print()
    print("4. Analysis:")

    if max_change < 1e-10:
        print(f"   [FAIL] Parameters barely changed (max change: {max_change:.2e})")
        print("   This indicates gradients were zeroed out by division by zero!")
        print()
        print("RESULT: BUG CONFIRMED - UPGD blocks learning when global_max_util = 0")
        return True

    elif max_change > 1e6:
        print(f"   [FAIL] Parameters exploded (max change: {max_change:.2e})")
        print("   This indicates numerical instability from division by zero!")
        print()
        print("RESULT: BUG CONFIRMED - UPGD causes numerical instability")
        return True

    else:
        print(f"   [OK] Parameters updated normally (max change: {max_change:.2e})")
        print()
        print("RESULT: BUG NOT FOUND - UPGD handles zero global_max_util correctly")
        return False


def test_upgd_with_negative_utility():
    """Test UPGD when all avg_utility values are negative."""

    print()
    print("=" * 80)
    print("Additional Test: UPGD with negative avg_utility")
    print("=" * 80)
    print()

    # Create model with negative parameters
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    # Set all parameters to negative values
    for param in model.parameters():
        param.data.fill_(-0.1)

    optimizer = UPGD(model.parameters(), lr=0.01)

    # Run one step to initialize optimizer state
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    output = model(x)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    # Force avg_utility to be negative
    # avg_utility = -grad * param
    # If param < 0 and grad > 0, then avg_utility < 0
    print("1. Checking avg_utility values after first step...")

    optimizer.step()

    # Check optimizer state
    for i, param in enumerate(model.parameters()):
        if param in optimizer.state:
            avg_util = optimizer.state[param]["avg_utility"]
            max_util = avg_util.max().item()
            min_util = avg_util.min().item()
            print(f"   Param {i}: avg_utility range = [{min_util:.6f}, {max_util:.6f}]")

    print()
    print("2. Running second step to check behavior...")

    output = model(x)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    params_before = [p.data.clone() for p in model.parameters()]

    try:
        optimizer.step()
        print("   [OK] Second step completed")
    except Exception as e:
        print(f"   [FAIL] Second step crashed: {e}")
        return True

    # Check parameter updates
    params_after = [p.data for p in model.parameters()]
    max_change = max(
        (after - before).abs().max().item() for before, after in zip(params_before, params_after)
    )

    print(f"   Max parameter change: {max_change:.10f}")

    if max_change < 1e-10:
        print("   [FAIL] Parameters not updated with negative avg_utility")
        return True
    else:
        print("   [OK] Parameters updated normally")
        return False


if __name__ == "__main__":
    print("\n")

    bug1 = test_upgd_division_by_zero()

    bug2 = test_upgd_with_negative_utility()

    print("\n")
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if bug1 or bug2:
        print("BUG EXISTS [FAIL]")
        print("Severity: CRITICAL - UPGD division by zero blocks learning")
        print()
        print("Scenarios:")
        if bug1:
            print("  ✗ Zero global_max_util (all params = 0)")
        if bug2:
            print("  ✗ Negative avg_utility (negative params)")
        print()
        print("Impact:")
        print("- Gradients zeroed out when global_max_util = 0")
        print("- Learning completely blocked in certain parameter states")
        print("- Numerical instability (inf/nan)")
        sys.exit(1)
    else:
        print("BUG NOT FOUND [OK]")
        print("UPGD handles edge cases correctly")
        sys.exit(0)
