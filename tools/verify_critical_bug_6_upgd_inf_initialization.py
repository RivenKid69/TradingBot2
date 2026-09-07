"""
Critical Bug #6 Verification: UPGD инициализация с -inf

Hypothesis: UPGD инициализирует global_max_util с -inf. Если на первом шаге
нет параметров с градиентами, или все avg_utility отрицательны и max < 0,
то global_max_util остается -inf, что приводит к странному поведению:
util / -inf → 0, sigmoid(0) → 0.5, что дает непредсказуемое поведение.

Expected: Непредсказуемое поведение на первых шагах обучения
Location: optimizers/upgd.py:94
"""

import sys
import torch
import torch.nn as nn
from optimizers.upgd import UPGD


def test_upgd_inf_initialization():
    """Test UPGD behavior with -inf initialization."""

    print("=" * 80)
    print("CRITICAL BUG #6: UPGD инициализация с -inf")
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

    print("1. Testing first optimizer step with no prior state...")
    print()

    # First step: optimizer state is empty, avg_utility will be initialized
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    output = model(x)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    print(f"   Loss: {loss.item():.6f}")

    # Check gradients before step
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"   Parameters with gradients: {has_grads}")

    # Save parameters before step
    params_before = [p.data.clone() for p in model.parameters()]

    try:
        optimizer.step()
        print(f"   [OK] First step completed")
    except Exception as e:
        print(f"   [FAIL] First step crashed: {e}")
        print()
        print("RESULT: BUG CONFIRMED - UPGD crashes on first step")
        return True

    print()
    print("2. Checking parameter updates...")

    params_after = [p.data for p in model.parameters()]

    param_changes = []
    for i, (before, after) in enumerate(zip(params_before, params_after)):
        change = (after - before).abs().max().item()
        param_changes.append(change)
        print(f"   Parameter {i}: max change = {change:.10f}")

    max_change = max(param_changes)
    avg_change = sum(param_changes) / len(param_changes)

    print()
    print("3. Analysis:")
    print(f"   Max parameter change: {max_change:.10f}")
    print(f"   Avg parameter change: {avg_change:.10f}")

    # Check for anomalies
    anomaly_found = False

    if max_change < 1e-12:
        print()
        print("   [WARNING] Parameters barely changed on first step")
        print("   This may indicate -inf initialization causing issues")
        anomaly_found = True

    elif max_change > 100.0:
        print()
        print("   [WARNING] Parameters changed dramatically on first step")
        print("   This may indicate numerical instability from -inf")
        anomaly_found = True

    # Check for NaN or Inf
    for i, after in enumerate(params_after):
        if torch.isnan(after).any():
            print()
            print(f"   [FAIL] Parameter {i} contains NaN!")
            anomaly_found = True
        if torch.isinf(after).any():
            print()
            print(f"   [FAIL] Parameter {i} contains Inf!")
            anomaly_found = True

    print()
    print("4. Checking optimizer internal state...")

    # Inspect optimizer state after first step
    for i, param in enumerate(model.parameters()):
        if param in optimizer.state:
            state = optimizer.state[param]
            avg_utility = state["avg_utility"]
            step_count = state["step"]

            util_min = avg_utility.min().item()
            util_max = avg_utility.max().item()

            print(f"   Param {i}: step={step_count}, avg_utility=[{util_min:.6f}, {util_max:.6f}]")

            # Check for -inf or inf in avg_utility
            if torch.isinf(avg_utility).any():
                print(f"            [WARNING] avg_utility contains inf!")
                anomaly_found = True

    if anomaly_found:
        print()
        print("RESULT: BUG CONFIRMED - UPGD has issues with -inf initialization")
        return True
    else:
        print()
        print("RESULT: BUG NOT FOUND - UPGD handles first step correctly")
        return False


def test_upgd_empty_gradients():
    """Test UPGD behavior when some parameters have no gradients."""

    print()
    print("=" * 80)
    print("Additional Test: UPGD with empty gradients")
    print("=" * 80)
    print()

    # Create model where some parameters won't have gradients
    class PartialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 5)
            self.layer2 = nn.Linear(5, 1)
            self.unused_param = nn.Parameter(torch.randn(5, 5))  # Won't get gradients

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return x

    model = PartialModel()
    optimizer = UPGD(model.parameters(), lr=0.01)

    print("1. Running step with unused parameter (no gradient)...")

    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    output = model(x)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    # Check which parameters have gradients
    params_with_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"   Parameters with gradients: {params_with_grads}/{total_params}")
    print(f"   unused_param has gradient: {model.unused_param.grad is not None}")

    try:
        optimizer.step()
        print(f"   [OK] Step completed with partial gradients")
    except Exception as e:
        print(f"   [FAIL] Step crashed with partial gradients: {e}")
        return True

    # Run second step to see if global_max_util is stable
    output = model(x)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    try:
        optimizer.step()
        print(f"   [OK] Second step completed")
        print()
        print("RESULT: UPGD handles partial gradients correctly")
        return False
    except Exception as e:
        print(f"   [FAIL] Second step crashed: {e}")
        print()
        print("RESULT: BUG CONFIRMED - UPGD fails with partial gradients")
        return True


if __name__ == "__main__":
    print("\n")

    bug1 = test_upgd_inf_initialization()

    bug2 = test_upgd_empty_gradients()

    print("\n")
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if bug1 or bug2:
        print("BUG EXISTS [FAIL]")
        print("Severity: MEDIUM - UPGD initialization issues")
        print()
        print("Scenarios:")
        if bug1:
            print("  ✗ First step with -inf initialization")
        if bug2:
            print("  ✗ Partial gradients causing issues")
        print()
        print("Impact:")
        print("- Unpredictable behavior on first training steps")
        print("- Potential numerical instability")
        print("- May cause training to diverge early")
        sys.exit(1)
    else:
        print("BUG NOT FOUND [OK]")
        print("UPGD initialization works correctly")
        sys.exit(0)
