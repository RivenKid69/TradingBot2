"""
Deep analysis script to verify UPGD and VGS implementation correctness.

This script tests:
1. UPGD utility normalization (check for "freezing" bug)
2. VGS variance computation (spatial vs stochastic)
"""

import torch
import torch.nn as nn
import numpy as np
from optimizers import UPGD
from variance_gradient_scaler import VarianceGradientScaler


def test_upgd_utility_normalization():
    """Test UPGD utility normalization for correctness."""
    print("\n" + "=" * 80)
    print("TEST 1: UPGD Utility Normalization")
    print("=" * 80)

    # Create simple model
    model = nn.Linear(10, 5, bias=False)
    optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.9, sigma=0.001)

    # Initialize weights to specific values
    with torch.no_grad():
        model.weight.copy_(torch.randn_like(model.weight))

    # Run several optimization steps
    utilities_history = []
    scaled_utilities_history = []

    for step in range(20):
        # Create known gradients
        model.zero_grad()
        model.weight.grad = torch.randn_like(model.weight) * 0.1

        # Store params before update
        params_before = model.weight.data.clone()
        grads = model.weight.grad.clone()

        # Perform step
        optimizer.step()

        # Extract utility from state
        state = optimizer.state[model.weight]
        avg_utility = state["avg_utility"]

        # Compute what scaled_utility would be (reconstructed from algorithm)
        # This requires access to internal computation, but we can check output sanity
        utilities_history.append(avg_utility.clone())

        print(f"\nStep {step + 1}:")
        print(f"  Utility range: [{avg_utility.min().item():.6f}, {avg_utility.max().item():.6f}]")
        print(f"  Utility mean: {avg_utility.mean().item():.6f}")
        print(f"  Utility std: {avg_utility.std().item():.6f}")

    # Check for extreme values that indicate normalization bug
    print("\n" + "-" * 80)
    print("ANALYSIS:")
    final_utility = utilities_history[-1]

    # Check if utilities are in reasonable range
    if final_utility.min() < -1e6 or final_utility.max() > 1e6:
        print("[FAIL] Extreme utility values detected (possible normalization bug)")
        print(f"   Range: [{final_utility.min().item():.2e}, {final_utility.max().item():.2e}]")
        return False
    else:
        print("[PASS] Utility values in reasonable range")
        print(f"   Range: [{final_utility.min().item():.6f}, {final_utility.max().item():.6f}]")

    # Check if parameters are updating (not frozen)
    param_changes = []
    for i in range(1, len(utilities_history)):
        change = (utilities_history[i] - utilities_history[i - 1]).abs().mean().item()
        param_changes.append(change)

    avg_change = np.mean(param_changes)
    if avg_change < 1e-10:
        print("[FAIL] Parameters appear frozen (no updates)")
        print(f"   Avg utility change: {avg_change:.2e}")
        return False
    else:
        print("[PASS] Parameters are updating normally")
        print(f"   Avg utility change: {avg_change:.6f}")

    return True


def test_vgs_spatial_vs_stochastic():
    """
    Test whether VGS computes spatial or stochastic variance.

    KEY DIFFERENCE:
    - Spatial variance: variance across parameter elements at ONE timestep
    - Stochastic variance: variance of gradient estimates OVER TIME for same parameter
    """
    print("\n" + "=" * 80)
    print("TEST 2: VGS - Spatial vs Stochastic Variance")
    print("=" * 80)

    # Create parameter with known properties
    param = nn.Parameter(torch.randn(100))
    vgs = VarianceGradientScaler([param], warmup_steps=5, beta=0.9, alpha=0.1)

    print("\n--- Scenario 1: CONSTANT gradients (zero temporal variance) ---")
    # Apply CONSTANT gradient (no temporal variation)
    # Stochastic variance should be ZERO
    # Spatial variance can still be non-zero (if gradient elements differ)

    constant_grad = torch.linspace(-1.0, 1.0, 100)  # Spatial heterogeneity

    for step in range(30):
        param.grad = constant_grad.clone()  # SAME gradient every time
        vgs.scale_gradients()
        vgs.step()

    var_scenario1 = vgs.get_normalized_variance()
    print(f"Normalized variance: {var_scenario1:.6f}")

    # For CONSTANT gradients:
    # - TRUE stochastic variance should be ~0 (no temporal variation)
    # - Spatial variance would be ~variance(linspace(-1, 1)) / mean^2 (non-zero)

    if var_scenario1 < 0.01:
        print("-> LOW variance detected (consistent with stochastic variance)")
    else:
        print("-> HIGH variance detected (consistent with spatial variance)")

    # Reset VGS
    vgs.reset_statistics()

    print("\n--- Scenario 2: NOISY gradients (high temporal variance) ---")
    # Apply NOISY gradients (high temporal variation)
    # Stochastic variance should be HIGH
    # Spatial variance depends on individual gradient's heterogeneity

    for step in range(30):
        # Mean gradient = 1.0, but with large temporal noise
        param.grad = torch.ones(100) * 1.0 + torch.randn(100) * 5.0
        vgs.scale_gradients()
        vgs.step()

    var_scenario2 = vgs.get_normalized_variance()
    print(f"Normalized variance: {var_scenario2:.6f}")

    print("\n" + "-" * 80)
    print("ANALYSIS:")

    # CRITICAL TEST:
    # If VGS computes STOCHASTIC variance:
    #   - Scenario 1 (constant grads) -> LOW variance
    #   - Scenario 2 (noisy grads) -> HIGH variance
    #
    # If VGS computes SPATIAL variance:
    #   - Both scenarios may have similar variance (depends on spatial heterogeneity)

    print(f"Scenario 1 (constant gradients): {var_scenario1:.6f}")
    print(f"Scenario 2 (noisy gradients): {var_scenario2:.6f}")
    print(f"Ratio (noisy / constant): {var_scenario2 / (var_scenario1 + 1e-8):.2f}x")

    # For TRUE stochastic variance, we expect:
    # - Scenario 2 >> Scenario 1 (at least 10x higher)

    if var_scenario2 > var_scenario1 * 10:
        print("\n[PASS] CONSISTENT with stochastic variance (temporal)")
        print("   -> Noisy gradients produce much higher variance")
        result = "stochastic"
    elif var_scenario1 > 0.01:
        print("\n[FAIL] CONSISTENT with spatial variance (NOT stochastic)")
        print("   -> Constant gradients still show high variance")
        print("   -> Variance reflects spatial heterogeneity, not temporal noise")
        result = "spatial"
    else:
        print("\n[WARN]  AMBIGUOUS result")
        result = "ambiguous"

    # Let's dig deeper into what VGS actually computes
    print("\n--- Deep Dive: What VGS Actually Computes ---")

    # Reset and apply uniform constant gradient (no spatial OR temporal variation)
    vgs.reset_statistics()

    print("\nTest: Uniform CONSTANT gradient (no spatial, no temporal variation)")
    for step in range(30):
        param.grad = torch.ones(100) * 1.0  # Uniform AND constant
        vgs.scale_gradients()
        vgs.step()

    var_uniform_constant = vgs.get_normalized_variance()
    print(f"Normalized variance: {var_uniform_constant:.6f}")

    # For TRUE stochastic variance with uniform constant gradient:
    # - Var[g] = E[(g - E[g])^2] = E[(1.0 - 1.0)^2] = 0
    # - Result should be ~0

    # For spatial variance with uniform gradient:
    # - Var_spatial[g] = variance(ones(100)) = 0
    # - Result should also be ~0

    # This test is inconclusive, but let's check hetero gradients

    print("\nTest: Uniform NOISY gradient (no spatial, yes temporal variation)")
    vgs.reset_statistics()

    for step in range(30):
        # All elements get SAME value, but it changes over time
        noise_value = torch.randn(1).item() * 2.0 + 1.0
        param.grad = torch.ones(100) * noise_value  # Uniform spatially, noisy temporally
        vgs.scale_gradients()
        vgs.step()

    var_uniform_noisy = vgs.get_normalized_variance()
    print(f"Normalized variance: {var_uniform_noisy:.6f}")

    # For TRUE stochastic variance:
    # - Var[g] = Var[noise_value] (high - temporal noise)
    # - BUT: torch.var(torch.ones(100) * noise_value) = 0 (all elements same)
    # - If VGS uses torch.var() at each step, it would see 0 variance!

    # For spatial variance:
    # - Var_spatial = 0 (all elements same at each timestep)
    # - Result should be ~0

    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS:")
    print("=" * 80)

    if var_uniform_noisy < 0.01:
        print("[FAIL] VGS computes SPATIAL variance (torch.var at each timestep)")
        print("   -> Uniform gradients (even with temporal noise) -> zero variance")
        print("   -> This is the BUG!")
        print("\n   Current implementation:")
        print("     grad_variance = grad.var()  # Spatial variance")
        print("     variance_ema = beta * variance_ema + (1-beta) * grad_variance")
        print("\n   Should be:")
        print("     grad_mean_ema = beta * grad_mean_ema + (1-beta) * grad.mean()")
        print("     grad_sq_ema = beta * grad_sq_ema + (1-beta) * (grad**2).mean()")
        print("     stochastic_variance = grad_sq_ema - grad_mean_ema**2")
        return False
    else:
        print("[PASS] VGS computes TRUE stochastic variance")
        print("   -> Temporal noise is correctly captured")
        return True


def main():
    """Run all analysis tests."""
    print("\n" + "=" * 80)
    print("DEEP ANALYSIS: UPGD and VGS Implementation Correctness")
    print("=" * 80)

    results = {}

    # Test 1: UPGD normalization
    try:
        results["upgd_normalization"] = test_upgd_utility_normalization()
    except Exception as e:
        print(f"\n[FAIL] UPGD test CRASHED: {e}")
        results["upgd_normalization"] = False

    # Test 2: VGS spatial vs stochastic
    try:
        results["vgs_variance_type"] = test_vgs_spatial_vs_stochastic()
    except Exception as e:
        print(f"\n[FAIL] VGS test CRASHED: {e}")
        results["vgs_variance_type"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[PASS] All tests passed - implementation appears correct")
    else:
        print("\n[FAIL] Some tests failed - bugs detected")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
