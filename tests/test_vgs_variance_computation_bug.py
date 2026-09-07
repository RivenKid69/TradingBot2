"""
Test to verify the VGS stochastic variance computation bug.

ISSUE: variance_gradient_scaler.py:279-280 computes:
    grad_mean_current = grad.mean().item()
    grad_sq_current = grad_mean_current ** 2  # Square of mean (E[g])²

But for stochastic variance Var[g] = E[g²] - E[g]², we need:
    E[g²] = mean of squares, NOT square of mean!

This test demonstrates the mathematical difference and impact.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from variance_gradient_scaler import VarianceGradientScaler


def test_variance_computation_difference():
    """Test showing the difference between square-of-mean vs mean-of-squares."""
    print("\n" + "=" * 80)
    print("TEST 1: Variance Computation - Square of Mean vs Mean of Squares")
    print("=" * 80)

    # Example gradient tensor
    grad = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])

    # CURRENT CODE (POTENTIALLY WRONG)
    grad_mean = grad.mean().item()  # E[g]
    grad_sq_wrong = grad_mean**2  # (E[g])² - square of mean

    # PROPOSED FIX (CORRECT?)
    grad_sq_correct = (grad**2).mean().item()  # E[g²] - mean of squares

    print(f"\nGradient: {grad.tolist()}")
    print(f"\nCURRENT CODE:")
    print(f"  E[g] = {grad_mean:.6f}")
    print(f"  (E[g])^2 = {grad_sq_wrong:.6f}")

    print(f"\nPROPOSED FIX:")
    print(f"  E[g] = {grad_mean:.6f}")
    print(f"  E[g^2] = {grad_sq_correct:.6f}")

    print(f"\nDIFFERENCE: {abs(grad_sq_correct - grad_sq_wrong):.6f}")
    print(f"RATIO: E[g^2] / (E[g])^2 = {grad_sq_correct / (grad_sq_wrong + 1e-10):.2f}x")

    # For variance computation
    # If we track E[g] and E[g^2] over time, then:
    # Var[g] = E[g^2] - E[g]^2
    # But current code tracks E[g] and E[(E[g])^2], which is different!

    assert grad_sq_correct != grad_sq_wrong, "These should be different!"
    print("\n[OK] CONFIRMED: Square of mean != Mean of squares")


def test_variance_computation_with_heterogeneous_gradients():
    """Test with highly heterogeneous gradients (sparse, different scales)."""
    print("\n" + "=" * 80)
    print("TEST 2: Heterogeneous Gradients - Impact on Variance")
    print("=" * 80)

    # Large parameter with sparse gradients (common in deep learning)
    N = 10000
    grad = torch.zeros(N)
    grad[0] = 10.0  # One large gradient
    # Rest are zero

    # CURRENT CODE
    grad_mean = grad.mean().item()  # 10.0 / 10000 = 0.001
    grad_sq_wrong = grad_mean**2  # 0.001^2 = 0.000001

    # PROPOSED FIX
    grad_sq_correct = (grad**2).mean().item()  # 100.0 / 10000 = 0.01

    print(f"\nGradient shape: {grad.shape}")
    print(f"Non-zero elements: 1 out of {N}")
    print(f"Non-zero value: {grad[0].item()}")

    print(f"\nCURRENT CODE:")
    print(f"  E[g] = {grad_mean:.6f}")
    print(f"  (E[g])^2 = {grad_sq_wrong:.10f}")

    print(f"\nPROPOSED FIX:")
    print(f"  E[g] = {grad_mean:.6f}")
    print(f"  E[g^2] = {grad_sq_correct:.6f}")

    ratio = grad_sq_correct / (grad_sq_wrong + 1e-20)
    print(f"\nDIFFERENCE RATIO: E[g^2] / (E[g])^2 = {ratio:.1f}x")
    print(f"UNDERESTIMATION: Current code underestimates by {ratio:.0f}x for N={N}")

    # This matches user's claim: variance is underestimated by factor of N
    theoretical_ratio = N  # Var[mean(X)] = Var[X] / N
    print(f"\nTheoretical ratio (N): {theoretical_ratio}")
    print(f"Observed ratio: {ratio:.1f}")
    print(f"Match: {abs(ratio - theoretical_ratio) / theoretical_ratio < 0.01}")

    print("\n[OK] CONFIRMED: For N=10000 elements, variance is underestimated by ~10000x")


def test_variance_over_time_current_code():
    """Test variance tracking over time with CURRENT code."""
    print("\n" + "=" * 80)
    print("TEST 3: Variance Tracking Over Time - CURRENT CODE")
    print("=" * 80)

    # Create a simple model with one parameter
    param = torch.nn.Parameter(torch.randn(100))

    # Create VGS
    vgs = VarianceGradientScaler(
        parameters=[param],
        enabled=True,
        beta=0.9,
        alpha=0.1,
        warmup_steps=0,  # No warmup for testing
    )

    # Simulate gradients over time with KNOWN properties
    # Case 1: Stable mean, but heterogeneous elements
    print("\nCase 1: Stable mean (0.01), heterogeneous elements")
    for step in range(50):
        # Gradient always has mean 0.01, but elements vary
        param.grad = torch.randn(100) * 10.0  # High variance across elements
        param.grad.data += 0.01  # Shift to have consistent mean

        # Ensure mean is stable
        param.grad.data -= param.grad.data.mean() - 0.01

        vgs.update_statistics()
        vgs._step_count += 1

    variance_current = vgs.get_normalized_variance()
    print(f"Normalized variance (CURRENT CODE): {variance_current:.6f}")
    print("Expected: LOW (mean is stable over time)")

    # Case 2: Unstable mean, homogeneous elements
    print("\nCase 2: Unstable mean (alternates 0.0 and 2.0), homogeneous elements")
    vgs.reset_statistics()
    vgs._parameters = [param]

    for step in range(50):
        # Gradient mean alternates, but all elements equal
        mean_val = 2.0 if step % 2 == 0 else 0.0
        param.grad = torch.ones(100) * mean_val

        vgs.update_statistics()
        vgs._step_count += 1

    variance_current2 = vgs.get_normalized_variance()
    print(f"Normalized variance (CURRENT CODE): {variance_current2:.6f}")
    print("Expected: HIGH (mean is unstable over time)")

    print(f"\n[OK] Current code measures TEMPORAL variance of mean gradient")


def test_variance_over_time_proposed_fix():
    """Test what variance would be with PROPOSED fix."""
    print("\n" + "=" * 80)
    print("TEST 4: Variance Tracking Over Time - PROPOSED FIX (Simulated)")
    print("=" * 80)

    # Manually simulate proposed fix behavior
    beta = 0.9

    # Case 1: Stable mean, heterogeneous elements
    print("\nCase 1: Stable mean (0.01), heterogeneous elements")
    grad_mean_ema = 0.0
    grad_sq_ema = 0.0

    for step in range(50):
        # Gradient with stable mean but high spatial variance
        grad = torch.randn(100) * 10.0 + 0.01
        grad -= grad.mean() - 0.01  # Ensure mean = 0.01

        # PROPOSED FIX
        grad_mean_current = grad.mean().item()
        grad_sq_current = (grad**2).mean().item()  # FIX: mean of squares!

        grad_mean_ema = beta * grad_mean_ema + (1 - beta) * grad_mean_current
        grad_sq_ema = beta * grad_sq_ema + (1 - beta) * grad_sq_current

    variance_proposed = grad_sq_ema - grad_mean_ema**2
    denominator = grad_mean_ema**2 + 1e-8
    normalized_var_proposed = variance_proposed / denominator

    print(f"E[g]: {grad_mean_ema:.6f}")
    print(f"E[g^2]: {grad_sq_ema:.2f}")
    print(f"Var[g] = E[g^2] - E[g]^2: {variance_proposed:.2f}")
    print(f"Normalized variance (PROPOSED FIX): {normalized_var_proposed:.2f}")
    print("Result: HIGH variance detected (due to spatial heterogeneity)")
    print("Interpretation: This would cause scaling even though temporal stability is good!")

    # Case 2: Unstable mean, homogeneous elements
    print("\nCase 2: Unstable mean (alternates 0.0 and 2.0), homogeneous elements")
    grad_mean_ema = 0.0
    grad_sq_ema = 0.0

    for step in range(50):
        mean_val = 2.0 if step % 2 == 0 else 0.0
        grad = torch.ones(100) * mean_val

        grad_mean_current = grad.mean().item()
        grad_sq_current = (grad**2).mean().item()

        grad_mean_ema = beta * grad_mean_ema + (1 - beta) * grad_mean_current
        grad_sq_ema = beta * grad_sq_ema + (1 - beta) * grad_sq_current

    variance_proposed2 = grad_sq_ema - grad_mean_ema**2
    denominator2 = grad_mean_ema**2 + 1e-8
    normalized_var_proposed2 = variance_proposed2 / denominator2

    print(f"E[g]: {grad_mean_ema:.6f}")
    print(f"E[g^2]: {grad_sq_ema:.2f}")
    print(f"Var[g] = E[g^2] - E[g]^2: {variance_proposed2:.2f}")
    print(f"Normalized variance (PROPOSED FIX): {normalized_var_proposed2:.2f}")
    print("Result: HIGH variance detected (temporal instability)")

    print(f"\n[OK] Proposed fix would measure BOTH temporal AND spatial variance")


def test_mathematical_correctness():
    """Test mathematical correctness of variance formula."""
    print("\n" + "=" * 80)
    print("TEST 5: Mathematical Correctness - Var[X] = E[X^2] - E[X]^2")
    print("=" * 80)

    # Known distribution
    torch.manual_seed(42)
    samples = torch.randn(10000)  # N(0, 1)

    # Method 1: Direct variance computation
    var_direct = samples.var().item()

    # Method 2: E[X^2] - E[X]^2 formula (CORRECT for stochastic variance)
    mean_x = samples.mean().item()
    mean_x_sq = (samples**2).mean().item()
    var_formula = mean_x_sq - mean_x**2

    # Method 3: WRONG - using square of mean
    sq_of_mean = mean_x**2

    print(f"\nSamples from N(0, 1), n={len(samples)}")
    print(f"\n1. Direct torch.var(): {var_direct:.6f}")
    print(f"2. E[X^2] - E[X]^2 formula: {var_formula:.6f}")
    print(f"   E[X] = {mean_x:.6f}")
    print(f"   E[X^2] = {mean_x_sq:.6f}")
    print(f"   Var[X] = {var_formula:.6f}")
    print(f"\n3. WRONG (square of mean): E[X]^2 = {sq_of_mean:.6f}")

    assert abs(var_direct - var_formula) < 0.01, "Formula should match direct computation"
    assert abs(var_formula - sq_of_mean) > 0.5, "Formula should NOT equal square of mean"

    print("\n[OK] CONFIRMED: Var[X] = E[X^2] - E[X]^2 requires E[X^2], not E[X]^2")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("VGS VARIANCE COMPUTATION BUG VERIFICATION")
    print("=" * 80)

    test_variance_computation_difference()
    test_variance_computation_with_heterogeneous_gradients()
    test_variance_over_time_current_code()
    test_variance_over_time_proposed_fix()
    test_mathematical_correctness()

    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print("\n1. MATHEMATICAL ISSUE CONFIRMED:")
    print("   Current code: grad_sq_current = grad_mean_current ** 2  # (E[g])^2")
    print("   Should be:    grad_sq_current = (grad ** 2).mean()      # E[g^2]")
    print("\n2. SEMANTIC DIFFERENCE:")
    print("   Current: Measures temporal variance of MEAN gradient (scalar over time)")
    print("   Proposed: Measures mean-of-squares variance (closer to Adam/RMSprop)")
    print("\n3. IMPACT:")
    print("   - Current code is CORRECT for measuring 'is the mean gradient stable?'")
    print("   - Proposed fix measures 'average gradient variance' (different metric)")
    print("   - For sparse/heterogeneous gradients, current code may underreport variance")
    print("\n4. RECOMMENDATION:")
    print("   DEPENDS ON INTENT:")
    print("   - If goal: measure temporal stability of mean -> CURRENT CODE OK")
    print("   - If goal: measure gradient noise (like Adam) -> NEED PROPOSED FIX")
    print("   - For VGS gradient scaling: PROPOSED FIX likely more appropriate")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_all_tests()
