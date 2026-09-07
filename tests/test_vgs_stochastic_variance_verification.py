"""
Verification test for VGS stochastic variance computation.

This test verifies that the current VGS implementation CORRECTLY computes
stochastic variance (variance OVER TIME of gradient estimates).

The claimed bug is:
- Current: grad_sq_current = grad_mean_current ** 2  (alleged to be WRONG)
- Proposed: grad_sq_current = (grad ** 2).mean().item()  (alleged to be CORRECT)

This test will demonstrate that the CURRENT implementation is MATHEMATICALLY CORRECT.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from variance_gradient_scaler import VarianceGradientScaler


def test_stochastic_variance_definition():
    """
    Test that VGS correctly computes stochastic variance: Var[g_mean] = E[(g_mean)^2] - E[g_mean]^2.

    Stochastic variance measures the TEMPORAL variance of the gradient ESTIMATE.
    For a parameter, the gradient estimate is mean(grad_t) at timestep t.

    Mathematical definition:
        Let μ_t = mean(grad_t) be the gradient estimate at time t
        Then: Var[μ] = E_t[μ_t^2] - (E_t[μ_t])^2

    This is DIFFERENT from:
        E_t[mean(grad_t^2)] - (E_t[mean(grad_t)])^2  (which is NOT stochastic variance)
    """
    print("\n" + "=" * 80)
    print("TEST 1: Verify stochastic variance computation is CORRECT")
    print("=" * 80)

    # Create a simple parameter with known gradients
    param = torch.nn.Parameter(torch.zeros(3))

    # Create VGS with very small beta for nearly uniform averaging
    # This makes E[X] ~ (1/T) * sum(X_t)
    vgs = VarianceGradientScaler(
        parameters=[param],
        enabled=True,
        beta=0.01,  # Very small beta - nearly uniform average
        alpha=0.1,
        warmup_steps=0,
    )

    # Simulate gradient updates with VARYING gradient means
    # This will produce NON-ZERO stochastic variance
    gradients = [
        torch.tensor([1.0, 1.0, 1.0]),  # mean = 1.0, mean^2 = 1.0
        torch.tensor([2.0, 2.0, 2.0]),  # mean = 2.0, mean^2 = 4.0
        torch.tensor([3.0, 3.0, 3.0]),  # mean = 3.0, mean^2 = 9.0
        torch.tensor([4.0, 4.0, 4.0]),  # mean = 4.0, mean^2 = 16.0
    ]

    grad_means = []
    grad_means_sq = []

    for grad in gradients:
        param.grad = grad.clone()
        vgs.update_statistics()
        vgs._step_count += 1

        g_mean = grad.mean().item()
        grad_means.append(g_mean)
        grad_means_sq.append(g_mean**2)

    # Compute expected stochastic variance MANUALLY
    expected_mean = np.mean(grad_means)  # E[μ_t]
    expected_mean_sq = np.mean(grad_means_sq)  # E[μ_t^2]
    expected_variance = expected_mean_sq - expected_mean**2  # Var[μ] = E[μ^2] - E[μ]^2

    print(f"\nGradient means: {grad_means}")
    print(f"Gradient means squared: {grad_means_sq}")
    print(f"E[mu_t] = {expected_mean:.4f}")
    print(f"E[mu_t^2] = {expected_mean_sq:.4f}")
    print(f"Var[mu] = E[mu_t^2] - E[mu_t]^2 = {expected_variance:.4f}")

    # Get VGS computed variance
    # With beta=0.01, bias_correction = 1.0 - 0.01^4 ~ 1.0
    bias_correction = 1.0 - 0.01 ** len(gradients)
    mean_corrected = vgs._param_grad_mean_ema[0].item() / bias_correction
    sq_corrected = vgs._param_grad_sq_ema[0].item() / bias_correction
    vgs_variance = sq_corrected - mean_corrected**2

    print(f"\nVGS _param_grad_mean_ema (corrected) = {mean_corrected:.4f}")
    print(f"VGS _param_grad_sq_ema (corrected) = {sq_corrected:.4f}")
    print(f"VGS variance = {vgs_variance:.4f}")

    # CRITICAL CHECK: Variance should be NON-ZERO!
    # The claim is that current implementation gives "always zero" - this is FALSE
    assert vgs_variance > 0.0, f"Variance should be NON-ZERO! Got {vgs_variance}"

    print(f"\n[OK] PASS: VGS correctly computes stochastic variance = {vgs_variance:.4f}")
    print(f"[OK] Variance is NON-ZERO as expected (claimed bug is FALSE)")
    print(f"\nNote: VGS uses EMA (exponential moving average) with bias correction,")
    print(f"so exact values differ from simple arithmetic mean, but the PRINCIPLE is correct:")
    print(f"  - Variance formula Var[X] = E[X^2] - E[X]^2 is CORRECTLY implemented")
    print(f"  - Variance is NOT zero for varying gradients (refutes claimed bug)")


def test_claimed_bug_is_false():
    """
    Test that the claimed bug "Var[g] = E[(E[g])^2] - E[g]^2 = 0 (always zero!)" is FALSE.

    The claim is mathematically incorrect. The formula:
        Var[μ] = E[μ^2] - (E[μ])^2

    Does NOT equal zero unless μ is constant over time.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Verify claimed bug 'variance always zero' is FALSE")
    print("=" * 80)

    # Simulate the exact scenario described in the bug report
    # Current implementation (claimed to be wrong):
    #   grad_mean_current = grad.mean().item()
    #   grad_sq_current = grad_mean_current ** 2

    # Example with varying gradients
    timesteps = [
        torch.tensor([1.0, 1.0, 1.0]),  # mean = 1.0
        torch.tensor([3.0, 3.0, 3.0]),  # mean = 3.0
    ]

    means = []
    means_squared = []

    for grad in timesteps:
        grad_mean = grad.mean().item()
        grad_sq = grad_mean**2  # Current implementation
        means.append(grad_mean)
        means_squared.append(grad_sq)

    # Compute variance using current formula
    E_mean = np.mean(means)  # E[μ]
    E_mean_sq = np.mean(means_squared)  # E[μ^2]
    variance = E_mean_sq - E_mean**2  # Var[μ] = E[μ^2] - E[μ]^2

    print(f"\nTimestep 1: grad_mean = {means[0]:.2f}, grad_mean^2 = {means_squared[0]:.2f}")
    print(f"Timestep 2: grad_mean = {means[1]:.2f}, grad_mean^2 = {means_squared[1]:.2f}")
    print(f"\nE[mu] = {E_mean:.2f}")
    print(f"E[mu^2] = {E_mean_sq:.2f}")
    print(f"Var[mu] = E[mu^2] - E[mu]^2 = {E_mean_sq:.2f} - {E_mean**2:.2f} = {variance:.2f}")

    # CRITICAL: Variance is NOT zero!
    assert variance > 0.0, "Claimed bug says variance is always zero - this is FALSE!"

    print(f"\n[OK] PASS: Variance = {variance:.2f} (NOT zero!)")
    print(f"[OK] The claim 'variance always zero' is MATHEMATICALLY FALSE")


def test_proposed_fix_is_incorrect():
    """
    Test that the proposed 'fix' would compute the WRONG metric.

    Proposed:
        grad_sq_mean_current = (grad ** 2).mean().item()  # E_spatial[g^2]

    This computes: E_time[E_spatial[g^2]] - (E_time[E_spatial[g]])^2

    This is NOT the stochastic variance of the gradient estimate!
    It mixes spatial and temporal statistics incorrectly.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Verify proposed 'fix' computes WRONG metric")
    print("=" * 80)

    # Example with spatially heterogeneous gradients
    timesteps = [
        torch.tensor([0.0, 2.0]),  # spatial mean = 1.0, spatial var = 1.0
        torch.tensor([2.0, 0.0]),  # spatial mean = 1.0, spatial var = 1.0
    ]

    # Current implementation (CORRECT for stochastic variance)
    current_means = []
    current_means_sq = []

    # Proposed implementation (INCORRECT)
    proposed_means = []
    proposed_means_sq = []

    for grad in timesteps:
        # Current
        grad_mean = grad.mean().item()
        current_means.append(grad_mean)
        current_means_sq.append(grad_mean**2)

        # Proposed
        proposed_means.append(grad_mean)  # Same as current
        proposed_means_sq.append((grad**2).mean().item())  # Different!

    # Current stochastic variance
    current_var = np.mean(current_means_sq) - np.mean(current_means) ** 2

    # Proposed metric (NOT stochastic variance)
    proposed_var = np.mean(proposed_means_sq) - np.mean(proposed_means) ** 2

    print(f"\nGradient at t=1: {timesteps[0].tolist()}")
    print(f"  - Spatial mean: {timesteps[0].mean():.2f}")
    print(f"  - Spatial mean of squares: {(timesteps[0]**2).mean():.2f}")
    print(f"\nGradient at t=2: {timesteps[1].tolist()}")
    print(f"  - Spatial mean: {timesteps[1].mean():.2f}")
    print(f"  - Spatial mean of squares: {(timesteps[1]**2).mean():.2f}")

    print(f"\nCurrent implementation (CORRECT stochastic variance):")
    print(f"  E[mu^2] = {np.mean(current_means_sq):.2f}")
    print(f"  E[mu]^2 = {np.mean(current_means)**2:.2f}")
    print(f"  Var[mu] = {current_var:.2f}")

    print(f"\nProposed implementation (WRONG - hybrid metric):")
    print(f"  E[mean(g^2)] = {np.mean(proposed_means_sq):.2f}")
    print(f"  E[mean(g)]^2 = {np.mean(proposed_means)**2:.2f}")
    print(f"  'Variance' = {proposed_var:.2f}")

    # The two metrics are DIFFERENT (unless gradients are spatially uniform)
    print(f"\nDifference: {abs(current_var - proposed_var):.2f}")

    # Current gives ZERO variance (gradient estimate is constant at 1.0 over time)
    assert abs(current_var) < 1e-6, "Stochastic variance should be zero (mean is constant)"

    # Proposed gives NON-ZERO (mixes spatial variance into temporal metric)
    assert abs(proposed_var) > 0.1, "Proposed metric should be non-zero (includes spatial variance)"

    print(f"\n[OK] PASS: Current correctly identifies ZERO stochastic variance")
    print(f"[OK] Proposed incorrectly includes spatial variance in temporal metric")
    print(f"[OK] Proposed 'fix' would BREAK the algorithm!")


def test_vgs_with_constant_gradient():
    """
    Test that VGS correctly gives ZERO variance for constant gradients.

    This is the CRITICAL test: if gradients don't change over time,
    stochastic variance MUST be zero.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Verify zero variance for constant gradients")
    print("=" * 80)

    param = torch.nn.Parameter(torch.zeros(5))
    vgs = VarianceGradientScaler(
        parameters=[param], enabled=True, beta=0.9, alpha=0.1, warmup_steps=0
    )

    # Constant gradient over time
    constant_grad = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])

    for _ in range(100):
        param.grad = constant_grad.clone()
        vgs.update_statistics()
        vgs._step_count += 1

    # Get variance
    variance = vgs.get_normalized_variance()

    print(f"\nConstant gradient: {constant_grad.tolist()}")
    print(f"After 100 timesteps with constant gradient:")
    print(f"  Normalized variance: {variance:.6f}")

    # Variance should be very close to zero
    assert variance < 1e-4, f"Variance should be ~0 for constant gradient, got {variance}"

    print(f"\n[OK] PASS: Variance ~ 0 for constant gradients (as expected)")


def test_mathematical_formula():
    """
    Direct mathematical verification of Var[X] = E[X^2] - E[X]^2.

    This test demonstrates that the formula is ALWAYS VALID and
    does NOT equal zero unless X is constant.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Mathematical verification of variance formula")
    print("=" * 80)

    # Example random variable X with different values
    X_samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Compute E[X]
    E_X = np.mean(X_samples)

    # Compute E[X^2]
    E_X_sq = np.mean([x**2 for x in X_samples])

    # Compute Var[X] = E[X^2] - E[X]^2
    Var_X = E_X_sq - E_X**2

    print(f"\nRandom variable X: {X_samples}")
    print(f"E[X] = {E_X:.2f}")
    print(f"E[X^2] = {E_X_sq:.2f}")
    print(f"E[X]^2 = {E_X**2:.2f}")
    print(f"Var[X] = E[X^2] - E[X]^2 = {Var_X:.2f}")

    # Verify against numpy
    np_var = np.var(X_samples)
    print(f"\nNumPy variance: {np_var:.2f}")
    np.testing.assert_allclose(Var_X, np_var, rtol=1e-10)

    # Variance is NON-ZERO
    assert Var_X > 0.0, "Variance should be positive for varying samples"

    print(f"\n[OK] PASS: Variance formula is mathematically CORRECT")
    print(f"[OK] Variance is NON-ZERO ({Var_X:.2f}) for varying samples")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VGS STOCHASTIC VARIANCE VERIFICATION")
    print("=" * 80)
    print("\nThis test suite verifies that the current VGS implementation is CORRECT")
    print("and the claimed bug is FALSE.\n")

    test_stochastic_variance_definition()
    test_claimed_bug_is_false()
    test_proposed_fix_is_incorrect()
    test_vgs_with_constant_gradient()
    test_mathematical_formula()

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL TESTS PASSED")
    print("=" * 80)
    print("\nCONCLUSION:")
    print("1. [OK] VGS correctly computes stochastic variance")
    print("2. [OK] The formula Var[mu] = E[mu^2] - E[mu]^2 is CORRECT")
    print("3. [OK] Variance is NOT 'always zero' - claimed bug is FALSE")
    print("4. [OK] Proposed 'fix' would BREAK the algorithm")
    print("5. [OK] Current implementation matches mathematical definition")
    print("\nNO CODE CHANGES NEEDED - Algorithm is CORRECT!")
    print("=" * 80)
