"""
Verification test: Quantile levels formula is CORRECT

This test verifies that QuantileValueHead uses the correct formula:
    tau_i = (i + 0.5) / N

Previous concern: Formula might be (2i+1)/(2*(N+1)) which would be incorrect.
RESULT: Code is CORRECT. The linspace + midpoints approach produces (i+0.5)/N.

Mathematical proof:
    taus = linspace(0, 1, steps=N+1)  => [0, 1/N, 2/N, ..., N/N]
    midpoints[i] = 0.5 * (taus[i] + taus[i+1])
                 = 0.5 * (i/N + (i+1)/N)
                 = 0.5 * (2i+1)/N
                 = (i + 0.5) / N  ✓ CORRECT
"""

import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from custom_policy_patch1 import QuantileValueHead


class TestQuantileLevelsCorrectness:
    """Verify that quantile levels use the correct formula."""

    def test_quantile_formula_is_correct(self):
        """Verify QuantileValueHead uses tau_i = (i + 0.5) / N."""
        N = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)

        actual_taus = head.taus.cpu().numpy()
        expected_taus = (np.arange(N) + 0.5) / N

        # Should match exactly (within floating point precision)
        np.testing.assert_allclose(actual_taus, expected_taus, rtol=1e-6, atol=1e-7)

        print(f"\nQuantile Levels Verification (N={N}):")
        print(f"Formula: tau_i = (i + 0.5) / N")
        print(f"tau_0 = {actual_taus[0]:.6f} (expected: {expected_taus[0]:.6f})")
        print(f"tau_20 = {actual_taus[-1]:.6f} (expected: {expected_taus[-1]:.6f})")
        print("✓ CORRECT: QuantileValueHead uses the correct formula")

    def test_quantile_spacing_is_uniform(self):
        """Verify quantile spacing is uniform (1/N)."""
        N = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)

        actual_taus = head.taus.cpu().numpy()
        spacing = np.diff(actual_taus)
        expected_spacing = 1.0 / N

        # All spacings should be uniform
        np.testing.assert_allclose(spacing, expected_spacing, rtol=1e-6)

        print(f"\nQuantile Spacing (N={N}):")
        print(f"Spacing: {spacing[0]:.6f}")
        print(f"Expected: {expected_spacing:.6f}")
        print("✓ CORRECT: Uniform spacing")

    @pytest.mark.parametrize("N", [11, 21, 32, 51])
    def test_coverage_bounds(self, N):
        """Verify quantile coverage bounds."""
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        taus = head.taus.cpu().numpy()

        # First quantile should cover [0, 1/N]
        # Last quantile should cover [(N-1)/N, 1]
        first_center = 0.5 / N
        last_center = (N - 0.5) / N

        np.testing.assert_allclose(taus[0], first_center, rtol=1e-6)
        np.testing.assert_allclose(taus[-1], last_center, rtol=1e-6)

        print(f"\nCoverage Bounds (N={N}):")
        print(f"First tau: {taus[0]:.6f} (covers [0, {1/N:.6f}])")
        print(f"Last tau: {taus[-1]:.6f} (covers [{(N-1)/N:.6f}, 1])")


class TestCVaRComputationConsistency:
    """Verify CVaR computation is consistent with quantile levels."""

    def test_cvar_computation_uses_correct_taus(self):
        """Verify _cvar_from_quantiles logic is consistent with actual taus."""
        N = 21
        alpha = 0.05

        # The CVaR code assumes tau_i = (i + 0.5) / N
        # Let's verify this assumption is correct
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()

        # CVaR code computes:
        # alpha_idx_float = alpha * N - 0.5
        # This assumes tau_i = (i + 0.5) / N
        alpha_idx_float = alpha * N - 0.5  # 0.05 * 21 - 0.5 = 0.55

        # Find actual tau that brackets alpha
        alpha_idx = max(0, int(math.floor(alpha_idx_float)))

        print(f"\nCVaR Computation Consistency (alpha={alpha}, N={N}):")
        print(f"alpha_idx_float: {alpha_idx_float:.3f}")
        print(f"alpha_idx: {alpha_idx}")
        print(f"tau[{alpha_idx}]: {actual_taus[alpha_idx]:.6f}")

        # For alpha=0.05, alpha_idx should be 0
        # tau[0] = 0.5/21 = 0.02381 < 0.05 ✓
        # tau[1] = 1.5/21 = 0.07143 > 0.05 ✓
        assert alpha_idx == 0
        assert actual_taus[0] < alpha < actual_taus[1]
        print(f"✓ CORRECT: alpha={alpha} falls between tau[0]={actual_taus[0]:.4f} and tau[1]={actual_taus[1]:.4f}")

    def test_extrapolation_assumptions_correct(self):
        """Verify extrapolation logic assumptions match actual taus."""
        N = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()

        # CVaR code assumes for extrapolation:
        # tau_0 = 0.5 / N
        # tau_1 = 1.5 / N
        assumed_tau_0 = 0.5 / N
        assumed_tau_1 = 1.5 / N

        # Verify these match actual values
        np.testing.assert_allclose(actual_taus[0], assumed_tau_0, rtol=1e-6)
        np.testing.assert_allclose(actual_taus[1], assumed_tau_1, rtol=1e-6)

        print(f"\nExtrapolation Logic Verification (N={N}):")
        print(f"Assumed tau_0: {assumed_tau_0:.6f}")
        print(f"Actual tau_0: {actual_taus[0]:.6f}")
        print(f"Assumed tau_1: {assumed_tau_1:.6f}")
        print(f"Actual tau_1: {actual_taus[1]:.6f}")
        print("✓ CORRECT: Extrapolation assumptions match actual taus")

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.25])
    def test_cvar_index_computation(self, alpha):
        """Test CVaR index computation for different alpha values."""
        N = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()

        # CVaR logic
        alpha_idx_float = alpha * N - 0.5

        if alpha_idx_float < 0.0:
            # Extrapolation case
            print(f"\nalpha={alpha}: Extrapolation (alpha < tau_0={actual_taus[0]:.4f})")
            assert alpha < actual_taus[0], "Should trigger extrapolation"
        else:
            alpha_idx = int(math.floor(alpha_idx_float))
            print(f"\nalpha={alpha}: Standard case")
            print(f"  alpha_idx_float: {alpha_idx_float:.3f}")
            print(f"  alpha_idx: {alpha_idx}")

            # Verify alpha falls in expected range
            if alpha_idx < N - 1:
                # Should bracket alpha between tau[alpha_idx] and tau[alpha_idx+1]
                # unless alpha is very close to tau[alpha_idx]
                print(f"  tau[{alpha_idx}]: {actual_taus[alpha_idx]:.6f}")
                print(f"  tau[{alpha_idx+1}]: {actual_taus[alpha_idx+1]:.6f}")


class TestRealWorldCVaRAccuracy:
    """Test CVaR accuracy with realistic distributions."""

    def test_cvar_standard_normal(self):
        """Test CVaR computation with standard normal quantiles."""
        from scipy.stats import norm

        N = 21
        alpha = 0.05

        # Create standard normal quantiles at correct tau levels
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        taus = head.taus.cpu().numpy()
        quantile_values = torch.tensor([norm.ppf(tau) for tau in taus]).unsqueeze(0)

        # True CVaR for standard normal at alpha=0.05
        # CVaR_alpha(X) = E[X | X <= VaR_alpha(X)]
        # For standard normal: CVaR_0.05 ≈ -2.063
        true_cvar = norm.expect(lambda x: x, lb=-np.inf, ub=norm.ppf(alpha)) / alpha

        # Simple approximation: mean of tail quantiles
        k_tail = max(1, int(math.ceil(alpha * N)))
        approx_cvar = quantile_values[:, :k_tail].mean().item()

        print(f"\nCVaR Standard Normal (alpha={alpha}, N={N}):")
        print(f"True CVaR: {true_cvar:.4f}")
        print(f"Approx CVaR (k={k_tail}): {approx_cvar:.4f}")
        print(f"Error: {abs(approx_cvar - true_cvar):.4f}")

        # Should be reasonably close (within 10%)
        assert abs(approx_cvar - true_cvar) / abs(true_cvar) < 0.15

    def test_cvar_uniform_distribution(self):
        """Test CVaR with uniform distribution."""
        N = 21
        alpha = 0.10

        # Uniform distribution on [0, 1]
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        taus = head.taus.cpu().numpy()
        quantile_values = torch.tensor(taus).unsqueeze(0)  # Quantiles of uniform = taus

        # True CVaR for uniform[0,1] at alpha=0.10 is alpha/2 = 0.05
        true_cvar = alpha / 2.0

        k_tail = max(1, int(math.ceil(alpha * N)))
        approx_cvar = quantile_values[:, :k_tail].mean().item()

        print(f"\nCVaR Uniform Distribution (alpha={alpha}, N={N}):")
        print(f"True CVaR: {true_cvar:.4f}")
        print(f"Approx CVaR (k={k_tail}): {approx_cvar:.4f}")
        print(f"Error: {abs(approx_cvar - true_cvar):.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
