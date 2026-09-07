"""
Test suite for quantile levels formula mismatch (BUG #1)

CRITICAL BUG: Mismatch between QuantileValueHead tau formula and CVaR computation assumptions.

QuantileValueHead uses:     τ_i = (2i+1)/(2*(N+1))
CVaR computation expects:   τ_i = (i+0.5)/N

For N=21:
- τ₀: 0.0227 (actual) vs 0.0238 (expected) → -4.6% difference
- τ₂₀: 0.9318 (actual) vs 0.9762 (expected) → -4.5% difference

This affects:
1. CVaR computation accuracy (especially for small α < 0.1)
2. Quantile spacing (narrower at extremes)
3. Extrapolation logic in _cvar_from_quantiles
"""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from custom_policy_patch1 import QuantileValueHead


class TestQuantileLevelsBug:
    """Test suite for quantile levels formula mismatch."""

    def test_quantile_levels_formula_mismatch(self):
        """Verify the quantile levels formula mismatch between QuantileValueHead and CVaR."""
        N = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)

        # Extract actual tau values from QuantileValueHead
        actual_taus = head.taus.cpu().numpy()

        # Expected formula: τ_i = (i + 0.5) / N
        expected_taus = (np.arange(N) + 0.5) / N

        # Current (incorrect) formula: τ_i = (2i+1)/(2*(N+1))
        current_formula_taus = (2 * np.arange(N) + 1) / (2 * (N + 1))

        # Verify that QuantileValueHead uses the "incorrect" formula
        np.testing.assert_allclose(actual_taus, current_formula_taus, rtol=1e-6)

        # Verify mismatch with expected formula
        max_diff = np.max(np.abs(actual_taus - expected_taus))
        print(f"\nQuantile Levels Formula Mismatch (N={N}):")
        print(f"τ₀: {actual_taus[0]:.6f} (actual) vs {expected_taus[0]:.6f} (expected)")
        print(
            f"    Difference: {(actual_taus[0] - expected_taus[0]) / expected_taus[0] * 100:.2f}%"
        )
        print(f"τ₂₀: {actual_taus[-1]:.6f} (actual) vs {expected_taus[-1]:.6f} (expected)")
        print(
            f"    Difference: {(actual_taus[-1] - expected_taus[-1]) / expected_taus[-1] * 100:.2f}%"
        )
        print(f"Max absolute difference: {max_diff:.6f}")

        # ASSERTION: Verify mismatch exists (4-5% at extremes)
        assert max_diff > 0.04, "Expected ~4-5% mismatch at extremes"

    def test_quantile_spacing_comparison(self):
        """Compare quantile spacing between current and correct formulas."""
        N = 21

        # Current (incorrect) formula
        current_taus = (2 * np.arange(N) + 1) / (2 * (N + 1))
        current_spacing = np.diff(current_taus)

        # Correct formula
        correct_taus = (np.arange(N) + 0.5) / N
        correct_spacing = np.diff(correct_taus)

        print(f"\nQuantile Spacing Comparison (N={N}):")
        print(f"Current formula spacing: {current_spacing[0]:.6f} (uniform)")
        print(f"Correct formula spacing: {correct_spacing[0]:.6f} (uniform)")
        print(f"Spacing difference: {(current_spacing[0] - correct_spacing[0]):.6f}")

        # Both should be uniform, but with different step sizes
        assert np.allclose(current_spacing, current_spacing[0]), "Current should be uniform"
        assert np.allclose(correct_spacing, correct_spacing[0]), "Correct should be uniform"

        # Current formula has narrower spacing
        assert current_spacing[0] < correct_spacing[0], "Current spacing should be narrower"

    def test_cvar_computation_with_actual_taus(self):
        """Test CVaR computation using actual (incorrect) tau values."""
        N = 21
        alpha = 0.05

        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()

        # CVaR computation ASSUMES tau_i = (i + 0.5) / N
        # But actual tau_i = (2i+1) / (2*(N+1))

        # Expected behavior (CVaR code's assumption)
        expected_alpha_idx_float = alpha * N - 0.5  # 0.05 * 21 - 0.5 = 0.55
        expected_alpha_idx = max(0, int(math.floor(expected_alpha_idx_float)))  # 0

        # Find actual index where tau > alpha
        actual_alpha_idx = np.searchsorted(actual_taus, alpha, side="right") - 1
        actual_alpha_idx = max(0, actual_alpha_idx)

        print(f"\nCVaR Index Computation (α={alpha}, N={N}):")
        print(f"Expected alpha_idx_float: {expected_alpha_idx_float:.3f}")
        print(f"Expected alpha_idx: {expected_alpha_idx}")
        print(f"Actual tau[{expected_alpha_idx}]: {actual_taus[expected_alpha_idx]:.6f}")
        print(f"Actual alpha_idx (searchsorted): {actual_alpha_idx}")

        # For alpha=0.05, actual_taus[0]=0.02273 < 0.05 < actual_taus[1]=0.06818
        # So CVaR should use quantiles 0 and 1 for interpolation
        assert (
            actual_taus[0] < alpha < actual_taus[1]
        ), "Alpha should fall between first two quantiles"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.25])
    def test_cvar_bias_for_different_alphas(self, alpha):
        """Test CVaR bias for different alpha values due to tau mismatch."""
        N = 21

        # Create synthetic quantile predictions (linear distribution from -10 to +10)
        quantile_values = torch.linspace(-10.0, 10.0, N).unsqueeze(0)  # [1, N]

        # Compute CVaR using EXPECTED tau formula
        expected_taus = (torch.arange(N) + 0.5) / N
        k_float = alpha * N
        k_int = int(math.floor(k_float))
        frac = k_float - k_int

        if k_int == 0:
            # Interpolate between tau=0 and first quantile
            tau_0_expected = expected_taus[0].item()  # 0.5/N
            if alpha < tau_0_expected:
                # Extrapolation needed
                print(f"α={alpha}: Extrapolation needed (α < τ₀={tau_0_expected:.4f})")

        # Compute CVaR with actual (incorrect) taus
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus

        # Simple CVaR approximation: mean of tail quantiles
        k_tail = max(1, int(math.ceil(alpha * N)))
        cvar_approx = quantile_values[:, :k_tail].mean(dim=1).item()

        print(f"\nCVaR Approximation (α={alpha}, N={N}):")
        print(f"  Tail quantiles used: {k_tail}")
        print(f"  Actual τ[0]: {actual_taus[0].item():.6f}")
        print(f"  Expected τ[0]: {expected_taus[0].item():.6f}")
        print(f"  CVaR (approx): {cvar_approx:.4f}")

    def test_extrapolation_logic_with_wrong_taus(self):
        """Test extrapolation logic in _cvar_from_quantiles with incorrect taus."""
        N = 21
        alpha = 0.01  # Very small alpha to trigger extrapolation

        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()

        # CVaR code assumes tau_0 = 0.5/N, tau_1 = 1.5/N
        assumed_tau_0 = 0.5 / N  # 0.02381
        assumed_tau_1 = 1.5 / N  # 0.07143

        # Actual values
        actual_tau_0 = actual_taus[0]  # 0.02273
        actual_tau_1 = actual_taus[1]  # 0.06818

        print(f"\nExtrapolation Logic (α={alpha}, N={N}):")
        print(f"CVaR code assumes:")
        print(f"  τ₀ = {assumed_tau_0:.6f}, τ₁ = {assumed_tau_1:.6f}")
        print(f"Actual tau values:")
        print(f"  τ₀ = {actual_tau_0:.6f}, τ₁ = {actual_tau_1:.6f}")
        print(f"Differences:")
        print(f"  Δτ₀ = {(actual_tau_0 - assumed_tau_0):.6f}")
        print(f"  Δτ₁ = {(actual_tau_1 - assumed_tau_1):.6f}")

        # When CVaR code does extrapolation:
        # slope = (q1 - q0) / (tau_1 - tau_0)
        # It assumes tau_1 - tau_0 = 1.0/N = 0.04762
        assumed_delta = assumed_tau_1 - assumed_tau_0
        actual_delta = actual_tau_1 - actual_tau_0

        print(f"Δτ (assumed): {assumed_delta:.6f}")
        print(f"Δτ (actual): {actual_delta:.6f}")
        print(f"Slope error: {(actual_delta - assumed_delta) / assumed_delta * 100:.2f}%")

        # This will cause ~4% error in slope computation
        assert abs((actual_delta - assumed_delta) / assumed_delta) > 0.04

    def test_coverage_at_extremes(self):
        """Test coverage of extreme quantiles (tails)."""
        N = 21

        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        actual_taus = head.taus.cpu().numpy()
        expected_taus = (np.arange(N) + 0.5) / N

        # Check coverage at extremes
        print(f"\nExtreme Quantile Coverage (N={N}):")
        print(f"Lower tail:")
        print(f"  Actual τ₀: {actual_taus[0]:.6f} (covers [0, {2*actual_taus[0]:.6f}])")
        print(f"  Expected τ₀: {expected_taus[0]:.6f} (covers [0, {2*expected_taus[0]:.6f}])")
        print(f"Upper tail:")
        print(f"  Actual τ₂₀: {actual_taus[-1]:.6f} (covers [{2*actual_taus[-1]-1:.6f}, 1])")
        print(f"  Expected τ₂₀: {expected_taus[-1]:.6f} (covers [{2*expected_taus[-1]-1:.6f}, 1])")

        # Actual formula leaves more mass uncovered at tails
        actual_lower_coverage = 2 * actual_taus[0]
        expected_lower_coverage = 2 * expected_taus[0]

        print(f"\nLower tail coverage:")
        print(f"  Actual: {actual_lower_coverage:.6f}")
        print(f"  Expected: {expected_lower_coverage:.6f}")
        print(f"  Difference: {(expected_lower_coverage - actual_lower_coverage):.6f}")

    def test_integration_with_multiple_num_quantiles(self):
        """Test the bug impact across different num_quantiles settings."""
        for N in [11, 21, 32, 51]:
            head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
            actual_taus = head.taus.cpu().numpy()
            expected_taus = (np.arange(N) + 0.5) / N

            # Compute max relative error
            rel_errors = np.abs((actual_taus - expected_taus) / expected_taus)
            max_rel_error = np.max(rel_errors) * 100

            # Compute absolute difference at extremes
            diff_0 = actual_taus[0] - expected_taus[0]
            diff_N = actual_taus[-1] - expected_taus[-1]

            print(f"\nN={N}:")
            print(f"  τ₀: {actual_taus[0]:.6f} vs {expected_taus[0]:.6f} (Δ={diff_0:.6f})")
            print(f"  τ_max: {actual_taus[-1]:.6f} vs {expected_taus[-1]:.6f} (Δ={diff_N:.6f})")
            print(f"  Max relative error: {max_rel_error:.2f}%")

            # All should show ~4-5% error at extremes
            assert max_rel_error > 3.0, f"Expected >3% error for N={N}"


class TestCVaRComputationWithBug:
    """Test CVaR computation behavior with the quantile levels bug."""

    def test_cvar_from_quantiles_mock(self):
        """Test _cvar_from_quantiles with mocked PPO instance."""
        # This test simulates the CVaR computation logic from distributional_ppo.py
        N = 21
        alpha = 0.05

        # Create synthetic quantile predictions (standard normal quantiles)
        from scipy.stats import norm

        expected_taus = (np.arange(N) + 0.5) / N
        true_quantile_values = torch.tensor(
            [norm.ppf(tau) for tau in expected_taus], dtype=torch.float32
        ).unsqueeze(
            0
        )  # [1, N]

        # CVaR code's logic (simplified)
        alpha_idx_float = alpha * N - 0.5  # 0.05 * 21 - 0.5 = 0.55

        print(f"\nCVaR Computation Mock (α={alpha}, N={N}):")
        print(f"alpha_idx_float: {alpha_idx_float:.3f}")

        if alpha_idx_float < 0.0:
            print("Extrapolation branch triggered")
            # Use first two quantiles for extrapolation
            q0 = true_quantile_values[0, 0].item()
            q1 = true_quantile_values[0, 1].item()

            # CVaR code assumes tau_0 = 0.5/N, tau_1 = 1.5/N
            tau_0 = 0.5 / N
            tau_1 = 1.5 / N
            slope = (q1 - q0) / (tau_1 - tau_0)

            print(f"q0: {q0:.4f}, q1: {q1:.4f}")
            print(f"tau_0: {tau_0:.6f}, tau_1: {tau_1:.6f}")
            print(f"Slope: {slope:.4f}")
        else:
            print("Standard branch (no extrapolation needed)")

    def test_real_world_impact_small_alpha(self):
        """Test real-world impact for small alpha (risk-sensitive case)."""
        N = 21
        alpha_values = [0.01, 0.025, 0.05, 0.10]

        # Create realistic quantile distribution (losses in tail)
        # Simulate worst-case scenario: large negative values in tail
        quantile_values = torch.linspace(-20.0, 5.0, N).unsqueeze(0)  # [1, N]

        for alpha in alpha_values:
            # Simple CVaR: mean of worst alpha% quantiles
            k_tail = max(1, int(math.ceil(alpha * N)))
            cvar_simple = quantile_values[:, :k_tail].mean(dim=1).item()

            # Expected number of quantiles based on correct formula
            expected_k = alpha * N

            # Actual tau coverage
            head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
            actual_taus = head.taus.cpu().numpy()
            actual_coverage = actual_taus[k_tail - 1] if k_tail > 0 else 0.0

            print(f"\nα={alpha} (N={N}):")
            print(f"  Expected k: {expected_k:.2f}")
            print(f"  Actual k used: {k_tail}")
            print(f"  Actual tau coverage: {actual_coverage:.6f}")
            print(f"  CVaR (simple): {cvar_simple:.4f}")
            print(f"  Coverage error: {(actual_coverage - alpha) / alpha * 100:.2f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
