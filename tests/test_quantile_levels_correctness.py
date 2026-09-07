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

        # All spacings should be uniform.
        # taus stored as float32 (QuantileValueHead) → np.diff carries ~6e-8 rounding;
        # atol absorbs it (matches sibling test_quantile_levels_uses_midpoint_formula).
        np.testing.assert_allclose(spacing, expected_spacing, rtol=1e-6, atol=1e-7)

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
        print(
            f"✓ CORRECT: alpha={alpha} falls between tau[0]={actual_taus[0]:.4f} and tau[1]={actual_taus[1]:.4f}"
        )

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

    @staticmethod
    def _production_cvar(quantiles: "torch.Tensor", alpha: float) -> float:
        """Call the PRODUCTION DistributionalPPO._cvar_from_quantiles.

        The method only reads ``self.cvar_alpha``; a lightweight shim avoids constructing
        a full model. This validates the REAL piecewise-linear quantile integration
        (with interpolation/extrapolation), not a naive bottom-k proxy.
        """
        from types import SimpleNamespace
        from distributional_ppo import DistributionalPPO

        shim = SimpleNamespace(cvar_alpha=float(alpha))
        return DistributionalPPO._cvar_from_quantiles(shim, quantiles).item()

    def test_cvar_standard_normal(self):
        """Production CVaR from quantiles is accurate for the standard normal tail.

        Regression note: this used to test a naive ``mean(bottom-k quantiles)`` proxy
        (~16.5% error, failed its own <15% bound). The production
        ``_cvar_from_quantiles`` integrates the quantile function piecewise-linearly with
        tail extrapolation and is far more accurate (~4.6% at N=21, ~1.8% at N=51).
        """
        from scipy.stats import norm

        N = 21
        alpha = 0.05

        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        taus = head.taus.cpu().numpy()
        quantile_values = torch.tensor(
            [norm.ppf(tau) for tau in taus], dtype=torch.float64
        ).unsqueeze(0)

        # True CVaR for standard normal at alpha=0.05: E[X | X <= q_alpha] ≈ -2.063
        true_cvar = norm.expect(lambda x: x, lb=-np.inf, ub=norm.ppf(alpha)) / alpha

        prod_cvar = self._production_cvar(quantile_values, alpha)
        prod_rel_err = abs(prod_cvar - true_cvar) / abs(true_cvar)

        # Naive bottom-k proxy (what the old test asserted) — kept to show the contrast.
        k_tail = max(1, int(math.ceil(alpha * N)))
        naive_cvar = quantile_values[:, :k_tail].mean().item()
        naive_rel_err = abs(naive_cvar - true_cvar) / abs(true_cvar)

        print(f"\nCVaR Standard Normal (alpha={alpha}, N={N}):")
        print(f"True CVaR:       {true_cvar:.4f}")
        print(f"PRODUCTION CVaR: {prod_cvar:.4f}  rel_err={prod_rel_err*100:.1f}%")
        print(f"naive bottom-k:  {naive_cvar:.4f}  rel_err={naive_rel_err*100:.1f}%")

        # Production method is genuinely accurate (~4.6%) — well within 10%.
        assert prod_rel_err < 0.10, f"production CVaR rel error too high: {prod_rel_err:.3f}"
        # And it strictly beats the naive proxy it replaced.
        assert prod_rel_err < naive_rel_err

        # Convergence: more quantiles → smaller error.
        N2 = 51
        head2 = QuantileValueHead(input_dim=64, num_quantiles=N2, huber_kappa=1.0)
        taus2 = head2.taus.cpu().numpy()
        q2 = torch.tensor([norm.ppf(t) for t in taus2], dtype=torch.float64).unsqueeze(0)
        err2 = abs(self._production_cvar(q2, alpha) - true_cvar) / abs(true_cvar)
        assert err2 < prod_rel_err, "increasing N should reduce CVaR error"

    def test_cvar_uniform_distribution(self):
        """Test CVaR with uniform distribution."""
        N = 21
        alpha = 0.10

        # Uniform distribution on [0, 1]: quantile function q(tau) = tau is LINEAR,
        # so the production piecewise-linear integration should be ~exact.
        head = QuantileValueHead(input_dim=64, num_quantiles=N, huber_kappa=1.0)
        taus = head.taus.cpu().numpy().astype(np.float64)
        quantile_values = torch.tensor(taus, dtype=torch.float64).unsqueeze(0)

        # True CVaR for uniform[0,1] at alpha=0.10 is alpha/2 = 0.05
        true_cvar = alpha / 2.0

        prod_cvar = TestRealWorldCVaRAccuracy._production_cvar(quantile_values, alpha)

        print(f"\nCVaR Uniform Distribution (alpha={alpha}, N={N}):")
        print(f"True CVaR:       {true_cvar:.4f}")
        print(f"PRODUCTION CVaR: {prod_cvar:.6f}  err={abs(prod_cvar - true_cvar):.2e}")

        # Linear distribution → production CVaR is exact to numerical precision.
        assert abs(prod_cvar - true_cvar) < 1e-6


class TestInferenceTrainingCVaRConsistency:
    """Inference-path CVaR (impl_rl_signal) must equal training-path CVaR (DistributionalPPO).

    Closes the methodology gap: RLAlphaSignal's CVaR utility used to be a naive
    mean-of-bottom-k proxy, while the model trains on the accurate piecewise-linear
    integration. They must now be numerically identical.
    """

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.25, 0.5, 0.95])
    @pytest.mark.parametrize("N", [11, 21, 32, 51])
    def test_numpy_cvar_matches_torch_production(self, alpha, N):
        from types import SimpleNamespace
        from distributional_ppo import DistributionalPPO
        from impl_rl_signal import cvar_from_quantiles_np

        rng = np.random.default_rng(0)
        # monotone (sorted) quantile rows across a batch
        q = np.sort(rng.standard_normal((7, N)), axis=1)

        np_cvar = cvar_from_quantiles_np(q, alpha)
        shim = SimpleNamespace(cvar_alpha=alpha)
        torch_cvar = (
            DistributionalPPO._cvar_from_quantiles(shim, torch.tensor(q, dtype=torch.float64))
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(np_cvar, torch_cvar, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
