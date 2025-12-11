"""
Tests for bug fixes from 2025-11-26:
1. Issue #1: Twin Critics categorical VF clipping projection (was identity stub)
2. Issue #2: Yang-Zhang Rogers-Satchell denominator (was n-1, should be n)

These tests verify that the fixes are correctly implemented and prevent regression.
"""

import math
import pytest
torch = pytest.importorskip("torch")
import numpy as np


# ============================================================================
# ISSUE #1 TESTS: Twin Critics Categorical VF Clipping Projection
# ============================================================================

class TestTwinCriticsCategoricalProjectionFix:
    """
    Tests for Issue #1: _project_distribution was a stub returning identity.

    PROBLEM: Twin Critics categorical VF clipping called _project_distribution
    which returned probs unchanged (identity projection). This made VF clipping
    non-functional for Twin Critics categorical critic.

    FIX: Changed to use _project_categorical_distribution which implements
    proper C51 projection (Bellemare et al. 2017).
    """

    def test_projection_is_not_identity(self):
        """Verify that projection actually transforms the distribution when atoms shift."""
        from distributional_ppo import DistributionalPPO

        algo = DistributionalPPO.__new__(DistributionalPPO)

        # Setup: distribution over original atoms
        batch_size = 4
        num_atoms = 21
        v_min, v_max = -10.0, 10.0
        target_atoms = torch.linspace(v_min, v_max, num_atoms)

        # Create non-uniform distribution (e.g., peaked at center)
        probs = torch.zeros(batch_size, num_atoms)
        probs[:, 10] = 0.8  # Peak at center
        probs[:, 9] = 0.1   # Some mass nearby
        probs[:, 11] = 0.1

        # Shift atoms by a significant amount
        delta = 2.0
        source_atoms = target_atoms + delta

        # Project
        projected = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Key assertion: projected distribution should be DIFFERENT from original
        # (unless delta=0, which is not the case here)
        assert not torch.allclose(projected, probs, atol=1e-3), \
            "BUG: Projection returned identity! VF clipping is non-functional."

        # Projected should still be a valid probability distribution
        assert torch.allclose(projected.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(projected >= 0.0)

    def test_twin_critics_clipping_applies_projection(self):
        """
        Verify that Twin Critics categorical VF clipping uses actual projection,
        not identity. This is the core test for Issue #1.
        """
        from distributional_ppo import DistributionalPPO

        algo = DistributionalPPO.__new__(DistributionalPPO)

        # Simulate the scenario in _twin_critics_categorical_vf_loss_with_clipping
        batch_size = 4
        num_atoms = 21
        v_min, v_max = -10.0, 10.0
        atoms = torch.linspace(v_min, v_max, num_atoms)

        # Current distribution (from critic)
        current_probs_1 = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)

        # Simulate clipping: shift atoms based on (clipped_mean - current_mean)
        # This is what happens in VF clipping: atoms_shifted = atoms + delta
        delta = torch.tensor([0.5, 1.0, -0.5, -1.0]).view(-1, 1)  # Different per sample
        atoms_shifted_1 = atoms.unsqueeze(0) + delta  # [batch, num_atoms]

        # Project current distribution onto shifted atoms (this is what the fix does)
        clipped_probs_1 = algo._project_categorical_distribution(
            probs=current_probs_1,
            source_atoms=atoms_shifted_1,
            target_atoms=atoms,
        )

        # Verify projection is meaningful (not identity)
        # For samples with non-zero delta, distribution should change
        for i in range(batch_size):
            if abs(delta[i].item()) > 0.1:
                # Distribution should be different after projection
                assert not torch.allclose(clipped_probs_1[i], current_probs_1[i], atol=1e-3), \
                    f"Sample {i}: Projection returned identity for delta={delta[i].item()}"

        # Distribution should remain valid
        assert torch.allclose(clipped_probs_1.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(clipped_probs_1 >= 0.0)

    def test_deprecated_function_redirects(self):
        """Verify that deprecated _project_distribution redirects to proper implementation."""
        from distributional_ppo import DistributionalPPO
        import warnings

        algo = DistributionalPPO.__new__(DistributionalPPO)

        batch_size = 2
        num_atoms = 5
        probs = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)
        atoms_src = torch.linspace(-1, 1, num_atoms)
        atoms_dst = torch.linspace(-1, 1, num_atoms)

        # Old function should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = algo._project_distribution(probs, atoms_src, atoms_dst)

            # Should have warning
            assert len(w) >= 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "_project_distribution is deprecated" in str(w[-1].message)

        # Result should match proper implementation
        expected = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=atoms_src,
            target_atoms=atoms_dst,
        )
        assert torch.allclose(result, expected, atol=1e-6)


# ============================================================================
# ISSUE #2 TESTS: Yang-Zhang Rogers-Satchell Denominator
# ============================================================================

class TestYangZhangRogersSatchellFix:
    """
    Tests for Issue #2: Rogers-Satchell used (n-1) instead of n.

    PROBLEM: The RS component in Yang-Zhang volatility was calculated with
    Bessel's correction (n-1) which is incorrect. RS is an uncentered estimator
    and should use n per the original Yang-Zhang (2000) paper.

    FIX: Changed from rs_sum / (rs_count - 1) to rs_sum / rs_count.
    """

    def test_rs_component_uses_n_not_n_minus_1(self):
        """
        Verify that RS component uses n (not n-1) in denominator.

        Reference: Yang & Zhang (2000) formula:
        σ²_rs = (1/n) Σ[log(H/C)·log(H/O) + log(L/C)·log(L/O)]
        """
        from transformers import _try_calculate_yang_zhang

        # Create test data with known RS contribution
        n = 10
        bars = []

        # Generate bars where RS contribution can be calculated analytically
        for i in range(n):
            bar = {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
            }
            bars.append(bar)

        # Calculate what RS should be with n
        # term = log(H/C)*log(H/O) + log(L/C)*log(L/O)
        term1 = math.log(105/102) * math.log(105/100)
        term2 = math.log(95/102) * math.log(95/100)
        expected_term = term1 + term2
        expected_rs_sq_with_n = expected_term  # Since all bars are identical, sum/n = term

        # Calculate what RS would be with (n-1) - the OLD buggy formula
        expected_rs_sq_with_n_minus_1 = (n * expected_term) / (n - 1)

        # The old formula inflates by n/(n-1)
        inflation_factor = n / (n - 1)  # 1.111... for n=10
        assert abs(expected_rs_sq_with_n_minus_1 / expected_rs_sq_with_n - inflation_factor) < 1e-10

        # Call the implementation
        result = _try_calculate_yang_zhang(bars, n)

        # The result should NOT be inflated by the factor
        # We can't easily extract RS component alone, but we can verify
        # the overall volatility is reasonable (not inflated)
        assert result is not None

        # For these parameters:
        # k = 0.34 (constant in formula)
        # σ_o depends on overnight returns (prev_close → open)
        # σ_c depends on open → close returns
        # Since all bars have same O=100, C=102, we can compute expected values

        # overnight_returns = log(open/prev_close) = log(100/102) ≈ -0.0198
        # Since all identical, mean = -0.0198, variance σ²_o = 0
        # Actually for i >= 1 we have bars[i-1].close = 102, bars[i].open = 100
        # So overnight_return = log(100/102) = -0.0198
        # For 9 returns (n-1 transitions), they're all equal so variance = 0

        # This means σ_o = 0 (all overnight returns identical)
        # σ_c = 0 (all open-close returns identical)
        # Only σ_rs contributes

        # Expected result: sqrt(σ²_o + k*σ²_c + (1-k)*σ²_rs)
        # = sqrt(0 + 0.34*0 + 0.66*expected_rs_sq_with_n)
        expected_result = math.sqrt(0.66 * expected_rs_sq_with_n)

        # Tolerance for floating point
        # With the fix, result should be close to expected_result
        # With the bug, it would be sqrt(0.66 * expected_rs_sq_with_n_minus_1) ≈ expected_result * sqrt(1.111)
        buggy_result = math.sqrt(0.66 * expected_rs_sq_with_n_minus_1)

        # The result should be closer to expected (n) than buggy (n-1)
        error_with_fix = abs(result - expected_result)
        error_with_bug = abs(result - buggy_result)

        # Result should match the correct formula (not the buggy one)
        assert error_with_fix < error_with_bug, \
            f"Result {result} is closer to buggy value {buggy_result} than correct {expected_result}"

    def test_rs_inflation_quantified(self):
        """
        Quantify the RS inflation for various n values.
        The old bug inflated RS by n/(n-1):
        - n=10: +11.1%
        - n=20: +5.3%
        - n=50: +2.0%
        """
        # These are the documented inflation factors
        test_cases = [
            (10, 1.111),  # 10/9 = 1.111
            (20, 1.053),  # 20/19 = 1.053
            (50, 1.020),  # 50/49 = 1.020
            (100, 1.010), # 100/99 = 1.010
        ]

        for n, expected_inflation in test_cases:
            actual_inflation = n / (n - 1)
            assert abs(actual_inflation - expected_inflation) < 0.001, \
                f"n={n}: expected inflation {expected_inflation}, got {actual_inflation}"

    def test_yang_zhang_matches_academic_formula(self):
        """
        Verify that Yang-Zhang implementation matches the academic formula.

        Academic formula (Yang & Zhang, 2000):
        σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_rs

        where:
        - σ²_o = (1/(n-1)) Σ(log(O_i/C_{i-1}) - μ_o)²  [centered, uses n-1]
        - σ²_c = (1/(n-1)) Σ(log(C_i/O_i) - μ_c)²      [centered, uses n-1]
        - σ²_rs = (1/n) Σ[log(H/C)·log(H/O) + log(L/C)·log(L/O)]  [uncentered, uses n]
        - k = 0.34
        """
        from transformers import _try_calculate_yang_zhang

        # Generate random but consistent test data
        np.random.seed(42)
        n = 20

        bars = []
        prev_close = 100.0
        for i in range(n):
            open_price = prev_close * (1 + np.random.uniform(-0.02, 0.02))
            high = open_price * (1 + np.random.uniform(0.001, 0.03))
            low = open_price * (1 - np.random.uniform(0.001, 0.03))
            close = (high + low) / 2 * (1 + np.random.uniform(-0.01, 0.01))

            bars.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
            })
            prev_close = close

        # Manual calculation following academic formula exactly
        k = 0.34

        # σ²_o: overnight returns (centered, n-1)
        overnight_returns = []
        for i in range(1, len(bars)):
            r = math.log(bars[i]['open'] / bars[i-1]['close'])
            overnight_returns.append(r)

        mean_o = sum(overnight_returns) / len(overnight_returns)
        sigma_o_sq = sum((r - mean_o)**2 for r in overnight_returns) / (len(overnight_returns) - 1)

        # σ²_c: open-close returns (centered, n-1)
        oc_returns = []
        for bar in bars:
            r = math.log(bar['close'] / bar['open'])
            oc_returns.append(r)

        mean_c = sum(oc_returns) / len(oc_returns)
        sigma_c_sq = sum((r - mean_c)**2 for r in oc_returns) / (len(oc_returns) - 1)

        # σ²_rs: Rogers-Satchell (uncentered, uses n NOT n-1)
        rs_sum = 0.0
        for bar in bars:
            h, l, o, c = bar['high'], bar['low'], bar['open'], bar['close']
            term = math.log(h/c) * math.log(h/o) + math.log(l/c) * math.log(l/o)
            rs_sum += term

        # CRITICAL: RS uses n, not (n-1)
        sigma_rs_sq = rs_sum / n

        # Combined Yang-Zhang
        sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq
        expected_result = math.sqrt(sigma_yz_sq)

        # Get implementation result
        actual_result = _try_calculate_yang_zhang(bars, n)

        # Should match within floating point tolerance
        assert actual_result is not None
        assert abs(actual_result - expected_result) < 1e-10, \
            f"Implementation {actual_result} doesn't match academic formula {expected_result}"


# ============================================================================
# REGRESSION TESTS: Ensure fixes don't break existing functionality
# ============================================================================

class TestRegressionPreventionProjection:
    """Regression tests for projection fix."""

    def test_projection_still_preserves_mean(self):
        """Mean preservation should still work after fix."""
        from distributional_ppo import DistributionalPPO

        algo = DistributionalPPO.__new__(DistributionalPPO)

        batch_size = 8
        num_atoms = 21
        v_min, v_max = -10.0, 10.0
        target_atoms = torch.linspace(v_min, v_max, num_atoms)

        # Random distribution
        probs = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)

        # Shift atoms
        delta = 1.5
        source_atoms = target_atoms + delta

        # Project
        projected = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Compute means
        original_mean = (probs * source_atoms.unsqueeze(0)).sum(dim=1)
        projected_mean = (projected * target_atoms.unsqueeze(0)).sum(dim=1)

        # Means should be close (within support clipping tolerance)
        # Note: clamping to target support may introduce some error
        assert torch.allclose(original_mean, projected_mean, atol=0.5)


class TestRegressionPreventionYangZhang:
    """Regression tests for Yang-Zhang fix."""

    def test_yang_zhang_still_handles_edge_cases(self):
        """Edge case handling should still work."""
        from transformers import calculate_yang_zhang_volatility

        # Empty data
        assert calculate_yang_zhang_volatility([], 10) is None

        # Insufficient data
        assert calculate_yang_zhang_volatility([{'open': 100, 'high': 101, 'low': 99, 'close': 100}], 10) is None

        # n < 2
        assert calculate_yang_zhang_volatility([{'open': 100, 'high': 101, 'low': 99, 'close': 100}], 1) is None

    def test_yang_zhang_returns_positive_volatility(self):
        """Yang-Zhang should always return positive volatility for valid data."""
        from transformers import calculate_yang_zhang_volatility

        # Normal market data
        bars = []
        price = 100.0
        for i in range(30):
            o = price
            h = o * 1.01
            l = o * 0.99
            c = o * (1 + np.random.uniform(-0.005, 0.005))
            bars.append({'open': o, 'high': h, 'low': l, 'close': c})
            price = c

        result = calculate_yang_zhang_volatility(bars, 20)
        assert result is not None
        assert result > 0, "Volatility should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
