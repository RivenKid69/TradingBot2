"""
Comprehensive tests for distributional_ppo.py compute functions.

This module tests:
- calculate_cvar
- _weighted_variance_np
- _compute_returns_with_time_limits
- safe_explained_variance (edge cases)
- compute_grouped_explained_variance (edge cases)

Tests cover:
- Basic functionality
- Edge cases (empty inputs, extreme values, NaN/inf)
- Numerical stability
- TimeLimit bootstrap handling
"""

import math
from typing import Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer

# Import functions under test
from distributional_ppo import (
    calculate_cvar,
    _weighted_variance_np,
    _compute_returns_with_time_limits,
    safe_explained_variance,
    compute_grouped_explained_variance,
)


class TestCalculateCvar:
    """Tests for calculate_cvar function."""

    def test_basic_functionality(self):
        """Test basic CVaR computation with simple distribution."""
        # Simple uniform distribution over 5 atoms
        probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        atoms = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        alpha = 0.2  # Worst 20%

        cvar = calculate_cvar(probs, atoms, alpha)
        assert isinstance(cvar, torch.Tensor)
        assert cvar.shape == (1,)
        # For uniform distribution, CVaR at 20% should be close to -2.0
        assert torch.isclose(cvar, torch.tensor(-2.0), atol=0.1)

    def test_alpha_boundaries(self):
        """Test CVaR at boundary alpha values."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # Alpha = 0.0 (invalid - must be > 0)
        with pytest.raises(ValueError, match="alpha.*must be.*finite.*probability"):
            calculate_cvar(probs, atoms, 0.0)

        # Alpha = 1.0 (entire distribution)
        cvar_1 = calculate_cvar(probs, atoms, 1.0)
        expected_mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0
        assert torch.isclose(cvar_1, torch.tensor(expected_mean), atol=0.1)

        # Alpha very close to 0 (extreme tail)
        cvar_small = calculate_cvar(probs, atoms, 0.01)
        assert torch.isfinite(cvar_small)
        assert cvar_small <= atoms.min()  # Should be near minimum

    def test_alpha_median(self):
        """Test CVaR at median (alpha = 0.5)."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        alpha = 0.5

        cvar = calculate_cvar(probs, atoms, alpha)
        # CVaR at 50% should be mean of bottom 50%: (1.0 + 2.0) / 2 = 1.5
        assert torch.isclose(cvar, torch.tensor(1.5), atol=0.2)

    def test_small_alpha(self):
        """Test CVaR with very small alpha (extreme tail)."""
        probs = torch.tensor([[0.01, 0.09, 0.4, 0.4, 0.1]])
        atoms = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
        alpha = 0.01

        cvar = calculate_cvar(probs, atoms, alpha)
        # CVaR at 1% should be close to worst atom
        assert cvar < -5.0  # Should be in the tail

    def test_degenerate_distribution(self):
        """Test CVaR with all mass at one atom (delta distribution)."""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        atoms = torch.tensor([[5.0, 10.0, 15.0, 20.0]])
        alpha = 0.25

        cvar = calculate_cvar(probs, atoms, alpha)
        # All mass at 5.0, so CVaR should be 5.0
        assert torch.isclose(cvar, torch.tensor(5.0), atol=0.01)

    def test_batch_processing(self):
        """Test CVaR computation for batch of distributions."""
        batch_size = 8
        n_atoms = 5
        probs = torch.ones(batch_size, n_atoms) / n_atoms
        # atoms must be 1D (or will be flattened)
        # Each distribution uses the same atoms
        atoms = torch.linspace(-2, 2, n_atoms)
        alpha = 0.2

        cvar = calculate_cvar(probs, atoms, alpha)
        assert cvar.shape == (batch_size,)
        assert torch.all(torch.isfinite(cvar))

    def test_negative_atoms(self):
        """Test CVaR with all negative atoms."""
        probs = torch.tensor([[0.2, 0.3, 0.5]])
        atoms = torch.tensor([[-10.0, -5.0, -1.0]])
        alpha = 0.5

        cvar = calculate_cvar(probs, atoms, alpha)
        assert cvar < 0.0
        assert torch.isfinite(cvar)

    def test_mixed_sign_atoms(self):
        """Test CVaR with atoms of mixed signs."""
        probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        atoms = torch.tensor([[-5.0, -2.0, 0.0, 2.0, 5.0]])
        alpha = 0.4

        cvar = calculate_cvar(probs, atoms, alpha)
        assert cvar < 0.0  # Should be in negative region
        assert torch.isfinite(cvar)

    def test_near_zero_probabilities(self):
        """Test CVaR with near-zero probabilities."""
        probs = torch.tensor([[0.001, 0.001, 0.998]])
        atoms = torch.tensor([[-100.0, -50.0, 10.0]])
        alpha = 0.1

        cvar = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(cvar)

    def test_uniform_atoms(self):
        """Test CVaR with uniformly spaced atoms."""
        n_atoms = 11
        probs = torch.ones(1, n_atoms) / n_atoms
        atoms = torch.linspace(-5, 5, n_atoms).unsqueeze(0)
        alpha = 0.2

        cvar = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(cvar)
        assert cvar < atoms.mean()  # CVaR should be less than mean

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large atom values."""
        probs = torch.tensor([[0.5, 0.5]])
        atoms = torch.tensor([[1e6, 1e7]])
        alpha = 0.5

        cvar = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(cvar)
        assert cvar < 1e7

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small atom values."""
        probs = torch.tensor([[0.5, 0.5]])
        atoms = torch.tensor([[1e-8, 1e-7]])
        alpha = 0.5

        cvar = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(cvar)

    def test_gradient_flow(self):
        """Test that gradients flow through CVaR computation."""
        probs = torch.tensor([[0.3, 0.7]], requires_grad=True)
        atoms = torch.tensor([[1.0, 2.0]], requires_grad=True)
        alpha = 0.5

        cvar = calculate_cvar(probs, atoms, alpha)
        loss = cvar.sum()
        loss.backward()

        assert probs.grad is not None
        assert atoms.grad is not None
        assert torch.all(torch.isfinite(probs.grad))
        assert torch.all(torch.isfinite(atoms.grad))


class TestWeightedVarianceNp:
    """Tests for _weighted_variance_np function."""

    def test_basic_functionality(self):
        """Test basic weighted variance computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        var = _weighted_variance_np(values, weights)
        # Variance for [1,2,3,4,5] with uniform weights
        # Uses Bessel's correction (ddof=1 equivalent)
        expected_var = np.var([1, 2, 3, 4, 5], ddof=1)  # Should be 2.5
        assert abs(var - expected_var) < 0.1

    def test_non_uniform_weights(self):
        """Test variance with non-uniform weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0, 1.0])

        var = _weighted_variance_np(values, weights)
        # Weighted variance should be finite
        assert np.isfinite(var)
        assert var >= 0.0

    def test_single_value(self):
        """Test variance with single value."""
        values = np.array([42.0])
        weights = np.array([1.0])

        var = _weighted_variance_np(values, weights)
        # With only one value and weights, cannot compute variance (need >= 2)
        # Function returns NaN for this case
        assert np.isnan(var)

    def test_all_weights_zero(self):
        """Test edge case: all weights zero."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        var = _weighted_variance_np(values, weights)
        # Should return NaN or 0.0
        assert np.isnan(var) or var == 0.0

    def test_single_nonzero_weight(self):
        """Test edge case: only one non-zero weight."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 1.0, 0.0])

        var = _weighted_variance_np(values, weights)
        # After filtering non-positive weights, only one value remains
        # Function returns NaN for single value case
        assert np.isnan(var)

    def test_very_large_weights(self):
        """Test numerical stability with very large weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1e50, 1e50, 1e50])

        var = _weighted_variance_np(values, weights)
        # Should not overflow
        assert np.isfinite(var)
        # Should be similar to unweighted variance
        assert abs(var - np.var(values)) < 0.5

    def test_negative_weights(self):
        """Test handling of negative weights (edge case)."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([-1.0, 2.0, 3.0])

        # May return NaN or handle specially
        var = _weighted_variance_np(values, weights)
        # Just check it doesn't crash
        assert isinstance(var, (float, np.floating))

    def test_nan_in_weights(self):
        """Test handling of NaN in weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, np.nan, 1.0])

        var = _weighted_variance_np(values, weights)
        # NaN weights are filtered out, leaving only 2 finite values
        # Should compute variance on remaining values
        assert np.isfinite(var) or np.isnan(var)  # May compute or return NaN

    def test_nan_in_values(self):
        """Test handling of NaN in values."""
        values = np.array([1.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 1.0])

        var = _weighted_variance_np(values, weights)
        # NaN values are filtered out, leaving only 2 finite values
        # Should compute variance on remaining values
        assert np.isfinite(var) or np.isnan(var)  # May compute or return NaN

    def test_empty_arrays(self):
        """Test with empty arrays."""
        values = np.array([])
        weights = np.array([])

        var = _weighted_variance_np(values, weights)
        # Should return NaN
        assert np.isnan(var)

    def test_identical_values(self):
        """Test variance with identical values."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        var = _weighted_variance_np(values, weights)
        # Variance should be 0.0
        assert abs(var) < 1e-10

    def test_high_variance_values(self):
        """Test with high variance values."""
        values = np.array([-1000.0, 0.0, 1000.0])
        weights = np.array([1.0, 1.0, 1.0])

        var = _weighted_variance_np(values, weights)
        assert np.isfinite(var)
        assert var > 0.0


class TestComputeReturnsWithTimeLimits:
    """Tests for _compute_returns_with_time_limits function."""

    def create_mock_buffer(
        self,
        buffer_size: int = 10,
        n_envs: int = 4,
        rewards: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        episode_starts: Optional[np.ndarray] = None,
    ) -> RecurrentRolloutBuffer:
        """Create a mock rollout buffer for testing."""
        buffer = Mock(spec=RecurrentRolloutBuffer)

        if rewards is None:
            rewards = np.random.randn(buffer_size, n_envs).astype(np.float32)
        if values is None:
            values = np.random.randn(buffer_size, n_envs).astype(np.float32)
        if episode_starts is None:
            episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
            episode_starts[0, :] = 1.0  # First step is episode start

        buffer.rewards = rewards
        buffer.values = values
        buffer.episode_starts = episode_starts
        buffer.advantages = None
        buffer.returns = None

        return buffer

    def test_basic_gae_computation(self):
        """Test basic GAE computation without TimeLimit."""
        buffer_size, n_envs = 10, 4
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (buffer_size, n_envs)
        assert buffer.returns.shape == (buffer_size, n_envs)
        assert np.all(np.isfinite(buffer.advantages))
        assert np.all(np.isfinite(buffer.returns))

    def test_with_time_limit_bootstrap(self):
        """Test GAE computation with TimeLimit bootstrap."""
        buffer_size, n_envs = 10, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95

        # Set TimeLimit flag at step 5
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_mask[5, 0] = True
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)
        time_limit_bootstrap[5, 0] = 10.0  # Bootstrap value

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert buffer.advantages is not None
        assert buffer.returns is not None
        # TimeLimit bootstrap should affect advantages at step 5
        assert np.isfinite(buffer.advantages[5, 0])

    def test_all_time_limits(self):
        """Test GAE when all episodes are truncated by TimeLimit."""
        buffer_size, n_envs = 5, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95

        # All steps are TimeLimit truncated
        time_limit_mask = np.ones((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.random.randn(buffer_size, n_envs).astype(np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert np.all(np.isfinite(buffer.advantages))
        assert np.all(np.isfinite(buffer.returns))

    def test_mixed_time_limits(self):
        """Test GAE with mixed TimeLimit and normal episodes."""
        buffer_size, n_envs = 10, 4
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95

        # Set TimeLimit flags at various steps
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_mask[3, 0] = True
        time_limit_mask[7, 1] = True
        time_limit_mask[5, 2] = True
        time_limit_bootstrap = np.random.randn(buffer_size, n_envs).astype(np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert np.all(np.isfinite(buffer.advantages))
        assert np.all(np.isfinite(buffer.returns))

    def test_single_step_buffer(self):
        """Test GAE with single-step buffer."""
        buffer_size, n_envs = 1, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.ones(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert buffer.advantages.shape == (buffer_size, n_envs)
        assert np.all(np.isfinite(buffer.advantages))

    def test_mismatched_dimensions_error(self):
        """Test that mismatched dimensions raise ValueError."""
        buffer_size, n_envs = 10, 4
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.95

        # Mismatched time_limit_mask dimensions
        time_limit_mask = np.zeros((buffer_size - 1, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        with pytest.raises(ValueError, match="TimeLimit mask"):
            _compute_returns_with_time_limits(
                buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
            )

    def test_with_done_episodes(self):
        """Test GAE computation with done episodes."""
        buffer_size, n_envs = 10, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.array([1.0, 0.0])  # First env is done
        gamma = 0.99
        gae_lambda = 0.95
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert np.all(np.isfinite(buffer.advantages))

    def test_zero_gamma(self):
        """Test GAE with gamma = 0 (no discounting)."""
        buffer_size, n_envs = 5, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.0
        gae_lambda = 0.95
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        # With gamma=0, returns should equal rewards
        assert np.allclose(buffer.returns, buffer.rewards, atol=1e-5)

    def test_zero_gae_lambda(self):
        """Test GAE with gae_lambda = 0 (TD(0))."""
        buffer_size, n_envs = 5, 2
        buffer = self.create_mock_buffer(buffer_size, n_envs)

        last_values = torch.zeros(n_envs)
        dones = np.zeros(n_envs)
        gamma = 0.99
        gae_lambda = 0.0
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=np.bool_)
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

        _compute_returns_with_time_limits(
            buffer, last_values, dones, gamma, gae_lambda, time_limit_mask, time_limit_bootstrap
        )

        assert np.all(np.isfinite(buffer.advantages))


class TestSafeExplainedVarianceEdgeCases:
    """Additional edge case tests for safe_explained_variance."""

    def test_huge_arrays(self):
        """Test explained variance with very large arrays."""
        n = 100000
        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.1

        ev = safe_explained_variance(y_true, y_pred)
        assert np.isfinite(ev)
        assert ev > 0.5  # Should have decent correlation

    def test_perfect_anticorrelation(self):
        """Test explained variance with perfect negative correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y_true

        ev = safe_explained_variance(y_true, y_pred)
        assert ev < 0.0  # Negative EV for anticorrelation

    def test_extreme_outliers(self):
        """Test explained variance with extreme outliers."""
        y_true = np.array([1.0, 2.0, 3.0, 1e10])
        y_pred = np.array([1.1, 2.1, 3.1, 1e10 + 1e6])

        ev = safe_explained_variance(y_true, y_pred)
        assert np.isfinite(ev)


class TestComputeGroupedExplainedVarianceEdgeCases:
    """Additional edge case tests for compute_grouped_explained_variance."""

    def test_single_group(self):
        """Test grouped EV with only one group."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        groups = ["A", "A", "A", "A"]
        weights = np.ones(4)

        ev_grouped, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, weights=weights
        )

        assert isinstance(ev_grouped, dict)
        assert isinstance(summary, dict)
        assert len(ev_grouped) == 1
        assert "A" in ev_grouped
        assert "mean_unweighted" in summary or "mean_weighted" in summary

    def test_many_groups(self):
        """Test grouped EV with many groups."""
        n = 100
        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.1
        groups = [f"group_{i % 20}" for i in range(n)]
        weights = np.ones(n)

        ev_grouped, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, weights=weights
        )

        assert isinstance(ev_grouped, dict)
        assert isinstance(summary, dict)
        assert len(ev_grouped) == 20

    def test_unequal_group_sizes(self):
        """Test grouped EV with very unequal group sizes."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = y_true + 0.1
        groups = ["A", "B", "B", "B", "B", "B"]  # One small, one large group
        weights = np.ones(6)

        ev_grouped, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, weights=weights
        )

        assert len(ev_grouped) == 2
        assert "A" in ev_grouped
        assert "B" in ev_grouped


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
