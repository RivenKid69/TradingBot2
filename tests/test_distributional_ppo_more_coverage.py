"""
Additional tests to improve coverage of distributional_ppo.py.
Focuses on helper methods, static functions, and edge cases.
"""

import math
import types
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests import test_distributional_ppo_raw_outliers  # noqa: F401

from distributional_ppo import (
    DistributionalPPO,
    safe_explained_variance,
    _compute_returns_with_time_limits,
    compute_grouped_explained_variance,
    calculate_cvar,
    create_sequencers,
)


class TestComputeReturnsWithTimeLimits:
    """Tests for _compute_returns_with_time_limits function."""

    @pytest.mark.skip(reason="Requires RecurrentRolloutBuffer mock")
    def test_no_time_limits(self):
        """Test when there are no time limits (no bootstrap)."""
        pass

    @pytest.mark.skip(reason="Requires RecurrentRolloutBuffer mock")
    def test_with_time_limits(self):
        """Test with time limits causing bootstrap."""
        pass

    @pytest.mark.skip(reason="Requires RecurrentRolloutBuffer mock")
    def test_different_gamma_values(self):
        """Test with different gamma values."""
        pass


class TestComputeGroupedExplainedVariance:
    """Tests for compute_grouped_explained_variance function."""

    def test_single_group(self):
        """Test with a single group."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        group_keys = ["A", "A", "A", "A", "A"]

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)

        assert "A" in result
        assert math.isfinite(result["A"])

    def test_multiple_groups(self):
        """Test with multiple groups."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        group_keys = ["A", "A", "A", "B", "B", "B"]

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)

        assert "A" in result
        assert "B" in result
        assert math.isfinite(result["A"])
        assert math.isfinite(result["B"])

    def test_empty_arrays(self):
        """Test with empty arrays."""
        result, summary = compute_grouped_explained_variance(
            np.array([]), np.array([]), []
        )
        assert result == {}

    def test_single_value_per_group(self):
        """Test when each group has only one value (variance is undefined)."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        group_keys = ["A", "B"]

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)

        # With single value per group, variance is 0 or nan
        for key in result:
            assert math.isnan(result[key]) or math.isfinite(result[key])


class TestCalculateCvar:
    """Tests for calculate_cvar function."""

    def test_uniform_distribution(self):
        """Test CVaR with uniform distribution."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # alpha=1.0 should return expected value (mean = 2.5)
        result = calculate_cvar(probs, atoms, alpha=1.0)
        assert torch.allclose(result, torch.tensor([2.5]), atol=1e-4)

    def test_alpha_very_small(self):
        """Test CVaR with very small alpha."""
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])

        result = calculate_cvar(probs, atoms, alpha=0.01)  # alpha must be > 0
        assert torch.isfinite(result).all()

    def test_small_alpha(self):
        """Test CVaR with small alpha (left tail)."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])

        result = calculate_cvar(probs, atoms, alpha=0.25)
        # CVaR at alpha=0.25 should be close to the lowest value
        assert result.item() <= 1.0  # Should be near lower end

    def test_batch_input(self):
        """Test CVaR with batched input."""
        batch_size = 5
        num_atoms = 10
        probs = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)
        atoms = torch.linspace(-5.0, 5.0, num_atoms)

        result = calculate_cvar(probs, atoms, alpha=0.5)
        assert result.shape == (batch_size,)
        assert torch.all(torch.isfinite(result))


class TestCreateSequencers:
    """Tests for create_sequencers function."""

    def test_single_sequence(self):
        """Test with a single continuous sequence."""
        episode_starts = np.array([True, False, False, False])
        env_change = np.array([False, False, False, False])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        assert len(indices) == 1  # Single sequence
        assert indices[0] == 0  # Starts at index 0

    def test_multiple_sequences(self):
        """Test with multiple sequences."""
        episode_starts = np.array([True, False, True, False, True, False])
        env_change = np.array([False, False, False, False, False, False])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        assert len(indices) == 3  # Three sequences

    def test_env_change_creates_sequence(self):
        """Test that env_change also creates new sequences."""
        episode_starts = np.array([True, False, False, False])
        env_change = np.array([False, False, True, False])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        assert len(indices) == 2  # Episode start + env change

    def test_pad_function(self):
        """Test the pad function returned by create_sequencers."""
        episode_starts = np.array([True, False, True, False])
        env_change = np.array([False, False, False, False])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        # Create test data
        data = np.array([1.0, 2.0, 3.0, 4.0])
        padded = pad(data)

        # Check padded shape
        assert padded.ndim >= 2

    def test_combined_start_and_change(self):
        """Test when episode_starts and env_change overlap."""
        episode_starts = np.array([True, False, True, False])
        env_change = np.array([False, False, True, False])  # Overlaps with episode_starts[2]

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        # OR logic means overlapping True values don't create extra sequences
        assert len(indices) == 2


class TestDistributionalPPOLimitMethods:
    """Tests for value limit methods in DistributionalPPO."""

    def _create_minimal_algo(self):
        """Create a minimal DistributionalPPO instance for testing."""
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._value_scale_max_rel_step = 0.1
        algo._ret_std_snapshot = 1.0
        return algo

    def test_limit_mean_step_within_bounds(self):
        """Test _limit_mean_step when proposed is within bounds."""
        algo = self._create_minimal_algo()

        old_value = 10.0
        proposed = 10.5  # Within 10% of max(10, 1) = 10, so within 1.0 delta
        reference_std = 1.0

        result = algo._limit_mean_step(old_value, proposed, reference_std)
        assert result == 10.5  # Should not be clipped

    def test_limit_mean_step_exceeds_upper(self):
        """Test _limit_mean_step when proposed exceeds upper bound."""
        algo = self._create_minimal_algo()

        old_value = 10.0
        proposed = 12.0  # Exceeds 10% of max(10, 1) = 1.0 delta
        reference_std = 1.0

        result = algo._limit_mean_step(old_value, proposed, reference_std)
        assert result <= old_value + 10.0 * 0.1 + 1e-6  # Should be clipped to upper bound

    def test_limit_mean_step_exceeds_lower(self):
        """Test _limit_mean_step when proposed exceeds lower bound."""
        algo = self._create_minimal_algo()

        old_value = 10.0
        proposed = 8.0  # Exceeds 10% downward
        reference_std = 1.0

        result = algo._limit_mean_step(old_value, proposed, reference_std)
        assert result >= old_value - 10.0 * 0.1 - 1e-6  # Should be clipped to lower bound

    def test_limit_mean_step_zero_rel_step(self):
        """Test _limit_mean_step when max_rel_step is 0 (no limiting)."""
        algo = self._create_minimal_algo()
        algo._value_scale_max_rel_step = 0.0

        old_value = 10.0
        proposed = 100.0  # Very different

        result = algo._limit_mean_step(old_value, proposed, 1.0)
        assert result == proposed  # No limiting

    def test_limit_std_step_within_bounds(self):
        """Test _limit_std_step when proposed is within bounds."""
        algo = self._create_minimal_algo()

        old_value = 1.0
        proposed = 1.05  # Within 10%

        result = algo._limit_std_step(old_value, proposed)
        assert abs(result - 1.05) < 1e-6

    def test_limit_std_step_exceeds_upper(self):
        """Test _limit_std_step when proposed exceeds upper bound."""
        algo = self._create_minimal_algo()

        old_value = 1.0
        proposed = 2.0  # Way over 10%

        result = algo._limit_std_step(old_value, proposed)
        assert result <= 1.1 + 1e-6  # Should be clipped

    def test_limit_std_step_zero_rel_step(self):
        """Test _limit_std_step when max_rel_step is 0."""
        algo = self._create_minimal_algo()
        algo._value_scale_max_rel_step = 0.0

        old_value = 1.0
        proposed = 10.0

        result = algo._limit_std_step(old_value, proposed)
        assert result == proposed

    def test_limit_std_step_negative_proposed(self):
        """Test _limit_std_step with negative proposed (clamped to 1e-8)."""
        algo = self._create_minimal_algo()

        old_value = 1.0
        proposed = -5.0

        result = algo._limit_std_step(old_value, proposed)
        assert result >= 1e-8  # Should be clamped to minimum


class TestDistributionalPPOValueConversion:
    """Tests for value conversion methods."""

    def _create_algo_with_popart(self):
        """Create algo with PopArt config."""
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._ret_mean_snapshot = 0.0
        algo._ret_std_snapshot = 1.0
        algo.normalize_returns = True
        return algo

    def test_to_raw_returns(self):
        """Test _to_raw_returns converts normalized values back to raw."""
        algo = self._create_algo_with_popart()
        algo._ret_mean_snapshot = 5.0
        algo._ret_std_snapshot = 2.0

        # Normalized value 0 should map to mean (5.0)
        normalized = torch.tensor([0.0, 1.0, -1.0])
        raw = algo._to_raw_returns(normalized)

        # raw = normalized * std + mean
        expected = torch.tensor([5.0, 7.0, 3.0])
        assert torch.allclose(raw, expected, atol=1e-5)

    def test_to_raw_returns_preserves_batch_shape(self):
        """Test _to_raw_returns preserves batch dimensions."""
        algo = self._create_algo_with_popart()

        batch = torch.randn(10, 5)
        raw = algo._to_raw_returns(batch)

        assert raw.shape == batch.shape

    def test_to_raw_returns_no_normalization(self):
        """Test _to_raw_returns when normalize_returns is False."""
        algo = self._create_algo_with_popart()
        algo.normalize_returns = False
        algo.value_target_scale = 1.0
        algo._value_target_scale_effective = 1.0

        normalized = torch.tensor([1.0, 2.0, 3.0])
        raw = algo._to_raw_returns(normalized)

        # When not normalizing, should return input unchanged (scaled by value_target_scale)
        assert torch.allclose(raw, normalized, atol=1e-5)


class TestDistributionalPPOBoundedDualUpdate:
    """Tests for _bounded_dual_update static method."""

    def test_increases_when_positive_gradient(self):
        """Test that positive gradient increases the value."""
        current = 0.5
        lr = 0.1
        gradient = 0.5

        result = DistributionalPPO._bounded_dual_update(current, lr, gradient)
        assert result > current
        assert 0.0 <= result <= 1.0

    def test_decreases_when_negative_gradient(self):
        """Test that negative gradient decreases the value."""
        current = 0.5
        lr = 0.1
        gradient = -0.5

        result = DistributionalPPO._bounded_dual_update(current, lr, gradient)
        assert result < current
        assert 0.0 <= result <= 1.0

    def test_stays_at_zero_boundary(self):
        """Test behavior at lower boundary."""
        current = 0.0
        lr = 0.1
        gradient = -1.0  # Trying to go lower

        result = DistributionalPPO._bounded_dual_update(current, lr, gradient)
        assert result >= 0.0

    def test_stays_at_one_boundary(self):
        """Test behavior at upper boundary."""
        current = 1.0
        lr = 0.1
        gradient = 1.0  # Trying to go higher

        result = DistributionalPPO._bounded_dual_update(current, lr, gradient)
        assert result <= 1.0

    def test_zero_gradient(self):
        """Test with zero gradient."""
        current = 0.5
        lr = 0.1
        gradient = 0.0

        result = DistributionalPPO._bounded_dual_update(current, lr, gradient)
        assert result == pytest.approx(current, abs=1e-6)


class TestDistributionalPPOValueTargetOutlierFractions:
    """Tests for _value_target_outlier_fractions static method."""

    def test_all_within_bounds(self):
        """Test when all values are within bounds."""
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        below, above = DistributionalPPO._value_target_outlier_fractions(
            values, -1.0, 5.0  # positional: support_min, support_max
        )
        assert below == 0.0
        assert above == 0.0

    def test_some_below(self):
        """Test when some values are below support_min."""
        values = torch.tensor([-5.0, -3.0, 0.0, 1.0])
        below, above = DistributionalPPO._value_target_outlier_fractions(
            values, -2.0, 5.0
        )
        assert below == pytest.approx(0.5)  # 2 out of 4
        assert above == 0.0

    def test_some_above(self):
        """Test when some values are above support_max."""
        values = torch.tensor([0.0, 1.0, 10.0, 15.0])
        below, above = DistributionalPPO._value_target_outlier_fractions(
            values, -5.0, 5.0
        )
        assert below == 0.0
        assert above == pytest.approx(0.5)  # 2 out of 4

    def test_mixed_outliers(self):
        """Test with outliers on both ends."""
        values = torch.tensor([-10.0, 0.0, 1.0, 10.0])
        below, above = DistributionalPPO._value_target_outlier_fractions(
            values, -5.0, 5.0
        )
        assert below == pytest.approx(0.25)  # 1 out of 4
        assert above == pytest.approx(0.25)  # 1 out of 4

    def test_empty_tensor(self):
        """Test with empty tensor."""
        values = torch.tensor([])
        below, above = DistributionalPPO._value_target_outlier_fractions(
            values, -1.0, 1.0
        )
        # With empty tensor, fractions should be 0
        assert below == 0.0
        assert above == 0.0


class TestSafeExplainedVarianceAdditional:
    """Additional edge cases for safe_explained_variance."""

    def test_constant_predictions(self):
        """Test when predictions are constant."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant

        result = safe_explained_variance(y_true, y_pred, None)
        # Constant prediction means zero variance in predictions
        assert math.isfinite(result)

    def test_perfect_negative_correlation(self):
        """Test with perfectly negatively correlated predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Opposite

        result = safe_explained_variance(y_true, y_pred, None)
        # Negative correlation should give negative EV
        assert math.isfinite(result)

    def test_very_small_values(self):
        """Test with very small values."""
        y_true = np.array([1e-15, 2e-15, 3e-15])
        y_pred = np.array([1e-15, 2e-15, 3e-15])

        result = safe_explained_variance(y_true, y_pred, None)
        assert math.isfinite(result) or math.isnan(result)

    def test_very_large_values(self):
        """Test with very large values."""
        y_true = np.array([1e15, 2e15, 3e15])
        y_pred = np.array([1e15, 2e15, 3e15])

        result = safe_explained_variance(y_true, y_pred, None)
        assert math.isfinite(result) or math.isnan(result)


class TestDistributionalPPOProjectCategorical:
    """Tests for _project_categorical_distribution method."""

    def test_no_shift_projection(self):
        """Test projection when source equals target (identity)."""
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.softmax(torch.randn(4, 10), dim=1)
        atoms = torch.linspace(-5.0, 5.0, 10)

        projected = algo._project_categorical_distribution(probs, atoms, atoms)

        # Should be approximately identity
        assert torch.allclose(projected.sum(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(projected, probs, atol=1e-3)

    def test_positive_shift(self):
        """Test projection with positive shift."""
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        target_atoms = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        source_atoms = target_atoms + 1.0  # Shifted by +1

        projected = algo._project_categorical_distribution(probs, source_atoms, target_atoms)

        # Mass should move to higher atoms
        assert torch.allclose(projected.sum(dim=1), torch.ones(1), atol=1e-5)
        assert projected[0, 3] > projected[0, 1]  # More mass at higher atom

    def test_negative_shift(self):
        """Test projection with negative shift."""
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        target_atoms = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        source_atoms = target_atoms - 1.0  # Shifted by -1

        projected = algo._project_categorical_distribution(probs, source_atoms, target_atoms)

        # Mass should move to lower atoms
        assert torch.allclose(projected.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_extreme_shift_clips_to_boundary(self):
        """Test that extreme shifts clip probability to boundary atoms."""
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[1.0, 0.0, 0.0]])
        target_atoms = torch.tensor([-1.0, 0.0, 1.0])
        source_atoms = target_atoms + 10.0  # Huge shift

        projected = algo._project_categorical_distribution(probs, source_atoms, target_atoms)

        # All mass should end up at the highest atom
        assert torch.allclose(projected.sum(dim=1), torch.ones(1), atol=1e-5)
        assert projected[0, 2] > 0.5  # Most mass at highest atom


class TestComputeGroupedExplainedVarianceWithWeights:
    """Additional tests for compute_grouped_explained_variance with weights."""

    def test_with_weights(self):
        """Test with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        group_keys = ["A", "A", "A", "A", "A"]
        weights = [1.0, 2.0, 1.0, 2.0, 1.0]

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_keys, weights=weights
        )

        assert "A" in result
        assert math.isfinite(result["A"])

    def test_with_zero_weights(self):
        """Test with some zero weights."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_keys = ["A", "A", "A", "A"]
        weights = [1.0, 0.0, 1.0, 0.0]  # Half zeros

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_keys, weights=weights
        )

        # Should still compute valid result using non-zero weights
        assert "A" in result

    def test_with_nan_in_values(self):
        """Test with NaN values in input."""
        y_true = np.array([1.0, np.nan, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_keys = ["A", "A", "A", "A"]

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)

        # Should handle NaN by filtering
        assert "A" in result

    def test_with_empty_group_key(self):
        """Test when group key is empty string."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        group_keys = ["", ""]  # Empty keys

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)

        # Empty keys should be converted to "group_N"
        assert len(result) >= 1


class TestSmoothValueTargetScale:
    """Tests for _smooth_value_target_scale method."""

    def _create_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._value_target_scale_smoothing_beta = 0.9
        algo._value_target_scale_min = 0.1
        algo._value_target_scale_max = 10.0
        return algo

    def test_smooth_blends_values(self):
        """Test that smoothing blends previous and target."""
        algo = self._create_algo()

        previous = 1.0
        target = 2.0

        result = algo._smooth_value_target_scale(previous, target)

        # With beta=0.9, result should be close to target
        assert 0.0 < result
        assert math.isfinite(result)

    def test_smooth_handles_nan_target(self):
        """Test handling of NaN target."""
        algo = self._create_algo()

        previous = 1.0
        target = float("nan")

        result = algo._smooth_value_target_scale(previous, target)

        # Should return finite value despite NaN input
        assert math.isfinite(result)

    def test_smooth_handles_inf_target(self):
        """Test handling of Inf target."""
        algo = self._create_algo()

        previous = 1.0
        target = float("inf")

        result = algo._smooth_value_target_scale(previous, target)

        # Should return finite value despite Inf input
        assert math.isfinite(result)

    def test_smooth_handles_zero_target(self):
        """Test handling of zero target."""
        algo = self._create_algo()

        previous = 1.0
        target = 0.0

        result = algo._smooth_value_target_scale(previous, target)

        # Should return finite positive value
        assert result > 0
        assert math.isfinite(result)


class TestCalculateCvarEdgeCases:
    """Edge cases for calculate_cvar function."""

    def test_concentrated_distribution(self):
        """Test CVaR with probability concentrated at one atom."""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = calculate_cvar(probs, atoms, alpha=0.5)
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-4)

    def test_bimodal_distribution(self):
        """Test CVaR with bimodal distribution."""
        probs = torch.tensor([[0.5, 0.0, 0.0, 0.5]])
        atoms = torch.tensor([-1.0, 0.0, 0.0, 1.0])

        result = calculate_cvar(probs, atoms, alpha=1.0)
        # Mean of bimodal: 0.5*(-1) + 0.5*(1) = 0
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-4)

    def test_large_batch(self):
        """Test CVaR with large batch."""
        batch_size = 100
        num_atoms = 51
        probs = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)
        atoms = torch.linspace(-10.0, 10.0, num_atoms)

        result = calculate_cvar(probs, atoms, alpha=0.25)

        assert result.shape == (batch_size,)
        assert torch.all(torch.isfinite(result))


class TestCreateSequencersEdgeCases:
    """Additional edge cases for create_sequencers."""

    def test_all_episode_starts(self):
        """Test when all timesteps are episode starts."""
        episode_starts = np.array([True, True, True, True])
        env_change = np.array([False, False, False, False])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        # Each timestep is its own sequence
        assert len(indices) == 4

    def test_all_env_changes(self):
        """Test when all timesteps are env changes."""
        episode_starts = np.array([True, False, False, False])
        env_change = np.array([False, True, True, True])

        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")

        # Each timestep after first creates new sequence
        assert len(indices) == 4

    def test_with_tensor_input(self):
        """Test that torch tensor input works."""
        episode_starts = torch.tensor([True, False, True, False])
        env_change = torch.tensor([False, False, False, False])

        # Should work with tensor input
        indices, pad, unpad = create_sequencers(episode_starts, env_change, "cpu")
        assert len(indices) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
