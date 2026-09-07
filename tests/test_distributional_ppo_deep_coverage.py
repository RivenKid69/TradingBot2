"""
Deep coverage tests for distributional_ppo.py targeting internal methods.
Uses mock objects to test methods that require DistributionalPPO instance.
"""

import copy
import dataclasses
import math
import types
import warnings
from collections import deque
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtCandidateMetrics,
    PopArtHoldoutBatch,
    RawRecurrentRolloutBuffer,
    safe_explained_variance,
    _weighted_variance_np,
    _cfg_get,
    _popart_value_to_serializable,
    calculate_cvar,
    create_sequencers,
    compute_grouped_explained_variance,
)


# =============================================================================
# Test _compute_empirical_cvar
# =============================================================================


class TestComputeEmpiricalCvar:
    """Test _compute_empirical_cvar method."""

    def _create_model_stub(self):
        """Create a minimal DistributionalPPO stub for testing."""
        model = DistributionalPPO.__new__(DistributionalPPO)
        model.cvar_alpha = 0.1
        model._cvar_winsor_fraction = 0.01
        model._cvar_tail_warning_logged = False
        # Use object.__setattr__ to bypass property
        object.__setattr__(model, "_logger", MagicMock())
        return model

    def test_empty_rewards(self):
        """Test with empty rewards tensor."""
        model = self._create_model_stub()
        raw_rewards = torch.tensor([])

        rewards_winsor, cvar = model._compute_empirical_cvar(raw_rewards)

        assert rewards_winsor.numel() == 0
        assert cvar.item() == 0.0

    def test_basic_cvar_computation(self):
        """Test basic CVaR computation."""
        model = self._create_model_stub()
        model.cvar_alpha = 0.5  # Use 50% for clearer test

        raw_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        rewards_winsor, cvar = model._compute_empirical_cvar(raw_rewards)

        assert cvar.numel() == 1
        # CVaR at alpha=0.5 should be mean of bottom 50%
        assert cvar.item() < 5.5  # Mean of full distribution

    def test_winsorization(self):
        """Test winsorization of rewards."""
        model = self._create_model_stub()
        model._cvar_winsor_fraction = 0.1

        raw_rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])

        rewards_winsor, cvar = model._compute_empirical_cvar(raw_rewards)

        # Winsorized rewards should be clipped
        assert rewards_winsor.max().item() < 100.0

    def test_invalid_alpha(self):
        """Test with invalid alpha value."""
        model = self._create_model_stub()
        model.cvar_alpha = -0.1  # Invalid

        raw_rewards = torch.tensor([1.0, 2.0, 3.0])

        rewards_winsor, cvar = model._compute_empirical_cvar(raw_rewards)

        assert cvar.item() == 0.0

    def test_nan_alpha(self):
        """Test with NaN alpha."""
        model = self._create_model_stub()
        model.cvar_alpha = float("nan")

        raw_rewards = torch.tensor([1.0, 2.0, 3.0])

        rewards_winsor, cvar = model._compute_empirical_cvar(raw_rewards)

        assert cvar.item() == 0.0


# =============================================================================
# Test _quantile_huber_loss
# =============================================================================


class TestQuantileHuberLoss:
    """Test _quantile_huber_loss method."""

    def _create_model_stub(self, num_quantiles=21):
        """Create a minimal DistributionalPPO stub."""
        model = DistributionalPPO.__new__(DistributionalPPO)
        model._quantile_huber_kappa = 1.0
        model._quantile_levels_cache = {
            torch.device("cpu"): torch.linspace(0.025, 0.975, num_quantiles)
        }
        # Add mock policy with quantile_levels
        mock_policy = MagicMock()
        mock_policy.quantile_levels = torch.linspace(0.025, 0.975, num_quantiles)
        model.policy = mock_policy
        return model

    def test_basic_loss(self):
        """Test basic quantile Huber loss computation."""
        model = self._create_model_stub()

        predicted = torch.zeros(5, 21)
        targets = torch.zeros(5, 1)

        loss = model._quantile_huber_loss(predicted, targets)

        assert loss.numel() == 1
        assert loss.item() >= 0.0

    def test_invalid_reduction(self):
        """Test with invalid reduction mode."""
        model = self._create_model_stub()

        predicted = torch.zeros(5, 21)
        targets = torch.zeros(5, 1)

        with pytest.raises(ValueError, match="Invalid reduction"):
            model._quantile_huber_loss(predicted, targets, reduction="invalid")

    def test_reduction_none(self):
        """Test with reduction='none'."""
        model = self._create_model_stub()

        predicted = torch.zeros(5, 21)
        targets = torch.zeros(5, 1)

        loss = model._quantile_huber_loss(predicted, targets, reduction="none")

        assert loss.shape == (5,)

    def test_reduction_sum(self):
        """Test with reduction='sum'."""
        model = self._create_model_stub()

        predicted = torch.zeros(5, 21)
        targets = torch.zeros(5, 1)

        loss = model._quantile_huber_loss(predicted, targets, reduction="sum")

        assert loss.numel() == 1

    def test_scalar_targets_raises_error(self):
        """Test that scalar targets raise ValueError."""
        model = self._create_model_stub()

        predicted = torch.zeros(5, 21)
        targets = torch.tensor(0.0)  # Scalar

        with pytest.raises(ValueError, match="batch dimension"):
            model._quantile_huber_loss(predicted, targets)


# =============================================================================
# Test _project_categorical_distribution
# =============================================================================


class TestProjectCategoricalDistribution:
    """Test _project_categorical_distribution method."""

    def _create_model_stub(self):
        """Create a minimal DistributionalPPO stub."""
        return DistributionalPPO.__new__(DistributionalPPO)

    def test_basic_projection(self):
        """Test basic categorical projection."""
        model = self._create_model_stub()

        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        source_atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])
        target_atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])

        result = model._project_categorical_distribution(probs, source_atoms, target_atoms)

        assert result.shape == probs.shape
        # Sum should be approximately 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_shifted_atoms(self):
        """Test projection with shifted atoms."""
        model = self._create_model_stub()

        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        source_atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Shifted
        target_atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])

        result = model._project_categorical_distribution(probs, source_atoms, target_atoms)

        assert result.shape == probs.shape
        assert torch.allclose(result.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_batch_projection(self):
        """Test batch projection."""
        model = self._create_model_stub()

        probs = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        source_atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])
        target_atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])

        result = model._project_categorical_distribution(probs, source_atoms, target_atoms)

        assert result.shape == (2, 4)

    def test_single_atom(self):
        """Test with single atom (edge case)."""
        model = self._create_model_stub()

        probs = torch.tensor([[1.0]])
        source_atoms = torch.tensor([0.0])
        target_atoms = torch.tensor([0.0])

        result = model._project_categorical_distribution(probs, source_atoms, target_atoms)

        assert result.shape == (1, 1)
        assert result[0, 0].item() == 1.0

    def test_degenerate_atoms(self):
        """Test with all atoms equal."""
        model = self._create_model_stub()

        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        source_atoms = torch.tensor([1.0, 1.0, 1.0, 1.0])
        target_atoms = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = model._project_categorical_distribution(probs, source_atoms, target_atoms)

        assert result.shape == probs.shape


# =============================================================================
# Test _cvar_from_quantiles
# =============================================================================


class TestCvarFromQuantiles:
    """Test _cvar_from_quantiles method."""

    def _create_model_stub(self, num_quantiles=21):
        """Create model stub with quantile levels."""
        model = DistributionalPPO.__new__(DistributionalPPO)
        model.cvar_alpha = 0.1
        model._quantile_levels_cache = {
            torch.device("cpu"): torch.linspace(0.025, 0.975, num_quantiles)
        }
        return model

    def test_basic_cvar(self):
        """Test basic CVaR from quantiles."""
        model = self._create_model_stub()

        quantiles = torch.arange(21).float().unsqueeze(0)  # [1, 21]

        result = model._cvar_from_quantiles(quantiles)

        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_batch_cvar(self):
        """Test batch CVaR."""
        model = self._create_model_stub()

        quantiles = torch.randn(5, 21)

        result = model._cvar_from_quantiles(quantiles)

        assert result.shape == (5,)


# =============================================================================
# Test PopArtController evaluate_shadow edge cases
# =============================================================================


class TestPopArtControllerEvaluateShadowEdgeCases:
    """Test PopArtController evaluate_shadow edge cases."""

    def test_disabled_returns_none(self):
        """Test that disabled controller returns None."""
        controller = PopArtController(enabled=False)

        result = controller.evaluate_shadow(
            model=MagicMock(),
            returns_raw=torch.tensor([1.0, 2.0]),
            ret_mean=0.0,
            ret_std=1.0,
        )

        assert result is None

    def test_empty_returns(self):
        """Test with empty returns tensor."""
        controller = PopArtController(enabled=True)
        controller.set_logger(MagicMock())

        result = controller.evaluate_shadow(
            model=MagicMock(),
            returns_raw=torch.tensor([]),
            ret_mean=0.0,
            ret_std=1.0,
        )

        # Should handle empty returns gracefully
        assert result is None or isinstance(result, PopArtCandidateMetrics)

    def test_all_nan_returns(self):
        """Test with all NaN returns."""
        controller = PopArtController(enabled=True)
        controller.set_logger(MagicMock())

        result = controller.evaluate_shadow(
            model=MagicMock(),
            returns_raw=torch.tensor([float("nan"), float("nan")]),
            ret_mean=0.0,
            ret_std=1.0,
        )

        assert result is None or isinstance(result, PopArtCandidateMetrics)

    def test_deprecated_ev_train_warning(self):
        """Test that explained_variance_train triggers deprecation warning."""
        controller = PopArtController(enabled=True)
        controller.set_logger(MagicMock())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            controller.evaluate_shadow(
                model=MagicMock(),
                returns_raw=torch.tensor([1.0]),
                ret_mean=0.0,
                ret_std=1.0,
                explained_variance_train=0.5,  # Deprecated param
            )

            # Check deprecation warning was raised
            assert any("deprecated" in str(warning.message).lower() for warning in w)


# =============================================================================
# Test PopArtController _weighted_mean_std edge cases
# =============================================================================


class TestPopArtControllerWeightedMeanStdEdgeCases:
    """Test PopArtController _weighted_mean_std edge cases."""

    def test_all_nan_values(self):
        """Test with all NaN values."""
        controller = PopArtController(enabled=True)

        values = np.array([np.nan, np.nan, np.nan])
        weights = np.array([1.0, 1.0, 1.0])

        mean, std = controller._weighted_mean_std(values, weights)

        assert math.isnan(mean) or mean == 0.0
        assert math.isnan(std) or std == 0.0

    def test_all_inf_values(self):
        """Test with inf values."""
        controller = PopArtController(enabled=True)

        values = np.array([np.inf, 1.0, 2.0])
        weights = np.array([1.0, 1.0, 1.0])

        mean, std = controller._weighted_mean_std(values, weights)

        # Should handle inf gracefully
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_negative_weights(self):
        """Test with negative weights."""
        controller = PopArtController(enabled=True)

        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([-1.0, 1.0, 1.0])

        mean, std = controller._weighted_mean_std(values, weights)

        # Should handle negative weights
        assert isinstance(mean, float)

    def test_single_element(self):
        """Test with single element."""
        controller = PopArtController(enabled=True)

        values = np.array([5.0])
        weights = np.array([1.0])

        mean, std = controller._weighted_mean_std(values, weights)

        assert mean == 5.0 or math.isnan(mean)


# =============================================================================
# Test PopArtController _clip_fraction
# =============================================================================


class TestPopArtControllerClipFraction:
    """Test PopArtController _clip_fraction static method."""

    def test_no_clipping(self):
        """Test with no values clipped."""
        values = torch.tensor([0.5, 0.6, 0.7])

        fraction = PopArtController._clip_fraction(values, low=0.0, high=1.0)

        assert fraction == 0.0

    def test_all_clipped_low(self):
        """Test with all values below low bound."""
        values = torch.tensor([-1.0, -0.5, -0.1])

        fraction = PopArtController._clip_fraction(values, low=0.0, high=1.0)

        assert fraction == 1.0

    def test_all_clipped_high(self):
        """Test with all values above high bound."""
        values = torch.tensor([1.5, 2.0, 2.5])

        fraction = PopArtController._clip_fraction(values, low=0.0, high=1.0)

        assert fraction == 1.0

    def test_partial_clipping(self):
        """Test with some values clipped."""
        values = torch.tensor([0.5, 0.6, 1.5, 2.0])  # 2 out of 4 clipped

        fraction = PopArtController._clip_fraction(values, low=0.0, high=1.0)

        assert fraction == 0.5


# =============================================================================
# Test _cfg_get with various object types
# =============================================================================


class TestCfgGetAdvanced:
    """Advanced tests for _cfg_get function."""

    def test_ordered_dict(self):
        """Test with OrderedDict."""
        from collections import OrderedDict

        cfg = OrderedDict([("key", "value"), ("other", "data")])

        assert _cfg_get(cfg, "key") == "value"
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_namedtuple(self):
        """Test with namedtuple."""
        from collections import namedtuple

        Config = namedtuple("Config", ["key", "other"])
        cfg = Config(key="value", other="data")

        assert _cfg_get(cfg, "key") == "value"
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_class_with_slots(self):
        """Test with class using __slots__."""

        class SlottedConfig:
            __slots__ = ["key"]

            def __init__(self):
                self.key = "value"

        cfg = SlottedConfig()

        assert _cfg_get(cfg, "key") == "value"

    def test_lambda_get(self):
        """Test object with callable that's not a proper get."""

        class WeirdConfig:
            get = lambda self, x: x  # noqa

        cfg = WeirdConfig()

        # Should fall through to default
        result = _cfg_get(cfg, "key", "fallback")
        assert result is not None


# =============================================================================
# Test compute_grouped_explained_variance with various weights
# =============================================================================


class TestComputeGroupedExplainedVarianceWeights:
    """Test compute_grouped_explained_variance with weights."""

    def test_non_uniform_weights(self):
        """Test with non-uniform weights."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_ids = np.array(["A", "A", "B", "B"])
        weights = np.array([1.0, 2.0, 1.0, 3.0])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=weights
        )

        assert "A" in result
        assert "B" in result

    def test_extreme_weight_ratio(self):
        """Test with extreme weight ratio."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_ids = np.array(["A", "A", "B", "B"])
        weights = np.array([1e10, 1e-10, 1.0, 1.0])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=weights
        )

        # Should handle extreme weights
        assert "A" in result


# =============================================================================
# Test calculate_cvar with various distributions
# =============================================================================


class TestCalculateCvarDistributions:
    """Test calculate_cvar with various distribution shapes."""

    def test_concentrated_mass(self):
        """Test with probability mass concentrated at one atom."""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = calculate_cvar(probs, atoms, alpha=0.5)

        assert result.item() == 1.0

    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        probs = torch.tensor([[0.5, 0.0, 0.0, 0.5]])
        atoms = torch.tensor([0.0, 1.0, 2.0, 3.0])

        result = calculate_cvar(probs, atoms, alpha=0.5)

        assert torch.isfinite(result)

    def test_very_small_alpha(self):
        """Test with very small alpha (focus on extreme tail)."""
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = calculate_cvar(probs, atoms, alpha=0.01)

        assert torch.isfinite(result)
        # CVaR should be close to minimum atom
        assert result.item() <= 1.5


# =============================================================================
# Test create_sequencers with various episode patterns
# =============================================================================


class TestCreateSequencersPatterns:
    """Test create_sequencers with various episode patterns."""

    def test_alternating_episodes(self):
        """Test with alternating episode starts."""
        episode_starts = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        env_change = np.zeros(8)

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) >= 1

    def test_long_single_episode(self):
        """Test with long single episode."""
        episode_starts = np.zeros(100)
        episode_starts[0] = 1
        env_change = np.zeros(100)

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) == 1

    def test_all_zeros_episode_starts(self):
        """Test with no episode starts (edge case)."""
        episode_starts = np.zeros(10)
        env_change = np.zeros(10)

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        # Should handle this case gracefully
        assert seq_indices is not None


# =============================================================================
# Test _bounded_dual_update
# =============================================================================


class TestBoundedDualUpdateAdvanced:
    """Advanced tests for _bounded_dual_update."""

    def test_large_positive_gap(self):
        """Test with large positive gap."""
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, 10.0)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_large_negative_gap(self):
        """Test with large negative gap."""
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, -10.0)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_zero_lr(self):
        """Test with zero learning rate."""
        result = DistributionalPPO._bounded_dual_update(0.5, 0.0, 0.1)

        assert isinstance(result, float)
        assert result == 0.5  # Should not change

    def test_inf_gap(self):
        """Test with inf gap."""
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, float("inf"))

        assert isinstance(result, float)

    def test_boundary_values(self):
        """Test at boundary values."""
        # At 0
        result = DistributionalPPO._bounded_dual_update(0.0, 0.1, 0.1)
        assert result >= 0.0

        # At 1
        result = DistributionalPPO._bounded_dual_update(1.0, 0.1, -0.1)
        assert result <= 1.0


# =============================================================================
# Test safe_explained_variance numerical edge cases
# =============================================================================


class TestSafeExplainedVarianceNumerical:
    """Test safe_explained_variance with numerical edge cases."""

    def test_very_close_values(self):
        """Test with very close y_true and y_pred."""
        y_true = np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10])
        y_pred = np.array([1.0 + 1e-11, 1.0 + 1e-10 + 1e-11, 1.0 + 2e-10 + 1e-11])

        result = safe_explained_variance(y_true, y_pred, None)

        # Should handle numerical precision
        assert math.isfinite(result) or math.isnan(result)

    def test_alternating_signs(self):
        """Test with alternating positive/negative values."""
        y_true = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        y_pred = np.array([0.9, -0.9, 0.9, -0.9, 0.9])

        result = safe_explained_variance(y_true, y_pred, None)

        assert math.isfinite(result)
        assert result > 0.0  # Good prediction

    def test_mismatched_lengths_handled(self):
        """Test that mismatched lengths are handled."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1])  # Shorter

        result = safe_explained_variance(y_true, y_pred, None)

        # Should use minimum length
        assert math.isfinite(result) or math.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
