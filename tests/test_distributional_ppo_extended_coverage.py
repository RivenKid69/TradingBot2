"""
Extended tests for additional coverage of distributional_ppo.py.
Focuses on __init__ paths, train() helpers, and remaining uncovered methods.
"""

import math
import types
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    RawRecurrentRolloutBuffer,
    safe_explained_variance,
    _weighted_variance_np,
    _cfg_get,
    compute_grouped_explained_variance,
    calculate_cvar,
    create_sequencers,
)


# =============================================================================
# Additional DistributionalPPO Static Methods
# =============================================================================


class TestDistributionalPPORobustStdFromReturns:
    """Tests for DistributionalPPO._robust_std_from_returns."""

    def _make_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._value_scale_std_floor = 1e-8
        return algo

    def test_empty_returns(self):
        algo = self._make_algo()
        result = algo._robust_std_from_returns(torch.tensor([]))
        assert result == algo._value_scale_std_floor

    def test_normal_returns(self):
        algo = self._make_algo()
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = algo._robust_std_from_returns(returns)
        assert result > 0.0
        assert math.isfinite(result)

    def test_all_zeros(self):
        algo = self._make_algo()
        returns = torch.zeros(10)
        result = algo._robust_std_from_returns(returns)
        assert result == algo._value_scale_std_floor

    def test_with_nan_values(self):
        algo = self._make_algo()
        returns = torch.tensor([1.0, float("nan"), 3.0])
        result = algo._robust_std_from_returns(returns)
        # Should handle NaN gracefully
        assert math.isfinite(result) or result == algo._value_scale_std_floor


class TestDistributionalPPOExtractRmsStats:
    """Tests for DistributionalPPO._extract_rms_stats."""

    def _make_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        return algo

    def test_none_rms(self):
        algo = self._make_algo()
        result = algo._extract_rms_stats(None)
        assert result is None

    def test_valid_rms(self):
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        algo = self._make_algo()
        rms = RunningMeanStd(shape=())
        # Manually update to get some statistics
        rms.update(np.array([1.0, 2.0, 3.0]))

        result = algo._extract_rms_stats(rms)
        assert result is not None
        mean, var, count = result
        assert math.isfinite(mean)
        assert math.isfinite(var)
        assert count > 0

    def test_empty_rms(self):
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        algo = self._make_algo()
        rms = RunningMeanStd(shape=())
        # No updates - count is 0

        result = algo._extract_rms_stats(rms)
        assert result is None


class TestDistributionalPPOSmoothValueTargetScale:
    """Tests for DistributionalPPO._smooth_value_target_scale."""

    def _make_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._value_target_scale_ema_beta = 0.9
        return algo

    def test_ema_smoothing(self):
        algo = self._make_algo()
        result = algo._smooth_value_target_scale(1.0, 2.0)
        # Should be between old and new
        assert 1.0 <= result <= 2.0


class TestDistributionalPPOToRawReturns:
    """Tests for DistributionalPPO._to_raw_returns."""

    def _make_algo(self, normalize_returns=False):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.normalize_returns = normalize_returns
        algo._ret_mean_snapshot = 0.0
        algo._ret_std_snapshot = 1.0
        algo._value_target_scale_effective = 1.0
        algo.value_target_scale = 1.0
        return algo

    def test_normalized_returns(self):
        algo = self._make_algo(normalize_returns=True)
        algo._ret_mean_snapshot = 10.0
        algo._ret_std_snapshot = 2.0

        x = torch.tensor([1.0, 2.0])
        result = algo._to_raw_returns(x)
        # x * std + mean
        expected = x * 2.0 + 10.0
        assert torch.allclose(result, expected)

    def test_non_normalized_returns(self):
        algo = self._make_algo(normalize_returns=False)
        algo._value_target_scale_effective = 100.0
        algo.value_target_scale = 100.0

        x = torch.tensor([1.0, 2.0])
        result = algo._to_raw_returns(x)
        assert torch.allclose(result, x)


class TestDistributionalPPOResolveValueScaleSafe:
    """Tests for DistributionalPPO._resolve_value_scale_safe."""

    def test_normal_scale(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.value_target_scale = 100.0
        result = algo._resolve_value_scale_safe()
        assert result == 100.0

    def test_near_zero_scale(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.value_target_scale = 1e-10
        result = algo._resolve_value_scale_safe()
        assert result == 1.0


class TestDistributionalPPODecodeReturnsScaleOnly:
    """Tests for DistributionalPPO._decode_returns_scale_only."""

    def test_decode(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.value_target_scale = 100.0

        returns = torch.tensor([0.01, 0.02])
        decoded, scale = algo._decode_returns_scale_only(returns)

        assert scale == 100.0
        expected = returns * 100.0
        assert torch.allclose(decoded, expected)


class TestDistributionalPPOGetCvarLimitRaw:
    """Tests for DistributionalPPO._get_cvar_limit_raw."""

    def test_get_limit(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.cvar_limit = -0.05
        result = algo._get_cvar_limit_raw()
        assert result == -0.05


class TestDistributionalPPOGetCvarNormalizationParams:
    """Tests for DistributionalPPO._get_cvar_normalization_params."""

    def test_normalized_returns(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.normalize_returns = True
        algo._ret_mean_snapshot = 0.5
        algo._ret_std_snapshot = 0.1
        algo._value_scale_std_floor = 1e-8

        offset, scale = algo._get_cvar_normalization_params()
        assert offset == 0.5
        assert scale == 0.1

    def test_non_normalized_returns(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.normalize_returns = False
        algo._value_target_scale_robust = 100.0

        offset, scale = algo._get_cvar_normalization_params()
        assert offset == 0.0
        assert scale == 100.0

    def test_invalid_robust_scale(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.normalize_returns = False
        algo._value_target_scale_robust = float("nan")
        algo.value_target_scale = 100.0

        offset, scale = algo._get_cvar_normalization_params()
        assert offset == 0.0
        assert scale == 100.0


class TestPopArtControllerEvaluateHoldout:
    """Tests for PopArtController._evaluate_holdout."""

    def _make_mock_model(self, use_quantile=False):
        model = types.SimpleNamespace()
        model._use_quantile_value = use_quantile
        model.normalize_returns = False
        model._value_clip_limit_unscaled = 1.0

        policy = types.SimpleNamespace()
        policy.device = torch.device("cpu")
        policy.training = False
        policy.eval = lambda: None
        policy.train = lambda: None
        policy.recurrent_initial_state = None

        model.policy = policy
        model._policy_value_outputs = lambda obs, states, starts: torch.randn(obs.shape[0], 1)
        model._to_raw_returns = lambda x: x

        return model

    def test_holdout_evaluation_non_quantile(self):
        controller = PopArtController(enabled=True)
        model = self._make_mock_model(use_quantile=False)

        holdout = PopArtHoldoutBatch(
            observations=torch.randn(10, 4),
            returns_raw=torch.randn(10, 1),
            episode_starts=torch.zeros(10, dtype=torch.bool),
            lstm_states=None,
            mask=None,
        )

        result = controller._evaluate_holdout(
            model=model,
            holdout=holdout,
            old_mean=0.0,
            old_std=1.0,
            new_mean=0.1,
            new_std=1.1,
        )

        assert isinstance(result, dppo.PopArtHoldoutEvaluation)
        assert math.isfinite(result.ev_before) or math.isnan(result.ev_before)
        assert math.isfinite(result.ev_after) or math.isnan(result.ev_after)


class TestSafeExplainedVarianceAdditional:
    """Additional edge cases for safe_explained_variance."""

    def test_identical_arrays(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        result = safe_explained_variance(y_true, y_pred, None)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_opposite_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = safe_explained_variance(y_true, y_pred, None)
        # Should be negative
        assert result < 0.0

    def test_with_weights(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = safe_explained_variance(y_true, y_pred, weights)
        assert math.isfinite(result)


class TestWeightedVarianceNpAdditional:
    """Additional edge cases for _weighted_variance_np."""

    def test_large_variance(self):
        values = np.array([0.0, 1e6])
        result = _weighted_variance_np(values, None)
        assert math.isfinite(result)
        assert result > 0

    def test_tiny_variance(self):
        values = np.array([1.0, 1.0 + 1e-10])
        result = _weighted_variance_np(values, None)
        assert math.isfinite(result) or math.isnan(result)


class TestCreateSequencersAdditional:
    """Additional tests for create_sequencers."""

    def test_all_episode_starts(self):
        episode_starts = np.ones(10, dtype=bool)
        env_change = np.zeros(10, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert indices is not None

    def test_no_episode_starts(self):
        episode_starts = np.zeros(10, dtype=bool)
        episode_starts[0] = True  # At least first must be True
        env_change = np.zeros(10, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert indices is not None


class TestCalculateCvarAdditional:
    """Additional tests for calculate_cvar."""

    def test_small_alpha(self):
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        alpha = 0.1

        result = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(result).all()
        # CVaR at alpha=0.1 should be close to worst outcomes
        assert result.item() <= 0  # Should be negative or zero

    def test_batch_processing(self):
        probs = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.3, 0.15, 0.05],
            ]
        )
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        alpha = 0.5

        result = calculate_cvar(probs, atoms, alpha)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()


class TestDistributionalPPOQuantileLevelsTensor:
    """Tests for DistributionalPPO._quantile_levels_tensor."""

    def test_with_policy_quantile_levels(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        quantile_levels = torch.linspace(0.1, 0.9, 5)
        algo.policy = types.SimpleNamespace(quantile_levels=quantile_levels)

        result = algo._quantile_levels_tensor(torch.device("cpu"))
        assert torch.equal(result, quantile_levels)

    def test_missing_quantile_levels_raises(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.policy = types.SimpleNamespace()

        with pytest.raises(RuntimeError):
            algo._quantile_levels_tensor(torch.device("cpu"))


class TestDistributionalPPOBuildSupportDistribution:
    """Tests for DistributionalPPO._build_support_distribution."""

    def _make_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.policy = types.SimpleNamespace(
            v_min=-1.0,
            v_max=1.0,
            delta_z=0.1,
        )
        return algo

    def test_empty_template(self):
        algo = self._make_algo()
        returns_norm = torch.tensor([])
        template = torch.tensor([])

        result = algo._build_support_distribution(returns_norm, template)
        assert result.numel() == 0

    def test_basic_distribution(self):
        algo = self._make_algo()
        returns_norm = torch.tensor([[0.5]])
        template = torch.zeros(1, 21)

        result = algo._build_support_distribution(returns_norm, template)
        assert result.shape == template.shape
        # Sum should be 1 for each row
        assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-5)


class TestDistributionalPPOPolicyValueOutputs:
    """Tests for DistributionalPPO._policy_value_outputs."""

    def test_quantile_value_head(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        def mock_value_quantiles(obs, states, starts):
            return torch.randn(obs.shape[0], 21)

        algo.policy = types.SimpleNamespace(
            uses_quantile_value_head=True,
            value_quantiles=mock_value_quantiles,
        )

        obs = torch.randn(10, 4)
        result = algo._policy_value_outputs(obs, None, torch.zeros(10, dtype=torch.bool))
        assert result.shape == (10, 21)

    def test_predict_values(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        def mock_predict_values(obs, states, starts):
            return torch.randn(obs.shape[0], 1)

        algo.policy = types.SimpleNamespace(
            uses_quantile_value_head=False,
            predict_values=mock_predict_values,
        )

        obs = torch.randn(10, 4)
        result = algo._policy_value_outputs(obs, None, torch.zeros(10, dtype=torch.bool))
        assert result.shape == (10, 1)

    def test_no_value_method_raises(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.policy = types.SimpleNamespace(uses_quantile_value_head=False)

        obs = torch.randn(10, 4)
        with pytest.raises(AttributeError):
            algo._policy_value_outputs(obs, None, torch.zeros(10, dtype=torch.bool))


class TestMakeClipRangeCallable:
    """Additional tests for _make_clip_range_callable."""

    def test_zero_clip_range(self):
        from distributional_ppo import _make_clip_range_callable

        fn = _make_clip_range_callable(0.0)
        assert fn() == 0.0

    def test_large_clip_range(self):
        from distributional_ppo import _make_clip_range_callable

        fn = _make_clip_range_callable(10.0)
        assert fn() == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
