"""
Full integration tests for DistributionalPPO with focus on helper methods.

These tests cover helper methods that can be tested without full policy integration.
"""

import gc
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except ImportError:
    RecurrentActorCriticPolicy = None
    RecurrentRolloutBuffer = None
    RNNStates = None

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    RawRecurrentRolloutBuffer,
    _compute_returns_with_time_limits,
)


# =============================================================================
# Test Environment
# =============================================================================

class FullInfoEnv(gymnasium.Env):
    """Environment that returns all possible info fields for coverage."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 42, max_steps: int = 10):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = max_steps
        self._state = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        return self._state.copy(), {
            "episode_start": True,
            "ev_group_key": "test_group",
        }

    def step(self, action):
        self._step_count += 1
        action_scalar = float(action[0]) if hasattr(action, '__len__') else float(action)
        self._state = np.clip(
            self._state + 0.1 * action_scalar + 0.01 * self._rng.standard_normal(4),
            -10.0, 10.0
        ).astype(np.float32)

        reward = float(-np.sum(self._state ** 2) * 0.01 + 0.1)
        terminated = self._step_count >= self._max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "episode_win": 1 if reward > 0 else 0,
            "TimeLimit.truncated": truncated,
            "reward_raw_fraction": reward * 0.5,
            "reward_used_fraction": reward * 0.4,
            "reward_clip_bound_fraction": 0.1,
            "reward_clip_hard_cap_fraction": 0.05,
            "reward_costs_fraction": 0.02,
            "reward_robust_clip_fraction": 0.01,
            "equity": 100.0 + reward * 10,
        }

        if terminated:
            info["terminal_observation"] = self._state.copy()
            info["episode"] = {"r": reward * self._max_steps, "l": self._step_count}

        return self._state.copy(), reward, terminated, truncated, info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


def make_vec_env(env_cls=FullInfoEnv, n_envs=2, seed=42, **kwargs):
    """Create vectorized environment."""
    def make_env():
        return env_cls(seed=seed, **kwargs)
    return DummyVecEnv([make_env for _ in range(n_envs)])


# =============================================================================
# Tests for additional helper methods
# =============================================================================

class TestDistributionalPPOHelpers:
    """Test helper methods in DistributionalPPO."""

    @pytest.fixture
    def model(self):
        """Create a basic model instance."""
        env = make_vec_env()
        m = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        return m

    def test_extract_critic_states_with_rnn_states(self, model):
        """Test _extract_critic_states with RNN states."""
        lstm_states = RNNStates(
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
        )

        result = model._extract_critic_states(lstm_states)

        assert result is not None

    def test_extract_critic_states_none(self, model):
        """Test _extract_critic_states with None."""
        result = model._extract_critic_states(None)
        assert result is None

    def test_extract_actor_states_with_rnn_states(self, model):
        """Test _extract_actor_states with RNN states."""
        lstm_states = RNNStates(
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
        )

        result = model._extract_actor_states(lstm_states)

        assert result is not None

    def test_extract_actor_states_none(self, model):
        """Test _extract_actor_states with None."""
        result = model._extract_actor_states(None)
        assert result is None

    def test_clone_states_to_device(self, model):
        """Test _clone_states_to_device."""
        lstm_states = RNNStates(
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
        )

        result = model._clone_states_to_device(lstm_states, torch.device("cpu"))

        assert result is not None

    def test_clone_observations_to_device_dict(self, model):
        """Test _clone_observations_to_device with dict obs."""
        obs = {"obs": torch.randn(2, 4)}

        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))

        assert "obs" in result

    def test_clone_observations_to_device_tensor(self, model):
        """Test _clone_observations_to_device with tensor obs."""
        obs = torch.randn(2, 4)

        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))

        assert isinstance(result, torch.Tensor)

    def test_detach_observations_to_cpu_dict(self, model):
        """Test _detach_observations_to_cpu with dict obs."""
        obs = {"obs": torch.randn(2, 4, requires_grad=True)}

        result = DistributionalPPO._detach_observations_to_cpu(obs)

        assert "obs" in result
        assert not result["obs"].requires_grad

    def test_detach_observations_to_cpu_tensor(self, model):
        """Test _detach_observations_to_cpu with tensor obs."""
        obs = torch.randn(2, 4, requires_grad=True)

        result = DistributionalPPO._detach_observations_to_cpu(obs)

        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad

    def test_detach_states_to_cpu(self, model):
        """Test _detach_states_to_cpu."""
        lstm_states = RNNStates(
            (torch.randn(1, 2, 256, requires_grad=True), torch.randn(1, 2, 256)),
            (torch.randn(1, 2, 256), torch.randn(1, 2, 256)),
        )

        result = model._detach_states_to_cpu(lstm_states)

        assert result is not None

    def test_detach_episode_starts_to_cpu(self, model):
        """Test _detach_episode_starts_to_cpu."""
        episode_starts = torch.zeros(10)

        result = DistributionalPPO._detach_episode_starts_to_cpu(episode_starts)

        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"


class TestDistributionalPPOOptimizer:
    """Test optimizer-related methods."""

    @pytest.fixture
    def model(self):
        """Create a basic model."""
        env = make_vec_env()
        m = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        return m

    def test_get_optimizer_class(self, model):
        """Test _get_optimizer_class."""
        cls = model._get_optimizer_class()
        assert cls is not None

    def test_get_optimizer_kwargs(self, model):
        """Test _get_optimizer_kwargs."""
        kwargs = model._get_optimizer_kwargs()
        assert isinstance(kwargs, dict)


class TestDistributionalPPOValueScale:
    """Test value scale related methods."""

    @pytest.fixture
    def model(self):
        """Create a model."""
        env = make_vec_env()
        m = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        return m

    def test_resolve_value_scale_safe(self, model):
        """Test _resolve_value_scale_safe."""
        result = model._resolve_value_scale_safe()
        assert isinstance(result, float)
        assert result > 0

    def test_to_raw_returns(self, model):
        """Test _to_raw_returns."""
        normalized = torch.tensor([0.0, 1.0, -1.0])
        result = model._to_raw_returns(normalized)
        assert result.shape == normalized.shape


class TestBoundedDualUpdate:
    """Test _bounded_dual_update static method."""

    def test_positive_gap(self):
        """Test with positive gap (violation)."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.5,
            lr=0.1,
            gap_unit=0.1,
        )
        assert result > 0.5  # Should increase

    def test_negative_gap(self):
        """Test with negative gap (no violation)."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.5,
            lr=0.1,
            gap_unit=-0.1,
        )
        assert result < 0.5  # Should decrease

    def test_clamped_to_zero(self):
        """Test clamping to zero."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.1,
            lr=1.0,
            gap_unit=-10.0,
        )
        assert result == 0.0

    def test_clamped_to_one(self):
        """Test clamping to one."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.9,
            lr=1.0,
            gap_unit=10.0,
        )
        assert result == 1.0

    def test_nan_lambda(self):
        """Test with NaN lambda."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=float("nan"),
            lr=0.1,
            gap_unit=0.1,
        )
        assert math.isfinite(result)

    def test_nan_lr(self):
        """Test with NaN lr."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.5,
            lr=float("nan"),
            gap_unit=0.1,
        )
        assert result == 0.5  # No change

    def test_nan_gap(self):
        """Test with NaN gap."""
        result = DistributionalPPO._bounded_dual_update(
            lambda_value=0.5,
            lr=0.1,
            gap_unit=float("nan"),
        )
        assert result == 0.5  # No change


class TestValueTargetOutlierFractions:
    """Test _value_target_outlier_fractions static method."""

    def test_no_outliers(self):
        """Test with no outliers."""
        values = torch.tensor([0.0, 0.5, 1.0])
        low, high = DistributionalPPO._value_target_outlier_fractions(
            values, support_min=-1.0, support_max=2.0
        )
        assert low == 0.0
        assert high == 0.0

    def test_all_below(self):
        """Test with all values below support_min."""
        values = torch.tensor([-5.0, -4.0, -3.0])
        low, high = DistributionalPPO._value_target_outlier_fractions(
            values, support_min=-1.0, support_max=2.0
        )
        assert low == 1.0
        assert high == 0.0

    def test_all_above(self):
        """Test with all values above support_max."""
        values = torch.tensor([5.0, 6.0, 7.0])
        low, high = DistributionalPPO._value_target_outlier_fractions(
            values, support_min=-1.0, support_max=2.0
        )
        assert low == 0.0
        assert high == 1.0

    def test_mixed(self):
        """Test with mixed values."""
        values = torch.tensor([-5.0, 0.0, 5.0])  # 1/3 below, 1/3 above
        low, high = DistributionalPPO._value_target_outlier_fractions(
            values, support_min=-1.0, support_max=2.0
        )
        assert abs(low - 1/3) < 0.01
        assert abs(high - 1/3) < 0.01

    def test_empty_tensor(self):
        """Test with empty tensor."""
        values = torch.tensor([])
        low, high = DistributionalPPO._value_target_outlier_fractions(
            values, support_min=-1.0, support_max=2.0
        )
        assert low == 0.0
        assert high == 0.0


class TestProjectDistributions:
    """Test distribution projection methods."""

    @pytest.fixture
    def model(self):
        """Create a model."""
        env = make_vec_env()
        m = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        return m

    def test_project_categorical_distribution(self, model):
        """Test _project_categorical_distribution."""
        probs = torch.rand(2, 51)
        probs = probs / probs.sum(dim=1, keepdim=True)
        atoms = torch.linspace(-1, 1, 51)
        targets = torch.tensor([0.5, -0.5])

        result = model._project_categorical_distribution(probs, atoms, targets)

        assert result.shape == probs.shape


class TestCvarFromQuantiles:
    """Test _cvar_from_quantiles method."""

    @pytest.fixture
    def model(self):
        """Create a model."""
        env = make_vec_env()
        m = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.5,
            verbose=0,
        )
        m.num_quantiles = 5
        return m

    def test_basic_cvar(self, model):
        """Test basic CVaR from quantiles."""
        quantiles = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

        result = model._cvar_from_quantiles(quantiles)

        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_batch_cvar(self, model):
        """Test batched CVaR from quantiles."""
        quantiles = torch.randn(10, 5)

        result = model._cvar_from_quantiles(quantiles)

        assert result.shape == (10,)


# =============================================================================
# Cleanup
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
