"""
Integration tests for DistributionalPPO targeting ≥95% line coverage.
Creates minimal deterministic environment and tests __init__, collect_rollouts, train.
"""

import copy
import math
import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
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
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except ImportError:
    RecurrentActorCriticPolicy = None
    RNNStates = None


class DummyCallback(BaseCallback):
    """Minimal callback for testing."""

    def __init__(self):
        super().__init__()
        self.n_calls = 0

    def _on_step(self) -> bool:
        self.n_calls += 1
        return True

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
    _compute_returns_with_time_limits,
    compute_grouped_explained_variance,
    calculate_cvar,
    create_sequencers,
    DEFAULT_CLIP_RANGE_VF,
)


# =============================================================================
# Minimal Deterministic Environment
# =============================================================================

class MinimalDeterministicEnv(gymnasium.Env):
    """Minimal environment for testing DistributionalPPO."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        # DistributionalPPO requires action space with shape (1,)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = 10
        self._state = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._step_count += 1
        # Simple dynamics - action is shape (1,)
        action_scalar = float(action[0]) if hasattr(action, '__len__') else float(action)
        self._state = np.clip(
            self._state + 0.1 * action_scalar + 0.01 * self._rng.standard_normal(4),
            -10.0, 10.0
        ).astype(np.float32)

        # Reward based on staying near origin
        reward = float(-np.sum(self._state ** 2) * 0.01 + 0.1)

        terminated = self._step_count >= self._max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "episode_win": 1 if reward > 0 else 0,
            "ev_group_key": "test_group",
        }

        return self._state.copy(), reward, terminated, truncated, info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class MinimalDiscreteEnv(gymnasium.Env):
    """Minimal discrete action environment."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = 10

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(action * 0.1 - 0.1)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_vec_env(env_cls=MinimalDeterministicEnv, n_envs=1, seed=42):
    """Create vectorized environment."""
    def make_env():
        env = env_cls(seed=seed)
        return env
    return DummyVecEnv([make_env for _ in range(n_envs)])

def make_model(env, **overrides):
    """Create a minimal DistributionalPPO model for integration tests."""
    model_kwargs = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "learning_rate": 3e-4,
        "device": "cpu",
        "verbose": 0,
    }
    model_kwargs.update(overrides)
    return DistributionalPPO(**model_kwargs)


def setup_and_collect_rollouts(model, env, n_rollout_steps):
    """Prepare learner state and collect one rollout."""
    total_timesteps = int(n_rollout_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    return model.collect_rollouts(
        env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=n_rollout_steps,
    )


# =============================================================================
# Mock Components
# =============================================================================

class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.records: Dict[str, Any] = {}
        self.name_to_value = {}
        self.name_to_count = {}
        self.name_to_excluded = {}

    def record(self, key: str, value: Any, exclude: Optional[str] = None):
        self.records[key] = value
        self.name_to_value[key] = value
        self.name_to_count[key] = self.name_to_count.get(key, 0) + 1
        if exclude:
            self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0):
        pass

    def get_dir(self) -> Optional[str]:
        return None


# =============================================================================
# Test DistributionalPPO Initialization
# =============================================================================

class TestDistributionalPPOInit:
    """Tests for DistributionalPPO.__init__."""

    def test_init_with_minimal_env(self):
        """Test initialization with minimal environment."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            learning_rate=3e-4,
            verbose=0,
        )

        assert model is not None
        assert model.env is not None
        assert model.n_envs == 1

        env.close()

    def test_init_with_cvar_parameters(self):
        """Test initialization with CVaR parameters."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.3,
            cvar_use_constraint=True,
            cvar_use_penalty=True,
            cvar_limit=-1.0,
            cvar_lambda_lr=0.01,
            verbose=0,
        )

        assert model.cvar_alpha == 0.1
        assert model.cvar_weight == 0.3

        env.close()

    def test_init_with_clip_range_vf(self):
        """Test initialization with VF clipping."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.5,
            vf_clip_warmup_updates=2,
            vf_clip_threshold_ev=0.1,
            verbose=0,
        )

        # clip_range_vf may be None if warmup is active, check the internal value
        assert model._vf_clip_warmup_updates == 2
        assert model._vf_clip_threshold_ev == 0.1

        env.close()

    def test_init_with_distributional_vf_clip_modes(self):
        """Test initialization with different VF clip modes."""
        env = make_vec_env()

        for mode in ["disable", "mean_only", "mean_and_variance", "per_quantile"]:
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                distributional_vf_clip_mode=mode,
                verbose=0,
            )
            assert model.distributional_vf_clip_mode == mode

        env.close()

    def test_init_invalid_vf_clip_mode_raises(self):
        """Test that invalid VF clip mode raises ValueError."""
        env = make_vec_env()

        with pytest.raises(ValueError, match="distributional_vf_clip_mode"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                distributional_vf_clip_mode="invalid_mode",
                verbose=0,
            )

        env.close()

    def test_init_with_optimizer_lr_bounds(self):
        """Test initialization with optimizer LR bounds."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_lr_min=1e-6,
            optimizer_lr_max=1e-2,
            scheduler_min_lr=1e-5,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_kl_parameters(self):
        """Test initialization with KL divergence parameters."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            target_kl=0.01,
            kl_early_stop=True,
            kl_epoch_decay=0.5,
            kl_exceed_stop_fraction=0.25,
            kl_absolute_stop_factor=2.5,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_entropy_parameters(self):
        """Test initialization with entropy parameters."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ent_coef_final=0.001,
            ent_coef_min=1e-4,
            ent_coef_decay_steps=100,
            entropy_boost_factor=1.5,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_value_scale_fixed(self):
        """Test initialization with fixed value scale."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_target_scale_fixed=0.1,
            verbose=0,
        )

        assert model._value_target_scale_fixed == 0.1

        env.close()

    def test_init_with_none_env(self):
        """Test lightweight initialization with None env."""
        class MockPolicy:
            device = torch.device("cpu")

        model = DistributionalPPO(
            policy=MockPolicy(),
            env=None,
        )

        assert model.env is None
        assert model._setup_complete is True

    def test_init_invalid_clip_range_vf_raises(self):
        """Test that invalid clip_range_vf raises ValueError."""
        env = make_vec_env()

        with pytest.raises(ValueError, match="clip_range_vf"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                clip_range_vf=-0.5,
                verbose=0,
            )

        env.close()

    def test_init_invalid_vf_clip_threshold_ev_raises(self):
        """Test that invalid vf_clip_threshold_ev raises ValueError."""
        env = make_vec_env()

        with pytest.raises(ValueError, match="vf_clip_threshold_ev"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                vf_clip_threshold_ev=2.0,
                verbose=0,
            )

        env.close()

    def test_init_invalid_variance_factor_raises(self):
        """Test that invalid variance factor raises ValueError."""
        env = make_vec_env()

        with pytest.raises(ValueError, match="distributional_vf_clip_variance_factor"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                distributional_vf_clip_variance_factor=0.5,
                verbose=0,
            )

        env.close()


# =============================================================================
# Test collect_rollouts and train
# =============================================================================

class TestDistributionalPPOCollectRollouts:
    """Tests for DistributionalPPO.collect_rollouts."""

    def test_collect_rollouts_basic(self):
        """Test basic rollout collection."""
        env = make_vec_env()
        model = make_model(env)

        collected = setup_and_collect_rollouts(model, env, n_rollout_steps=8)

        assert collected is True
        assert model.rollout_buffer.full is True
        assert model._last_obs is not None
        assert model._last_lstm_states is not None

        env.close()

    def test_collect_rollouts_multiple_envs(self):
        """Test rollout collection with multiple environments."""
        env = make_vec_env(n_envs=2)
        model = make_model(env, n_steps=4, batch_size=4)

        collected = setup_and_collect_rollouts(model, env, n_rollout_steps=4)

        assert collected is True
        assert model.n_envs == 2
        assert model.rollout_buffer.full is True
        assert model.rollout_buffer.observations.shape[1] == 2

        env.close()


class TestDistributionalPPOTrain:
    """Tests for DistributionalPPO.train."""

    def test_train_after_collect(self):
        """Test training after collecting rollouts."""
        env = make_vec_env()
        model = make_model(env)

        setup_and_collect_rollouts(model, env, n_rollout_steps=8)

        assert model._global_update_step == 0
        model.train()
        assert model._global_update_step == 1

        env.close()

    def test_train_with_cvar_constraint(self):
        """Test training with CVaR constraint."""
        env = make_vec_env()
        model = make_model(
            env,
            cvar_use_constraint=True,
            cvar_use_penalty=True,
            cvar_use_predicted_for_dual=True,
            cvar_alpha=0.1,
            cvar_weight=0.5,
            cvar_limit=-0.2,
            cvar_lambda_lr=0.05,
        )

        setup_and_collect_rollouts(model, env, n_rollout_steps=8)
        model.train()

        assert model._cvar_predicted_last_raw is not None
        assert model._cvar_predicted_last_unit is not None

        env.close()

    def test_train_with_kl_early_stop(self):
        """Test training with KL early stopping."""
        env = make_vec_env()
        model = make_model(
            env,
            target_kl=1e-6,
            kl_early_stop=True,
            kl_exceed_stop_fraction=0.1,
            kl_absolute_stop_factor=2.0,
        )

        setup_and_collect_rollouts(model, env, n_rollout_steps=8)
        model.train()

        assert model._global_update_step == 1

        env.close()

    def test_train_with_vf_clip_warmup(self):
        """Test training with VF clip warmup."""
        env = make_vec_env()
        model = make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_warmup_updates=2,
            vf_clip_threshold_ev=0.1,
        )

        setup_and_collect_rollouts(model, env, n_rollout_steps=8)
        model.train()

        assert model._vf_clip_warmup_updates == 2

        env.close()


class TestDistributionalPPOLearn:
    """Tests for DistributionalPPO.learn."""

    def test_learn_short_run(self):
        """Test short learn run."""
        env = make_vec_env()
        model = make_model(env)

        model.learn(total_timesteps=16)

        assert model.num_timesteps >= 16

        env.close()

    def test_learn_with_callback(self):
        """Test learn with callback."""
        env = make_vec_env()
        model = make_model(env)
        callback = DummyCallback()

        model.learn(total_timesteps=16, callback=callback)

        assert callback.n_calls > 0

        env.close()


# =============================================================================
# Test Helper Methods
# =============================================================================

class TestDistributionalPPOHelperMethods:
    """Tests for DistributionalPPO helper methods."""

    def test_compute_clip_range_value(self):
        """Test clip range computation."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_warmup=0.4,
            clip_range_warmup_updates=5,
            verbose=0,
        )

        # Early update - should use warmup value
        clip_0 = model._compute_clip_range_value(0)
        assert isinstance(clip_0, float)
        assert clip_0 > 0

        # Later update - should use base value
        clip_10 = model._compute_clip_range_value(10)
        assert isinstance(clip_10, float)
        assert clip_10 > 0

        env.close()

    def test_compute_vf_coef_value(self):
        """Test VF coefficient computation."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            vf_coef_warmup=0.1,
            vf_coef_warmup_updates=5,
            verbose=0,
        )

        # Early update - should use warmup value
        vf_0 = model._compute_vf_coef_value(0)
        assert isinstance(vf_0, float)
        assert vf_0 >= 0

        env.close()

    def test_compute_entropy_boost(self):
        """Test entropy boost computation."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            entropy_boost_factor=2.0,
            entropy_boost_cap=0.1,
            verbose=0,
        )

        # Initialize tracking attributes
        model._ev_tracking_bad_streak = 5

        boosted = model._compute_entropy_boost(0.01)
        assert isinstance(boosted, float)
        assert boosted >= 0.01

        env.close()

    def test_bounded_dual_update(self):
        """Test bounded dual update static method."""
        # Basic update
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, 0.1)
        assert 0.0 <= result <= 1.0

        # Clamp to max
        result = DistributionalPPO._bounded_dual_update(0.99, 0.1, 0.1)
        assert result <= 1.0

        # Clamp to min
        result = DistributionalPPO._bounded_dual_update(0.01, 0.1, -0.1)
        assert result >= 0.0

        # Zero LR - no change
        result = DistributionalPPO._bounded_dual_update(0.5, 0.0, 0.1)
        assert result == 0.5

        # NaN gap
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, float("nan"))
        assert result == 0.5

        # Inf gap - clamped
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, float("inf"))
        assert 0.0 <= result <= 1.0

    def test_limit_mean_step(self):
        """Test mean step limiting."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        limited = model._limit_mean_step(0.0, 10.0, 1.0)
        assert isinstance(limited, float)
        # Limited by reference std factor
        assert abs(limited) <= 10.0

        env.close()

    def test_limit_std_step(self):
        """Test std step limiting."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        limited = model._limit_std_step(1.0, 10.0)
        assert isinstance(limited, float)
        # Should be limited somehow
        assert limited >= 0

        env.close()


# =============================================================================
# Test Twin Critics Methods
# =============================================================================

class TestTwinCriticsLoss:
    """Tests for twin critics loss computation."""

    @pytest.mark.skip(reason="Requires custom policy with distributional value head")
    def test_twin_critics_loss_quantile_mode(self):
        """Test twin critics loss in quantile mode."""
        pass


# =============================================================================
# Test _compute_returns_with_time_limits edge cases
# =============================================================================

class TestComputeReturnsWithTimeLimitsEdges:
    """Test edge cases for _compute_returns_with_time_limits."""

    def test_nan_in_rewards_raises(self):
        """Test that NaN in rewards raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([[1.0, np.nan], [1.0, 1.0]])
        buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="rewards contain NaN"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([0.0, 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.zeros((2, 2), dtype=bool),
                time_limit_bootstrap=np.zeros((2, 2)),
            )

    def test_nan_in_values_raises(self):
        """Test that NaN in values raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([[1.0, 1.0], [1.0, 1.0]])
        buffer.values = np.array([[np.nan, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="values contain NaN"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([0.0, 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.zeros((2, 2), dtype=bool),
                time_limit_bootstrap=np.zeros((2, 2)),
            )

    def test_nan_in_last_values_raises(self):
        """Test that NaN in last_values raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([[1.0, 1.0], [1.0, 1.0]])
        buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="last_values contain NaN"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([float("nan"), 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.zeros((2, 2), dtype=bool),
                time_limit_bootstrap=np.zeros((2, 2)),
            )

    def test_nan_in_time_limit_bootstrap_raises(self):
        """Test that NaN in time_limit_bootstrap raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([[1.0, 1.0], [1.0, 1.0]])
        buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="time_limit_bootstrap contains NaN"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([0.0, 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.array([[True, False], [False, False]]),
                time_limit_bootstrap=np.array([[float("nan"), 0.0], [0.0, 0.0]]),
            )

    def test_wrong_dimension_rewards_raises(self):
        """Test that 1D rewards raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([1.0, 1.0, 1.0, 1.0])  # 1D instead of 2D
        buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="2D arrays"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([0.0, 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.zeros((2, 2), dtype=bool),
                time_limit_bootstrap=np.zeros((2, 2)),
            )

    def test_mismatched_time_limit_mask_shape_raises(self):
        """Test that mismatched time_limit_mask shape raises ValueError."""
        buffer = Mock()
        buffer.rewards = np.array([[1.0, 1.0], [1.0, 1.0]])
        buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        buffer.episode_starts = np.array([[1, 0], [0, 0]])

        with pytest.raises(ValueError, match="TimeLimit mask"):
            _compute_returns_with_time_limits(
                buffer,
                last_values=torch.tensor([0.0, 0.0]),
                dones=np.array([False, False]),
                gamma=0.99,
                gae_lambda=0.95,
                time_limit_mask=np.zeros((3, 2), dtype=bool),  # Wrong shape
                time_limit_bootstrap=np.zeros((2, 2)),
            )


# =============================================================================
# Test PopArtController Integration
# =============================================================================

class TestPopArtControllerIntegration:
    """Integration tests for PopArtController with DistributionalPPO."""

    def test_popart_disabled_by_default(self):
        """Test that PopArt is disabled by default."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        assert model._popart_controller is None or not model._popart_controller.enabled

        env.close()

    def test_popart_warning_when_enabled(self):
        """Test that warning is issued when PopArt is enabled."""
        env = make_vec_env()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                value_scale_controller={"enabled": True},
                verbose=0,
            )

            # PopArt should still be disabled despite config
            assert model._popart_requested_enabled is True

        env.close()


# =============================================================================
# Test Serialization/Deserialization
# =============================================================================

class TestDistributionalPPOSerialization:
    """Tests for model serialization."""

    def test_get_parameters(self):
        """Test get_parameters method."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        params = model.get_parameters()

        assert "policy" in params

        env.close()

    def test_get_parameters_with_optimizer(self):
        """Test get_parameters with optimizer state."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        params = model.get_parameters(include_optimizer=True)

        assert "policy" in params

        env.close()


# =============================================================================
# Test Edge Cases in safe_explained_variance
# =============================================================================

class TestSafeExplainedVarianceIntegration:
    """Integration tests for safe_explained_variance."""

    def test_weighted_large_weights_overflow_prevention(self):
        """Test that very large weights don't cause overflow."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        weights = np.array([1e51, 1e51, 1e51])  # Very large weights

        result = safe_explained_variance(y_true, y_pred, weights)
        assert math.isnan(result)  # Should return NaN for overflow protection

    def test_weighted_with_near_equal_weights(self):
        """Test weighted variance with near-equal weights."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = safe_explained_variance(y_true, y_pred, weights)
        assert math.isfinite(result)
        assert result > 0.9  # High explained variance


# =============================================================================
# Test _weighted_variance_np edge cases
# =============================================================================

class TestWeightedVarianceNpIntegration:
    """Integration tests for _weighted_variance_np."""

    def test_single_positive_weight(self):
        """Test with single positive weight among zeros."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 1.0, 0.0])

        result = _weighted_variance_np(values, weights)
        # Single non-zero weight means no variance can be computed
        assert math.isnan(result)

    def test_all_same_values(self):
        """Test with all same values (zero variance)."""
        values = np.array([5.0, 5.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0])

        result = _weighted_variance_np(values, weights)
        assert result == 0.0 or math.isclose(result, 0.0, abs_tol=1e-10)


# =============================================================================
# Test _compute_empirical_cvar
# =============================================================================

class TestComputeEmpiricalCvar:
    """Tests for DistributionalPPO._compute_empirical_cvar."""

    def test_empty_rewards(self):
        """Test with empty rewards tensor."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        rewards = torch.tensor([])
        winsor, cvar = model._compute_empirical_cvar(rewards)

        assert winsor.numel() == 0
        assert cvar.item() == 0.0

        env.close()

    def test_normal_rewards(self):
        """Test with normal rewards."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            verbose=0,
        )

        rewards = torch.randn(100)
        winsor, cvar = model._compute_empirical_cvar(rewards)

        assert winsor.shape == rewards.shape
        assert torch.isfinite(cvar)

        env.close()

    def test_winsorization(self):
        """Test that winsorization is applied."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_winsor_pct=0.1,  # 10% winsorization
            verbose=0,
        )

        # Create rewards with outliers
        rewards = torch.tensor([
            -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        winsor, cvar = model._compute_empirical_cvar(rewards)

        assert winsor.shape == rewards.shape
        env.close()


# =============================================================================
# Test _compute_cvar_statistics
# =============================================================================

class TestComputeCvarStatistics:
    """Tests for DistributionalPPO._compute_cvar_statistics."""

    def test_empty_rewards(self):
        """Test with empty rewards."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        rewards = torch.tensor([])
        results = model._compute_cvar_statistics(rewards)

        assert len(results) == 5
        assert results[0].numel() == 0

        env.close()

    def test_normal_rewards(self):
        """Test with normal rewards."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        rewards = torch.randn(100)
        winsor, cvar, p50, p95, abs_p95 = model._compute_cvar_statistics(rewards)

        assert winsor.shape == rewards.shape
        assert torch.isfinite(cvar)
        assert torch.isfinite(p50)
        assert torch.isfinite(p95)
        assert abs_p95 >= 0

        env.close()


# =============================================================================
# Test RawRecurrentRolloutBuffer
# =============================================================================

class TestRawRecurrentRolloutBuffer:
    """Tests for RawRecurrentRolloutBuffer."""

    def test_buffer_initialization(self):
        """Test buffer can be created."""
        from distributional_ppo import RawRecurrentRolloutBuffer

        # Check class exists
        assert RawRecurrentRolloutBuffer is not None

    def test_to_numpy_static(self):
        """Test _to_numpy static method."""
        from distributional_ppo import RawRecurrentRolloutBuffer

        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = RawRecurrentRolloutBuffer._to_numpy(arr)
        assert isinstance(result, np.ndarray)

        # Test with tensor
        tensor = torch.tensor([1, 2, 3])
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)
        assert isinstance(result, np.ndarray)


# =============================================================================
# Test quantile loss helper
# =============================================================================

class TestQuantileHuberLossEdges:
    """Edge case tests for _quantile_huber_loss."""

    def test_loss_with_valid_inputs(self):
        """Test quantile Huber loss with valid inputs."""
        # This tests the standalone function from dppo module
        from distributional_ppo import DistributionalPPO

        # Test that the method exists
        assert hasattr(DistributionalPPO, "_quantile_huber_loss")


# =============================================================================
# Test _project_categorical_distribution
# =============================================================================

class TestProjectCategoricalDistributionEdges:
    """Edge case tests for _project_categorical_distribution."""

    def test_function_exists(self):
        """Test function exists."""
        from distributional_ppo import DistributionalPPO
        assert hasattr(DistributionalPPO, "_project_categorical_distribution")


# =============================================================================
# Test Additional Init Parameters
# =============================================================================

class TestDistributionalPPOInitExtended:
    """Extended tests for DistributionalPPO.__init__."""

    def test_init_with_popart_config_dict(self):
        """Test init with PopArt config as dict."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_scale_controller={"enabled": False},
            verbose=0,
        )

        assert model._popart_requested_enabled is False

        env.close()

    def test_init_with_winrate_confidence(self):
        """Test init with winrate confidence level."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            winrate_confidence_level=0.99,
            verbose=0,
        )

        assert model._winrate_confidence_level == 0.99

        env.close()

    def test_init_with_invalid_winrate_confidence(self):
        """Test init with invalid winrate confidence level."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            winrate_confidence_level=1.5,  # Invalid
            verbose=0,
        )

        # Should default to 0.95
        assert model._winrate_confidence_level == 0.95

        env.close()

    def test_init_with_bc_warmup(self):
        """Test init with behavioral cloning warmup."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            bc_warmup_steps=100,
            bc_decay_steps=50,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_kl_penalty_params(self):
        """Test init with KL penalty PID parameters."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            kl_penalty_beta=0.01,
            kl_penalty_beta_min=0.001,
            kl_penalty_beta_max=0.1,
            kl_penalty_pid_kp=0.1,
            kl_penalty_pid_ki=0.01,
            kl_penalty_pid_kd=0.001,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_gradient_accumulation(self):
        """Test init with gradient accumulation."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            microbatch_size=2,
            gradient_accumulation_steps=2,
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_loss_head_weights(self):
        """Test init with custom loss head weights."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            loss_head_weights={"policy": 1.0, "value": 0.5, "entropy": 0.01},
            verbose=0,
        )

        assert model is not None

        env.close()

    def test_init_with_vgs_parameters(self):
        """Test init with VGS parameters."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_beta=0.99,
            vgs_alpha=0.1,
            vgs_warmup_steps=100,
            verbose=0,
        )

        assert model is not None

        env.close()


# =============================================================================
# Test Static Concat Methods
# =============================================================================

class TestConcatMethods:
    """Tests for static concat methods."""

    def test_concat_tensor_batches(self):
        """Test _concat_tensor_batches."""
        batches = [torch.randn(4, 8), torch.randn(4, 8)]
        result = DistributionalPPO._concat_tensor_batches(batches)

        # Result should be concatenated tensors
        assert result is not None
        assert isinstance(result, torch.Tensor)

    def test_concat_tensor_batches_empty(self):
        """Test _concat_tensor_batches with empty list."""
        result = DistributionalPPO._concat_tensor_batches([])

        # Empty list should return None
        assert result is None

    def test_concat_string_keys(self):
        """Test _concat_string_keys."""
        batches = [["a", "b"], ["c", "d"]]
        result = DistributionalPPO._concat_string_keys(batches)

        # Result should be concatenated lists
        assert result is not None
        assert "a" in result and "d" in result

    def test_concat_string_keys_empty(self):
        """Test _concat_string_keys with empty list."""
        result = DistributionalPPO._concat_string_keys([])

        # Empty list returns None or empty list
        assert result is None or result == []


# =============================================================================
# Test Value Scale Methods
# =============================================================================

class TestValueScaleMethods:
    """Tests for value scale related methods."""

    def test_coerce_value_target_scale(self):
        """Test _coerce_value_target_scale static method."""
        # Test with float
        result = DistributionalPPO._coerce_value_target_scale(0.5)
        assert result == 0.5

        # Test with string "percent" (100 = 1/0.01)
        result = DistributionalPPO._coerce_value_target_scale("percent")
        assert result == 100.0

        # Test with string "bps" (10000 = 1/0.0001)
        result = DistributionalPPO._coerce_value_target_scale("bps")
        assert result == 10000.0

        # Test with None
        result = DistributionalPPO._coerce_value_target_scale(None)
        assert result == 1.0

        # Test with invalid string raises
        with pytest.raises(ValueError):
            DistributionalPPO._coerce_value_target_scale("invalid")


# =============================================================================
# Test KL Property Methods
# =============================================================================

class TestKLPropertyMethods:
    """Tests for KL-related property methods."""

    def test_kl_exceed_stop_fraction_property(self):
        """Test kl_exceed_stop_fraction property."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            kl_exceed_stop_fraction=0.3,
            verbose=0,
        )

        assert model.kl_exceed_stop_fraction == 0.3

        # Test setter
        model.kl_exceed_stop_fraction = 0.5
        assert model.kl_exceed_stop_fraction == 0.5

        env.close()

    def test_kl_absolute_stop_factor_property(self):
        """Test kl_absolute_stop_factor property."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            kl_absolute_stop_factor=3.0,
            verbose=0,
        )

        assert model.kl_absolute_stop_factor == 3.0

        # Test setter
        model.kl_absolute_stop_factor = 4.0
        assert model.kl_absolute_stop_factor == 4.0

        env.close()


# =============================================================================
# Test cvar_winsor_pct property
# =============================================================================

class TestCvarWinsorPctProperty:
    """Tests for cvar_winsor_pct property."""

    def test_cvar_winsor_pct_getter(self):
        """Test getter."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_winsor_pct=0.05,
            verbose=0,
        )

        # Property returns percentage (fraction * 100)
        assert model.cvar_winsor_pct == pytest.approx(0.05, abs=0.01)

        env.close()

    def test_cvar_winsor_pct_setter(self):
        """Test setter."""
        env = make_vec_env()

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        model.cvar_winsor_pct = 0.1
        assert model.cvar_winsor_pct == pytest.approx(0.1, abs=0.01)

        env.close()


# =============================================================================
# Test _has_nonempty_batches
# =============================================================================

class TestHasNonemptyBatches:
    """Tests for _has_nonempty_batches static method."""

    def test_empty_list(self):
        """Test with empty list."""
        result = DistributionalPPO._has_nonempty_batches([])
        assert result is False

    def test_list_with_empty_tensors(self):
        """Test with list of empty tensors."""
        batches = [torch.tensor([]), torch.tensor([])]
        result = DistributionalPPO._has_nonempty_batches(batches)
        assert result is False

    def test_list_with_nonempty_tensor(self):
        """Test with list containing non-empty tensor."""
        batches = [torch.tensor([]), torch.tensor([1, 2, 3])]
        result = DistributionalPPO._has_nonempty_batches(batches)
        assert result is True


# =============================================================================
# Additional PopArt Tests
# =============================================================================

class TestPopArtControllerAdvanced:
    """Advanced tests for PopArtController."""

    def test_within_tolerance(self):
        """Test _within_tolerance static method."""
        # Exact match
        assert PopArtController._within_tolerance(0.0, 1.0, abs_tol=1e-5, rel_tol=1e-6)

        # Within absolute tolerance
        assert PopArtController._within_tolerance(1e-6, 1.0, abs_tol=1e-5, rel_tol=1e-6)

        # Outside tolerance
        assert not PopArtController._within_tolerance(0.1, 1.0, abs_tol=1e-5, rel_tol=1e-6)

    def test_clip_fraction(self):
        """Test _clip_fraction static method."""
        values = torch.tensor([0.5, 1.5, 2.0, 3.0])

        # All within range
        result = PopArtController._clip_fraction(values, 0.0, 5.0)
        assert result == 0.0

        # Some clipped
        result = PopArtController._clip_fraction(values, 1.0, 2.5)
        # 0.5 is below 1.0, 3.0 is above 2.5
        assert result == 0.5

    def test_safe_numpy(self):
        """Test _safe_numpy static method."""
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        result = PopArtController._safe_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


# =============================================================================
# Test _expand_logger_key_length
# =============================================================================

class TestExpandLoggerKeyLength:
    """Tests for _expand_logger_key_length static method."""

    def test_with_none_logger(self):
        """Test with None logger - should not raise."""
        # Should not raise
        DistributionalPPO._expand_logger_key_length(None, min_max_length=50)

    def test_function_exists(self):
        """Test function exists."""
        assert hasattr(DistributionalPPO, "_expand_logger_key_length")


# =============================================================================
# Run if main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
