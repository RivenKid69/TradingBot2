"""
Additional coverage tests for distributional_ppo.py - simplified version.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import (
    DistributionalPPO,
    safe_explained_variance,
    compute_grouped_explained_variance,
)

np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 10) -> DummyVecEnv:
    def _env_fn():
        import gymnasium

        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
                self.observation_space = spaces.Box(-10.0, 10.0, (4,), np.float32)
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                return np.zeros(4, dtype=np.float32), 0.1, done, False, {"TimeLimit.truncated": done}

        return _Env()

    return DummyVecEnv([_env_fn])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env, **kwargs):
    defaults = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    defaults.update(kwargs)
    return DistributionalPPO(**defaults)


def _setup_and_collect(model, env, n_steps=8):
    total = int(n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=n_steps)
    return model


class TestEnforceOptimizerLrBounds:
    """Test _enforce_optimizer_lr_bounds."""

    def test_lr_below_min(self):
        env = _make_env()
        model = _make_model(env, optimizer_lr_min=1e-3, optimizer_lr_max=1e-1)
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = 1e-6
        model._enforce_optimizer_lr_bounds(log_values=True, warn_on_floor=True)
        for pg in model.policy.optimizer.param_groups:
            assert pg["lr"] >= 1e-3
        env.close()

    def test_lr_above_max(self):
        env = _make_env()
        model = _make_model(env, optimizer_lr_min=1e-6, optimizer_lr_max=1e-3)
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = 1.0
        model._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=False)
        for pg in model.policy.optimizer.param_groups:
            assert pg["lr"] <= 1e-3
        env.close()


class TestSafeExplainedVarianceMore:
    """Additional safe_explained_variance tests."""

    def test_zero_variance(self):
        y_pred = np.ones(10)
        y_true = np.ones(10)
        result = safe_explained_variance(y_pred, y_true)
        assert isinstance(result, float)

    def test_with_weights(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.1, 2.1, 3.1])
        weights = np.array([1.0, 1.0, 1.0])
        result = safe_explained_variance(y_pred, y_true, weights=weights)
        assert isinstance(result, float)


class TestComputeGroupedExplainedVariance:
    """Test compute_grouped_explained_variance."""

    def test_grouped_ev(self):
        y_pred = np.random.randn(100)
        y_true = y_pred + np.random.randn(100) * 0.1
        groups = np.random.randint(0, 3, size=100)
        result = compute_grouped_explained_variance(y_pred, y_true, groups)
        assert result is not None


class TestResetLstmStates:
    """Test reset_lstm_states_to_initial."""

    def test_reset_lstm(self):
        env = _make_env()
        model = _make_model(env)
        if hasattr(model, "reset_lstm_states_to_initial"):
            model.reset_lstm_states_to_initial()
        env.close()


class TestFinalizeReturnStats:
    """Test _finalize_return_stats."""

    def test_finalize_stats(self):
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model._finalize_return_stats()
        env.close()


class TestSetupDependentComponents:
    """Test _setup_dependent_components."""

    def test_setup_with_popart(self):
        env = _make_env()
        model = _make_model(
            env,
            value_scale_controller={"enabled": True, "mode": "ema"},
        )
        env.close()


class TestTrainWithVGS:
    """Test train loop."""

    def test_train_basic(self):
        env = _make_env(max_steps=8)
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCollectRolloutsEdgeCases:
    """Test collect_rollouts."""

    def test_collect_basic(self):
        env = _make_env(max_steps=8)
        model = _make_model(env)
        _setup_and_collect(model, env)
        env.close()


class TestPredictWithState:
    """Test predict."""

    def test_predict_with_state_none(self):
        env = _make_env()
        model = _make_model(env)
        obs = np.zeros((1, 4), dtype=np.float32)
        action, state = model.predict(obs, state=None, deterministic=True)
        assert action is not None
        env.close()


class TestGetOptimizerClass:
    """Test _get_optimizer_class."""

    def test_get_optimizer_adamw(self):
        env = _make_env()
        model = _make_model(env, optimizer_class="AdamW")
        opt_class = model._get_optimizer_class()
        assert "Adam" in opt_class.__name__
        env.close()


class TestRebuildScheduler:
    """Test _rebuild_scheduler_if_needed."""

    def test_rebuild(self):
        env = _make_env()
        model = _make_model(env)
        model._rebuild_scheduler_if_needed()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
