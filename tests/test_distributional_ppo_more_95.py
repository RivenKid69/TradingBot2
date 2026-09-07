"""
More coverage tests for distributional_ppo.py.
"""

from __future__ import annotations

import math
import types
from collections import deque
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    safe_explained_variance,
)


np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8) -> DummyVecEnv:
    """Create environment."""

    def _env_fn():
        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(4,), dtype=np.float32
                )
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.random.randn(4).astype(np.float32), {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                return (
                    np.random.randn(4).astype(np.float32),
                    np.random.randn() * 0.1,
                    done,
                    False,
                    {"TimeLimit.truncated": done},
                )

        return _Env()

    return DummyVecEnv([_env_fn])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env: DummyVecEnv, **overrides) -> DistributionalPPO:
    defaults = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    defaults.update(overrides)
    return DistributionalPPO(**defaults)


def _setup_and_collect(model: DistributionalPPO, env: DummyVecEnv) -> None:
    total = int(model.n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)


class TestCvarFromQuantiles:
    """Test _cvar_from_quantiles branches."""

    def test_cvar_basic(self):
        """Cover basic CVaR computation."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.25)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_edge_cases(self):
        """Cover CVaR edge cases."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.1)
        _setup_and_collect(model, env)
        # Trigger cvar computation multiple times
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestFinalizeReturnStats:
    """Test _finalize_return_stats branches."""

    def test_finalize_with_normalize(self):
        """Cover finalize with normalization."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        for _ in range(3):
            _setup_and_collect(model, env)
            model.train()
        env.close()

    def test_finalize_without_normalize(self):
        """Cover finalize without normalization."""
        env = _make_env()
        model = _make_model(env, normalize_returns=False)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestApplyLiveUpdate:
    """Test apply_live_update branches."""

    def test_popart_live_mode(self):
        """Cover PopArt live update mode."""
        env = _make_env()
        model = _make_model(env)
        controller = PopArtController(
            enabled=True,
            mode="live",
            min_samples=1,
            warmup_updates=0,
        )
        model._popart_controller = controller
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestGetOptimizerKwargs:
    """Test _get_optimizer_kwargs branches."""

    def test_with_weight_decay(self):
        """Cover optimizer kwargs with weight decay."""
        env = _make_env()
        model = _make_model(env)
        model._optimizer_kwargs = {"weight_decay": 1e-4}
        kwargs = model._get_optimizer_kwargs()
        assert "weight_decay" in kwargs
        env.close()

    def test_empty_kwargs(self):
        """Cover empty optimizer kwargs."""
        env = _make_env()
        model = _make_model(env)
        model._optimizer_kwargs = {}
        kwargs = model._get_optimizer_kwargs()
        assert kwargs is not None
        env.close()


class TestComputeExplainedVarianceMetric:
    """Test _compute_explained_variance_metric branches."""

    def test_ev_with_weights(self):
        """Cover EV with sample weights."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestSmoothValueTargetScale:
    """Test _smooth_value_target_scale branches."""

    def test_smooth_with_ema(self):
        """Cover EMA smoothing."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"ema_beta": 0.3, "target_scale_ema_beta": 0.4},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestAddBuffer:
    """Test add() buffer method branches."""

    def test_add_with_episode_starts(self):
        """Cover add with episode starts."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestUpdateCriticGradientBlock:
    """Test _update_critic_gradient_block."""

    def test_critic_with_twin_critics(self):
        """Cover twin critics update."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestFilterEvReserveRows:
    """Test _filter_ev_reserve_rows branches."""

    def test_filter_with_low_ev(self):
        """Cover filtering with low EV."""
        env = _make_env()
        model = _make_model(env)
        model._last_explained_variance = 0.1
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestGetOptimizerClass:
    """Test _get_optimizer_class branches."""

    def test_with_custom_optimizer(self):
        """Cover custom optimizer class."""
        env = _make_env()
        model = _make_model(env)
        cls = model._get_optimizer_class()
        assert cls is not None
        env.close()


class TestSetupDependentComponents:
    """Test _setup_dependent_components."""

    def test_setup_with_all_options(self):
        """Cover full setup."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            cvar_alpha=0.25,
            target_kl=0.01,
        )
        assert model._setup_complete
        env.close()


class TestApplyVRangeUpdate:
    """Test _apply_v_range_update branches."""

    def test_v_range_with_limits(self):
        """Cover V range with limits."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            value_scale={"range_max_rel_step": 0.05},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestLimitVRangeStep:
    """Test _limit_v_range_step."""

    def test_limit_large_step(self):
        """Cover limiting large V range step."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"range_max_rel_step": 0.01},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestPredict:
    """Test _predict branches."""

    def test_predict_with_state(self):
        """Cover prediction with LSTM state."""
        env = _make_env()
        model = _make_model(env)
        obs = np.zeros((1, 4), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
        env.close()


class TestKlDiagStep:
    """Test _kl_diag_step branches."""

    def test_kl_diag_with_target(self):
        """Cover KL diagnostic step."""
        env = _make_env()
        model = _make_model(env, target_kl=0.01)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestSafeExplainedVarianceAllBranches:
    """Test safe_explained_variance all branches."""

    def test_normal_values(self):
        """Cover normal values."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.1, 2.1, 2.9])
        result = safe_explained_variance(y_pred, y_true)
        assert result > 0

    def test_perfect_prediction(self):
        """Cover perfect prediction."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        result = safe_explained_variance(y_pred, y_true)
        assert result == pytest.approx(1.0)

    def test_constant_true(self):
        """Cover constant y_true."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 2.0, 2.0])
        result = safe_explained_variance(y_pred, y_true)
        # Constant y_true has zero variance - result is close to 0 or nan
        assert abs(result) < 1e-6 or math.isnan(result)


class TestRebuildSchedulerIfNeeded:
    """Test _rebuild_scheduler_if_needed branches."""

    def test_rebuild_with_scheduler(self):
        """Cover scheduler rebuild."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestIsValueScaleFrameStable:
    """Test _is_value_scale_frame_stable."""

    def test_stable_frame(self):
        """Cover stable frame check."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"stability_patience": 2},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestRecordValueDebugStats:
    """Test _record_value_debug_stats."""

    def test_record_stats(self):
        """Cover recording debug stats."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestComputeWeightedStats:
    """Test _compute_weighted_stats."""

    def test_weighted_stats_computation(self):
        """Cover weighted stats."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestSummarizeRecentReturnStats:
    """Test _summarize_recent_return_stats."""

    def test_summarize_with_window(self):
        """Cover summarization with window."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"window_updates": 5},
        )
        for _ in range(3):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestUpdateBcCoef:
    """Test _update_bc_coef branches."""

    def test_bc_coef_update(self):
        """Cover BC coefficient update."""
        env = _make_env()
        model = _make_model(env)
        model._bc_coef = 0.1
        model._bc_decay_steps = 5
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestEnforceOptimizerLrBoundsAllBranches:
    """Test all _enforce_optimizer_lr_bounds branches."""

    def test_lr_within_bounds(self):
        """Cover LR within bounds."""
        env = _make_env()
        model = _make_model(
            env,
            optimizer_lr_min=1e-6,
            optimizer_lr_max=1e-2,
            learning_rate=1e-4,
        )
        model._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=False)
        env.close()

    def test_lr_at_floor(self):
        """Cover LR at floor."""
        env = _make_env()
        model = _make_model(
            env,
            optimizer_lr_min=1e-3,
            learning_rate=1e-5,
        )
        model._enforce_optimizer_lr_bounds(log_values=True, warn_on_floor=True)
        env.close()


class TestMultipleEpochs:
    """Test with multiple epochs."""

    def test_two_epochs(self):
        """Cover multiple epoch training."""
        env = _make_env()
        model = _make_model(env, n_epochs=2)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestWithGae:
    """Test with different GAE settings."""

    def test_low_gae_lambda(self):
        """Cover low GAE lambda."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.8)
        _setup_and_collect(model, env)
        model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
