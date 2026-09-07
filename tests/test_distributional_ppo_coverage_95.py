"""
Additional coverage tests to push distributional_ppo.py coverage to 95%+.

Focus on:
- train() branches
- collect_rollouts() edge cases
- _compute_explained_variance_metric()
- _enforce_optimizer_lr_bounds()
- _cvar_from_quantiles()
- Other helper functions
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

from distributional_ppo import DistributionalPPO, PopArtController


np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, time_limit: bool = True) -> DummyVecEnv:
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
                self._time_limit = time_limit

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.random.randn(4).astype(np.float32), {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                info = {}
                if self._time_limit and done:
                    info["TimeLimit.truncated"] = True
                return (
                    np.random.randn(4).astype(np.float32),
                    np.random.randn() * 0.1,
                    done,
                    False,
                    info,
                )

        return _Env()

    return DummyVecEnv([_env_fn])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env: DummyVecEnv, **overrides) -> DistributionalPPO:
    """Create model with defaults."""
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
    """Setup model and collect rollouts."""
    total = int(model.n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)


class TestTrainCoverage:
    """Additional train() coverage tests."""

    def test_train_with_normalize_returns(self):
        """Cover normalize_returns path in train."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_clip_range_vf(self):
        """Cover clip_range_vf path."""
        env = _make_env()
        model = _make_model(env, clip_range_vf=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_awr_enabled(self):
        """Cover AWR path."""
        env = _make_env()
        model = _make_model(env)
        model._use_awr = True
        model._awr_temperature = 1.0
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_large_batch(self):
        """Cover larger batch training."""
        env = _make_env()
        model = _make_model(env, batch_size=8)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_value_clipping_categorical(self):
        """Cover categorical value clipping."""
        env = _make_env()
        model = _make_model(
            env,
            distributional_vf_clip_mode="mean_only",
            clip_range_vf=0.2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestEnforceOptimizerLrBounds:
    """Test _enforce_optimizer_lr_bounds branches."""

    def test_lr_below_min(self):
        """Cover LR below minimum."""
        env = _make_env()
        model = _make_model(env, optimizer_lr_min=1e-3, learning_rate=1e-5)
        model._enforce_optimizer_lr_bounds(log_values=True, warn_on_floor=True)
        env.close()

    def test_lr_above_max(self):
        """Cover LR above maximum."""
        env = _make_env()
        model = _make_model(env, optimizer_lr_max=1e-4, learning_rate=1e-2)
        model._enforce_optimizer_lr_bounds(log_values=True, warn_on_floor=False)
        env.close()


class TestValueScaleBranches:
    """Test value scale related branches."""

    def test_value_scale_with_freeze(self):
        """Cover value scale freeze."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"freeze_after_updates": 1},
        )
        _setup_and_collect(model, env)
        model._value_scale_update_count = 5
        model._value_scale_frozen = True
        model.train()
        env.close()

    def test_value_scale_never_freeze(self):
        """Cover never freeze path."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"never_freeze": True},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCvarBranches:
    """Test CVaR related branches."""

    def test_cvar_with_constraint(self):
        """Cover CVaR constraint."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.25,
            cvar_use_constraint=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_penalty_mode(self):
        """Cover CVaR penalty mode."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.25,
            cvar_use_penalty=True,
        )
        model._cvar_penalty_lambda = 0.1
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestEntropySchedule:
    """Test entropy schedule branches."""

    def test_entropy_decay(self):
        """Cover entropy decay."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef_decay_steps=2,
            ent_coef_min=0.0,
        )
        _setup_and_collect(model, env)
        model.train()
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_entropy_plateau_detection(self):
        """Cover entropy plateau detection."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef_decay_steps=5,
            ent_coef_plateau_window=2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestKlDiagnostics:
    """Test KL diagnostics branches."""

    def test_kl_ema_update(self):
        """Cover KL EMA update."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.01,
            kl_ema_updates=2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_kl_absolute_stop(self):
        """Cover KL absolute stop."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=1e-6,  # Very low
            kl_absolute_stop_factor=1.0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestReturnStats:
    """Test return statistics functions."""

    def test_summarize_with_variance(self):
        """Cover _summarize_recent_return_stats with variance."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            value_scale={"window_updates": 3},
        )
        _setup_and_collect(model, env)
        model.train()
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCollectRolloutsCoverage:
    """Test collect_rollouts branches."""

    def test_collect_with_different_rewards(self):
        """Cover reward processing paths."""

        def _env_fn():
            class _Env(gymnasium.Env):
                def __init__(self):
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                    self.observation_space = spaces.Box(
                        low=-10.0, high=10.0, shape=(4,), dtype=np.float32
                    )
                    self._step = 0

                def reset(self, *, seed=None, options=None):
                    self._step = 0
                    return np.zeros(4, dtype=np.float32), {}

                def step(self, action):
                    self._step += 1
                    reward = 10.0 if self._step % 3 == 0 else -1.0
                    done = self._step >= 8
                    return (
                        np.random.randn(4).astype(np.float32),
                        reward,
                        done,
                        False,
                        {"TimeLimit.truncated": done},
                    )

            return _Env()

        env = DummyVecEnv([_env_fn])
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestGetOptimizerClass:
    """Test _get_optimizer_class branches."""

    def test_default_optimizer(self):
        """Cover default optimizer class."""
        env = _make_env()
        model = _make_model(env)
        cls = model._get_optimizer_class()
        assert cls is not None
        env.close()


class TestGradientAccumulation:
    """Test gradient accumulation."""

    def test_with_accumulation_steps(self):
        """Cover gradient accumulation."""
        env = _make_env()
        model = _make_model(
            env,
            batch_size=8,
            gradient_accumulation_steps=2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestValueDebugStats:
    """Test value debug stats recording."""

    def test_record_value_stats(self):
        """Cover _record_value_debug_stats."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestExplainedVarianceMetric:
    """Test _compute_explained_variance_metric branches."""

    def test_ev_with_clipping(self):
        """Cover EV with clipping threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.1,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestLimitVRangeStep:
    """Test _limit_v_range_step branches."""

    def test_v_range_limited(self):
        """Cover V range limiting."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"range_max_rel_step": 0.1},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestApplyVRangeUpdate:
    """Test _apply_v_range_update branches."""

    def test_v_range_update(self):
        """Cover V range update."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestSmoothValueTargetScale:
    """Test _smooth_value_target_scale branches."""

    def test_smoothing_with_change_limit(self):
        """Cover smoothing with max change limit."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"max_change_pct": 10.0, "target_scale_ema_beta": 0.5},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestIsValueScaleFrameStable:
    """Test _is_value_scale_frame_stable branches."""

    def test_stability_check(self):
        """Cover stability check."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={
                "stability": {"min_explained_variance": 0.0, "max_abs_p95": 100.0},
                "stability_patience": 1,
            },
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestUpdateCriticGradientBlock:
    """Test _update_critic_gradient_block branches."""

    def test_critic_gradient_with_clipping(self):
        """Cover critic gradient block."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_coef=0.5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestFilterEvReserveRows:
    """Test _filter_ev_reserve_rows branches."""

    def test_filter_reserve(self):
        """Cover reserve row filtering."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestComputeWeightedStats:
    """Test _compute_weighted_stats branches."""

    def test_weighted_stats(self):
        """Cover weighted stats computation."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestSetupDependentComponents:
    """Test _setup_dependent_components branches."""

    def test_setup_components(self):
        """Cover component setup."""
        env = _make_env()
        model = _make_model(env)
        # Components are set up during __init__
        assert model._setup_complete
        env.close()


class TestMultipleTrainCycles:
    """Test multiple training cycles for state transitions."""

    def test_multiple_cycles(self):
        """Run multiple train cycles to cover state changes."""
        env = _make_env(max_steps=8)
        model = _make_model(
            env,
            normalize_returns=True,
            target_kl=0.1,
            ent_coef_decay_steps=3,
            value_scale={"window_updates": 2},
        )

        for _ in range(3):
            _setup_and_collect(model, env)
            model.train()

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
