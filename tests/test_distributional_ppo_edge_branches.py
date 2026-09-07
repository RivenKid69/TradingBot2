"""
Targeted tests for edge branches in distributional_ppo.py to push coverage to 95%+.
Focuses on:
- Twin critics VF clipping modes
- Optimizer class resolution
- Advantage normalization edge cases
- EV filtering paths
"""

from __future__ import annotations

import math
import types
from collections import deque
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

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
    """Create simple test environment."""

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
                    np.random.randn() * 0.5 - 0.1,
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


class TestVfClippingModes:
    """Test VF clipping mode branches."""

    def test_vf_clip_mean_only_mode(self):
        """Cover mean_only mode for VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            distributional_vf_clip_mode="mean_only",
            clip_range_vf=0.2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mean_and_variance_mode(self):
        """Cover mean_and_variance mode."""
        env = _make_env()
        model = _make_model(
            env,
            distributional_vf_clip_mode="mean_and_variance",
            clip_range_vf=0.2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_per_quantile_mode(self):
        """Cover per_quantile mode."""
        env = _make_env()
        model = _make_model(
            env,
            distributional_vf_clip_mode="per_quantile",
            clip_range_vf=0.2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_with_normalize_returns(self):
        """Cover VF clipping with normalize_returns enabled."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            clip_range_vf=0.15,
            distributional_vf_clip_mode="mean_only",
        )
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestOptimizerClassResolution:
    """Test _get_optimizer_class branches."""

    def test_default_adamw(self):
        """Cover default AdamW optimizer."""
        env = _make_env()
        model = _make_model(env)
        # The optimizer class should be available
        assert model.policy.optimizer is not None
        env.close()

    def test_with_high_learning_rate(self):
        """Cover with high learning rate."""
        env = _make_env()
        model = _make_model(env, learning_rate=1e-2)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_with_low_learning_rate(self):
        """Cover with low learning rate."""
        env = _make_env()
        model = _make_model(env, learning_rate=1e-6)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_with_optimizer_kwargs(self):
        """Cover with optimizer kwargs."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestAdvantageNormalizationEdgeCases:
    """Test advantage normalization edge cases."""

    def test_advantages_with_zero_std(self):
        """Cover advantage normalization with zero std."""
        env = _make_env()
        model = _make_model(env, normalize_advantage=True)
        _setup_and_collect(model, env)
        # Manipulate buffer to have zero-std advantages
        if hasattr(model.rollout_buffer, "advantages"):
            model.rollout_buffer.advantages = np.ones_like(model.rollout_buffer.advantages) * 0.1
        model.train()
        env.close()

    def test_advantages_near_zero(self):
        """Cover advantage normalization with values near zero."""
        env = _make_env()
        model = _make_model(env, normalize_advantage=True)
        _setup_and_collect(model, env)
        if hasattr(model.rollout_buffer, "advantages"):
            model.rollout_buffer.advantages = np.random.randn(8, 1).astype(np.float32) * 1e-8
        model.train()
        env.close()


class TestEvFilteringPaths:
    """Test explained variance filtering paths."""

    def test_ev_with_filter_reserve_rows(self):
        """Cover EV filtering with reserve rows."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        model._ev_filter_reserve_frac = 0.2
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_ev_with_low_variance_target(self):
        """Cover EV with low variance in target values."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        # Check if EV was computed
        assert model._last_explained_variance is not None
        env.close()


class TestValueScaleEdgeCases:
    """Test value scale edge cases."""

    def test_value_scale_with_outlier_detection(self):
        """Cover value scale with outlier detection."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={
                "outlier_clip_factor": 3.0,
                "target_scale_clip": 100.0,
            },
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_warmup_period(self):
        """Cover warmup period in value scale."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"warmup_updates": 2},
        )
        for _ in range(4):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestPopArtControllerEdgeCases:
    """Test PopArtController edge cases."""

    def test_popart_controller_creation(self):
        """Cover PopArt controller creation."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            warmup_updates=0,
            min_samples=1,
        )
        assert controller.enabled == True

    def test_popart_controller_disabled(self):
        """Cover disabled PopArt controller."""
        controller = PopArtController(
            enabled=False,
        )
        assert controller.enabled == False

    def test_popart_live_mode(self):
        """Cover PopArt live mode."""
        controller = PopArtController(
            enabled=True,
            mode="live",
            warmup_updates=0,
            min_samples=1,
        )
        assert controller.mode == "live"

    def test_popart_with_model(self):
        """Cover PopArt with model."""
        env = _make_env()
        model = _make_model(env)
        controller = PopArtController(
            enabled=True,
            mode="shadow",
        )
        model._popart_controller = controller
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestKlDivergenceEdgeCases:
    """Test KL divergence edge cases."""

    def test_kl_with_very_low_target(self):
        """Cover KL with very low target."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=1e-5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_kl_with_high_target(self):
        """Cover KL with high target."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestEntropyEdgeCases:
    """Test entropy coefficient edge cases."""

    def test_entropy_at_minimum(self):
        """Cover entropy at minimum value."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef=0.0,
            ent_coef_min=0.0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_entropy_decay_complete(self):
        """Cover complete entropy decay."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef=0.01,
            ent_coef_decay_steps=2,
            ent_coef_min=0.0,
        )
        for _ in range(5):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestGradientClippingEdgeCases:
    """Test gradient clipping edge cases."""

    def test_high_max_grad_norm(self):
        """Cover high max grad norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=10.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_low_max_grad_norm(self):
        """Cover low max grad norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfCoefEdgeCases:
    """Test VF coefficient edge cases."""

    def test_zero_vf_coef(self):
        """Cover zero VF coefficient."""
        env = _make_env()
        model = _make_model(env, vf_coef=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_high_vf_coef(self):
        """Cover high VF coefficient."""
        env = _make_env()
        model = _make_model(env, vf_coef=2.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestBatchSizeEdgeCases:
    """Test batch size edge cases."""

    def test_batch_size_equals_buffer(self):
        """Cover batch size equal to buffer size."""
        env = _make_env()
        model = _make_model(env, n_steps=8, batch_size=8)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestMultiEnvEdgeCases:
    """Test multi-environment edge cases."""

    def test_two_envs(self):
        """Cover two environments."""

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
                    return np.random.randn(4).astype(np.float32), {}

                def step(self, action):
                    self._step += 1
                    done = self._step >= 8
                    return (
                        np.random.randn(4).astype(np.float32),
                        np.random.randn(),
                        done,
                        False,
                        {"TimeLimit.truncated": done},
                    )

            return _Env()

        env = DummyVecEnv([_env_fn, _env_fn])
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestReturnClippingEdgeCases:
    """Test return clipping edge cases."""

    def test_ret_clip_low(self):
        """Cover low return clipping."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True, ret_clip=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_ret_clip_high(self):
        """Cover high return clipping."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True, ret_clip=100.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestGaeEdgeCases:
    """Test GAE edge cases."""

    def test_gae_lambda_zero(self):
        """Cover GAE lambda = 0 (TD(0))."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_gae_lambda_one(self):
        """Cover GAE lambda = 1 (MC)."""
        env = _make_env()
        model = _make_model(env, gae_lambda=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCvarExtreme:
    """Test CVaR extreme cases."""

    def test_cvar_alpha_very_low(self):
        """Cover very low CVaR alpha."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.01)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_alpha_high(self):
        """Cover high CVaR alpha (near 1)."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.9)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestLearningRateSchedule:
    """Test learning rate schedule edge cases."""

    def test_constant_lr(self):
        """Cover constant learning rate."""
        env = _make_env()
        model = _make_model(env, learning_rate=1e-4)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_very_low_lr(self):
        """Cover very low learning rate."""
        env = _make_env()
        model = _make_model(env, learning_rate=1e-7)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestClipRangeEdgeCases:
    """Test clip range edge cases."""

    def test_clip_range_low(self):
        """Cover low clip range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_clip_range_high(self):
        """Cover high clip range."""
        env = _make_env()
        model = _make_model(env, clip_range=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestNEpochsEdgeCases:
    """Test n_epochs edge cases."""

    def test_many_epochs(self):
        """Cover many epochs."""
        env = _make_env()
        model = _make_model(env, n_epochs=5)
        _setup_and_collect(model, env)
        model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
