"""
Tests for VF clipping paths in distributional_ppo.py.
Focuses on different VF clip modes and combinations.
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

from distributional_ppo import DistributionalPPO


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
                    np.random.randn() * 0.5,
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


class TestVfClippingQuantile:
    """Test VF clipping for quantile critics."""

    def test_vf_clip_per_quantile_enabled(self):
        """Cover VF clipping with per_quantile mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mean_only_enabled(self):
        """Cover VF clipping with mean_only mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mean_and_variance_enabled(self):
        """Cover VF clipping with mean_and_variance mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClippingWithNormalization:
    """Test VF clipping with return normalization."""

    def test_vf_clip_with_normalize_returns(self):
        """Cover VF clipping with normalize_returns."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()

    def test_vf_clip_per_quantile_with_normalize(self):
        """Cover per_quantile mode with normalize_returns."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            clip_range_vf=0.15,
            distributional_vf_clip_mode="per_quantile",
        )
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()

    def test_vf_clip_mean_variance_with_normalize(self):
        """Cover mean_and_variance mode with normalize_returns."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            clip_range_vf=0.15,
            distributional_vf_clip_mode="mean_and_variance",
        )
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestVfClipOptions:
    """Test different VF clip options."""

    def test_no_vf_clip(self):
        """Cover without VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=None,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_only(self):
        """Cover single critic with VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClipModes:
    """Test different VF clip modes."""

    def test_all_modes_sequentially(self):
        """Cover all VF clip modes sequentially."""
        modes = ["per_quantile", "mean_only", "mean_and_variance"]
        for mode in modes:
            env = _make_env()
            model = _make_model(
                env,
                clip_range_vf=0.2,
                distributional_vf_clip_mode=mode,
            )
            _setup_and_collect(model, env)
            model.train()
            env.close()

    def test_mode_with_high_clip_range(self):
        """Cover VF clip modes with high clip range."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mode_with_low_clip_range(self):
        """Cover VF clip modes with low clip range."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.05,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClipWarmup:
    """Test VF clipping warmup period."""

    def test_vf_clip_warmup_period(self):
        """Cover VF clipping warmup."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            vf_clip_warmup_updates=2,
        )
        # Train multiple times to pass warmup
        for _ in range(4):
            _setup_and_collect(model, env)
            model.train()
        env.close()

    def test_vf_clip_warmup_zero(self):
        """Cover VF clipping without warmup."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            vf_clip_warmup_updates=0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestValuePredictionClipping:
    """Test value prediction clipping paths."""

    def test_value_clip_limit(self):
        """Cover value clip limit path."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCriticLossAggregation:
    """Test critic loss aggregation paths."""

    def test_loss_aggregation(self):
        """Cover critic loss aggregation."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.15,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClipLogging:
    """Test VF clipping logging paths."""

    def test_vf_clip_logging_with_logger(self):
        """Cover VF clipping logging."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            verbose=1,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestMultipleEpochsWithVfClip:
    """Test multiple epochs with VF clipping."""

    def test_multiple_epochs_vf_clip(self):
        """Cover multiple epochs with VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            n_epochs=3,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_multiple_epochs_mean_only(self):
        """Cover multiple epochs with mean_only VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            n_epochs=2,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClipWithCvar:
    """Test VF clipping combined with CVaR."""

    def test_vf_clip_with_cvar(self):
        """Cover VF clipping with CVaR enabled."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            cvar_alpha=0.25,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_per_quantile_with_cvar(self):
        """Cover per_quantile VF clipping with CVaR."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            cvar_alpha=0.2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestVfClipWithEntropy:
    """Test VF clipping combined with entropy scheduling."""

    def test_vf_clip_with_entropy_decay(self):
        """Cover VF clipping with entropy decay."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            ent_coef_decay_steps=5,
        )
        for _ in range(3):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestVfClipCombinations:
    """Test various VF clip combinations."""

    def test_vf_clip_with_target_kl(self):
        """Cover VF clipping with target_kl."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            target_kl=0.01,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_with_high_vf_coef(self):
        """Cover VF clipping with high VF coefficient."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            vf_coef=1.0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_all_options(self):
        """Cover VF clipping with many options."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.15,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
            cvar_alpha=0.2,
            target_kl=0.02,
        )
        for _ in range(2):
            _setup_and_collect(model, env)
            model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
