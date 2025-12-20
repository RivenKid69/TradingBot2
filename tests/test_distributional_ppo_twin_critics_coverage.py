"""
Tests for Twin Critics VF clipping paths in distributional_ppo.py.
Targets the uncovered lines in train() related to:
- Twin Critics VF clipping (quantile and categorical)
- distributional_vf_clip_mode settings
- normalize_returns with VF clipping
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import DistributionalPPO

np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, n_envs: int = 1) -> DummyVecEnv:
    """Create minimal test environment."""
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
                obs = np.random.randn(4).astype(np.float32) * 0.1
                return obs, {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                obs = np.random.randn(4).astype(np.float32) * 0.1
                reward = np.random.randn() * 0.1
                return obs, reward, done, False, {"TimeLimit.truncated": done}

        return _Env()

    return DummyVecEnv([_env_fn for _ in range(n_envs)])


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


# =============================================================================
# Distributional VF Clip Mode Tests
# =============================================================================

class TestDistributionalVfClipModes:
    """Test all distributional_vf_clip_mode settings."""

    def test_vf_clip_mode_mean_only(self):
        """Test mean_only VF clip mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mode_mean_and_variance(self):
        """Test mean_and_variance VF clip mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mode_per_quantile(self):
        """Test per_quantile VF clip mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mode_disable(self):
        """Test disable VF clip mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="disable",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_mode_none(self):
        """Test None VF clip mode (default)."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode=None,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Clip with Normalize Returns
# =============================================================================

class TestVfClipWithNormalizeReturns:
    """Test VF clipping with normalize_returns enabled."""

    def test_vf_clip_normalize_returns_mean_only(self):
        """VF clip with normalize_returns and mean_only mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            normalize_returns=True,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_normalize_returns_per_quantile(self):
        """VF clip with normalize_returns and per_quantile mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            normalize_returns=True,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_normalize_returns_mean_and_variance(self):
        """VF clip with normalize_returns and mean_and_variance mode."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            normalize_returns=True,
            distributional_vf_clip_mode="mean_and_variance",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Clip Variance Factor
# =============================================================================

class TestVfClipVarianceFactor:
    """Test distributional_vf_clip_variance_factor settings."""

    def test_variance_factor_default(self):
        """Test default variance factor."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_variance_factor_high(self):
        """Test high variance factor."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            distributional_vf_clip_variance_factor=5.0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_variance_factor_low(self):
        """Test low variance factor (1.0 is minimum)."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            distributional_vf_clip_variance_factor=1.0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Clip Warmup
# =============================================================================

class TestVfClipWarmup:
    """Test VF clip warmup behavior."""

    def test_vf_clip_warmup_initial(self):
        """Test VF clip during warmup period."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_warmup_updates=100,  # Long warmup
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_warmup_short(self):
        """Test VF clip with short warmup."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_warmup_updates=1,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Clip Threshold EV
# =============================================================================

class TestVfClipThresholdEv:
    """Test vf_clip_threshold_ev settings."""

    def test_threshold_ev_high(self):
        """Test with high EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_threshold_ev=0.9,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_threshold_ev_low(self):
        """Test with low EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_threshold_ev=-0.5,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_threshold_ev_zero(self):
        """Test with zero EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_clip_threshold_ev=0.0,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Multiple Epochs with VF Clip
# =============================================================================

class TestMultipleEpochsVfClip:
    """Test VF clipping with multiple training epochs."""

    def test_multiple_epochs_mean_only(self):
        """Multiple epochs with mean_only mode."""
        env = _make_env()
        model = _make_model(
            env,
            n_epochs=3,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_multiple_epochs_per_quantile(self):
        """Multiple epochs with per_quantile mode."""
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


# =============================================================================
# Small and Large Clip Ranges
# =============================================================================

class TestClipRangeSizes:
    """Test different clip range sizes."""

    def test_very_small_clip_range(self):
        """Test very small clip range."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.001,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_very_large_clip_range(self):
        """Test very large clip range."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=10.0,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Combined Configurations
# =============================================================================

class TestCombinedConfigurations:
    """Test combined configuration scenarios."""

    def test_vf_clip_with_target_kl(self):
        """VF clip with target_kl enabled."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            target_kl=0.01,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_with_ent_coef_decay(self):
        """VF clip with entropy coefficient decay."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            ent_coef_decay_steps=5,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_with_max_grad_norm(self):
        """VF clip with gradient clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            max_grad_norm=0.5,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_all_modes_sequential(self):
        """Test all VF clip modes sequentially."""
        env = _make_env()
        for mode in ["mean_only", "mean_and_variance", "per_quantile"]:
            model = _make_model(
                env,
                clip_range_vf=0.2,
                distributional_vf_clip_mode=mode,
            )
            _setup_and_collect(model, env)
            model.train()
        env.close()


# =============================================================================
# Multi-Environment with VF Clip
# =============================================================================

class TestMultiEnvVfClip:
    """Test VF clipping with multiple environments."""

    def test_multi_env_vf_clip(self):
        """VF clip with multiple environments."""
        env = _make_env(n_envs=2)
        model = _make_model(
            env,
            n_steps=4,
            batch_size=4,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()

    def test_multi_env_vf_clip_per_quantile(self):
        """VF clip per_quantile with multiple environments."""
        env = _make_env(n_envs=2)
        model = _make_model(
            env,
            n_steps=4,
            batch_size=4,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
        )
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()


# =============================================================================
# VF Clip with Different VF Coefficients
# =============================================================================

class TestVfClipWithVfCoef:
    """Test VF clipping with different vf_coef values."""

    def test_vf_clip_low_vf_coef(self):
        """VF clip with low vf_coef."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_coef=0.1,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_high_vf_coef(self):
        """VF clip with high vf_coef."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            vf_coef=1.0,
            distributional_vf_clip_mode="mean_only",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Learn Method with VF Clip
# =============================================================================

class TestLearnWithVfClip:
    """Test learn() method with VF clipping."""

    def test_learn_with_vf_clip(self):
        """Full learn() with VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_learn_with_vf_clip_per_quantile(self):
        """Full learn() with per_quantile VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_learn_with_vf_clip_mean_and_variance(self):
        """Full learn() with mean_and_variance VF clipping."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
