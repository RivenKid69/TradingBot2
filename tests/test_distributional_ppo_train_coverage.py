"""
Comprehensive train() loop coverage tests.
Target the 494 uncovered lines in train() method.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import DistributionalPPO

np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, n_envs: int = 1, reward_scale: float = 0.1) -> DummyVecEnv:
    """Create minimal test environment."""

    def _env_fn():
        import gymnasium

        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
                self.observation_space = spaces.Box(-10.0, 10.0, (4,), np.float32)
                self._step = 0
                self._max = max_steps
                self._reward_scale = reward_scale

            def reset(self, *, seed=None, options=None):
                self._step = 0
                obs = np.random.randn(4).astype(np.float32) * 0.1
                return obs, {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                obs = np.random.randn(4).astype(np.float32) * 0.1
                reward = np.random.randn() * self._reward_scale
                return obs, reward, done, False, {"TimeLimit.truncated": done}

        return _Env()

    return DummyVecEnv([_env_fn for _ in range(n_envs)])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env, **kwargs):
    """Create DistributionalPPO model with defaults."""
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
    """Setup model and collect rollouts."""
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
# VF Clipping Modes with normalize_returns combinations
# =============================================================================


class TestVfClipNormalizeReturnsCombos:
    """Test all VF clip mode + normalize_returns combinations."""

    def test_per_quantile_normalize_true(self):
        """per_quantile + normalize_returns=True."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_per_quantile_normalize_false(self):
        """per_quantile + normalize_returns=False."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=False,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mean_only_normalize_true(self):
        """mean_only + normalize_returns=True."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mean_only_normalize_false(self):
        """mean_only + normalize_returns=False."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=False,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mean_and_variance_normalize_true(self):
        """mean_and_variance + normalize_returns=True."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mean_and_variance_normalize_false(self):
        """mean_and_variance + normalize_returns=False."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=False,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_disable_mode_normalize_true(self):
        """disable mode + normalize_returns=True."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="disable",
            normalize_returns=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_disable_mode_normalize_false(self):
        """disable mode + normalize_returns=False."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="disable",
            normalize_returns=False,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Multi-epoch and batch size variations
# =============================================================================


class TestEpochBatchVariations:
    """Test different epoch and batch size configurations."""

    def test_multiple_epochs_small_batch(self):
        """Multiple epochs with small batch."""
        env = _make_env()
        model = _make_model(env, n_epochs=3, batch_size=2)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_single_epoch_full_batch(self):
        """Single epoch with full batch."""
        env = _make_env()
        model = _make_model(env, n_epochs=1, batch_size=8)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_many_epochs_normalized(self):
        """Many epochs with normalization."""
        env = _make_env()
        model = _make_model(
            env,
            n_epochs=5,
            batch_size=4,
            normalize_returns=True,
            normalize_advantage=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Value scale variations
# =============================================================================


class TestValueScaleVariations:
    """Test value scale configurations."""

    def test_value_scale_percent(self):
        """Test with percent scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale="percent")
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_bps(self):
        """Test with bps scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale="bps")
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_numeric(self):
        """Test with numeric scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale=50.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_fixed(self):
        """Test with fixed value scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale_fixed=100.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# CVaR configurations
# =============================================================================


class TestCvarConfigurations:
    """Test CVaR configurations."""

    def test_cvar_low_alpha(self):
        """Test CVaR with low alpha."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_high_alpha(self):
        """Test CVaR with high alpha."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_with_constraint(self):
        """Test CVaR with constraint."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.1,
            cvar_use_constraint=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_with_penalty(self):
        """Test CVaR with penalty."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.1,
            cvar_use_penalty=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Multi-environment configurations
# =============================================================================


class TestMultiEnvConfigurations:
    """Test multi-environment configurations."""

    def test_two_envs_train(self):
        """Test with 2 environments."""
        env = _make_env(n_envs=2)
        model = _make_model(env, n_steps=4, batch_size=4)
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()

    def test_three_envs_train(self):
        """Test with 3 environments."""
        env = _make_env(n_envs=3)
        model = _make_model(env, n_steps=4, batch_size=6)
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()

    def test_four_envs_normalized(self):
        """Test with 4 environments and normalization."""
        env = _make_env(n_envs=4)
        model = _make_model(
            env,
            n_steps=4,
            batch_size=8,
            normalize_returns=True,
        )
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()


# =============================================================================
# GAE lambda variations
# =============================================================================


class TestGaeLambdaVariations:
    """Test GAE lambda variations."""

    def test_gae_lambda_zero(self):
        """Test with gae_lambda=0 (TD(0))."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_gae_lambda_half(self):
        """Test with gae_lambda=0.5."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_gae_lambda_high(self):
        """Test with gae_lambda=0.99."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.99)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_gae_lambda_one(self):
        """Test with gae_lambda=1.0."""
        env = _make_env()
        model = _make_model(env, gae_lambda=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Entropy coefficient variations
# =============================================================================


class TestEntCoefVariations:
    """Test entropy coefficient variations."""

    def test_ent_coef_zero(self):
        """Test with ent_coef=0."""
        env = _make_env()
        model = _make_model(env, ent_coef=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_ent_coef_small(self):
        """Test with small ent_coef."""
        env = _make_env()
        model = _make_model(env, ent_coef=0.001)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_ent_coef_large(self):
        """Test with large ent_coef."""
        env = _make_env()
        model = _make_model(env, ent_coef=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF coefficient variations
# =============================================================================


class TestVfCoefVariations:
    """Test VF coefficient variations."""

    def test_vf_coef_zero(self):
        """Test with vf_coef=0."""
        env = _make_env()
        model = _make_model(env, vf_coef=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_coef_small(self):
        """Test with small vf_coef."""
        env = _make_env()
        model = _make_model(env, vf_coef=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_coef_large(self):
        """Test with large vf_coef."""
        env = _make_env()
        model = _make_model(env, vf_coef=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Gradient norm clipping variations
# =============================================================================


class TestGradNormVariations:
    """Test gradient norm clipping variations."""

    def test_max_grad_norm_zero(self):
        """Test with max_grad_norm=0 (no clipping)."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_max_grad_norm_small(self):
        """Test with small max_grad_norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_max_grad_norm_large(self):
        """Test with large max_grad_norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=10.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Target KL variations
# =============================================================================


class TestTargetKlVariations:
    """Test target KL variations."""

    def test_target_kl_none(self):
        """Test with target_kl=None."""
        env = _make_env()
        model = _make_model(env, target_kl=None)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_target_kl_very_low(self):
        """Test with very low target_kl (likely early stopping)."""
        env = _make_env()
        model = _make_model(env, target_kl=0.0001, n_epochs=3)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_target_kl_high(self):
        """Test with high target_kl."""
        env = _make_env()
        model = _make_model(env, target_kl=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Clip range variations
# =============================================================================


class TestClipRangeVariations:
    """Test clip range variations."""

    def test_clip_range_small(self):
        """Test with small clip_range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_clip_range_large(self):
        """Test with large clip_range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF clip warmup variations
# =============================================================================


class TestVfClipWarmupVariations:
    """Test VF clip warmup variations."""

    def test_vf_clip_warmup_updates(self):
        """Test with VF clip warmup updates."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            vf_clip_warmup_updates=1,
        )
        model.learn(total_timesteps=24, progress_bar=False)
        env.close()

    def test_vf_clip_threshold_ev_low(self):
        """Test with low VF clip EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            vf_clip_threshold_ev=-0.5,
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_vf_clip_threshold_ev_high(self):
        """Test with high VF clip EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            vf_clip_threshold_ev=0.9,
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Combined complex configurations
# =============================================================================


class TestCombinedConfigurations:
    """Test complex combined configurations."""

    def test_full_config_1(self):
        """Full config: VF clip + normalize + CVaR."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            cvar_alpha=0.1,
            n_epochs=2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_full_config_2(self):
        """Full config: No VF clip + no normalize + constraint."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=None,
            normalize_returns=False,
            cvar_alpha=0.2,
            cvar_use_constraint=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_full_config_3(self):
        """Full config: Multi-env + mean_and_variance + normalize."""
        env = _make_env(n_envs=2)
        model = _make_model(
            env,
            n_steps=4,
            batch_size=4,
            clip_range_vf=0.5,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
            n_epochs=2,
        )
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()

    def test_full_config_4(self):
        """Full config: Low learning rate + gradient clipping."""
        env = _make_env()
        model = _make_model(
            env,
            learning_rate=1e-5,
            max_grad_norm=0.5,
            n_epochs=3,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Learn method variations
# =============================================================================


class TestLearnMethodVariations:
    """Test learn() method variations."""

    def test_learn_short(self):
        """Test short learning run."""
        env = _make_env()
        model = _make_model(env)
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_learn_with_callback(self):
        """Test learning with callback."""
        env = _make_env()
        model = _make_model(env)
        model.learn(total_timesteps=16, callback=_DummyCallback(), progress_bar=False)
        env.close()

    def test_learn_reset_timesteps_false(self):
        """Test learning without resetting timesteps."""
        env = _make_env()
        model = _make_model(env)
        model.learn(total_timesteps=16, progress_bar=False)
        model.learn(total_timesteps=16, reset_num_timesteps=False, progress_bar=False)
        env.close()

    def test_learn_multi_call(self):
        """Test multiple learning calls."""
        env = _make_env()
        model = _make_model(env)
        for _ in range(3):
            model.learn(total_timesteps=8, reset_num_timesteps=False, progress_bar=False)
        env.close()


# =============================================================================
# Reward scale variations (affects value processing)
# =============================================================================


class TestRewardScaleVariations:
    """Test with different reward scales."""

    def test_small_rewards(self):
        """Test with small rewards."""
        env = _make_env(reward_scale=0.001)
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_large_rewards(self):
        """Test with large rewards."""
        env = _make_env(reward_scale=10.0)
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_mixed_scale_normalized(self):
        """Test with normalized returns and various scales."""
        env = _make_env(reward_scale=1.0)
        model = _make_model(
            env,
            normalize_returns=True,
            normalize_advantage=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
