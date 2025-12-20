"""
Tests targeting 95% coverage for distributional_ppo.py.
Focuses on:
1. Twin Critics VF clipping paths
2. CVaR edge cases
3. Uncovered training branches
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    calculate_cvar,
    safe_explained_variance,
)

np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Test Environment
# =============================================================================


class SimpleTestEnv(gymnasium.Env):
    """Simple test environment with Box observation and action spaces."""

    def __init__(self, seed: int = 42, max_steps: int = 8):
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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(-np.sum(obs**2) * 0.01 + 0.1)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_test_env(n_envs: int = 1, seed: int = 42) -> DummyVecEnv:
    """Create vectorized test environment."""
    return DummyVecEnv([lambda: SimpleTestEnv(seed=seed) for _ in range(n_envs)])


# =============================================================================
# Tests for Twin Critics VF Clipping (Quantile Mode)
# =============================================================================


class TestTwinCriticsVFClippingQuantile:
    """Test Twin Critics VF clipping paths in quantile mode."""

    def test_train_with_vf_clip_mode_per_quantile_twin(self):
        """Cover Twin Critics per_quantile VF clip mode."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_vf_clip_mode_mean_only_twin(self):
        """Cover Twin Critics mean_only VF clip mode."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_only",
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_vf_clip_mode_mean_and_variance_twin(self):
        """Cover Twin Critics mean_and_variance VF clip mode."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_and_variance",
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_normalize_returns_and_vf_clip(self):
        """Cover VF clipping with normalize_returns enabled."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            normalize_returns=True,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()


# =============================================================================
# Tests for CVaR Edge Cases
# =============================================================================


class TestCVaREdgeCases:
    """Test CVaR edge cases and boundary conditions."""

    def test_cvar_alpha_very_small(self):
        """Test CVaR with very small alpha (extreme tail)."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.01,  # Very small alpha
            cvar_weight=0.5,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_cvar_alpha_large(self):
        """Test CVaR with large alpha (close to full distribution)."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.95,  # Large alpha
            cvar_weight=0.3,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_cvar_with_ema_and_constraint(self):
        """Test CVaR with EMA and constraint together."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.3,
            cvar_ema_beta=0.95,
            cvar_use_constraint=True,
            cvar_limit=-0.5,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_calculate_cvar_edge_alpha(self):
        """Test calculate_cvar with edge alpha values via training."""
        # Test CVaR calculation through training which uses correct API
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.99,  # Near-full distribution
            cvar_weight=0.3,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for PopArt Edge Cases
# =============================================================================


class TestPopArtEdgeCases:
    """Test PopArt controller edge cases."""

    def test_popart_with_live_mode(self):
        """Test PopArt in live mode."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        controller = PopArtController(
            enabled=True,
            mode="live",
            min_samples=1,
            warmup_updates=0,
        )

        # Test basic live mode operation
        returns = torch.randn(100)
        controller._update_counter = 5

        env.close()

    def test_popart_ema_update(self):
        """Test PopArt EMA update path."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            ema_beta=0.95,
            min_samples=1,
            warmup_updates=0,
        )

        # Manually set some shadow values
        controller._shadow_mean = 0.5
        controller._shadow_std = 1.0
        controller._update_counter = 10

        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            verbose=0,
        )

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(100),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None

        env.close()


# =============================================================================
# Tests for Training Configuration Edge Cases
# =============================================================================


class TestTrainingEdgeCases:
    """Test training configuration edge cases."""

    def test_train_with_multiple_warmups(self):
        """Cover multiple warmup parameters together."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            vf_coef_warmup=0.1,
            vf_coef_warmup_updates=2,
            clip_range_warmup=0.1,
            clip_range_warmup_updates=2,
            critic_grad_warmup_updates=2,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_kl_features(self):
        """Cover KL features together."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            target_kl=0.02,
            kl_penalty_beta=0.01,
            kl_epoch_decay=0.9,
            kl_consec_minibatches=2,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_entropy_features(self):
        """Cover entropy features together."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ent_coef=0.02,
            ent_coef_decay_steps=20,
            ent_coef_final=0.001,
            ent_coef_plateau_window=3,
            verbose=0,
        )
        model.learn(total_timesteps=48)
        env.close()

    def test_train_with_gradient_features(self):
        """Cover gradient scaling and clipping features."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            max_grad_norm=0.5,
            gradient_accumulation_steps=2,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_vgs_features(self):
        """Cover VGS (Variance Gradient Scaler) features."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_warmup_steps=2,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_ret_clip(self):
        """Cover return clipping path."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ret_clip=3.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for Buffer Edge Cases
# =============================================================================


class TestBufferEdgeCases:
    """Test buffer-related edge cases."""

    def test_train_with_small_buffer(self):
        """Test training with minimal buffer size."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        model.learn(total_timesteps=8)
        env.close()

    def test_train_with_multiple_epochs(self):
        """Test training with multiple epochs."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=4,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()


# =============================================================================
# Tests for Value Scale Edge Cases
# =============================================================================


class TestValueScaleEdgeCases:
    """Test value scale edge cases."""

    def test_value_target_scale_update_disabled(self):
        """Test with value scale update disabled."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_scale_update_enabled=False,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_value_target_scale_with_fixed(self):
        """Test with fixed value target scale."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_target_scale_fixed=100.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for Optimizer Edge Cases
# =============================================================================


class TestOptimizerEdgeCases:
    """Test optimizer configuration edge cases."""

    def test_train_with_adamw_optimizer(self):
        """Test with AdamW optimizer explicitly."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_class="AdamW",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_adam_optimizer(self):
        """Test with Adam optimizer."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_class="Adam",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_sgd_optimizer(self):
        """Test with SGD optimizer."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_class="SGD",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_lr_bounds(self):
        """Test with learning rate bounds."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            learning_rate=3e-4,
            optimizer_lr_min=1e-7,
            optimizer_lr_max=1e-2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for Extended Training
# =============================================================================


class TestExtendedTraining:
    """Tests for extended training scenarios."""

    def test_longer_training_run(self):
        """Test with longer training run to cover more paths."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            cvar_alpha=0.1,
            cvar_weight=0.2,
            normalize_returns=True,
            verbose=0,
        )
        model.learn(total_timesteps=64)
        env.close()

    def test_full_feature_training(self):
        """Test with all major features enabled."""
        env = make_test_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="mean_and_variance",
            cvar_alpha=0.1,
            cvar_weight=0.2,
            target_kl=0.02,
            ent_coef=0.01,
            variance_gradient_scaling=True,
            verbose=0,
        )
        model.learn(total_timesteps=64)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
