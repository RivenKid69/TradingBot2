"""
Additional integration tests for DistributionalPPO coverage.

Targets:
- KL early stopping branches
- Various edge cases in train loop
"""

import os
import warnings
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv

from distributional_ppo import DistributionalPPO


# =============================================================================
# Test Environment
# =============================================================================


class MinimalEnv(gymnasium.Env):
    """Minimal environment for coverage tests."""

    def __init__(self, seed: int = 42, max_steps: int = 10):
        super().__init__()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = max_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(-np.sum(obs**2) * 0.01)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_vec_env(n_envs=1, seed=42):
    """Create vectorized environment."""
    return DummyVecEnv([lambda: MinimalEnv(seed=seed) for _ in range(n_envs)])


# =============================================================================
# Tests for KL Early Stopping
# =============================================================================


class TestKLEarlyStopping:
    """Tests for KL early stopping branches."""

    def test_kl_early_stop_with_target_kl(self):
        """Test training with target_kl for KL early stopping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=10,  # More epochs to trigger KL stopping
            target_kl=0.001,  # Very small target_kl to trigger early stop
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=64, progress_bar=False)
        env.close()

    def test_kl_absolute_stop_factor(self):
        """Test training with _kl_absolute_stop_factor."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=10,
            target_kl=0.001,
            verbose=0,
            device="cpu",
        )
        # Manually set the absolute stop factor to trigger this branch
        model._kl_absolute_stop_factor = 2.0
        model.learn(total_timesteps=64, progress_bar=False)
        env.close()

    def test_no_target_kl(self):
        """Test training without target_kl."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            target_kl=None,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for VF Coefficient Warmup
# =============================================================================


class TestVFCoefWarmup:
    """Tests for VF coefficient warmup."""

    def test_vf_coef_warmup_positive(self):
        """Test training with VF coefficient warmup."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            vf_coef=0.5,
            vf_coef_warmup_updates=2,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=64, progress_bar=False)
        env.close()

    def test_vf_coef_warmup_zero(self):
        """Test training with no VF coefficient warmup."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            vf_coef=0.5,
            vf_coef_warmup_updates=0,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for CVaR Constraint
# =============================================================================


class TestCVaRConstraint:
    """Tests for CVaR constraint."""

    def test_cvar_with_constraint(self):
        """Test training with CVaR constraint."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.1,
            cvar_use_constraint=True,
            cvar_limit=-1.0,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_cvar_with_penalty(self):
        """Test training with CVaR penalty."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.1,
            cvar_use_penalty=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_cvar_with_both(self):
        """Test training with both CVaR constraint and penalty."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.1,
            cvar_use_constraint=True,
            cvar_use_penalty=True,
            cvar_limit=-1.0,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Advantage Normalization
# =============================================================================


class TestAdvantageNormalization:
    """Tests for advantage normalization."""

    def test_normalize_advantage_enabled(self):
        """Test training with advantage normalization enabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            normalize_advantage=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_normalize_advantage_disabled(self):
        """Test training with advantage normalization disabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            normalize_advantage=False,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Entropy Coefficient
# =============================================================================


class TestEntropyCoefficient:
    """Tests for entropy coefficient."""

    def test_ent_coef_constant(self):
        """Test training with constant entropy coefficient."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ent_coef=0.01,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_ent_coef_zero(self):
        """Test training with zero entropy coefficient."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ent_coef=0.0,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Multiple Epochs
# =============================================================================


class TestMultipleEpochs:
    """Tests for training with multiple epochs."""

    def test_multiple_epochs_small_batch(self):
        """Test training with multiple epochs and small batches."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=16,
            batch_size=4,
            n_epochs=3,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_single_epoch_full_batch(self):
        """Test training with single epoch and full batch."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,  # Full batch
            n_epochs=1,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Max Grad Norm
# =============================================================================


class TestMaxGradNorm:
    """Tests for gradient clipping."""

    def test_grad_clipping_enabled(self):
        """Test training with gradient clipping enabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            max_grad_norm=0.5,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_grad_clipping_high(self):
        """Test training with high gradient clipping threshold."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            max_grad_norm=10.0,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for GAE Lambda
# =============================================================================


class TestGAELambda:
    """Tests for GAE lambda variations."""

    def test_gae_lambda_high(self):
        """Test training with high GAE lambda."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gae_lambda=0.99,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_gae_lambda_low(self):
        """Test training with low GAE lambda."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gae_lambda=0.5,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Gamma
# =============================================================================


class TestGamma:
    """Tests for gamma (discount factor)."""

    def test_gamma_high(self):
        """Test training with high gamma."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gamma=0.999,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_gamma_low(self):
        """Test training with low gamma."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gamma=0.9,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Tests for Clip Range
# =============================================================================


class TestClipRange:
    """Tests for clip range variations."""

    def test_clip_range_high(self):
        """Test training with high clip range."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range=0.4,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_clip_range_low(self):
        """Test training with low clip range."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range=0.1,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
