"""
Integration tests targeting VF clipping code paths in DistributionalPPO.

Covers:
- _twin_critics_vf_clipping_loss() method (lines 3320-3680)
- Train loop VF clipping for quantile critics (lines 11063-11353)
- Train loop VF clipping for categorical critics (lines 11492-11773)
"""

import os
import warnings
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv


import distributional_ppo as dppo
from distributional_ppo import DistributionalPPO


# =============================================================================
# Test Environment
# =============================================================================

class MinimalVFClipEnv(gymnasium.Env):
    """Minimal environment for VF clipping tests."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = 8

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(-np.sum(obs ** 2) * 0.01)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_vec_env(n_envs=1, seed=42):
    """Create vectorized environment."""
    return DummyVecEnv([lambda: MinimalVFClipEnv(seed=seed) for _ in range(n_envs)])


# =============================================================================
# Integration Tests: Train Loop VF Clipping (Quantile)
# =============================================================================

class TestTrainLoopVFClippingQuantile:
    """Integration tests for train loop VF clipping with quantile critics."""

    def test_train_quantile_vf_clip_per_quantile_normalize_returns(self):
        """Test training with quantile VF clipping per_quantile mode + normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_per_quantile_no_normalize(self):
        """Test training with quantile VF clipping per_quantile mode without normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=False,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_mean_only_normalize_returns(self):
        """Test training with quantile VF clipping mean_only mode + normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_mean_only_no_normalize(self):
        """Test training with quantile VF clipping mean_only mode without normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=False,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_mean_and_variance_normalize_returns(self):
        """Test training with quantile VF clipping mean_and_variance mode + normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_mean_and_variance_no_normalize(self):
        """Test training with quantile VF clipping mean_and_variance mode without normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=False,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_vf_clip_disabled(self):
        """Test training with VF clipping disabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="disable",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_train_quantile_no_vf_clip(self):
        """Test training without VF clipping (clip_range_vf=None)."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=None,
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Integration Tests: Train Loop VF Clipping with Twin Critics
# =============================================================================

class TestTrainLoopVFClippingTwinCritics:
    """Integration tests for train loop VF clipping with Twin Critics."""

    def test_twin_critics_vf_clip_per_quantile(self):
        """Test training with twin critics VF clipping per_quantile mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "num_quantiles": 8,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_twin_critics_vf_clip_mean_only(self):
        """Test training with twin critics VF clipping mean_only mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "num_quantiles": 8,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_twin_critics_vf_clip_mean_and_variance(self):
        """Test training with twin critics VF clipping mean_and_variance mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "num_quantiles": 8,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_twin_critics_vf_clip_no_normalize_returns(self):
        """Test training with twin critics VF clipping without normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=False,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "num_quantiles": 8,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_twin_critics_no_vf_clip(self):
        """Test training with twin critics but no VF clipping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=None,
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "num_quantiles": 8,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Integration Tests: Categorical Critic VF Clipping
# =============================================================================

class TestTrainLoopVFClippingCategorical:
    """Integration tests for categorical critic VF clipping."""

    def test_categorical_vf_clip_mean_only(self):
        """Test training with categorical VF clipping mean_only mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_categorical_vf_clip_mean_and_variance(self):
        """Test training with categorical VF clipping mean_and_variance mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_categorical_vf_clip_per_quantile(self):
        """Test training with categorical VF clipping per_quantile mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_categorical_vf_clip_no_normalize_returns(self):
        """Test training with categorical VF clipping without normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=False,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_categorical_vf_clip_disabled(self):
        """Test training with categorical VF clipping disabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="disable",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Integration Tests: Categorical Critic with Twin Critics VF Clipping
# =============================================================================

class TestTrainLoopVFClippingCategoricalTwin:
    """Integration tests for categorical critic with Twin Critics VF clipping."""

    def test_categorical_twin_vf_clip_mean_only(self):
        """Test training with categorical twin critics VF clipping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_categorical_twin_vf_clip_per_quantile(self):
        """Test training with categorical twin critics per_quantile VF clipping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "arch_params": {
                    "critic": {
                        "distributional": True,
                        "value_type": "categorical",
                        "num_atoms": 51,
                        "v_min": -10.0,
                        "v_max": 10.0,
                        "use_twin_critics": True,
                    }
                }
            },
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Integration Tests: VF Clip Warmup
# =============================================================================

class TestVFClipWarmup:
    """Integration tests for VF clipping warmup."""

    def test_vf_clip_warmup_with_threshold(self):
        """Test VF clipping warmup with EV threshold."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            vf_clip_warmup_updates=1,
            vf_clip_threshold_ev=0.1,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_vf_clip_warmup_zero_updates(self):
        """Test VF clipping with no warmup."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            vf_clip_warmup_updates=0,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Integration Tests: Various Configurations
# =============================================================================

class TestVariousConfigurations:
    """Integration tests for various model configurations."""

    def test_multi_env_vf_clip(self):
        """Test VF clipping with multiple environments."""
        env = make_vec_env(n_envs=2)
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_larger_batch_size_vf_clip(self):
        """Test VF clipping with larger batch size."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=32, progress_bar=False)
        env.close()

    def test_vf_clip_with_cvar(self):
        """Test VF clipping with CVaR enabled."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            cvar_alpha=0.1,
            cvar_weight=0.1,
            normalize_returns=True,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_vf_clip_with_gae_lambda_variations(self):
        """Test VF clipping with different GAE lambda."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gae_lambda=0.9,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            normalize_returns=True,
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
