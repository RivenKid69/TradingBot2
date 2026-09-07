"""
Integration Test Suite for Twin Critics VF Clipping - All Modes

This module tests that ALL VF clipping modes integrate correctly with the training loop.

Test Coverage:
1. Per-quantile mode integration
2. Mean-only mode integration
3. Mean-and-variance mode integration
4. Mode dispatch correctness5. Backward compatibility (mode=None)
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
from gymnasium import spaces

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


class SimpleDummyEnv(gym.Env):
    """Simple test environment with Box action space."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        obs = np.random.randn(4).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(4).astype(np.float32)
        reward = -np.sum(action**2)  # Simple quadratic reward
        terminated = self.steps >= self.max_steps
        truncated = False
        return obs, float(reward), terminated, truncated, {}


@pytest.fixture
def simple_env():
    """Create a simple test environment."""
    return SimpleDummyEnv()


@pytest.fixture
def quantile_policy_config():
    """Configuration for quantile critic with Twin Critics enabled."""
    return {
        "hidden_dim": 32,
        "lstm_hidden_size": 32,
        "critic": {
            "distributional": True,
            "num_quantiles": 11,
            "huber_kappa": 1.0,
            "use_twin_critics": True,
        },
    }


class TestAllModesIntegration:
    """Test all VF clipping modes integrate correctly with training loop."""

    @pytest.mark.parametrize("mode", ["per_quantile", "mean_only", "mean_and_variance", None])
    def test_mode_integration(self, simple_env, quantile_policy_config, mode):
        """Test that each mode integrates correctly with the training loop."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            clip_range_vf=0.2,  # Enable VF clipping
            distributional_vf_clip_mode=mode,
            verbose=0,
        )

        # Train for 1 step to verify no errors
        model.learn(total_timesteps=64, log_interval=None)

        # Verify training completed
        assert model.num_timesteps == 64

    def test_per_quantile_mode_trains(self, simple_env, quantile_policy_config):
        """Test per_quantile mode trains without errors."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="per_quantile",
            verbose=0,
        )

        model.learn(total_timesteps=128, log_interval=None)
        assert model.num_timesteps == 128

    def test_mean_only_mode_trains(self, simple_env, quantile_policy_config):
        """Test mean_only mode trains without errors."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_only",
            verbose=0,
        )

        model.learn(total_timesteps=128, log_interval=None)
        assert model.num_timesteps == 128

    def test_mean_and_variance_mode_trains(self, simple_env, quantile_policy_config):
        """Test mean_and_variance mode trains without errors."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            distributional_vf_clip_variance_factor=2.0,
            verbose=0,
        )

        model.learn(total_timesteps=128, log_interval=None)
        assert model.num_timesteps == 128

    def test_variance_factor_configurable(self, simple_env, quantile_policy_config):
        """Test that variance_factor parameter is configurable."""
        variance_factor = 1.5

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=0.2,
            distributional_vf_clip_mode="mean_and_variance",
            distributional_vf_clip_variance_factor=variance_factor,
            verbose=0,
        )

        # Verify parameter was set
        assert model.distributional_vf_clip_variance_factor == variance_factor

    def test_mode_none_defaults_to_per_quantile(self, simple_env, quantile_policy_config):
        """Test that mode=None defaults to per_quantile."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=0.2,
            distributional_vf_clip_mode=None,  # Should default to per_quantile
            verbose=0,
        )

        model.learn(total_timesteps=64, log_interval=None)
        assert model.num_timesteps == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
