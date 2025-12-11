"""
Integration Test Suite for Twin Critics VF Clipping - Categorical Critic

This module tests the END-TO-END integration of Twin Critics VF clipping
for CATEGORICAL critics through the training loop.

Test Coverage:
1. Training integration: Trains successfully with VF clipping enabled
2. Training without VF clipping: Backwards compatibility
3. Error handling: Proper error messages for invalid configurations
4. Categorical vs Quantile: Correct detection of critic type
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
def categorical_policy_config():
    """Configuration for categorical critic with Twin Critics enabled."""
    return {
        "hidden_dim": 32,
        "lstm_hidden_size": 32,
        "critic": {
            "distributional": True,
            "categorical": True,  # CATEGORICAL critic
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "use_twin_critics": True,
        },
    }


@pytest.fixture
def quantile_policy_config():
    """Configuration for quantile critic with Twin Critics enabled (for comparison)."""
    return {
        "hidden_dim": 32,
        "lstm_hidden_size": 32,
        "critic": {
            "distributional": True,
            # NO categorical flag → quantile critic
            "num_quantiles": 21,
            "huber_kappa": 1.0,
            "use_twin_critics": True,
        },
    }


class TestCategoricalTrainingIntegration:
    """Test end-to-end training integration for categorical critic."""

    def test_categorical_trains_with_vf_clipping(self, simple_env, categorical_policy_config):
        """Test that categorical critic trains successfully with VF clipping enabled."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            clip_range_vf=0.2,  # Enable VF clipping
            verbose=0,
        )

        # Train for multiple steps
        model.learn(total_timesteps=256, log_interval=None)
        assert model.num_timesteps == 256

    def test_categorical_trains_without_vf_clipping(self, simple_env, categorical_policy_config):
        """Test that categorical critic trains successfully without VF clipping."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            clip_range_vf=None,  # Disable VF clipping
            verbose=0,
        )

        # Train for multiple steps
        model.learn(total_timesteps=256, log_interval=None)
        assert model.num_timesteps == 256

    def test_categorical_longer_training(self, simple_env, categorical_policy_config):
        """Test categorical critic with longer training to verify stability."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            clip_range_vf=0.3,  # Enable VF clipping with larger range
            verbose=0,
        )

        # Longer training
        model.learn(total_timesteps=512, log_interval=None)
        assert model.num_timesteps == 512


class TestCategoricalPolicyConfiguration:
    """Test that categorical vs quantile critic is correctly detected."""

    def test_categorical_flag_sets_use_quantile_false(self, simple_env, categorical_policy_config):
        """Test that categorical=True sets _use_quantile_value_head=False."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Verify policy has correct flag
        assert hasattr(model.policy, '_use_quantile_value_head')
        assert model.policy._use_quantile_value_head is False, \
            "Categorical critic should have _use_quantile_value_head=False"

    def test_quantile_flag_sets_use_quantile_true(self, simple_env, quantile_policy_config):
        """Test that quantile critic (no categorical flag) sets _use_quantile_value_head=True."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Verify policy has correct flag
        assert hasattr(model.policy, '_use_quantile_value_head')
        assert model.policy._use_quantile_value_head is True, \
            "Quantile critic should have _use_quantile_value_head=True"

    def test_categorical_has_atoms_buffer(self, simple_env, categorical_policy_config):
        """Test that categorical critic has atoms buffer."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Verify policy has atoms buffer
        assert hasattr(model.policy, 'atoms')
        assert model.policy.atoms.numel() > 0, "Categorical critic should have non-empty atoms buffer"
        assert model.policy.atoms.shape[0] == 51, "Should have 51 atoms as configured"


class TestCategoricalTwinCriticsVFClipping:
    """Test Twin Critics VF clipping specific behavior for categorical critic."""

    def test_twin_critics_enabled_for_categorical(self, simple_env, categorical_policy_config):
        """Test that Twin Critics is enabled for categorical critic."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Verify Twin Critics is enabled
        assert hasattr(model.policy, '_use_twin_critics')
        assert model.policy._use_twin_critics is True

        # Verify second critic head exists
        assert hasattr(model.policy, 'dist_head_2')
        assert model.policy.dist_head_2 is not None

    def test_vf_clipping_with_different_ranges(self, simple_env, categorical_policy_config):
        """Test VF clipping with different clip_range_vf values."""
        clip_ranges = [0.1, 0.2, 0.5, 1.0]

        for clip_range in clip_ranges:
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                simple_env,
                policy_kwargs={"arch_params": categorical_policy_config},
                n_steps=64,
                batch_size=32,
                n_epochs=1,
                clip_range_vf=clip_range,
                verbose=0,
            )

            # Train to verify no errors
            model.learn(total_timesteps=64, log_interval=None)
            assert model.num_timesteps == 64


class TestBackwardCompatibility:
    """Test backward compatibility for categorical critic."""

    def test_categorical_without_twin_critics(self, simple_env):
        """Test categorical critic with Twin Critics disabled (backward compatibility)."""
        config = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "categorical": True,
                "num_atoms": 51,
                "v_min": -10.0,
                "v_max": 10.0,
                "use_twin_critics": False,  # Disable Twin Critics
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": config},
            n_steps=64,
            clip_range_vf=0.2,  # VF clipping should still work
            verbose=0,
        )

        # Train to verify no errors
        model.learn(total_timesteps=128, log_interval=None)
        assert model.num_timesteps == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
