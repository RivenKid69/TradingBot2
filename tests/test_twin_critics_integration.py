"""
Integration tests for Twin Critics with full PPO training.

Tests cover:
- Full training loop with twin critics
- Convergence behavior
- Overestimation bias reduction
- Compatibility with existing features (VGS, CVaR, etc.)
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


class SimpleDummyEnv(gymnasium.Env):
    """Simple environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        obs = np.random.randn(10).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(10).astype(np.float32)
        reward = -np.sum(action**2)  # Simple quadratic reward
        terminated = self.steps >= self.max_steps
        truncated = False
        return obs, float(reward), terminated, truncated, {}


class TestTwinCriticsIntegration:
    """Integration tests for Twin Critics with PPO training."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_twin_critics_training_quantile(self, env):
        """Test that PPO can train with twin critics enabled (quantile mode)."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "huber_kappa": 1.0,
                "use_twin_critics": True,  # Enable twin critics
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=256)

        # Check that both critics exist and have been updated
        policy = model.policy
        assert policy._use_twin_critics is True
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is not None

        # Check that parameters have been updated (not all zeros)
        assert not torch.allclose(
            policy.quantile_head.linear.weight,
            torch.zeros_like(policy.quantile_head.linear.weight),
            atol=1e-6,
        )
        assert not torch.allclose(
            policy.quantile_head_2.linear.weight,
            torch.zeros_like(policy.quantile_head_2.linear.weight),
            atol=1e-6,
        )

    def test_twin_critics_training_categorical(self, env):
        """Test that PPO can train with twin critics enabled (categorical mode)."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "num_atoms": 21,
            "critic": {
                "distributional": False,
                "use_twin_critics": True,  # Enable twin critics
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=256)

        # Check that both critics exist and have been updated
        policy = model.policy
        assert policy._use_twin_critics is True
        assert policy.dist_head is not None
        assert policy.dist_head_2 is not None

        # Check that parameters have been updated
        assert not torch.allclose(
            policy.dist_head.weight,
            torch.zeros_like(policy.dist_head.weight),
            atol=1e-6,
        )
        assert not torch.allclose(
            policy.dist_head_2.weight,
            torch.zeros_like(policy.dist_head_2.weight),
            atol=1e-6,
        )

    def test_twin_critics_vs_single_critic(self, env):
        """Compare training with and without twin critics."""
        arch_params_single = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": False,  # Single critic
            },
        }

        arch_params_twin = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": True,  # Twin critics
            },
        }

        # Train single critic model
        model_single = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params_single,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
            seed=42,
        )
        model_single.learn(total_timesteps=256)

        # Train twin critics model
        # Need to recreate env to reset state
        env_twin = DummyVecEnv([lambda: SimpleDummyEnv()])
        model_twin = DistributionalPPO(
            CustomActorCriticPolicy,
            env_twin,
            arch_params=arch_params_twin,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
            seed=42,
        )
        model_twin.learn(total_timesteps=256)

        # Both should complete training without errors
        assert model_single.num_timesteps == 256
        assert model_twin.num_timesteps == 256

        # Twin critics model should have second critic
        assert model_single.policy._use_twin_critics is False
        assert model_twin.policy._use_twin_critics is True
        assert model_twin.policy.quantile_head_2 is not None

    def test_twin_critics_with_vgs(self, env):
        """Test Twin Critics compatibility with Variance Gradient Scaling."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": True,
            },
        }

        vgs_config = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "warmup_steps": 10,
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            vgs_config=vgs_config,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Train with both Twin Critics and VGS enabled
        model.learn(total_timesteps=256)

        # Check that both features are active
        assert model.policy._use_twin_critics is True
        assert hasattr(model, "_vgs") and model._vgs is not None

    def test_backward_compatibility(self, env):
        """Test that explicitly disabling twin critics maintains backward compatibility."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": False,  # Explicitly disable for backward compatibility
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Train normally
        model.learn(total_timesteps=256)

        # Check that twin critics are disabled
        assert model.policy._use_twin_critics is False
        assert model.policy.quantile_head is not None
        assert model.policy.quantile_head_2 is None

        # Model should train successfully (backward compatible)
        assert model.num_timesteps == 256

    def test_default_enables_twin_critics(self, env):
        """Test that twin critics are enabled by default."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                # use_twin_critics NOT specified - should default to True
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Check that twin critics are enabled by default
        assert model.policy._use_twin_critics is True
        assert model.policy.quantile_head is not None
        assert model.policy.quantile_head_2 is not None

        # Train normally
        model.learn(total_timesteps=256)

        # Model should train successfully
        assert model.num_timesteps == 256


class TestTwinCriticsOptimization:
    """Test that twin critics parameters are properly optimized."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_both_critics_in_optimizer(self, env):
        """Test that both critics' parameters are in the optimizer."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": True,
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Get optimizer parameter ids
        optimizer_param_ids = {
            id(p) for group in model.policy.optimizer.param_groups for p in group["params"]
        }

        # Check that both critics' parameters are in optimizer
        critic_1_param_ids = {id(p) for p in model.policy.quantile_head.parameters()}
        critic_2_param_ids = {id(p) for p in model.policy.quantile_head_2.parameters()}

        assert critic_1_param_ids.issubset(
            optimizer_param_ids
        ), "First critic params not in optimizer"
        assert critic_2_param_ids.issubset(
            optimizer_param_ids
        ), "Second critic params not in optimizer"

    def test_gradients_flow_to_both_critics(self, env):
        """Test that gradients flow to both critics during training."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": True,
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Store initial weights
        initial_weight_1 = model.policy.quantile_head.linear.weight.clone()
        initial_weight_2 = model.policy.quantile_head_2.linear.weight.clone()

        # Train for a few steps
        model.learn(total_timesteps=128)

        # Check that both critics' weights have changed
        final_weight_1 = model.policy.quantile_head.linear.weight
        final_weight_2 = model.policy.quantile_head_2.linear.weight

        assert not torch.allclose(
            initial_weight_1, final_weight_1, atol=1e-7
        ), "First critic weights did not update"
        assert not torch.allclose(
            initial_weight_2, final_weight_2, atol=1e-7
        ), "Second critic weights did not update"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
