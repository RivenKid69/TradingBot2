"""
Comprehensive tests for Twin Critics default behavior.

Tests verify that:
- Twin Critics are enabled by default in all modes
- Explicit enabling/disabling works correctly
- Edge cases are handled properly
- All combinations of settings work as expected
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

gym = pytest.importorskip("gymnasium")
from gymnasium import spaces
from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleDummyEnv(gym.Env):
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
        reward = -np.sum(action**2)
        terminated = self.steps >= self.max_steps
        truncated = False
        return obs, float(reward), terminated, truncated, {}


class TestDefaultBehavior:
    """Test default Twin Critics behavior."""

    def test_default_quantile_mode(self):
        """Test that Twin Critics are enabled by default in quantile mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                # use_twin_critics NOT specified
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is True, "Twin Critics should be enabled by default"
        assert policy._use_quantile_value_head is True
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is not None

    def test_default_categorical_mode(self):
        """Test that Twin Critics are enabled by default in categorical mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "num_atoms": 51,
            "critic": {
                "distributional": False,
                # use_twin_critics NOT specified
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is True, "Twin Critics should be enabled by default"
        assert policy._use_quantile_value_head is False
        assert policy.dist_head is not None
        assert policy.dist_head_2 is not None

    def test_default_minimal_config(self):
        """Test default behavior with minimal arch_params."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Minimal config without critic section
        arch_params = {"hidden_dim": 32}

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        # Should still enable Twin Critics by default
        assert policy._use_twin_critics is True

    def test_default_empty_critic_config(self):
        """Test default behavior with empty critic config."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {"hidden_dim": 32, "critic": {}}  # Empty critic config

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is True, "Should default to True even with empty config"


class TestExplicitControl:
    """Test explicit enabling/disabling of Twin Critics."""

    def test_explicit_enable_quantile(self):
        """Test explicit enabling in quantile mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "use_twin_critics": True,
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is True
        assert policy.quantile_head_2 is not None

    def test_explicit_disable_quantile(self):
        """Test explicit disabling in quantile mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "use_twin_critics": False,
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is False
        assert policy.quantile_head_2 is None

    def test_explicit_enable_categorical(self):
        """Test explicit enabling in categorical mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": False,
                "use_twin_critics": True,
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is True
        assert policy.dist_head_2 is not None

    def test_explicit_disable_categorical(self):
        """Test explicit disabling in categorical mode."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": False,
                "use_twin_critics": False,
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        assert policy._use_twin_critics is False
        assert policy.dist_head_2 is None


class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_various_value_types_for_flag(self):
        """Test that various value types are handled correctly."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Test integer 0 (should be False)
        arch_params = {
            "hidden_dim": 32,
            "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": 0},
        }
        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        assert policy._use_twin_critics is False

        # Test integer 1 (should be True)
        arch_params = {
            "hidden_dim": 32,
            "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": 1},
        }
        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        assert policy._use_twin_critics is True

        # Test string "true" (should be True)
        arch_params = {
            "hidden_dim": 32,
            "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": "true"},
        }
        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        assert policy._use_twin_critics is True

        # Test string "false" (should be False)
        arch_params = {
            "hidden_dim": 32,
            "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": "false"},
        }
        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        assert policy._use_twin_critics is False

    def test_none_value(self):
        """Test that None defaults to True."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": None},
        }
        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        # None should use default value (True)
        assert policy._use_twin_critics is True  # None uses fallback value (True)


class TestPPOIntegration:
    """Test Twin Critics default behavior in PPO training."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_ppo_default_enables_twin_critics(self, env):
        """Test that PPO enables Twin Critics by default."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                # use_twin_critics NOT specified
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            policy_kwargs={"arch_params": arch_params},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Should enable Twin Critics by default
        assert model.policy._use_twin_critics is True
        assert model.policy.quantile_head_2 is not None

        # Should train successfully
        model.learn(total_timesteps=128)
        assert model.num_timesteps == 128

    def test_ppo_explicit_disable(self, env):
        """Test that PPO respects explicit disabling."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                "use_twin_critics": False,
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            policy_kwargs={"arch_params": arch_params},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Should respect explicit disable
        assert model.policy._use_twin_critics is False
        assert model.policy.quantile_head_2 is None

        # Should still train successfully
        model.learn(total_timesteps=128)
        assert model.num_timesteps == 128

    def test_ppo_value_predictions_use_min(self, env):
        """Test that PPO uses min(V1, V2) for predictions by default."""
        arch_params = {
            "hidden_dim": 32,
            "lstm_hidden_size": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 8,
                # use_twin_critics defaults to True
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            policy_kwargs={"arch_params": arch_params},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Collect some rollouts
        model.learn(total_timesteps=64)

        # Test value prediction
        obs = torch.randn(4, 10)
        lstm_states = model.policy.recurrent_initial_state
        episode_starts = torch.zeros(4, dtype=torch.bool)

        with torch.no_grad():
            value_pred = model.policy.predict_values(obs, lstm_states, episode_starts)

            # Get individual critic values
            features = model.policy.extract_features(obs, model.policy.vf_features_extractor)
            latent_vf, _ = model.policy._process_sequence(
                features, lstm_states.vf, episode_starts, model.policy.lstm_critic
            )
            latent_vf = model.policy.mlp_extractor.forward_critic(latent_vf)

            logits_1 = model.policy._get_value_logits(latent_vf)
            logits_2 = model.policy._get_value_logits_2(latent_vf)

            value_1 = logits_1.mean(dim=-1, keepdim=True)
            value_2 = logits_2.mean(dim=-1, keepdim=True)
            expected_min = torch.min(value_1, value_2)

        assert torch.allclose(
            value_pred, expected_min, atol=1e-6
        ), "Should use min(V1, V2) by default"


class TestOptimizerInclusion:
    """Test that both critics are included in optimizer by default."""

    def test_default_includes_both_critics(self):
        """Test that optimizer includes both critics by default."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                # use_twin_critics defaults to True
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )

        # Get optimizer parameter IDs
        optimizer_param_ids = {
            id(p) for group in policy.optimizer.param_groups for p in group["params"]
        }

        # Get critic parameter IDs
        critic1_ids = {id(p) for p in policy.quantile_head.parameters()}
        critic2_ids = {id(p) for p in policy.quantile_head_2.parameters()}

        # Both should be in optimizer
        assert critic1_ids.issubset(optimizer_param_ids), "Critic 1 missing from optimizer"
        assert critic2_ids.issubset(optimizer_param_ids), "Critic 2 missing from optimizer"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
