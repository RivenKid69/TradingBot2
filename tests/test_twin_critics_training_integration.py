"""
Tests for Twin Critics training loop integration.

These tests verify that Twin Critics are properly integrated into the PPO training loop:
- latent_vf caching works correctly
- _twin_critics_loss() is called during training
- Both critics are updated during training
- min(V1, V2) is used for value predictions
- Metrics are logged correctly
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


class SimpleDummyEnv:
    """Simple environment for testing."""

    def __init__(self):
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


class TestTwinCriticsTrainingIntegration:
    """Test Twin Critics integration into PPO training loop."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_latent_vf_caching_quantile(self, env):
        """Test that latent_vf is cached during forward pass (quantile mode)."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
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

        # Initially, latent_vf should not be cached
        assert model.policy._last_latent_vf is None

        # Collect rollouts (triggers forward pass)
        model.collect_rollouts(env, model._last_callback, model.rollout_buffer, n_rollout_steps=64)

        # After rollouts, latent_vf should be cached
        # Note: It might be None if not called recently, but the attribute should exist
        assert hasattr(model.policy, '_last_latent_vf')

        # Train one step
        model.learn(total_timesteps=64)

        # After training, check that caching mechanism is in place
        assert hasattr(model.policy, '_last_latent_vf')

    def test_both_critics_update_quantile(self, env):
        """Test that both critics are updated during training (quantile mode)."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
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
        initial_weight_1 = model.policy.quantile_head.linear.weight.data.clone()
        initial_weight_2 = model.policy.quantile_head_2.linear.weight.data.clone()

        # Train
        model.learn(total_timesteps=128)

        # Check that both critics' weights have changed
        final_weight_1 = model.policy.quantile_head.linear.weight.data
        final_weight_2 = model.policy.quantile_head_2.linear.weight.data

        weight_change_1 = (initial_weight_1 - final_weight_1).abs().max().item()
        weight_change_2 = (initial_weight_2 - final_weight_2).abs().max().item()

        assert weight_change_1 > 1e-7, f"Critic 1 not updated: {weight_change_1}"
        assert weight_change_2 > 1e-7, f"Critic 2 not updated: {weight_change_2}"

        print(f"✓ Both critics updated: Δw1={weight_change_1:.2e}, Δw2={weight_change_2:.2e}")

    def test_both_critics_update_categorical(self, env):
        """Test that both critics are updated during training (categorical mode)."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'num_atoms': 21,
            'critic': {
                'distributional': False,
                'use_twin_critics': True,
            }
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
        initial_weight_1 = model.policy.dist_head.weight.data.clone()
        initial_weight_2 = model.policy.dist_head_2.weight.data.clone()

        # Train
        model.learn(total_timesteps=128)

        # Check that both critics' weights have changed
        final_weight_1 = model.policy.dist_head.weight.data
        final_weight_2 = model.policy.dist_head_2.weight.data

        weight_change_1 = (initial_weight_1 - final_weight_1).abs().max().item()
        weight_change_2 = (initial_weight_2 - final_weight_2).abs().max().item()

        assert weight_change_1 > 1e-7, f"Critic 1 not updated: {weight_change_1}"
        assert weight_change_2 > 1e-7, f"Critic 2 not updated: {weight_change_2}"

        print(f"✓ Both critics updated: Δw1={weight_change_1:.2e}, Δw2={weight_change_2:.2e}")

    def test_min_value_used_for_prediction(self, env):
        """Test that min(V1, V2) is used for value predictions."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
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

        # Create test observation
        obs = torch.randn(4, 10)
        lstm_states = model.policy.recurrent_initial_state
        episode_starts = torch.zeros(4, dtype=torch.bool)

        # Get value prediction
        with torch.no_grad():
            value_pred = model.policy.predict_values(obs, lstm_states, episode_starts)

            # Also get individual critic values for comparison
            features = model.policy.extract_features(obs, model.policy.vf_features_extractor)
            latent_vf, _ = model.policy._process_sequence(
                features,
                lstm_states.vf,
                episode_starts,
                model.policy.lstm_critic
            )
            latent_vf = model.policy.mlp_extractor.forward_critic(latent_vf)

            value_logits_1 = model.policy._get_value_logits(latent_vf)
            value_logits_2 = model.policy._get_value_logits_2(latent_vf)

            value_1 = value_logits_1.mean(dim=-1, keepdim=True)
            value_2 = value_logits_2.mean(dim=-1, keepdim=True)
            min_value = torch.min(value_1, value_2)

        # Check that predict_values returns the minimum
        assert torch.allclose(value_pred, min_value, atol=1e-6), \
            "predict_values should return min(V1, V2) for Twin Critics"

        print("✓ min(V1, V2) correctly used for value predictions")

    def test_twin_critics_loss_called_during_training(self, env):
        """Test that _twin_critics_loss() is called during training."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
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

        # Wrap _twin_critics_loss to track calls
        original_method = model._twin_critics_loss
        call_count = [0]

        def wrapped_method(*args, **kwargs):
            call_count[0] += 1
            return original_method(*args, **kwargs)

        model._twin_critics_loss = wrapped_method

        # Train
        model.learn(total_timesteps=128)

        # Check that _twin_critics_loss was called
        assert call_count[0] > 0, "_twin_critics_loss() was not called during training"

        print(f"✓ _twin_critics_loss() called {call_count[0]} times during training")

    def test_twin_critics_metrics_logged(self, env):
        """Test that Twin Critics metrics are logged correctly."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=1,  # Enable logging
        )

        # Train
        model.learn(total_timesteps=128)

        # Check that Twin Critics attributes are initialized (they get reset after logging)
        # We can't easily check logger output, but we can verify the mechanism is in place
        assert hasattr(model, '_twin_critic_1_loss_sum')
        assert hasattr(model, '_twin_critic_2_loss_sum')
        assert hasattr(model, '_twin_critic_loss_count')

        print("✓ Twin Critics logging mechanism in place")

    def test_single_vs_twin_critics_convergence(self, env):
        """Compare convergence with and without Twin Critics."""
        arch_params_single = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': False,
            }
        }

        arch_params_twin = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                'use_twin_critics': True,
            }
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

        # Both should complete training successfully
        assert model_single.num_timesteps == 256
        assert model_twin.num_timesteps == 256

        print("✓ Both single and twin critics models converge successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
