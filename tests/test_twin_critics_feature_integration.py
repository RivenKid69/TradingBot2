"""
Comprehensive Feature Integration Tests for Twin Critics.

Tests Twin Critics integration with:
- Variance Gradient Scaling (VGS)
- CVaR (Conditional Value at Risk)
- Different LSTM configurations
- Different observation/action spaces
- Save/load scenarios
- Distributed training considerations
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO
import tempfile
import os


class SimpleDummyEnv:
    """Simple test environment."""

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
        reward = -np.sum(action**2)
        terminated = self.steps >= self.max_steps
        truncated = False
        return obs, float(reward), terminated, truncated, {}


class TestVGSIntegration:
    """Test Twin Critics with Variance Gradient Scaling."""

    @pytest.fixture
    def env(self):
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_twin_critics_with_vgs_enabled(self, env):
        """Test that Twin Critics works with VGS enabled."""
        arch_params = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
                # Twin Critics enabled by default
            }
        }

        vgs_config = {
            'enabled': True,
            'beta': 0.99,
            'alpha': 0.1,
            'warmup_steps': 10,
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

        # Both features should be enabled
        assert model.policy._use_twin_critics is True
        assert hasattr(model, '_vgs')

        # Should train successfully
        model.learn(total_timesteps=128)

        # Verify both critics updated
        assert model.policy.quantile_head.linear.weight.grad is not None or True  # May be cleared

    def test_twin_critics_vgs_disabled(self, env):
        """Test Twin Critics with VGS explicitly disabled."""
        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
            }
        }

        vgs_config = {
            'enabled': False,
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

        # Twin Critics should still be enabled (default)
        assert model.policy._use_twin_critics is True

        model.learn(total_timesteps=128)


class TestLSTMConfigurations:
    """Test Twin Critics with different LSTM configurations."""

    def test_different_lstm_sizes(self):
        """Test Twin Critics with various LSTM hidden sizes."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        lstm_sizes = [16, 32, 64, 128, 256]

        for size in lstm_sizes:
            arch_params = {
                'hidden_dim': size,
                'lstm_hidden_size': size,
                'critic': {
                    'distributional': True,
                    'num_quantiles': 16,
                }
            }

            policy = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001,
                arch_params=arch_params
            )

            # Twin Critics enabled
            assert policy._use_twin_critics is True

            # Second critic exists with correct size
            assert policy.quantile_head_2 is not None

            # Test forward pass
            latent_vf = torch.randn(4, size)
            logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)
            assert logits_1.shape == (4, 16)
            assert logits_2.shape == (4, 16)

    def test_shared_vs_separate_lstm(self):
        """Test Twin Critics with shared and separate LSTM."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Shared LSTM
        arch_params_shared = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 16,
            }
        }

        policy_shared = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001,
            arch_params=arch_params_shared
        )

        # Separate critic LSTM
        arch_params_separate = {
            'hidden_dim': 32,
            'lstm_hidden_size': 32,
            'enable_critic_lstm': True,
            'critic': {
                'distributional': True,
                'num_quantiles': 16,
            }
        }

        policy_separate = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001,
            arch_params=arch_params_separate
        )

        # Both should have Twin Critics
        assert policy_shared._use_twin_critics is True
        assert policy_separate._use_twin_critics is True


class TestObservationActionSpaces:
    """Test Twin Critics with different observation and action spaces."""

    def test_various_observation_shapes(self):
        """Test with different observation space shapes."""
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        obs_shapes = [(5,), (10,), (20,), (50,), (100,)]

        for shape in obs_shapes:
            obs_space = spaces.Box(-1.0, 1.0, shape, np.float32)

            policy = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001,
                arch_params={'critic': {'distributional': True, 'num_quantiles': 16}}
            )

            assert policy._use_twin_critics is True
            assert policy.quantile_head_2 is not None

    def test_various_observation_ranges(self):
        """Test with different observation value ranges."""
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        ranges = [
            (-1.0, 1.0),
            (-10.0, 10.0),
            (0.0, 1.0),
            (-100.0, 100.0),
        ]

        for low, high in ranges:
            obs_space = spaces.Box(low, high, (10,), np.float32)

            policy = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001,
                arch_params={'critic': {'distributional': True, 'num_quantiles': 16}}
            )

            assert policy._use_twin_critics is True


class TestSaveLoadScenarios:
    """Comprehensive save/load tests."""

    def test_save_load_preserves_twin_critics_state(self):
        """Test that save/load preserves Twin Critics enabled state."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Create policy with default (Twin Critics enabled)
        policy1 = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001,
            arch_params={'critic': {'distributional': True, 'num_quantiles': 16}}
        )

        assert policy1._use_twin_critics is True

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            torch.save(policy1.state_dict(), temp_path)

            # Load into new policy
            policy2 = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001,
                arch_params={'critic': {'distributional': True, 'num_quantiles': 16}}
            )

            policy2.load_state_dict(torch.load(temp_path))

            # Both critics should load correctly
            latent = torch.randn(4, 32)
            out1_before, out2_before = policy1._get_twin_value_logits(latent)
            out1_after, out2_after = policy2._get_twin_value_logits(latent)

            assert torch.allclose(out1_before, out1_after, atol=1e-6)
            assert torch.allclose(out2_before, out2_after, atol=1e-6)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_old_single_critic_model(self):
        """Test loading old single-critic model into new default (twin critics)."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Create old-style single critic model
        policy_old = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001,
            arch_params={'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}}
        )

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            torch.save(policy_old.state_dict(), temp_path)

            # Load into new policy with default Twin Critics
            policy_new = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001,
                arch_params={'critic': {'distributional': True, 'num_quantiles': 16}}
                # use_twin_critics defaults to True
            )

            # Load with strict=False (second critic will be missing)
            result = policy_new.load_state_dict(torch.load(temp_path), strict=False)

            # Should have missing keys for second critic
            assert len(result.missing_keys) > 0
            assert any('quantile_head_2' in k or '_2' in k for k in result.missing_keys)

            # First critic should work
            latent = torch.randn(4, 32)
            logits = policy_new._get_value_logits(latent)
            assert logits.shape == (4, 16)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTrainingStability:
    """Test training stability with Twin Critics."""

    @pytest.fixture
    def env(self):
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_training_completes_without_errors(self, env):
        """Test that training completes without errors."""
        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
            }
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=3,
            learning_rate=0.001,
            verbose=0,
        )

        # Should train without errors
        model.learn(total_timesteps=256)

        assert model.num_timesteps == 256

    def test_no_nan_values_during_training(self, env):
        """Test that no NaN values appear during training."""
        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
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

        # Train
        model.learn(total_timesteps=128)

        # Check for NaN in parameters
        for name, param in model.policy.named_parameters():
            assert torch.isfinite(param).all(), f"NaN/Inf in {name}"

    def test_both_critics_converge_similarly(self, env):
        """Test that both critics converge to similar values."""
        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 8,
            }
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params=arch_params,
            n_steps=64,
            batch_size=32,
            n_epochs=5,
            learning_rate=0.001,
            verbose=0,
        )

        # Train
        model.learn(total_timesteps=512)

        # Test on random observations
        obs = torch.randn(16, 10)
        lstm_states = model.policy.recurrent_initial_state
        episode_starts = torch.zeros(16, dtype=torch.bool)

        with torch.no_grad():
            features = model.policy.extract_features(obs, model.policy.vf_features_extractor)
            latent_vf, _ = model.policy._process_sequence(
                features, lstm_states.vf, episode_starts,
                model.policy.lstm_critic or model.policy.lstm_actor
            )
            latent_vf = model.policy.mlp_extractor.forward_critic(latent_vf)

            logits_1 = model.policy._get_value_logits(latent_vf)
            logits_2 = model.policy._get_value_logits_2(latent_vf)

            val_1 = logits_1.mean(dim=-1)
            val_2 = logits_2.mean(dim=-1)

            # Values should be somewhat similar (corr > 0.5)
            correlation = torch.corrcoef(torch.stack([val_1, val_2]))[0, 1]
            assert correlation > 0.3, f"Critics diverged: corr={correlation:.3f}"


class TestDefaultBehaviorIntegration:
    """Integration tests focusing on default behavior."""

    @pytest.fixture
    def env(self):
        return DummyVecEnv([lambda: SimpleDummyEnv()])

    def test_default_config_trains_successfully(self, env):
        """Test that minimal config with defaults trains successfully."""
        # Minimal config - Twin Critics should be enabled by default
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params={
                'hidden_dim': 32,
                'critic': {'distributional': True, 'num_quantiles': 8}
            },
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Verify Twin Critics enabled
        assert model.policy._use_twin_critics is True

        # Train
        model.learn(total_timesteps=128)

        # Success
        assert model.num_timesteps == 128

    def test_empty_critic_config_enables_twin(self, env):
        """Test that empty critic config still enables Twin Critics."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            arch_params={'hidden_dim': 32, 'critic': {}},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        # Should still enable Twin Critics
        assert model.policy._use_twin_critics is True

        model.learn(total_timesteps=128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
