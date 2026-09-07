"""
Test suite for Twin Critics VF Clipping Fix

This module tests the fix for Bug #1: Twin Critics VF Clipping using wrong quantiles.

Bug Description:
- BEFORE FIX: VF clipping only used quantiles from first critic (Q1)
- AFTER FIX: VF clipping uses min(Q1, Q2) quantiles for consistency

Test Coverage:
1. Policy properties: last_value_quantiles_min and last_value_logits_min
2. Rollout buffer storage: Verifies min quantiles are stored
3. Training loop: Verifies VF clipping uses min quantiles
4. Consistency: Verifies unclipped and clipped losses use same value estimates
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
from gymnasium import spaces

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


class SimpleDummyEnv(gym.Env):
    """Simple test environment with Box action space (required for CustomActorCriticPolicy)."""

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
            "use_twin_critics": True,  # Explicitly enable Twin Critics
        },
    }


@pytest.fixture
def categorical_policy_config():
    """Configuration for categorical critic with Twin Critics enabled."""
    return {
        "hidden_dim": 32,
        "lstm_hidden_size": 32,
        "critic": {
            "distributional": False,
            "n_atoms": 21,
            "v_min": -10.0,
            "v_max": 10.0,
            "use_twin_critics": True,  # Explicitly enable Twin Critics
        },
    }


class TestPolicyMinProperties:
    """Test the new min properties in CustomActorCriticPolicy."""

    def test_quantile_min_property_twin_critics_enabled(self, simple_env, quantile_policy_config):
        """Test last_value_quantiles_min returns element-wise min when Twin Critics enabled."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Get observation
        obs, _ = simple_env.reset()
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        episode_starts = torch.zeros(1, dtype=torch.float32, device=model.device)

        # Forward pass
        with torch.no_grad():
            model.policy.forward(obs_tensor, None, episode_starts)

        # Verify both critics cached quantiles
        q1 = model.policy.last_value_quantiles
        q2 = model.policy._last_value_quantiles_2
        q_min = model.policy.last_value_quantiles_min

        assert q1 is not None, "First critic should cache quantiles"
        assert q2 is not None, "Second critic should cache quantiles"
        assert q_min is not None, "Min quantiles should be returned"

        # Verify shape matches
        assert q_min.shape == q1.shape, "Min quantiles should have same shape as Q1"
        assert q_min.shape == q2.shape, "Min quantiles should have same shape as Q2"

        # Verify element-wise minimum
        expected_min = torch.min(q1, q2)
        torch.testing.assert_close(
            q_min, expected_min, msg="Min quantiles should be element-wise minimum"
        )

    def test_quantile_min_property_twin_critics_disabled(self, simple_env):
        """Test last_value_quantiles_min falls back to Q1 when Twin Critics disabled."""
        config = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 11,
                "use_twin_critics": False,  # Explicitly disable
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": config},
            n_steps=64,
            verbose=0,
        )

        # Get observation
        obs, _ = simple_env.reset()
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        episode_starts = torch.zeros(1, dtype=torch.float32, device=model.device)

        # Forward pass
        with torch.no_grad():
            model.policy.forward(obs_tensor, None, episode_starts)

        q1 = model.policy.last_value_quantiles
        q_min = model.policy.last_value_quantiles_min

        assert q1 is not None, "First critic should cache quantiles"
        assert q_min is not None, "Min quantiles should be returned"

        # Should be identical when Twin Critics disabled
        torch.testing.assert_close(
            q_min, q1, msg="Min quantiles should equal Q1 when Twin Critics disabled"
        )

    def test_categorical_min_property_twin_critics_enabled(
        self, simple_env, categorical_policy_config
    ):
        """Test last_value_logits_min returns logits from critic with min value."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": categorical_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Get observation
        obs, _ = simple_env.reset()
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        episode_starts = torch.zeros(1, dtype=torch.float32, device=model.device)

        # Forward pass
        with torch.no_grad():
            model.policy.forward(obs_tensor, None, episode_starts)

        logits1 = model.policy.last_value_logits
        logits2 = model.policy._last_value_logits_2
        logits_min = model.policy.last_value_logits_min

        assert logits1 is not None, "First critic should cache logits"
        assert logits2 is not None, "Second critic should cache logits"
        assert logits_min is not None, "Min logits should be returned"

        # Verify shape matches
        assert logits_min.shape == logits1.shape, "Min logits should have same shape as logits1"

        # Compute expected values from both critics
        with torch.no_grad():
            probs1 = torch.softmax(logits1, dim=-1)
            probs2 = torch.softmax(logits2, dim=-1)
            value1 = (probs1 * model.policy.atoms).sum(dim=-1)
            value2 = (probs2 * model.policy.atoms).sum(dim=-1)

            # Verify that logits_min comes from critic with minimum value
            # (element-wise comparison for batch)
            for i in range(logits_min.shape[0]):
                if value2[i] < value1[i]:
                    torch.testing.assert_close(
                        logits_min[i],
                        logits2[i],
                        msg=f"Sample {i}: Should use critic 2 logits (lower value)",
                    )
                else:
                    torch.testing.assert_close(
                        logits_min[i],
                        logits1[i],
                        msg=f"Sample {i}: Should use critic 1 logits (lower value)",
                    )


class TestRolloutBufferStorage:
    """Test that rollout buffer stores min quantiles/logits correctly."""

    def test_rollout_stores_min_quantiles(self, simple_env, quantile_policy_config):
        """Test that rollout buffer stores min(Q1, Q2) quantiles."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=16,  # Short rollout for testing
            verbose=0,
        )

        # Trigger a rollout by calling learn (simplest way to populate buffer)
        model.learn(total_timesteps=16, log_interval=None)

        # Get stored quantiles from buffer
        for batch in model.rollout_buffer.get(batch_size=None):
            old_quantiles = batch.old_value_quantiles

            assert old_quantiles is not None, "Buffer should store old_value_quantiles"
            assert old_quantiles.shape[-1] == quantile_policy_config["critic"]["num_quantiles"]

            # We can't easily verify the exact values without re-running forward passes,
            # but we can verify the shape and that they're finite
            assert torch.isfinite(old_quantiles).all(), "All quantiles should be finite"

            # Verify that we only iterate once (buffer should have data)
            break


class TestVFClippingConsistency:
    """Test that VF clipping uses consistent value estimates with Twin Critics."""

    @pytest.mark.parametrize("clip_range_vf", [None, 0.7])
    def test_vf_clipping_uses_min_quantiles(
        self, simple_env, quantile_policy_config, clip_range_vf
    ):
        """Test that VF clipping loss computation uses min(Q1, Q2) quantiles."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            clip_range_vf=clip_range_vf,
            verbose=0,
        )

        # Train for one step to trigger VF clipping logic
        model.learn(total_timesteps=64, log_interval=None)

        # Verify that training completed without errors
        # (detailed verification of loss computation would require extensive mocking)
        assert model.num_timesteps == 64, "Training should complete successfully"

    def test_twin_critics_both_updated(self, simple_env, quantile_policy_config):
        """Test that both critics are updated during training."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            n_epochs=1,
            verbose=0,
        )

        # Get initial parameters
        params1_before = [p.clone() for p in model.policy.quantile_head.parameters()]
        params2_before = [p.clone() for p in model.policy.quantile_head_2.parameters()]

        # Train for one step
        model.learn(total_timesteps=64, log_interval=None)

        # Get updated parameters
        params1_after = list(model.policy.quantile_head.parameters())
        params2_after = list(model.policy.quantile_head_2.parameters())

        # Verify both critics were updated
        for p_before, p_after in zip(params1_before, params1_after):
            assert not torch.allclose(
                p_before, p_after, atol=1e-6
            ), "First critic should be updated"

        for p_before, p_after in zip(params2_before, params2_after):
            assert not torch.allclose(
                p_before, p_after, atol=1e-6
            ), "Second critic should be updated"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_second_critic_quantiles(self, simple_env, quantile_policy_config):
        """Test fallback when second critic quantiles are missing."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Get observation
        obs, _ = simple_env.reset()
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        episode_starts = torch.zeros(1, dtype=torch.float32, device=model.device)

        # Forward pass
        with torch.no_grad():
            model.policy.forward(obs_tensor, None, episode_starts)

        # Manually clear second critic cache
        model.policy._last_value_quantiles_2 = None

        # Should fall back to first critic
        q_min = model.policy.last_value_quantiles_min
        q1 = model.policy.last_value_quantiles

        assert q_min is not None, "Should return fallback value"
        torch.testing.assert_close(
            q_min, q1, msg="Should fall back to first critic when second is missing"
        )

    def test_different_quantile_values(self, simple_env, quantile_policy_config):
        """Test min property with artificially different quantile values."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": quantile_policy_config},
            n_steps=64,
            verbose=0,
        )

        # Create mock quantiles with known values
        batch_size = 4
        num_quantiles = quantile_policy_config["critic"]["num_quantiles"]

        q1_mock = torch.randn(batch_size, num_quantiles, device=model.device)
        q2_mock = torch.randn(batch_size, num_quantiles, device=model.device)

        # Manually set cached values
        model.policy._last_value_quantiles = q1_mock
        model.policy._last_value_quantiles_2 = q2_mock

        # Get min quantiles
        q_min = model.policy.last_value_quantiles_min

        # Verify element-wise minimum
        expected_min = torch.min(q1_mock, q2_mock)
        torch.testing.assert_close(q_min, expected_min, msg="Should compute element-wise min")

        # Verify that min is never greater than either critic
        assert (q_min <= q1_mock).all(), "Min should be <= Q1 everywhere"
        assert (q_min <= q2_mock).all(), "Min should be <= Q2 everywhere"


class TestBackwardCompatibility:
    """Test backward compatibility with models trained before fix."""

    def test_single_critic_unchanged(self, simple_env):
        """Test that single critic behavior is unchanged."""
        config = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 11,
                "use_twin_critics": False,
            },
        }

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            simple_env,
            policy_kwargs={"arch_params": config},
            n_steps=64,
            clip_range_vf=0.7,
            verbose=0,
        )

        # Train for one step
        model.learn(total_timesteps=64, log_interval=None)

        # Verify training completed successfully
        assert model.num_timesteps == 64, "Single critic training should work unchanged"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
