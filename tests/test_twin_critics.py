"""
Comprehensive tests for Twin Critics integration.

Tests cover:
- Policy architecture with twin critics
- Forward pass with both critics
- Loss computation for both critics
- Minimum value selection
- Integration with DistributionalPPO
- Backward compatibility (twin critics disabled)
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from custom_policy_patch1 import CustomActorCriticPolicy, QuantileValueHead
from distributional_ppo import DistributionalPPO


class TestTwinCriticsArchitecture:
    """Test Twin Critics network architecture creation."""

    def test_twin_critics_quantile_creation(self):
        """Test that both quantile critics are created when twin critics enabled."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "huber_kappa": 1.0,
                "use_twin_critics": True,  # Enable twin critics
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        # Check that twin critics flag is set
        assert policy._use_twin_critics is True

        # Check that both quantile heads exist
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is not None
        assert isinstance(policy.quantile_head, QuantileValueHead)
        assert isinstance(policy.quantile_head_2, QuantileValueHead)

        # Check that both heads have same architecture
        assert policy.quantile_head.num_quantiles == policy.quantile_head_2.num_quantiles
        assert policy.quantile_head.huber_kappa == policy.quantile_head_2.huber_kappa

        # Check that heads have independent parameters
        assert (
            policy.quantile_head.linear.weight.data_ptr()
            != policy.quantile_head_2.linear.weight.data_ptr()
        )

    def test_twin_critics_categorical_creation(self):
        """Test that both categorical critics are created when twin critics enabled."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "num_atoms": 51,
            "critic": {
                "distributional": False,
                "use_twin_critics": True,  # Enable twin critics
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        # Check that twin critics flag is set
        assert policy._use_twin_critics is True

        # Check that both categorical heads exist
        assert policy.dist_head is not None
        assert policy.dist_head_2 is not None
        assert isinstance(policy.dist_head, nn.Linear)
        assert isinstance(policy.dist_head_2, nn.Linear)

        # Check that heads have same output dimension
        assert policy.dist_head.out_features == policy.dist_head_2.out_features

        # Check that heads have independent parameters
        assert policy.dist_head.weight.data_ptr() != policy.dist_head_2.weight.data_ptr()

    def test_twin_critics_enabled_by_default(self):
        """Test that twin critics are enabled by default."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                # use_twin_critics NOT set - should default to True
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        # Check that twin critics are enabled by default
        assert policy._use_twin_critics is True
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is not None

    def test_twin_critics_explicit_disable(self):
        """Test that twin critics can be explicitly disabled."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "use_twin_critics": False,  # Explicitly disable
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        # Check that twin critics are disabled
        assert policy._use_twin_critics is False
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is None
        assert policy.dist_head_2 is None


class TestTwinCriticsForward:
    """Test forward passes through twin critics."""

    @pytest.fixture
    def twin_critics_policy(self):
        """Create policy with twin critics enabled."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "huber_kappa": 1.0,
                "use_twin_critics": True,
            },
        }

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

    def test_get_twin_value_logits(self, twin_critics_policy):
        """Test retrieving both critics' outputs."""
        batch_size = 4
        latent_dim = 32

        latent_vf = torch.randn(batch_size, latent_dim)

        # Get both critics' logits
        logits_1, logits_2 = twin_critics_policy._get_twin_value_logits(latent_vf)

        # Check shapes
        assert logits_1.shape == (batch_size, 16)  # 16 quantiles
        assert logits_2.shape == (batch_size, 16)

        # Check that outputs are different (independent networks)
        assert not torch.allclose(logits_1, logits_2)

    def test_get_min_twin_values(self, twin_critics_policy):
        """Test minimum value selection from twin critics."""
        batch_size = 4
        latent_dim = 32

        latent_vf = torch.randn(batch_size, latent_dim)

        # Get minimum values
        min_values = twin_critics_policy._get_min_twin_values(latent_vf)

        # Check shape
        assert min_values.shape == (batch_size, 1)

        # Get individual critic values for verification
        logits_1, logits_2 = twin_critics_policy._get_twin_value_logits(latent_vf)
        value_1 = logits_1.mean(dim=-1, keepdim=True)
        value_2 = logits_2.mean(dim=-1, keepdim=True)

        # Check that min_values is indeed the minimum
        expected_min = torch.min(value_1, value_2)
        assert torch.allclose(min_values, expected_min)

    def test_get_value_logits_2_error_when_disabled(self):
        """Test that accessing second critic raises error when twin critics disabled."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "use_twin_critics": False,  # Disabled
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        latent_vf = torch.randn(4, 32)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Second critic is not enabled"):
            policy._get_value_logits_2(latent_vf)


class TestTwinCriticsLoss:
    """Test loss computation for twin critics."""

    def test_twin_critics_loss_quantile(self):
        """Test twin critics loss computation for quantile critics."""
        from unittest.mock import Mock

        # Create mock policy
        policy = Mock()
        policy._use_twin_critics = True
        policy._use_quantile_value_head = True

        # Create mock PPO instance
        ppo = DistributionalPPO.__new__(DistributionalPPO)
        ppo.policy = policy
        ppo._quantile_huber_kappa = 1.0

        # Create mock quantile levels
        num_quantiles = 16
        quantile_levels = torch.linspace(0.0, 1.0, num_quantiles + 1)[:-1] + 0.5 / num_quantiles
        ppo._quantile_levels_tensor = lambda device: quantile_levels

        # Setup mock returns for policy methods
        batch_size = 4
        quantiles_1 = torch.randn(batch_size, num_quantiles)
        quantiles_2 = torch.randn(batch_size, num_quantiles)

        policy._get_value_logits = Mock(return_value=quantiles_1)
        policy._get_value_logits_2 = Mock(return_value=quantiles_2)

        # Create latent and targets
        latent_vf = torch.randn(batch_size, 32)
        targets = torch.randn(batch_size, 1)

        # Compute twin critics loss
        loss_1, loss_2, min_values = ppo._twin_critics_loss(latent_vf, targets, reduction="mean")

        # Check that both losses are computed
        assert loss_1 is not None
        assert loss_2 is not None
        assert isinstance(loss_1, torch.Tensor)
        assert isinstance(loss_2, torch.Tensor)

        # Check that both are scalars (reduction='mean')
        assert loss_1.ndim == 0
        assert loss_2.ndim == 0

        # Check that min_values is computed
        assert min_values is not None
        assert min_values.shape == (batch_size, 1)

    def test_twin_critics_loss_disabled(self):
        """Test that twin critics loss returns None for second critic when disabled."""
        from unittest.mock import Mock

        # Create mock policy with twin critics disabled
        policy = Mock()
        policy._use_twin_critics = False
        policy._use_quantile_value_head = True

        # Create mock PPO instance
        ppo = DistributionalPPO.__new__(DistributionalPPO)
        ppo.policy = policy
        ppo._quantile_huber_kappa = 1.0

        # Create mock quantile levels
        num_quantiles = 16
        quantile_levels = torch.linspace(0.0, 1.0, num_quantiles + 1)[:-1] + 0.5 / num_quantiles
        ppo._quantile_levels_tensor = lambda device: quantile_levels

        # Setup mock return
        batch_size = 4
        quantiles_1 = torch.randn(batch_size, num_quantiles)
        policy._get_value_logits = Mock(return_value=quantiles_1)

        # Create latent and targets
        latent_vf = torch.randn(batch_size, 32)
        targets = torch.randn(batch_size, 1)

        # Compute loss
        loss_1, loss_2, min_values = ppo._twin_critics_loss(latent_vf, targets, reduction="mean")

        # Check that only first loss is computed
        assert loss_1 is not None
        assert loss_2 is None
        assert min_values is None


class TestTwinCriticsGradients:
    """Test gradient flow through twin critics."""

    def test_independent_gradients(self):
        """Test that twin critics have independent gradients."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "huber_kappa": 1.0,
                "use_twin_critics": True,
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        batch_size = 4
        latent_vf = torch.randn(batch_size, 32, requires_grad=True)

        # Forward pass through both critics
        logits_1 = policy._get_value_logits(latent_vf)
        logits_2 = policy._get_value_logits_2(latent_vf)

        # Create dummy loss
        loss_1 = logits_1.mean()
        loss_2 = logits_2.mean()

        # Backward through first critic only
        loss_1.backward()

        # Check that only first critic has gradients
        assert policy.quantile_head.linear.weight.grad is not None
        assert policy.quantile_head_2.linear.weight.grad is None  # Should be None

        # Reset gradients
        policy.zero_grad()

        # Backward through second critic only
        logits_2 = policy._get_value_logits_2(latent_vf)
        loss_2 = logits_2.mean()
        loss_2.backward()

        # Check that only second critic has gradients
        assert policy.quantile_head.linear.weight.grad is None
        assert policy.quantile_head_2.linear.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
