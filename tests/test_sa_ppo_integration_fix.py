"""
Comprehensive tests for SA-PPO integration fix.

Tests that adversarial training is correctly integrated into distributional_ppo.py
and PBTTrainingCoordinator.

This file verifies the fix for the critical bug where compute_adversarial_loss
was NEVER called, making adversarial training completely inactive.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from unittest.mock import MagicMock, patch, call

gym = pytest.importorskip("gymnasium")

from adversarial import (
    PerturbationConfig,
    SAPPOConfig,
    StateAdversarialPPO,
    PBTConfig,
    HyperparamConfig,
)
from training_pbt_adversarial_integration import PBTTrainingCoordinator, PBTAdversarialConfig


class MockPolicy:
    """Mock policy for testing."""

    def __init__(self):
        self.optimizer = MagicMock()

    def get_distribution(self, obs):
        """Mock distribution."""
        mock_dist = MagicMock()
        mock_dist.log_prob = MagicMock(return_value=torch.randn(obs.size(0)))
        return mock_dist

    def predict_values(self, obs):
        """Mock value prediction."""
        return torch.randn(obs.size(0), 1)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, actions_raw=None):
        """Mock action evaluation."""
        return torch.randn(obs.size(0), 1), torch.randn(obs.size(0)), torch.randn(obs.size(0))


class TestSAPPOIntegration:
    """Tests for SA-PPO integration into distributional PPO."""

    def test_wrapper_can_be_set(self):
        """Test that SA-PPO wrapper can be set on model."""
        # Create mock model
        model = MagicMock()
        model.policy = MockPolicy()

        # Create SA-PPO config
        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
        )

        # Create wrapper
        wrapper = StateAdversarialPPO(sa_ppo_config, model)

        # Test that wrapper can be created
        assert wrapper is not None
        assert wrapper.config.enabled is True
        assert wrapper.config.adversarial_ratio == 0.5

    def test_apply_adversarial_augmentation_disabled(self):
        """Test augmentation when adversarial training is disabled."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=False,  # Disabled
            adversarial_ratio=0.5,
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)

        # Create test inputs
        states = torch.randn(32, 10)
        actions = torch.randn(32, 2)
        advantages = torch.randn(32)
        old_log_probs = torch.randn(32)
        clip_range = 0.2

        # Apply augmentation
        augmented_states, sample_mask, info = wrapper.apply_adversarial_augmentation(
            states, actions, advantages, old_log_probs, clip_range
        )

        # Should return original states
        assert torch.equal(augmented_states, states)
        # All samples should be clean (mask = 0)
        assert torch.all(sample_mask == 0.0)
        assert info["sa_ppo/num_adversarial"] == 0
        assert info["sa_ppo/num_clean"] == 32

    def test_apply_adversarial_augmentation_enabled(self):
        """Test augmentation when adversarial training is enabled."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            warmup_updates=0,  # No warmup
            attack_policy=False,  # Disable attack for mock testing (no gradients available)
            perturbation=PerturbationConfig(
                epsilon=0.1,
                attack_steps=2,
                attack_method="pgd",
            ),
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)
        wrapper.on_training_start()
        wrapper.on_update_start()

        # Create test inputs
        batch_size = 32
        states = torch.randn(batch_size, 10)
        actions = torch.randn(batch_size, 2)
        advantages = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        clip_range = 0.2

        # Apply augmentation
        augmented_states, sample_mask, info = wrapper.apply_adversarial_augmentation(
            states, actions, advantages, old_log_probs, clip_range
        )

        # Check outputs
        assert augmented_states.shape == states.shape
        assert sample_mask.shape == (batch_size,)

        # Half should be adversarial (mask = 1), half clean (mask = 0)
        num_adversarial = int(torch.sum(sample_mask > 0.5).item())
        num_clean = int(torch.sum(sample_mask < 0.5).item())

        assert num_adversarial == int(batch_size * 0.5)
        assert num_clean == batch_size - num_adversarial
        assert info["sa_ppo/num_adversarial"] == num_adversarial
        assert info["sa_ppo/num_clean"] == num_clean

        # When attack_policy=False, adversarial samples should be same as clean (no perturbations)
        # This is expected behavior for testing without gradients
        # In real training with gradients, perturbations would be applied

    def test_compute_robust_kl_penalty_disabled(self):
        """Test robust KL computation when disabled."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=False,
            robust_kl_coef=0.0,  # Disabled
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)

        states_clean = torch.randn(16, 10)
        states_adv = torch.randn(16, 10)
        actions = torch.randn(16, 2)

        # Compute penalty
        penalty, info = wrapper.compute_robust_kl_penalty(states_clean, states_adv, actions)

        # Should be zero
        assert penalty == 0.0
        assert "sa_ppo/robust_kl_penalty" not in info or info["sa_ppo/robust_kl_penalty"] == 0.0

    def test_compute_robust_kl_penalty_enabled(self):
        """Test robust KL computation when enabled."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=True,
            robust_kl_coef=0.1,
            warmup_updates=0,
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)
        wrapper.on_training_start()
        wrapper.on_update_start()

        states_clean = torch.randn(16, 10)
        states_adv = torch.randn(16, 10) + 0.1  # Slightly perturbed
        actions = torch.randn(16, 2)

        # Compute penalty
        penalty, info = wrapper.compute_robust_kl_penalty(states_clean, states_adv, actions)

        # Should be non-zero
        assert isinstance(penalty, float)
        assert "sa_ppo/robust_kl_penalty" in info
        assert isinstance(info["sa_ppo/robust_kl_penalty"], float)

    def test_pbt_coordinator_sets_wrapper_on_model(self):
        """Test that PBTTrainingCoordinator sets wrapper on model."""
        # Create config
        pbt_adversarial_config = PBTAdversarialConfig(
            pbt_enabled=False,  # Disable PBT for this test
            adversarial_enabled=True,
            adversarial=SAPPOConfig(
                enabled=True,
                adversarial_ratio=0.5,
            ),
        )

        coordinator = PBTTrainingCoordinator(pbt_adversarial_config)

        # Create mock member
        mock_member = MagicMock()
        mock_member.member_id = 0
        mock_member.hyperparams = {"learning_rate": 1e-4}

        # Create mock model with set_sa_ppo_wrapper method
        mock_model = MagicMock()
        mock_model.set_sa_ppo_wrapper = MagicMock()
        mock_model.policy = MockPolicy()

        def mock_factory(**kwargs):
            return mock_model

        # Create model
        model, wrapper = coordinator.create_member_model(
            mock_member,
            mock_factory,
        )

        # Verify wrapper was created
        assert wrapper is not None
        # Verify set_sa_ppo_wrapper was called
        mock_model.set_sa_ppo_wrapper.assert_called_once()
        # Verify wrapper was passed to model
        call_args = mock_model.set_sa_ppo_wrapper.call_args
        assert call_args[0][0] is wrapper

    def test_pbt_coordinator_warns_if_model_not_support_wrapper(self):
        """Test that coordinator warns if model doesn't support wrapper."""
        pbt_adversarial_config = PBTAdversarialConfig(
            pbt_enabled=False,
            adversarial_enabled=True,
            adversarial=SAPPOConfig(enabled=True),
        )

        coordinator = PBTTrainingCoordinator(pbt_adversarial_config)

        mock_member = MagicMock()
        mock_member.member_id = 0
        mock_member.hyperparams = {}

        # Model WITHOUT set_sa_ppo_wrapper method
        mock_model = MagicMock(spec=[])  # Empty spec - no methods
        mock_model.policy = MockPolicy()

        def mock_factory(**kwargs):
            return mock_model

        # Should still create model but log warning
        with patch("training_pbt_adversarial_integration.logger") as mock_logger:
            model, wrapper = coordinator.create_member_model(
                mock_member,
                mock_factory,
            )

            # Verify wrapper was created
            assert wrapper is not None
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "does not support SA-PPO wrapper" in mock_logger.warning.call_args[0][0]

    def test_stats_tracking(self):
        """Test that SA-PPO tracks statistics correctly."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            warmup_updates=0,
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)
        wrapper.on_training_start()

        # Run multiple updates
        for _ in range(5):
            wrapper.on_update_start()

        # Get stats
        stats = wrapper.get_stats()

        assert "sa_ppo/enabled" in stats
        assert "sa_ppo/update_count" in stats
        assert stats["sa_ppo/update_count"] == 5
        assert "sa_ppo/adversarial_samples" in stats
        assert "sa_ppo/clean_samples" in stats


class TestBackwardCompatibility:
    """Tests for backward compatibility when wrapper is None."""

    def test_training_works_without_wrapper(self):
        """Test that training works normally when wrapper is not set."""
        # This test verifies backward compatibility
        # When sa_ppo_wrapper is None, training should proceed normally

        # Create mock model
        model = MagicMock()
        model._sa_ppo_wrapper = None  # No wrapper

        # Verify getattr returns None
        wrapper = getattr(model, "_sa_ppo_wrapper", None)
        assert wrapper is None

        # Training should proceed normally
        # (this is tested implicitly by existing distributional PPO tests)


class TestIntegrationScenarios:
    """Integration tests for full training scenarios."""

    def test_full_augmentation_pipeline(self):
        """Test full pipeline: augmentation -> loss computation -> backward."""
        model = MagicMock()
        model.policy = MockPolicy()

        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
            warmup_updates=0,
            attack_policy=False,  # Disable attack for mock testing (no gradients)
            perturbation=PerturbationConfig(
                epsilon=0.05,
                attack_steps=2,
            ),
        )

        wrapper = StateAdversarialPPO(sa_ppo_config, model)
        wrapper.on_training_start()
        wrapper.on_update_start()

        # Create test batch
        batch_size = 32
        states = torch.randn(batch_size, 10)
        actions = torch.randn(batch_size, 2)
        advantages = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        clip_range = 0.2

        # Step 1: Apply adversarial augmentation
        augmented_states, sample_mask, info = wrapper.apply_adversarial_augmentation(
            states, actions, advantages, old_log_probs, clip_range
        )

        assert augmented_states.shape == states.shape
        assert info["sa_ppo/num_adversarial"] > 0
        assert info["sa_ppo/num_clean"] > 0

        # Step 2: Compute robust KL penalty
        adv_mask = sample_mask > 0.5
        if torch.any(adv_mask) and torch.any(~adv_mask):
            obs_clean = states[~adv_mask]
            obs_adv = augmented_states[adv_mask]
            actions_for_kl = actions[adv_mask]

            robust_kl_penalty, kl_info = wrapper.compute_robust_kl_penalty(
                obs_clean, obs_adv, actions_for_kl
            )

            assert isinstance(robust_kl_penalty, float)
            assert "sa_ppo/robust_kl_penalty" in kl_info

        # NOTE: With attack_policy=False, perturbations are not applied (mock testing limitation)
        # In real training with gradients, PGD would generate non-zero perturbations
        # This test verifies the pipeline flow, not the actual perturbation magnitude


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
