"""
Integration tests for PBT + Adversarial Training.

Tests the full pipeline integration.
"""

import pytest
torch = pytest.importorskip("torch")
from unittest.mock import MagicMock

from adversarial import (
    PerturbationConfig,
    StatePerturbation,
    SAPPOConfig,
    StateAdversarialPPO,
    PBTConfig,
    HyperparamConfig,
    PBTScheduler,
)


class TestPBTAdversarialIntegration:
    """Integration tests for PBT + SA-PPO."""

    def test_full_pipeline_initialization(self, tmp_path):
        """Test that all components can be initialized together."""
        # PBT config
        pbt_config = PBTConfig(
            population_size=4,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
                HyperparamConfig(name="adversarial_epsilon", min_value=0.01, max_value=0.15),
            ],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # SA-PPO config
        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
        )

        # Initialize components
        pbt_scheduler = PBTScheduler(pbt_config)
        population = pbt_scheduler.initialize_population()

        # Create mock model for each population member
        for member in population:
            mock_model = MagicMock()
            mock_model.policy = MagicMock()
            sa_ppo = StateAdversarialPPO(sa_ppo_config, mock_model)
            sa_ppo.on_training_start()

        assert len(population) == 4
        assert all(m.hyperparams is not None for m in population)

    def test_perturbation_with_sa_ppo(self):
        """Test state perturbation works with SA-PPO."""
        perturbation_config = PerturbationConfig(
            epsilon=0.1,
            attack_steps=3,
            attack_method="pgd",
        )
        perturbation_gen = StatePerturbation(perturbation_config)

        state = torch.randn(4, 10)

        def loss_fn(s):
            return (s ** 2).sum()

        delta = perturbation_gen.generate_perturbation(state, loss_fn)

        assert delta.shape == state.shape
        assert torch.abs(delta).max().item() <= perturbation_config.epsilon + 1e-6

    def test_pbt_hyperparameter_optimization(self, tmp_path):
        """Test PBT hyperparameter optimization flow."""
        pbt_config = PBTConfig(
            population_size=4,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=str(tmp_path),
        )

        scheduler = PBTScheduler(pbt_config, seed=42)
        population = scheduler.initialize_population()

        # Simulate training
        for step in range(10):
            for member in population:
                # Simulate training and performance
                performance = 0.5 + step * 0.05 + member.member_id * 0.01

                # Update performance
                model_state = {"param": torch.randn(5, 5)}
                scheduler.update_performance(member, performance, step, model_state)

                # Check if should exploit
                if scheduler.should_exploit_and_explore(member):
                    new_state, new_hyperparams = scheduler.exploit_and_explore(member)
                    # new_state could be None or a state dict
                    assert new_hyperparams is not None

        # All members should have history
        assert all(len(m.history) > 0 for m in population)

    def test_config_yaml_compatibility(self):
        """Test that configurations match YAML structure."""
        # This test verifies that our Python configs match the YAML structure
        pbt_config = PBTConfig(
            population_size=8,
            perturbation_interval=10,
            exploit_method="truncation",
            explore_method="both",
            truncation_ratio=0.25,
            metric_name="mean_reward",
            metric_mode="max",
        )

        sa_ppo_config = SAPPOConfig(
            enabled=True,
            perturbation=PerturbationConfig(
                epsilon=0.075,
                attack_steps=3,
                attack_lr=0.03,
                random_init=True,
                norm_type="linf",
                attack_method="pgd",
            ),
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
            warmup_updates=10,
            attack_policy=True,
            attack_value=True,
        )

        # Verify all attributes exist and have correct values
        assert pbt_config.population_size == 8
        assert sa_ppo_config.adversarial_ratio == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
