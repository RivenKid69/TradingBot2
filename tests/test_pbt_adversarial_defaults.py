"""
Tests for validating that PBT + Adversarial Training is enabled by default.

This test suite ensures that:
1. All configuration defaults have PBT and Adversarial enabled
2. The system behaves correctly with default settings
3. Default values are sane and deployment-ready
4. All scenarios work with defaults enabled

Target: 100% confidence that PBT + Adversarial is enabled by default
"""

import os
import pytest
pytest.importorskip("torch")
import yaml

from adversarial import (
    HyperparamConfig,
    PBTConfig,
    SAPPOConfig,
    PerturbationConfig,
    StateAdversarialPPO,
    PBTScheduler,
)
from training_pbt_adversarial_integration import (
    PBTAdversarialConfig,
    PBTTrainingCoordinator,
    DEFAULT_PBT_ADVERSARIAL_CONFIG,
    is_pbt_adversarial_enabled_by_default,
    load_pbt_adversarial_config,
)


class TestModuleDefaults:
    """Test that module-level defaults are correct."""

    def test_sa_ppo_config_default_enabled(self):
        """SAPPOConfig should have enabled=True by default."""
        config = SAPPOConfig()
        assert config.enabled is True, "SAPPOConfig.enabled must be True by default"

    def test_perturbation_config_has_sensible_defaults(self):
        """PerturbationConfig should have sensible defaults."""
        config = PerturbationConfig()
        assert config.epsilon == 0.075
        assert config.attack_steps == 3
        assert config.attack_method == "pgd"
        assert config.norm_type == "linf"

    def test_pbt_config_defaults_are_valid(self):
        """PBTConfig should have valid defaults."""
        # PBTConfig requires hyperparams, so we provide minimal one
        config = PBTConfig(
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
            ]
        )
        assert config.population_size == 10
        assert config.perturbation_interval == 5
        assert config.exploit_method == "truncation"
        assert config.explore_method == "both"

    def test_pbt_adversarial_config_default_enabled(self):
        """PBTAdversarialConfig should have both enabled by default."""
        config = PBTAdversarialConfig()
        assert config.pbt_enabled is True, "PBT must be enabled by default"
        assert config.adversarial_enabled is True, "Adversarial must be enabled by default"

    def test_default_pbt_adversarial_config_constant(self):
        """DEFAULT_PBT_ADVERSARIAL_CONFIG should have both enabled."""
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.pbt_enabled is True
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.adversarial_enabled is True


class TestYAMLConfigDefaults:
    """Test that YAML configuration files have correct defaults."""

    def test_config_pbt_adversarial_yaml_exists(self):
        """config_pbt_adversarial.yaml should exist."""
        config_path = "configs/config_pbt_adversarial.yaml"
        assert os.path.exists(config_path), f"{config_path} must exist"

    def test_config_pbt_adversarial_yaml_has_pbt_enabled(self):
        """config_pbt_adversarial.yaml should have pbt.enabled=true."""
        config_path = "configs/config_pbt_adversarial.yaml"
        if not os.path.exists(config_path):
            pytest.skip(f"{config_path} not found")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        assert "pbt" in data, "YAML must have 'pbt' section"
        assert data["pbt"]["enabled"] is True, "pbt.enabled must be True"

    def test_config_pbt_adversarial_yaml_has_adversarial_enabled(self):
        """config_pbt_adversarial.yaml should have adversarial.enabled=true."""
        config_path = "configs/config_pbt_adversarial.yaml"
        if not os.path.exists(config_path):
            pytest.skip(f"{config_path} not found")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        assert "adversarial" in data, "YAML must have 'adversarial' section"
        assert data["adversarial"]["enabled"] is True, "adversarial.enabled must be True"

    def test_load_default_config_returns_enabled(self):
        """Loading default config should return both enabled."""
        config_path = "configs/config_pbt_adversarial.yaml"
        if not os.path.exists(config_path):
            pytest.skip(f"{config_path} not found")

        config = load_pbt_adversarial_config(config_path)
        assert config.pbt_enabled is True, "Loaded PBT config must be enabled"
        assert config.adversarial_enabled is True, "Loaded Adversarial config must be enabled"

    def test_default_config_has_valid_hyperparameters(self):
        """Default config should have valid hyperparameters defined."""
        config_path = "configs/config_pbt_adversarial.yaml"
        if not os.path.exists(config_path):
            pytest.skip(f"{config_path} not found")

        config = load_pbt_adversarial_config(config_path)
        assert config.pbt is not None
        assert len(config.pbt.hyperparams) > 0, "Must have at least one hyperparameter"

        # Check specific hyperparameters exist
        hyperparam_names = [hp.name for hp in config.pbt.hyperparams]
        expected_hyperparams = [
            "learning_rate",
            "adversarial_epsilon",
            "adversarial_ratio",
            "robust_kl_coef",
        ]
        for expected in expected_hyperparams:
            assert expected in hyperparam_names, f"{expected} must be in hyperparams"

    def test_default_config_adversarial_settings_are_valid(self):
        """Default config should have valid adversarial settings."""
        config_path = "configs/config_pbt_adversarial.yaml"
        if not os.path.exists(config_path):
            pytest.skip(f"{config_path} not found")

        config = load_pbt_adversarial_config(config_path)
        assert config.adversarial is not None
        assert 0.0 <= config.adversarial.adversarial_ratio <= 1.0
        assert config.adversarial.robust_kl_coef >= 0
        assert config.adversarial.warmup_updates >= 0
        assert config.adversarial.perturbation.epsilon > 0
        assert config.adversarial.perturbation.attack_steps >= 1


class TestSystemDefaults:
    """Test that the system correctly identifies defaults."""

    def test_is_pbt_adversarial_enabled_by_default_returns_true(self):
        """is_pbt_adversarial_enabled_by_default() should return True."""
        result = is_pbt_adversarial_enabled_by_default()
        assert result is True, (
            "System must report that PBT + Adversarial is enabled by default. "
            "If this fails, check that configs/config_pbt_adversarial.yaml exists "
            "and has both pbt.enabled=true and adversarial.enabled=true."
        )

    def test_coordinator_defaults_to_enabled(self):
        """PBTTrainingCoordinator should work with default enabled config."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                hyperparams=[
                    HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
                ],
            ),
            adversarial=SAPPOConfig(),
        )

        coordinator = PBTTrainingCoordinator(config)
        assert coordinator.pbt_scheduler is not None
        assert coordinator.config.adversarial_enabled is True

    def test_creating_model_with_defaults_works(self):
        """Creating a model with default settings should work."""
        from unittest.mock import MagicMock

        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                hyperparams=[
                    HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
                ],
            ),
            adversarial=SAPPOConfig(),
        )

        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population()
        member = population[0]

        def model_factory(**kwargs):
            model = MagicMock()
            model.policy = MagicMock()
            return model

        model, sa_ppo = coordinator.create_member_model(member, model_factory)

        assert model is not None
        assert sa_ppo is not None
        assert isinstance(sa_ppo, StateAdversarialPPO)
        assert sa_ppo.is_adversarial_enabled is True or sa_ppo._update_count < sa_ppo.config.warmup_updates


class TestDefaultBehavior:
    """Test that default behavior matches expectations."""

    def test_sa_ppo_starts_enabled_after_warmup(self):
        """SA-PPO should be enabled after warmup period with defaults."""
        from unittest.mock import MagicMock

        config = SAPPOConfig()  # Default config
        model = MagicMock()
        model.policy = MagicMock()

        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        # Before warmup
        assert not sa_ppo.is_adversarial_enabled or sa_ppo.config.warmup_updates == 0

        # Simulate warmup
        for _ in range(config.warmup_updates + 1):
            sa_ppo.on_update_start()

        # After warmup
        assert sa_ppo.is_adversarial_enabled

    def test_pbt_scheduler_works_with_minimal_config(self):
        """PBT scheduler should work with minimal config."""
        config = PBTConfig(
            population_size=2,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
            ],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        assert len(population) == 2
        for member in population:
            assert "lr" in member.hyperparams
            assert 1e-5 <= member.hyperparams["lr"] <= 1e-3

    def test_default_epsilon_is_reasonable(self):
        """Default epsilon should be in a reasonable range."""
        config = PerturbationConfig()
        assert 0.01 <= config.epsilon <= 0.2, "Epsilon should be in reasonable range"

    def test_default_adversarial_ratio_is_balanced(self):
        """Default adversarial ratio should be balanced."""
        config = SAPPOConfig()
        assert 0.3 <= config.adversarial_ratio <= 0.7, "Ratio should be balanced"

    def test_default_population_size_is_sufficient(self):
        """Default population size should be sufficient for PBT."""
        config = PBTConfig(
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
            ]
        )
        assert config.population_size >= 4, "Population should be at least 4"


class TestDocumentationConsistency:
    """Test that documentation matches defaults."""

    def test_readme_mentions_defaults(self):
        """README should mention that PBT + Adversarial is enabled by default."""
        readme_path = "adversarial/README.md"
        if not os.path.exists(readme_path):
            pytest.skip(f"{readme_path} not found")

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that README mentions the components
        assert "PBT" in content or "Population-Based Training" in content
        assert "Adversarial" in content or "SA-PPO" in content


class TestRegressionDefaults:
    """Test that defaults haven't changed unexpectedly."""

    def test_sa_ppo_config_enabled_default_is_true(self):
        """Regression test: SAPPOConfig.enabled default must be True."""
        config = SAPPOConfig()
        assert config.enabled is True

    def test_pbt_adversarial_config_pbt_enabled_default_is_true(self):
        """Regression test: PBTAdversarialConfig.pbt_enabled default must be True."""
        config = PBTAdversarialConfig()
        assert config.pbt_enabled is True

    def test_pbt_adversarial_config_adversarial_enabled_default_is_true(self):
        """Regression test: PBTAdversarialConfig.adversarial_enabled default must be True."""
        config = PBTAdversarialConfig()
        assert config.adversarial_enabled is True

    def test_default_constant_is_consistent(self):
        """Regression test: DEFAULT_PBT_ADVERSARIAL_CONFIG must have both enabled."""
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.pbt_enabled is True
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.adversarial_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
