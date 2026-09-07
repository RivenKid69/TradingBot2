"""
Comprehensive tests for PBT + Adversarial Training Integration.

Tests cover:
- Configuration loading from YAML
- PBT Training Coordinator initialization
- Population management
- Model creation with SA-PPO wrapper
- Training lifecycle (on_update_start, on_update_end)
- Statistics collection
- Default settings validation
- Integration scenarios

Target: 100% code coverage for training_pbt_adversarial_integration.py
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

torch = pytest.importorskip("torch")
import yaml

from training_pbt_adversarial_integration import (
    PBTAdversarialConfig,
    PBTTrainingCoordinator,
    create_sappo_wrapper,
    is_pbt_adversarial_enabled_by_default,
    load_pbt_adversarial_config,
    DEFAULT_PBT_ADVERSARIAL_CONFIG,
)
from adversarial import (
    HyperparamConfig,
    PBTConfig,
    PopulationMember,
    SAPPOConfig,
    PerturbationConfig,
    StateAdversarialPPO,
)


class TestPBTAdversarialConfig:
    """Tests for PBTAdversarialConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = PBTAdversarialConfig()
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is True
        assert config.pbt is None
        assert config.adversarial is None
        assert config.base_config_path is None

    def test_custom_initialization(self):
        """Test custom configuration values."""
        pbt_config = PBTConfig(population_size=4, perturbation_interval=5)
        sa_ppo_config = SAPPOConfig(adversarial_ratio=0.6)

        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=pbt_config,
            adversarial=sa_ppo_config,
            base_config_path="configs/config_train.yaml",
        )

        assert config.pbt_enabled is True
        assert config.adversarial_enabled is True
        assert config.pbt == pbt_config
        assert config.adversarial == sa_ppo_config
        assert config.base_config_path == "configs/config_train.yaml"

    def test_disabled_pbt(self):
        """Test configuration with PBT disabled."""
        config = PBTAdversarialConfig(pbt_enabled=False, adversarial_enabled=True)
        assert config.pbt_enabled is False
        assert config.adversarial_enabled is True

    def test_disabled_adversarial(self):
        """Test configuration with adversarial disabled."""
        config = PBTAdversarialConfig(pbt_enabled=True, adversarial_enabled=False)
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is False


class TestLoadPBTAdversarialConfig:
    """Tests for load_pbt_adversarial_config function."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            "pbt": {
                "enabled": True,
                "population_size": 4,
                "perturbation_interval": 10,
                "exploit_method": "truncation",
                "explore_method": "both",
                "truncation_ratio": 0.25,
                "checkpoint_dir": "test_checkpoints",
                "metric_name": "test_metric",
                "metric_mode": "max",
                "ready_percentage": 0.75,
                "hyperparams": [
                    {
                        "name": "learning_rate",
                        "min_value": 1e-5,
                        "max_value": 1e-3,
                        "perturbation_factor": 1.2,
                        "resample_probability": 0.25,
                        "is_log_scale": True,
                    },
                    {
                        "name": "epsilon",
                        "min_value": 0.01,
                        "max_value": 0.15,
                    },
                ],
            },
            "adversarial": {
                "enabled": True,
                "perturbation": {
                    "epsilon": 0.1,
                    "attack_steps": 5,
                    "attack_lr": 0.02,
                    "random_init": True,
                    "norm_type": "linf",
                    "attack_method": "pgd",
                },
                "adversarial_ratio": 0.6,
                "robust_kl_coef": 0.15,
                "warmup_updates": 20,
                "attack_policy": True,
                "attack_value": True,
                "adaptive_epsilon": True,
                "epsilon_schedule": "cosine",
                "epsilon_final": 0.03,
            },
            "base_config_path": "configs/config_train.yaml",
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    def test_load_full_config(self, temp_config_file):
        """Test loading full configuration from YAML."""
        config = load_pbt_adversarial_config(str(temp_config_file))

        # Check top-level flags
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is True

        # Check PBT config
        assert config.pbt is not None
        assert config.pbt.population_size == 4
        assert config.pbt.perturbation_interval == 10
        assert config.pbt.exploit_method == "truncation"
        assert len(config.pbt.hyperparams) == 2
        assert config.pbt.hyperparams[0].name == "learning_rate"
        assert config.pbt.hyperparams[0].is_log_scale is True

        # Check Adversarial config
        assert config.adversarial is not None
        assert config.adversarial.perturbation.epsilon == 0.1
        assert config.adversarial.perturbation.attack_steps == 5
        assert config.adversarial.adversarial_ratio == 0.6
        assert config.adversarial.robust_kl_coef == 0.15
        assert config.adversarial.adaptive_epsilon is True
        assert config.adversarial.epsilon_schedule == "cosine"

    def test_load_pbt_disabled(self, tmp_path):
        """Test loading config with PBT disabled."""
        config_path = tmp_path / "config_pbt_disabled.yaml"
        config_data = {
            "pbt": {"enabled": False},
            "adversarial": {
                "enabled": True,
                "adversarial_ratio": 0.5,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_pbt_adversarial_config(str(config_path))
        assert config.pbt_enabled is False
        assert config.pbt is None
        assert config.adversarial_enabled is True

    def test_load_adversarial_disabled(self, tmp_path):
        """Test loading config with adversarial disabled."""
        config_path = tmp_path / "config_adv_disabled.yaml"
        config_data = {
            "pbt": {
                "enabled": True,
                "population_size": 4,
                "hyperparams": [],
            },
            "adversarial": {"enabled": False},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_pbt_adversarial_config(str(config_path))
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is False
        assert config.adversarial is None


class TestCreateSAPPOWrapper:
    """Tests for create_sappo_wrapper function."""

    def test_create_wrapper(self):
        """Test creating SA-PPO wrapper."""
        mock_model = MagicMock()
        sa_ppo_config = SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
        )

        wrapper = create_sappo_wrapper(mock_model, sa_ppo_config)

        assert isinstance(wrapper, StateAdversarialPPO)
        assert wrapper.config == sa_ppo_config
        assert wrapper.model == mock_model
        assert wrapper._adversarial_enabled is True

    def test_wrapper_initialized(self):
        """Test that wrapper is properly initialized."""
        mock_model = MagicMock()
        sa_ppo_config = SAPPOConfig()

        wrapper = create_sappo_wrapper(mock_model, sa_ppo_config)

        # on_training_start should have been called
        assert wrapper._adversarial_enabled is True


class TestPBTTrainingCoordinator:
    """Tests for PBTTrainingCoordinator class."""

    @pytest.fixture
    def pbt_config(self, tmp_path):
        """Create PBT configuration."""
        return PBTConfig(
            population_size=4,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

    @pytest.fixture
    def sa_ppo_config(self):
        """Create SA-PPO configuration."""
        return SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
        )

    @pytest.fixture
    def full_config(self, pbt_config, sa_ppo_config):
        """Create full PBT + Adversarial configuration."""
        return PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=pbt_config,
            adversarial=sa_ppo_config,
        )

    def test_initialization(self, full_config):
        """Test coordinator initialization."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)

        assert coordinator.config == full_config
        assert coordinator.seed == 42
        assert coordinator.pbt_scheduler is not None
        assert len(coordinator.sa_ppo_wrappers) == 0

    def test_initialization_pbt_disabled(self):
        """Test coordinator initialization with PBT disabled."""
        config = PBTAdversarialConfig(pbt_enabled=False, adversarial_enabled=True)
        coordinator = PBTTrainingCoordinator(config)

        assert coordinator.pbt_scheduler is None

    def test_initialize_population(self, full_config):
        """Test population initialization."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()

        assert len(population) == 4
        assert all(isinstance(m, PopulationMember) for m in population)
        assert all(m.hyperparams is not None for m in population)

    def test_initialize_population_no_pbt(self):
        """Test that initializing population without PBT raises error."""
        config = PBTAdversarialConfig(pbt_enabled=False)
        coordinator = PBTTrainingCoordinator(config)

        with pytest.raises(RuntimeError, match="PBT scheduler not initialized"):
            coordinator.initialize_population()

    def test_create_member_model(self, full_config):
        """Test creating model for a population member."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()
        member = population[0]

        # Mock model factory
        def model_factory(**kwargs):
            model = MagicMock()
            model.hyperparams = kwargs
            return model

        model, sa_ppo_wrapper = coordinator.create_member_model(
            member, model_factory, extra_param="test"
        )

        assert model is not None
        assert "lr" in model.hyperparams
        assert model.hyperparams["extra_param"] == "test"
        assert sa_ppo_wrapper is not None
        assert isinstance(sa_ppo_wrapper, StateAdversarialPPO)
        assert member.member_id in coordinator.sa_ppo_wrappers

    def test_create_member_model_no_adversarial(self, pbt_config):
        """Test creating model without adversarial training."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=pbt_config,
        )
        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population()
        member = population[0]

        def model_factory(**kwargs):
            return MagicMock()

        model, sa_ppo_wrapper = coordinator.create_member_model(member, model_factory)

        assert model is not None
        assert sa_ppo_wrapper is None
        assert len(coordinator.sa_ppo_wrappers) == 0

    def test_on_member_update_start(self, full_config):
        """Test on_member_update_start callback."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()
        member = population[0]

        # Create model to register SA-PPO wrapper
        def model_factory(**kwargs):
            return MagicMock()

        coordinator.create_member_model(member, model_factory)

        # Mock the on_update_start method
        wrapper = coordinator.sa_ppo_wrappers[member.member_id]
        wrapper.on_update_start = MagicMock()

        coordinator.on_member_update_start(member)

        wrapper.on_update_start.assert_called_once()

    def test_on_member_update_end(self, full_config, tmp_path):
        """Test on_member_update_end callback."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()
        member = population[0]

        # Create model
        def model_factory(**kwargs):
            return MagicMock()

        coordinator.create_member_model(member, model_factory)

        # Mock the on_update_end method
        wrapper = coordinator.sa_ppo_wrappers[member.member_id]
        wrapper.on_update_end = MagicMock()

        # Update member
        model_state = {"param": torch.randn(5, 5)}
        new_state, new_hyperparams, checkpoint_format = coordinator.on_member_update_end(
            member, performance=0.8, step=1, model_state_dict=model_state
        )

        wrapper.on_update_end.assert_called_once()
        assert new_hyperparams is not None

    def test_on_member_update_end_pbt_step(self, full_config, tmp_path):
        """Test PBT step during on_member_update_end."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()

        # Update all members with different performances
        for i, member in enumerate(population):
            model_state = {"param": torch.randn(5, 5)}
            coordinator.on_member_update_end(
                member,
                performance=0.5 + i * 0.1,
                step=5,  # perturbation_interval
                model_state_dict=model_state,
            )

        # Now trigger PBT step for worst performer
        worst_member = population[0]
        model_state = {"param": torch.randn(5, 5)}
        new_state, new_hyperparams, checkpoint_format = coordinator.on_member_update_end(
            worst_member,
            performance=0.5,
            step=10,  # Another perturbation_interval
            model_state_dict=model_state,
        )

        # PBT should have modified hyperparams
        assert new_hyperparams is not None

    def test_get_stats(self, full_config):
        """Test getting combined statistics."""
        coordinator = PBTTrainingCoordinator(full_config, seed=42)
        population = coordinator.initialize_population()

        # Create models for all members
        def model_factory(**kwargs):
            return MagicMock()

        for member in population:
            coordinator.create_member_model(member, model_factory)

        # Update some stats
        for member in population:
            model_state = {"param": torch.randn(5, 5)}
            coordinator.on_member_update_end(
                member, performance=0.7, step=1, model_state_dict=model_state
            )

        stats = coordinator.get_stats()

        # Should contain PBT stats
        assert "pbt/population_size" in stats
        assert "pbt/ready_members" in stats
        assert "pbt/mean_performance" in stats

        # Should contain aggregated SA-PPO stats
        assert any("sa_ppo" in key for key in stats.keys())


class TestDefaultSettings:
    """Tests for default settings and configuration."""

    def test_default_pbt_adversarial_config(self):
        """Test DEFAULT_PBT_ADVERSARIAL_CONFIG has correct defaults."""
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.pbt_enabled is True
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.adversarial_enabled is True

    def test_is_pbt_adversarial_enabled_by_default(self):
        """Test is_pbt_adversarial_enabled_by_default function."""
        # This test checks if the default config file exists and is enabled
        result = is_pbt_adversarial_enabled_by_default()
        assert isinstance(result, bool)

        # If config file exists, should return True
        if os.path.exists("configs/config_pbt_adversarial.yaml"):
            assert result is True

    def test_sa_ppo_config_enabled_by_default(self):
        """Test that SAPPOConfig has enabled=True by default."""
        config = SAPPOConfig()
        assert config.enabled is True

    def test_pbt_adversarial_config_enabled_by_default(self):
        """Test that PBTAdversarialConfig has both enabled by default."""
        config = PBTAdversarialConfig()
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is True


class TestIntegrationScenarios:
    """Integration tests for complete training scenarios."""

    @pytest.fixture
    def full_yaml_config(self, tmp_path):
        """Create a full YAML config file."""
        config_path = tmp_path / "full_config.yaml"
        config_data = {
            "pbt": {
                "enabled": True,
                "population_size": 2,
                "perturbation_interval": 5,
                "hyperparams": [
                    {"name": "lr", "min_value": 1e-5, "max_value": 1e-3},
                ],
            },
            "adversarial": {
                "enabled": True,
                "adversarial_ratio": 0.5,
                "robust_kl_coef": 0.1,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    def test_full_training_cycle(self, full_yaml_config, tmp_path):
        """Test a complete training cycle with PBT and adversarial."""
        # Load config
        config = load_pbt_adversarial_config(str(full_yaml_config))
        assert config.pbt_enabled is True
        assert config.adversarial_enabled is True

        # Create coordinator
        coordinator = PBTTrainingCoordinator(config, seed=42)

        # Initialize population
        population = coordinator.initialize_population()
        assert len(population) == 2

        # Create models
        def model_factory(**kwargs):
            model = MagicMock()
            model.policy = MagicMock()
            return model

        models = []
        for member in population:
            model, sa_ppo = coordinator.create_member_model(member, model_factory)
            models.append((model, sa_ppo))

        # Simulate training updates
        for step in range(10):
            for i, member in enumerate(population):
                # Start update
                coordinator.on_member_update_start(member)

                # Simulate training...

                # End update
                model_state = {"param": torch.randn(5, 5)}
                performance = 0.5 + i * 0.1 + step * 0.01
                new_state, new_hyperparams, checkpoint_format = coordinator.on_member_update_end(
                    member,
                    performance=performance,
                    step=step,
                    model_state_dict=model_state,
                )

        # Get final stats
        stats = coordinator.get_stats()
        assert "pbt/population_size" in stats
        assert stats["pbt/population_size"] == 2

    def test_pbt_only_scenario(self, tmp_path):
        """Test training with PBT only (no adversarial)."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
                ],
                checkpoint_dir=str(tmp_path / "checkpoints"),
            ),
        )

        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population()

        def model_factory(**kwargs):
            return MagicMock()

        for member in population:
            model, sa_ppo = coordinator.create_member_model(member, model_factory)
            assert sa_ppo is None  # No adversarial wrapper

    def test_adversarial_only_scenario(self):
        """Test training with adversarial only (no PBT)."""
        config = PBTAdversarialConfig(
            pbt_enabled=False,
            adversarial_enabled=True,
            adversarial=SAPPOConfig(adversarial_ratio=0.5),
        )

        coordinator = PBTTrainingCoordinator(config)
        assert coordinator.pbt_scheduler is None

        # Cannot initialize population without PBT
        with pytest.raises(RuntimeError):
            coordinator.initialize_population()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
