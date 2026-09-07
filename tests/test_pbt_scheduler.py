"""
Comprehensive tests for PBT (Population-Based Training) Scheduler.

Tests cover:
- Configuration validation
- Population initialization
- Exploitation strategies (truncation, binary tournament)
- Exploration strategies (perturb, resample, both)
- Hyperparameter mutation
- Performance tracking
- Checkpoint management
- Edge cases

Target: 100% code coverage
"""

import os
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from adversarial.pbt_scheduler import (
    HyperparamConfig,
    PBTConfig,
    PopulationMember,
    PBTScheduler,
)


class TestHyperparamConfig:
    """Tests for HyperparamConfig dataclass."""

    def test_continuous_hyperparam(self):
        """Test continuous hyperparameter configuration."""
        config = HyperparamConfig(
            name="learning_rate",
            min_value=1e-5,
            max_value=1e-3,
            perturbation_factor=1.2,
        )
        assert config.name == "learning_rate"
        assert config.is_continuous
        assert not config.is_categorical
        assert config.min_value == 1e-5
        assert config.max_value == 1e-3

    def test_categorical_hyperparam(self):
        """Test categorical hyperparameter configuration."""
        config = HyperparamConfig(
            name="activation",
            values=["relu", "tanh", "sigmoid"],
        )
        assert config.name == "activation"
        assert config.is_categorical
        assert not config.is_continuous
        assert config.values == ["relu", "tanh", "sigmoid"]

    def test_validation_no_range_or_values(self):
        """Test that missing both range and values raises error."""
        with pytest.raises(ValueError, match="Either 'values' or 'min_value'/'max_value'"):
            HyperparamConfig(name="test")

    def test_validation_both_range_and_values(self):
        """Test that specifying both range and values raises error."""
        with pytest.raises(ValueError, match="Cannot specify both 'values' and 'min_value'"):
            HyperparamConfig(
                name="test",
                min_value=0.0,
                max_value=1.0,
                values=[1, 2, 3],
            )

    def test_validation_invalid_range(self):
        """Test that min_value >= max_value raises error."""
        with pytest.raises(ValueError, match="min_value must be < max_value"):
            HyperparamConfig(
                name="test",
                min_value=1.0,
                max_value=0.5,
            )

    def test_validation_invalid_resample_probability(self):
        """Test that invalid resample_probability raises error."""
        with pytest.raises(ValueError, match="resample_probability must be in"):
            HyperparamConfig(
                name="test",
                min_value=0.0,
                max_value=1.0,
                resample_probability=1.5,
            )


class TestPBTConfig:
    """Tests for PBTConfig dataclass."""

    def test_default_initialization(self):
        """Test default PBT configuration."""
        config = PBTConfig()
        assert config.population_size == 10
        assert config.perturbation_interval == 5
        assert config.exploit_method == "truncation"
        assert config.explore_method == "both"
        assert config.truncation_ratio == 0.2
        assert config.metric_mode == "max"

    def test_custom_initialization(self):
        """Test custom PBT configuration."""
        hyperparams = [
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
        ]
        config = PBTConfig(
            population_size=5,
            perturbation_interval=10,
            hyperparams=hyperparams,
            exploit_method="binary_tournament",
            explore_method="perturb",
            metric_name="loss",
            metric_mode="min",
        )
        assert config.population_size == 5
        assert config.perturbation_interval == 10
        assert len(config.hyperparams) == 1
        assert config.exploit_method == "binary_tournament"
        assert config.explore_method == "perturb"
        assert config.metric_name == "loss"
        assert config.metric_mode == "min"

    def test_validation_small_population(self):
        """Test that population_size < 2 raises error."""
        with pytest.raises(ValueError, match="population_size must be >= 2"):
            PBTConfig(population_size=1)

    def test_validation_zero_perturbation_interval(self):
        """Test that perturbation_interval < 1 raises error."""
        with pytest.raises(ValueError, match="perturbation_interval must be >= 1"):
            PBTConfig(perturbation_interval=0)

    def test_validation_invalid_exploit_method(self):
        """Test that invalid exploit_method raises error."""
        with pytest.raises(ValueError, match="exploit_method must be"):
            PBTConfig(exploit_method="invalid")

    def test_validation_invalid_explore_method(self):
        """Test that invalid explore_method raises error."""
        with pytest.raises(ValueError, match="explore_method must be"):
            PBTConfig(explore_method="invalid")

    def test_validation_invalid_truncation_ratio(self):
        """Test that invalid truncation_ratio raises error."""
        with pytest.raises(ValueError, match="truncation_ratio must be in"):
            PBTConfig(truncation_ratio=1.5)

    def test_validation_invalid_metric_mode(self):
        """Test that invalid metric_mode raises error."""
        with pytest.raises(ValueError, match="metric_mode must be"):
            PBTConfig(metric_mode="invalid")

    def test_validation_invalid_ready_percentage(self):
        """Test that invalid ready_percentage raises error."""
        with pytest.raises(ValueError, match="ready_percentage must be in"):
            PBTConfig(ready_percentage=1.5)


class TestPopulationMember:
    """Tests for PopulationMember dataclass."""

    def test_initialization(self):
        """Test PopulationMember initialization."""
        member = PopulationMember(
            member_id=0,
            hyperparams={"lr": 0.001, "momentum": 0.9},
        )
        assert member.member_id == 0
        assert member.hyperparams == {"lr": 0.001, "momentum": 0.9}
        assert member.performance is None
        assert member.step == 0
        assert member.checkpoint_path is None
        assert len(member.history) == 0

    def test_record_step(self):
        """Test recording training steps."""
        member = PopulationMember(
            member_id=0,
            hyperparams={"lr": 0.001},
        )

        member.record_step(100, 0.8, {"lr": 0.001})

        assert member.step == 100
        assert member.performance == 0.8
        assert len(member.history) == 1
        assert member.history[0]["step"] == 100
        assert member.history[0]["performance"] == 0.8
        assert member.history[0]["hyperparams"] == {"lr": 0.001}

    def test_record_multiple_steps(self):
        """Test recording multiple training steps."""
        member = PopulationMember(
            member_id=0,
            hyperparams={"lr": 0.001},
        )

        member.record_step(100, 0.8, {"lr": 0.001})
        member.record_step(200, 0.85, {"lr": 0.002})
        member.record_step(300, 0.9, {"lr": 0.002})

        assert member.step == 300
        assert member.performance == 0.9
        assert len(member.history) == 3


class TestPBTScheduler:
    """Tests for PBTScheduler class."""

    @pytest.fixture
    def config(self, tmp_path):
        """Default PBT configuration."""
        hyperparams = [
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
            HyperparamConfig(name="momentum", values=[0.8, 0.9, 0.95, 0.99]),
        ]
        return PBTConfig(
            population_size=4,
            perturbation_interval=5,
            hyperparams=hyperparams,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            metric_name="reward",
            metric_mode="max",
        )

    @pytest.fixture
    def scheduler(self, config):
        """PBT scheduler instance."""
        return PBTScheduler(config, seed=42)

    def test_initialization(self, config):
        """Test PBT scheduler initialization."""
        scheduler = PBTScheduler(config, seed=42)
        assert scheduler.config == config
        assert len(scheduler.population) == 0
        assert scheduler._exploitation_count == 0
        assert scheduler._exploration_count == 0
        assert os.path.exists(config.checkpoint_dir)

    def test_initialize_population_random(self, scheduler):
        """Test random population initialization."""
        population = scheduler.initialize_population()

        assert len(population) == scheduler.config.population_size
        for i, member in enumerate(population):
            assert member.member_id == i
            assert "lr" in member.hyperparams
            assert "momentum" in member.hyperparams
            assert member.performance is None
            assert member.step == 0

    def test_initialize_population_provided(self, scheduler):
        """Test population initialization with provided hyperparameters."""
        initial_hyperparams = [
            {"lr": 1e-4, "momentum": 0.9},
            {"lr": 2e-4, "momentum": 0.95},
            {"lr": 3e-4, "momentum": 0.99},
            {"lr": 4e-4, "momentum": 0.8},
        ]

        population = scheduler.initialize_population(initial_hyperparams)

        assert len(population) == 4
        for i, member in enumerate(population):
            assert member.hyperparams == initial_hyperparams[i]

    def test_initialize_population_wrong_size(self, scheduler):
        """Test population initialization with wrong number of hyperparams."""
        initial_hyperparams = [
            {"lr": 1e-4, "momentum": 0.9},
            {"lr": 2e-4, "momentum": 0.95},
        ]

        with pytest.raises(ValueError, match="initial_hyperparams length"):
            scheduler.initialize_population(initial_hyperparams)

    def test_should_exploit_and_explore(self, scheduler):
        """Test should_exploit_and_explore condition."""
        scheduler.initialize_population()
        member = scheduler.population[0]

        # Should not exploit at step 0
        assert not scheduler.should_exploit_and_explore(member)

        # Should exploit at perturbation_interval
        member.step = scheduler.config.perturbation_interval
        assert scheduler.should_exploit_and_explore(member)

        # Should not exploit at non-multiples
        member.step = 3
        assert not scheduler.should_exploit_and_explore(member)

    def test_update_performance(self, scheduler):
        """Test performance update and checkpoint saving."""
        scheduler.initialize_population()
        member = scheduler.population[0]

        model_state = {"param1": torch.randn(10, 10)}

        scheduler.update_performance(member, 0.8, 100, model_state)

        assert member.performance == 0.8
        assert member.step == 100
        assert len(member.history) == 1
        assert member.checkpoint_path is not None
        assert os.path.exists(member.checkpoint_path)

    def test_rank_population_max_mode(self, scheduler):
        """Test population ranking in max mode."""
        scheduler.initialize_population()

        # Set different performances
        scheduler.population[0].performance = 0.5
        scheduler.population[1].performance = 0.9
        scheduler.population[2].performance = 0.7
        scheduler.population[3].performance = 0.3

        ranked = scheduler._rank_population()

        # Should be sorted in descending order (best first)
        assert ranked[0].performance == 0.9
        assert ranked[1].performance == 0.7
        assert ranked[2].performance == 0.5
        assert ranked[3].performance == 0.3

    def test_rank_population_min_mode(self, scheduler):
        """Test population ranking in min mode."""
        scheduler.config.metric_mode = "min"
        scheduler.initialize_population()

        # Set different performances
        scheduler.population[0].performance = 0.5
        scheduler.population[1].performance = 0.9
        scheduler.population[2].performance = 0.7
        scheduler.population[3].performance = 0.3

        ranked = scheduler._rank_population()

        # Should be sorted in ascending order (best first)
        assert ranked[0].performance == 0.3
        assert ranked[1].performance == 0.5
        assert ranked[2].performance == 0.7
        assert ranked[3].performance == 0.9

    def test_sample_hyperparams_continuous(self, scheduler):
        """Test sampling continuous hyperparameter."""
        config = HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)
        value = scheduler._sample_hyperparam(config)

        assert config.min_value <= value <= config.max_value

    def test_sample_hyperparams_continuous_log_scale(self, scheduler):
        """Test sampling continuous hyperparameter with log scale."""
        config = HyperparamConfig(
            name="lr",
            min_value=1e-5,
            max_value=1e-3,
            is_log_scale=True,
        )
        value = scheduler._sample_hyperparam(config)

        assert config.min_value <= value <= config.max_value

    def test_sample_hyperparams_categorical(self, scheduler):
        """Test sampling categorical hyperparameter."""
        config = HyperparamConfig(name="activation", values=["relu", "tanh", "sigmoid"])
        value = scheduler._sample_hyperparam(config)

        assert value in config.values

    def test_perturb_hyperparam_continuous(self, scheduler):
        """Test perturbing continuous hyperparameter."""
        config = HyperparamConfig(
            name="lr", min_value=1e-5, max_value=1e-3, perturbation_factor=1.2
        )
        current_value = 5e-4

        new_value = scheduler._perturb_hyperparam(config, current_value)

        # Should be either multiplied or divided by perturbation_factor
        assert new_value != current_value
        assert config.min_value <= new_value <= config.max_value

    def test_perturb_hyperparam_categorical(self, scheduler):
        """Test perturbing categorical hyperparameter."""
        config = HyperparamConfig(name="activation", values=["relu", "tanh", "sigmoid"])
        current_value = "tanh"

        new_value = scheduler._perturb_hyperparam(config, current_value)

        # Should be an adjacent value
        assert new_value in config.values
        # For 3 values, adjacent means either "relu" or "sigmoid"
        assert new_value in ["relu", "sigmoid"]

    def test_explore_hyperparams_perturb(self, scheduler):
        """Test exploration with perturb method."""
        scheduler.config.explore_method = "perturb"
        current = {"lr": 1e-4, "momentum": 0.9}

        new_hyperparams = scheduler._explore_hyperparams(current)

        # Should have same keys
        assert set(new_hyperparams.keys()) == set(current.keys())
        # Values should be different (with high probability)
        # Note: might be same by chance, but unlikely

    def test_explore_hyperparams_resample(self, scheduler):
        """Test exploration with resample method."""
        scheduler.config.explore_method = "resample"
        current = {"lr": 1e-4, "momentum": 0.9}

        new_hyperparams = scheduler._explore_hyperparams(current)

        # Should have same keys
        assert set(new_hyperparams.keys()) == set(current.keys())
        # Values should be resampled

    def test_explore_hyperparams_both(self, scheduler):
        """Test exploration with both method."""
        scheduler.config.explore_method = "both"
        current = {"lr": 1e-4, "momentum": 0.9}

        new_hyperparams = scheduler._explore_hyperparams(current)

        # Should have same keys
        assert set(new_hyperparams.keys()) == set(current.keys())

    def test_should_exploit_truncation_bottom(self, scheduler):
        """Test exploitation decision with truncation (bottom performer)."""
        scheduler.config.exploit_method = "truncation"
        scheduler.config.truncation_ratio = 0.5  # Truncate bottom 50%
        scheduler.initialize_population()

        # Set performances
        scheduler.population[0].performance = 0.3  # Worst
        scheduler.population[1].performance = 0.9  # Best
        scheduler.population[2].performance = 0.7
        scheduler.population[3].performance = 0.5

        # With 0.5 ratio: threshold = int(4*0.5) = 2
        # Ranked: [0.9, 0.7, 0.5, 0.3], worst is at rank 3
        # Should exploit if rank >= 4-2 = 2, so ranks 2,3 exploit
        # Member with 0.3 is at rank 3, should exploit
        assert scheduler._should_exploit(scheduler.population[0])

    def test_should_exploit_truncation_top(self, scheduler):
        """Test exploitation decision with truncation (top performer)."""
        scheduler.config.exploit_method = "truncation"
        scheduler.config.truncation_ratio = 0.5  # Truncate bottom 50%
        scheduler.initialize_population()

        # Set performances
        scheduler.population[0].performance = 0.3
        scheduler.population[1].performance = 0.9  # Best
        scheduler.population[2].performance = 0.7
        scheduler.population[3].performance = 0.5

        # With 0.5 ratio: threshold = int(4*0.5) = 2
        # Ranked: [0.9, 0.7, 0.5, 0.3], best is at rank 0
        # Should exploit if rank >= 4-2 = 2, so ranks 2,3 exploit
        # Member with 0.9 is at rank 0, should NOT exploit
        assert not scheduler._should_exploit(scheduler.population[1])

    def test_should_exploit_binary_tournament(self, scheduler):
        """Test exploitation decision with binary tournament."""
        scheduler.config.exploit_method = "binary_tournament"
        scheduler.initialize_population()

        # Set performances
        scheduler.population[0].performance = 0.3
        scheduler.population[1].performance = 0.9
        scheduler.population[2].performance = 0.7
        scheduler.population[3].performance = 0.5

        # Result depends on random selection, just check it doesn't crash
        result = scheduler._should_exploit(scheduler.population[0])
        assert isinstance(result, bool)

    def test_select_source_member_truncation(self, scheduler):
        """Test source member selection with truncation."""
        scheduler.config.exploit_method = "truncation"
        scheduler.config.truncation_ratio = 0.5  # Select from top 50%
        scheduler.initialize_population()

        # Set performances
        scheduler.population[0].performance = 0.3  # Worst
        scheduler.population[1].performance = 0.9  # Best
        scheduler.population[2].performance = 0.7  # 2nd best
        scheduler.population[3].performance = 0.5

        source = scheduler._select_source_member(scheduler.population[0])

        # With 0.5 ratio: threshold = int(4*0.5) = 2
        # Top 2 members have performance 0.9 and 0.7
        # Should select from top performers (>= 0.7)
        assert source is not None
        assert source.performance >= 0.7

    def test_exploit_and_explore_not_ready(self, scheduler):
        """Test exploit_and_explore when population not ready."""
        scheduler.initialize_population()
        member = scheduler.population[0]

        # Most members have no performance yet
        scheduler.population[0].performance = 0.5

        new_state, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(member)

        # Should skip exploitation
        assert new_state is None
        assert checkpoint_format is None

    def test_exploit_and_explore_ready(self, scheduler):
        """Test exploit_and_explore when population ready."""
        scheduler.initialize_population()

        # Set all performances
        for i, member in enumerate(scheduler.population):
            member.performance = 0.5 + i * 0.1
            # Create dummy checkpoint
            checkpoint_path = os.path.join(
                scheduler.config.checkpoint_dir, f"member_{member.member_id}_step_0.pt"
            )
            torch.save({"dummy": torch.randn(5, 5)}, checkpoint_path)
            member.checkpoint_path = checkpoint_path

        # Test with worst performer
        worst_member = scheduler.population[0]
        new_state, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(worst_member)

        # Note: Exploitation may or may not occur depending on population size and truncation_ratio
        # Main test: method returns correct tuple (3 values) and doesn't crash
        # Should have new hyperparams from exploration (always happens)
        assert new_hyperparams is not None
        # If exploitation occurred, checkpoint_format should be v1_policy_only
        if new_state is not None:
            assert checkpoint_format == "v1_policy_only"

    def test_get_stats(self, scheduler):
        """Test getting PBT statistics."""
        scheduler.initialize_population()

        # Set performances
        for i, member in enumerate(scheduler.population):
            member.performance = 0.5 + i * 0.1

        stats = scheduler.get_stats()

        assert stats["pbt/population_size"] == 4
        assert stats["pbt/ready_members"] == 4
        assert stats["pbt/mean_performance"] > 0
        assert stats["pbt/std_performance"] >= 0
        assert stats["pbt/min_performance"] == 0.5
        assert stats["pbt/max_performance"] == 0.8

    def test_get_stats_empty_population(self, scheduler):
        """Test getting stats with no performances."""
        scheduler.initialize_population()

        stats = scheduler.get_stats()

        assert stats["pbt/ready_members"] == 0
        assert stats["pbt/mean_performance"] == 0.0
        assert stats["pbt/std_performance"] == 0.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_perturbation_clipping_to_bounds(self):
        """Test that perturbation respects hyperparameter bounds."""
        config = HyperparamConfig(
            name="lr",
            min_value=1e-5,
            max_value=1e-4,
            perturbation_factor=10.0,  # Large factor
        )
        scheduler = PBTScheduler(
            PBTConfig(
                population_size=2,
                hyperparams=[config],
            )
        )

        # Perturb value near boundary
        current_value = 9e-5
        new_value = scheduler._perturb_hyperparam(config, current_value)

        # Should be clipped to valid range
        assert config.min_value <= new_value <= config.max_value

    def test_categorical_wrapping(self):
        """Test that categorical perturbation wraps around."""
        config = HyperparamConfig(name="size", values=[32, 64, 128])
        scheduler = PBTScheduler(
            PBTConfig(
                population_size=2,
                hyperparams=[config],
            )
        )

        # Test wrapping from first to last
        # (Depends on random, but we can test it doesn't crash)
        new_value = scheduler._perturb_hyperparam(config, 32)
        assert new_value in [64, 128]

    def test_empty_hyperparams_list(self):
        """Test PBT with no hyperparameters to optimize."""
        config = PBTConfig(
            population_size=2,
            hyperparams=[],  # No hyperparams
        )
        scheduler = PBTScheduler(config)
        population = scheduler.initialize_population()

        # Should still create population
        assert len(population) == 2
        for member in population:
            assert len(member.hyperparams) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
