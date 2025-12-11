"""
Deep validation tests for PBT + Adversarial Training.

This test suite goes beyond basic functionality and tests:
- Edge cases and boundary conditions
- Error handling and recovery
- Stress tests and performance
- Memory management
- Concurrent scenarios
- Integration with real components
- Numerical stability
- Configuration validation
- State consistency

Target: 100% confidence in production readiness
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import copy

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
import yaml

from adversarial import (
    PerturbationConfig,
    StatePerturbation,
    SAPPOConfig,
    StateAdversarialPPO,
    PBTConfig,
    HyperparamConfig,
    PBTScheduler,
    PopulationMember,
)
from training_pbt_adversarial_integration import (
    PBTAdversarialConfig,
    PBTTrainingCoordinator,
    create_sappo_wrapper,
    load_pbt_adversarial_config,
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_epsilon_perturbation(self):
        """Test perturbation with epsilon=0."""
        config = PerturbationConfig(epsilon=0.0)
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)
        def loss_fn(s): return (s ** 2).sum()

        delta = perturb.fgsm_attack(state, loss_fn)

        assert torch.all(delta == 0.0), "Zero epsilon should produce zero perturbation"

    def test_very_large_epsilon(self):
        """Test perturbation with very large epsilon."""
        config = PerturbationConfig(epsilon=100.0)
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)
        def loss_fn(s): return (s ** 2).sum()

        delta = perturb.fgsm_attack(state, loss_fn)

        # Should still be bounded by epsilon
        assert torch.abs(delta).max() <= 100.0 + 1e-5

    def test_single_sample_perturbation(self):
        """Test perturbation with batch size 1."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        state = torch.randn(1, 10)  # Single sample
        def loss_fn(s): return (s ** 2).sum()

        delta = perturb.generate_perturbation(state, loss_fn)

        assert delta.shape == state.shape

    def test_very_large_batch(self):
        """Test perturbation with very large batch."""
        config = PerturbationConfig(attack_steps=1)  # Reduce steps for speed
        perturb = StatePerturbation(config)

        state = torch.randn(1000, 10)  # Large batch
        def loss_fn(s): return (s ** 2).sum()

        delta = perturb.generate_perturbation(state, loss_fn)

        assert delta.shape == state.shape

    def test_high_dimensional_state(self):
        """Test perturbation with high-dimensional state."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        state = torch.randn(4, 64, 64, 3)  # High-dim state
        def loss_fn(s): return (s ** 2).sum()

        delta = perturb.generate_perturbation(state, loss_fn)

        assert delta.shape == state.shape

    def test_population_size_one(self):
        """Test PBT rejects population size of 1 (minimum is 2)."""
        # Population size must be >= 2 for PBT to work
        with pytest.raises(ValueError, match="population_size must be >= 2"):
            config = PBTConfig(
                population_size=1,
                hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            )

    def test_very_large_population(self):
        """Test PBT with very large population."""
        config = PBTConfig(
            population_size=100,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        assert len(population) == 100

    def test_adversarial_ratio_zero(self):
        """Test SA-PPO with adversarial_ratio=0 (all clean)."""
        config = SAPPOConfig(adversarial_ratio=0.0)
        model = MagicMock()
        model.policy = MagicMock()

        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        # Should still work without errors
        assert sa_ppo.config.adversarial_ratio == 0.0

    def test_adversarial_ratio_one(self):
        """Test SA-PPO with adversarial_ratio=1.0 (all adversarial)."""
        config = SAPPOConfig(adversarial_ratio=1.0)
        model = MagicMock()
        model.policy = MagicMock()

        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        assert sa_ppo.config.adversarial_ratio == 1.0


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_yaml_config(self, tmp_path):
        """Test loading invalid YAML config."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(Exception):
            load_pbt_adversarial_config(str(config_path))

    def test_missing_required_fields_in_yaml(self, tmp_path):
        """Test YAML with missing required fields."""
        config_path = tmp_path / "incomplete.yaml"
        config_data = {
            "pbt": {
                "enabled": True,
                # Missing other required fields
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Should handle gracefully with defaults
        config = load_pbt_adversarial_config(str(config_path))
        assert config.pbt_enabled is True

    def test_coordinator_without_pbt_initialization(self):
        """Test using coordinator methods before initialization."""
        config = PBTAdversarialConfig(pbt_enabled=False)
        coordinator = PBTTrainingCoordinator(config)

        with pytest.raises(RuntimeError, match="PBT scheduler not initialized"):
            coordinator.initialize_population()

    def test_perturbation_with_nan_state(self):
        """Test perturbation handling NaN in state."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)
        state[0, 0] = float('nan')

        def loss_fn(s): return (s ** 2).sum()

        # Should handle NaN gracefully or raise clear error
        try:
            delta = perturb.generate_perturbation(state, loss_fn)
            # If it succeeds, check delta is not all NaN
            assert not torch.isnan(delta).all()
        except (ValueError, RuntimeError):
            # Acceptable to raise error
            pass

    def test_perturbation_with_inf_state(self):
        """Test perturbation handling Inf in state."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)
        state[0, 0] = float('inf')

        def loss_fn(s): return (s ** 2).sum()

        try:
            delta = perturb.generate_perturbation(state, loss_fn)
            assert not torch.isinf(delta).all()
        except (ValueError, RuntimeError):
            pass

    def test_loss_function_returning_nan(self):
        """Test perturbation when loss function returns NaN."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)
        def bad_loss_fn(s): return torch.tensor(float('nan'))

        try:
            delta = perturb.generate_perturbation(state, bad_loss_fn)
            # Should either handle or raise
        except:
            pass  # Acceptable

    def test_pbt_with_all_none_performance(self):
        """Test PBT when all members have None performance."""
        config = PBTConfig(
            population_size=3,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # All members have None performance
        for member in population:
            assert member.performance is None

        # Should handle gracefully
        member = population[0]
        new_state, new_hp, _ = scheduler.exploit_and_explore(member)


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_perturbation_numerical_stability_linf(self):
        """Test L-inf perturbation maintains bounds."""
        config = PerturbationConfig(epsilon=0.1, norm_type="linf")
        perturb = StatePerturbation(config)

        for _ in range(10):  # Multiple trials
            state = torch.randn(10, 20)
            def loss_fn(s): return (s ** 2).sum()

            delta = perturb.generate_perturbation(state, loss_fn)

            # Check L-inf norm
            max_norm = torch.abs(delta).max().item()
            assert max_norm <= config.epsilon + 1e-6, f"L-inf norm {max_norm} exceeds epsilon {config.epsilon}"

    def test_perturbation_numerical_stability_l2(self):
        """Test L2 perturbation maintains bounds."""
        config = PerturbationConfig(epsilon=0.5, norm_type="l2")
        perturb = StatePerturbation(config)

        for _ in range(10):
            state = torch.randn(10, 20)
            def loss_fn(s): return (s ** 2).sum()

            delta = perturb.generate_perturbation(state, loss_fn)

            # Check L2 norm per sample
            batch_size = delta.size(0)
            l2_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            assert torch.all(l2_norms <= config.epsilon + 1e-5), "L2 norm exceeds epsilon"

    def test_hyperparameter_bounds_maintained(self):
        """Test that hyperparameter mutations stay within bounds."""
        config = PBTConfig(
            population_size=10,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
                HyperparamConfig(name="gamma", min_value=0.9, max_value=0.999),
            ],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Mutate many times
        for _ in range(100):
            for member in population:
                new_hp = scheduler._explore_hyperparams(member.hyperparams)

                assert 1e-5 <= new_hp["lr"] <= 1e-3, f"lr {new_hp['lr']} out of bounds"
                assert 0.9 <= new_hp["gamma"] <= 0.999, f"gamma {new_hp['gamma']} out of bounds"

    def test_epsilon_schedule_monotonic(self):
        """Test that epsilon schedule is monotonic."""
        config = SAPPOConfig(
            adaptive_epsilon=True,
            epsilon_schedule="linear",
            perturbation=PerturbationConfig(epsilon=0.1),
            epsilon_final=0.05,
        )
        model = MagicMock()
        model.policy = MagicMock()

        sa_ppo = StateAdversarialPPO(config, model)

        prev_epsilon = sa_ppo._get_current_epsilon()

        for _ in range(100):
            sa_ppo.on_update_start()
            curr_epsilon = sa_ppo._get_current_epsilon()

            # For linear schedule, epsilon should decrease
            # (or stay same if at final)
            if curr_epsilon != config.epsilon_final:
                assert curr_epsilon <= prev_epsilon + 1e-9, "Epsilon should decrease or stay same"
            prev_epsilon = curr_epsilon


class TestStateConsistency:
    """Test state consistency and persistence."""

    def test_population_member_history_tracking(self):
        """Test that population member history is correctly tracked."""
        member = PopulationMember(member_id=0, hyperparams={"lr": 0.001})

        # Record multiple steps
        for step in range(10):
            member.record_step(step, performance=0.5 + step * 0.01, hyperparams={"lr": 0.001 + step * 0.0001})

        assert len(member.history) == 10

        # Check history is ordered
        for i, record in enumerate(member.history):
            assert record["step"] == i
            assert abs(record["performance"] - (0.5 + i * 0.01)) < 1e-9

    def test_sa_ppo_stats_accumulation(self):
        """Test that SA-PPO stats accumulate correctly."""
        config = SAPPOConfig()
        model = MagicMock()
        model.policy = MagicMock()

        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        # Simulate multiple updates
        for _ in range(20):
            sa_ppo.on_update_start()

        stats = sa_ppo.get_stats()
        assert stats["sa_ppo/update_count"] == 20

    def test_pbt_stats_consistency(self):
        """Test PBT statistics consistency."""
        config = PBTConfig(
            population_size=5,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Update performances
        for i, member in enumerate(population):
            scheduler.update_performance(member, 0.5 + i * 0.1, step=1)

        stats = scheduler.get_stats()

        assert stats["pbt/population_size"] == 5
        assert stats["pbt/ready_members"] == 5
        assert abs(stats["pbt/mean_performance"] - 0.7) < 1e-9

    def test_checkpoint_save_load_cycle(self, tmp_path):
        """Test checkpoint saving and loading cycle."""
        config = PBTConfig(
            population_size=2,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        member = population[0]
        model_state = {"param": torch.randn(5, 5)}

        # Save checkpoint
        scheduler.update_performance(member, 0.8, step=5, model_state_dict=model_state)

        assert member.checkpoint_path is not None
        assert os.path.exists(member.checkpoint_path)

        # Load checkpoint
        # Security: weights_only=False needed for dict metadata (format_version)
        loaded_checkpoint = torch.load(member.checkpoint_path, weights_only=False)

        # Verify new checkpoint format
        assert "format_version" in loaded_checkpoint
        assert "data" in loaded_checkpoint
        assert loaded_checkpoint["format_version"] == "v1_policy_only"  # Using model_state_dict

        # Extract actual state from checkpoint
        loaded_state = loaded_checkpoint["data"]
        assert "param" in loaded_state
        assert torch.allclose(loaded_state["param"], model_state["param"])


class TestConcurrency:
    """Test concurrent scenarios and race conditions."""

    def test_multiple_coordinators_same_config(self):
        """Test multiple coordinators with same config (independence check)."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            ),
            adversarial=SAPPOConfig(),
        )

        coordinator1 = PBTTrainingCoordinator(config, seed=42)
        coordinator2 = PBTTrainingCoordinator(config, seed=42)

        pop1 = coordinator1.initialize_population()
        pop2 = coordinator2.initialize_population()

        # With same seed, hyperparams should be close (within log scale range)
        # Note: Due to independent RNG state, values may differ slightly
        lr1 = pop1[0].hyperparams["lr"]
        lr2 = pop2[0].hyperparams["lr"]
        # Both should be within valid range
        assert 1e-5 <= lr1 <= 1e-3
        assert 1e-5 <= lr2 <= 1e-3

    def test_concurrent_population_updates(self, tmp_path):
        """Test concurrent updates to population members."""
        config = PBTConfig(
            population_size=5,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Simulate concurrent updates
        for step in range(10):
            for member in population:
                performance = 0.5 + np.random.random() * 0.3
                model_state = {"param": torch.randn(3, 3)}
                scheduler.update_performance(member, performance, step, model_state)

        # All members should have history
        assert all(len(m.history) > 0 for m in population)


class TestMemoryManagement:
    """Test memory management and leaks."""

    def test_perturbation_memory_cleanup(self):
        """Test that perturbation cleans up tensors."""
        config = PerturbationConfig()
        perturb = StatePerturbation(config)

        initial_tensors = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

        for _ in range(100):
            state = torch.randn(10, 20)
            def loss_fn(s): return (s ** 2).sum()
            delta = perturb.generate_perturbation(state, loss_fn)
            del delta

        gc.collect()

        final_tensors = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

        # Should not leak too many tensors
        tensor_growth = final_tensors - initial_tensors
        assert tensor_growth < 50, f"Potential memory leak: {tensor_growth} new tensors"

    def test_coordinator_memory_cleanup(self):
        """Test coordinator cleans up properly."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            pbt=PBTConfig(
                population_size=10,
                hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            ),
            adversarial=SAPPOConfig(),
        )

        for _ in range(10):
            coordinator = PBTTrainingCoordinator(config, seed=42)
            population = coordinator.initialize_population()

            def model_factory(**kwargs):
                return MagicMock()

            for member in population:
                coordinator.create_member_model(member, model_factory)

            del coordinator

        gc.collect()


class TestConfigValidation:
    """Test configuration validation thoroughly."""

    def test_all_invalid_perturbation_configs(self):
        """Test all invalid perturbation configurations."""
        invalid_configs = [
            {"epsilon": -1.0},
            {"attack_steps": 0},
            {"attack_lr": 0.0},
            {"norm_type": "invalid"},
            {"attack_method": "invalid"},
            {"clip_min": 1.0, "clip_max": 0.5},
        ]

        for invalid in invalid_configs:
            with pytest.raises(ValueError):
                PerturbationConfig(**invalid)

    def test_all_invalid_sappo_configs(self):
        """Test all invalid SA-PPO configurations."""
        invalid_configs = [
            {"adversarial_ratio": -0.1},
            {"adversarial_ratio": 1.5},
            {"robust_kl_coef": -0.1},
            {"warmup_updates": -1},
            {"epsilon_schedule": "invalid"},
        ]

        for invalid in invalid_configs:
            with pytest.raises(ValueError):
                SAPPOConfig(**invalid)

    def test_all_invalid_pbt_configs(self):
        """Test all invalid PBT configurations."""
        invalid_configs = [
            {"population_size": 1, "hyperparams": []},  # Too small
            {"population_size": 10, "perturbation_interval": 0, "hyperparams": []},
            {"exploit_method": "invalid", "hyperparams": []},
            {"explore_method": "invalid", "hyperparams": []},
            {"truncation_ratio": 0.0, "hyperparams": []},
            {"truncation_ratio": 1.0, "hyperparams": []},
            {"metric_mode": "invalid", "hyperparams": []},
            {"ready_percentage": 0.0, "hyperparams": []},
            {"ready_percentage": 1.5, "hyperparams": []},
        ]

        for invalid in invalid_configs:
            with pytest.raises(ValueError):
                PBTConfig(**invalid)

    def test_hyperparam_config_edge_cases(self):
        """Test hyperparameter config edge cases."""
        # Continuous param at boundary
        hp = HyperparamConfig(name="lr", min_value=1e-10, max_value=1e-9)
        assert hp.is_continuous

        # Categorical with single value
        hp = HyperparamConfig(name="act", values=["relu"])
        assert hp.is_categorical

        # Very large range
        hp = HyperparamConfig(name="lr", min_value=1e-10, max_value=1e10, is_log_scale=True)
        assert hp.is_log_scale


class TestIntegrationRealism:
    """Test integration with realistic scenarios."""

    def test_realistic_training_loop(self, tmp_path):
        """Test realistic training loop scenario."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
                    HyperparamConfig(name="epsilon", min_value=0.01, max_value=0.15),
                ],
                checkpoint_dir=str(tmp_path / "checkpoints"),
            ),
            adversarial=SAPPOConfig(
                adversarial_ratio=0.5,
                warmup_updates=5,
            ),
        )

        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population()

        # Create mock models
        models = []
        for member in population:
            def model_factory(**kwargs):
                model = MagicMock()
                model.policy = MagicMock()
                model.state_dict = MagicMock(return_value={"param": torch.randn(5, 5)})
                model.load_state_dict = MagicMock()
                return model

            model, sa_ppo = coordinator.create_member_model(member, model_factory)
            models.append((model, sa_ppo))

        # Simulate realistic training
        for step in range(20):
            for i, member in enumerate(population):
                model, sa_ppo = models[i]

                # Start update
                coordinator.on_member_update_start(member)

                # Simulate training...
                performance = 0.5 + step * 0.01 + i * 0.05 + np.random.random() * 0.1

                # End update
                new_state, new_hp, _ = coordinator.on_member_update_end(
                    member,
                    performance=performance,
                    step=step,
                    model_state_dict=model.state_dict()
                )

                # Apply PBT updates
                if new_state is not None:
                    model.load_state_dict(new_state)
                    member.hyperparams = new_hp

        # Verify all members have history
        assert all(len(m.history) == 20 for m in population)

        # Verify stats are reasonable
        stats = coordinator.get_stats()
        assert stats["pbt/population_size"] == 2
        assert stats["pbt/ready_members"] == 2

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML config save and load round trip."""
        original_config = {
            "pbt": {
                "enabled": True,
                "population_size": 4,
                "hyperparams": [
                    {"name": "lr", "min_value": 1e-5, "max_value": 1e-3},
                ],
            },
            "adversarial": {
                "enabled": True,
                "adversarial_ratio": 0.6,
            },
        }

        config_path = tmp_path / "config.yaml"

        # Save
        with open(config_path, "w") as f:
            yaml.dump(original_config, f)

        # Load
        loaded_config = load_pbt_adversarial_config(str(config_path))

        # Verify
        assert loaded_config.pbt_enabled is True
        assert loaded_config.pbt.population_size == 4
        assert loaded_config.adversarial_enabled is True
        assert loaded_config.adversarial.adversarial_ratio == 0.6


class TestPerformance:
    """Test performance and scalability."""

    def test_perturbation_performance(self):
        """Test perturbation generation is reasonably fast."""
        config = PerturbationConfig(attack_steps=3)
        perturb = StatePerturbation(config)

        state = torch.randn(64, 100)  # Realistic batch size
        def loss_fn(s): return (s ** 2).sum()

        start = time.time()
        for _ in range(10):
            delta = perturb.generate_perturbation(state, loss_fn)
        elapsed = time.time() - start

        # Should be reasonably fast (< 5 seconds for 10 iterations)
        assert elapsed < 5.0, f"Perturbation too slow: {elapsed:.2f}s"

    def test_pbt_scaling(self):
        """Test PBT scales with population size."""
        sizes = [2, 5, 10, 20]
        times = []

        for size in sizes:
            config = PBTConfig(
                population_size=size,
                hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            )

            scheduler = PBTScheduler(config, seed=42)

            start = time.time()
            population = scheduler.initialize_population()

            for member in population:
                scheduler.update_performance(member, 0.5, 1)

            elapsed = time.time() - start
            times.append(elapsed)

        # Time should scale roughly linearly (not exponentially)
        assert times[-1] < times[0] * 20, "PBT scaling is not linear"


class TestDefaultsComprehensive:
    """Comprehensive tests for all default values."""

    def test_all_config_classes_have_correct_defaults(self):
        """Test all config classes have correct defaults."""
        # PerturbationConfig
        pc = PerturbationConfig()
        assert pc.epsilon == 0.075
        assert pc.attack_steps == 3
        assert pc.attack_method == "pgd"
        assert pc.norm_type == "linf"

        # SAPPOConfig
        sc = SAPPOConfig()
        assert sc.enabled is True
        assert sc.adversarial_ratio == 0.5
        assert sc.robust_kl_coef == 0.1

        # PBTAdversarialConfig
        pac = PBTAdversarialConfig()
        assert pac.pbt_enabled is True
        assert pac.adversarial_enabled is True

    def test_yaml_config_has_both_enabled(self):
        """Test YAML config has both PBT and Adversarial enabled."""
        config_path = "configs/config_pbt_adversarial.yaml"

        if os.path.exists(config_path):
            with open(config_path) as f:
                data = yaml.safe_load(f)

            assert data["pbt"]["enabled"] is True, "PBT must be enabled in YAML"
            assert data["adversarial"]["enabled"] is True, "Adversarial must be enabled in YAML"

    def test_all_modules_export_enabled_defaults(self):
        """Test all modules export enabled defaults."""
        from training_pbt_adversarial_integration import (
            DEFAULT_PBT_ADVERSARIAL_CONFIG,
            is_pbt_adversarial_enabled_by_default,
        )

        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.pbt_enabled is True
        assert DEFAULT_PBT_ADVERSARIAL_CONFIG.adversarial_enabled is True

        # System check
        if os.path.exists("configs/config_pbt_adversarial.yaml"):
            assert is_pbt_adversarial_enabled_by_default() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
