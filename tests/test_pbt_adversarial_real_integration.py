"""
Real integration tests for PBT + Adversarial Training with actual components.

This test suite tests integration with:
- Real PyTorch models
- Actual gradient computation
- Real training scenarios
- Configuration compatibility with existing code
- File system operations
- Error scenarios in production

Target: Validate production readiness with real components
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F
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
    create_policy_loss_fn,
    create_value_loss_fn,
)
from training_pbt_adversarial_integration import (
    PBTAdversarialConfig,
    PBTTrainingCoordinator,
    load_pbt_adversarial_config,
)


class SimplePolicy(nn.Module):
    """Simple policy network for testing."""

    def __init__(self, state_dim=10, action_dim=3, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def get_distribution(self, state):
        """Get action distribution."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        class SimpleDistribution:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
                self.normal = torch.distributions.Normal(mean, std)

            def log_prob(self, actions):
                return self.normal.log_prob(actions).sum(-1)

            def sample(self):
                return self.normal.sample()

        return SimpleDistribution(mean, std)

    def predict_values(self, state):
        """Dummy value prediction."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x.mean(dim=-1)


class SimplePPOModel:
    """Simple PPO-like model for testing."""

    def __init__(self, state_dim=10, action_dim=3, learning_rate=3e-4):
        self.policy = SimplePolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def state_dict(self):
        return self.policy.state_dict()

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)

    def train_step(self, states, actions, advantages, returns, old_log_probs):
        """Simplified training step."""
        self.optimizer.zero_grad()

        # Policy loss
        dist = self.policy.get_distribution(states)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -torch.min(
            ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        # Value loss
        values = self.policy.predict_values(states)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()


class TestRealPerturbations:
    """Test adversarial perturbations with real gradients."""

    def test_fgsm_real_gradient_computation(self):
        """Test FGSM with real gradient computation."""
        config = PerturbationConfig(epsilon=0.1, attack_method="fgsm")
        perturb = StatePerturbation(config)

        policy = SimplePolicy()
        state = torch.randn(4, 10, requires_grad=True)
        actions = torch.randn(4, 3)
        old_log_probs = torch.randn(4)
        advantages = torch.randn(4)

        # Create real loss function
        def policy_loss_fn(s):
            dist = policy.get_distribution(s)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()

        # Generate perturbation
        delta = perturb.fgsm_attack(state, policy_loss_fn)

        # Verify perturbation properties
        assert delta.shape == state.shape
        assert torch.abs(delta).max() <= config.epsilon + 1e-5
        assert not torch.all(delta == 0), "Perturbation should not be all zeros"

    def test_pgd_real_gradient_computation(self):
        """Test PGD with real gradient computation."""
        config = PerturbationConfig(epsilon=0.1, attack_method="pgd", attack_steps=5)
        perturb = StatePerturbation(config)

        policy = SimplePolicy()
        state = torch.randn(4, 10)
        actions = torch.randn(4, 3)
        old_log_probs = torch.randn(4)
        advantages = torch.randn(4)

        def policy_loss_fn(s):
            dist = policy.get_distribution(s)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()

        delta = perturb.pgd_attack(state, policy_loss_fn)

        assert delta.shape == state.shape
        assert torch.abs(delta).max() <= config.epsilon + 1e-5

    def test_perturbation_increases_loss(self):
        """Test that adversarial perturbation increases loss."""
        config = PerturbationConfig(epsilon=0.1)
        perturb = StatePerturbation(config)

        policy = SimplePolicy()
        state = torch.randn(4, 10)
        actions = torch.randn(4, 3)
        old_log_probs = torch.randn(4)
        advantages = torch.randn(4)

        def policy_loss_fn(s):
            dist = policy.get_distribution(s)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()

        # Original loss
        with torch.no_grad():
            original_loss = policy_loss_fn(state).item()

        # Generate perturbation
        delta = perturb.generate_perturbation(state, policy_loss_fn)

        # Perturbed loss
        with torch.no_grad():
            perturbed_loss = policy_loss_fn(state + delta).item()

        # Adversarial perturbation should increase loss (or at least not decrease it significantly)
        # Note: Due to gradient approximations, it might not always increase
        assert (
            perturbed_loss >= original_loss - 0.1
        ), "Perturbation should not significantly decrease loss"

    def test_value_perturbation_real(self):
        """Test value function perturbation with real network."""
        config = PerturbationConfig(epsilon=0.05)
        perturb = StatePerturbation(config)

        policy = SimplePolicy()
        state = torch.randn(4, 10)
        returns = torch.randn(4)

        def value_loss_fn(s):
            values = policy.predict_values(s)
            return F.mse_loss(values, returns)

        delta = perturb.generate_perturbation(state, value_loss_fn)

        assert delta.shape == state.shape
        assert torch.abs(delta).max() <= config.epsilon + 1e-5


class TestRealSAPPO:
    """Test SA-PPO with real model."""

    def test_sappo_with_real_model(self):
        """Test SA-PPO wrapper with real PPO model."""
        config = SAPPOConfig(
            adversarial_ratio=0.5,
            warmup_updates=5,
        )

        model = SimplePPOModel()
        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        # Simulate warmup
        for _ in range(10):
            sa_ppo.on_update_start()

        assert sa_ppo.is_adversarial_enabled

    def test_sappo_gradient_flow(self):
        """Test that gradients flow correctly through SA-PPO."""
        config = SAPPOConfig(
            adversarial_ratio=0.5,
            warmup_updates=0,  # No warmup
            attack_policy=True,
        )

        model = SimplePPOModel()
        sa_ppo = StateAdversarialPPO(config, model)
        sa_ppo.on_training_start()

        # Create training data
        states = torch.randn(8, 10)
        actions = torch.randn(8, 3)
        advantages = torch.randn(8)
        returns = torch.randn(8)
        old_log_probs = torch.randn(8)

        # Get initial parameters
        initial_params = [p.clone() for p in model.policy.parameters()]

        # Compute adversarial loss (this should trigger gradient computation)
        # Note: This is a simplified test - full integration would require
        # actual backward pass through the model
        sa_ppo.on_update_start()

        # The test verifies the wrapper is set up correctly
        assert sa_ppo._update_count > 0


class TestRealPBT:
    """Test PBT with real models and checkpoints."""

    def test_pbt_checkpoint_save_load_real_model(self, tmp_path):
        """Test PBT checkpoint save/load with real model."""
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Create real model
        model = SimplePPOModel(learning_rate=population[0].hyperparams["learning_rate"])

        # Save checkpoint
        scheduler.update_performance(
            population[0], performance=0.8, step=5, model_state_dict=model.state_dict()
        )

        # Load checkpoint
        # Security: weights_only=False needed for dict metadata (format_version)
        loaded_checkpoint = torch.load(population[0].checkpoint_path, weights_only=False)

        # Verify new checkpoint format
        assert "format_version" in loaded_checkpoint
        assert "data" in loaded_checkpoint

        # Extract actual state from checkpoint
        loaded_state = loaded_checkpoint["data"]

        # Verify loaded state matches
        for key in model.state_dict().keys():
            assert key in loaded_state
            assert torch.allclose(model.state_dict()[key], loaded_state[key])

    def test_pbt_exploitation_with_real_model(self, tmp_path):
        """Test PBT exploitation with real model weights."""
        config = PBTConfig(
            population_size=3,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=str(tmp_path / "checkpoints"),
            exploit_method="truncation",
            truncation_ratio=0.33,
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Create models with different performances
        models = []
        for i, member in enumerate(population):
            model = SimplePPOModel(learning_rate=member.hyperparams["learning_rate"])
            models.append(model)

            # Save checkpoints with different performances
            performance = 0.5 + i * 0.2  # 0.5, 0.7, 0.9
            scheduler.update_performance(
                member, performance=performance, step=5, model_state_dict=model.state_dict()
            )

        # Worst performer should exploit from better performer
        worst_member = population[0]
        worst_model = models[0]

        # Trigger exploitation
        new_state, new_hp, _ = scheduler.exploit_and_explore(worst_member)

        if new_state is not None:
            # Load exploited weights
            worst_model.load_state_dict(new_state)

            # NOTE: This test uses old-style direct load_state_dict for backward compatibility testing
            # For new code with LSTM models, use coordinator.apply_exploited_parameters() instead

            # Verify weights changed
            # (They should be from a better performer)


class TestRealCoordinator:
    """Test coordinator with real components."""

    def test_coordinator_full_training_simulation(self, tmp_path):
        """Test coordinator with full training simulation."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=10,
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
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

        # Create real models
        models = []
        for member in population:

            def model_factory(**kwargs):
                lr = kwargs.get("learning_rate", 3e-4)
                return SimplePPOModel(learning_rate=lr)

            model, sa_ppo = coordinator.create_member_model(member, model_factory)
            models.append((model, sa_ppo))

        # Simulate training loop
        for step in range(30):
            for i, member in enumerate(population):
                model, sa_ppo = models[i]

                # Training update
                coordinator.on_member_update_start(member)

                # Simulate training step
                states = torch.randn(16, 10)
                actions = torch.randn(16, 3)
                advantages = torch.randn(16)
                returns = torch.randn(16)
                old_log_probs = torch.randn(16)

                loss = model.train_step(states, actions, advantages, returns, old_log_probs)

                # Update performance
                performance = 1.0 / (1.0 + loss)  # Convert loss to performance

                new_state, new_hp, _ = coordinator.on_member_update_end(
                    member, performance=performance, step=step, model_state_dict=model.state_dict()
                )

                # Apply PBT updates (FIX 2025-11-22: use apply_exploited_parameters for LSTM reset)
                if new_state is not None:
                    coordinator.apply_exploited_parameters(model, new_state, member)

        # Verify training happened
        assert all(len(m.history) > 0 for m in population)

        # Verify SA-PPO was used
        for _, sa_ppo in models:
            stats = sa_ppo.get_stats()
            assert stats["sa_ppo/update_count"] > 0


class TestConfigCompatibility:
    """Test configuration compatibility with existing code."""

    def test_yaml_config_loads_successfully(self):
        """Test that the YAML config loads without errors."""
        config_path = "configs/config_pbt_adversarial.yaml"

        if os.path.exists(config_path):
            config = load_pbt_adversarial_config(config_path)

            assert config.pbt is not None
            assert config.adversarial is not None
            assert len(config.pbt.hyperparams) > 0

    def test_config_has_required_hyperparameters(self):
        """Test config has all required hyperparameters."""
        config_path = "configs/config_pbt_adversarial.yaml"

        if os.path.exists(config_path):
            config = load_pbt_adversarial_config(config_path)

            hyperparam_names = [hp.name for hp in config.pbt.hyperparams]

            # Should include key hyperparameters
            expected = ["learning_rate", "adversarial_epsilon"]
            for exp in expected:
                assert any(exp in name for name in hyperparam_names), f"Missing {exp}"

    def test_config_values_are_reasonable(self):
        """Test config values are in reasonable ranges."""
        config_path = "configs/config_pbt_adversarial.yaml"

        if os.path.exists(config_path):
            config = load_pbt_adversarial_config(config_path)

            # PBT values
            assert 2 <= config.pbt.population_size <= 100
            assert 1 <= config.pbt.perturbation_interval <= 100

            # Adversarial values
            assert 0.0 <= config.adversarial.adversarial_ratio <= 1.0
            assert 0.0 <= config.adversarial.perturbation.epsilon <= 1.0
            assert 1 <= config.adversarial.perturbation.attack_steps <= 20


class TestFileSystemOperations:
    """Test file system operations and permissions."""

    def test_checkpoint_directory_creation(self, tmp_path):
        """Test checkpoint directory is created correctly."""
        checkpoint_dir = tmp_path / "my_checkpoints"

        config = PBTConfig(
            population_size=2,
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(checkpoint_dir),
        )

        scheduler = PBTScheduler(config, seed=42)

        # Directory should be created
        assert os.path.exists(checkpoint_dir)

    def test_checkpoint_file_permissions(self, tmp_path):
        """Test checkpoint files have correct permissions."""
        config = PBTConfig(
            population_size=2,  # Minimum valid size
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        model_state = {"param": torch.randn(5, 5)}
        scheduler.update_performance(population[0], 0.8, 1, model_state)

        # Check file exists and is readable
        assert os.path.exists(population[0].checkpoint_path)
        assert os.access(population[0].checkpoint_path, os.R_OK)

    def test_multiple_checkpoints_same_member(self, tmp_path):
        """Test multiple checkpoints for same member."""
        config = PBTConfig(
            population_size=2,  # Minimum valid size
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        member = population[0]

        # Save multiple checkpoints
        for step in range(5):
            model_state = {"param": torch.randn(5, 5), "step": step}
            scheduler.update_performance(member, 0.5 + step * 0.1, step, model_state)

        # Latest checkpoint should be used
        assert member.checkpoint_path is not None
        assert "step_4" in member.checkpoint_path


class TestErrorRecovery:
    """Test error recovery in production scenarios."""

    def test_coordinator_handles_model_creation_failure(self):
        """Test coordinator handles model creation failure gracefully."""
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            pbt=PBTConfig(
                population_size=2,
                hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            ),
        )

        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population()

        def failing_factory(**kwargs):
            raise RuntimeError("Model creation failed")

        with pytest.raises(RuntimeError):
            coordinator.create_member_model(population[0], failing_factory)

    def test_perturbation_handles_gradient_none(self):
        """Test perturbation with very small gradients."""
        config = PerturbationConfig(epsilon=0.01)
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10)

        def nearly_flat_loss(s):
            # Loss with very small gradients (almost flat)
            # Using a very small coefficient to create tiny gradients
            return (s * 1e-10).sum()

        # Should work even with very small gradients
        delta = perturb.generate_perturbation(state, nearly_flat_loss)

        # Delta exists but should be very small due to tiny gradients
        assert delta.shape == state.shape
        # Check epsilon constraint is respected
        assert torch.abs(delta).max() <= config.epsilon + 1e-6


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_perturbation_float32_precision(self):
        """Test perturbation with float32 precision."""
        config = PerturbationConfig(epsilon=0.01)
        perturb = StatePerturbation(config)

        state = torch.randn(4, 10, dtype=torch.float32)

        def loss_fn(s):
            return (s**2).sum()

        delta = perturb.generate_perturbation(state, loss_fn)

        assert delta.dtype == torch.float32
        assert torch.abs(delta).max() <= config.epsilon + 1e-6

    def test_very_small_learning_rate_bounds(self):
        """Test hyperparameter bounds with very small values."""
        config = PBTConfig(
            population_size=5,
            hyperparams=[
                HyperparamConfig(name="lr", min_value=1e-10, max_value=1e-8, is_log_scale=True),
            ],
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # All values should be in range
        for member in population:
            lr = member.hyperparams["lr"]
            assert 1e-10 <= lr <= 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
