"""
Comprehensive tests for PBT optimizer state handling (RESET and COPY strategies).

These tests verify that the fix for optimizer state mismatch works correctly
for both strategies:
- 'reset': Reset optimizer state after exploit (recommended, default)
- 'copy': Copy optimizer state from source agent (advanced)

The fix ensures no performance drops after PBT exploit operation.
"""

import os
import tempfile
from typing import Dict, Any

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np

from adversarial.pbt_scheduler import (
    HyperparamConfig,
    PBTConfig,
    PopulationMember,
    PBTScheduler,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestPBTOptimizerStateReset:
    """Tests for 'reset' optimizer exploit strategy."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_reset_strategy_removes_optimizer_state(self, tmp_checkpoint_dir):
        """
        Test that 'reset' strategy removes optimizer state from exploit parameters.

        This ensures that after exploit, the caller can reset the optimizer
        with fresh state, avoiding mismatch with the new model weights.
        """
        # Create PBT scheduler with RESET strategy
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,
            optimizer_exploit_strategy="reset",  # RESET strategy
        )
        scheduler = PBTScheduler(config, seed=42)

        # Initialize population
        population = scheduler.initialize_population(
            [
                {"learning_rate": 1e-4},
                {"learning_rate": 2e-4},
            ]
        )

        # Create two agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both to build optimizer state
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Save Agent 2 checkpoint WITH optimizer state
        agent2_parameters = {
            "policy": model2.state_dict(),
            "optimizer_state": optimizer2.state_dict(),  # Include optimizer state
        }

        member1 = population[0]  # Worse performer
        member2 = population[1]  # Better performer
        member1.performance = 0.5
        member2.performance = 0.9

        scheduler.update_performance(
            member2,
            performance=0.9,
            step=10,
            model_parameters=agent2_parameters,
        )

        # Agent 1 exploits from Agent 2
        member1.step = config.perturbation_interval
        new_parameters, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(member1)

        assert new_parameters is not None, "Exploit should have occurred"

        # CRITICAL CHECK: Optimizer state should be REMOVED with reset strategy
        assert (
            "optimizer_state" not in new_parameters
        ), "RESET strategy should REMOVE optimizer_state from parameters"

        print("[OK] RESET strategy correctly removes optimizer state")
        print("     Caller can now reset optimizer with fresh state")

    def test_reset_strategy_allows_fresh_optimizer(self, tmp_checkpoint_dir):
        """
        Test that with 'reset' strategy, agent can create fresh optimizer after exploit.

        This demonstrates the recommended pattern: after exploit, reset optimizer.
        """
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,
            optimizer_exploit_strategy="reset",
        )
        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population(
            [
                {"learning_rate": 1e-4},
                {"learning_rate": 2e-4},
            ]
        )

        # Setup agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Save checkpoint
        agent2_parameters = {
            "policy": model2.state_dict(),
            "optimizer_state": optimizer2.state_dict(),
        }

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        scheduler.update_performance(member2, 0.9, 10, model_parameters=agent2_parameters)

        # Exploit
        member1.step = config.perturbation_interval
        new_parameters, new_hyperparams, _ = scheduler.exploit_and_explore(member1)

        # Load new weights
        model1.load_state_dict(new_parameters["policy"])

        # RESET OPTIMIZER (recommended pattern)
        new_lr = new_hyperparams["learning_rate"]
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=new_lr)

        # Verify optimizer has fresh state (no momentum)
        optimizer_state = optimizer1.state_dict()["state"]
        assert len(optimizer_state) == 0, "Fresh optimizer should have empty state"

        print("[OK] RESET strategy allows fresh optimizer creation")
        print("     Optimizer has no momentum/velocity (fresh state)")


class TestPBTOptimizerStateCopy:
    """Tests for 'copy' optimizer exploit strategy."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_copy_strategy_preserves_optimizer_state(self, tmp_checkpoint_dir):
        """
        Test that 'copy' strategy preserves optimizer state in exploit parameters.

        This allows advanced users to copy optimizer state from source agent.
        """
        # Create PBT scheduler with COPY strategy
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,
            optimizer_exploit_strategy="copy",  # COPY strategy
        )
        scheduler = PBTScheduler(config, seed=42)

        # Initialize population
        population = scheduler.initialize_population(
            [
                {"learning_rate": 1e-4},
                {"learning_rate": 2e-4},
            ]
        )

        # Create agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Save Agent 2 checkpoint WITH optimizer state
        agent2_parameters = {
            "policy": model2.state_dict(),
            "optimizer_state": optimizer2.state_dict(),
        }

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        scheduler.update_performance(member2, 0.9, 10, model_parameters=agent2_parameters)

        # Exploit
        member1.step = config.perturbation_interval
        new_parameters, new_hyperparams, _ = scheduler.exploit_and_explore(member1)

        assert new_parameters is not None, "Exploit should have occurred"

        # CRITICAL CHECK: Optimizer state should be PRESERVED with copy strategy
        assert (
            "optimizer_state" in new_parameters
        ), "COPY strategy should PRESERVE optimizer_state in parameters"

        print("[OK] COPY strategy correctly preserves optimizer state")
        print("     Optimizer state can be loaded from source agent")

    def test_copy_strategy_transfers_momentum(self, tmp_checkpoint_dir):
        """
        Test that 'copy' strategy successfully transfers optimizer momentum.

        This demonstrates that momentum from source agent is copied correctly.
        """
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,
            optimizer_exploit_strategy="copy",
        )
        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population(
            [
                {"learning_rate": 1e-4},
                {"learning_rate": 2e-4},
            ]
        )

        # Create agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Capture Agent 2 optimizer state before saving
        agent2_optimizer_state_before = optimizer2.state_dict()["state"][0]
        agent2_momentum_before = agent2_optimizer_state_before["exp_avg"].clone()

        # Save checkpoint
        agent2_parameters = {
            "policy": model2.state_dict(),
            "optimizer_state": optimizer2.state_dict(),
        }

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        scheduler.update_performance(member2, 0.9, 10, model_parameters=agent2_parameters)

        # Exploit
        member1.step = config.perturbation_interval
        new_parameters, new_hyperparams, _ = scheduler.exploit_and_explore(member1)

        # Load new weights AND optimizer state
        model1.load_state_dict(new_parameters["policy"])
        optimizer1.load_state_dict(new_parameters["optimizer_state"])

        # Verify optimizer momentum was transferred
        agent1_optimizer_state_after = optimizer1.state_dict()["state"][0]
        agent1_momentum_after = agent1_optimizer_state_after["exp_avg"]

        # Momentum should match Agent 2's momentum
        momentum_diff = torch.norm(agent1_momentum_after - agent2_momentum_before).item()
        assert (
            momentum_diff < 1e-6
        ), f"Momentum should be transferred from source agent, but diff={momentum_diff}"

        print("[OK] COPY strategy successfully transfers momentum")
        print(f"     Momentum difference: {momentum_diff:.2e} (< 1e-6)")


class TestPBTOptimizerStateFix:
    """Integration tests demonstrating the fix for optimizer state bug."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_no_performance_drop_with_reset_strategy(self, tmp_checkpoint_dir):
        """
        Test that RESET strategy prevents performance drops after exploit.

        Before fix: Performance drops because optimizer state is mismatched
        After fix: No performance drop because optimizer is reset properly
        """
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,
            optimizer_exploit_strategy="reset",
        )
        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population(
            [
                {"learning_rate": 1e-4},
                {"learning_rate": 2e-4},
            ]
        )

        # Create agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both
        for _ in range(20):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Save checkpoint
        agent2_parameters = {
            "policy": model2.state_dict(),
            "optimizer_state": optimizer2.state_dict(),
        }

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        scheduler.update_performance(member2, 0.9, 10, model_parameters=agent2_parameters)

        # Exploit
        member1.step = config.perturbation_interval
        new_parameters, new_hyperparams, _ = scheduler.exploit_and_explore(member1)

        # Load new weights
        model1.load_state_dict(new_parameters["policy"])

        # RESET optimizer (FIX!)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=new_hyperparams["learning_rate"])

        # Test performance after exploit
        x_test = torch.randn(32, 10)
        y_test = torch.randn(32, 1)

        losses_after_exploit = []
        for step in range(5):
            y_pred = model1(x_test)
            loss = ((y_pred - y_test) ** 2).mean()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            losses_after_exploit.append(loss.item())

        # Loss should decrease (no performance drop)
        assert (
            losses_after_exploit[-1] < losses_after_exploit[0]
        ), "Loss should decrease after exploit (no performance drop)"

        print("[OK] RESET strategy prevents performance drops")
        print(
            f"     Loss decreased from {losses_after_exploit[0]:.6f} to {losses_after_exploit[-1]:.6f}"
        )

    def test_config_validation(self):
        """Test that PBTConfig validates optimizer_exploit_strategy."""
        # Valid strategies
        config_reset = PBTConfig(population_size=2, optimizer_exploit_strategy="reset")
        assert config_reset.optimizer_exploit_strategy == "reset"

        config_copy = PBTConfig(population_size=2, optimizer_exploit_strategy="copy")
        assert config_copy.optimizer_exploit_strategy == "copy"

        # Invalid strategy should raise ValueError
        with pytest.raises(ValueError, match="optimizer_exploit_strategy must be"):
            PBTConfig(population_size=2, optimizer_exploit_strategy="invalid")

        print("[OK] Config validation works correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
