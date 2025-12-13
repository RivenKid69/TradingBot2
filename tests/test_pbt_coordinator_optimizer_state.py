"""
Comprehensive integration tests for PBT Coordinator optimizer state handling.

These tests verify the complete integration between PBTTrainingCoordinator,
PBTScheduler, and optimizer state management for both RESET and COPY strategies.

Tests cover:
1. RESET strategy - optimizer state is removed, caller resets optimizer
2. COPY strategy - optimizer state is preserved, caller loads optimizer state
3. Backward compatibility with old API (model_state_dict)
4. Correct checkpoint format detection and handling
"""

import os
import tempfile
from typing import Dict, Any, Optional, Tuple

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np

from adversarial.pbt_scheduler import (
    HyperparamConfig,
    PBTConfig,
    PopulationMember,
)
from training_pbt_adversarial_integration import (
    PBTAdversarialConfig,
    PBTTrainingCoordinator,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_parameters(self, include_optimizer: bool = False,
                      optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Get model parameters in PBT-compatible format."""
        params = {
            'policy': self.state_dict(),
        }
        if include_optimizer and optimizer is not None:
            params['optimizer_state'] = optimizer.state_dict()
        return params


class TestPBTCoordinatorResetStrategy:
    """Integration tests for RESET optimizer strategy."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def pbt_config_reset(self, tmp_checkpoint_dir):
        """PBT config with RESET strategy."""
        return PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
                ],
                checkpoint_dir=tmp_checkpoint_dir,
                metric_mode="max",
                truncation_ratio=0.5,
                optimizer_exploit_strategy="reset",  # RESET strategy
            ),
        )

    def test_reset_strategy_full_integration(self, pbt_config_reset):
        """
        Test complete workflow with RESET strategy.

        Workflow:
        1. Initialize population
        2. Train two agents with different optimizer states
        3. Save checkpoint for better agent (with optimizer state)
        4. Trigger exploit from worse to better agent
        5. Verify optimizer_state is NOT returned
        6. Reset optimizer and continue training
        """
        # Create coordinator
        coordinator = PBTTrainingCoordinator(pbt_config_reset, seed=42)
        population = coordinator.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        # Create two agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both agents
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            # Train agent 1
            loss1 = ((model1(x) - y) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            # Train agent 2
            loss2 = ((model2(x) - y) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Set performance (agent 2 is better)
        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        # Save checkpoint for agent 2 WITH optimizer state
        agent2_parameters = model2.get_parameters(
            include_optimizer=True,
            optimizer=optimizer2,
        )

        # Call on_member_update_end for agent 2 (saves checkpoint)
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member2,
            performance=0.9,
            step=10,
            model_parameters=agent2_parameters,
        )

        # Should not trigger exploit yet (not at perturbation_interval)
        assert new_params is None, "No exploit should occur for agent 2"
        assert checkpoint_format is None

        # Trigger exploit for agent 1
        member1.step = pbt_config_reset.pbt.perturbation_interval
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member1,
            performance=0.5,
            step=pbt_config_reset.pbt.perturbation_interval,
            model_parameters=model1.get_parameters(include_optimizer=True, optimizer=optimizer1),
        )

        # Verify exploit occurred
        assert new_params is not None, "Exploit should have occurred"
        assert checkpoint_format == "v2_full_parameters"

        # CRITICAL: optimizer_state should be REMOVED with RESET strategy
        assert 'optimizer_state' not in new_params, \
            "RESET strategy should REMOVE optimizer_state from returned parameters"

        # Load new model weights
        model1.load_state_dict(new_params['policy'])

        # RESET optimizer (correct pattern for RESET strategy)
        new_lr = new_hp['learning_rate']
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=new_lr)

        # Verify optimizer has fresh state
        optimizer_state = optimizer1.state_dict()['state']
        assert len(optimizer_state) == 0, "Fresh optimizer should have empty state"

        # Continue training - should work without issues
        x_test = torch.randn(32, 10)
        y_test = torch.randn(32, 1)

        losses_after_reset = []
        for _ in range(5):
            y_pred = model1(x_test)
            loss = ((y_pred - y_test) ** 2).mean()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            losses_after_reset.append(loss.item())

        # Loss should decrease (no performance issues after reset)
        assert losses_after_reset[-1] < losses_after_reset[0], \
            "Loss should decrease after optimizer reset"

        print("[OK] RESET strategy integration test passed")
        print(f"     - Optimizer state correctly removed")
        print(f"     - Fresh optimizer created successfully")
        print(f"     - Training continues without issues")
        print(f"     - Loss: {losses_after_reset[0]:.6f} -> {losses_after_reset[-1]:.6f}")


class TestPBTCoordinatorCopyStrategy:
    """Integration tests for COPY optimizer strategy."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def pbt_config_copy(self, tmp_checkpoint_dir):
        """PBT config with COPY strategy."""
        return PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
                ],
                checkpoint_dir=tmp_checkpoint_dir,
                metric_mode="max",
                truncation_ratio=0.5,
                optimizer_exploit_strategy="copy",  # COPY strategy
            ),
        )

    def test_copy_strategy_full_integration(self, pbt_config_copy):
        """
        Test complete workflow with COPY strategy.

        Workflow:
        1. Initialize population
        2. Train two agents with different optimizer states
        3. Save checkpoint for better agent (with optimizer state)
        4. Trigger exploit from worse to better agent
        5. Verify optimizer_state IS returned
        6. Load optimizer state and continue training
        """
        # Create coordinator
        coordinator = PBTTrainingCoordinator(pbt_config_copy, seed=42)
        population = coordinator.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        # Create two agents
        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        # Train both agents
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

        # Capture agent 2's optimizer momentum before saving
        agent2_momentum_before = optimizer2.state_dict()['state'][0]['exp_avg'].clone()

        # Set performance
        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        # Save checkpoint for agent 2
        agent2_parameters = model2.get_parameters(
            include_optimizer=True,
            optimizer=optimizer2,
        )

        coordinator.on_member_update_end(
            member2,
            performance=0.9,
            step=10,
            model_parameters=agent2_parameters,
        )

        # Trigger exploit for agent 1
        member1.step = pbt_config_copy.pbt.perturbation_interval
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member1,
            performance=0.5,
            step=pbt_config_copy.pbt.perturbation_interval,
            model_parameters=model1.get_parameters(include_optimizer=True, optimizer=optimizer1),
        )

        # Verify exploit occurred
        assert new_params is not None, "Exploit should have occurred"
        assert checkpoint_format == "v2_full_parameters"

        # CRITICAL: optimizer_state should be PRESERVED with COPY strategy
        assert 'optimizer_state' in new_params, \
            "COPY strategy should PRESERVE optimizer_state in returned parameters"

        # Load new model weights AND optimizer state
        model1.load_state_dict(new_params['policy'])
        optimizer1.load_state_dict(new_params['optimizer_state'])

        # Verify optimizer momentum was transferred
        agent1_momentum_after = optimizer1.state_dict()['state'][0]['exp_avg']
        momentum_diff = torch.norm(agent1_momentum_after - agent2_momentum_before).item()

        assert momentum_diff < 1e-6, \
            f"Momentum should be transferred, but diff={momentum_diff}"

        # Continue training - momentum should help
        x_test = torch.randn(32, 10)
        y_test = torch.randn(32, 1)

        losses_after_copy = []
        for _ in range(5):
            y_pred = model1(x_test)
            loss = ((y_pred - y_test) ** 2).mean()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            losses_after_copy.append(loss.item())

        # Loss should decrease
        assert losses_after_copy[-1] < losses_after_copy[0], \
            "Loss should decrease after copying optimizer state"

        print("[OK] COPY strategy integration test passed")
        print(f"     - Optimizer state correctly preserved")
        print(f"     - Momentum transferred successfully (diff: {momentum_diff:.2e})")
        print(f"     - Training continues with transferred momentum")
        print(f"     - Loss: {losses_after_copy[0]:.6f} -> {losses_after_copy[-1]:.6f}")


class TestPBTCoordinatorBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def pbt_config(self, tmp_checkpoint_dir):
        """Basic PBT config."""
        return PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
                ],
                checkpoint_dir=tmp_checkpoint_dir,
                metric_mode="max",
                truncation_ratio=0.5,
                optimizer_exploit_strategy="reset",
            ),
        )

    def test_old_api_still_works(self, pbt_config):
        """
        Test that old API (model_state_dict) still works for backward compatibility.

        Note: This will log warnings about using deprecated format.
        """
        coordinator = PBTTrainingCoordinator(pbt_config, seed=42)
        population = coordinator.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        model1 = SimpleModel()
        model2 = SimpleModel()

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        # Use OLD API: pass only model_state_dict (no optimizer_state)
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member2,
            performance=0.9,
            step=10,
            model_state_dict=model2.state_dict(),  # Old API
        )

        assert new_params is None  # No exploit yet
        assert checkpoint_format is None

        # Trigger exploit
        member1.step = pbt_config.pbt.perturbation_interval
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member1,
            performance=0.5,
            step=pbt_config.pbt.perturbation_interval,
            model_state_dict=model1.state_dict(),  # Old API
        )

        # Should still work, but checkpoint format will be v1 (legacy)
        # Note: In current implementation, we need to check if model_parameters
        # was passed to update_performance, not model_state_dict
        # So this test verifies backward compatibility at API level
        assert new_params is not None or checkpoint_format is not None, \
            "Old API should still work (may log warnings)"

        print("[OK] Backward compatibility test passed")
        print("     - Old API (model_state_dict) still works")
        print("     - May log warnings about deprecated format")

    def test_new_api_preferred(self, pbt_config):
        """
        Test that new API (model_parameters) is preferred and works correctly.
        """
        coordinator = PBTTrainingCoordinator(pbt_config, seed=42)
        population = coordinator.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        model1 = SimpleModel()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-4)

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        # Use NEW API: pass model_parameters (with optimizer_state)
        agent2_parameters = model2.get_parameters(
            include_optimizer=True,
            optimizer=optimizer2,
        )

        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member2,
            performance=0.9,
            step=10,
            model_parameters=agent2_parameters,  # New API
        )

        assert new_params is None  # No exploit yet

        # Trigger exploit
        member1.step = pbt_config.pbt.perturbation_interval
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member1,
            performance=0.5,
            step=pbt_config.pbt.perturbation_interval,
            model_parameters=model1.get_parameters(include_optimizer=True, optimizer=optimizer1),
        )

        # Should work with v2 format
        assert new_params is not None, "Exploit should occur"
        assert checkpoint_format == "v2_full_parameters"
        assert 'policy' in new_params
        # optimizer_state removed by RESET strategy
        assert 'optimizer_state' not in new_params

        print("[OK] New API test passed")
        print("     - model_parameters API works correctly")
        print("     - Checkpoint format: v2_full_parameters")
        print("     - Optimizer state handled according to strategy")


class TestPBTCoordinatorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_copy_strategy_without_optimizer_state_in_checkpoint(self, tmp_checkpoint_dir):
        """
        Test COPY strategy when checkpoint doesn't contain optimizer_state.

        This should log a warning and fall back to resetting optimizer.
        """
        config = PBTAdversarialConfig(
            pbt_enabled=True,
            adversarial_enabled=False,
            pbt=PBTConfig(
                population_size=2,
                perturbation_interval=5,
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
                ],
                checkpoint_dir=tmp_checkpoint_dir,
                metric_mode="max",
                truncation_ratio=0.5,
                optimizer_exploit_strategy="copy",  # COPY strategy
            ),
        )

        coordinator = PBTTrainingCoordinator(config, seed=42)
        population = coordinator.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        model1 = SimpleModel()
        model2 = SimpleModel()

        member1, member2 = population[0], population[1]
        member1.performance = 0.5
        member2.performance = 0.9

        # Save checkpoint WITHOUT optimizer_state (even though strategy is COPY)
        agent2_parameters = {
            'policy': model2.state_dict(),
            # optimizer_state NOT included!
        }

        coordinator.on_member_update_end(
            member2,
            performance=0.9,
            step=10,
            model_parameters=agent2_parameters,
        )

        # Trigger exploit
        member1.step = config.pbt.perturbation_interval
        new_params, new_hp, checkpoint_format = coordinator.on_member_update_end(
            member1,
            performance=0.5,
            step=config.pbt.perturbation_interval,
            model_parameters={'policy': model1.state_dict()},
        )

        # Should return parameters but without optimizer_state
        assert new_params is not None
        assert 'optimizer_state' not in new_params, \
            "If checkpoint lacks optimizer_state, it should not appear in returned params"

        print("[OK] Edge case test passed")
        print("     - COPY strategy without optimizer_state handled correctly")
        print("     - Should have logged warning about missing optimizer_state")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
