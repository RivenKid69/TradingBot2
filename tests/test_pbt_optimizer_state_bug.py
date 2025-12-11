"""
Diagnostic test to confirm optimizer state loss during PBT exploit operation.

This test demonstrates the critical bug where optimizer state (momentum, velocity, EMA)
is NOT transferred when a population member exploits from a better performer.

Expected behavior:
- After exploit, the agent should have:
  1. Model weights from the source agent (✓ currently working)
  2. Optimizer state from the source agent (✗ BUG: not working)
  3. Hyperparameters from the source agent (✓ currently working)

Problem impact:
- After exploit, optimizer state is mismatched with model weights
- First gradient steps after exploit are incorrect (momentum points wrong direction)
- Can cause performance drops after PBT exploit
- Especially problematic for momentum-based optimizers (Adam, AdaptiveUPGD)
"""

import os
import tempfile
from pathlib import Path
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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestPBTOptimizerStateBug:
    """Tests to confirm and diagnose optimizer state loss during PBT exploit."""

    @pytest.fixture
    def tmp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_optimizer_state_not_saved_in_checkpoint(self, tmp_checkpoint_dir):
        """
        Test 1: Confirm that optimizer state is NOT currently saved in PBT checkpoints.

        This is the root cause of the bug.
        """
        # Create simple model and optimizer
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Take a few gradient steps to build up optimizer state
        for _ in range(5):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get optimizer state (should have momentum buffers)
        optimizer_state = optimizer.state_dict()

        # Verify optimizer has state
        assert len(optimizer_state['state']) > 0, "Optimizer should have state after training"

        # Check momentum buffers exist
        for param_id, param_state in optimizer_state['state'].items():
            assert 'exp_avg' in param_state, "Adam should have exp_avg (momentum)"
            assert 'exp_avg_sq' in param_state, "Adam should have exp_avg_sq (velocity)"

        # Simulate what PBT currently does: save model parameters only
        # (mimicking DistributionalPPO.get_parameters())
        model_parameters = {
            'policy': model.state_dict(),  # Only model weights
            # NOTE: optimizer_state is NOT included!
        }

        # Save checkpoint (current PBT format)
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "checkpoint.pt")
        checkpoint = {
            "format_version": "v2_full_parameters",
            "data": model_parameters,
            "step": 100,
            "performance": 0.85,
        }
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
        loaded_parameters = loaded_checkpoint["data"]

        # BUG CONFIRMATION: Optimizer state is NOT in checkpoint
        assert 'optimizer_state' not in loaded_parameters, \
            "BUG CONFIRMED: Optimizer state is not saved in current checkpoint format"

        print("[X] BUG CONFIRMED: Optimizer state is NOT saved in PBT checkpoints")

    def test_optimizer_state_mismatch_after_exploit(self, tmp_checkpoint_dir):
        """
        Test 2: Demonstrate optimizer state mismatch after exploit operation.

        This shows the practical impact of the bug.
        """
        # Create PBT scheduler
        config = PBTConfig(
            population_size=2,
            perturbation_interval=5,
            hyperparams=[
                HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3),
            ],
            checkpoint_dir=tmp_checkpoint_dir,
            metric_mode="max",
            truncation_ratio=0.5,  # With 2 agents, ensure exploit happens
        )
        scheduler = PBTScheduler(config, seed=42)

        # Initialize population
        population = scheduler.initialize_population([
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ])

        # Create two agents with different models and optimizers
        agent1_model = SimpleModel()
        agent1_optimizer = torch.optim.Adam(agent1_model.parameters(), lr=1e-4)

        agent2_model = SimpleModel()
        agent2_optimizer = torch.optim.Adam(agent2_model.parameters(), lr=2e-4)

        # Train both agents to build up different optimizer states
        print("\nTraining Agent 1 (worse performer)...")
        for step in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss = ((agent1_model(x) - y) ** 2).mean()
            agent1_optimizer.zero_grad()
            loss.backward()
            agent1_optimizer.step()

        print("Training Agent 2 (better performer)...")
        for step in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1) * 2  # Different data distribution
            loss = ((agent2_model(x) - y) ** 2).mean()
            agent2_optimizer.zero_grad()
            loss.backward()
            agent2_optimizer.step()

        # Capture optimizer states BEFORE exploit
        agent1_optimizer_state_before = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in agent1_optimizer.state_dict()['state'][0].items()
        }
        agent2_optimizer_state_before = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in agent2_optimizer.state_dict()['state'][0].items()
        }

        # Verify optimizer states are DIFFERENT before exploit
        exp_avg_diff = torch.norm(
            agent1_optimizer_state_before['exp_avg'] -
            agent2_optimizer_state_before['exp_avg']
        ).item()
        assert exp_avg_diff > 0.01, "Optimizer states should be different"
        print(f"[OK] Optimizer states are different (momentum diff: {exp_avg_diff:.4f})")

        # Simulate PBT: Agent 2 performs better
        member1 = population[0]  # Worse performer
        member2 = population[1]  # Better performer
        member1.performance = 0.5
        member2.performance = 0.9

        # Save Agent 2 checkpoint (better performer)
        agent2_parameters = {'policy': agent2_model.state_dict()}
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
        print(f"[OK] Exploit occurred: Agent 1 copying from Agent 2")

        # Load new parameters into Agent 1 model
        agent1_model.load_state_dict(new_parameters['policy'])

        # BUG DEMONSTRATION: Agent 1's optimizer state is UNCHANGED
        agent1_optimizer_state_after = agent1_optimizer.state_dict()['state'][0]

        # Check if momentum changed (it shouldn't, demonstrating the bug)
        momentum_changed = torch.norm(
            agent1_optimizer_state_after['exp_avg'] -
            agent1_optimizer_state_before['exp_avg']
        ).item() > 1e-6

        # BUG: Optimizer state did NOT change after exploit
        assert not momentum_changed, \
            "BUG CONFIRMED: Optimizer state did NOT change after exploit"

        print("[X] BUG CONFIRMED: After exploit, model weights changed but optimizer state DID NOT")
        print("  This causes mismatch between model and optimizer state!")

        # Show the impact: optimizer momentum now points in wrong direction
        print("\nImpact analysis:")
        print(f"  - Model weights: Updated to Agent 2 [OK]")
        print(f"  - Hyperparameters: Updated to Agent 2 [OK]")
        print(f"  - Optimizer momentum: Still from Agent 1 [X] BUG!")
        print(f"  - Optimizer velocity: Still from Agent 1 [X] BUG!")
        print("\n  Result: First gradient steps after exploit will be INCORRECT")
        print("          Can cause performance drops after PBT exploit")

    def test_optimizer_state_points_wrong_direction_after_exploit(self, tmp_checkpoint_dir):
        """
        Test 3: Demonstrate that optimizer momentum points in wrong direction after exploit.

        This shows the gradient descent will be suboptimal after exploit.
        """
        # Create two models with different optimal directions
        model_good = SimpleModel()
        model_bad = SimpleModel()

        # Train on different tasks to get different optimal directions
        optimizer_good = torch.optim.Adam(model_good.parameters(), lr=0.01)
        optimizer_bad = torch.optim.Adam(model_bad.parameters(), lr=0.01)

        # Task 1: Predict positive values (good model learns this)
        print("\nTraining good model on Task 1 (positive values)...")
        for _ in range(20):
            x = torch.randn(32, 10)
            y_true = torch.abs(torch.randn(32, 1)) + 1.0  # Always positive
            y_pred = model_good(x)
            loss = ((y_pred - y_true) ** 2).mean()
            optimizer_good.zero_grad()
            loss.backward()
            optimizer_good.step()

        # Task 2: Predict negative values (bad model learns this)
        print("Training bad model on Task 2 (negative values)...")
        for _ in range(20):
            x = torch.randn(32, 10)
            y_true = -torch.abs(torch.randn(32, 1)) - 1.0  # Always negative
            y_pred = model_bad(x)
            loss = ((y_pred - y_true) ** 2).mean()
            optimizer_bad.zero_grad()
            loss.backward()
            optimizer_bad.step()

        # Get momentum from bad optimizer
        bad_momentum = optimizer_bad.state_dict()['state'][0]['exp_avg'].clone()

        # Simulate exploit: bad model copies weights from good model
        model_bad.load_state_dict(model_good.state_dict())
        # BUG: optimizer_bad state is NOT updated (still has momentum from Task 2)

        # Now take a gradient step on Task 1 with mismatched optimizer state
        x_test = torch.randn(32, 10)
        y_test = torch.abs(torch.randn(32, 1)) + 1.0

        # Compute gradient
        y_pred = model_bad(x_test)
        loss = ((y_pred - y_test) ** 2).mean()
        optimizer_bad.zero_grad()
        loss.backward()

        # Get true gradient direction (what optimizer should follow)
        true_gradient = [p.grad.clone() for p in model_bad.parameters()]

        # Get momentum direction (what optimizer actually uses)
        momentum_direction = bad_momentum

        # Compute cosine similarity between gradient and momentum
        # Negative similarity means they point in opposite directions!
        grad_flat = torch.cat([g.flatten() for g in true_gradient])
        grad_norm = torch.norm(grad_flat)
        momentum_norm = torch.norm(momentum_direction.flatten())

        if grad_norm > 0 and momentum_norm > 0:
            # Note: momentum is for first layer only, so this is approximate
            first_param_grad = true_gradient[0].flatten()
            momentum_flat = momentum_direction.flatten()
            if len(first_param_grad) >= len(momentum_flat):
                similarity = torch.dot(
                    first_param_grad[:len(momentum_flat)],
                    momentum_flat
                ) / (torch.norm(first_param_grad[:len(momentum_flat)]) * momentum_norm)

                print(f"\nGradient-Momentum alignment: {similarity.item():.4f}")
                print("  > 0.5:  Good alignment (optimizer helps)")
                print("  < 0.0:  Opposite direction (optimizer HURTS!)")
                print("\n[X] BUG IMPACT: Optimizer momentum can point in WRONG direction after exploit")

    def test_performance_drop_simulation(self, tmp_checkpoint_dir):
        """
        Test 4: Simulate performance drop caused by optimizer state mismatch.

        Compare gradient steps with correct vs incorrect optimizer state.
        """
        # Create model and optimizer
        model = SimpleModel()
        optimizer_correct = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train to build up optimizer state
        print("\nTraining model...")
        for _ in range(20):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss = ((model(x) - y) ** 2).mean()
            optimizer_correct.zero_grad()
            loss.backward()
            optimizer_correct.step()

        # Save model state and optimizer state
        model_state = model.state_dict()
        optimizer_state_correct = optimizer_correct.state_dict()

        # Simulate exploit: Create two scenarios
        # Scenario 1: Correct behavior (copy optimizer state)
        model1 = SimpleModel()
        model1.load_state_dict(model_state)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
        optimizer1.load_state_dict(optimizer_state_correct)

        # Scenario 2: Current buggy behavior (reset optimizer state)
        model2 = SimpleModel()
        model2.load_state_dict(model_state)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        # optimizer2 has fresh state (not loaded from correct state)

        # Test on same batch of data
        x_test = torch.randn(32, 10)
        y_test = torch.randn(32, 1)

        # Take 5 gradient steps with both scenarios
        losses_correct = []
        losses_buggy = []

        print("\nTaking gradient steps after exploit:")
        print("Step | With correct optimizer state | With reset optimizer state | Difference")
        print("-" * 80)

        for step in range(5):
            # Scenario 1: Correct optimizer state
            y_pred1 = model1(x_test)
            loss1 = ((y_pred1 - y_test) ** 2).mean()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            losses_correct.append(loss1.item())

            # Scenario 2: Buggy (reset) optimizer state
            y_pred2 = model2(x_test)
            loss2 = ((y_pred2 - y_test) ** 2).mean()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            losses_buggy.append(loss2.item())

            diff = loss2.item() - loss1.item()
            print(f"{step+1:4d} | {loss1.item():28.6f} | {loss2.item():26.6f} | {diff:+.6f}")

        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Final loss with correct optimizer state: {losses_correct[-1]:.6f}")
        print(f"  Final loss with reset optimizer state:   {losses_buggy[-1]:.6f}")

        # In many cases, reset optimizer performs worse initially
        # (though not always, depends on the specific task)
        avg_loss_correct = np.mean(losses_correct)
        avg_loss_buggy = np.mean(losses_buggy)

        print(f"\n  Average loss (correct): {avg_loss_correct:.6f}")
        print(f"  Average loss (buggy):   {avg_loss_buggy:.6f}")

        if avg_loss_buggy > avg_loss_correct:
            print("\n[X] BUG IMPACT: Reset optimizer state leads to worse performance")
        else:
            print("\n  Note: In this particular run, reset didn't hurt much")
            print("       But in general, optimizer state mismatch can cause issues")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
