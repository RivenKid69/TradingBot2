#!/usr/bin/env python3
"""
Simple Test Runner for UPGD Integration (without pytest)

Runs basic validation tests for UPGD + PBT + Twin Critics + VGS integration.
"""

import sys
import traceback
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from optimizers import UPGD, AdaptiveUPGD, UPGDW
from variance_gradient_scaler import VarianceGradientScaler
from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name, test_func):
        """Run a single test function."""
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        try:
            test_func()
            print("✓ PASSED")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            traceback.print_exc()
            self.failed += 1
            self.errors.append((name, str(e)))

    def summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")

        if self.errors:
            print(f"\n{'='*60}")
            print("FAILURES:")
            print(f"{'='*60}")
            for name, error in self.errors:
                print(f"\n{name}:")
                print(f"  {error}")

        print(f"\n{'='*60}")
        if self.failed == 0:
            print("✓ ALL TESTS PASSED!")
            return 0
        else:
            print(f"✗ {self.failed} TEST(S) FAILED")
            return 1


def make_simple_env():
    """Create simple test environment."""
    return DummyVecEnv([lambda: gym.make("CartPole-v1")])


# ============================================================================
# Test Definitions
# ============================================================================


def test_upgd_basic_instantiation():
    """Test basic UPGD optimizer instantiation."""
    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0


def test_adaptive_upgd_instantiation():
    """Test AdaptiveUPGD optimizer instantiation."""
    model = nn.Linear(4, 2)
    optimizer = AdaptiveUPGD(model.parameters(), lr=1e-3)
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0


def test_upgd_single_step():
    """Test UPGD can perform a single optimization step."""
    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)

    x = torch.randn(8, 4)
    target = torch.randn(8, 2)

    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).sum()
    loss.backward()
    optimizer.step()

    # Check optimizer state was created
    assert len(optimizer.state) > 0


def test_upgd_numerical_stability():
    """Test UPGD maintains numerical stability over many steps."""
    model = nn.Linear(4, 2)
    optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4, sigma=0.01)

    for step in range(100):
        x = torch.randn(8, 4)
        target = torch.randint(0, 2, (8,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Check parameters are finite
        for param in model.parameters():
            assert torch.all(torch.isfinite(param)), f"NaN/Inf at step {step}"


def test_vgs_basic_functionality():
    """Test Variance Gradient Scaler basic functionality."""
    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)
    vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=10)

    x = torch.randn(8, 4)
    target = torch.randint(0, 2, (8,))

    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()

    scaling_factor = vgs.scale_gradients()
    assert 0.0 < scaling_factor <= 1.0

    optimizer.step()
    vgs.step()

    assert vgs._step_count == 1


def test_upgd_vgs_integration():
    """Test UPGD with VGS integration over multiple steps."""
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))

    optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4, sigma=0.01)
    vgs = VarianceGradientScaler(
        model.parameters(), enabled=True, beta=0.99, alpha=0.1, warmup_steps=20
    )

    for step in range(50):
        x = torch.randn(16, 4)
        target = torch.randint(0, 2, (16,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        vgs.scale_gradients()
        optimizer.step()
        vgs.step()

        # Check no NaN/Inf
        for param in model.parameters():
            assert torch.all(torch.isfinite(param)), f"NaN/Inf at step {step}"

        # Check VGS statistics
        if vgs._grad_mean_ema is not None:
            assert np.isfinite(vgs._grad_mean_ema)
            assert np.isfinite(vgs.get_normalized_variance())


def test_pbt_scheduler_initialization():
    """Test PBT scheduler initialization."""
    config = PBTConfig(
        population_size=3,
        perturbation_interval=5,
        hyperparams=[
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
            HyperparamConfig(name="sigma", min_value=0.001, max_value=0.1),
        ],
    )

    scheduler = PBTScheduler(config, seed=42)
    population = scheduler.initialize_population()

    assert len(population) == 3
    for member in population:
        assert "lr" in member.hyperparams
        assert "sigma" in member.hyperparams


def test_pbt_exploit_and_explore():
    """Test PBT exploit and explore operations."""
    config = PBTConfig(
        population_size=4,
        perturbation_interval=10,
        hyperparams=[
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
        ],
    )

    scheduler = PBTScheduler(config, seed=123)
    population = scheduler.initialize_population()

    # Assign different performances
    for i, member in enumerate(population):
        performance = 10.0 + i * 2.0
        scheduler.update_performance(
            member,
            performance=performance,
            step=5,
            model_state_dict={"dummy": torch.randn(2, 2)},
        )

    # Trigger exploit/explore for worst performer
    worst_member = population[0]
    worst_member.step = 10

    new_state_dict, new_hyperparams = scheduler.exploit_and_explore(
        worst_member,
        model_state_dict={"dummy": torch.randn(2, 2)},
    )

    assert "lr" in new_hyperparams


def test_upgd_with_ppo():
    """Test UPGD integration with DistributionalPPO."""
    env = make_simple_env()

    model = DistributionalPPO(
        "MlpPolicy",
        env,
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 3e-4, "sigma": 0.01},
        n_steps=64,
        n_epochs=2,
        batch_size=64,
        verbose=0,
    )

    assert isinstance(model.policy.optimizer, AdaptiveUPGD)

    # Train
    model.learn(total_timesteps=256)

    # Check optimizer state exists
    assert len(model.policy.optimizer.state) > 0

    # Check parameters are finite
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param))

    env.close()


def test_twin_critics_with_upgd():
    """Test Twin Critics with UPGD optimizer."""
    env = make_simple_env()

    model = DistributionalPPO(
        "MlpPolicy",
        env,
        optimizer_class="adaptive_upgd",
        use_twin_critics=True,
        adversarial_training=True,
        n_steps=64,
        n_epochs=2,
        verbose=0,
    )

    # Check Twin Critics is active
    assert hasattr(model.policy, "critics")
    assert len(model.policy.critics) == 2

    # Train
    model.learn(total_timesteps=256)

    # Check parameters are finite
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param))

    env.close()


def test_full_integration():
    """Test full integration of UPGD + Twin Critics + VGS."""
    env = make_simple_env()

    model = DistributionalPPO(
        "MlpPolicy",
        env,
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 3e-4, "sigma": 0.01},
        use_twin_critics=True,
        adversarial_training=True,
        vgs_enabled=True,
        vgs_alpha=0.1,
        vgs_warmup_steps=30,
        n_steps=128,
        n_epochs=2,
        batch_size=64,
        verbose=0,
    )

    # Train
    model.learn(total_timesteps=512)

    # Check all parameters are finite
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters contain NaN/Inf"

    # Check optimizer state
    optimizer = model.policy.optimizer
    for p in optimizer.state:
        state = optimizer.state[p]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"Optimizer state {key} contains NaN/Inf"

    env.close()


def test_utility_computation():
    """Test UPGD utility computation correctness."""
    model = nn.Linear(2, 2, bias=False)
    optimizer = UPGD(model.parameters(), lr=1e-3, beta_utility=0.9)

    with torch.no_grad():
        model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    x = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([0.0, 1.0])

    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).sum()
    loss.backward()

    grad = model.weight.grad.clone()
    param = model.weight.data.clone()

    expected_utility = -grad * param

    optimizer.step()

    state = optimizer.state[model.weight]
    avg_utility = state["avg_utility"]

    # First step: avg_utility = (1 - beta) * utility = 0.1 * expected_utility
    expected_avg_utility = 0.1 * expected_utility

    assert torch.allclose(avg_utility, expected_avg_utility, atol=1e-6)


def test_vgs_warmup_behavior():
    """Test VGS warmup behavior."""
    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)
    vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=10)

    scaling_factors = []

    for step in range(20):
        x = torch.randn(8, 4)
        target = torch.randint(0, 2, (8,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        scaling_factor = vgs.scale_gradients()
        scaling_factors.append(scaling_factor)

        optimizer.step()
        vgs.step()

    # During warmup, scaling should be 1.0
    for i in range(10):
        assert scaling_factors[i] == 1.0, f"Warmup step {i} should have scaling=1.0"


def test_weight_decay():
    """Test UPGDW weight decay."""
    model = nn.Linear(4, 2, bias=False)

    with torch.no_grad():
        model.weight.data = torch.randn(2, 4) * 10.0

    optimizer = UPGDW(model.parameters(), lr=1e-3, weight_decay=0.1)

    initial_norm = model.weight.data.norm().item()

    # Run with minimal gradients
    for _ in range(10):
        optimizer.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight.data) + 1e-6
        optimizer.step()

    final_norm = model.weight.data.norm().item()

    # Weight decay should reduce norm
    assert final_norm < initial_norm


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run all tests."""
    print("=" * 60)
    print("UPGD + PBT + Twin Critics + VGS - Simple Test Suite")
    print("=" * 60)

    runner = TestRunner()

    # Basic tests
    runner.run_test("UPGD Basic Instantiation", test_upgd_basic_instantiation)
    runner.run_test("AdaptiveUPGD Instantiation", test_adaptive_upgd_instantiation)
    runner.run_test("UPGD Single Step", test_upgd_single_step)
    runner.run_test("UPGD Numerical Stability", test_upgd_numerical_stability)

    # VGS tests
    runner.run_test("VGS Basic Functionality", test_vgs_basic_functionality)
    runner.run_test("UPGD + VGS Integration", test_upgd_vgs_integration)
    runner.run_test("VGS Warmup Behavior", test_vgs_warmup_behavior)

    # PBT tests
    runner.run_test("PBT Scheduler Initialization", test_pbt_scheduler_initialization)
    runner.run_test("PBT Exploit and Explore", test_pbt_exploit_and_explore)

    # Integration tests
    runner.run_test("UPGD with PPO", test_upgd_with_ppo)
    runner.run_test("Twin Critics with UPGD", test_twin_critics_with_upgd)
    runner.run_test("Full Integration", test_full_integration)

    # Deep validation
    runner.run_test("Utility Computation", test_utility_computation)
    runner.run_test("Weight Decay", test_weight_decay)

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
