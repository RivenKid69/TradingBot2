#!/usr/bin/env python3
"""
Standalone Integration Tests for UPGD + PBT + Twin Critics + VGS

Runs without pytest dependency - suitable for CI/CD environments.
Tests the full integration of all advanced optimization components.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import numpy
        print(f"[OK] NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        import gymnasium as gym
        print(f"[OK] Gymnasium {gym.__version__}")
    except ImportError:
        missing.append("gymnasium")

    try:
        import stable_baselines3
        print(f"[OK] Stable-Baselines3 {stable_baselines3.__version__}")
    except ImportError:
        missing.append("stable-baselines3")

    if missing:
        print(f"\n[FAIL] Missing dependencies: {', '.join(missing)}")
        print("Cannot run tests without required packages.")
        return False

    print("[OK] All dependencies available\n")
    return True


def run_test(test_name, test_func):
    """Run a single test and report result."""
    print(f"{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    try:
        test_func()
        print(f"[OK] PASSED: {test_name}\n")
        return True
    except AssertionError as e:
        print(f"[FAIL] FAILED: {test_name}")
        print(f"  Assertion Error: {e}\n")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {test_name}")
        print(f"  Exception: {e}\n")
        traceback.print_exc()
        return False


# ===========================================================================
# Test Implementations
# ===========================================================================

def test_01_upgd_basic():
    """Test basic UPGD optimizer functionality."""
    import torch
    import torch.nn as nn
    from optimizers import UPGD

    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)

    # Create training data
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)

    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).sum()
    loss.backward()
    optimizer.step()

    # Verify optimizer state
    assert len(optimizer.state) > 0, "Optimizer should have state"
    assert model.weight in optimizer.state, "Weight should be in optimizer state"

    state = optimizer.state[model.weight]
    assert 'step' in state, "Should track step count"
    assert 'avg_utility' in state, "Should track utility"
    assert state['step'] == 1, "Step count should be 1"

    print("  [OK] UPGD optimizer initialized and step completed")
    print(f"  [OK] Optimizer state created for {len(optimizer.state)} parameters")


def test_02_adaptive_upgd():
    """Test AdaptiveUPGD with Adam-style moments."""
    import torch
    import torch.nn as nn
    from optimizers import AdaptiveUPGD

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    optimizer = AdaptiveUPGD(
        model.parameters(),
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        sigma=0.01
    )

    # Multiple training steps
    for i in range(10):
        x = torch.randn(16, 4)
        target = torch.randint(0, 2, (16,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    # Verify moments are tracked
    has_moments = False
    for param in model.parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            assert 'first_moment' in state, "Should track first moment"
            assert 'sec_moment' in state, "Should track second moment"
            assert 'avg_utility' in state, "Should track utility"
            has_moments = True

    assert has_moments, "At least some parameters should have moment statistics"
    print("  [OK] AdaptiveUPGD moments tracked correctly")
    print(f"  [OK] Trained for 10 steps without errors")


def test_03_variance_gradient_scaler():
    """Test Variance Gradient Scaler functionality."""
    import torch
    import torch.nn as nn
    from optimizers import UPGD
    from variance_gradient_scaler import VarianceGradientScaler

    model = nn.Linear(4, 2)
    optimizer = UPGD(model.parameters(), lr=1e-3)
    vgs = VarianceGradientScaler(
        model.parameters(),
        enabled=True,
        beta=0.99,
        alpha=0.1,
        warmup_steps=5
    )

    # Train with VGS
    scaling_factors = []
    for step in range(15):
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

    # Verify warmup behavior
    for i in range(5):
        assert scaling_factors[i] == 1.0, f"Warmup step {i} should not scale"

    # Verify VGS state
    assert vgs._step_count == 15, "Should track 15 steps"
    assert vgs._grad_mean_ema is not None, "Should track gradient mean"
    assert vgs._grad_var_ema is not None, "Should track gradient variance"

    print("  [OK] VGS warmup behavior correct (first 5 steps unscaled)")
    print("  [OK] VGS statistics tracked over 15 steps")
    print(f"  [OK] Final scaling factor: {scaling_factors[-1]:.4f}")


def test_04_upgd_numerical_stability():
    """Test UPGD numerical stability over extended training."""
    import torch
    import torch.nn as nn
    from optimizers import AdaptiveUPGD

    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4, sigma=0.01)

    # Extended training
    for step in range(100):
        x = torch.randn(32, 8)
        target = torch.randint(0, 2, (32,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Check for NaN/Inf
        for param in model.parameters():
            assert torch.all(torch.isfinite(param)), f"NaN/Inf in parameters at step {step}"

    # Check optimizer state
    for param in optimizer.state:
        state = optimizer.state[param]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"NaN/Inf in optimizer state '{key}'"

    print("  [OK] No NaN/Inf in parameters after 100 steps")
    print("  [OK] Optimizer state remained stable")


def test_05_pbt_scheduler():
    """Test Population-Based Training scheduler."""
    import torch
    from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig

    config = PBTConfig(
        population_size=4,
        perturbation_interval=10,
        hyperparams=[
            HyperparamConfig(
                name="lr",
                min_value=1e-5,
                max_value=1e-3,
                perturbation_factor=1.2
            ),
            HyperparamConfig(
                name="sigma",
                min_value=0.001,
                max_value=0.1,
                perturbation_factor=1.5
            ),
        ],
    )

    scheduler = PBTScheduler(config, seed=42)
    population = scheduler.initialize_population()

    assert len(population) == 4, "Should create 4 population members"

    # Verify hyperparameters
    for member in population:
        assert "lr" in member.hyperparams, "Should have lr"
        assert "sigma" in member.hyperparams, "Should have sigma"
        assert 1e-5 <= member.hyperparams["lr"] <= 1e-3, "lr in valid range"
        assert 0.001 <= member.hyperparams["sigma"] <= 0.1, "sigma in valid range"

    # Simulate performance updates
    for i, member in enumerate(population):
        performance = 10.0 + i * 2.0
        scheduler.update_performance(
            member,
            performance=performance,
            step=5,
            model_state_dict={"dummy": torch.randn(2, 2)}
        )

    # Test exploit/explore
    worst_member = population[0]
    worst_member.step = 10

    new_state, new_params, _ = scheduler.exploit_and_explore(
        worst_member,
        model_state_dict={"dummy": torch.randn(2, 2)}
    )

    assert "lr" in new_params, "Should return new hyperparameters"
    print(f"  [OK] PBT scheduler initialized with {len(population)} members")
    print(f"  [OK] Exploit/explore completed")
    print(f"  [OK] New lr: {new_params['lr']:.6f}, sigma: {new_params['sigma']:.6f}")


def test_06_upgd_with_ppo():
    """Test UPGD integration with DistributionalPPO."""
    import torch
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from distributional_ppo import DistributionalPPO
    from optimizers import AdaptiveUPGD

    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    # We must use "DistributionalPolicy" alias which maps to DistributionalActorCriticPolicy
    model = DistributionalPPO(
        "DistributionalPolicy",
        env,
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 3e-4, "sigma": 0.01},
        n_steps=64,
        n_epochs=2,
        batch_size=64,
        value_scale_max_rel_step=0.1,  # Required parameter
        verbose=0,
    )

    assert isinstance(model.policy.optimizer, AdaptiveUPGD), "Should use AdaptiveUPGD"

    # Train
    model.learn(total_timesteps=256)

    # Verify training worked
    assert len(model.policy.optimizer.state) > 0, "Optimizer should have state"

    # Check parameters are finite
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters should be finite"

    env.close()
    print("  [OK] UPGD integrated with PPO successfully")
    print("  [OK] Training completed without errors")


def test_07_twin_critics_with_upgd():
    """Test Twin Critics with UPGD optimizer."""
    import torch
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from distributional_ppo import DistributionalPPO

    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    # We must use "DistributionalPolicy" alias
    model = DistributionalPPO(
        "DistributionalPolicy",
        env,
        optimizer_class="adaptive_upgd",
        policy_kwargs={
            'arch_params': {
                'critic': {
                    'use_twin_critics': True
                }
            }
        },
        n_steps=64,
        n_epochs=2,
        value_scale_max_rel_step=0.1,  # Required parameter
        verbose=0,
    )

    # Verify Twin Critics is active
    assert getattr(model.policy, '_use_twin_critics', False) is True, "Twin critics should be enabled"
    assert (model.policy.dist_head_2 is not None) or (model.policy.quantile_head_2 is not None), "Second critic head should exist"

    # Train
    model.learn(total_timesteps=256)

    # Check stability
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters should be finite"

    env.close()
    print("  [OK] Twin Critics initialized correctly")
    print("  [OK] Training with adversarial critics successful")


def test_08_full_integration():
    """Test full integration: UPGD + Twin Critics + VGS."""
    import torch
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from distributional_ppo import DistributionalPPO

    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    # We must use "DistributionalPolicy" alias
    model = DistributionalPPO(
        "DistributionalPolicy",
        env,
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 3e-4, "sigma": 0.01, "beta_utility": 0.999},
        policy_kwargs={
            'arch_params': {
                'critic': {
                    'use_twin_critics': True
                }
            }
        },
        variance_gradient_scaling=True,
        vgs_alpha=0.15,
        vgs_warmup_steps=30,
        n_steps=128,
        n_epochs=2,
        batch_size=64,
        max_grad_norm=0.5,
        value_scale_max_rel_step=0.1,  # Required parameter
        verbose=0,
    )

    # Train for reasonable duration
    model.learn(total_timesteps=512)

    # Comprehensive checks
    param_count = 0
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters contain NaN/Inf"
        param_count += 1

    # Check optimizer state
    optimizer = model.policy.optimizer
    state_count = 0
    for p in optimizer.state:
        state = optimizer.state[p]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"Optimizer state '{key}' contains NaN/Inf"
        state_count += 1

    env.close()
    print(f"  [OK] Full integration successful")
    print(f"  [OK] Checked {param_count} parameters - all finite")
    print(f"  [OK] Checked {state_count} optimizer states - all finite")
    print(f"  [OK] UPGD + Twin Critics + VGS working together")


# ===========================================================================
# Main Test Runner
# ===========================================================================

def main():
    """Run all tests and report results."""
    print("="*70)
    print("UPGD + PBT + Twin Critics + VGS")
    print("Standalone Integration Test Suite")
    print("="*70)
    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Define all tests
    tests = [
        ("Basic UPGD Functionality", test_01_upgd_basic),
        ("AdaptiveUPGD with Moments", test_02_adaptive_upgd),
        ("Variance Gradient Scaler", test_03_variance_gradient_scaler),
        ("UPGD Numerical Stability", test_04_upgd_numerical_stability),
        ("PBT Scheduler", test_05_pbt_scheduler),
        ("UPGD with PPO", test_06_upgd_with_ppo),
        ("Twin Critics with UPGD", test_07_twin_critics_with_upgd),
        ("Full Integration", test_08_full_integration),
    ]

    # Run all tests
    results = []
    for test_name, test_func in tests:
        success = run_test(test_name, test_func)
        results.append((test_name, success))

    # Print summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    failed = sum(1 for _, success in results if not success)
    total = len(results)

    for test_name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*70)

    if failed == 0:
        print()
        print("ALL TESTS PASSED!")
        print()
        print("Summary of validated features:")
        print("  - UPGD optimizer (utility-based weight protection)")
        print("  - AdaptiveUPGD (UPGD + Adam moments)")
        print("  - Variance Gradient Scaling (adaptive gradient scaling)")
        print("  - Population-Based Training (hyperparameter optimization)")
        print("  - Twin Critics (adversarial value functions)")
        print("  - Full integration (all components working together)")
        print("  - Numerical stability over extended training")
        print()
        return 0
    else:
        print()
        print(f"{failed} TEST(S) FAILED")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
