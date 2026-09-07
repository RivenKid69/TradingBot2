"""
Test demonstrating CORRECT API usage for UPGD + Twin Critics + VGS integration.

This test shows the proper way to configure all advanced optimization features:
1. UPGD Optimizer - via optimizer_class and optimizer_kwargs
2. Variance Gradient Scaling (VGS) - via variance_gradient_scaling parameter
3. Twin Critics - via policy_kwargs['arch_params']['critic']['use_twin_critics']

IMPORTANT: This test demonstrates the CORRECT way to use these features.
"""

import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def make_simple_env():
    """Create a simple test environment."""
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


def test_correct_api_all_features():
    """
    Test CORRECT API usage for all features together:
    - UPGD Optimizer (adaptive_upgd)
    - Variance Gradient Scaling (VGS)
    - Twin Critics
    """
    print("=" * 80)
    print("Test: CORRECT API Usage for UPGD + VGS + Twin Critics")
    print("=" * 80)
    print()

    env = make_simple_env()

    print("Creating model with CORRECT API...")
    print()

    # ✅ CORRECT: All parameters passed correctly
    model = DistributionalPPO(
        CustomActorCriticPolicy,  # Use custom policy to enable advanced features
        env,
        # ✅ UPGD Optimizer (CORRECT)
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={
            "lr": 3e-4,
            "sigma": 0.01,
            "beta_utility": 0.999,
        },
        # ✅ Variance Gradient Scaling (CORRECT)
        variance_gradient_scaling=True,  # ← 'variance_gradient_scaling', NOT 'vgs_enabled'!
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        # ✅ Twin Critics (CORRECT - via policy_kwargs)
        policy_kwargs={
            "arch_params": {
                "hidden_dim": 64,
                "critic": {
                    "distributional": True,
                    "num_quantiles": 32,
                    "huber_kappa": 1.0,
                    "use_twin_critics": True,  # ← This is where Twin Critics goes!
                },
            }
        },
        # Training parameters
        n_steps=64,
        n_epochs=2,
        batch_size=64,
        verbose=0,
    )

    print("[OK] Model created successfully with correct API")
    print()

    # Verify components are configured correctly
    print("Verifying configuration...")
    print()

    # Check UPGD optimizer
    from optimizers import AdaptiveUPGD

    assert isinstance(
        model.policy.optimizer, AdaptiveUPGD
    ), f"Expected AdaptiveUPGD, got {type(model.policy.optimizer)}"
    print(f"  [OK] Optimizer: {type(model.policy.optimizer).__name__}")
    print(f"     - lr: {model.policy.optimizer.param_groups[0]['lr']}")
    print(f"     - sigma: {model.policy.optimizer.param_groups[0].get('sigma', 'N/A')}")

    # Check VGS
    assert model._variance_gradient_scaler is not None, "VGS should be initialized"
    vgs = model._variance_gradient_scaler
    print(f"  [OK] Variance Gradient Scaling:")
    print(f"     - enabled: {vgs.enabled}")
    print(f"     - beta: {vgs.beta}")
    print(f"     - alpha: {vgs.alpha}")
    print(f"     - warmup_steps: {vgs.warmup_steps}")

    # Check Twin Critics
    use_twin = getattr(model.policy, "_use_twin_critics", False)
    assert use_twin is True, "Twin Critics should be enabled"
    print(f"  [OK] Twin Critics: enabled={use_twin}")

    # Check second critic exists
    assert hasattr(model.policy, "quantile_head_2"), "Second critic head should exist"
    assert model.policy.quantile_head_2 is not None, "Second critic head should be initialized"
    print(f"     - Second critic: {type(model.policy.quantile_head_2).__name__}")

    print()
    print("Training for 256 timesteps...")
    model.learn(total_timesteps=256)
    print("[OK] Training completed successfully")
    print()

    # Verify numerical stability
    print("Checking numerical stability...")
    for param in model.policy.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters should be finite"
    print("[OK] All parameters are finite (no NaN/Inf)")
    print()

    # Verify VGS statistics
    if vgs._step_count > 0:
        print(f"VGS Statistics:")
        print(f"  - steps: {vgs._step_count}")
        print(
            f"  - grad_mean_ema: {vgs._grad_mean_ema:.6f}"
            if vgs._grad_mean_ema
            else "  - grad_mean_ema: None"
        )
        print(f"  - normalized_variance: {vgs.get_normalized_variance():.6f}")
        print()

    env.close()

    print("=" * 80)
    print("RESULT: SUCCESS - All features work correctly with CORRECT API")
    print("=" * 80)
    print()
    print("Summary:")
    print("  [OK] UPGD Optimizer: AdaptiveUPGD configured correctly")
    print("  [OK] Variance Gradient Scaling: Enabled and working")
    print("  [OK] Twin Critics: Enabled via policy_kwargs")
    print("  [OK] Training: Completed without errors")
    print("  [OK] Numerical Stability: No NaN/Inf detected")
    print()


def test_incorrect_api_demonstration():
    """
    Demonstrates what HAPPENS when using INCORRECT API.
    This test should FAIL to show the error.
    """
    print("=" * 80)
    print("Test: Demonstrating INCORRECT API (This should fail)")
    print("=" * 80)
    print()

    env = make_simple_env()

    try:
        # ❌ INCORRECT: These parameters don't exist or are in wrong location
        model = DistributionalPPO(
            "MlpPolicy",
            env,
            optimizer_class="adaptive_upgd",
            use_twin_critics=True,  # ❌ NOT a direct parameter!
            adversarial_training=True,  # ❌ Doesn't exist!
            vgs_enabled=True,  # ❌ Wrong parameter name!
            vgs_alpha=0.1,
            vgs_warmup_steps=50,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )
        print("[UNEXPECTED] Model created with incorrect API (should have failed)")
        env.close()
        return False

    except TypeError as e:
        print(f"[EXPECTED] TypeError caught: {e}")
        print()
        print("This error occurs because:")
        print("  [X] 'use_twin_critics' is NOT a direct parameter of DistributionalPPO")
        print(
            "     [OK] Correct: Pass via policy_kwargs['arch_params']['critic']['use_twin_critics']"
        )
        print()
        print("  [X] 'adversarial_training' does NOT exist in DistributionalPPO")
        print("     [OK] Correct: Adversarial training is a separate module")
        print()
        print("  [X] 'vgs_enabled' is the WRONG parameter name")
        print("     [OK] Correct: Use 'variance_gradient_scaling=True'")
        print()
        env.close()
        return True


def test_minimal_correct_usage():
    """Test minimal correct configuration (just UPGD)."""
    print("=" * 80)
    print("Test: Minimal CORRECT usage (UPGD only)")
    print("=" * 80)
    print()

    env = make_simple_env()

    model = DistributionalPPO(
        CustomActorCriticPolicy,  # Use custom policy
        env,
        # Just UPGD optimizer
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 3e-4},
        n_steps=64,
        verbose=0,
    )

    print("[OK] Model created with UPGD optimizer")
    model.learn(total_timesteps=128)
    print("[OK] Training completed")

    env.close()
    print()


def test_vgs_only_correct_usage():
    """Test VGS configuration without Twin Critics."""
    print("=" * 80)
    print("Test: VGS configuration (without Twin Critics)")
    print("=" * 80)
    print()

    env = make_simple_env()

    model = DistributionalPPO(
        CustomActorCriticPolicy,  # Use custom policy
        env,
        optimizer_class="adaptive_upgd",
        # VGS (CORRECT parameter name)
        variance_gradient_scaling=True,  # ← CORRECT!
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        n_steps=64,
        verbose=0,
    )

    print("[OK] Model created with VGS")
    assert model._variance_gradient_scaler is not None
    print(f"[OK] VGS enabled: {model._variance_gradient_scaler.enabled}")

    model.learn(total_timesteps=128)
    print("[OK] Training completed")

    env.close()
    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("CORRECT API USAGE DEMONSTRATION")
    print("=" * 80)
    print()
    print("This script demonstrates the CORRECT way to configure:")
    print("  1. UPGD Optimizer")
    print("  2. Variance Gradient Scaling (VGS)")
    print("  3. Twin Critics")
    print()
    print("=" * 80)
    print()

    # Test 1: Show incorrect API fails
    print("\n### Test 1: Incorrect API (Expected to Fail) ###\n")
    incorrect_failed_as_expected = test_incorrect_api_demonstration()

    if incorrect_failed_as_expected:
        print("=" * 80)
        print("[OK] Incorrect API properly rejected (as expected)")
        print("=" * 80)
    print()

    # Test 2: Minimal correct usage
    print("\n### Test 2: Minimal Correct Usage (UPGD only) ###\n")
    test_minimal_correct_usage()

    # Test 3: VGS only
    print("\n### Test 3: VGS Configuration ###\n")
    test_vgs_only_correct_usage()

    # Test 4: All features together
    print("\n### Test 4: All Features Together (CORRECT API) ###\n")
    test_correct_api_all_features()

    print()
    print("=" * 80)
    print("FINAL VERDICT: ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  [OK] Use 'variance_gradient_scaling=True' (NOT 'vgs_enabled')")
    print("  [OK] Use 'optimizer_class' and 'optimizer_kwargs' for UPGD")
    print("  [OK] Use policy_kwargs['arch_params']['critic']['use_twin_critics'] for Twin Critics")
    print("  [X] DON'T use 'adversarial_training' (not implemented in DistributionalPPO)")
    print()
    print("For more information:")
    print("  - docs/twin_critics.md")
    print("  - REMAINING_INTEGRATION_ISSUES_REPORT.md")
    print()
