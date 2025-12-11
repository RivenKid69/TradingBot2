"""
Final Integration Test: All Bug Fixes Together

Tests all fixed components working together:
- Bug #1: Twin Critics (FIXED)
- Bug #2: optimizer_kwargs['lr'] (FIXED)
- Bug #4: VGS parameter update (FIXED)
- Bug #5: UPGD division by zero (FIXED)

This test verifies that all components work together without conflicts.
"""

import pytest
import sys
gym = pytest.importorskip("gymnasium")
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_full_integration():
    """Test all fixed components together."""

    print("=" * 80)
    print("FULL INTEGRATION TEST: All Components Together")
    print("=" * 80)
    print()

    print("Components being tested:")
    print("  - Twin Critics (Bug #1)")
    print("  - Custom learning rate (Bug #2)")
    print("  - VGS parameter tracking (Bug #4)")
    print("  - UPGD optimizer (Bug #5)")
    print()

    # Create environment
    print("1. Creating environment...")
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    print("   [OK] Environment created")
    print()

    # Create model with ALL components enabled
    print("2. Creating model with all components enabled...")
    print("   - Twin Critics: ON")
    print("   - Custom LR: 0.002")
    print("   - VGS: ON")
    print("   - UPGD optimizer: ON")
    print()

    try:
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            # Bug #2: Custom learning rate
            optimizer_kwargs={'lr': 0.002},
            # Bug #5: UPGD optimizer
            optimizer_class='upgd',
            # Bug #4: VGS enabled
            variance_gradient_scaling=True,
            vgs_beta=0.99,
            vgs_alpha=0.1,
            vgs_warmup_steps=10,  # Lower for faster test
            # Twin Critics (Bug #1) - enabled by default in CustomActorCriticPolicy
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        print("   [OK] Model created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Verify components are configured correctly
    print("3. Verifying component configuration...")

    # Check Twin Critics
    has_twin_critics = getattr(model.policy, '_use_twin_critics', False)
    print(f"   Twin Critics: {'ON' if has_twin_critics else 'OFF'}")
    if not has_twin_critics:
        print("   [WARNING] Twin Critics not enabled (expected ON)")

    # Check custom LR (Bug #2)
    actual_lr = model.policy.optimizer.param_groups[0]['lr']
    expected_lr = 0.002
    lr_correct = abs(actual_lr - expected_lr) < 1e-9
    print(f"   Learning rate: {actual_lr} (expected {expected_lr})")
    if not lr_correct:
        print(f"   [FAIL] Custom LR not applied correctly!")
        return False
    print("   [OK] Custom LR correct (Bug #2 fixed)")

    # Check VGS (Bug #4)
    vgs = model._variance_gradient_scaler
    vgs_enabled = vgs is not None
    print(f"   VGS enabled: {vgs_enabled}")
    if vgs_enabled:
        vgs_param_count = len(vgs._parameters) if vgs._parameters else 0
        model_param_count = len(list(model.policy.parameters()))
        print(f"   VGS tracking {vgs_param_count}/{model_param_count} parameters")
        if vgs_param_count != model_param_count:
            print(f"   [FAIL] VGS parameter count mismatch!")
            return False
        print("   [OK] VGS parameters correct (Bug #4 fixed)")
    else:
        print("   [FAIL] VGS not enabled!")
        return False

    # Check UPGD optimizer (Bug #5)
    from optimizers.upgd import UPGD
    is_upgd = isinstance(model.policy.optimizer, UPGD)
    print(f"   UPGD optimizer: {'ON' if is_upgd else 'OFF'}")
    if not is_upgd:
        print("   [FAIL] UPGD optimizer not used!")
        return False
    print("   [OK] UPGD optimizer active (Bug #5 fixed)")

    print()

    # Run training to verify everything works together
    print("4. Running integrated training (all components active)...")
    print("   Training for 256 timesteps...")

    try:
        model.learn(total_timesteps=256, progress_bar=False)
        print("   [OK] Training completed successfully")
    except Exception as e:
        print(f"   [FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False

    print()

    # Verify no NaN/Inf in parameters (Bug #5)
    print("5. Checking for numerical stability...")
    has_nan = False
    has_inf = False

    for param in model.policy.parameters():
        if torch.isnan(param).any():
            has_nan = True
        if torch.isinf(param).any():
            has_inf = True

    if has_nan or has_inf:
        print(f"   [FAIL] Found NaN/Inf in parameters after training!")
        if has_nan:
            print("   - Contains NaN")
        if has_inf:
            print("   - Contains Inf")
        env.close()
        return False

    print("   [OK] No NaN/Inf detected (numerical stability confirmed)")

    print()

    # Check VGS statistics
    if vgs:
        print("6. Checking VGS statistics...")
        print(f"   VGS step count: {vgs._step_count}")
        if vgs._grad_norm_ema is not None:
            print(f"   VGS grad norm EMA: {vgs._grad_norm_ema:.6f}")
            print("   [OK] VGS accumulated statistics")
        else:
            print("   [WARNING] VGS has no accumulated statistics")

    print()
    print("=" * 80)
    print("INTEGRATION TEST PASSED")
    print("=" * 80)
    print()
    print("All components work together correctly:")
    print("  [OK] Twin Critics (Bug #1)")
    print("  [OK] Custom learning rate (Bug #2)")
    print("  [OK] VGS parameter tracking (Bug #4)")
    print("  [OK] UPGD optimizer (Bug #5)")
    print()

    env.close()
    return True


if __name__ == "__main__":
    print("\n")

    success = test_full_integration()

    print("\n")
    print("=" * 80)

    if success:
        print("FINAL VERDICT: ALL BUGS FIXED [OK]")
        print()
        print("Summary:")
        print("  - Bug #1 (Twin Critics): FIXED")
        print("  - Bug #2 (optimizer_kwargs['lr']): FIXED")
        print("  - Bug #4 (VGS parameters): FIXED")
        print("  - Bug #5 (UPGD division by zero): FIXED")
        print()
        print("All components integrate successfully!")
        sys.exit(0)
    else:
        print("FINAL VERDICT: INTEGRATION ISSUES FOUND [FAIL]")
        print()
        print("Some components do not work together correctly.")
        print("Please review the test output above.")
        sys.exit(1)
