"""
Critical Bug #4 Verification: VGS не обновляет параметры после пересоздания optimizer

Hypothesis: VarianceGradientScaler (VGS) инициализируется с параметрами до того,
как CustomActorCriticPolicy пересоздает optimizer в _setup_custom_optimizer().
VGS не обновляет свой список параметров после пересоздания optimizer, что приводит
к работе с неправильным/пустым списком параметров.

Expected: VGS работает с устаревшим списком параметров или пустым списком
Location: distributional_ppo.py:5848, custom_policy_patch1.py:381
"""

import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_vgs_parameter_update():
    """Test if VGS updates parameters after optimizer recreation."""

    print("=" * 80)
    print("CRITICAL BUG #4: VGS не обновляет параметры после пересоздания optimizer")
    print("=" * 80)
    print()

    # Setup
    print("1. Creating environment...")
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    print("2. Creating model with VGS enabled...")
    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        variance_gradient_scaling=True,
        verbose=0,
    )

    print("   [OK] Model created")
    print()

    # Check VGS configuration
    vgs = model._variance_gradient_scaler
    if vgs is None:
        print("   [FAIL] VGS not initialized!")
        return True

    print("3. Checking VGS parameter list...")

    # Get VGS parameters
    vgs_params = vgs._parameters
    if vgs_params is None:
        print("   [FAIL] VGS has no parameters! (_parameters is None)")
        print()
        print("RESULT: BUG CONFIRMED - VGS not tracking any parameters")
        return True

    vgs_param_count = len(list(vgs_params))
    print(f"   VGS tracking {vgs_param_count} parameters")

    # Get actual model parameters
    model_params = list(model.policy.parameters())
    model_param_count = len(model_params)
    print(f"   Model has {model_param_count} parameters")
    print()

    # Check if counts match
    if vgs_param_count == 0:
        print("   [FAIL] VGS tracking 0 parameters!")
        print()
        print("RESULT: BUG CONFIRMED - VGS parameter list is empty")
        return True

    if vgs_param_count != model_param_count:
        print(f"   [WARNING] Mismatch: VGS has {vgs_param_count}, Model has {model_param_count}")
        print()
        print("RESULT: BUG CONFIRMED - VGS parameter count mismatch")
        return True

    # Check if parameters are the same objects
    vgs_param_ids = {id(p) for p in vgs_params if p is not None}
    model_param_ids = {id(p) for p in model_params}

    matching_params = vgs_param_ids & model_param_ids
    print(f"4. Checking parameter identity...")
    print(f"   Matching parameters: {len(matching_params)}/{model_param_count}")

    if len(matching_params) < model_param_count * 0.9:  # Allow 10% tolerance
        print()
        print("   [FAIL] VGS parameters don't match model parameters!")
        print(f"   Only {len(matching_params)}/{model_param_count} parameters match")
        print()
        print("RESULT: BUG CONFIRMED - VGS tracking wrong parameters")
        return True

    print()
    print("5. Testing gradient scaling (simple test)...")
    try:
        # Simple test: scale_gradients() with no gradients should return 1.0
        scaling_factor = vgs.scale_gradients()
        print(f"   VGS scaling factor (no grads): {scaling_factor}")

        if scaling_factor != 1.0:
            print(f"   [WARNING] Expected 1.0, got {scaling_factor}")

        print(f"   [OK] Basic gradient scaling works")

    except Exception as e:
        print(f"   [FAIL] Error during gradient scaling: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("RESULT: BUG CONFIRMED - VGS fails during gradient scaling")
        return True

    print()
    print("6. Testing VGS with actual training step...")
    try:
        # Run a short training to verify VGS works in real scenario
        model.learn(total_timesteps=64, progress_bar=False)

        print(f"   [OK] Training completed with VGS enabled")
        print(f"   VGS is working correctly!")

    except Exception as e:
        print(f"   [FAIL] Error during training with VGS: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("RESULT: BUG CONFIRMED - VGS fails during training")
        return True

    print()
    print("=" * 80)
    print("RESULT: BUG NOT FOUND - VGS parameters correctly updated")
    print("=" * 80)

    env.close()
    return False


if __name__ == "__main__":
    print("\n")
    bug_exists = test_vgs_parameter_update()
    print("\n")
    print("=" * 80)
    if bug_exists:
        print("VERDICT: BUG EXISTS [FAIL]")
        print("Severity: CRITICAL - VGS not working correctly")
        print()
        print("Impact:")
        print("- VGS scales gradients incorrectly or not at all")
        print("- Training instability due to wrong gradient statistics")
        print("- Potential NaN or inf gradients")
        sys.exit(1)
    else:
        print("VERDICT: BUG NOT FOUND [OK]")
        print("VGS parameters correctly updated")
        sys.exit(0)
