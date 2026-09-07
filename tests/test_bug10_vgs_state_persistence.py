"""
Specialized Test for Bug #10: VGS State Not Preserved Across Save/Load

This test precisely localizes the issue where VGS internal state
(step_count, EMAs) is not correctly preserved during model save/load cycles.
"""

import pytest
import tempfile
from pathlib import Path

gym = pytest.importorskip("gymnasium")
import numpy as np

torch = pytest.importorskip("torch")
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def make_test_env():
    """Create simple test environment."""
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


def test_vgs_state_preservation():
    """
    Test that VGS state (step_count, EMAs) is preserved across save/load.

    This is Bug #10: VGS step_count resets to 0 after load instead of
    preserving the trained value.
    """
    print("=" * 80)
    print("BUG #10: VGS STATE PERSISTENCE TEST")
    print("=" * 80)
    print()

    # Create environment and model with VGS
    print("1. Creating model with VGS enabled...")
    env = make_test_env()

    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=1.0,
        vgs_warmup_steps=5,
        n_steps=128,
        batch_size=64,
        verbose=0,
    )
    print(f"   Model created")
    print()

    # Train to accumulate VGS state
    print("2. Training model to accumulate VGS state...")
    initial_step_count = model._variance_gradient_scaler._step_count
    print(f"   Initial VGS step_count: {initial_step_count}")

    model.learn(total_timesteps=2048)

    after_train_step_count = model._variance_gradient_scaler._step_count
    print(f"   After training VGS step_count: {after_train_step_count}")
    print(f"   VGS stepped: {after_train_step_count > initial_step_count}")
    print()

    # Capture full VGS state before save
    print("3. Capturing VGS state before save...")
    vgs_before = model._variance_gradient_scaler
    state_before = {
        "step_count": vgs_before._step_count,
        "grad_mean_ema": vgs_before._grad_mean_ema,
        "grad_var_ema": vgs_before._grad_var_ema,
        "grad_norm_ema": vgs_before._grad_norm_ema,
    }

    print(f"   step_count: {state_before['step_count']}")
    print(f"   grad_mean_ema: {state_before['grad_mean_ema']}")
    print(f"   grad_var_ema: {state_before['grad_var_ema']}")
    print(f"   grad_norm_ema: {state_before['grad_norm_ema']}")
    print()

    # Save model
    print("4. Saving model...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "vgs_state_test.zip"
        model.save(save_path)
        print(f"   Model saved to {save_path}")
        print()

        # Check what's actually saved
        print("5. Inspecting saved state_dict...")
        import zipfile

        with zipfile.ZipFile(save_path, "r") as archive:
            files_in_archive = archive.namelist()
            print(f"   Files in archive: {files_in_archive}")

            # Try to load pytorch variables
            if "pytorch_variables.pth" in files_in_archive:
                with archive.open("pytorch_variables.pth") as f:
                    import io

                    # Security Note: weights_only=False is acceptable here because:
                    # 1. This is a test file with controlled checkpoint creation
                    # 2. The checkpoint is created within this test (not from external source)
                    # 3. We need access to full object structure for debugging
                    # WARNING: In production code, always use weights_only=True for untrusted checkpoints
                    pytorch_vars = torch.load(io.BytesIO(f.read()), weights_only=False)
                    print(f"   Keys in pytorch_variables.pth:")
                    for key in pytorch_vars.keys():
                        print(f"     - {key}")
                        if "vgs" in key.lower() or "variance" in key.lower():
                            print(f"       VGS-related key found: {key}")
                            print(f"       Value: {pytorch_vars[key]}")
        print()

        # Load model
        print("6. Loading model...")
        loaded_model = DistributionalPPO.load(save_path, env=env)
        print("   Model loaded")
        print()

        # Capture VGS state after load
        print("7. Capturing VGS state after load...")
        vgs_after = loaded_model._variance_gradient_scaler
        state_after = {
            "step_count": vgs_after._step_count,
            "grad_mean_ema": vgs_after._grad_mean_ema,
            "grad_var_ema": vgs_after._grad_var_ema,
            "grad_norm_ema": vgs_after._grad_norm_ema,
        }

        print(f"   step_count: {state_after['step_count']}")
        print(f"   grad_mean_ema: {state_after['grad_mean_ema']}")
        print(f"   grad_var_ema: {state_after['grad_var_ema']}")
        print(f"   grad_norm_ema: {state_after['grad_norm_ema']}")
        print()

        # Compare states
        print("8. Comparing states...")
        print("-" * 80)

        step_count_matches = state_after["step_count"] == state_before["step_count"]
        print(f"   Step count preserved: {step_count_matches}")
        print(f"     Before: {state_before['step_count']}")
        print(f"     After:  {state_after['step_count']}")

        if state_before["grad_mean_ema"] is not None:
            if state_after["grad_mean_ema"] is not None:
                mean_ema_matches = (
                    abs(state_after["grad_mean_ema"] - state_before["grad_mean_ema"]) < 1e-6
                )
                print(f"   Mean EMA preserved: {mean_ema_matches}")
                print(f"     Before: {state_before['grad_mean_ema']}")
                print(f"     After:  {state_after['grad_mean_ema']}")
            else:
                mean_ema_matches = False
                print(f"   Mean EMA preserved: False (reset to None)")
                print(f"     Before: {state_before['grad_mean_ema']}")
                print(f"     After:  None")
        else:
            mean_ema_matches = True
            print(f"   Mean EMA: N/A (not initialized)")

        if state_before["grad_var_ema"] is not None:
            if state_after["grad_var_ema"] is not None:
                var_ema_matches = (
                    abs(state_after["grad_var_ema"] - state_before["grad_var_ema"]) < 1e-6
                )
                print(f"   Var EMA preserved: {var_ema_matches}")
                print(f"     Before: {state_before['grad_var_ema']}")
                print(f"     After:  {state_after['grad_var_ema']}")
            else:
                var_ema_matches = False
                print(f"   Var EMA preserved: False (reset to None)")
                print(f"     Before: {state_before['grad_var_ema']}")
                print(f"     After:  None")
        else:
            var_ema_matches = True
            print(f"   Var EMA: N/A (not initialized)")

        if state_before["grad_norm_ema"] is not None:
            if state_after["grad_norm_ema"] is not None:
                norm_ema_matches = (
                    abs(state_after["grad_norm_ema"] - state_before["grad_norm_ema"]) < 1e-6
                )
                print(f"   Norm EMA preserved: {norm_ema_matches}")
                print(f"     Before: {state_before['grad_norm_ema']}")
                print(f"     After:  {state_after['grad_norm_ema']}")
            else:
                norm_ema_matches = False
                print(f"   Norm EMA preserved: False (reset to None)")
                print(f"     Before: {state_before['grad_norm_ema']}")
                print(f"     After:  None")
        else:
            norm_ema_matches = True
            print(f"   Norm EMA: N/A (not initialized)")

        print()
        print("=" * 80)
        print("TEST RESULT")
        print("=" * 80)

        all_match = step_count_matches and mean_ema_matches and var_ema_matches and norm_ema_matches

        if all_match:
            print("[PASS] All VGS state correctly preserved")
            return True
        else:
            print("[FAIL] VGS state NOT preserved")
            print()
            print("BUGS FOUND:")
            if not step_count_matches:
                print(f"  - Step count reset to 0 (expected {state_before['step_count']})")
            if not mean_ema_matches:
                print(f"  - Mean EMA not preserved")
            if not var_ema_matches:
                print(f"  - Var EMA not preserved")
            if not norm_ema_matches:
                print(f"  - Norm EMA not preserved")
            return False

    env.close()


if __name__ == "__main__":
    success = test_vgs_state_preservation()
    exit(0 if success else 1)
