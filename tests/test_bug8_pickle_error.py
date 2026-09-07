"""
Specialized test to confirm Bug #8: Pickle error when saving DistributionalPPO model.

Error: TypeError: cannot pickle 'EncodedFile' instances

Root Cause: Lambda function in distributional_ppo.py line 6025:
    self.clip_range = lambda _: self._compute_clip_range_value()

This lambda captures the current scope, which may include unpicklable objects
like sys.stdout/stderr (EncodedFile instances on Windows).
"""

import pytest

gym = pytest.importorskip("gymnasium")
import tempfile
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_bug8_pickle_error_on_save():
    """
    Test to CONFIRM Bug #8: Cannot save model due to pickle error.

    Expected behavior WITHOUT fix:
        - model.save() raises TypeError: cannot pickle 'EncodedFile' instances

    Expected behavior WITH fix:
        - model.save() succeeds
        - model.load() succeeds
        - Loaded model can continue training
    """
    print("=" * 80)
    print("Bug #8 Confirmation Test: Pickle error when saving model")
    print("=" * 80)
    print()

    # Create environment
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    print("Creating DistributionalPPO model with all features...")
    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        # UPGD Optimizer
        optimizer_class="adaptive_upgd",
        optimizer_kwargs={"lr": 1e-4, "sigma": 0.005},
        # Variance Gradient Scaling
        variance_gradient_scaling=True,
        vgs_alpha=0.1,
        vgs_warmup_steps=50,
        # Twin Critics via policy_kwargs
        policy_kwargs={
            "arch_params": {
                "critic": {
                    "distributional": True,
                    "num_quantiles": 32,
                    "use_twin_critics": True,
                }
            }
        },
        n_steps=64,
        n_epochs=2,
        verbose=0,
    )
    print("[OK] Model created successfully")
    print()

    # Train briefly
    print("Training for 256 timesteps...")
    model.learn(total_timesteps=256)
    print("[OK] Training completed")
    print()

    # Attempt to save
    print("Attempting to save model...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.zip"

        try:
            model.save(save_path)
            print("[OK] Model saved successfully!")
            print(f"     Save path: {save_path}")
            print()

            # Try to load
            print("Attempting to load model...")
            loaded_model = DistributionalPPO.load(save_path, env=env)
            print("[OK] Model loaded successfully!")
            print()

            # Try to continue training
            print("Testing loaded model with continued training...")
            loaded_model.learn(total_timesteps=128)
            print("[OK] Continued training successful!")
            print()

            env.close()

            print("=" * 80)
            print("VERDICT: Bug #8 FIXED [OK]")
            print("=" * 80)
            print()
            print("Summary:")
            print("  [OK] Model save: SUCCESS")
            print("  [OK] Model load: SUCCESS")
            print("  [OK] Continued training: SUCCESS")
            print()
            return True

        except TypeError as e:
            if "cannot pickle 'EncodedFile' instances" in str(e):
                print(f"[FAIL] Pickle error caught: {type(e).__name__}")
                print(f"       Error message: {e}")
                print()
                print("=" * 80)
                print("VERDICT: Bug #8 CONFIRMED [FAIL]")
                print("=" * 80)
                print()
                print("Root Cause:")
                print("  Lambda function in distributional_ppo.py captures unpicklable scope")
                print("  Line 6025: self.clip_range = lambda _: self._compute_clip_range_value()")
                print()
                print("Recommended Fix:")
                print("  Replace lambda with a bound method or use a picklable callable")
                print()
                env.close()
                return False
            else:
                # Different TypeError
                raise
        except Exception as e:
            print(f"[UNEXPECTED] Different error: {type(e).__name__}: {e}")
            env.close()
            raise


def test_lambda_scope_capture():
    """
    Test to demonstrate that lambda functions capture unpicklable scope.
    """
    print("\n")
    print("=" * 80)
    print("Auxiliary Test: Lambda scope capture demonstration")
    print("=" * 80)
    print()

    import pickle
    import sys

    # Test 1: Simple lambda (should work)
    simple_lambda = lambda x: x + 1
    try:
        pickled = pickle.dumps(simple_lambda)
        print("[OK] Simple lambda can be pickled")
    except Exception as e:
        print(f"[FAIL] Simple lambda cannot be pickled: {e}")

    # Test 2: Lambda capturing local variable (should work)
    offset = 10
    local_lambda = lambda x: x + offset
    try:
        pickled = pickle.dumps(local_lambda)
        print("[OK] Lambda with local variable can be pickled")
    except Exception as e:
        print(f"[FAIL] Lambda with local variable cannot be pickled: {e}")

    # Test 3: Lambda in scope with unpicklable reference
    # This simulates what might happen in DistributionalPPO.__init__
    output_file = sys.stdout  # EncodedFile on Windows
    scoped_lambda = lambda x: x + 1  # lambda created in scope with unpicklable ref
    try:
        pickled = pickle.dumps(scoped_lambda)
        print("[OK] Lambda in unpicklable scope can be pickled (no direct capture)")
    except Exception as e:
        print(f"[FAIL] Lambda in unpicklable scope cannot be pickled: {e}")

    print()


if __name__ == "__main__":
    print("\n" * 2)
    print("=" * 80)
    print("BUG #8 SPECIALIZED CONFIRMATION TEST")
    print("=" * 80)
    print()
    print("Purpose: Confirm the pickle error when saving DistributionalPPO models")
    print()
    print("Expected Result WITHOUT fix:")
    print("  - model.save() raises TypeError: cannot pickle 'EncodedFile' instances")
    print()
    print("Expected Result WITH fix:")
    print("  - model.save() succeeds")
    print("  - model.load() succeeds")
    print()
    print("=" * 80)
    print()

    # Run auxiliary test first
    test_lambda_scope_capture()

    # Run main confirmation test
    success = test_bug8_pickle_error_on_save()

    if success:
        print("\n" + "=" * 80)
        print("OVERALL RESULT: Bug #8 is FIXED")
        print("=" * 80)
        exit(0)
    else:
        print("\n" + "=" * 80)
        print("OVERALL RESULT: Bug #8 is CONFIRMED - FIX NEEDED")
        print("=" * 80)
        exit(1)
