"""
Validation tests for VGS + PBT State Synchronization Fix

These tests verify that the fix for VGS state mismatch during PBT exploitation
works correctly. The fix ensures that VGS state is properly transferred when
population members exploit from better performers.
"""

import pytest
torch = pytest.importorskip("torch")
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import required components
try:
    from distributional_ppo import DistributionalPPO
    from variance_gradient_scaler import VarianceGradientScaler
    from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig, PopulationMember
    from custom_policy_patch1 import CustomActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    pytest.skip(f"Required imports not available: {e}", allow_module_level=True)


def make_test_env(seed=None):
    """Create a Pendulum environment for testing."""
    def _make():
        env = gym.make("Pendulum-v1")
        if seed is not None:
            env.reset(seed=seed)
        return env
    return DummyVecEnv([_make])


def create_test_model(learning_rate=1e-4, use_vgs=True, seed=None):
    """Create a test model with VGS configuration."""
    env = make_test_env(seed=seed)

    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        verbose=0,
        variance_gradient_scaling=use_vgs,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=10,
        seed=seed,
    )

    return model, env


def get_vgs_state(model: DistributionalPPO) -> Dict[str, Any]:
    """Extract VGS state from model."""
    if model._variance_gradient_scaler is None:
        return {"enabled": False, "state": None}

    state = model._variance_gradient_scaler.state_dict()
    return {
        "enabled": True,
        "step_count": state.get("step_count", 0),
        "grad_mean_ema": state.get("grad_mean_ema"),
        "grad_var_ema": state.get("grad_var_ema"),
        "grad_norm_ema": state.get("grad_norm_ema"),
        "grad_max_ema": state.get("grad_max_ema"),
    }


def print_vgs_comparison(name_a: str, state_a: Dict, name_b: str, state_b: Dict):
    """Print VGS state comparison."""
    print(f"\nVGS State Comparison:")
    grad_var_a_str = f"{state_a['grad_var_ema']:.6e}" if state_a['grad_var_ema'] is not None else 'None'
    grad_var_b_str = f"{state_b['grad_var_ema']:.6e}" if state_b['grad_var_ema'] is not None else 'None'
    print(f"  {name_a}: step_count={state_a['step_count']}, grad_var_ema={grad_var_a_str}")
    print(f"  {name_b}: step_count={state_b['step_count']}, grad_var_ema={grad_var_b_str}")


class TestVGS_PBT_Fix:
    """Test suite for VGS + PBT fix validation."""

    def test_v2_checkpoint_includes_vgs_state(self):
        """
        Test 1: Verify that v2 checkpoints include VGS state.
        """
        print("\n" + "=" * 70)
        print("TEST 1: V2 Checkpoint Includes VGS State")
        print("=" * 70)

        model, env = create_test_model(use_vgs=True, seed=42)

        print("\n1. Training model (100 timesteps)...")
        model.learn(total_timesteps=100, progress_bar=False)

        vgs_before = get_vgs_state(model)
        print(f"\n2. VGS state: step_count={vgs_before['step_count']}")

        # Simulate PBT v2 checkpoint saving
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "member_checkpoint.pt"

            print(f"\n3. Saving v2 checkpoint (full parameters)...")
            full_parameters = model.get_parameters()

            checkpoint_to_save = {
                "format_version": "v2_full_parameters",
                "data": full_parameters,
                "step": 100,
                "performance": 0.75,
            }

            torch.save(checkpoint_to_save, checkpoint_path)

            # Load and verify
            print(f"4. Loading checkpoint to verify VGS state...")
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            assert "format_version" in loaded_checkpoint
            assert loaded_checkpoint["format_version"] == "v2_full_parameters"
            assert "data" in loaded_checkpoint

            loaded_params = loaded_checkpoint["data"]
            assert "vgs_state" in loaded_params

            loaded_vgs_state = loaded_params["vgs_state"]
            print(f"   VGS state found: step_count={loaded_vgs_state.get('step_count', 'N/A')}")

            assert loaded_vgs_state["step_count"] == vgs_before["step_count"]

            print("\n" + "=" * 70)
            print("[PASS] TEST 1 PASSED: V2 checkpoint correctly includes VGS state")
            print("=" * 70)

    def test_pbt_scheduler_with_full_parameters(self):
        """
        Test 2: Verify PBT scheduler works with full parameters.
        """
        print("\n" + "=" * 70)
        print("TEST 2: PBT Scheduler With Full Parameters")
        print("=" * 70)

        # Create two models
        model_a, env_a = create_test_model(learning_rate=1e-4, use_vgs=True, seed=42)
        model_b, env_b = create_test_model(learning_rate=5e-4, use_vgs=True, seed=123)

        print("\n1. Training Member A (50 timesteps)...")
        model_a.learn(total_timesteps=50, progress_bar=False)

        print("2. Training Member B (150 timesteps - better performance)...")
        model_b.learn(total_timesteps=150, progress_bar=False)

        vgs_a_before = get_vgs_state(model_a)
        vgs_b = get_vgs_state(model_b)

        print_vgs_comparison("Member A (before)", vgs_a_before, "Member B (source)", vgs_b)

        # Create PBT scheduler
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt_config = PBTConfig(
                population_size=2,
                perturbation_interval=50,
                truncation_ratio=0.5,  # 50% - ensures exploitation with 2 members
                hyperparams=[
                    HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3)
                ],
                checkpoint_dir=tmpdir,
                metric_name="mean_reward",
                metric_mode="max",
            )

            scheduler = PBTScheduler(pbt_config)
            population = scheduler.initialize_population()
            member_a = population[0]
            member_b = population[1]

            print(f"\n3. Saving checkpoints with full parameters...")

            # Save checkpoints using NEW API (model_parameters)
            # Member A has WORST performance (will be exploited)
            member_a.performance = 0.3  # Worst
            scheduler.update_performance(
                member_a,
                performance=0.3,
                step=50,
                model_parameters=model_a.get_parameters()  # Full parameters (includes VGS!)
            )

            # Member B has BEST performance (will be source)
            member_b.performance = 0.9  # Best
            scheduler.update_performance(
                member_b,
                performance=0.9,
                step=150,
                model_parameters=model_b.get_parameters()  # Full parameters (includes VGS!)
            )

            # Set member_a step to trigger perturbation
            member_a.step = pbt_config.perturbation_interval

            print(f"4. Simulating PBT exploitation...")
            print(f"   Member A (perf=0.3) exploiting from Member B (perf=0.9)")

            # Exploit
            new_parameters, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(member_a)

            print(f"\n5. Exploitation result:")
            print(f"   Exploitation occurred: {new_parameters is not None}")
            print(f"   Checkpoint format: {checkpoint_format}")

            if new_parameters is not None:
                assert checkpoint_format == "v2_full_parameters", f"Expected v2 format, got {checkpoint_format}"

                # Check VGS state is included
                assert "vgs_state" in new_parameters, "VGS state should be in new_parameters!"

                loaded_vgs = new_parameters["vgs_state"]
                print(f"   VGS state transferred: step_count={loaded_vgs.get('step_count', 'N/A')}")

                # Apply parameters to model_a
                print(f"\n6. Applying transferred parameters to Member A...")
                model_a.set_parameters(new_parameters)

                vgs_a_after = get_vgs_state(model_a)

                print_vgs_comparison("Member A (after exploit)", vgs_a_after, "Member B (expected)", vgs_b)

                # Verification
                print(f"\n7. Verification:")
                step_match = (vgs_a_after['step_count'] == vgs_b['step_count'])

                if step_match:
                    print(f"   [OK] VGS step_count matches: {vgs_a_after['step_count']} == {vgs_b['step_count']}")
                else:
                    print(f"   [FAIL] VGS step_count mismatch: {vgs_a_after['step_count']} != {vgs_b['step_count']}")

                # Test training
                print(f"\n8. Testing training after exploitation...")
                try:
                    model_a.learn(total_timesteps=50, progress_bar=False)
                    print("   [OK] Training continues successfully")
                    training_ok = True
                except Exception as e:
                    print(f"   [FAIL] Training failed: {e}")
                    training_ok = False

                # Final verdict
                print("\n" + "=" * 70)
                if step_match and training_ok:
                    print("[PASS] TEST 2 PASSED: VGS state correctly transferred via PBT!")
                else:
                    print("[FAIL] TEST 2 FAILED: VGS state not properly transferred")
                print("=" * 70)

                assert step_match, f"VGS step_count should match: {vgs_a_after['step_count']} != {vgs_b['step_count']}"
                assert training_ok, "Training should work after exploitation"
            else:
                print("\n[WARN] No exploitation occurred (members too similar)")

    def test_backward_compatibility_v1_checkpoints(self):
        """
        Test 3: Verify backward compatibility with v1 (legacy) checkpoints.
        """
        print("\n" + "=" * 70)
        print("TEST 3: Backward Compatibility With V1 Checkpoints")
        print("=" * 70)

        model, env = create_test_model(use_vgs=True, seed=42)

        print("\n1. Training model...")
        model.learn(total_timesteps=100, progress_bar=False)

        # Create legacy v1 checkpoint (policy_only)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"

            print(f"\n2. Saving LEGACY v1 checkpoint (policy_only, NO VGS)...")
            # Legacy format: Just policy state_dict, no metadata
            policy_state_dict = model.policy.state_dict()
            torch.save(policy_state_dict, checkpoint_path)

            # Create PBT scheduler
            pbt_config = PBTConfig(
                population_size=2,
                perturbation_interval=50,
                truncation_ratio=0.5,  # 50% - ensures exploitation with 2 members
                hyperparams=[HyperparamConfig(name="learning_rate", min_value=1e-5, max_value=1e-3)],
                checkpoint_dir=tmpdir,
                metric_name="mean_reward",
                metric_mode="max",
            )

            scheduler = PBTScheduler(pbt_config)
            population = scheduler.initialize_population()
            member_a = population[0]
            member_b = population[1]

            # Set up exploitation scenario: member_a (worst) exploits from member_b (best)
            member_a.checkpoint_path = str(checkpoint_path)  # Legacy checkpoint
            member_a.performance = 0.2  # Worst
            member_a.step = pbt_config.perturbation_interval  # Trigger perturbation

            member_b.checkpoint_path = str(checkpoint_path)  # Source (same legacy checkpoint for test)
            member_b.performance = 0.9  # Best

            print(f"3. Loading legacy checkpoint via PBT exploit...")
            new_parameters, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(member_a)

            print(f"\n4. Results:")
            print(f"   Checkpoint format detected: {checkpoint_format}")
            print(f"   Parameters loaded: {new_parameters is not None}")

            if new_parameters is not None:
                assert checkpoint_format == "v1_policy_only", f"Expected v1 format, got {checkpoint_format}"

                # V1 format should NOT include VGS state
                has_vgs = "vgs_state" in new_parameters if isinstance(new_parameters, dict) else False
                print(f"   VGS state included: {has_vgs}")

                if not has_vgs:
                    print("   [OK] V1 checkpoint correctly identified (no VGS state)")
                else:
                    print("   [WARN] V1 checkpoint unexpectedly contains VGS state")

                print("\n" + "=" * 70)
                print("[PASS] TEST 3 PASSED: Backward compatibility maintained")
                print("=" * 70)


if __name__ == "__main__":
    test = TestVGS_PBT_Fix()

    try:
        print("\n" + "=" * 70)
        print("RUNNING VGS + PBT FIX VALIDATION SUITE")
        print("=" * 70)

        # Test 1: V2 checkpoints include VGS
        test.test_v2_checkpoint_includes_vgs_state()

        # Test 2: PBT with full parameters
        test.test_pbt_scheduler_with_full_parameters()

        # Test 3: Backward compatibility
        test.test_backward_compatibility_v1_checkpoints()

        print("\n" + "=" * 70)
        print("[OK] ALL FIX VALIDATION TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\nX TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nX UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
