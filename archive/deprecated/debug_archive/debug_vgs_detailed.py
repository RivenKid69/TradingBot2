"""
Detailed debug script with instrumentation in _setup_dependent_components.
"""

import gymnasium as gym
import tempfile
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy
import torch


# Monkey-patch to add debug output
original_setup = DistributionalPPO._setup_dependent_components


def instrumented_setup(self):
    """Instrumented version of _setup_dependent_components."""
    print("\n[DEBUG] _setup_dependent_components() called")

    # Check if policy exists
    if not hasattr(self, "policy") or self.policy is None:
        print("[DEBUG] Policy does NOT exist yet - skipping")
        return

    # Get IDs before setup
    params_before = list(self.policy.parameters())
    ids_before = [id(p) for p in params_before]
    print(f"[DEBUG] Policy parameter IDs at START: {ids_before[:3]}... (showing first 3)")

    # Check if VGS exists
    if hasattr(self, "_variance_gradient_scaler") and self._variance_gradient_scaler is not None:
        vgs_params_before = self._variance_gradient_scaler._parameters
        if vgs_params_before:
            vgs_ids_before = [id(p) for p in vgs_params_before]
            print(f"[DEBUG] VGS parameter IDs BEFORE setup: {vgs_ids_before[:3]}... (showing first 3)")

    # Call original
    original_setup(self)

    # Get IDs after setup
    params_after = list(self.policy.parameters())
    ids_after = [id(p) for p in params_after]
    print(f"[DEBUG] Policy parameter IDs at END: {ids_after[:3]}... (showing first 3)")

    if hasattr(self, "_variance_gradient_scaler") and self._variance_gradient_scaler is not None:
        vgs_params_after = self._variance_gradient_scaler._parameters
        if vgs_params_after:
            vgs_ids_after = [id(p) for p in vgs_params_after]
            print(f"[DEBUG] VGS parameter IDs AFTER setup: {vgs_ids_after[:3]}... (showing first 3)")

            # Check if they match
            match_count = sum(1 for vid in vgs_ids_after if vid in set(ids_after))
            print(f"[DEBUG] VGS-Policy ID matches: {match_count}/{len(vgs_ids_after)}")


DistributionalPPO._setup_dependent_components = instrumented_setup


env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

print("=" * 80)
print("Creating and training model")
print("=" * 80)
model = DistributionalPPO(
    CustomActorCriticPolicy,
    env,
    variance_gradient_scaling=True,
    n_steps=64,
    verbose=0,
)
model.learn(total_timesteps=256)

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "debug_vgs.zip"

    print("\n" + "=" * 80)
    print("Saving model")
    print("=" * 80)
    model.save(save_path)

    print("\n" + "=" * 80)
    print("Loading model")
    print("=" * 80)
    loaded_model = DistributionalPPO.load(save_path, env=env)

    print("\n" + "=" * 80)
    print("Final verification")
    print("=" * 80)
    params_final = list(loaded_model.policy.parameters())
    vgs_params_final = loaded_model._variance_gradient_scaler._parameters

    ids_final = [id(p) for p in params_final]
    vgs_ids_final = [id(p) for p in vgs_params_final]

    match_count = sum(1 for vid in vgs_ids_final if vid in set(ids_final))
    print(f"Final VGS-Policy ID matches: {match_count}/{len(vgs_params_final)}")

    if match_count == 0:
        print("\n[BUG CONFIRMED] VGS tracks completely different parameter objects!")

        # Check if it's the optimizer's params
        if hasattr(loaded_model.policy, "optimizer") and loaded_model.policy.optimizer is not None:
            opt_param_ids = set()
            for group in loaded_model.policy.optimizer.param_groups:
                for p in group['params']:
                    opt_param_ids.add(id(p))

            vgs_in_opt = sum(1 for vid in vgs_ids_final if vid in opt_param_ids)
            print(f"VGS params found in optimizer.param_groups: {vgs_in_opt}/{len(vgs_params_final)}")

            opt_in_policy = sum(1 for oid in opt_param_ids if oid in set(ids_final))
            print(f"Optimizer params matching current policy: {opt_in_policy}/{len(opt_param_ids)}")

env.close()