"""
Debug script to trace VGS parameter tracking through save/load cycle.

Goal: Understand where the "mysterious" parameters come from that VGS tracks
after load, which are neither old (before save) nor current (after load).
"""

import gymnasium as gym
import tempfile
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def get_param_details(params):
    """Get detailed info about parameters."""
    param_list = list(params)
    ids = [id(p) for p in param_list]
    shapes = [tuple(p.shape) for p in param_list]
    return param_list, ids, shapes


env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

print("=" * 80)
print("STEP 1: Create model")
print("=" * 80)
model = DistributionalPPO(
    CustomActorCriticPolicy,
    env,
    variance_gradient_scaling=True,
    n_steps=64,
    verbose=0,
)

params_1, ids_1, shapes_1 = get_param_details(model.policy.parameters())
vgs_params_1, vgs_ids_1, vgs_shapes_1 = get_param_details(model._variance_gradient_scaler._parameters)

print(f"Policy parameters: {len(params_1)}")
print(f"VGS parameters:    {len(vgs_params_1)}")
print(f"Shapes match: {shapes_1 == vgs_shapes_1}")
print(f"IDs match:    {ids_1 == vgs_ids_1}")
print()

print("=" * 80)
print("STEP 2: Train briefly")
print("=" * 80)
model.learn(total_timesteps=256)
print("Training done")
print()

params_2, ids_2, shapes_2 = get_param_details(model.policy.parameters())
vgs_params_2, vgs_ids_2, vgs_shapes_2 = get_param_details(model._variance_gradient_scaler._parameters)

print(f"Policy IDs same as before training: {ids_2 == ids_1}")
print(f"VGS IDs same as before training:    {vgs_ids_2 == vgs_ids_1}")
print()

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "debug_vgs.zip"

    print("=" * 80)
    print("STEP 3: Save model")
    print("=" * 80)
    params_before_save, ids_before_save, shapes_before_save = get_param_details(model.policy.parameters())
    vgs_params_before_save, vgs_ids_before_save, vgs_shapes_before_save = get_param_details(
        model._variance_gradient_scaler._parameters
    )

    print(f"Policy parameters before save: {len(ids_before_save)} IDs")
    print(f"VGS parameters before save:    {len(vgs_ids_before_save)} IDs")
    print(f"IDs match: {set(ids_before_save) == set(vgs_ids_before_save)}")
    print()

    model.save(save_path)
    print("Model saved")
    print()

    print("=" * 80)
    print("STEP 4: Load model")
    print("=" * 80)
    loaded_model = DistributionalPPO.load(save_path, env=env)

    params_after_load, ids_after_load, shapes_after_load = get_param_details(loaded_model.policy.parameters())
    vgs_params_after_load, vgs_ids_after_load, vgs_shapes_after_load = get_param_details(
        loaded_model._variance_gradient_scaler._parameters
    )

    print(f"Policy parameters after load: {len(ids_after_load)} IDs")
    print(f"VGS parameters after load:    {len(vgs_ids_after_load)} IDs")
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Shapes same: {shapes_after_load == vgs_shapes_after_load}")
    print(f"Number of params: policy={len(ids_after_load)}, vgs={len(vgs_ids_after_load)}")
    print()

    # Check ID overlaps
    set_before_save = set(ids_before_save)
    set_after_load = set(ids_after_load)
    set_vgs_after_load = set(vgs_ids_after_load)

    print("ID Overlaps:")
    print(f"  Policy before save & Policy after load = {len(set_before_save & set_after_load)}")
    print(f"  Policy after load & VGS after load = {len(set_after_load & set_vgs_after_load)}")
    print(f"  VGS after load & VGS before save = {len(set_vgs_after_load & set(vgs_ids_before_save))}")
    print()

    if set_after_load == set_vgs_after_load:
        print("[OK] VGS tracks correct parameters")
    else:
        print("[BUG] VGS tracks wrong parameters")
        print()
        print("Let's check if VGS holds copies or references...")

        # Check if VGS params are the exact same objects (not copies)
        exact_match_count = 0
        for policy_p in params_after_load:
            for vgs_p in vgs_params_after_load:
                if policy_p is vgs_p:  # Same object reference
                    exact_match_count += 1
                    break

        print(f"  Exact object matches: {exact_match_count}/{len(params_after_load)}")

        if exact_match_count == 0:
            print("  [DIAGNOSIS] VGS is tracking DIFFERENT parameter objects!")
            print("  This suggests VGS was given a copy or a snapshot of parameters,")
            print("  not the actual current parameters.")

        # Let's check parameter values
        print()
        print("  Checking if parameter VALUES match...")
        values_match_count = 0
        for i, policy_p in enumerate(params_after_load):
            for j, vgs_p in enumerate(vgs_params_after_load):
                if policy_p.shape == vgs_p.shape:
                    if (policy_p == vgs_p).all():
                        values_match_count += 1
                        break

        print(f"  Parameters with matching values: {values_match_count}/{len(params_after_load)}")

        if values_match_count == len(params_after_load):
            print("  [DIAGNOSIS] Values match, but IDs don't!")
            print("  This means VGS has COPIES of the parameters, not references.")

env.close()