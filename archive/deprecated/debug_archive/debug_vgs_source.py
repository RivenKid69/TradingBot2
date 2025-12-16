"""
Debug script to trace where VGS with wrong parameters comes from.
"""

import gymnasium as gym
import tempfile
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy
import torch


# Monkey-patch VarianceGradientScaler to trace its creation
from variance_gradient_scaler import VarianceGradientScaler

original_vgs_init = VarianceGradientScaler.__init__


def traced_vgs_init(self, parameters=None, **kwargs):
    """Traced __init__."""
    if parameters is not None:
        params_list = list(parameters)
        ids = [id(p) for p in params_list[:3]]
        print(f"[VGS TRACE] Creating VGS with parameter IDs: {ids}... (first 3)")
    else:
        print(f"[VGS TRACE] Creating VGS with NO parameters")
    original_vgs_init(self, parameters, **kwargs)


VarianceGradientScaler.__init__ = traced_vgs_init

# Monkey-patch update_parameters
original_update_params = VarianceGradientScaler.update_parameters


def traced_update_params(self, parameters):
    """Traced update_parameters."""
    old_ids = [id(p) for p in self._parameters[:3]] if self._parameters else []
    # Convert to list to avoid exhausting generator
    params_list = list(parameters)
    new_ids = [id(p) for p in params_list[:3]]
    print(f"[VGS TRACE] update_parameters called")
    print(f"  Old IDs: {old_ids}...")
    print(f"  New IDs: {new_ids}...")
    # Pass the list instead of exhausted generator
    original_update_params(self, params_list)
    actual_ids = [id(p) for p in self._parameters[:3]] if self._parameters else []
    print(f"  Result IDs: {actual_ids}...")


VarianceGradientScaler.update_parameters = traced_update_params

# Monkey-patch load_state_dict
original_load_state = VarianceGradientScaler.load_state_dict


def traced_load_state(self, state_dict):
    """Traced load_state_dict."""
    print(f"[VGS TRACE] load_state_dict called")
    print(f"  step_count: {state_dict.get('step_count', 'N/A')}")
    param_ids = [id(p) for p in self._parameters[:3]] if self._parameters else []
    print(f"  Parameter IDs before load_state_dict: {param_ids}...")
    original_load_state(self, state_dict)
    param_ids_after = [id(p) for p in self._parameters[:3]] if self._parameters else []
    print(f"  Parameter IDs after load_state_dict: {param_ids_after}...")


VarianceGradientScaler.load_state_dict = traced_load_state

# Monkey-patch __setstate__
original_vgs_setstate = VarianceGradientScaler.__setstate__


def traced_vgs_setstate(self, state):
    """Traced __setstate__."""
    print(f"[VGS TRACE] __setstate__ called")
    if "_parameters" in state and state["_parameters"]:
        param_ids = [id(p) for p in state["_parameters"][:3]]
        print(f"  Parameter IDs in state: {param_ids}...")
    else:
        print(f"  No _parameters in state")
    original_vgs_setstate(self, state)
    if hasattr(self, "_parameters") and self._parameters:
        actual_ids = [id(p) for p in self._parameters[:3]]
        print(f"  Parameter IDs after restore: {actual_ids}...")


VarianceGradientScaler.__setstate__ = traced_vgs_setstate

# Monkey-patch _setup_dependent_components
from distributional_ppo import DistributionalPPO

original_setup_deps = DistributionalPPO._setup_dependent_components


def traced_setup_deps(self):
    """Traced _setup_dependent_components."""
    print(f"\n[SETUP TRACE] _setup_dependent_components() called")
    has_policy = hasattr(self, "policy") and self.policy is not None
    vgs_enabled = getattr(self, "_vgs_enabled", False)
    print(f"  has_policy: {has_policy}")
    print(f"  vgs_enabled: {vgs_enabled}")
    if has_policy and vgs_enabled:
        print(f"  Policy exists and VGS enabled - should create VGS")
    try:
        original_setup_deps(self)
        print(f"  After setup: VGS is {type(self._variance_gradient_scaler).__name__ if self._variance_gradient_scaler else 'None'}")
        if self._variance_gradient_scaler:
            print(f"  VGS._parameters is {type(self._variance_gradient_scaler._parameters).__name__ if self._variance_gradient_scaler._parameters else 'None'}")
    except Exception as e:
        print(f"  EXCEPTION in _setup_dependent_components: {e}")
        import traceback
        traceback.print_exc()
        raise


DistributionalPPO._setup_dependent_components = traced_setup_deps


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

    print("\n[LOAD TRACE] After load() returns:")
    print(f"  VGS type: {type(loaded_model._variance_gradient_scaler).__name__ if loaded_model._variance_gradient_scaler else 'None'}")
    if loaded_model._variance_gradient_scaler:
        print(f"  VGS._parameters type: {type(loaded_model._variance_gradient_scaler._parameters).__name__ if loaded_model._variance_gradient_scaler._parameters else 'None'}")
        if loaded_model._variance_gradient_scaler._parameters:
            param_ids = [id(p) for p in loaded_model._variance_gradient_scaler._parameters[:3]]
            print(f"  VGS parameter IDs: {param_ids}...")

    print("\n" + "=" * 80)
    print("Final verification")
    print("=" * 80)
    params_final = list(loaded_model.policy.parameters())
    vgs_params_final = loaded_model._variance_gradient_scaler._parameters

    ids_final = [id(p) for p in params_final]
    vgs_ids_final = [id(p) for p in vgs_params_final]

    match_count = sum(1 for vid in vgs_ids_final if vid in set(ids_final))
    print(f"Final VGS-Policy ID matches: {match_count}/{len(vgs_params_final)}")

env.close()
