"""Debug script to trace VGS state loading."""

import tempfile
from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def make_test_env():
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


print("=" * 80)
print("DEBUG: VGS State Loading Trace")
print("=" * 80)

# Create and train model
print("\n1. Creating model with VGS...")
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

print("2. Training...")
model.learn(total_timesteps=2048)

vgs_before = model._variance_gradient_scaler
print(f"\n3. Before save:")
print(f"   VGS exists: {vgs_before is not None}")
print(f"   step_count: {vgs_before._step_count if vgs_before else 'N/A'}")

# Save
with tempfile.TemporaryDirectory() as tmp_dir:
    save_path = Path(tmp_dir) / "test.zip"
    print(f"\n4. Saving to {save_path}...")
    model.save(save_path)

    # Check what was saved
    print("\n5. Checking get_parameters()...")
    params = model.get_parameters()
    vgs_state_saved = params.get("vgs_state")
    print(f"   vgs_state in params: {vgs_state_saved is not None}")
    if vgs_state_saved:
        print(f"   vgs_state['step_count']: {vgs_state_saved.get('step_count', 'N/A')}")

    # Load
    print(f"\n6. Loading from {save_path}...")

    # Patch _restore_vgs_state to add print
    original_restore = DistributionalPPO._restore_vgs_state

    def patched_restore(self, state):
        print(f"   [DEBUG] _restore_vgs_state called!")
        print(f"   [DEBUG] state is None: {state is None}")
        print(f"   [DEBUG] state type: {type(state)}")
        if state:
            print(f"   [DEBUG] state['step_count']: {state.get('step_count', 'N/A')}")
            print(f"   [DEBUG] VGS exists: {self._variance_gradient_scaler is not None}")
        return original_restore(self, state)

    DistributionalPPO._restore_vgs_state = patched_restore

    loaded_model = DistributionalPPO.load(save_path, env=env)

    # Restore original
    DistributionalPPO._restore_vgs_state = original_restore

    print("\n7. After load:")
    vgs_after = loaded_model._variance_gradient_scaler
    print(f"   VGS exists: {vgs_after is not None}")
    print(f"   step_count: {vgs_after._step_count if vgs_after else 'N/A'}")
    print(f"   grad_mean_ema: {vgs_after._grad_mean_ema if vgs_after else 'N/A'}")

    print("\n8. Checking _vgs_saved_state_for_restore attribute...")
    has_saved_state = hasattr(loaded_model, "_vgs_saved_state_for_restore")
    print(f"   Has attribute: {has_saved_state}")
    if has_saved_state:
        saved_state = getattr(loaded_model, "_vgs_saved_state_for_restore")
        print(f"   Value: {saved_state}")

env.close()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
