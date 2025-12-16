"""Debug script to understand VGS error."""

import traceback
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy

print("Creating model with VGS...")
env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
model = DistributionalPPO(
    CustomActorCriticPolicy,
    env,
    variance_gradient_scaling=True,
    verbose=0,
)

vgs = model._variance_gradient_scaler

print(f"VGS enabled: {vgs is not None}")
if vgs:
    print(f"VGS params: {len(list(vgs._parameters)) if vgs._parameters else 'None'}")
    print(f"First param: {next(iter(vgs._parameters)) if vgs._parameters else 'None'}")

    print("\nChecking parameter gradients...")
    for i, p in enumerate(vgs._parameters):
        if i < 3:  # Only check first 3
            print(f"  Param {i}: shape={p.shape if p is not None else 'None'}, grad={p.grad if p is not None else 'None'}")

print("\nTrying scale_gradients() with no gradients...")
try:
    result = vgs.scale_gradients()
    print(f"Success: scaling_factor={result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "="*80)
print("Creating gradients and trying again...")

# Create some gradients
obs = env.reset()
actions, _ = model.predict(obs, deterministic=True)

print("Resetting gradients...")
model.policy.optimizer.zero_grad()

print("Checking if gradients exist after zero_grad...")
grad_count = sum(1 for p in model.policy.parameters() if p.grad is not None)
print(f"  Parameters with gradients: {grad_count}")

print("\nTrying scale_gradients() after zero_grad...")
try:
    result = vgs.scale_gradients()
    print(f"Success: scaling_factor={result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

env.close()
