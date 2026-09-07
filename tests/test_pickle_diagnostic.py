"""Diagnostic test to find what exactly cannot be pickled."""

import pytest

pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
import cloudpickle
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_pickle_components():
    """Test pickling individual components to find the problematic one."""
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        optimizer_class="adaptive_upgd",
        variance_gradient_scaling=True,
        n_steps=64,
        verbose=0,
    )

    model.learn(total_timesteps=128)

    print("Testing pickle ability of model components...\n")

    # Test individual attributes
    attrs_to_test = [
        "clip_range",
        "_logger",
        "logger",
        "policy",
        "rollout_buffer",
        "lr_schedule",
        "observation_space",
        "action_space",
    ]

    for attr_name in attrs_to_test:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            try:
                cloudpickle.dumps(attr)
                print(f"[OK] {attr_name}: {type(attr)}")
            except Exception as e:
                print(f"[FAIL] {attr_name}: {type(attr)} - {type(e).__name__}: {e}")
        else:
            print(f"[N/A] {attr_name}: not found")

    # Test model.__dict__ keys
    print("\nTesting model.__dict__ items that might contain file handles...\n")

    for key in sorted(model.__dict__.keys()):
        value = model.__dict__[key]
        try:
            cloudpickle.dumps(value)
        except Exception as e:
            print(f"[FAIL] model.{key}: {type(value)} - {e}")

    env.close()


if __name__ == "__main__":
    test_pickle_components()
