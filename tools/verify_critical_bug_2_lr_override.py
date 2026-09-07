"""
Critical Bug #2 Verification: optimizer_kwargs['lr'] Ignored

Hypothesis: When passing custom learning rate via optimizer_kwargs, the value
is ignored and default learning rate (3e-4) is used instead.

Expected: Custom lr should be used, but default is used instead
Location: custom_policy_patch1.py - optimizer initialization
"""

import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_lr_override():
    """Test if custom learning rate in optimizer_kwargs is respected."""

    print("=" * 80)
    print("CRITICAL BUG #2: optimizer_kwargs['lr'] Ignored")
    print("=" * 80)
    print()

    # Test multiple custom lr values
    test_cases = [
        0.001,  # 1e-3
        0.005,  # 5e-3
        0.0001,  # 1e-4
        0.01,  # 1e-2
    ]

    bugs_found = []

    for custom_lr in test_cases:
        print(f"Test case: custom_lr = {custom_lr}")
        print("-" * 40)

        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

        try:
            # Create model with custom lr in optimizer_kwargs
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_kwargs={"lr": custom_lr},
                verbose=0,
            )

            # Check actual lr in optimizer
            actual_lr = model.policy.optimizer.param_groups[0]["lr"]

            print(f"  Expected lr: {custom_lr}")
            print(f"  Actual lr:   {actual_lr}")

            # Check if lr matches
            if abs(actual_lr - custom_lr) < 1e-9:
                print(f"  [PASS] Custom lr correctly applied")
            else:
                print(f"  [FAIL] Custom lr ignored!")
                print(f"         (Difference: {abs(actual_lr - custom_lr)})")
                bugs_found.append((custom_lr, actual_lr))

        except Exception as e:
            print(f"  [ERROR] {e}")
            bugs_found.append((custom_lr, None))
        finally:
            env.close()

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if bugs_found:
        print(f"BUG CONFIRMED: {len(bugs_found)}/{len(test_cases)} test cases failed")
        print()
        print("Failed cases:")
        for expected, actual in bugs_found:
            if actual is None:
                print(f"  - Expected {expected}, got ERROR")
            else:
                print(f"  - Expected {expected}, got {actual}")
        print()
        print("VERDICT: BUG EXISTS [X]")
        print("Severity: CRITICAL - Cannot configure learning rate")
        return True
    else:
        print(f"All {len(test_cases)} test cases passed")
        print()
        print("VERDICT: BUG NOT FOUND [OK]")
        print("optimizer_kwargs['lr'] works correctly")
        return False


if __name__ == "__main__":
    print("\n")
    bug_exists = test_lr_override()
    print("\n")
    sys.exit(1 if bug_exists else 0)
