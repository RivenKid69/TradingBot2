#!/usr/bin/env python3
"""Quick test to verify Twin Critics is enabled by default."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, "/home/user/AI-Powered Quantitative Research Platform")


def test_default_behavior():
    """Test that Twin Critics are enabled by default."""
    print("=" * 70)
    print("QUICK TEST: Twin Critics Default Behavior")
    print("=" * 70)

    try:
        # Import necessary modules
        print("\n[1/5] Importing modules...")
        import numpy as np
        import torch
        from gymnasium import spaces
        from custom_policy_patch1 import CustomActorCriticPolicy

        print("✓ Imports successful")

        # Create basic config WITHOUT use_twin_critics specified
        print("\n[2/5] Creating policy with default config (no use_twin_critics)...")
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                # use_twin_critics NOT specified - should default to True
            },
        }

        policy = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        print("✓ Policy created")

        # Check that Twin Critics is enabled by default
        print("\n[3/5] Checking Twin Critics default state...")
        assert policy._use_twin_critics is True, "FAIL: Twin Critics should be True by default"
        print(f"✓ policy._use_twin_critics = {policy._use_twin_critics} (ENABLED BY DEFAULT)")

        # Check that second critic exists
        print("\n[4/5] Verifying second critic exists...")
        assert policy.quantile_head is not None, "FAIL: First critic should exist"
        assert policy.quantile_head_2 is not None, "FAIL: Second critic should exist"
        print("✓ Both critic heads exist")

        # Test forward pass
        print("\n[5/5] Testing forward pass with both critics...")
        latent_vf = torch.randn(4, 32)
        logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)
        print(
            f"✓ Forward pass successful: logits_1.shape={logits_1.shape}, logits_2.shape={logits_2.shape}"
        )

        # Test explicit disable
        print("\n[BONUS] Testing explicit disable...")
        arch_params_disable = {
            "hidden_dim": 32,
            "critic": {
                "distributional": True,
                "num_quantiles": 16,
                "use_twin_critics": False,  # Explicitly disable
            },
        }

        policy_disabled = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params_disable
        )

        assert (
            policy_disabled._use_twin_critics is False
        ), "FAIL: Should be disabled when explicitly set"
        assert (
            policy_disabled.quantile_head_2 is None
        ), "FAIL: Second critic should not exist when disabled"
        print(
            f"✓ Explicit disable works: policy._use_twin_critics = {policy_disabled._use_twin_critics}"
        )

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - Twin Critics is ENABLED BY DEFAULT")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_default_behavior()
    sys.exit(0 if success else 1)
