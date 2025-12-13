"""
Tests for Twin Critics save/load functionality and backward compatibility.
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
import tempfile
import os
from gymnasium import spaces
from custom_policy_patch1 import CustomActorCriticPolicy
import sys


def test_save_load_twin_critics():
    """Test that Twin Critics models can be saved and loaded correctly."""
    print("\n=== Test: Save/Load Twin Critics ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    arch_params = {
        'hidden_dim': 32,
        'critic': {
            'distributional': True,
            'num_quantiles': 16,
            'use_twin_critics': True,
        }
    }

    # Create policy with twin critics
    print("Step 1: Create policy with Twin Critics")
    policy_original = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )

    # Get some outputs before saving
    test_latent = torch.randn(4, 32)
    outputs_before = policy_original._get_twin_value_logits(test_latent)
    print(f"  Outputs before save: {outputs_before[0].shape}, {outputs_before[1].shape}")

    # Save to temporary file
    print("Step 2: Save model")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name

    try:
        torch.save(policy_original.state_dict(), temp_path)
        print(f"  Saved to: {temp_path}")

        # Create new policy and load weights
        print("Step 3: Load into new policy")
        policy_loaded = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_params
        )
        policy_loaded.load_state_dict(torch.load(temp_path))
        print("  Loaded successfully")

        # Check outputs match
        print("Step 4: Verify outputs match")
        outputs_after = policy_loaded._get_twin_value_logits(test_latent)

        diff1 = (outputs_before[0] - outputs_after[0]).abs().max().item()
        diff2 = (outputs_before[1] - outputs_after[1]).abs().max().item()

        print(f"  Max diff critic 1: {diff1:.2e}")
        print(f"  Max diff critic 2: {diff2:.2e}")

        assert diff1 < 1e-6, f"Critic 1 outputs differ: {diff1}"
        assert diff2 < 1e-6, f"Critic 2 outputs differ: {diff2}"

        print("✓ Save/load preserves Twin Critics state")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("✅ Save/load test passed")


def test_load_single_critic_into_twin():
    """Test loading single critic model into twin critics architecture (should fail gracefully)."""
    print("\n=== Test: Load Single→Twin (Should Warn/Fail) ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Create single critic model
    print("Step 1: Create single critic model")
    arch_single = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}
    }
    policy_single = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_single
    )

    # Save single critic state
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name

    try:
        torch.save(policy_single.state_dict(), temp_path)
        print("  Saved single critic model")

        # Try to load into twin critics policy
        print("Step 2: Try loading into twin critics policy")
        arch_twin = {
            'hidden_dim': 32,
            'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': True}
        }
        policy_twin = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_twin
        )

        state_dict = torch.load(temp_path)

        # This should either fail or load with warnings (missing keys for second critic)
        try:
            result = policy_twin.load_state_dict(state_dict, strict=False)
            print(f"  Loaded with strict=False")
            print(f"  Missing keys: {result.missing_keys}")
            print(f"  Unexpected keys: {result.unexpected_keys}")

            # Second critic should have missing keys
            missing_second_critic = any('_2' in k or 'quantile_head_2' in k for k in result.missing_keys)
            assert missing_second_critic, "Should have missing keys for second critic"
            print("✓ Correctly identified missing second critic keys")

        except RuntimeError as e:
            print(f"  Failed as expected: {e}")
            print("✓ Strict loading correctly fails")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("✅ Single→Twin compatibility test passed")


def test_load_twin_into_single():
    """Test loading twin critics model into single critic architecture."""
    print("\n=== Test: Load Twin→Single ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Create twin critics model
    print("Step 1: Create twin critics model")
    arch_twin = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': True}
    }
    policy_twin = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_twin
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name

    try:
        torch.save(policy_twin.state_dict(), temp_path)
        print("  Saved twin critics model")

        # Load into single critic policy
        print("Step 2: Load into single critic policy")
        arch_single = {
            'hidden_dim': 32,
            'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}
        }
        policy_single = CustomActorCriticPolicy(
            observation_space, action_space, lambda x: 0.001, arch_params=arch_single
        )

        state_dict = torch.load(temp_path)
        result = policy_single.load_state_dict(state_dict, strict=False)

        print(f"  Unexpected keys: {result.unexpected_keys}")

        # Should have unexpected keys for second critic
        unexpected_second_critic = any('_2' in k or 'quantile_head_2' in k for k in result.unexpected_keys)
        assert unexpected_second_critic, "Should have unexpected keys for second critic"
        print("✓ Correctly ignored second critic keys")

        # First critic should be loaded correctly
        test_latent = torch.randn(4, 32)
        output = policy_single._get_value_logits(test_latent)
        assert output.shape == (4, 16)
        print("✓ First critic works after loading")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("✅ Twin→Single compatibility test passed")


def test_backward_compatibility_no_twin_critics():
    """Test that code without twin critics still works (backward compatibility)."""
    print("\n=== Test: Backward Compatibility ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Test 1: Default behavior (no twin critics key in config)
    print("Test 1: Default config (no 'use_twin_critics' key)")
    arch_params = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16}
        # Note: 'use_twin_critics' not specified - should default to True
    }
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    assert policy._use_twin_critics == True, "Should default to True"
    assert policy.quantile_head_2 is not None, "Should create second critic by default"
    print("✓ Default behavior: Twin Critics enabled")

    # Test 2: Explicit False
    print("Test 2: Explicit use_twin_critics=False")
    arch_params = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}
    }
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    assert policy._use_twin_critics == False
    assert policy.quantile_head_2 is None
    print("✓ Explicit False works")

    # Test 3: Forward pass works without twin critics
    print("Test 3: Forward pass without twin critics")
    latent = torch.randn(4, 32)
    output = policy._get_value_logits(latent)
    assert output.shape == (4, 16)
    print("✓ Forward pass works")

    # Test 4: _get_min_twin_values falls back to single critic when disabled
    print("Test 4: Fallback in _get_min_twin_values when twin critics disabled")
    min_vals = policy._get_min_twin_values(latent)
    single_vals = policy._get_value_from_latent(latent)
    assert torch.allclose(min_vals, single_vals), "Should return single critic value when disabled"
    print("✓ Fallback works when twin critics disabled")

    # Test 5: Old code should still work
    print("Test 5: Old code patterns still work")
    # Simulate old code that doesn't know about twin critics
    try:
        _ = policy._get_value_logits(latent)
        _ = policy._get_value_from_latent(latent)
        print("✓ Old code patterns work")
    except Exception as e:
        print(f"✗ Old code failed: {e}")
        raise

    print("✅ Backward compatibility test passed")


def test_optimizer_contains_twin_critics():
    """Test that optimizer contains parameters from both critics."""
    print("\n=== Test: Optimizer Contains Twin Critics ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    arch_params = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': True}
    }

    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )

    # Get all parameter IDs in optimizer
    print("Step 1: Check optimizer parameters")
    optimizer_param_ids = set()
    for param_group in policy.optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))

    print(f"  Total params in optimizer: {len(optimizer_param_ids)}")

    # Get parameter IDs from both critics
    critic1_param_ids = {id(p) for p in policy.quantile_head.parameters()}
    critic2_param_ids = {id(p) for p in policy.quantile_head_2.parameters()}

    print(f"  Critic 1 params: {len(critic1_param_ids)}")
    print(f"  Critic 2 params: {len(critic2_param_ids)}")

    # Check both critics in optimizer
    critic1_in_opt = critic1_param_ids.issubset(optimizer_param_ids)
    critic2_in_opt = critic2_param_ids.issubset(optimizer_param_ids)

    assert critic1_in_opt, "Critic 1 parameters missing from optimizer"
    assert critic2_in_opt, "Critic 2 parameters missing from optimizer"

    print("✓ Both critics in optimizer")

    # Test that optimizer updates both critics
    print("Step 2: Test optimizer updates")
    latent = torch.randn(4, 32, requires_grad=True)

    # Store initial weights
    weight1_before = policy.quantile_head.linear.weight.data.clone()
    weight2_before = policy.quantile_head_2.linear.weight.data.clone()

    # Compute loss and optimize
    logits1, logits2 = policy._get_twin_value_logits(latent)
    loss = logits1.mean() + logits2.mean()
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()

    # Check weights changed
    weight1_after = policy.quantile_head.linear.weight.data
    weight2_after = policy.quantile_head_2.linear.weight.data

    diff1 = (weight1_before - weight1_after).abs().max().item()
    diff2 = (weight2_before - weight2_after).abs().max().item()

    print(f"  Weight change critic 1: {diff1:.2e}")
    print(f"  Weight change critic 2: {diff2:.2e}")

    assert diff1 > 1e-8, f"Critic 1 not updated: {diff1}"
    assert diff2 > 1e-8, f"Critic 2 not updated: {diff2}"

    print("✓ Optimizer updates both critics")

    print("✅ Optimizer test passed")


def run_all_save_load_tests():
    """Run all save/load and compatibility tests."""
    print("\n" + "="*70)
    print("TWIN CRITICS SAVE/LOAD & COMPATIBILITY TEST SUITE")
    print("="*70)

    tests = [
        test_save_load_twin_critics,
        test_load_single_critic_into_twin,
        test_load_twin_into_single,
        test_backward_compatibility_no_twin_critics,
        test_optimizer_contains_twin_critics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)

    if failed > 0:
        sys.exit(1)

    print("\n🎉 ALL SAVE/LOAD TESTS PASSED!")


if __name__ == "__main__":
    run_all_save_load_tests()
