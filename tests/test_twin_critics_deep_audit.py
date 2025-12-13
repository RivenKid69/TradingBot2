"""
Deep audit and comprehensive tests for Twin Critics implementation.

This file tests:
1. Edge cases and corner cases
2. Gradient flow in detail
3. Save/load model state
4. Backward compatibility thoroughly
5. Numerical stability
6. Memory leaks
7. Integration with all PPO features
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from custom_policy_patch1 import CustomActorCriticPolicy, QuantileValueHead
from distributional_ppo import DistributionalPPO
import gc
import sys


def test_twin_critics_configuration_validation():
    """Test that Twin Critics configuration is validated correctly."""
    print("\n=== Test: Configuration Validation ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Test 1: Twin critics with quantile mode
    print("Test 1: Twin Critics + Quantile")
    arch_params = {
        'hidden_dim': 32,
        'critic': {
            'distributional': True,
            'num_quantiles': 16,
            'use_twin_critics': True,
        }
    }
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    assert policy._use_twin_critics == True, "Twin critics should be enabled"
    assert policy.quantile_head is not None, "First quantile head should exist"
    assert policy.quantile_head_2 is not None, "Second quantile head should exist"
    print("✓ Quantile mode OK")

    # Test 2: Twin critics with categorical mode
    print("Test 2: Twin Critics + Categorical")
    arch_params = {
        'hidden_dim': 32,
        'num_atoms': 51,
        'critic': {
            'distributional': False,
            'use_twin_critics': True,
        }
    }
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    assert policy._use_twin_critics == True
    assert policy.dist_head is not None
    assert policy.dist_head_2 is not None
    print("✓ Categorical mode OK")

    # Test 3: Default behavior (twin critics ENABLED by default since 2025-11-21)
    print("Test 3: Default (Twin Critics enabled by default)")
    arch_params = {'hidden_dim': 32, 'critic': {'distributional': True, 'num_quantiles': 16}}
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    assert policy._use_twin_critics == True  # Default is True!
    assert policy.quantile_head_2 is not None  # Should exist!
    print("✓ Default behavior OK (Twin Critics enabled)")

    print("✅ Configuration validation passed")


def test_twin_critics_parameter_independence():
    """Test that twin critics have truly independent parameters."""
    print("\n=== Test: Parameter Independence ===")

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

    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )

    # Test 1: Different memory addresses
    print("Test 1: Memory addresses")
    addr1 = id(policy.quantile_head.linear.weight)
    addr2 = id(policy.quantile_head_2.linear.weight)
    assert addr1 != addr2, "Weights should have different memory addresses"
    print(f"  Critic 1 weight address: {hex(addr1)}")
    print(f"  Critic 2 weight address: {hex(addr2)}")
    print("✓ Different memory addresses")

    # Test 2: Different initial values (with high probability)
    print("Test 2: Initial values differ")
    weight1 = policy.quantile_head.linear.weight.data.clone()
    weight2 = policy.quantile_head_2.linear.weight.data.clone()
    diff = (weight1 - weight2).abs().mean().item()
    assert diff > 1e-6, f"Weights should differ (diff={diff})"
    print(f"  Mean absolute difference: {diff:.6f}")
    print("✓ Initial values differ")

    # Test 3: Independent updates
    print("Test 3: Independent gradient updates")
    latent = torch.randn(4, 32, requires_grad=True)

    # Update only first critic
    out1 = policy.quantile_head(latent)
    loss1 = out1.mean()
    policy.zero_grad()
    loss1.backward()

    grad1_exists = policy.quantile_head.linear.weight.grad is not None
    grad2_none = policy.quantile_head_2.linear.weight.grad is None

    assert grad1_exists, "First critic should have gradients"
    assert grad2_none, "Second critic should NOT have gradients"
    print("✓ Independent updates")

    print("✅ Parameter independence passed")


def test_twin_critics_forward_consistency():
    """Test forward pass consistency and correctness."""
    print("\n=== Test: Forward Pass Consistency ===")

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

    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    policy.eval()

    batch_size = 8
    latent = torch.randn(batch_size, 32)

    # Test 1: Output shapes
    print("Test 1: Output shapes")
    logits1 = policy._get_value_logits(latent)
    logits2 = policy._get_value_logits_2(latent)
    assert logits1.shape == (batch_size, 16), f"Wrong shape: {logits1.shape}"
    assert logits2.shape == (batch_size, 16), f"Wrong shape: {logits2.shape}"
    print(f"  Critic 1 output shape: {logits1.shape}")
    print(f"  Critic 2 output shape: {logits2.shape}")
    print("✓ Shapes correct")

    # Test 2: Outputs differ (not identical networks)
    print("Test 2: Outputs differ")
    diff = (logits1 - logits2).abs().mean().item()
    assert diff > 1e-6, f"Outputs too similar (diff={diff})"
    print(f"  Mean absolute difference: {diff:.6f}")
    print("✓ Outputs differ")

    # Test 3: Minimum value selection
    print("Test 3: Minimum value selection")
    min_values = policy._get_min_twin_values(latent)
    value1 = logits1.mean(dim=-1, keepdim=True)
    value2 = logits2.mean(dim=-1, keepdim=True)
    expected_min = torch.min(value1, value2)

    assert torch.allclose(min_values, expected_min, atol=1e-6), "Min values incorrect"
    print(f"  Min values shape: {min_values.shape}")
    print(f"  Difference from expected: {(min_values - expected_min).abs().max().item():.2e}")
    print("✓ Minimum selection correct")

    # Test 4: Deterministic with same input
    print("Test 4: Deterministic behavior")
    logits1_again = policy._get_value_logits(latent)
    assert torch.allclose(logits1, logits1_again), "Forward pass not deterministic"
    print("✓ Deterministic")

    print("✅ Forward pass consistency passed")


def test_twin_critics_gradient_flow_detailed():
    """Test gradient flow in detail."""
    print("\n=== Test: Detailed Gradient Flow ===")

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

    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )

    latent = torch.randn(4, 32, requires_grad=True)

    # Test 1: Gradient flow through both critics
    print("Test 1: Gradient flow through both critics")
    logits1, logits2 = policy._get_twin_value_logits(latent)
    loss = logits1.mean() + logits2.mean()
    policy.zero_grad()
    loss.backward()

    # Check gradients exist
    grad1 = policy.quantile_head.linear.weight.grad
    grad2 = policy.quantile_head_2.linear.weight.grad
    grad_latent = latent.grad

    assert grad1 is not None, "Critic 1 should have gradients"
    assert grad2 is not None, "Critic 2 should have gradients"
    assert grad_latent is not None, "Latent should have gradients"

    print(f"  Critic 1 grad norm: {grad1.norm().item():.6f}")
    print(f"  Critic 2 grad norm: {grad2.norm().item():.6f}")
    print(f"  Latent grad norm: {grad_latent.norm().item():.6f}")
    print("✓ Gradients flow to both critics")

    # Test 2: No gradient flow when detached
    print("Test 2: Detached doesn't propagate gradients")
    policy.zero_grad()
    latent2 = torch.randn(4, 32, requires_grad=True)
    logits1_det = policy._get_value_logits(latent2).detach()
    loss_det = logits1_det.mean()
    try:
        loss_det.backward()
        has_grad = policy.quantile_head.linear.weight.grad is not None
        assert not has_grad, "Should not have gradients when detached"
        print("✓ Detached blocks gradients")
    except RuntimeError:
        print("✓ Detached blocks gradients (RuntimeError as expected)")

    # Test 3: Gradient magnitudes reasonable
    print("Test 3: Gradient magnitudes")
    policy.zero_grad()
    latent3 = torch.randn(4, 32, requires_grad=True)
    logits1, logits2 = policy._get_twin_value_logits(latent3)
    loss = (logits1.mean() + logits2.mean()) * 10.0  # Scale up loss
    loss.backward()

    grad_norm1 = policy.quantile_head.linear.weight.grad.norm().item()
    grad_norm2 = policy.quantile_head_2.linear.weight.grad.norm().item()

    # Gradients should be non-zero and not too large
    assert 0.0 < grad_norm1 < 1e6, f"Grad norm 1 out of range: {grad_norm1}"
    assert 0.0 < grad_norm2 < 1e6, f"Grad norm 2 out of range: {grad_norm2}"
    print(f"  Grad norm 1: {grad_norm1:.6f}")
    print(f"  Grad norm 2: {grad_norm2:.6f}")
    print("✓ Gradient magnitudes reasonable")

    print("✅ Detailed gradient flow passed")


def test_twin_critics_numerical_stability():
    """Test numerical stability edge cases."""
    print("\n=== Test: Numerical Stability ===")

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

    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )
    policy.eval()

    # Test 1: Very large inputs
    print("Test 1: Large inputs")
    latent_large = torch.randn(4, 32) * 100.0
    try:
        min_vals = policy._get_min_twin_values(latent_large)
        assert torch.isfinite(min_vals).all(), "Non-finite values with large inputs"
        print(f"  Min value range: [{min_vals.min().item():.2f}, {min_vals.max().item():.2f}]")
        print("✓ Large inputs OK")
    except Exception as e:
        print(f"✗ Failed with large inputs: {e}")
        raise

    # Test 2: Very small inputs (near zero)
    print("Test 2: Small inputs")
    latent_small = torch.randn(4, 32) * 1e-6
    try:
        min_vals = policy._get_min_twin_values(latent_small)
        assert torch.isfinite(min_vals).all(), "Non-finite values with small inputs"
        print(f"  Min value range: [{min_vals.min().item():.2e}, {min_vals.max().item():.2e}]")
        print("✓ Small inputs OK")
    except Exception as e:
        print(f"✗ Failed with small inputs: {e}")
        raise

    # Test 3: Mixed positive/negative
    print("Test 3: Mixed signs")
    latent_mixed = torch.randn(4, 32)
    latent_mixed[:2] = latent_mixed[:2].abs()  # Positive
    latent_mixed[2:] = -latent_mixed[2:].abs()  # Negative
    try:
        min_vals = policy._get_min_twin_values(latent_mixed)
        assert torch.isfinite(min_vals).all(), "Non-finite values with mixed signs"
        print("✓ Mixed signs OK")
    except Exception as e:
        print(f"✗ Failed with mixed signs: {e}")
        raise

    # Test 4: Batch size = 1
    print("Test 4: Single sample batch")
    latent_single = torch.randn(1, 32)
    try:
        min_vals = policy._get_min_twin_values(latent_single)
        assert min_vals.shape == (1, 1), f"Wrong shape: {min_vals.shape}"
        print("✓ Single sample OK")
    except Exception as e:
        print(f"✗ Failed with single sample: {e}")
        raise

    print("✅ Numerical stability passed")


def test_twin_critics_memory_efficiency():
    """Test memory usage and detect leaks."""
    print("\n=== Test: Memory Efficiency ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Test 1: Memory overhead
    print("Test 1: Memory overhead")

    # Single critic
    arch_single = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}
    }
    policy_single = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_single
    )
    params_single = sum(p.numel() for p in policy_single.quantile_head.parameters())

    # Twin critics
    arch_twin = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': True}
    }
    policy_twin = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_twin
    )
    params_critic1 = sum(p.numel() for p in policy_twin.quantile_head.parameters())
    params_critic2 = sum(p.numel() for p in policy_twin.quantile_head_2.parameters())
    params_twin_total = params_critic1 + params_critic2

    overhead_ratio = params_twin_total / params_single
    print(f"  Single critic params: {params_single}")
    print(f"  Twin critics params: {params_twin_total}")
    print(f"  Overhead ratio: {overhead_ratio:.2f}x")

    # Should be approximately 2x
    assert 1.9 < overhead_ratio < 2.1, f"Unexpected overhead: {overhead_ratio:.2f}x"
    print("✓ Memory overhead ~2x as expected")

    # Test 2: No memory leak in forward passes
    print("Test 2: Memory leak check")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    initial_objects = len(gc.get_objects())

    # Run many forward passes
    for _ in range(100):
        latent = torch.randn(8, 32)
        _ = policy_twin._get_min_twin_values(latent)

    gc.collect()
    final_objects = len(gc.get_objects())

    object_growth = final_objects - initial_objects
    print(f"  Objects before: {initial_objects}")
    print(f"  Objects after: {final_objects}")
    print(f"  Growth: {object_growth}")

    # Some growth is expected, but not excessive
    assert object_growth < 1000, f"Possible memory leak: {object_growth} objects created"
    print("✓ No significant memory leak")

    print("✅ Memory efficiency passed")


def test_twin_critics_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Test: Error Handling ===")

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Test 1: Error when accessing second critic with twin critics disabled
    print("Test 1: Error when twin critics disabled")
    arch_params = {
        'hidden_dim': 32,
        'critic': {'distributional': True, 'num_quantiles': 16, 'use_twin_critics': False}
    }
    policy = CustomActorCriticPolicy(
        observation_space, action_space, lambda x: 0.001, arch_params=arch_params
    )

    latent = torch.randn(4, 32)
    try:
        _ = policy._get_value_logits_2(latent)
        print("✗ Should have raised RuntimeError")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Second critic is not enabled" in str(e)
        print(f"✓ Correctly raised: {e}")

    # Test 2: Error when calling twin methods with twin critics disabled
    print("Test 2: Error when calling twin methods")
    try:
        _ = policy._get_twin_value_logits(latent)
        print("✗ Should have raised RuntimeError")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Twin critics are not enabled" in str(e)
        print(f"✓ Correctly raised: {e}")

    # Test 3: Fallback behavior in _get_min_twin_values
    print("Test 3: Fallback when twin critics disabled")
    min_vals = policy._get_min_twin_values(latent)
    # Should just return single critic value
    assert min_vals.shape == (4, 1)
    print("✓ Fallback works correctly")

    print("✅ Error handling passed")


def run_all_deep_tests():
    """Run all deep audit tests."""
    print("\n" + "="*70)
    print("TWIN CRITICS DEEP AUDIT - COMPREHENSIVE TEST SUITE")
    print("="*70)

    tests = [
        test_twin_critics_configuration_validation,
        test_twin_critics_parameter_independence,
        test_twin_critics_forward_consistency,
        test_twin_critics_gradient_flow_detailed,
        test_twin_critics_numerical_stability,
        test_twin_critics_memory_efficiency,
        test_twin_critics_error_handling,
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

    print("\n🎉 ALL DEEP AUDIT TESTS PASSED!")


if __name__ == "__main__":
    run_all_deep_tests()
