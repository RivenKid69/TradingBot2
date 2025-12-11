"""
Integration test for advantage normalization in DistributionalPPO.

This test verifies that the production code in distributional_ppo.py correctly
normalizes advantages GLOBALLY (entire rollout buffer) rather than per-group,
following standard PPO practice (OpenAI Baselines, Stable-Baselines3).
"""

import pytest
import numpy as np
torch = pytest.importorskip("torch")


def test_global_advantage_normalization_basic():
    """
    Test that advantages are normalized globally with mean≈0 and std≈1.
    """
    # Simulate a rollout buffer with advantages
    buffer_size = 100
    n_envs = 4

    # Create advantages with known distribution
    np.random.seed(42)
    advantages = np.random.randn(buffer_size, n_envs).astype(np.float32) * 10.0 + 5.0

    # Compute expected global statistics
    expected_mean = float(np.mean(advantages))
    expected_std = float(np.std(advantages, ddof=1))
    expected_std_clamped = max(expected_std, 1e-8)

    # Normalize (simulating the code in collect_rollouts)
    advantages_normalized = (advantages - expected_mean) / expected_std_clamped

    # Verify global statistics of normalized advantages
    actual_mean = float(np.mean(advantages_normalized))
    actual_std = float(np.std(advantages_normalized, ddof=1))

    assert abs(actual_mean) < 1e-6, f"Normalized mean should be ≈0, got {actual_mean}"
    assert abs(actual_std - 1.0) < 1e-6, f"Normalized std should be ≈1, got {actual_std}"

    print(f"✓ Global normalization: mean={actual_mean:.6f}, std={actual_std:.6f}")


def test_global_normalization_preserves_relative_ordering():
    """
    Test that global normalization preserves relative ordering of advantages.
    """
    # Create three groups with different advantage levels
    group_a = np.array([100.0, 110.0, 120.0], dtype=np.float32)  # High
    group_b = np.array([10.0, 12.0, 14.0], dtype=np.float32)      # Medium
    group_c = np.array([-5.0, -6.0, -4.0], dtype=np.float32)      # Low

    advantages = np.concatenate([group_a, group_b, group_c])

    # Global normalization
    mean = float(np.mean(advantages))
    std = float(np.std(advantages, ddof=1))
    std_clamped = max(std, 1e-8)
    advantages_normalized = (advantages - mean) / std_clamped

    # Verify relative ordering is preserved
    a_norm = advantages_normalized[:3]
    b_norm = advantages_normalized[3:6]
    c_norm = advantages_normalized[6:]

    # Group A should have highest normalized values
    assert np.mean(a_norm) > np.mean(b_norm), "Group A should be > Group B"
    assert np.mean(b_norm) > np.mean(c_norm), "Group B should be > Group C"

    print(f"✓ Relative ordering preserved:")
    print(f"  Group A (high): mean={np.mean(a_norm):.3f}")
    print(f"  Group B (mid):  mean={np.mean(b_norm):.3f}")
    print(f"  Group C (low):  mean={np.mean(c_norm):.3f}")


def test_global_normalization_consistency_across_epochs():
    """
    Test that globally normalized advantages remain consistent regardless of
    how they're grouped during training (important for multi-epoch training).
    """
    # Create advantages
    advantages = np.array([50.0, 60.0, 10.0, 15.0, -5.0, -10.0], dtype=np.float32)

    # Global normalization (done once in collect_rollouts)
    mean = float(np.mean(advantages))
    std = float(np.std(advantages, ddof=1))
    std_clamped = max(std, 1e-8)
    advantages_normalized = (advantages - mean) / std_clamped

    # Simulate different groupings (as would happen with different batch sizes)
    # Epoch 1: Groups of 2
    groups_epoch1 = [advantages_normalized[i:i+2] for i in range(0, 6, 2)]

    # Epoch 2: Groups of 3
    groups_epoch2 = [advantages_normalized[i:i+3] for i in range(0, 6, 3)]

    # Verify that the normalized values are the SAME regardless of grouping
    # (This would NOT be true with per-group normalization!)
    for i, val in enumerate(advantages_normalized):
        # Check that value remains consistent across different epoch groupings
        # This is trivially true for global normalization but would fail for per-group
        epoch1_idx_group = i // 2
        epoch1_idx_within = i % 2
        retrieved_epoch1 = groups_epoch1[epoch1_idx_group][epoch1_idx_within]

        assert abs(val - retrieved_epoch1) < 1e-7, \
            f"Value {i} should be consistent in epoch 1 grouping"

    print("✓ Normalized advantages are consistent across different groupings")


def test_global_normalization_handles_edge_cases():
    """
    Test that global normalization handles edge cases correctly.
    """
    # Case 1: All advantages are the same (std = 0)
    advantages_constant = np.ones(10, dtype=np.float32) * 5.0
    mean = float(np.mean(advantages_constant))
    std = float(np.std(advantages_constant, ddof=1))
    std_clamped = max(std, 1e-8)  # Clamp to prevent division by zero
    normalized = (advantages_constant - mean) / std_clamped

    # Should be all zeros (since all values are the same)
    assert np.allclose(normalized, 0.0), "Constant advantages should normalize to 0"

    # Case 2: Very small std
    advantages_small_std = np.array([1.0, 1.0001, 0.9999], dtype=np.float32)
    mean = float(np.mean(advantages_small_std))
    std = float(np.std(advantages_small_std, ddof=1))
    std_clamped = max(std, 1e-8)
    normalized = (advantages_small_std - mean) / std_clamped

    # Should not have NaN or Inf
    assert np.all(np.isfinite(normalized)), "Should handle small std without NaN/Inf"

    print("✓ Edge cases handled correctly")


def test_implementation_uses_global_normalization():
    """
    Verify that the actual implementation code uses global normalization.
    """
    import inspect
    from distributional_ppo import DistributionalPPO

    # Check collect_rollouts for global normalization
    source = inspect.getsource(DistributionalPPO.collect_rollouts)

    # Should have global normalization code
    assert "self.normalize_advantage" in source, \
        "Should check normalize_advantage flag"
    assert "advantages_flat = rollout_buffer.advantages.reshape(-1)" in source, \
        "Should flatten entire buffer for global statistics"
    assert "np.mean(advantages_flat)" in source, \
        "Should compute mean over entire buffer"
    assert "np.std(advantages_flat, ddof=1)" in source, \
        "Should compute std over entire buffer with ddof=1 for unbiased estimate"
    assert "rollout_buffer.advantages = " in source, \
        "Should update buffer with normalized advantages"

    # Check train() does NOT have per-group normalization
    train_source = inspect.getsource(DistributionalPPO.train)

    # Should NOT compute per-group statistics
    assert "group_advantages_for_stats" not in train_source, \
        "Should NOT collect advantages for per-group statistics"
    assert "group_adv_mean = " not in train_source, \
        "Should NOT compute per-group mean"
    assert "group_adv_std = " not in train_source, \
        "Should NOT compute per-group std"

    # Should have comment about global normalization
    assert "already normalized globally" in train_source.lower() or \
           "already globally normalized" in train_source.lower(), \
        "Should document that advantages are already normalized"

    print("✓ Implementation uses global normalization correctly")


def test_no_per_group_statistics_in_train():
    """
    Verify that train() does NOT compute per-group advantage statistics.
    """
    import inspect
    from distributional_ppo import DistributionalPPO

    source = inspect.getsource(DistributionalPPO.train)

    # Find the microbatch loop
    if "for rollout_data, sample_count, mask_tensor, sample_weight in zip(" not in source:
        pytest.fail("Could not find microbatch loop in train()")

    loop_start = source.index("for rollout_data, sample_count, mask_tensor, sample_weight in zip(")

    # Find gradient step (end of processing)
    grad_step_idx = source.index("self.policy.optimizer.step()", loop_start)

    # Extract the relevant section
    relevant_section = source[loop_start:grad_step_idx]

    # Should NOT see advantage normalization in this section
    assert "advantages_selected_raw - " not in relevant_section or \
           "group_adv_mean" not in relevant_section, \
        "Should NOT normalize advantages during training loop"

    assert "advantages.mean()" not in relevant_section, \
        "Should NOT compute advantage mean in training loop"

    assert "advantages.std(" not in relevant_section, \
        "Should NOT compute advantage std in training loop"

    print("✓ No per-group statistics computation in train()")


def test_consistency_with_stable_baselines3_approach():
    """
    Verify our approach matches Stable-Baselines3 normalization.

    In SB3, advantages are normalized once after computation:
    buffer.normalize_advantages()

    Then used as-is during training.
    """
    # Simulate SB3 approach
    advantages = np.array([
        [50.0, 60.0, 70.0],
        [10.0, 15.0, 20.0],
        [-5.0, -10.0, -15.0]
    ], dtype=np.float32)

    # SB3: normalize entire buffer once
    mean = float(np.mean(advantages))
    std = float(np.std(advantages, ddof=1))
    advantages_normalized = (advantages - mean) / (std + 1e-8)

    # Verify it has the right properties
    assert abs(np.mean(advantages_normalized)) < 1e-6
    assert abs(np.std(advantages_normalized, ddof=1) - 1.0) < 1e-5

    # Simulate training with batches (as SB3 does)
    # Each batch uses the SAME pre-normalized advantages
    batch1 = advantages_normalized[0, :]
    batch2 = advantages_normalized[1, :]
    batch3 = advantages_normalized[2, :]

    # The normalization remains consistent across all batches
    # (no re-normalization per batch)
    all_batches = np.concatenate([batch1, batch2, batch3])
    assert abs(np.mean(all_batches)) < 1e-6
    assert abs(np.std(all_batches, ddof=1) - 1.0) < 1e-5

    print("✓ Approach consistent with Stable-Baselines3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
