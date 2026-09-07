"""
Simple standalone test for advantage normalization (no pytest dependency).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import inspect


def test_global_advantage_normalization_basic():
    """Test that advantages are normalized globally with mean≈0 and std≈1."""
    print("\n[TEST] Global advantage normalization basic...")

    # Simulate a rollout buffer with advantages
    buffer_size = 100
    n_envs = 4

    # Create advantages with known distribution
    np.random.seed(42)
    advantages = np.random.randn(buffer_size, n_envs).astype(np.float32) * 10.0 + 5.0

    # Compute expected global statistics (using ddof=1 for sample std)
    expected_mean = float(np.mean(advantages))
    expected_std = float(np.std(advantages, ddof=1))
    expected_std_clamped = max(expected_std, 1e-8)

    # Normalize (simulating the code in collect_rollouts)
    advantages_normalized = (advantages - expected_mean) / expected_std_clamped

    # Verify global statistics of normalized advantages (using ddof=1)
    actual_mean = float(np.mean(advantages_normalized))
    actual_std = float(np.std(advantages_normalized, ddof=1))

    assert abs(actual_mean) < 1e-6, f"Normalized mean should be ≈0, got {actual_mean}"
    assert abs(actual_std - 1.0) < 1e-6, f"Normalized std should be ≈1, got {actual_std}"

    print(f"  ✓ Global normalization: mean={actual_mean:.6f}, std={actual_std:.6f}")


def test_global_normalization_preserves_relative_ordering():
    """Test that global normalization preserves relative ordering of advantages."""
    print("\n[TEST] Global normalization preserves relative ordering...")

    # Create three groups with different advantage levels
    group_a = np.array([100.0, 110.0, 120.0], dtype=np.float32)  # High
    group_b = np.array([10.0, 12.0, 14.0], dtype=np.float32)  # Medium
    group_c = np.array([-5.0, -6.0, -4.0], dtype=np.float32)  # Low

    advantages = np.concatenate([group_a, group_b, group_c])

    # Global normalization (using ddof=1 for sample std)
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

    print(f"  ✓ Relative ordering preserved:")
    print(f"    Group A (high): mean={np.mean(a_norm):.3f}")
    print(f"    Group B (mid):  mean={np.mean(b_norm):.3f}")
    print(f"    Group C (low):  mean={np.mean(c_norm):.3f}")


def test_implementation_uses_global_normalization():
    """Verify that the actual implementation code uses global normalization."""
    print("\n[TEST] Implementation uses global normalization...")

    from distributional_ppo import DistributionalPPO

    # Check collect_rollouts for global normalization
    source = inspect.getsource(DistributionalPPO.collect_rollouts)

    # Should have global normalization code
    assert "self.normalize_advantage" in source, "Should check normalize_advantage flag"
    assert (
        "advantages_flat = rollout_buffer.advantages.reshape(-1)" in source
    ), "Should flatten entire buffer for global statistics"
    assert "np.mean(advantages_flat)" in source, "Should compute mean over entire buffer"
    assert (
        "np.std(advantages_flat, ddof=1)" in source
    ), "Should compute std over entire buffer with ddof=1 for unbiased estimate"
    assert (
        "rollout_buffer.advantages = " in source
    ), "Should update buffer with normalized advantages"

    print("  ✓ collect_rollouts() has global normalization")

    # Check train() does NOT have per-group normalization
    train_source = inspect.getsource(DistributionalPPO.train)

    # Should NOT compute per-group statistics
    assert (
        "group_advantages_for_stats" not in train_source
    ), "Should NOT collect advantages for per-group statistics"
    assert "group_adv_mean = " not in train_source, "Should NOT compute per-group mean"
    assert "group_adv_std = " not in train_source, "Should NOT compute per-group std"

    # Should have comment about global normalization
    assert (
        "already normalized globally" in train_source.lower()
        or "already globally normalized" in train_source.lower()
    ), "Should document that advantages are already normalized"

    print("  ✓ train() does NOT use per-group normalization")
    print("  ✓ Implementation uses global normalization correctly")


def test_no_per_group_statistics_in_train():
    """Verify that train() does NOT compute per-group advantage statistics."""
    print("\n[TEST] No per-group statistics in train()...")

    from distributional_ppo import DistributionalPPO

    source = inspect.getsource(DistributionalPPO.train)

    # Find the microbatch loop
    if "for rollout_data, sample_count, mask_tensor, sample_weight in zip(" not in source:
        raise AssertionError("Could not find microbatch loop in train()")

    loop_start = source.index("for rollout_data, sample_count, mask_tensor, sample_weight in zip(")

    # Find gradient step (end of processing)
    grad_step_idx = source.index("self.policy.optimizer.step()", loop_start)

    # Extract the relevant section
    relevant_section = source[loop_start:grad_step_idx]

    # Should NOT see advantage normalization in this section
    assert (
        "advantages_selected_raw - " not in relevant_section
        or "group_adv_mean" not in relevant_section
    ), "Should NOT normalize advantages during training loop"

    assert (
        "advantages.mean()" not in relevant_section
    ), "Should NOT compute advantage mean in training loop"

    assert (
        "advantages.std(" not in relevant_section
    ), "Should NOT compute advantage std in training loop"

    print("  ✓ No per-group statistics computation in train()")


def main():
    print("=" * 70)
    print("Testing Global Advantage Normalization Implementation")
    print("=" * 70)

    tests = [
        test_global_advantage_normalization_basic,
        test_global_normalization_preserves_relative_ordering,
        test_implementation_uses_global_normalization,
        test_no_per_group_statistics_in_train,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
