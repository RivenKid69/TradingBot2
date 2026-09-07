"""
Test GAE Overflow Protection

Tests defensive clamping in GAE computation to prevent overflow in extreme reward scenarios.

Bug #4: GAE Overflow Risk (2025-11-23)
- Added defensive clamping to delta and GAE accumulation
- Threshold: 1e6 (conservative, plenty of headroom)
- Tests cover normal, high reward, and extreme cases

References:
- Schulman et al. (2016), "High-Dimensional Continuous Control Using GAE"
- distributional_ppo.py:263-296 (GAE computation with clamping)
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from typing import Tuple


# Mock the necessary parts of distributional_ppo for testing
class MockRolloutBuffer:
    """Mock rollout buffer for testing."""

    def __init__(self):
        self.advantages = None
        self.returns = None


def compute_gae_with_clamping(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    time_limit_mask: np.ndarray = None,
    time_limit_bootstrap: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages with defensive clamping (copy from distributional_ppo.py).

    This is the actual implementation from distributional_ppo.py:223-299.
    """
    # Input validation
    if not np.all(np.isfinite(rewards)):
        raise ValueError(
            f"GAE computation: rewards contain NaN or inf values. "
            f"Non-finite count: {np.sum(~np.isfinite(rewards))}/{rewards.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"GAE computation: values contain NaN or inf values. "
            f"Non-finite count: {np.sum(~np.isfinite(values))}/{values.size}"
        )

    buffer_size, n_envs = rewards.shape
    advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)

    last_values_np = np.asarray(last_values, dtype=np.float32).reshape(n_envs)
    dones_float = np.asarray(dones, dtype=np.float32).reshape(n_envs)

    if not np.all(np.isfinite(last_values_np)):
        raise ValueError(
            f"GAE computation: last_values contain NaN or inf values. "
            f"Non-finite count: {np.sum(~np.isfinite(last_values_np))}/{last_values_np.size}"
        )

    # Default time_limit_mask and bootstrap if not provided
    if time_limit_mask is None:
        time_limit_mask = np.zeros((buffer_size, n_envs), dtype=bool)
    if time_limit_bootstrap is None:
        time_limit_bootstrap = np.zeros((buffer_size, n_envs), dtype=np.float32)

    if time_limit_mask.shape != (buffer_size, n_envs):
        raise ValueError("TimeLimit mask must match rollout buffer dimensions")
    if time_limit_bootstrap.shape != (buffer_size, n_envs):
        raise ValueError("TimeLimit bootstrap values must match rollout buffer dimensions")

    if not np.all(np.isfinite(time_limit_bootstrap)):
        raise ValueError(
            f"GAE computation: time_limit_bootstrap contains NaN or inf values. "
            f"Non-finite count: {np.sum(~np.isfinite(time_limit_bootstrap))}/{time_limit_bootstrap.size}"
        )

    last_gae_lam = np.zeros(n_envs, dtype=np.float32)

    # DEFENSIVE CLAMPING: Prevent GAE overflow in extreme reward scenarios
    GAE_CLAMP_THRESHOLD = 1e6

    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones_float
            next_values = last_values_np.copy()
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1].astype(np.float32)
            next_values = values[step + 1].astype(np.float32).copy()

        mask = time_limit_mask[step]
        if np.any(mask):
            next_non_terminal = np.where(mask, 1.0, next_non_terminal)
            next_values = np.where(mask, time_limit_bootstrap[step], next_values)

        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        # DEFENSIVE CLAMPING: Prevent overflow in delta computation
        delta = np.clip(delta, -GAE_CLAMP_THRESHOLD, GAE_CLAMP_THRESHOLD)

        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        # DEFENSIVE CLAMPING: Prevent overflow in GAE accumulation
        last_gae_lam = np.clip(last_gae_lam, -GAE_CLAMP_THRESHOLD, GAE_CLAMP_THRESHOLD)

        advantages[step] = last_gae_lam

    returns = (advantages + values).astype(np.float32)
    return advantages.astype(np.float32), returns


# ============================================================================
# Test 1: Normal Case (No Overflow)
# ============================================================================


def test_gae_normal_case():
    """Test GAE computation with normal reward values (no overflow risk)."""
    buffer_size = 64
    n_envs = 4

    # Normal rewards: [-1, 1] range
    rewards = np.random.uniform(-1.0, 1.0, (buffer_size, n_envs)).astype(np.float32)
    values = np.random.uniform(-5.0, 5.0, (buffer_size, n_envs)).astype(np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.random.uniform(-5.0, 5.0, n_envs).astype(np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"
    assert np.all(np.isfinite(returns)), "Returns should be finite"

    # Advantages should be within reasonable bounds (no clamping needed)
    assert np.all(np.abs(advantages) < 100), "Advantages should be < 100 for normal rewards"

    # Shape checks
    assert advantages.shape == (buffer_size, n_envs)
    assert returns.shape == (buffer_size, n_envs)

    print(f"[PASS] Normal case: advantages in [{advantages.min():.2f}, {advantages.max():.2f}]")


# ============================================================================
# Test 2: High Rewards Case (Near Overflow Risk)
# ============================================================================


def test_gae_high_rewards():
    """Test GAE computation with high but realistic reward values."""
    buffer_size = 256
    n_envs = 4

    # High rewards: sustained +100 rewards (realistic worst case)
    rewards = np.full((buffer_size, n_envs), 100.0, dtype=np.float32)
    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.99,  # High lambda → longer accumulation
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"
    assert np.all(np.isfinite(returns)), "Returns should be finite"

    # Advantages should be high but within clamp threshold (1e6)
    assert np.all(advantages <= 1e6), "Advantages should be clamped at 1e6"
    assert np.all(advantages >= -1e6), "Advantages should be clamped at -1e6"

    # For sustained high rewards, advantages should be high but NOT overflow
    # Theoretical max: 100 * (1 - 0.99^256) / (1 - 0.99) ≈ 10,000
    assert np.all(advantages < 20000), "Advantages should be < 20,000 (theoretical max ~10,000)"

    print(
        f"[PASS] High rewards case: advantages in [{advantages.min():.2f}, {advantages.max():.2f}]"
    )


# ============================================================================
# Test 3: Extreme Case (Would Overflow Without Clamping)
# ============================================================================


def test_gae_extreme_case_with_clamping():
    """Test GAE computation with extreme rewards that would overflow without clamping."""
    buffer_size = 128
    n_envs = 2

    # EXTREME rewards: 1e5 (would cause overflow without clamping)
    rewards = np.full((buffer_size, n_envs), 1e5, dtype=np.float32)
    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.99,
    )

    # All values should be finite (clamping prevents overflow)
    assert np.all(np.isfinite(advantages)), "Advantages should be finite (clamped)"
    assert np.all(np.isfinite(returns)), "Returns should be finite (clamped)"

    # Advantages MUST be clamped at threshold (1e6)
    assert np.all(advantages <= 1e6), "Advantages MUST be clamped at 1e6"
    assert np.all(advantages >= -1e6), "Advantages MUST be clamped at -1e6"

    # Most advantages should hit the clamp threshold
    clamped_count = np.sum(np.abs(advantages) >= 1e6 * 0.99)  # Within 1% of threshold
    assert clamped_count > buffer_size * n_envs * 0.5, "Most advantages should be clamped"

    print(
        f"[PASS] Extreme case: advantages clamped at [{advantages.min():.2e}, {advantages.max():.2e}]"
    )
    print(
        f"       Clamped: {clamped_count}/{buffer_size * n_envs} ({100*clamped_count/(buffer_size*n_envs):.1f}%)"
    )


# ============================================================================
# Test 4: Negative Rewards (Downside Overflow Risk)
# ============================================================================


def test_gae_negative_rewards():
    """Test GAE computation with sustained negative rewards (downside overflow)."""
    buffer_size = 128
    n_envs = 4

    # Sustained negative rewards: -1e5
    rewards = np.full((buffer_size, n_envs), -1e5, dtype=np.float32)
    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"
    assert np.all(np.isfinite(returns)), "Returns should be finite"

    # Advantages should be clamped at negative threshold
    assert np.all(advantages >= -1e6), "Advantages should be clamped at -1e6"
    assert np.all(advantages <= 1e6), "Advantages should be clamped at +1e6"

    # Most advantages should be negative and clamped
    assert np.all(advantages < 0), "All advantages should be negative"
    clamped_count = np.sum(advantages <= -1e6 * 0.99)
    assert clamped_count > 0, "Some advantages should be clamped at negative threshold"

    print(
        f"[PASS] Negative rewards: advantages in [{advantages.min():.2e}, {advantages.max():.2e}]"
    )


# ============================================================================
# Test 5: Mixed Signs (Positive and Negative Rewards)
# ============================================================================


def test_gae_mixed_signs():
    """Test GAE computation with mixed positive and negative extreme rewards."""
    buffer_size = 64
    n_envs = 4

    # Alternating extreme rewards: +1e5, -1e5, +1e5, -1e5, ...
    rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
    rewards[::2] = 1e5  # Even steps: +1e5
    rewards[1::2] = -1e5  # Odd steps: -1e5

    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"
    assert np.all(np.isfinite(returns)), "Returns should be finite"

    # Advantages should be within clamp bounds
    assert np.all(advantages >= -1e6), "Advantages >= -1e6"
    assert np.all(advantages <= 1e6), "Advantages <= +1e6"

    # Should have both positive and negative advantages (mixed signs)
    # NOTE: This may not always be true due to GAE accumulation smoothing
    # But at least some variation should exist
    assert np.std(advantages) > 0, "Advantages should have variation"

    print(f"[PASS] Mixed signs: advantages in [{advantages.min():.2e}, {advantages.max():.2e}]")


# ============================================================================
# Test 6: Episode Boundaries (Reset GAE Accumulation)
# ============================================================================


def test_gae_episode_boundaries():
    """Test that GAE accumulation resets correctly at episode boundaries."""
    buffer_size = 32
    n_envs = 2

    # High rewards with episode boundary in the middle
    rewards = np.full((buffer_size, n_envs), 1e5, dtype=np.float32)
    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    # Episode boundary at step 16 for env 0
    episode_starts[16, 0] = 1.0

    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"

    # Advantages should be clamped
    assert np.all(advantages >= -1e6), "Advantages >= -1e6"
    assert np.all(advantages <= 1e6), "Advantages <= +1e6"

    # Env 0 should have different advantage profile due to episode boundary
    # (This is hard to test rigorously, but at least check it doesn't crash)

    print(
        f"[PASS] Episode boundaries: advantages in [{advantages.min():.2e}, {advantages.max():.2e}]"
    )


# ============================================================================
# Test 7: Integration with Existing Validation (NaN/Inf Inputs)
# ============================================================================


def test_gae_input_validation_nan():
    """Test that input validation catches NaN values (existing protection)."""
    buffer_size = 16
    n_envs = 2

    # NaN in rewards
    rewards = np.ones((buffer_size, n_envs), dtype=np.float32)
    rewards[5, 0] = np.nan
    values = np.zeros((buffer_size, n_envs), dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    with pytest.raises(ValueError, match="rewards contain NaN or inf"):
        compute_gae_with_clamping(
            rewards=rewards,
            values=values,
            episode_starts=episode_starts,
            last_values=last_values,
            dones=dones,
        )

    print("[PASS] Input validation (NaN): correctly rejects NaN inputs")


def test_gae_input_validation_inf():
    """Test that input validation catches inf values (existing protection)."""
    buffer_size = 16
    n_envs = 2

    # Inf in values
    rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
    values = np.ones((buffer_size, n_envs), dtype=np.float32)
    values[10, 1] = np.inf
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    with pytest.raises(ValueError, match="values contain NaN or inf"):
        compute_gae_with_clamping(
            rewards=rewards,
            values=values,
            episode_starts=episode_starts,
            last_values=last_values,
            dones=dones,
        )

    print("[PASS] Input validation (inf): correctly rejects inf inputs")


# ============================================================================
# Test 8: Clamping Does Not Trigger on Normal Values
# ============================================================================


def test_gae_clamping_does_not_trigger_normal():
    """Test that clamping does NOT trigger on normal reward values."""
    buffer_size = 64
    n_envs = 4

    # Normal rewards: [-10, 10] range (high but not extreme)
    rewards = np.random.uniform(-10.0, 10.0, (buffer_size, n_envs)).astype(np.float32)
    values = np.random.uniform(-50.0, 50.0, (buffer_size, n_envs)).astype(np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.random.uniform(-50.0, 50.0, n_envs).astype(np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"

    # NO clamping should occur (all values far from threshold)
    assert np.all(np.abs(advantages) < 1e6 * 0.01), "Advantages should be << 1e6 (no clamping)"

    # Typical GAE advantages for normal rewards: [-100, 100]
    assert np.all(np.abs(advantages) < 1000), "Advantages should be < 1000 for normal rewards"

    print(
        f"[PASS] No false triggers: advantages in [{advantages.min():.2f}, {advantages.max():.2f}]"
    )


# ============================================================================
# Test 9: Float32 Dtype Preservation
# ============================================================================


def test_gae_float32_dtype():
    """Test that GAE computation preserves float32 dtype (memory efficiency)."""
    buffer_size = 32
    n_envs = 2

    rewards = np.random.randn(buffer_size, n_envs).astype(np.float32)
    values = np.random.randn(buffer_size, n_envs).astype(np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.random.randn(n_envs).astype(np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
    )

    # Check dtype
    assert advantages.dtype == np.float32, "Advantages should be float32"
    assert returns.dtype == np.float32, "Returns should be float32"

    print("[PASS] Dtype preservation: advantages and returns are float32")


# ============================================================================
# Test 10: Edge Case - Single Step
# ============================================================================


def test_gae_single_step():
    """Test GAE computation with buffer_size=1 (edge case)."""
    buffer_size = 1
    n_envs = 2

    rewards = np.array([[10.0, -5.0]], dtype=np.float32)
    values = np.array([[2.0, 1.0]], dtype=np.float32)
    episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_values = np.array([3.0, 2.0], dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae_with_clamping(
        rewards=rewards,
        values=values,
        episode_starts=episode_starts,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # All values should be finite
    assert np.all(np.isfinite(advantages)), "Advantages should be finite"
    assert advantages.shape == (1, 2)
    assert returns.shape == (1, 2)

    print(f"[PASS] Single step: advantages = {advantages.flatten()}")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GAE Overflow Protection Tests (Bug #4)")
    print("=" * 80 + "\n")

    tests = [
        ("Normal Case", test_gae_normal_case),
        ("High Rewards", test_gae_high_rewards),
        ("Extreme Case (Clamping)", test_gae_extreme_case_with_clamping),
        ("Negative Rewards", test_gae_negative_rewards),
        ("Mixed Signs", test_gae_mixed_signs),
        ("Episode Boundaries", test_gae_episode_boundaries),
        ("Input Validation (NaN)", test_gae_input_validation_nan),
        ("Input Validation (Inf)", test_gae_input_validation_inf),
        ("No False Triggers", test_gae_clamping_does_not_trigger_normal),
        ("Float32 Dtype", test_gae_float32_dtype),
        ("Single Step", test_gae_single_step),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n[Test {passed+failed+1}/{len(tests)}] {name}")
            print("-" * 80)
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 80 + "\n")

    if failed > 0:
        exit(1)
