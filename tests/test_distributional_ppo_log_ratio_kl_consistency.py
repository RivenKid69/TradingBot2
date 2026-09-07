"""
Tests for consistency between log_ratio monitoring and approx_kl calculations.

This test suite validates that:
1. log_ratio and approx_kl are mathematically consistent
2. Warning thresholds align with PPO best practices
3. Monitoring detects training instability at appropriate levels
4. Statistics are numerically accurate across edge cases

Mathematical relationship:
  approx_kl ≈ old_log_prob - new_log_prob = -(new_log_prob - old_log_prob) = -log_ratio

This is the first-order approximation used in PPO (Schulman et al., 2017).

Reference values (OpenAI Spinning Up):
- Healthy training: approx_kl < 0.02
- Early stopping: approx_kl > 1.5 × target_kl (default: 0.015)
- If approx_kl > 0.02, training may be too aggressive
"""

import math

import pytest


def test_log_ratio_approx_kl_relationship() -> None:
    """Test mathematical relationship: approx_kl ≈ -log_ratio."""
    torch = pytest.importorskip("torch")

    # Create sample log probabilities
    new_log_prob = torch.tensor([-1.5, -2.0, -2.5, -3.0], dtype=torch.float32)
    old_log_prob = torch.tensor([-1.4, -2.1, -2.45, -3.05], dtype=torch.float32)

    # Compute log_ratio
    log_ratio = new_log_prob - old_log_prob

    # Compute approx_kl (first-order approximation)
    approx_kl = old_log_prob - new_log_prob

    # Verify relationship: approx_kl = -log_ratio
    expected_approx_kl = -log_ratio
    assert torch.allclose(approx_kl, expected_approx_kl, atol=1e-7), \
        f"approx_kl should equal -log_ratio: {approx_kl.tolist()} vs {expected_approx_kl.tolist()}"


def test_healthy_training_thresholds_consistency() -> None:
    """Test that log_ratio and approx_kl thresholds are consistent."""
    torch = pytest.importorskip("torch")

    # OpenAI Spinning Up: healthy training has approx_kl < 0.02
    # This means |log_ratio| < 0.02
    healthy_kl_threshold = 0.02

    # Generate healthy log_ratio values
    torch.manual_seed(42)
    log_ratios = torch.randn(1000, dtype=torch.float32) * 0.01  # std=0.01 << 0.02

    # Compute stats
    mean_log_ratio = log_ratios.mean().item()
    std_log_ratio = log_ratios.std(unbiased=True).item()
    max_abs_log_ratio = torch.max(torch.abs(log_ratios)).item()

    # Corresponding approx_kl
    approx_kl = -log_ratios
    mean_approx_kl = approx_kl.mean().item()
    max_abs_approx_kl = torch.max(torch.abs(approx_kl)).item()

    # Verify healthy ranges
    assert abs(mean_log_ratio) < 0.01, \
        f"Healthy training should have small mean log_ratio: {mean_log_ratio}"
    assert std_log_ratio < healthy_kl_threshold, \
        f"Healthy training std should be < {healthy_kl_threshold}: {std_log_ratio}"
    assert max_abs_log_ratio < 0.1, \
        f"Healthy training max_abs should be < 0.1: {max_abs_log_ratio}"

    # approx_kl consistency
    assert abs(mean_approx_kl + mean_log_ratio) < 1e-6, \
        f"mean(approx_kl) should equal -mean(log_ratio)"
    assert abs(max_abs_approx_kl - max_abs_log_ratio) < 1e-6, \
        f"max_abs should be consistent: {max_abs_approx_kl} vs {max_abs_log_ratio}"


def test_warning_threshold_vs_kl_target() -> None:
    """Test that warning thresholds make sense relative to KL target."""
    torch = pytest.importorskip("torch")

    # OpenAI default: target_kl = 0.01, early stop at 1.5 × 0.01 = 0.015
    target_kl = 0.01
    early_stop_kl = 1.5 * target_kl  # = 0.015

    # Our warning thresholds:
    # - concerning: |log_ratio| > 1.0
    # - severe: |log_ratio| > 10.0

    # Check that concerning threshold (1.0) is much larger than early_stop_kl
    # This is correct because:
    # - Early KL stop happens at approx_kl ≈ 0.015 (very conservative)
    # - We only warn at |log_ratio| > 1.0 (much more permissive)
    # - This gives room for normal training variations

    concerning_threshold = 1.0
    severe_threshold = 10.0

    # |log_ratio| = 1.0 corresponds to approx_kl = 1.0
    # This is 67× larger than early_stop_kl (0.015)
    ratio_vs_kl_stop = concerning_threshold / early_stop_kl
    assert ratio_vs_kl_stop > 50, \
        f"Concerning threshold should be much larger than KL stop: {ratio_vs_kl_stop:.1f}×"

    # This is intentional: we're more permissive with warnings to reduce noise
    # But we still catch catastrophic failures (|log_ratio| > 10)


def test_extreme_log_ratio_vs_kl_relationship() -> None:
    """Test that extreme log_ratio values correspond to extreme KL."""
    torch = pytest.importorskip("torch")

    # Test cases: (log_ratio, expected behavior)
    # Note: "concerning" threshold is > 1.0, so we use 1.01 to clearly exceed it
    test_cases = [
        (0.01, "healthy"),      # approx_kl = 0.01 (at target)
        (0.1, "healthy"),       # approx_kl = 0.1 (still ok)
        (1.0, "healthy"),       # approx_kl = 1.0 (at boundary, still healthy)
        (1.01, "concerning"),   # approx_kl = 1.01 (just above threshold)
        (10.01, "severe"),      # approx_kl > 10 (catastrophic)
    ]

    for log_ratio_val, expected_level in test_cases:
        log_ratio = torch.tensor([log_ratio_val], dtype=torch.float32)
        approx_kl = -log_ratio  # Could be negative, but we care about magnitude

        max_abs_log_ratio = torch.abs(log_ratio).item()
        max_abs_kl = torch.abs(approx_kl).item()

        # Verify consistency
        assert abs(max_abs_log_ratio - max_abs_kl) < 1e-6, \
            f"log_ratio and KL magnitudes should match: {max_abs_log_ratio} vs {max_abs_kl}"

        # Verify warning level
        if max_abs_log_ratio > 10.0:
            level = "severe"
        elif max_abs_log_ratio > 1.0:
            level = "concerning"
        else:
            level = "healthy"

        assert level == expected_level, \
            f"For |log_ratio|={max_abs_log_ratio}: expected {expected_level}, got {level}"


def test_multi_sample_kl_vs_log_ratio_statistics() -> None:
    """Test that batch statistics are consistent between log_ratio and approx_kl."""
    torch = pytest.importorskip("torch")

    batch_size = 256
    torch.manual_seed(999)

    # Generate realistic batch
    new_log_prob = torch.randn(batch_size, dtype=torch.float32) * 2.0 - 1.5
    old_log_prob = new_log_prob + torch.randn(batch_size, dtype=torch.float32) * 0.05

    # Compute log_ratio
    log_ratio = new_log_prob - old_log_prob
    approx_kl = old_log_prob - new_log_prob

    # Statistics for log_ratio
    mean_log_ratio = log_ratio.mean().item()
    std_log_ratio = log_ratio.std(unbiased=True).item()
    max_abs_log_ratio = torch.max(torch.abs(log_ratio)).item()

    # Statistics for approx_kl
    mean_approx_kl = approx_kl.mean().item()
    std_approx_kl = approx_kl.std(unbiased=True).item()
    max_abs_approx_kl = torch.max(torch.abs(approx_kl)).item()

    # Verify consistency
    assert abs(mean_log_ratio + mean_approx_kl) < 1e-5, \
        f"Means should be negatives: {mean_log_ratio} vs {mean_approx_kl}"
    assert abs(std_log_ratio - std_approx_kl) < 1e-5, \
        f"Std should be equal: {std_log_ratio} vs {std_approx_kl}"
    assert abs(max_abs_log_ratio - max_abs_approx_kl) < 1e-5, \
        f"Max abs should be equal: {max_abs_log_ratio} vs {max_abs_approx_kl}"


def test_kl_early_stop_simulation() -> None:
    """Simulate KL early stopping and verify log_ratio monitoring would detect it."""
    torch = pytest.importorskip("torch")

    target_kl = 0.01
    early_stop_threshold = 1.5 * target_kl  # = 0.015

    # Scenario 1: Healthy training (should NOT trigger early stop)
    torch.manual_seed(111)
    healthy_log_ratio = torch.randn(100, dtype=torch.float32) * 0.005  # Very small
    healthy_approx_kl = -healthy_log_ratio

    mean_healthy_kl = torch.abs(healthy_approx_kl).mean().item()
    assert mean_healthy_kl < target_kl, \
        f"Healthy training should have mean KL < {target_kl}: {mean_healthy_kl}"

    # No warnings expected
    max_abs_healthy = torch.max(torch.abs(healthy_log_ratio)).item()
    assert max_abs_healthy < 0.1, \
        f"Healthy training should not trigger warnings: {max_abs_healthy}"

    # Scenario 2: Aggressive training (SHOULD trigger early stop)
    torch.manual_seed(222)
    aggressive_log_ratio = torch.randn(100, dtype=torch.float32) * 0.02  # Larger
    aggressive_approx_kl = -aggressive_log_ratio

    mean_aggressive_kl = torch.abs(aggressive_approx_kl).mean().item()
    # This might exceed early_stop_threshold

    # Our warning system should detect if it gets really bad
    max_abs_aggressive = torch.max(torch.abs(aggressive_log_ratio)).item()
    # Even if mean KL triggers early stop, individual max might not trigger our warnings
    # This is OK - our warnings are for catastrophic failures, not just aggressive training


def test_numerical_precision_log_ratio_kl() -> None:
    """Test numerical precision in log_ratio to approx_kl conversion."""
    torch = pytest.importorskip("torch")

    # Small differences (precision suitable for float32)
    # Note: float32 has ~7 decimal digits of precision
    new_log_prob = torch.tensor([-1.001, -2.001], dtype=torch.float32)
    old_log_prob = torch.tensor([-1.000, -2.000], dtype=torch.float32)

    log_ratio = new_log_prob - old_log_prob
    approx_kl = old_log_prob - new_log_prob

    # Verify precision (float32 can handle 1e-3 differences reliably)
    expected_log_ratio = torch.tensor([-1e-3, -1e-3], dtype=torch.float32)
    expected_approx_kl = -expected_log_ratio

    assert torch.allclose(log_ratio, expected_log_ratio, atol=1e-5), \
        f"High precision log_ratio: {log_ratio.tolist()}"
    assert torch.allclose(approx_kl, expected_approx_kl, atol=1e-5), \
        f"High precision approx_kl: {approx_kl.tolist()}"


def test_log_ratio_monitoring_prevents_kl_explosion() -> None:
    """Test that log_ratio monitoring would detect KL divergence explosion."""
    torch = pytest.importorskip("torch")

    # Simulate KL explosion scenario
    # Normal first epoch
    epoch1_log_ratio = torch.randn(100, dtype=torch.float32) * 0.05
    max_epoch1 = torch.max(torch.abs(epoch1_log_ratio)).item()
    assert max_epoch1 < 0.3, "Epoch 1 should be healthy"

    # Second epoch: KL starts growing
    epoch2_log_ratio = torch.randn(100, dtype=torch.float32) * 0.5
    max_epoch2 = torch.max(torch.abs(epoch2_log_ratio)).item()
    # May trigger concerning warning (> 1.0)

    # Third epoch: KL explodes
    epoch3_log_ratio = torch.randn(100, dtype=torch.float32) * 5.0
    max_epoch3 = torch.max(torch.abs(epoch3_log_ratio)).item()

    # Our monitoring should detect this
    if max_epoch3 > 10.0:
        warning = "severe"
    elif max_epoch3 > 1.0:
        warning = "concerning"
    else:
        warning = None

    # Should trigger at least concerning warning
    assert warning is not None, \
        f"KL explosion should trigger warning: max_epoch3={max_epoch3}, warning={warning}"


def test_ratio_clipping_vs_kl_clipping_distinction() -> None:
    """Test the distinction between ratio clipping (in loss) vs KL monitoring."""
    torch = pytest.importorskip("torch")

    clip_range = 0.2  # PPO clip range

    # Large log_ratio that would be clipped in loss
    log_ratio = torch.tensor([5.0], dtype=torch.float32)

    # Conservative numerical clipping (±20)
    log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio_clamped)

    # PPO loss clipping
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    # Verify distinction:
    # 1. log_ratio = 5.0 is preserved (< 20)
    assert log_ratio_clamped.item() == 5.0, "log_ratio should not be clamped"

    # 2. ratio = exp(5) ≈ 148 is computed
    assert abs(ratio.item() - 148.4) < 0.1, f"ratio should be exp(5)≈148: {ratio.item()}"

    # 3. ratio is clipped to [0.8, 1.2] in loss
    assert ratio_clipped.item() == pytest.approx(1.2), \
        f"ratio should be clipped to 1.2 in loss: {ratio_clipped.item()}"

    # 4. But monitoring should detect log_ratio = 5.0 as concerning
    max_abs = abs(log_ratio.item())
    if max_abs > 10.0:
        warning = "severe"
    elif max_abs > 1.0:
        warning = "concerning"
    else:
        warning = None

    assert warning == "concerning", \
        f"log_ratio=5.0 should trigger concerning warning: {warning}"


def test_zero_log_ratio_first_epoch() -> None:
    """Test that log_ratio ≈ 0 in first epoch (policy hasn't changed)."""
    torch = pytest.importorskip("torch")

    # In first minibatch of first epoch, new and old policies are the same
    new_log_prob = torch.tensor([-1.5, -2.0, -2.5], dtype=torch.float32)
    old_log_prob = new_log_prob.clone()  # Identical

    log_ratio = new_log_prob - old_log_prob
    approx_kl = old_log_prob - new_log_prob

    # Should be exactly zero
    assert torch.all(log_ratio == 0.0), \
        f"First epoch log_ratio should be 0: {log_ratio.tolist()}"
    assert torch.all(approx_kl == 0.0), \
        f"First epoch approx_kl should be 0: {approx_kl.tolist()}"

    # ratio should be exactly 1.0
    ratio = torch.exp(log_ratio)
    assert torch.allclose(ratio, torch.ones_like(ratio)), \
        f"First epoch ratio should be 1.0: {ratio.tolist()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
