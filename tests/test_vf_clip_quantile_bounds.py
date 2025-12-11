"""
Test demonstrating quantile bounds violation in distributional VF clipping.

This test shows the exact problem described: when using parallel shift
(mean_only mode) or even variance constraint (mean_and_variance mode),
individual quantiles can still exceed the VF clipping bounds
[old_value - clip_delta, old_value + clip_delta].

Example from problem description:
- old_value = 10, clip_delta = 5
- New quantiles: [5, 20, 35], mean = 20
- After clipping mean to 15: [0, 15, 30] (parallel shift by -5)
- PROBLEM: Quantiles [0, 30] are outside bounds [5, 15]!
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np


def test_mean_only_quantile_bounds_violation():
    """
    Demonstrate that mean_only mode allows quantiles to exceed clipping bounds.

    This is the CORE problem: VF clipping should constrain value changes,
    but mean_only only clips the mean, not individual quantiles.
    """
    print("\n" + "="*80)
    print("TEST: Quantile Bounds Violation in mean_only Mode")
    print("="*80)

    # Example from problem description
    old_value = 10.0
    clip_delta = 5.0
    clip_min = old_value - clip_delta  # 5.0
    clip_max = old_value + clip_delta  # 15.0

    print(f"\nSetup:")
    print(f"  old_value = {old_value}")
    print(f"  clip_delta = {clip_delta}")
    print(f"  Clipping bounds: [{clip_min}, {clip_max}]")

    # New quantiles: wide distribution centered at 20
    new_quantiles = torch.tensor([[5.0, 20.0, 35.0]])  # mean = 20
    new_mean = new_quantiles.mean()

    print(f"\nNew distribution:")
    print(f"  Quantiles: {new_quantiles.tolist()}")
    print(f"  Mean: {new_mean.item():.2f}")

    # mean_only mode: clip mean, parallel shift
    clipped_mean = torch.clamp(
        new_mean,
        min=clip_min,
        max=clip_max
    )

    delta = clipped_mean - new_mean
    quantiles_clipped = new_quantiles + delta

    print(f"\nAfter mean_only clipping:")
    print(f"  Clipped mean: {clipped_mean.item():.2f}")
    print(f"  Delta (shift): {delta.item():.2f}")
    print(f"  Clipped quantiles: {quantiles_clipped.tolist()}")

    # Check bounds violation
    quantile_min = quantiles_clipped.min().item()
    quantile_max = quantiles_clipped.max().item()

    print(f"\nBounds check:")
    print(f"  Quantile range: [{quantile_min:.2f}, {quantile_max:.2f}]")
    print(f"  Expected bounds: [{clip_min:.2f}, {clip_max:.2f}]")

    violation_low = quantile_min < clip_min
    violation_high = quantile_max > clip_max

    if violation_low or violation_high:
        print(f"\n❌ BOUNDS VIOLATION DETECTED!")
        if violation_low:
            print(f"   - Minimum quantile {quantile_min:.2f} < {clip_min:.2f} (underflow by {clip_min - quantile_min:.2f})")
        if violation_high:
            print(f"   - Maximum quantile {quantile_max:.2f} > {clip_max:.2f} (overflow by {quantile_max - clip_max:.2f})")
        print(f"\n   This confirms the problem: mean_only mode does NOT constrain individual quantiles!")
    else:
        print(f"\n✓ No bounds violation (but this is not guaranteed in general)")

    # Verify the problem exists
    assert violation_low or violation_high, \
        "Expected bounds violation in mean_only mode, but none detected!"

    print("\n" + "="*80)
    return violation_low, violation_high


def test_mean_and_variance_still_allows_bounds_violation():
    """
    Demonstrate that even mean_and_variance mode can allow bounds violations.

    Variance constraint reduces the spread, but doesn't guarantee
    all quantiles stay within [old_value - clip_delta, old_value + clip_delta].
    """
    print("\n" + "="*80)
    print("TEST: Bounds Violation Possible in mean_and_variance Mode")
    print("="*80)

    old_value = 10.0
    clip_delta = 3.0  # Tighter clipping
    clip_min = old_value - clip_delta  # 7.0
    clip_max = old_value + clip_delta  # 13.0

    print(f"\nSetup:")
    print(f"  old_value = {old_value}")
    print(f"  clip_delta = {clip_delta}")
    print(f"  Clipping bounds: [{clip_min}, {clip_max}]")

    # Old quantiles: narrow distribution
    old_quantiles = torch.tensor([[9.0, 10.0, 11.0]])  # mean=10, variance~0.67
    old_mean = old_quantiles.mean()
    old_variance = ((old_quantiles - old_mean) ** 2).mean()

    # New quantiles: very wide distribution
    new_quantiles = torch.tensor([[-5.0, 20.0, 45.0]])  # mean=20, variance~433.33
    new_mean = new_quantiles.mean()
    new_variance = ((new_quantiles - new_mean) ** 2).mean()

    print(f"\nOld distribution:")
    print(f"  Quantiles: {old_quantiles.tolist()}")
    print(f"  Mean: {old_mean.item():.2f}")
    print(f"  Variance: {old_variance.item():.2f}")

    print(f"\nNew distribution:")
    print(f"  Quantiles: {new_quantiles.tolist()}")
    print(f"  Mean: {new_mean.item():.2f}")
    print(f"  Variance: {new_variance.item():.2f}")
    print(f"  Variance ratio: {(new_variance / old_variance).item():.2f}x")

    # mean_and_variance mode: clip mean + constrain variance
    clipped_mean = torch.clamp(
        new_mean,
        min=clip_min,
        max=clip_max
    )

    # Parallel shift
    delta = clipped_mean - new_mean
    quantiles_shifted = new_quantiles + delta

    # Constrain variance (factor = 2.0)
    variance_factor = 2.0
    quantiles_centered = quantiles_shifted - clipped_mean
    current_variance = (quantiles_centered ** 2).mean()

    max_variance = old_variance * (variance_factor ** 2)
    variance_ratio = torch.sqrt(current_variance / (old_variance + 1e-8))
    variance_scale = torch.clamp(variance_ratio, max=variance_factor)

    quantiles_clipped = clipped_mean + quantiles_centered * variance_scale

    clipped_variance = ((quantiles_clipped - clipped_mean) ** 2).mean()

    print(f"\nAfter mean_and_variance clipping:")
    print(f"  Clipped mean: {clipped_mean.item():.2f}")
    print(f"  Clipped quantiles: {quantiles_clipped.tolist()}")
    print(f"  Clipped variance: {clipped_variance.item():.2f}")
    print(f"  Variance factor applied: {variance_scale.item():.2f}")

    # Check bounds
    quantile_min = quantiles_clipped.min().item()
    quantile_max = quantiles_clipped.max().item()

    print(f"\nBounds check:")
    print(f"  Quantile range: [{quantile_min:.2f}, {quantile_max:.2f}]")
    print(f"  Expected bounds: [{clip_min:.2f}, {clip_max:.2f}]")

    violation_low = quantile_min < clip_min
    violation_high = quantile_max > clip_max

    if violation_low or violation_high:
        print(f"\n❌ BOUNDS VIOLATION DETECTED!")
        if violation_low:
            print(f"   - Minimum quantile {quantile_min:.2f} < {clip_min:.2f}")
        if violation_high:
            print(f"   - Maximum quantile {quantile_max:.2f} > {clip_max:.2f}")
        print(f"\n   Even mean_and_variance mode does NOT guarantee quantile bounds!")
    else:
        print(f"\n✓ No bounds violation in this case")
        print(f"   (But this is not guaranteed - depends on variance factor and distribution shape)")

    print("\n" + "="*80)


def test_per_quantile_clipping_solution():
    """
    Demonstrate the proposed solution: clip each quantile individually.

    Formula: quantile_clipped = old_value + clip(quantile - old_value, -ε, +ε)

    This GUARANTEES all quantiles stay within bounds [old_value - ε, old_value + ε].
    """
    print("\n" + "="*80)
    print("TEST: Per-Quantile Clipping Solution")
    print("="*80)

    old_value = 10.0
    clip_delta = 5.0
    clip_min = old_value - clip_delta  # 5.0
    clip_max = old_value + clip_delta  # 15.0

    print(f"\nSetup:")
    print(f"  old_value = {old_value}")
    print(f"  clip_delta = {clip_delta}")
    print(f"  Clipping bounds: [{clip_min}, {clip_max}]")

    # Same problematic case as test 1
    new_quantiles = torch.tensor([[5.0, 20.0, 35.0]])  # mean = 20
    new_mean = new_quantiles.mean()

    print(f"\nNew distribution:")
    print(f"  Quantiles: {new_quantiles.tolist()}")
    print(f"  Mean: {new_mean.item():.2f}")

    # Per-quantile clipping: clip each quantile relative to old_value
    # quantile_clipped = old_value + clip(quantile - old_value, -clip_delta, +clip_delta)
    quantiles_clipped = old_value + torch.clamp(
        new_quantiles - old_value,
        min=-clip_delta,
        max=clip_delta
    )

    clipped_mean = quantiles_clipped.mean()

    print(f"\nAfter per-quantile clipping:")
    print(f"  Clipped quantiles: {quantiles_clipped.tolist()}")
    print(f"  Clipped mean: {clipped_mean.item():.2f}")

    # Check bounds
    quantile_min = quantiles_clipped.min().item()
    quantile_max = quantiles_clipped.max().item()

    print(f"\nBounds check:")
    print(f"  Quantile range: [{quantile_min:.2f}, {quantile_max:.2f}]")
    print(f"  Expected bounds: [{clip_min:.2f}, {clip_max:.2f}]")

    # Verify all quantiles are within bounds
    within_bounds = (quantiles_clipped >= clip_min).all() and (quantiles_clipped <= clip_max).all()

    if within_bounds:
        print(f"\n✓ ALL QUANTILES WITHIN BOUNDS!")
        print(f"   Per-quantile clipping GUARANTEES bounds constraint!")
    else:
        print(f"\n❌ ERROR: Bounds violation detected (should not happen)")

    assert within_bounds, "Per-quantile clipping should guarantee bounds!"

    print(f"\nKey insight:")
    print(f"  - mean_only: Clips mean only → quantiles can overflow")
    print(f"  - mean_and_variance: Clips mean + constrains variance → quantiles can still overflow")
    print(f"  - per_quantile: Clips EACH quantile → GUARANTEES bounds")

    print("\n" + "="*80)
    return True


def analyze_impact_on_cvar():
    """
    Analyze why per-quantile clipping is important for CVaR.

    CVaR is computed from tail quantiles, so allowing tail quantiles
    to exceed bounds defeats the purpose of VF clipping for risk-sensitive RL.
    """
    print("\n" + "="*80)
    print("ANALYSIS: Impact on CVaR (Conditional Value at Risk)")
    print("="*80)

    old_value = 10.0
    clip_delta = 5.0

    # Example: 5 quantiles at τ = [0.1, 0.3, 0.5, 0.7, 0.9]
    new_quantiles = torch.tensor([[0.0, 10.0, 20.0, 30.0, 50.0]])  # mean = 22

    print(f"\nNew quantiles: {new_quantiles.tolist()}")
    print(f"Mean: {new_quantiles.mean().item():.2f}")

    # CVaR at α=0.25 uses quantiles 0.1, 0.3 (tail quantiles)
    cvar_alpha = 0.25
    tail_quantiles = new_quantiles[0, :2]  # [0.0, 10.0]
    cvar_original = tail_quantiles.mean()

    print(f"\nCVaR (α={cvar_alpha}) from tail quantiles [0.1, 0.3]:")
    print(f"  Original CVaR: {cvar_original.item():.2f}")

    # mean_only clipping
    new_mean = new_quantiles.mean()
    clipped_mean_only = torch.clamp(new_mean, old_value - clip_delta, old_value + clip_delta)
    delta = clipped_mean_only - new_mean
    quantiles_mean_only = new_quantiles + delta

    tail_quantiles_mean_only = quantiles_mean_only[0, :2]
    cvar_mean_only = tail_quantiles_mean_only.mean()

    print(f"\nmean_only mode:")
    print(f"  Clipped quantiles: {quantiles_mean_only.tolist()}")
    print(f"  Tail quantiles: {tail_quantiles_mean_only.tolist()}")
    print(f"  CVaR: {cvar_mean_only.item():.2f}")
    print(f"  PROBLEM: Tail quantile {tail_quantiles_mean_only[0].item():.2f} < {old_value - clip_delta:.2f}")

    # per_quantile clipping
    quantiles_per_quantile = old_value + torch.clamp(
        new_quantiles - old_value,
        min=-clip_delta,
        max=clip_delta
    )

    tail_quantiles_per_quantile = quantiles_per_quantile[0, :2]
    cvar_per_quantile = tail_quantiles_per_quantile.mean()

    print(f"\nper_quantile mode:")
    print(f"  Clipped quantiles: {quantiles_per_quantile.tolist()}")
    print(f"  Tail quantiles: {tail_quantiles_per_quantile.tolist()}")
    print(f"  CVaR: {cvar_per_quantile.item():.2f}")
    print(f"  ✓ All tail quantiles within bounds [{old_value - clip_delta:.2f}, {old_value + clip_delta:.2f}]")

    print(f"\nConclusion:")
    print(f"  CVaR is computed from TAIL quantiles (where risk is!)")
    print(f"  If tail quantiles can exceed bounds, VF clipping fails for risk-sensitive RL!")
    print(f"  Per-quantile clipping is ESSENTIAL for CVaR-based training!")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n")
    print("="*80)
    print("DISTRIBUTIONAL VF CLIPPING: QUANTILE BOUNDS VIOLATION")
    print("="*80)
    print("\nThis test suite demonstrates the problem described:")
    print("VF clipping for distributional critics must constrain EACH quantile,")
    print("not just the mean or variance!")

    # Run tests
    test_mean_only_quantile_bounds_violation()
    test_mean_and_variance_still_allows_bounds_violation()
    test_per_quantile_clipping_solution()
    analyze_impact_on_cvar()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Problem confirmed: mean_only and mean_and_variance modes")
    print("  can allow individual quantiles to exceed VF clipping bounds")
    print("\n✓ Solution validated: per_quantile mode guarantees")
    print("  all quantiles stay within [old_value - ε, old_value + ε]")
    print("\n✓ Impact: Critical for CVaR-based risk-sensitive RL,")
    print("  where tail quantiles must be properly constrained")
    print("\n" + "="*80)
    print("\n")
