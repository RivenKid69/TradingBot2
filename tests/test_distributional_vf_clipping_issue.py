"""
Test to demonstrate the conceptual issue with distributional VF clipping.

The current implementation:
1. Clips the MEAN of the distribution
2. Shifts ALL quantiles/atoms by the same delta

This means the SHAPE of the distribution (variance, skewness) is NOT constrained!

Example:
- Old distribution: quantiles=[0, 1, 2, 3, 4], mean=2, std≈1.41
- New distribution: quantiles=[-10, 0, 10, 20, 30], mean=10, std≈14.14 (10x variance!)
- With clip_delta=5:
  - clipped_mean = clamp(10, 2-5, 2+5) = 7
  - delta = 7 - 10 = -3
  - clipped_quantiles = [-13, -3, 7, 17, 27], mean=7, std≈14.14 (still 10x variance!)

The distribution changed RADICALLY (10x variance increase), but VF clipping allowed it!
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


def test_quantile_parallel_shift_does_not_constrain_variance():
    """Demonstrates that parallel shift of quantiles does not constrain variance changes."""

    # Old distribution from rollout (narrow, low variance)
    old_quantiles = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # mean=2, std≈1.41
    old_mean = old_quantiles.mean()
    old_std = old_quantiles.std()

    print(f"Old distribution:")
    print(f"  quantiles: {old_quantiles.tolist()}")
    print(f"  mean: {old_mean:.2f}")
    print(f"  std: {old_std:.2f}")
    print()

    # New distribution from current policy (wide, high variance - 10x!)
    new_quantiles = torch.tensor([-10.0, 0.0, 10.0, 20.0, 30.0])  # mean=10, std≈14.14
    new_mean = new_quantiles.mean()
    new_std = new_quantiles.std()

    print(f"New distribution (RADICALLY different - 10x variance!):")
    print(f"  quantiles: {new_quantiles.tolist()}")
    print(f"  mean: {new_mean:.2f}")
    print(f"  std: {new_std:.2f}")
    print(f"  variance ratio: {(new_std / old_std):.2f}x")
    print()

    # Current VF clipping implementation (parallel shift)
    clip_delta = 5.0
    clipped_mean = torch.clamp(new_mean, old_mean - clip_delta, old_mean + clip_delta)
    delta = clipped_mean - new_mean

    print(f"VF clipping with clip_delta={clip_delta}:")
    print(f"  clipped_mean: {clipped_mean:.2f}")
    print(f"  delta: {delta:.2f}")
    print()

    # Apply parallel shift (current implementation)
    clipped_quantiles = new_quantiles + delta
    clipped_mean_verify = clipped_quantiles.mean()
    clipped_std = clipped_quantiles.std()

    print(f"Clipped distribution (parallel shift):")
    print(f"  quantiles: {clipped_quantiles.tolist()}")
    print(f"  mean: {clipped_mean_verify:.2f}")
    print(f"  std: {clipped_std:.2f}")
    print(f"  variance ratio: {(clipped_std / old_std):.2f}x")
    print()

    # THE PROBLEM: Variance is NOT constrained!
    print("=" * 70)
    print("PROBLEM DEMONSTRATED:")
    print(f"  Mean was clipped: {abs(clipped_mean - old_mean):.2f} <= {clip_delta} ✓")
    print(f"  But variance INCREASED by {(clipped_std / old_std):.2f}x (UNCONSTRAINED!) ✗")
    print(f"  The distribution shape changed RADICALLY despite VF clipping!")
    print("=" * 70)
    print()

    # Verify the problem
    assert abs(clipped_mean - old_mean) <= clip_delta, "Mean should be clipped"
    assert clipped_std > old_std * 5, "Variance increased >5x despite clipping!"

    return {
        "old_std": old_std.item(),
        "new_std": new_std.item(),
        "clipped_std": clipped_std.item(),
        "variance_ratio": (clipped_std / old_std).item(),
    }


def test_categorical_shift_and_project():
    """
    Demonstrates that shift+project in categorical critic also has issues.

    While projection can change the shape, it's an indirect effect.
    The fundamental issue is: we're not directly constraining distribution changes.
    """

    # Fixed atoms for categorical critic (e.g., C51 with 51 atoms from -10 to 10)
    num_atoms = 51
    v_min, v_max = -10.0, 10.0
    atoms = torch.linspace(v_min, v_max, num_atoms)
    delta_z = (v_max - v_min) / (num_atoms - 1)

    print("Categorical Critic (C51-style):")
    print(f"  atoms: {num_atoms} atoms from {v_min} to {v_max}")
    print()

    # Old distribution: concentrated around 0 (low variance)
    old_probs = torch.zeros(num_atoms)
    center_idx = num_atoms // 2
    old_probs[center_idx - 2 : center_idx + 3] = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])
    old_mean = (old_probs * atoms).sum()
    old_std = torch.sqrt(((atoms - old_mean) ** 2 * old_probs).sum())

    print(f"Old distribution:")
    print(f"  mean: {old_mean:.2f}")
    print(f"  std: {old_std:.2f}")
    print()

    # New distribution: spread out across entire range (high variance)
    new_probs = torch.ones(num_atoms) / num_atoms  # Uniform distribution
    new_mean = (new_probs * atoms).sum()
    new_std = torch.sqrt(((atoms - new_mean) ** 2 * new_probs).sum())

    print(f"New distribution (RADICALLY different - uniform!):")
    print(f"  mean: {new_mean:.2f}")
    print(f"  std: {new_std:.2f}")
    print(f"  variance ratio: {(new_std / old_std):.2f}x")
    print()

    # VF clipping: shift atoms
    clip_delta = 2.0
    clipped_mean = torch.clamp(new_mean, old_mean - clip_delta, old_mean + clip_delta)
    delta = clipped_mean - new_mean

    print(f"VF clipping with clip_delta={clip_delta}:")
    print(f"  clipped_mean: {clipped_mean:.2f}")
    print(f"  delta: {delta:.2f}")
    print()

    # Shift atoms (current implementation before projection)
    atoms_shifted = atoms + delta

    # Project back to original atoms (simplified version)
    # This is where categorical differs from quantile
    # But the fundamental issue remains: we're not directly constraining distribution changes

    print("=" * 70)
    print("ISSUE: Even with projection, we're not directly constraining:")
    print("  1. How much the distribution SHAPE can change")
    print("  2. Changes in variance, skewness, or tail behavior")
    print("  3. The projection is an indirect effect, not a direct constraint")
    print("=" * 70)
    print()


def test_what_should_be_constrained():
    """
    What SHOULD VF clipping constrain in distributional critics?

    Options:
    1. Distance between distributions (Wasserstein, KL divergence)
    2. Individual quantiles/atoms
    3. Mean + variance/IQR
    4. Nothing (disable VF clipping for distributional critics)
    """

    print("=" * 70)
    print("WHAT SHOULD VF CLIPPING CONSTRAIN?")
    print("=" * 70)
    print()

    print("Option 1: Distance-based clipping (Wasserstein/KL)")
    print("  Pros: Theoretically sound, directly constrains distribution changes")
    print("  Cons: Expensive to compute, complex to implement")
    print()

    print("Option 2: Individual quantile/atom clipping")
    print("  Pros: Simple, directly constrains each element")
    print("  Cons: May break monotonicity (quantiles), structure (categorical)")
    print()

    print("Option 3: Mean + variance clipping")
    print("  Pros: Constrains first two moments, relatively simple")
    print("  Cons: Doesn't capture full distribution changes")
    print()

    print("Option 4: Disable VF clipping for distributional critics")
    print("  Pros: Safest, no theoretical issues")
    print("  Cons: May reduce training stability")
    print()

    print("RECOMMENDATION:")
    print("  Option 3 (Mean + variance) or Option 4 (Disable) are most practical")
    print("  Option 1 is theoretically best but computationally expensive")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEMONSTRATING DISTRIBUTIONAL VF CLIPPING ISSUE")
    print("=" * 70)
    print()

    # Test 1: Quantile critic
    print("\nTEST 1: QUANTILE CRITIC - PARALLEL SHIFT ISSUE")
    print("-" * 70)
    results = test_quantile_parallel_shift_does_not_constrain_variance()

    # Test 2: Categorical critic
    print("\n\nTEST 2: CATEGORICAL CRITIC - SHIFT+PROJECT ISSUE")
    print("-" * 70)
    test_categorical_shift_and_project()

    # Test 3: What should be constrained
    print("\n\nTEST 3: SOLUTION OPTIONS")
    print("-" * 70)
    test_what_should_be_constrained()

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("  The current VF clipping implementation for distributional critics")
    print("  does NOT effectively constrain distribution changes!")
    print("=" * 70)
    print()
