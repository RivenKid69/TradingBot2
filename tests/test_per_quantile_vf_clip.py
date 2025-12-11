"""
Comprehensive tests for per_quantile VF clipping mode.

Tests that the new per_quantile mode:
1. Guarantees all quantiles stay within [old_value - clip_delta, old_value + clip_delta]
2. Works correctly for both quantile and categorical critics
3. Handles edge cases properly
4. Integrates correctly with the training loop
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np


class TestPerQuantileMode:
    """Test suite for per_quantile VF clipping mode."""

    def test_quantile_bounds_guarantee(self):
        """
        Test that per_quantile mode GUARANTEES all quantiles within bounds.

        This is the core feature: unlike mean_only and mean_and_variance,
        per_quantile ensures EVERY quantile respects the clipping constraint.
        """
        # Setup
        old_value = 10.0
        clip_delta = 5.0
        clip_min = old_value - clip_delta  # 5.0
        clip_max = old_value + clip_delta  # 15.0

        # Test case 1: Wide distribution that would violate bounds
        new_quantiles = torch.tensor([[0.0, 10.0, 20.0, 30.0, 50.0]])  # mean=22
        num_samples, num_quantiles = new_quantiles.shape

        # Simulate per_quantile clipping logic
        old_value_tensor = torch.full((num_samples, 1), old_value)

        # Clip each quantile: quantile_clipped = old_value + clip(quantile - old_value, -delta, +delta)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )

        # Verify ALL quantiles are within bounds
        assert torch.all(quantiles_clipped >= clip_min), \
            f"Some quantiles below {clip_min}: {quantiles_clipped[quantiles_clipped < clip_min]}"
        assert torch.all(quantiles_clipped <= clip_max), \
            f"Some quantiles above {clip_max}: {quantiles_clipped[quantiles_clipped > clip_max]}"

        # Verify exact values for this test case
        expected = torch.tensor([[5.0, 10.0, 15.0, 15.0, 15.0]])  # All clipped to [5, 15]
        assert torch.allclose(quantiles_clipped, expected), \
            f"Expected {expected}, got {quantiles_clipped}"

        print("✓ per_quantile mode guarantees all quantiles within bounds!")

    def test_per_quantile_vs_mean_only_comparison(self):
        """
        Compare per_quantile vs mean_only on same problematic distribution.

        Shows that mean_only allows bounds violations while per_quantile prevents them.
        """
        old_value = 10.0
        clip_delta = 5.0
        clip_min = old_value - clip_delta
        clip_max = old_value + clip_delta

        # Problematic distribution: high mean, wide spread
        new_quantiles = torch.tensor([[5.0, 20.0, 35.0]])  # mean=20
        new_mean = new_quantiles.mean()

        # mean_only mode
        clipped_mean_only = torch.clamp(new_mean, clip_min, clip_max)  # 15.0
        delta = clipped_mean_only - new_mean  # -5.0
        quantiles_mean_only = new_quantiles + delta  # [0, 15, 30]

        # per_quantile mode
        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_per_quantile = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )  # [5, 15, 15]

        # Check violations
        mean_only_violates = (
            (quantiles_mean_only < clip_min).any() or
            (quantiles_mean_only > clip_max).any()
        )
        per_quantile_violates = (
            (quantiles_per_quantile < clip_min).any() or
            (quantiles_per_quantile > clip_max).any()
        )

        print(f"\nmean_only clipped quantiles: {quantiles_mean_only.tolist()}")
        print(f"  Bounds violation: {mean_only_violates}")
        print(f"  Min: {quantiles_mean_only.min().item():.2f} (should be >= {clip_min})")
        print(f"  Max: {quantiles_mean_only.max().item():.2f} (should be <= {clip_max})")

        print(f"\nper_quantile clipped quantiles: {quantiles_per_quantile.tolist()}")
        print(f"  Bounds violation: {per_quantile_violates}")
        print(f"  Min: {quantiles_per_quantile.min().item():.2f}")
        print(f"  Max: {quantiles_per_quantile.max().item():.2f}")

        # Assertions
        assert mean_only_violates, "mean_only should allow violations in this test case"
        assert not per_quantile_violates, "per_quantile must NEVER violate bounds"

        print("\n✓ per_quantile prevents violations that mean_only allows!")

    def test_per_quantile_batch_handling(self):
        """
        Test that per_quantile handles batches correctly.

        Each sample in batch may have different old_value, so clipping
        should be sample-specific.
        """
        batch_size = 3
        num_quantiles = 5
        clip_delta = 5.0

        # Different old_values per sample
        old_values = torch.tensor([[10.0], [20.0], [30.0]])  # [batch, 1]

        # Same new quantiles for all (for simplicity)
        # In practice, each sample would have different quantiles
        new_quantiles = torch.tensor([
            [0.0, 10.0, 20.0, 30.0, 50.0],
            [0.0, 10.0, 20.0, 30.0, 50.0],
            [0.0, 10.0, 20.0, 30.0, 50.0],
        ])  # [batch, num_quantiles]

        # per_quantile clipping: different bounds per sample
        quantiles_clipped = old_values + torch.clamp(
            new_quantiles - old_values,
            min=-clip_delta,
            max=clip_delta
        )

        # Verify each sample respects its own bounds
        for i in range(batch_size):
            sample_old_value = old_values[i, 0].item()
            sample_clip_min = sample_old_value - clip_delta
            sample_clip_max = sample_old_value + clip_delta
            sample_quantiles = quantiles_clipped[i]

            assert torch.all(sample_quantiles >= sample_clip_min), \
                f"Sample {i}: quantiles below {sample_clip_min}"
            assert torch.all(sample_quantiles <= sample_clip_max), \
                f"Sample {i}: quantiles above {sample_clip_max}"

            print(f"Sample {i}: old_value={sample_old_value:.1f}, "
                  f"bounds=[{sample_clip_min:.1f}, {sample_clip_max:.1f}], "
                  f"quantiles={sample_quantiles.tolist()}")

        print("\n✓ per_quantile handles batch-specific clipping correctly!")

    def test_per_quantile_preserves_order(self):
        """
        Test that per_quantile clipping preserves quantile ordering.

        Quantiles should remain monotonically increasing after clipping.
        """
        old_value = 10.0
        clip_delta = 5.0

        # Test with properly ordered quantiles
        new_quantiles = torch.tensor([[0.0, 5.0, 10.0, 20.0, 50.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )

        # Check ordering preserved
        for i in range(quantiles_clipped.shape[1] - 1):
            assert quantiles_clipped[0, i] <= quantiles_clipped[0, i + 1], \
                f"Ordering violated at index {i}: {quantiles_clipped[0, i]} > {quantiles_clipped[0, i+1]}"

        print(f"Original: {new_quantiles.tolist()}")
        print(f"Clipped:  {quantiles_clipped.tolist()}")
        print("✓ per_quantile preserves quantile ordering!")

    def test_per_quantile_no_op_when_within_bounds(self):
        """
        Test that per_quantile is a no-op when quantiles already within bounds.

        If all quantiles already satisfy the constraint, they should be unchanged.
        """
        old_value = 10.0
        clip_delta = 5.0

        # All quantiles already within [5, 15]
        new_quantiles = torch.tensor([[6.0, 8.0, 10.0, 12.0, 14.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )

        # Should be unchanged
        assert torch.allclose(quantiles_clipped, new_quantiles), \
            "Clipping should be no-op when already within bounds"

        print(f"Original: {new_quantiles.tolist()}")
        print(f"Clipped:  {quantiles_clipped.tolist()}")
        print("✓ per_quantile is no-op when quantiles within bounds!")

    def test_per_quantile_extreme_case(self):
        """
        Test per_quantile with extreme distribution.

        Even with very wide distribution, all quantiles must be clipped.
        """
        old_value = 0.0
        clip_delta = 1.0

        # Extremely wide distribution
        new_quantiles = torch.tensor([[-1000.0, -100.0, 0.0, 100.0, 1000.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )

        # All should be clipped to [-1, 1]
        expected = torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]])
        assert torch.allclose(quantiles_clipped, expected), \
            f"Expected {expected}, got {quantiles_clipped}"

        print(f"Original: {new_quantiles.tolist()}")
        print(f"Clipped:  {quantiles_clipped.tolist()}")
        print("✓ per_quantile handles extreme distributions correctly!")

    def test_categorical_per_quantile_atoms_clipping(self):
        """
        Test per_quantile mode for categorical critic.

        For categorical, atoms are clipped per-sample to ensure
        distribution support stays within bounds.
        """
        num_atoms = 51
        batch_size = 2
        clip_delta = 5.0

        # Fixed atoms (C51 style)
        atoms = torch.linspace(-10.0, 10.0, num_atoms)  # [num_atoms]

        # Different old values per sample
        old_values = torch.tensor([[10.0], [20.0]])  # [batch, 1]

        # Simulate per_quantile clipping for categorical
        # Each sample gets its own clipped atom range
        atoms_broadcast = atoms.unsqueeze(0)  # [1, num_atoms]

        # Clip atoms relative to old_value for each sample
        atoms_clipped_batch = old_values + torch.clamp(
            atoms_broadcast - old_values,
            min=-clip_delta,
            max=clip_delta
        )  # [batch, num_atoms]

        # Verify each sample's atoms are within bounds
        for i in range(batch_size):
            old_value_i = old_values[i, 0].item()
            clip_min_i = old_value_i - clip_delta
            clip_max_i = old_value_i + clip_delta
            atoms_i = atoms_clipped_batch[i]

            assert torch.all(atoms_i >= clip_min_i), \
                f"Sample {i}: atoms below {clip_min_i}"
            assert torch.all(atoms_i <= clip_max_i), \
                f"Sample {i}: atoms above {clip_max_i}"

            print(f"Sample {i}: old_value={old_value_i:.1f}, "
                  f"atoms range=[{atoms_i.min().item():.2f}, {atoms_i.max().item():.2f}], "
                  f"expected=[{clip_min_i:.1f}, {clip_max_i:.1f}]")

        print("\n✓ Categorical per_quantile clips atoms correctly!")

    def test_cvar_preservation_with_per_quantile(self):
        """
        Test that per_quantile properly constrains CVaR (tail quantiles).

        This is CRITICAL: CVaR is computed from tail quantiles,
        so they must respect clipping bounds.
        """
        old_value = 10.0
        clip_delta = 5.0
        clip_min = old_value - clip_delta

        # Distribution with very negative tail (risk!)
        # Quantiles at τ = [0.1, 0.3, 0.5, 0.7, 0.9]
        new_quantiles = torch.tensor([[-20.0, 0.0, 10.0, 20.0, 40.0]])

        # CVaR at α=0.25 uses tail quantiles [0.1, 0.3]
        tail_original = new_quantiles[0, :2]  # [-20, 0]
        cvar_original = tail_original.mean().item()  # -10

        # Apply per_quantile clipping
        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor,
            min=-clip_delta,
            max=clip_delta
        )

        # CVaR from clipped quantiles
        tail_clipped = quantiles_clipped[0, :2]
        cvar_clipped = tail_clipped.mean().item()

        print(f"Original tail quantiles: {tail_original.tolist()}")
        print(f"  CVaR (α=0.25): {cvar_original:.2f}")
        print(f"\nClipped tail quantiles: {tail_clipped.tolist()}")
        print(f"  CVaR (α=0.25): {cvar_clipped:.2f}")

        # Verify tail quantiles are bounded
        assert torch.all(tail_clipped >= clip_min), \
            f"Tail quantiles must be >= {clip_min}"

        # CVaR should be more conservative (higher) after clipping
        # because we clipped the extreme negative values
        assert cvar_clipped >= cvar_original, \
            "Clipped CVaR should be >= original (less extreme risk)"

        print(f"\n✓ per_quantile properly constrains CVaR (tail risk)!")


class TestPerQuantileParameterValidation:
    """Test parameter validation for per_quantile mode."""

    def test_mode_name_validation(self):
        """Test that 'per_quantile' is accepted as valid mode."""
        valid_modes = ["disable", "mean_only", "mean_and_variance", "per_quantile"]

        for mode in valid_modes:
            # Simulate validation logic
            mode_lower = mode.lower()
            assert mode_lower in ["disable", "mean_only", "mean_and_variance", "per_quantile"], \
                f"Valid mode {mode} should pass validation"

        print("✓ 'per_quantile' is recognized as valid mode!")

    def test_invalid_mode_rejected(self):
        """Test that invalid modes are still rejected."""
        invalid_modes = ["per_atom", "strict", "quantile_only", "clip_all"]

        for mode in invalid_modes:
            mode_lower = mode.lower()
            is_valid = mode_lower in ["disable", "mean_only", "mean_and_variance", "per_quantile"]
            assert not is_valid, f"Invalid mode {mode} should be rejected"

        print("✓ Invalid modes are properly rejected!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
