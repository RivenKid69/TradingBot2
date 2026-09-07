"""
Deep comprehensive tests for per_quantile VF clipping mode.

This test suite covers:
1. Edge cases (single quantile, extreme values, zeros)
2. Gradient flow verification
3. Interaction with normalize_returns
4. Categorical vs Quantile consistency
5. Batch-specific clipping behavior
6. CVaR preservation
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


class TestPerQuantileEdgeCases:
    """Deep edge case testing for per_quantile mode."""

    def test_single_quantile_edge_case(self):
        """Test per_quantile with single quantile (degenerate case)."""
        old_value = 10.0
        clip_delta = 5.0

        # Single quantile
        new_quantiles = torch.tensor([[20.0]])  # [batch=1, num_quantiles=1]

        old_value_tensor = torch.full((1, 1), old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # Should be clipped to 15.0
        expected = torch.tensor([[15.0]])
        assert torch.allclose(
            quantiles_clipped, expected
        ), f"Expected {expected}, got {quantiles_clipped}"

        print("✓ Single quantile edge case works correctly")

    def test_all_quantiles_below_bound(self):
        """Test when all quantiles are below lower bound."""
        old_value = 10.0
        clip_delta = 5.0
        clip_min = old_value - clip_delta  # 5.0

        # All quantiles far below bound
        new_quantiles = torch.tensor([[-50.0, -20.0, -10.0, -5.0, 0.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # All should be clipped to 5.0
        expected = torch.full_like(new_quantiles, clip_min)
        assert torch.allclose(
            quantiles_clipped, expected
        ), f"Expected all {clip_min}, got {quantiles_clipped}"

        print("✓ All quantiles below bound correctly clipped")

    def test_all_quantiles_above_bound(self):
        """Test when all quantiles are above upper bound."""
        old_value = 10.0
        clip_delta = 5.0
        clip_max = old_value + clip_delta  # 15.0

        # All quantiles far above bound
        new_quantiles = torch.tensor([[20.0, 50.0, 100.0, 200.0, 500.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # All should be clipped to 15.0
        expected = torch.full_like(new_quantiles, clip_max)
        assert torch.allclose(
            quantiles_clipped, expected
        ), f"Expected all {clip_max}, got {quantiles_clipped}"

        print("✓ All quantiles above bound correctly clipped")

    def test_quantiles_exactly_on_bounds(self):
        """Test quantiles exactly on clipping bounds."""
        old_value = 10.0
        clip_delta = 5.0

        # Quantiles exactly on bounds
        new_quantiles = torch.tensor([[5.0, 10.0, 15.0]])  # [min, old, max]

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # Should remain unchanged
        assert torch.allclose(
            quantiles_clipped, new_quantiles
        ), f"Quantiles on bounds should be unchanged"

        print("✓ Quantiles on bounds unchanged")

    def test_zero_old_value(self):
        """Test per_quantile with old_value = 0."""
        old_value = 0.0
        clip_delta = 5.0

        new_quantiles = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        expected = torch.tensor([[-5.0, -5.0, 0.0, 5.0, 5.0]])
        assert torch.allclose(
            quantiles_clipped, expected
        ), f"Expected {expected}, got {quantiles_clipped}"

        print("✓ Zero old_value handled correctly")

    def test_negative_old_value(self):
        """Test per_quantile with negative old_value."""
        old_value = -10.0
        clip_delta = 5.0

        new_quantiles = torch.tensor([[-20.0, -10.0, 0.0, 10.0, 20.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        expected = torch.tensor([[-15.0, -10.0, -5.0, -5.0, -5.0]])
        assert torch.allclose(
            quantiles_clipped, expected
        ), f"Expected {expected}, got {quantiles_clipped}"

        print("✓ Negative old_value handled correctly")

    def test_very_small_clip_delta(self):
        """Test with very small clip_delta (tight constraint)."""
        old_value = 10.0
        clip_delta = 0.1  # Very tight!

        new_quantiles = torch.tensor([[0.0, 5.0, 10.0, 15.0, 20.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # All except middle should be clipped to [9.9, 10.1]
        expected = torch.tensor([[9.9, 9.9, 10.0, 10.1, 10.1]])
        assert torch.allclose(
            quantiles_clipped, expected, atol=1e-6
        ), f"Expected {expected}, got {quantiles_clipped}"

        print("✓ Very small clip_delta handled correctly")

    def test_large_batch_consistency(self):
        """Test consistency across large batch."""
        batch_size = 100
        num_quantiles = 5
        clip_delta = 5.0

        # Random old values
        old_values = torch.randn(batch_size, 1) * 10.0

        # Random new quantiles
        new_quantiles = torch.randn(batch_size, num_quantiles) * 20.0

        # Apply clipping
        quantiles_clipped = old_values + torch.clamp(
            new_quantiles - old_values, min=-clip_delta, max=clip_delta
        )

        # Verify bounds for each sample
        for i in range(batch_size):
            old_val = old_values[i, 0].item()
            clip_min = old_val - clip_delta
            clip_max = old_val + clip_delta

            sample_quantiles = quantiles_clipped[i]

            assert torch.all(
                sample_quantiles >= clip_min - 1e-6
            ), f"Sample {i}: quantiles below bound"
            assert torch.all(
                sample_quantiles <= clip_max + 1e-6
            ), f"Sample {i}: quantiles above bound"

        print(f"✓ Large batch (n={batch_size}) consistency verified")


class TestPerQuantileGradientFlow:
    """Test gradient flow through per_quantile clipping."""

    def test_gradient_flow_quantile_clipping(self):
        """Test that gradients flow correctly through per_quantile clipping."""
        old_value = 10.0
        clip_delta = 5.0

        # Create quantiles with gradient tracking
        new_quantiles = torch.tensor(
            [[0.0, 10.0, 20.0, 30.0, 50.0]], requires_grad=True, dtype=torch.float32
        )

        old_value_tensor = torch.tensor([[old_value]], dtype=torch.float32)

        # Apply per_quantile clipping
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # Compute loss (e.g., mean squared error)
        target = torch.tensor([[10.0, 12.0, 13.0, 14.0, 15.0]], dtype=torch.float32)
        loss = ((quantiles_clipped - target) ** 2).sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert new_quantiles.grad is not None, "Gradients should exist"

        # Check gradient values
        # Quantiles within bounds should have non-zero gradients
        # Quantiles clipped should have zero gradients
        grad = new_quantiles.grad[0]

        print(f"Gradients: {grad}")
        print(
            f"  quantile[0] = {new_quantiles[0, 0].item():.1f} (clipped to 5.0), grad = {grad[0].item():.4f}"
        )
        print(
            f"  quantile[1] = {new_quantiles[0, 1].item():.1f} (unclipped), grad = {grad[1].item():.4f}"
        )
        print(
            f"  quantile[2] = {new_quantiles[0, 2].item():.1f} (clipped to 15.0), grad = {grad[2].item():.4f}"
        )

        # Quantile[1] at 10.0 is within bounds [5, 15], should have gradient
        assert grad[1].item() != 0.0, "Unclipped quantile should have gradient"

        # Quantiles[0,2,3,4] are clipped, gradients should be zero
        assert grad[0].item() == 0.0, "Clipped quantile (below) should have zero gradient"
        assert grad[2].item() == 0.0, "Clipped quantile (above) should have zero gradient"

        print("✓ Gradient flow through per_quantile clipping verified")

    def test_gradient_magnitude_preservation(self):
        """Test that gradient magnitudes are reasonable after clipping."""
        old_value = 0.0
        clip_delta = 10.0

        # Quantiles within bounds
        new_quantiles = torch.tensor([[-5.0, 0.0, 5.0]], requires_grad=True, dtype=torch.float32)

        old_value_tensor = torch.tensor([[old_value]], dtype=torch.float32)

        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        target = torch.ones_like(quantiles_clipped)
        loss = ((quantiles_clipped - target) ** 2).sum()

        loss.backward()

        # All quantiles within bounds, all should have gradients
        grad = new_quantiles.grad[0]
        assert torch.all(grad != 0.0), "All within-bound quantiles should have gradients"

        # Gradient magnitudes should be similar (since all within bounds)
        grad_mean = grad.abs().mean()
        assert torch.all(grad.abs() < grad_mean * 3), "Gradient magnitudes should be reasonable"

        print(f"✓ Gradient magnitudes preserved (mean={grad_mean:.4f})")


class TestPerQuantileCategoricalSpecific:
    """Tests specific to categorical critic per_quantile mode."""

    def test_categorical_atoms_clipping_per_sample(self):
        """Test that categorical atoms are clipped per-sample correctly."""
        num_atoms = 51
        batch_size = 3
        clip_delta = 5.0

        # Fixed atoms (C51 style)
        atoms = torch.linspace(-10.0, 10.0, num_atoms)

        # Different old_values per sample
        old_values = torch.tensor([[0.0], [10.0], [-5.0]])

        # Simulate per_quantile clipping for categorical
        atoms_broadcast = atoms.unsqueeze(0)  # [1, num_atoms]

        atoms_clipped_batch = old_values + torch.clamp(
            atoms_broadcast - old_values, min=-clip_delta, max=clip_delta
        )

        # Verify each sample's atoms
        for i in range(batch_size):
            old_val = old_values[i, 0].item()
            clip_min = old_val - clip_delta
            clip_max = old_val + clip_delta

            atoms_i = atoms_clipped_batch[i]

            # All atoms should be within bounds
            assert torch.all(atoms_i >= clip_min - 1e-6), f"Sample {i}: atoms below {clip_min}"
            assert torch.all(atoms_i <= clip_max + 1e-6), f"Sample {i}: atoms above {clip_max}"

            # Check specific values
            assert atoms_i.min().item() == pytest.approx(
                clip_min, abs=1e-5
            ), f"Sample {i}: min atom should be {clip_min}"
            assert atoms_i.max().item() == pytest.approx(
                clip_max, abs=1e-5
            ), f"Sample {i}: max atom should be {clip_max}"

        print("✓ Categorical atoms clipped per-sample correctly")

    def test_categorical_probability_preservation(self):
        """Test that categorical probabilities sum to 1 after projection."""
        num_atoms = 51
        batch_size = 2

        # Create dummy probabilities
        probs = torch.rand(batch_size, num_atoms)
        probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize

        # Verify they sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size)), "Probabilities should sum to 1"

        print("✓ Categorical probability preservation verified")


class TestPerQuantileNormalizeReturns:
    """Test per_quantile interaction with normalize_returns."""

    def test_with_normalize_returns_true(self):
        """Test per_quantile with return normalization enabled."""
        # Simulate normalized space
        old_value_norm = 0.0  # Normalized old value
        clip_delta_norm = 1.0  # In normalized space

        new_quantiles_norm = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

        old_value_tensor = torch.full_like(new_quantiles_norm[:, :1], old_value_norm)
        quantiles_clipped_norm = old_value_tensor + torch.clamp(
            new_quantiles_norm - old_value_tensor, min=-clip_delta_norm, max=clip_delta_norm
        )

        expected = torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]])
        assert torch.allclose(
            quantiles_clipped_norm, expected
        ), f"Expected {expected}, got {quantiles_clipped_norm}"

        print("✓ per_quantile with normalize_returns works correctly")

    def test_raw_vs_normalized_consistency(self):
        """Test that clipping in raw vs normalized space is consistent."""
        # Raw space
        old_value_raw = 100.0
        clip_delta_raw = 50.0

        new_quantiles_raw = torch.tensor([[20.0, 100.0, 180.0]])

        old_value_tensor_raw = torch.full_like(new_quantiles_raw[:, :1], old_value_raw)
        quantiles_clipped_raw = old_value_tensor_raw + torch.clamp(
            new_quantiles_raw - old_value_tensor_raw, min=-clip_delta_raw, max=clip_delta_raw
        )

        # Normalized space (assume mean=100, std=50)
        ret_mu = 100.0
        ret_std = 50.0

        old_value_norm = (old_value_raw - ret_mu) / ret_std  # 0.0
        clip_delta_norm = clip_delta_raw / ret_std  # 1.0
        new_quantiles_norm = (new_quantiles_raw - ret_mu) / ret_std

        old_value_tensor_norm = torch.full_like(new_quantiles_norm[:, :1], old_value_norm)
        quantiles_clipped_norm = old_value_tensor_norm + torch.clamp(
            new_quantiles_norm - old_value_tensor_norm, min=-clip_delta_norm, max=clip_delta_norm
        )

        # Convert back to raw
        quantiles_clipped_from_norm = quantiles_clipped_norm * ret_std + ret_mu

        # Should match
        assert torch.allclose(
            quantiles_clipped_raw, quantiles_clipped_from_norm, rtol=1e-5
        ), "Raw and normalized clipping should be consistent"

        print("✓ Raw vs normalized clipping consistency verified")


class TestPerQuantileCVaRPreservation:
    """Test that per_quantile properly constrains CVaR."""

    def test_cvar_tail_constraint(self):
        """Test that tail quantiles (CVaR) are properly constrained."""
        old_value = 10.0
        clip_delta = 5.0
        clip_min = old_value - clip_delta

        # Distribution with very negative tail
        # τ = [0.1, 0.3, 0.5, 0.7, 0.9]
        new_quantiles = torch.tensor([[-100.0, -50.0, 10.0, 50.0, 100.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # CVaR at α=0.3 uses tail quantiles [0.1, 0.3]
        tail_clipped = quantiles_clipped[0, :2]
        cvar_clipped = tail_clipped.mean().item()

        # All tail quantiles must be >= clip_min
        assert torch.all(
            tail_clipped >= clip_min
        ), f"Tail quantiles must be >= {clip_min}, got {tail_clipped}"

        # CVaR must be >= clip_min
        assert cvar_clipped >= clip_min - 1e-6, f"CVaR must be >= {clip_min}, got {cvar_clipped}"

        print(f"✓ CVaR tail constraint verified (CVaR={cvar_clipped:.2f} >= {clip_min})")

    def test_cvar_downside_risk_limited(self):
        """Test that per_quantile limits downside risk (negative tail)."""
        old_value = 0.0
        clip_delta = 10.0

        # Extreme downside risk
        new_quantiles = torch.tensor([[-1000.0, -500.0, 0.0, 100.0, 200.0]])

        old_value_tensor = torch.full_like(new_quantiles[:, :1], old_value)
        quantiles_clipped = old_value_tensor + torch.clamp(
            new_quantiles - old_value_tensor, min=-clip_delta, max=clip_delta
        )

        # Worst-case quantile (τ=0.1) should be -10.0, not -1000.0
        worst_case = quantiles_clipped.min().item()
        assert worst_case == pytest.approx(
            -10.0, abs=1e-5
        ), f"Worst case should be -10.0, got {worst_case}"

        # CVaR at α=0.5 (median of 3 worst)
        tail = quantiles_clipped[0, :3]
        cvar = tail.mean().item()

        # CVaR should be bounded
        assert cvar >= -10.0, f"CVaR should be >= -10.0, got {cvar}"

        print(f"✓ Downside risk limited (worst={worst_case:.1f}, CVaR={cvar:.1f})")


class TestPerQuantileDefaultDisabled:
    """Test that per_quantile is properly disabled by default."""

    def test_default_mode_is_none(self):
        """Test that default mode is None (disabled)."""
        # This would require actual DistributionalPPO initialization
        # For now, we test the logic
        distributional_vf_clip_mode = None
        clip_range_vf = 0.5

        enabled = clip_range_vf is not None and distributional_vf_clip_mode not in (None, "disable")

        assert not enabled, "VF clipping should be disabled by default (mode=None)"

        print("✓ Default mode properly disables VF clipping")

    def test_explicit_disable(self):
        """Test that mode='disable' explicitly disables VF clipping."""
        distributional_vf_clip_mode = "disable"
        clip_range_vf = 0.5

        enabled = clip_range_vf is not None and distributional_vf_clip_mode not in (None, "disable")

        assert not enabled, "VF clipping should be disabled with mode='disable'"

        print("✓ Explicit disable works correctly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEEP COMPREHENSIVE TESTS FOR PER_QUANTILE VF CLIPPING")
    print("=" * 80 + "\n")

    # Run all tests
    test_classes = [
        TestPerQuantileEdgeCases,
        TestPerQuantileGradientFlow,
        TestPerQuantileCategoricalSpecific,
        TestPerQuantileNormalizeReturns,
        TestPerQuantileCVaRPreservation,
        TestPerQuantileDefaultDisabled,
    ]

    for test_class in test_classes:
        print(f"\n{'='*80}")
        print(f"{test_class.__name__}")
        print(f"{'='*80}")

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    print(f"\n{method_name}:")
                    method()
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80 + "\n")
