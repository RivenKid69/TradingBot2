"""
INTEGRATION TEST: Test actual DistributionalPPO _quantile_huber_loss implementation

This test validates the VF clipping fix by testing the _quantile_huber_loss method directly.
Uses minimal mocking to test real code paths.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
import numpy as np
import pytest


class TestQuantileHuberLoss:
    """Test suite for _quantile_huber_loss method with reduction parameter."""

    @pytest.fixture
    def setup_method_vars(self):
        """Setup test fixtures."""
        from distributional_ppo import DistributionalPPO

        # Create a partial instance just for testing the method
        class PartialPPO:
            def __init__(self):
                self._quantile_huber_kappa = 1.0
                self.num_quantiles = 5
                self._quantile_levels = torch.linspace(0.1, 0.9, self.num_quantiles)

            def _quantile_levels_tensor(self, device=None):
                """Return quantile levels tensor (optionally on specific device)."""
                if device is not None:
                    return self._quantile_levels.to(device)
                return self._quantile_levels

            # Copy the actual method from DistributionalPPO
            _quantile_huber_loss = DistributionalPPO._quantile_huber_loss

        return PartialPPO()

    def test_reduction_none(self, setup_method_vars):
        """Test reduction='none' returns per-sample losses."""
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        predicted = torch.randn(batch_size, num_quantiles, requires_grad=True)
        targets = torch.randn(batch_size, 1)

        loss = ppo._quantile_huber_loss(predicted, targets, reduction="none")

        assert loss.shape == (
            batch_size,
        ), f"reduction='none' should return [{batch_size}], got {loss.shape}"
        assert torch.all(torch.isfinite(loss)), "All losses should be finite"
        print(f"✓ reduction='none' shape: {loss.shape}")
        print(f"✓ loss values: {loss.tolist()}")

    def test_reduction_mean(self, setup_method_vars):
        """Test reduction='mean' returns scalar."""
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        predicted = torch.randn(batch_size, num_quantiles, requires_grad=True)
        targets = torch.randn(batch_size, 1)

        loss = ppo._quantile_huber_loss(predicted, targets, reduction="mean")

        assert loss.shape == (), f"reduction='mean' should return scalar, got {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"
        print(f"✓ reduction='mean' shape: {loss.shape}")
        print(f"✓ loss value: {loss.item():.6f}")

    def test_reduction_sum(self, setup_method_vars):
        """Test reduction='sum' returns scalar."""
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        predicted = torch.randn(batch_size, num_quantiles, requires_grad=True)
        targets = torch.randn(batch_size, 1)

        loss = ppo._quantile_huber_loss(predicted, targets, reduction="sum")

        assert loss.shape == (), f"reduction='sum' should return scalar, got {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"
        print(f"✓ reduction='sum' shape: {loss.shape}")
        print(f"✓ loss value: {loss.item():.6f}")

    def test_reduction_relationships(self, setup_method_vars):
        """Test mathematical relationships between reduction modes."""
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        predicted = torch.randn(batch_size, num_quantiles, requires_grad=True)
        targets = torch.randn(batch_size, 1)

        loss_none = ppo._quantile_huber_loss(predicted, targets, reduction="none")
        loss_mean = ppo._quantile_huber_loss(predicted, targets, reduction="mean")
        loss_sum = ppo._quantile_huber_loss(predicted, targets, reduction="sum")

        # Verify mathematical relationships
        assert torch.allclose(
            loss_mean, loss_none.mean(), atol=1e-6
        ), "reduction='mean' should equal mean of reduction='none'"
        assert torch.allclose(
            loss_sum, loss_none.sum(), atol=1e-6
        ), "reduction='sum' should equal sum of reduction='none'"

        print(
            f"✓ mean relationship verified: {loss_mean.item():.6f} == {loss_none.mean().item():.6f}"
        )
        print(f"✓ sum relationship verified: {loss_sum.item():.6f} == {loss_none.sum().item():.6f}")

    def test_backward_compatibility_default_mean(self, setup_method_vars):
        """Test backward compatibility - default reduction='mean'."""
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        predicted = torch.randn(batch_size, num_quantiles)
        targets = torch.randn(batch_size, 1)

        # Call without specifying reduction (should default to 'mean')
        loss_default = ppo._quantile_huber_loss(predicted, targets)

        # Call with explicit reduction='mean'
        loss_explicit = ppo._quantile_huber_loss(predicted, targets, reduction="mean")

        assert torch.allclose(
            loss_default, loss_explicit
        ), "Default should equal explicit reduction='mean'"
        assert loss_default.shape == (), "Default should return scalar"

        print(
            f"✓ Default matches explicit mean: {loss_default.item():.6f} == {loss_explicit.item():.6f}"
        )

    def test_invalid_reduction_raises_error(self, setup_method_vars):
        """Test that invalid reduction mode raises ValueError."""
        ppo = setup_method_vars

        predicted = torch.randn(3, 5)
        targets = torch.randn(3, 1)

        with pytest.raises(ValueError, match="Invalid reduction mode"):
            ppo._quantile_huber_loss(predicted, targets, reduction="invalid")

        print("✓ Invalid reduction correctly raises ValueError")

    def test_gradients_flow_correctly(self, setup_method_vars):
        """Test that gradients flow correctly for all reduction modes."""
        ppo = setup_method_vars

        for reduction in ["none", "mean", "sum"]:
            predicted = torch.randn(4, 5, requires_grad=True)
            targets = torch.randn(4, 1)

            loss = ppo._quantile_huber_loss(predicted, targets, reduction=reduction)

            # Compute scalar for backward (if needed)
            if reduction == "none":
                loss = loss.mean()

            loss.backward()

            assert predicted.grad is not None, f"Gradients should exist for reduction={reduction}"
            assert torch.all(
                torch.isfinite(predicted.grad)
            ), f"Gradients should be finite for reduction={reduction}"
            assert (
                predicted.grad.norm() > 0
            ), f"Gradients should be non-zero for reduction={reduction}"

            print(
                f"✓ Gradients flow correctly for reduction='{reduction}': norm={predicted.grad.norm().item():.6f}"
            )

    def test_per_sample_shapes_various_batch_sizes(self, setup_method_vars):
        """Test per-sample loss shapes for various batch sizes."""
        ppo = setup_method_vars

        num_quantiles = 5
        batch_sizes = [1, 4, 16, 64]

        for batch_size in batch_sizes:
            predicted = torch.randn(batch_size, num_quantiles)
            targets = torch.randn(batch_size, 1)

            loss = ppo._quantile_huber_loss(predicted, targets, reduction="none")

            assert loss.shape == (
                batch_size,
            ), f"Batch {batch_size}: expected shape ({batch_size},), got {loss.shape}"

            print(f"✓ Batch size {batch_size:3d}: loss shape {loss.shape}")

    def test_mean_of_max_vs_max_of_mean(self, setup_method_vars):
        """
        Test that we can correctly compute mean(max(...)) using reduction='none'.

        This demonstrates the building block for VF clipping fix.
        """
        ppo = setup_method_vars

        batch_size = 4
        num_quantiles = 5

        # Create test data where mean(max) != max(mean)
        predicted_unclipped = torch.tensor(
            [
                [1.0, 1.5, 2.0, 2.5, 3.0],
                [5.0, 5.1, 5.2, 5.3, 5.4],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            dtype=torch.float32,
        )

        predicted_clipped = torch.tensor(
            [
                [1.5, 2.0, 2.5, 3.0, 3.5],
                [4.0, 4.5, 5.0, 5.5, 6.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([[2.0], [5.0], [3.0], [10.0]], dtype=torch.float32)

        # Get per-sample losses
        loss_unclipped = ppo._quantile_huber_loss(predicted_unclipped, targets, reduction="none")
        loss_clipped = ppo._quantile_huber_loss(predicted_clipped, targets, reduction="none")

        # CORRECT: mean(max(...))
        correct_vf_loss = torch.mean(torch.max(loss_unclipped, loss_clipped))

        # INCORRECT (old bug): max(mean(...))
        incorrect_vf_loss = torch.max(loss_unclipped.mean(), loss_clipped.mean())

        print(f"Loss unclipped per-sample: {loss_unclipped.tolist()}")
        print(f"Loss clipped per-sample:   {loss_clipped.tolist()}")
        print(f"\nCorrect VF loss (mean of max):   {correct_vf_loss.item():.6f}")
        print(f"Incorrect VF loss (max of means): {incorrect_vf_loss.item():.6f}")
        print(f"Difference: {abs(correct_vf_loss.item() - incorrect_vf_loss.item()):.6f}")

        # Verify they are different (proving the bug matters)
        assert not torch.allclose(
            correct_vf_loss, incorrect_vf_loss, atol=1e-4
        ), "Correct and incorrect should differ in this scenario"

        print("✓ mean(max) correctly differs from max(mean)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
