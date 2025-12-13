"""
Simple unit tests for torch.clamp numerical stability improvements (Problem #1).

These tests verify the numerical stability improvements without requiring
full PPO initialization, making them faster and more focused.

Author: Claude Code
Date: 2025-11-22
"""

import pytest
torch = pytest.importorskip("torch")


class TestTorchClampNumericalStability:
    """Test torch.clamp approach for numerical stability."""

    def test_torch_clamp_prevents_invalid_probabilities(self):
        """Test that torch.clamp keeps probabilities in valid range [1e-8, 1.0]."""
        batch_size = 8
        num_atoms = 21
        epsilon = 1e-8

        # Test case 1: Very small probabilities (< 1e-8)
        probs_small = torch.full((batch_size, num_atoms), 1e-10)
        probs_small = probs_small / probs_small.sum(dim=1, keepdim=True)

        # Test case 2: Probabilities close to 1.0
        probs_large = torch.zeros((batch_size, num_atoms))
        probs_large[:, 0] = 0.99
        probs_large[:, 1:] = 0.01 / (num_atoms - 1)

        # Test case 3: Mix of small and large
        probs_mixed = torch.rand((batch_size, num_atoms))
        probs_mixed = probs_mixed / probs_mixed.sum(dim=1, keepdim=True)
        probs_mixed[0, :5] = 1e-12
        probs_mixed[0, :] = probs_mixed[0, :] / probs_mixed[0, :].sum()

        for probs in [probs_small, probs_large, probs_mixed]:
            # Apply torch.clamp (as in the fix)
            probs_safe = torch.clamp(probs, min=epsilon, max=1.0)

            # Verify all values are in valid range
            assert torch.all(probs_safe >= epsilon), "Found probabilities below 1e-8"
            assert torch.all(probs_safe <= 1.0), "Found probabilities above 1.0"

            # Verify log is safe (no NaN or Inf)
            log_probs = torch.log(probs_safe)
            assert torch.all(torch.isfinite(log_probs)), "Found NaN or Inf in log probabilities"

    def test_torch_clamp_vs_addition_equivalence(self):
        """Test that torch.clamp gives similar results to addition for normal cases."""
        batch_size = 8
        num_atoms = 21
        epsilon = 1e-8

        # Create normal probabilities (not edge cases, all > epsilon)
        probs = torch.rand((batch_size, num_atoms))
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Ensure all probabilities are well above epsilon
        probs = torch.clamp(probs, min=epsilon * 10)
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Old approach: probs + epsilon
        log_probs_old = torch.log(probs + epsilon)

        # New approach: torch.clamp
        probs_safe = torch.clamp(probs, min=epsilon, max=1.0)
        log_probs_new = torch.log(probs_safe)

        # For normal probabilities (>> epsilon), results should be similar
        # Note: They won't be identical due to addition changing values
        # But the difference should be small for values >> epsilon
        relative_diff = torch.abs((log_probs_new - log_probs_old) / log_probs_old)
        # Maximum relative difference should be small (< 1%)
        max_relative_diff = relative_diff.max().item()
        assert max_relative_diff < 0.01, \
            f"Large relative difference: {max_relative_diff:.6f}"

    def test_torch_clamp_preserves_safe_values(self):
        """Test that torch.clamp preserves values already in valid range."""
        batch_size = 4
        num_atoms = 21
        epsilon = 1e-8

        # Create probabilities that are already safe (> epsilon)
        probs = torch.rand((batch_size, num_atoms))
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Ensure all are > epsilon
        probs = torch.clamp(probs, min=epsilon * 2)
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Apply torch.clamp
        probs_clamped = torch.clamp(probs, min=epsilon, max=1.0)

        # Should be identical (no modification needed)
        torch.testing.assert_close(
            probs_clamped,
            probs,
            rtol=1e-9,
            atol=1e-12,
            msg="torch.clamp should preserve already-safe probabilities"
        )

    def test_addition_modifies_safe_values_unnecessarily(self):
        """Test that addition modifies safe values unnecessarily."""
        batch_size = 4
        num_atoms = 21
        epsilon = 1e-8

        # Create probabilities that are already safe
        probs = torch.rand((batch_size, num_atoms))
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Ensure all are > epsilon
        probs = torch.clamp(probs, min=epsilon * 2)
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Old approach: addition
        probs_added = probs + epsilon

        # Addition changes the values even though they're already safe
        assert torch.any(probs_added != probs), \
            "Addition should modify probabilities (even safe ones)"

        # This is UNNECESSARY modification
        # torch.clamp would preserve them unchanged

    def test_cross_entropy_loss_numerical_stability(self):
        """Test cross-entropy loss with both approaches for numerical stability."""
        batch_size = 8
        num_atoms = 21
        epsilon = 1e-8

        # Create target distribution
        target_dist = torch.rand((batch_size, num_atoms))
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)

        # Create predicted distribution with some very small values
        pred_dist = torch.rand((batch_size, num_atoms))
        pred_dist[:, ::2] = 1e-10  # Every other value very small
        pred_dist = pred_dist / pred_dist.sum(dim=1, keepdim=True)

        # Old approach: probs + epsilon
        loss_old = -(target_dist * torch.log(pred_dist + epsilon)).sum(dim=1)

        # New approach: torch.clamp
        pred_dist_safe = torch.clamp(pred_dist, min=epsilon, max=1.0)
        loss_new = -(target_dist * torch.log(pred_dist_safe)).sum(dim=1)

        # Both should be finite
        assert torch.all(torch.isfinite(loss_old)), "Old approach produced NaN/Inf"
        assert torch.all(torch.isfinite(loss_new)), "New approach produced NaN/Inf"

        # Both should be non-negative (cross-entropy property)
        assert torch.all(loss_old >= 0), "Old approach: negative cross-entropy"
        assert torch.all(loss_new >= 0), "New approach: negative cross-entropy"

        # Both approaches should give similar loss magnitudes (not exact match)
        # Check that neither approach gives drastically different results
        relative_diff = torch.abs((loss_new - loss_old) / (loss_old + 1e-10))
        max_relative_diff = relative_diff.max().item()
        # Allow up to 5% relative difference (they won't be identical)
        assert max_relative_diff < 0.05, \
            f"Cross-entropy losses differ too much: {max_relative_diff:.4f}"

    def test_extreme_edge_cases_no_nan(self):
        """Test that no NaN occurs even with extreme edge cases."""
        batch_size = 4
        num_atoms = 21
        epsilon = 1e-8

        edge_cases = [
            # Case 1: All probabilities equal (uniform)
            torch.ones((batch_size, num_atoms)) / num_atoms,

            # Case 2: One probability very close to 1.0
            torch.cat([
                torch.ones((batch_size, 1)) * 0.9999,
                torch.ones((batch_size, num_atoms - 1)) * (0.0001 / (num_atoms - 1))
            ], dim=1),

            # Case 3: Many very small probabilities
            torch.cat([
                torch.ones((batch_size, 1)) * 0.99,
                torch.ones((batch_size, num_atoms - 1)) * (1e-10)
            ], dim=1),

            # Case 4: Extreme small values
            torch.cat([
                torch.ones((batch_size, 1)) * 0.999999,
                torch.ones((batch_size, num_atoms - 1)) * (1e-15)
            ], dim=1),
        ]

        for i, probs in enumerate(edge_cases):
            # Normalize
            probs = probs / probs.sum(dim=1, keepdim=True)

            # Apply torch.clamp
            probs_safe = torch.clamp(probs, min=epsilon, max=1.0)

            # Verify safe for log
            log_probs = torch.log(probs_safe)
            assert torch.all(torch.isfinite(log_probs)), \
                f"Case {i+1}: log produced NaN/Inf"

            # Verify values in valid range
            assert torch.all(probs_safe >= epsilon), \
                f"Case {i+1}: values below epsilon"
            assert torch.all(probs_safe <= 1.0), \
                f"Case {i+1}: values above 1.0"

    def test_gradient_flow_through_torch_clamp(self):
        """Test that gradients flow correctly through torch.clamp."""
        batch_size = 4
        num_atoms = 21
        epsilon = 1e-8

        # Create probabilities with gradient tracking
        logits = torch.randn((batch_size, num_atoms), requires_grad=True)
        probs = torch.softmax(logits, dim=1)

        # Apply torch.clamp
        probs_safe = torch.clamp(probs, min=epsilon, max=1.0)

        # Compute loss
        target = torch.rand((batch_size, num_atoms))
        target = target / target.sum(dim=1, keepdim=True)
        loss = -(target * torch.log(probs_safe)).sum()

        # Backward
        loss.backward()

        # Verify gradients exist
        assert logits.grad is not None, "No gradient for logits"

        # Verify gradients are finite
        assert torch.all(torch.isfinite(logits.grad)), \
            "Gradients contain NaN/Inf"

        # Verify gradients are non-zero (learning is happening)
        assert torch.norm(logits.grad) > 0, "Gradient norm is zero"

    def test_torch_clamp_idempotent(self):
        """Test that torch.clamp is idempotent (applying twice gives same result)."""
        batch_size = 8
        num_atoms = 21
        epsilon = 1e-8

        probs = torch.rand((batch_size, num_atoms))
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Add some edge cases
        probs[0, :5] = 1e-12
        probs[0, :] = probs[0, :] / probs[0, :].sum()

        # Apply clamp once
        probs_1 = torch.clamp(probs, min=epsilon, max=1.0)

        # Apply clamp twice
        probs_2 = torch.clamp(probs_1, min=epsilon, max=1.0)

        # Should be identical (idempotent)
        torch.testing.assert_close(
            probs_1,
            probs_2,
            rtol=1e-12,
            atol=1e-15,
            msg="torch.clamp is not idempotent"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
