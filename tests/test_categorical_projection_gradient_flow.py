"""
Test gradient flow through categorical distribution projection.

This test verifies that gradients properly flow through the _project_categorical_distribution
function when used for VF clipping. This is CRITICAL because the projection is applied to
predicted probabilities (not targets), so gradients must backpropagate correctly.

Background:
-----------
In distributional_ppo.py:8794, the projection is used to compute clipped predictions:
    pred_probs_clipped = self._project_categorical_distribution(
        probs=pred_probs_fp32,  # Current network predictions
        source_atoms=atoms_shifted,
        target_atoms=atoms_original,
    )

These clipped predictions are then used in the loss function:
    critic_loss_clipped = -(log_predictions_clipped * target_probs).sum(dim=1)

Therefore, gradients MUST flow from the loss back through the projection to pred_probs_fp32
and ultimately to the network parameters.

Issue:
------
The current implementation uses .item() to extract indices and Python loops for indexing,
which can break the computational graph in PyTorch.
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np

from distributional_ppo import DistributionalPPO


class TestCategoricalProjectionGradientFlow:
    """Test that gradients flow correctly through categorical projection."""

    def test_gradient_flow_simple_case(self):
        """
        Test that gradients flow through projection in a simple case.

        This test creates a minimal scenario where:
        1. Input probabilities require gradients
        2. Projection is applied
        3. Simple loss is computed
        4. We verify gradients exist and are reasonable
        """
        algo = DistributionalPPO.__new__(DistributionalPPO)

        # Setup
        batch_size = 4
        num_atoms = 5
        target_atoms = torch.linspace(-2.0, 2.0, num_atoms)

        # Create input probabilities that require gradients
        logits = torch.randn(batch_size, num_atoms, requires_grad=True)
        probs = torch.softmax(logits, dim=1)

        # Shift atoms (simulating VF clipping scenario)
        delta = 0.5
        source_atoms = target_atoms + delta

        # Project distribution
        projected_probs = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Compute a simple loss (e.g., negative log likelihood with uniform target)
        target_probs = torch.ones(batch_size, num_atoms) / num_atoms
        log_probs = torch.log(projected_probs.clamp(min=1e-8))
        loss = -(log_probs * target_probs).sum()

        # Backpropagate
        loss.backward()

        # CRITICAL CHECK: Gradients should exist and be non-zero
        assert logits.grad is not None, "Gradient should exist on input logits"
        assert not torch.allclose(
            logits.grad, torch.zeros_like(logits.grad)
        ), "Gradient should not be all zeros (gradient flow is broken)"

        # Check gradient is finite and reasonable
        assert torch.all(torch.isfinite(logits.grad)), "Gradient should be finite"

        # Gradient magnitude should be reasonable (not too small or too large)
        grad_norm = logits.grad.norm()
        assert grad_norm > 1e-6, f"Gradient norm {grad_norm} is too small"
        assert grad_norm < 1e3, f"Gradient norm {grad_norm} is too large"

    def test_gradient_flow_with_same_bounds_correction(self):
        """
        Test gradient flow when same_bounds correction is triggered.

        This is the specific case where the bug is suspected:
        When source atoms exactly match target atoms, the correction code
        uses .item() and Python indexing which may break gradients.
        """
        algo = DistributionalPPO.__new__(DistributionalPPO)

        # Setup: Create a case where some atoms will have same_bounds
        batch_size = 2
        num_atoms = 7
        target_atoms = torch.linspace(-3.0, 3.0, num_atoms)

        # Create input probabilities that require gradients
        logits = torch.randn(batch_size, num_atoms, requires_grad=True)
        probs = torch.softmax(logits, dim=1)

        # Create delta that will cause some atoms to land exactly on target atoms
        # This triggers the same_bounds correction path
        delta = torch.tensor([[0.0], [1.0]])  # First batch: no shift, second: shift by 1
        source_atoms = target_atoms.unsqueeze(0) + delta

        # Project distribution
        projected_probs = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Compute loss
        target_probs = torch.ones(batch_size, num_atoms) / num_atoms
        log_probs = torch.log(projected_probs.clamp(min=1e-8))
        loss = -(log_probs * target_probs).sum()

        # Backpropagate
        loss.backward()

        # CRITICAL CHECK: Gradients should exist for BOTH batch items
        assert logits.grad is not None, "Gradient should exist"

        # Check gradient exists for both rows (including the same_bounds case)
        for i in range(batch_size):
            row_grad = logits.grad[i]
            assert not torch.allclose(
                row_grad, torch.zeros_like(row_grad)
            ), f"Gradient for batch item {i} should not be all zeros"

        assert torch.all(torch.isfinite(logits.grad)), "Gradient should be finite"

    def test_gradient_flow_matches_expected_direction(self):
        """
        Test that gradients point in the expected direction.

        When we increase probability on an atom, the loss should change accordingly,
        and the gradient should reflect that.
        """
        algo = DistributionalPPO.__new__(DistributionalPPO)

        batch_size = 1
        num_atoms = 5
        target_atoms = torch.linspace(-2.0, 2.0, num_atoms)
        source_atoms = target_atoms + 0.3  # Small shift

        # Create a specific probability distribution
        logits = torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0]], requires_grad=True)
        probs = torch.softmax(logits, dim=1)

        # Project
        projected_probs = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Target: prefer middle atom
        target_probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])

        # Cross-entropy loss
        log_probs = torch.log(projected_probs.clamp(min=1e-8))
        loss = -(log_probs * target_probs).sum()

        loss.backward()

        # Gradient should exist and be non-zero
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))

        # The middle logit should have a meaningful gradient
        # (since target prefers middle atom)
        assert abs(logits.grad[0, 2].item()) > 1e-6

    def test_gradient_flow_end_to_end_scenario(self):
        """
        Test gradient flow in a scenario that mimics actual VF clipping usage.

        This simulates the exact pattern used in distributional_ppo.py:8788-8819:
        1. Compute delta from clipping
        2. Shift atoms by delta
        3. Project probabilities
        4. Compute loss with projected probabilities
        5. Verify gradients flow back
        """
        algo = DistributionalPPO.__new__(DistributionalPPO)

        batch_size = 3
        num_atoms = 21
        v_min, v_max = -5.0, 5.0
        atoms_original = torch.linspace(v_min, v_max, num_atoms)

        # Simulate predicted probabilities from network
        logits = torch.randn(batch_size, num_atoms, requires_grad=True)
        pred_probs = torch.softmax(logits, dim=1)

        # Simulate VF clipping: compute predicted mean and clip it
        mean_pred = (pred_probs * atoms_original).sum(dim=1, keepdim=True)
        mean_old = torch.zeros(batch_size, 1)
        clip_range = 2.0

        mean_clipped = torch.clamp(
            mean_pred,
            min=mean_old - clip_range,
            max=mean_old + clip_range
        )

        # Compute delta and shift atoms (exactly as in distributional_ppo.py:8790)
        delta_norm = mean_clipped - mean_pred
        atoms_shifted = atoms_original.unsqueeze(0) + delta_norm

        # Project (exactly as in distributional_ppo.py:8794)
        pred_probs_clipped = algo._project_categorical_distribution(
            probs=pred_probs,
            source_atoms=atoms_shifted,
            target_atoms=atoms_original,
        )

        # Compute loss with clipped predictions (as in distributional_ppo.py:8819)
        target_probs = torch.ones(batch_size, num_atoms) / num_atoms
        log_predictions_clipped = torch.log(pred_probs_clipped.clamp(min=1e-8))
        critic_loss_clipped = -(log_predictions_clipped * target_probs).sum(dim=1)
        loss = critic_loss_clipped.mean()

        # Backpropagate
        loss.backward()

        # CRITICAL: Gradients MUST flow back to the network logits
        assert logits.grad is not None, "Gradient must exist on network output"
        assert not torch.allclose(
            logits.grad, torch.zeros_like(logits.grad)
        ), "Gradient must be non-zero (this is the VF clipping bug)"

        # All batch items should have gradients
        for i in range(batch_size):
            row_grad = logits.grad[i]
            assert torch.any(torch.abs(row_grad) > 1e-6), \
                f"Batch item {i} should have non-negligible gradients"

        # Gradient should be finite
        assert torch.all(torch.isfinite(logits.grad)), "Gradient should be finite"

        print("✓ Gradient flow test passed - gradients successfully backpropagate")
        print(f"  Gradient norm: {logits.grad.norm():.6f}")
        print(f"  Gradient mean: {logits.grad.mean():.6f}")
        print(f"  Gradient std: {logits.grad.std():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
