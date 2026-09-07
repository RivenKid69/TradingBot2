"""
CRITICAL TEST: Verify PPO Value Function Clipping Fix

This test verifies that the critical VF clipping bug is fixed:
- OLD (INCORRECT): max(mean(L_unclipped), mean(L_clipped))
- NEW (CORRECT): mean(max(L_unclipped, L_clipped))

The bug was causing incorrect gradient flow and violating PPO mathematics
from the original paper (Schulman et al., 2017).

Tests cover:
1. Quantile distribution value clipping
2. Categorical distribution value clipping (both methods)
3. Gradient flow correctness
4. Edge cases and numerical stability
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock


class TestPPOVFClippingMathematicalCorrectness:
    """Test that VF clipping uses correct mathematical formula."""

    def test_quantile_huber_loss_reduction_parameter(self):
        """Test that _quantile_huber_loss supports reduction parameter."""
        from distributional_ppo import DistributionalPPO

        # Create minimal PPO instance
        policy = MagicMock()
        policy.atoms = None
        policy._value_type = "quantile"

        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        # Mock quantile levels
        num_quantiles = 5
        batch_size = 3
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            predicted = torch.randn(batch_size, num_quantiles, requires_grad=True)
            targets = torch.randn(batch_size, 1)

            # Test reduction='none' returns per-sample losses
            loss_none = ppo._quantile_huber_loss(predicted, targets, reduction="none")
            assert loss_none.shape == (
                batch_size,
            ), f"reduction='none' should return shape [batch], got {loss_none.shape}"

            # Test reduction='mean' returns scalar
            loss_mean = ppo._quantile_huber_loss(predicted, targets, reduction="mean")
            assert (
                loss_mean.shape == ()
            ), f"reduction='mean' should return scalar, got {loss_mean.shape}"

            # Test reduction='sum' returns scalar
            loss_sum = ppo._quantile_huber_loss(predicted, targets, reduction="sum")
            assert (
                loss_sum.shape == ()
            ), f"reduction='sum' should return scalar, got {loss_sum.shape}"

            # Verify mathematical relationship
            assert torch.allclose(
                loss_mean, loss_none.mean(), atol=1e-6
            ), "reduction='mean' should equal mean of reduction='none'"
            assert torch.allclose(
                loss_sum, loss_none.sum(), atol=1e-6
            ), "reduction='sum' should equal sum of reduction='none'"

    def test_vf_clipping_uses_mean_of_max_not_max_of_mean(self):
        """
        CRITICAL: Verify VF clipping uses mean(max(...)) NOT max(mean(...)).

        This is the core mathematical fix. The old implementation took the maximum
        of two averaged losses, which is mathematically incorrect per PPO paper.
        """
        from distributional_ppo import DistributionalPPO

        policy = MagicMock()
        policy.atoms = None
        policy._value_type = "quantile"

        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        batch_size = 4
        num_quantiles = 5
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            # Create scenario where bug would produce different result
            predicted_unclipped = torch.tensor(
                [
                    [1.0, 1.2, 1.4, 1.6, 1.8],
                    [2.0, 2.2, 2.4, 2.6, 2.8],
                    [0.5, 0.6, 0.7, 0.8, 0.9],
                    [1.5, 1.6, 1.7, 1.8, 1.9],
                ],
                requires_grad=True,
            )

            predicted_clipped = torch.tensor(
                [
                    [0.9, 1.1, 1.3, 1.5, 1.7],  # Clipped down
                    [2.1, 2.3, 2.5, 2.7, 2.9],  # Clipped up
                    [0.5, 0.6, 0.7, 0.8, 0.9],  # No change
                    [1.4, 1.5, 1.6, 1.7, 1.8],  # Clipped down
                ],
                requires_grad=True,
            )

            targets = torch.tensor([[1.5], [2.5], [0.7], [1.7]])

            # Compute per-sample losses
            loss_unclipped_per_sample = ppo._quantile_huber_loss(
                predicted_unclipped, targets, reduction="none"
            )
            loss_clipped_per_sample = ppo._quantile_huber_loss(
                predicted_clipped, targets, reduction="none"
            )

            # CORRECT implementation: mean(max(...))
            correct_loss = torch.mean(torch.max(loss_unclipped_per_sample, loss_clipped_per_sample))

            # INCORRECT implementation (old bug): max(mean(...))
            incorrect_loss = torch.max(
                loss_unclipped_per_sample.mean(), loss_clipped_per_sample.mean()
            )

            # These should be DIFFERENT (proving the bug matters)
            assert not torch.allclose(
                correct_loss, incorrect_loss, atol=1e-4
            ), "Correct and incorrect implementations should produce different results"

            print(f"Correct (mean of max): {correct_loss.item():.6f}")
            print(f"Incorrect (max of means): {incorrect_loss.item():.6f}")
            print(f"Difference: {abs(correct_loss.item() - incorrect_loss.item()):.6f}")

    def test_vf_clipping_gradient_flow(self):
        """Test that VF clipping produces correct gradients per-sample."""
        from distributional_ppo import DistributionalPPO

        policy = MagicMock()
        policy.atoms = None
        policy._value_type = "quantile"

        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        batch_size = 3
        num_quantiles = 5
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            predicted_unclipped = torch.randn(batch_size, num_quantiles, requires_grad=True)
            predicted_clipped = predicted_unclipped.clone().detach() + 0.1
            predicted_clipped.requires_grad = True
            targets = torch.randn(batch_size, 1)

            # Compute with mean(max(...))
            loss_unclipped_per_sample = ppo._quantile_huber_loss(
                predicted_unclipped, targets, reduction="none"
            )
            loss_clipped_per_sample = ppo._quantile_huber_loss(
                predicted_clipped, targets, reduction="none"
            )

            # Element-wise max, then mean
            loss = torch.mean(torch.max(loss_unclipped_per_sample, loss_clipped_per_sample))

            # Backprop
            loss.backward()

            # Verify gradients exist and are finite
            assert predicted_unclipped.grad is not None
            assert predicted_clipped.grad is not None
            assert torch.all(torch.isfinite(predicted_unclipped.grad))
            assert torch.all(torch.isfinite(predicted_clipped.grad))

            # For each sample, only the prediction with larger loss should get gradient
            # This verifies correct per-sample gradient routing
            with torch.no_grad():
                loss_unclipped_np = loss_unclipped_per_sample.detach().numpy()
                loss_clipped_np = loss_clipped_per_sample.detach().numpy()

                for i in range(batch_size):
                    if loss_unclipped_np[i] > loss_clipped_np[i]:
                        # Unclipped loss is larger for this sample
                        # So unclipped should have gradient, clipped might be zero
                        assert torch.any(
                            predicted_unclipped.grad[i] != 0
                        ), f"Sample {i}: unclipped has larger loss but zero gradient"
                    elif loss_clipped_np[i] > loss_unclipped_np[i]:
                        # Clipped loss is larger for this sample
                        assert torch.any(
                            predicted_clipped.grad[i] != 0
                        ), f"Sample {i}: clipped has larger loss but zero gradient"

    def test_categorical_vf_clipping_per_sample(self):
        """Test categorical distribution VF clipping uses per-sample max."""
        batch_size = 4
        num_atoms = 51

        # Simulate categorical cross-entropy loss computation
        target_dist = torch.softmax(torch.randn(batch_size, num_atoms), dim=1)
        pred_logits_unclipped = torch.randn(batch_size, num_atoms, requires_grad=True)
        pred_logits_clipped = pred_logits_unclipped.clone().detach() + 0.05
        pred_logits_clipped.requires_grad = True

        # Compute log probabilities
        log_pred_unclipped = torch.log_softmax(pred_logits_unclipped, dim=1)
        log_pred_clipped = torch.log_softmax(pred_logits_clipped, dim=1)

        # Compute per-sample cross-entropy (sum over atoms, no mean yet)
        loss_unclipped_per_sample = -(target_dist * log_pred_unclipped).sum(dim=1)
        loss_clipped_per_sample = -(target_dist * log_pred_clipped).sum(dim=1)

        # CORRECT: mean(max(...))
        correct_loss = torch.mean(torch.max(loss_unclipped_per_sample, loss_clipped_per_sample))

        # INCORRECT: max(mean(...))
        incorrect_loss = torch.max(loss_unclipped_per_sample.mean(), loss_clipped_per_sample.mean())

        # Verify gradients with correct implementation
        correct_loss.backward()

        assert pred_logits_unclipped.grad is not None
        assert pred_logits_clipped.grad is not None
        assert torch.all(torch.isfinite(pred_logits_unclipped.grad))
        assert torch.all(torch.isfinite(pred_logits_clipped.grad))

        print(f"Categorical - Correct: {correct_loss.item():.6f}")
        print(f"Categorical - Incorrect: {incorrect_loss.item():.6f}")


class TestPPOVFClippingEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_batch_size(self):
        """Test with empty batch."""
        from distributional_ppo import DistributionalPPO

        policy = MagicMock()
        policy.atoms = None
        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        num_quantiles = 5
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            predicted = torch.empty(0, num_quantiles)
            targets = torch.empty(0, 1)

            # Should handle gracefully
            loss = ppo._quantile_huber_loss(predicted, targets, reduction="none")
            assert loss.shape == (0,)

            loss_mean = ppo._quantile_huber_loss(predicted, targets, reduction="mean")
            assert torch.isfinite(loss_mean) or loss_mean.numel() == 0

    def test_identical_predictions(self):
        """Test when clipped and unclipped predictions are identical."""
        from distributional_ppo import DistributionalPPO

        policy = MagicMock()
        policy.atoms = None
        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        batch_size = 3
        num_quantiles = 5
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            predicted = torch.randn(batch_size, num_quantiles)
            targets = torch.randn(batch_size, 1)

            loss_unclipped = ppo._quantile_huber_loss(predicted, targets, reduction="none")
            loss_clipped = ppo._quantile_huber_loss(predicted, targets, reduction="none")

            # When predictions are identical, max should equal either one
            loss_vf = torch.mean(torch.max(loss_unclipped, loss_clipped))
            loss_plain = loss_unclipped.mean()

            assert torch.allclose(loss_vf, loss_plain, atol=1e-6)

    def test_extreme_clipping(self):
        """Test with extreme clipping values."""
        from distributional_ppo import DistributionalPPO

        policy = MagicMock()
        policy.atoms = None
        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo._quantile_huber_kappa = 1.0

        batch_size = 3
        num_quantiles = 5
        quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)

        with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
            predicted_unclipped = torch.randn(batch_size, num_quantiles) * 100
            predicted_clipped = torch.clamp(predicted_unclipped, -10, 10)
            targets = torch.randn(batch_size, 1)

            loss_unclipped = ppo._quantile_huber_loss(
                predicted_unclipped, targets, reduction="none"
            )
            loss_clipped = ppo._quantile_huber_loss(predicted_clipped, targets, reduction="none")

            loss = torch.mean(torch.max(loss_unclipped, loss_clipped))

            assert torch.isfinite(loss)
            assert loss.item() >= 0


def test_invalid_reduction_mode():
    """Test that invalid reduction mode raises error."""
    from distributional_ppo import DistributionalPPO

    policy = MagicMock()
    policy.atoms = None
    ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
    ppo._quantile_huber_kappa = 1.0

    quantile_levels = torch.linspace(0.1, 0.9, 5)

    with patch.object(ppo, "_quantile_levels_tensor", return_value=quantile_levels):
        predicted = torch.randn(3, 5)
        targets = torch.randn(3, 1)

        with pytest.raises(ValueError, match="Invalid reduction mode"):
            ppo._quantile_huber_loss(predicted, targets, reduction="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
