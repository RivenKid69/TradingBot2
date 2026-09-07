"""
Test suite for categorical critic numerical stability improvements.

Tests the torch.clamp approach for numerical stability in categorical critic
VF clipping (Problem #1 from TWIN_CRITICS_NUMERICAL_STABILITY_ANALYSIS.md).

Author: Sultan Suyunbayev
Date: 2025-11-22
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np
from unittest.mock import MagicMock, patch

# Import DistributionalPPO for testing
from distributional_ppo import DistributionalPPO


class TestCategoricalCriticNumericalStability:
    """Test categorical critic VF clipping numerical stability improvements."""

    @pytest.fixture
    def setup_categorical_ppo(self):
        """Setup PPO with categorical critic for testing."""
        # Create minimal PPO instance with mocked dependencies
        # Following the pattern from test_actual_ppo_vf_clipping_integration.py

        policy = MagicMock()
        policy.atoms = torch.linspace(-10.0, 10.0, 21)  # Categorical critic
        policy._value_type = "categorical"

        ppo = DistributionalPPO(policy=policy, env=None, verbose=0)
        ppo.clip_range_vf = 0.7  # Enable VF clipping
        ppo._use_twin_critics = True

        return ppo

    def test_torch_clamp_prevents_invalid_probabilities(self, setup_categorical_ppo):
        """Test that torch.clamp prevents probabilities outside [1e-8, 1.0]."""
        ppo = setup_categorical_ppo

        # Create test probabilities with edge cases
        batch_size = 4
        num_atoms = 21

        # Test case 1: Very small probabilities (< 1e-8)
        probs_small = torch.full((batch_size, num_atoms), 1e-10)
        probs_small = probs_small / probs_small.sum(dim=1, keepdim=True)  # Normalize

        # Test case 2: Probabilities close to 1.0
        probs_large = torch.zeros((batch_size, num_atoms))
        probs_large[:, 0] = 0.99
        probs_large[:, 1:] = 0.01 / (num_atoms - 1)

        # Test case 3: Mix of small and large
        probs_mixed = torch.rand((batch_size, num_atoms))
        probs_mixed = probs_mixed / probs_mixed.sum(dim=1, keepdim=True)
        probs_mixed[0, :5] = 1e-12  # Very small
        probs_mixed[0, :] = probs_mixed[0, :] / probs_mixed[0, :].sum()

        for probs in [probs_small, probs_large, probs_mixed]:
            # Apply torch.clamp (as in the fix)
            probs_safe = torch.clamp(probs, min=1e-8, max=1.0)

            # Verify all values are in valid range
            assert torch.all(probs_safe >= 1e-8), "Found probabilities below 1e-8"
            assert torch.all(probs_safe <= 1.0), "Found probabilities above 1.0"

            # Verify log is safe (no NaN or Inf)
            log_probs = torch.log(probs_safe)
            assert torch.all(torch.isfinite(log_probs)), "Found NaN or Inf in log probabilities"

    def test_categorical_vf_clipping_numerical_stability(self, setup_categorical_ppo):
        """Test categorical VF clipping with edge case probabilities."""
        ppo = setup_categorical_ppo
        policy = ppo.policy

        batch_size = 8
        latent_dim = 32
        num_atoms = 21

        # Create mock latent features
        latent_vf = torch.randn(batch_size, latent_dim, requires_grad=True)

        # Create target distribution with some very small probabilities
        target_distribution = torch.rand(batch_size, num_atoms)
        target_distribution[:, ::2] = 1e-10  # Make every other probability very small
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)

        # Create old quantiles/probs
        old_quantiles_1 = torch.randn(batch_size, num_atoms)
        old_quantiles_2 = torch.randn(batch_size, num_atoms)
        old_probs_1 = torch.rand(batch_size, num_atoms)
        old_probs_1 = old_probs_1 / old_probs_1.sum(dim=1, keepdim=True)
        old_probs_2 = torch.rand(batch_size, num_atoms)
        old_probs_2 = old_probs_2 / old_probs_2.sum(dim=1, keepdim=True)

        # Make some old probs very small (edge case)
        old_probs_1[0, :5] = 1e-12
        old_probs_1[0, :] = old_probs_1[0, :] / old_probs_1[0, :].sum()
        old_probs_2[1, :3] = 1e-11
        old_probs_2[1, :] = old_probs_2[1, :] / old_probs_2[1, :].sum()

        # Compute VF clipping loss
        try:
            clipped_loss_avg, loss_c1, loss_c2, loss_unclipped = ppo._twin_critics_vf_clipping_loss(
                latent_vf=latent_vf,
                targets=None,  # Categorical critic doesn't use targets
                old_quantiles_critic1=old_quantiles_1,
                old_quantiles_critic2=old_quantiles_2,
                clip_delta=0.7,
                reduction="mean",
                old_probs_critic1=old_probs_1,
                old_probs_critic2=old_probs_2,
                target_distribution=target_distribution,
            )

            # Verify all losses are finite (no NaN or Inf)
            assert torch.isfinite(clipped_loss_avg), f"clipped_loss_avg is not finite: {clipped_loss_avg}"
            assert torch.isfinite(loss_c1), f"loss_c1 is not finite: {loss_c1}"
            assert torch.isfinite(loss_c2), f"loss_c2 is not finite: {loss_c2}"
            assert torch.isfinite(loss_unclipped), f"loss_unclipped is not finite: {loss_unclipped}"

            # Verify losses are non-negative (cross-entropy should be >= 0)
            assert clipped_loss_avg >= 0, f"Negative clipped_loss_avg: {clipped_loss_avg}"
            assert loss_c1 >= 0, f"Negative loss_c1: {loss_c1}"
            assert loss_c2 >= 0, f"Negative loss_c2: {loss_c2}"
            assert loss_unclipped >= 0, f"Negative loss_unclipped: {loss_unclipped}"

        except Exception as e:
            pytest.fail(f"VF clipping failed with edge case probabilities: {e}")

    def test_gradient_flow_through_categorical_vf_clipping(self, setup_categorical_ppo):
        """Test that gradients flow correctly through categorical VF clipping."""
        ppo = setup_categorical_ppo

        batch_size = 4
        latent_dim = 32
        num_atoms = 21

        # Create latent features with gradient tracking
        latent_vf = torch.randn(batch_size, latent_dim, requires_grad=True)

        # Create target distribution
        target_distribution = torch.rand(batch_size, num_atoms)
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)

        # Create old quantiles/probs
        old_quantiles_1 = torch.randn(batch_size, num_atoms)
        old_quantiles_2 = torch.randn(batch_size, num_atoms)
        old_probs_1 = torch.rand(batch_size, num_atoms)
        old_probs_1 = old_probs_1 / old_probs_1.sum(dim=1, keepdim=True)
        old_probs_2 = torch.rand(batch_size, num_atoms)
        old_probs_2 = old_probs_2 / old_probs_2.sum(dim=1, keepdim=True)

        # Compute loss
        clipped_loss_avg, _, _, _ = ppo._twin_critics_vf_clipping_loss(
            latent_vf=latent_vf,
            targets=None,
            old_quantiles_critic1=old_quantiles_1,
            old_quantiles_critic2=old_quantiles_2,
            clip_delta=0.7,
            reduction="mean",
            old_probs_critic1=old_probs_1,
            old_probs_critic2=old_probs_2,
            target_distribution=target_distribution,
        )

        # Backward pass
        clipped_loss_avg.backward()

        # Verify gradients exist and are finite
        assert latent_vf.grad is not None, "No gradient computed for latent_vf"
        assert torch.all(torch.isfinite(latent_vf.grad)), "Gradients contain NaN or Inf"

        # Verify gradients are non-zero (network is learning)
        grad_norm = torch.norm(latent_vf.grad)
        assert grad_norm > 0, f"Gradient norm is zero: {grad_norm}"

    def test_loss_consistency_across_reduction_modes(self, setup_categorical_ppo):
        """Test that loss is consistent across different reduction modes."""
        ppo = setup_categorical_ppo

        batch_size = 8
        latent_dim = 32
        num_atoms = 21

        # Create inputs
        latent_vf = torch.randn(batch_size, latent_dim)
        target_distribution = torch.rand(batch_size, num_atoms)
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)
        old_quantiles_1 = torch.randn(batch_size, num_atoms)
        old_quantiles_2 = torch.randn(batch_size, num_atoms)
        old_probs_1 = torch.rand(batch_size, num_atoms)
        old_probs_1 = old_probs_1 / old_probs_1.sum(dim=1, keepdim=True)
        old_probs_2 = torch.rand(batch_size, num_atoms)
        old_probs_2 = old_probs_2 / old_probs_2.sum(dim=1, keepdim=True)

        # Compute with reduction='none'
        _, loss_c1_none, loss_c2_none, _ = ppo._twin_critics_vf_clipping_loss(
            latent_vf=latent_vf,
            targets=None,
            old_quantiles_critic1=old_quantiles_1,
            old_quantiles_critic2=old_quantiles_2,
            clip_delta=0.7,
            reduction="none",
            old_probs_critic1=old_probs_1,
            old_probs_critic2=old_probs_2,
            target_distribution=target_distribution,
        )

        # Compute with reduction='mean'
        _, loss_c1_mean, loss_c2_mean, _ = ppo._twin_critics_vf_clipping_loss(
            latent_vf=latent_vf,
            targets=None,
            old_quantiles_critic1=old_quantiles_1,
            old_quantiles_critic2=old_quantiles_2,
            clip_delta=0.7,
            reduction="mean",
            old_probs_critic1=old_probs_1,
            old_probs_critic2=old_probs_2,
            target_distribution=target_distribution,
        )

        # Verify consistency: mean(loss_none) ≈ loss_mean
        expected_c1_mean = loss_c1_none.mean()
        expected_c2_mean = loss_c2_none.mean()

        torch.testing.assert_close(
            loss_c1_mean,
            expected_c1_mean,
            rtol=1e-5,
            atol=1e-8,
            msg=f"Critic 1 loss inconsistent: {loss_c1_mean} vs {expected_c1_mean}",
        )
        torch.testing.assert_close(
            loss_c2_mean,
            expected_c2_mean,
            rtol=1e-5,
            atol=1e-8,
            msg=f"Critic 2 loss inconsistent: {loss_c2_mean} vs {expected_c2_mean}",
        )

    def test_no_nan_with_extreme_edge_cases(self, setup_categorical_ppo):
        """Test that no NaN occurs even with extreme edge cases."""
        ppo = setup_categorical_ppo

        batch_size = 4
        latent_dim = 32
        num_atoms = 21

        latent_vf = torch.randn(batch_size, latent_dim, requires_grad=True)

        # Extreme edge cases
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

            # Case 4: Extreme small values (would cause NaN with probs + eps)
            torch.cat([
                torch.ones((batch_size, 1)) * 0.999999,
                torch.ones((batch_size, num_atoms - 1)) * (1e-15)
            ], dim=1),
        ]

        for i, target_dist in enumerate(edge_cases):
            # Normalize to ensure valid probability distribution
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)

            old_quantiles_1 = torch.randn(batch_size, num_atoms)
            old_quantiles_2 = torch.randn(batch_size, num_atoms)
            old_probs_1 = target_dist.clone()
            old_probs_2 = target_dist.clone()

            # Compute loss
            clipped_loss_avg, loss_c1, loss_c2, loss_unclipped = ppo._twin_critics_vf_clipping_loss(
                latent_vf=latent_vf,
                targets=None,
                old_quantiles_critic1=old_quantiles_1,
                old_quantiles_critic2=old_quantiles_2,
                clip_delta=0.7,
                reduction="mean",
                old_probs_critic1=old_probs_1,
                old_probs_critic2=old_probs_2,
                target_distribution=target_dist,
            )

            # Verify no NaN or Inf
            assert torch.isfinite(clipped_loss_avg), f"Case {i+1}: clipped_loss_avg is not finite"
            assert torch.isfinite(loss_c1), f"Case {i+1}: loss_c1 is not finite"
            assert torch.isfinite(loss_c2), f"Case {i+1}: loss_c2 is not finite"
            assert torch.isfinite(loss_unclipped), f"Case {i+1}: loss_unclipped is not finite"

            # Test backward pass
            latent_vf.grad = None
            clipped_loss_avg.backward()
            assert torch.all(torch.isfinite(latent_vf.grad)), f"Case {i+1}: Gradients contain NaN or Inf"

    def test_torch_clamp_vs_addition_equivalence(self):
        """Test that torch.clamp approach gives similar results to addition for normal cases."""
        batch_size = 8
        num_atoms = 21
        epsilon = 1e-8

        # Create normal probabilities (not edge cases)
        probs = torch.rand((batch_size, num_atoms))
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Old approach: probs + epsilon
        log_probs_old = torch.log(probs + epsilon)

        # New approach: torch.clamp
        probs_safe = torch.clamp(probs, min=epsilon, max=1.0)
        log_probs_new = torch.log(probs_safe)

        # For normal probabilities (> epsilon), results should be nearly identical
        torch.testing.assert_close(
            log_probs_new,
            log_probs_old,
            rtol=1e-6,
            atol=1e-9,
            msg="torch.clamp and addition approaches differ for normal probabilities",
        )

    def test_torch_clamp_handles_edge_cases_better(self):
        """Test that torch.clamp handles edge cases better than addition."""
        batch_size = 4
        num_atoms = 21
        epsilon = 1e-8

        # Edge case 1: Very small probabilities (would be changed by addition)
        probs_small = torch.zeros((batch_size, num_atoms))
        probs_small[:, 0] = 0.99
        probs_small[:, 1:] = 0.01 / (num_atoms - 1)
        # Make some probabilities smaller than epsilon
        probs_small[0, 5:10] = 1e-12
        probs_small[0, :] = probs_small[0, :] / probs_small[0, :].sum()  # Renormalize

        # Old approach: probs + epsilon
        # This CHANGES the actual probability values even when they're already > epsilon
        probs_old = probs_small + epsilon
        # For values already > epsilon, addition changes them unnecessarily
        mask_already_safe = probs_small > epsilon
        if torch.any(mask_already_safe):
            # These values got modified even though they were already safe
            assert torch.any(probs_old[mask_already_safe] != probs_small[mask_already_safe]), \
                "Old approach modifies safe probabilities unnecessarily"

        # New approach: torch.clamp
        # This PRESERVES values that are already >= epsilon
        probs_new = torch.clamp(probs_small, min=epsilon, max=1.0)
        # Values already >= epsilon should be unchanged
        if torch.any(mask_already_safe):
            torch.testing.assert_close(
                probs_new[mask_already_safe],
                probs_small[mask_already_safe],
                rtol=1e-9,
                atol=1e-12,
                msg="New approach should preserve safe probabilities"
            )

        # Edge case 2: Ensure both approaches protect against very small values
        probs_tiny = torch.ones((batch_size, num_atoms)) * 1e-15
        probs_tiny = probs_tiny / probs_tiny.sum(dim=1, keepdim=True)

        # Both approaches should make these safe for log
        probs_old_tiny = probs_tiny + epsilon
        probs_new_tiny = torch.clamp(probs_tiny, min=epsilon, max=1.0)

        # Both should be >= epsilon
        assert torch.all(probs_old_tiny >= epsilon), "Old approach should protect tiny values"
        assert torch.all(probs_new_tiny >= epsilon), "New approach should protect tiny values"

        # Both should allow safe log
        log_old = torch.log(probs_old_tiny)
        log_new = torch.log(probs_new_tiny)
        assert torch.all(torch.isfinite(log_old)), "Old approach log should be finite"
        assert torch.all(torch.isfinite(log_new)), "New approach log should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
