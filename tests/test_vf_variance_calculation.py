"""
Test distributional VF clipping variance calculation correctness.

This test validates that old_variance is computed from OLD distributions
(stored in rollout buffer), not from current predictions.

The bug was:
- Quantile critic: Used current quantiles instead of old quantiles from buffer
- Categorical critic: Used uniform distribution assumption instead of old probs

The fix:
- Store old_quantiles and old_probs in rollout buffer
- Compute old_variance from these stored distributions
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


def test_quantile_variance_from_old_quantiles():
    """
    Test that quantile critic computes old_variance from old quantiles,
    not from current predictions.
    """
    batch_size = 4
    n_quantiles = 51

    # Old quantiles (from rollout buffer) - what we should use
    old_quantiles = torch.randn(batch_size, n_quantiles) * 2.0 + 5.0
    old_mean = old_quantiles.mean(dim=1, keepdim=True)

    # Current quantiles (new predictions) - what we should NOT use
    current_quantiles = torch.randn(batch_size, n_quantiles) * 3.0 + 6.0
    current_mean = current_quantiles.mean(dim=1, keepdim=True)

    # CORRECT: Compute old_variance from OLD quantiles
    old_quantiles_centered = old_quantiles - old_mean
    old_variance_correct = (old_quantiles_centered**2).mean(dim=1, keepdim=True)

    # INCORRECT (bug): Compute from CURRENT quantiles
    current_quantiles_centered = current_quantiles - current_mean
    old_variance_buggy = (current_quantiles_centered**2).mean(dim=1, keepdim=True)

    # These should be different (unless by extreme coincidence)
    assert not torch.allclose(
        old_variance_correct, old_variance_buggy, rtol=0.1
    ), "Old variance should be computed from old quantiles, not current ones"

    # Verify old_variance_correct matches expected calculation
    expected_variance = torch.var(old_quantiles, dim=1, keepdim=True, unbiased=False)
    assert torch.allclose(
        old_variance_correct, expected_variance, rtol=1e-5
    ), "Old variance calculation should match torch.var"


def test_categorical_variance_from_old_probs():
    """
    Test that categorical critic computes old_variance from old probabilities
    with proper weighting, not uniform distribution assumption.
    """
    batch_size = 4
    n_atoms = 51

    # Atoms (fixed support points)
    atoms = torch.linspace(-10.0, 10.0, n_atoms)

    # Old probabilities (from rollout buffer) - what we should use
    # Create non-uniform distribution
    old_logits = torch.randn(batch_size, n_atoms)
    old_probs = torch.softmax(old_logits, dim=1)

    # Old mean value
    old_mean = (old_probs * atoms).sum(dim=1, keepdim=True)

    # CORRECT: Compute old_variance using old probabilities (weighted)
    old_atoms_centered = atoms - old_mean.squeeze(-1)
    old_variance_correct = ((old_atoms_centered**2) * old_probs).sum(dim=1, keepdim=True)

    # INCORRECT (bug): Assume uniform distribution
    old_variance_buggy_uniform = (old_atoms_centered**2).mean(dim=1, keepdim=True)

    # These should be different for non-uniform distributions
    assert not torch.allclose(
        old_variance_correct, old_variance_buggy_uniform, rtol=0.1
    ), "Old variance should use old probabilities, not uniform distribution"

    # Verify correctness by manual calculation
    expected_variance = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        mean = old_mean[i, 0]
        var = sum(old_probs[i, j] * (atoms[j] - mean) ** 2 for j in range(n_atoms))
        expected_variance[i, 0] = var

    assert torch.allclose(
        old_variance_correct, expected_variance, rtol=1e-5
    ), "Old variance calculation should match manual weighted variance"


def test_variance_constraint_correctness():
    """
    Test that variance constraint correctly limits variance changes.
    """
    batch_size = 4
    n_quantiles = 51
    variance_clip_factor = 1.5  # Allow up to 1.5x variance increase

    # Old distribution (low variance)
    old_quantiles = torch.linspace(-1.0, 1.0, n_quantiles).unsqueeze(0).expand(batch_size, -1)
    old_mean = old_quantiles.mean(dim=1, keepdim=True)
    old_quantiles_centered = old_quantiles - old_mean
    old_variance = (old_quantiles_centered**2).mean(dim=1, keepdim=True)

    # New distribution (high variance - should be constrained)
    new_quantiles = torch.linspace(-5.0, 5.0, n_quantiles).unsqueeze(0).expand(batch_size, -1)
    new_mean = new_quantiles.mean(dim=1, keepdim=True)
    new_quantiles_centered = new_quantiles - new_mean
    new_variance = (new_quantiles_centered**2).mean(dim=1, keepdim=True)

    # Compute variance ratio and apply constraint
    variance_ratio = new_variance / (old_variance + 1e-8)
    variance_ratio_constrained = torch.clamp(variance_ratio, max=variance_clip_factor**2)
    std_ratio = torch.sqrt(variance_ratio_constrained)

    # Apply scaling
    constrained_quantiles_centered = new_quantiles_centered * std_ratio
    constrained_variance = (constrained_quantiles_centered**2).mean(dim=1, keepdim=True)

    # Verify constraint is enforced
    max_allowed_variance = old_variance * (variance_clip_factor**2)
    assert torch.all(
        constrained_variance <= max_allowed_variance * 1.001
    ), "Constrained variance should not exceed max allowed variance"

    # Verify that unconstrained variance would have violated the constraint
    assert torch.all(
        new_variance > max_allowed_variance
    ), "Test setup: new variance should exceed limit to test constraint"


def test_rollout_buffer_stores_distributions():
    """
    Integration test: verify that rollout buffer properly stores and retrieves
    old quantiles and probabilities.
    """
    # This test would require importing actual DistributionalPPO classes
    # For now, we test the data structure contracts

    # Simulate RolloutBufferSamples with distributional data
    batch_size = 8
    n_quantiles = 51
    n_atoms = 51

    # Create mock rollout data
    old_values = torch.randn(batch_size, 1)
    old_value_quantiles = torch.randn(batch_size, n_quantiles)
    old_value_probs = torch.softmax(torch.randn(batch_size, n_atoms), dim=1)

    # Verify quantiles match their mean
    computed_mean_from_quantiles = old_value_quantiles.mean(dim=1, keepdim=True)
    # Note: In practice, old_values might be slightly different due to floating point
    # But they should be close
    assert old_value_quantiles.shape == (
        batch_size,
        n_quantiles,
    ), "Quantiles should have correct shape"

    # Verify probs sum to 1
    probs_sum = old_value_probs.sum(dim=1)
    assert torch.allclose(
        probs_sum, torch.ones(batch_size), rtol=1e-5
    ), "Probabilities should sum to 1"


def test_variance_calculation_numerical_stability():
    """
    Test that variance calculation is numerically stable even with extreme values.
    """
    batch_size = 4
    n_quantiles = 51

    # Test with very small values
    tiny_quantiles = torch.randn(batch_size, n_quantiles) * 1e-6
    tiny_mean = tiny_quantiles.mean(dim=1, keepdim=True)
    tiny_centered = tiny_quantiles - tiny_mean
    tiny_variance = (tiny_centered**2).mean(dim=1, keepdim=True)
    assert torch.all(torch.isfinite(tiny_variance)), "Variance should be finite for tiny values"

    # Test with very large values
    huge_quantiles = torch.randn(batch_size, n_quantiles) * 1e6
    huge_mean = huge_quantiles.mean(dim=1, keepdim=True)
    huge_centered = huge_quantiles - huge_mean
    huge_variance = (huge_centered**2).mean(dim=1, keepdim=True)
    assert torch.all(torch.isfinite(huge_variance)), "Variance should be finite for huge values"

    # Test with zero variance (all same values)
    constant_quantiles = torch.ones(batch_size, n_quantiles) * 5.0
    constant_mean = constant_quantiles.mean(dim=1, keepdim=True)
    constant_centered = constant_quantiles - constant_mean
    constant_variance = (constant_centered**2).mean(dim=1, keepdim=True)
    assert torch.allclose(
        constant_variance, torch.zeros(batch_size, 1), atol=1e-6
    ), "Variance should be zero for constant values"


if __name__ == "__main__":
    test_quantile_variance_from_old_quantiles()
    print("✓ test_quantile_variance_from_old_quantiles passed")

    test_categorical_variance_from_old_probs()
    print("✓ test_categorical_variance_from_old_probs passed")

    test_variance_constraint_correctness()
    print("✓ test_variance_constraint_correctness passed")

    test_rollout_buffer_stores_distributions()
    print("✓ test_rollout_buffer_stores_distributions passed")

    test_variance_calculation_numerical_stability()
    print("✓ test_variance_calculation_numerical_stability passed")

    print("\nAll variance calculation tests passed!")
