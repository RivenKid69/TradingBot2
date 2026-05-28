"""
Quick smoke test for categorical VF clipping implementation.
This can be run without pytest to verify basic functionality.
"""

import pytest
torch = pytest.importorskip("torch")
from distributional_ppo import DistributionalPPO


def test_projection_basic():
    """Basic smoke test for projection function."""
    print("Testing _project_categorical_distribution...")

    algo = DistributionalPPO.__new__(DistributionalPPO)

    # Simple test: uniform distribution, shifted atoms
    batch_size = 2
    num_atoms = 5
    probs = torch.ones(batch_size, num_atoms) / num_atoms
    target_atoms = torch.linspace(-2.0, 2.0, num_atoms)
    source_atoms = target_atoms + 0.5  # Shift by 0.5

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    # Verify output shape and validity
    assert projected.shape == probs.shape, f"Shape mismatch: {projected.shape} != {probs.shape}"

    total_prob = projected.sum(dim=1)
    assert torch.allclose(total_prob, torch.ones(batch_size), atol=1e-4), \
        f"Probability doesn't sum to 1: {total_prob}"

    assert torch.all(projected >= 0.0), "Negative probabilities detected"

    # Check mean is approximately preserved
    original_mean = (probs * source_atoms).sum(dim=1)
    projected_mean = (projected * target_atoms).sum(dim=1)

    print(f"  Original mean: {original_mean}")
    print(f"  Projected mean: {projected_mean}")
    print(f"  Difference: {torch.abs(original_mean - projected_mean)}")

    assert torch.allclose(original_mean, projected_mean, atol=0.3), \
        f"Mean not preserved: {original_mean} vs {projected_mean}"

    print("  ✓ Projection test passed!")


def test_projection_identity():
    """Test projection is identity when atoms don't change."""
    print("\nTesting projection identity property...")

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 3
    num_atoms = 7
    atoms = torch.linspace(-3.0, 3.0, num_atoms)

    # Random distribution
    logits = torch.randn(batch_size, num_atoms)
    probs = torch.softmax(logits, dim=1)

    # Project to same atoms
    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=atoms, target_atoms=atoms
    )

    diff = torch.abs(projected - probs).max()
    print(f"  Max difference from identity: {diff}")

    assert torch.allclose(projected, probs, atol=1e-3), \
        f"Identity projection failed, max diff: {diff}"

    print("  ✓ Identity test passed!")


def test_projection_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    algo = DistributionalPPO.__new__(DistributionalPPO)

    # Single atom
    probs_single = torch.tensor([[1.0]])
    atoms_single = torch.tensor([0.0])
    projected = algo._project_categorical_distribution(
        probs=probs_single, source_atoms=atoms_single, target_atoms=atoms_single
    )
    assert torch.allclose(projected, probs_single), "Single atom test failed"
    print("  ✓ Single atom test passed!")

    # Two atoms
    probs_two = torch.tensor([[0.3, 0.7]])
    atoms_two = torch.tensor([-1.0, 1.0])
    source_two = atoms_two + 0.2
    projected = algo._project_categorical_distribution(
        probs=probs_two, source_atoms=source_two, target_atoms=atoms_two
    )
    assert projected.shape == probs_two.shape, "Two atoms shape mismatch"
    assert torch.allclose(projected.sum(dim=1), torch.ones(1), atol=1e-5), \
        "Two atoms probability sum failed"
    print("  ✓ Two atoms test passed!")


def test_code_has_vf_clipping():
    """Verify the code has VF clipping for categorical."""
    print("\nVerifying VF clipping code structure...")

    import inspect
    import distributional_ppo

    source = inspect.getsource(distributional_ppo.DistributionalPPO.train)

    # Check for VF clipping keywords
    checks = {
        "critic_loss_unclipped in categorical section": "critic_loss_unclipped_per_sample" in source,
        "critic_loss_clipped in categorical section": "critic_loss_clipped_per_sample" in source,
        "max(loss_unclipped, loss_clipped)": "torch.max(" in source,
        "_project_categorical_distribution call": "_project_categorical_distribution" in source,
        "PPO VF clipping comment": "ppo vf clipping" in source.lower(),
    }

    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        assert passed, f"Failed check: {check_name}"

    print("  ✓ Code structure verification passed!")


def main():
    print("=" * 60)
    print("Categorical VF Clipping Smoke Tests")
    print("=" * 60)

    try:
        test_projection_basic()
        test_projection_identity()
        test_projection_edge_cases()
        test_code_has_vf_clipping()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
