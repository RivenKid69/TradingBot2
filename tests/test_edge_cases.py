#!/usr/bin/env python3
"""
Edge case tests for categorical projection.
Tests all corner cases to ensure robustness.
"""

import sys
import pytest

torch = pytest.importorskip("torch")
from conftest import TORCH_AVAILABLE

pytest.importorskip("gymnasium")

from distributional_ppo import DistributionalPPO


def test_edge_case_single_atom():
    """Test with single atom (should be identity)"""
    print("\n" + "=" * 60)
    print("TEST: Single Atom")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    probs = torch.tensor([[1.0]], requires_grad=True)
    atoms = torch.tensor([0.0])

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=atoms, target_atoms=atoms
    )

    assert torch.allclose(projected, probs), "Single atom should be identity"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert probs.grad is not None, "Gradient should exist"

    print("✅ Single atom test passed")
    return True


def test_edge_case_all_same_bounds():
    """Test when ALL atoms have same_bounds (no shift)"""
    print("\n" + "=" * 60)
    print("TEST: All Same Bounds (Identity Projection)")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 4
    num_atoms = 10
    atoms = torch.linspace(-5.0, 5.0, num_atoms)

    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    # No shift - all atoms match exactly
    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=atoms, target_atoms=atoms
    )

    # Should be approximately identity
    assert torch.allclose(projected, probs, atol=1e-4), "Identity projection should preserve probs"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist"
    assert not torch.allclose(
        logits.grad, torch.zeros_like(logits.grad)
    ), "Gradient should be non-zero"

    print(f"✅ All same bounds test passed")
    print(f"   Max difference from identity: {(projected - probs).abs().max().item():.2e}")
    return True


def test_edge_case_no_same_bounds():
    """Test when NO atoms have same_bounds (large shift)"""
    print("\n" + "=" * 60)
    print("TEST: No Same Bounds (Large Shift)")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 4
    num_atoms = 10
    target_atoms = torch.linspace(-5.0, 5.0, num_atoms)

    # Large shift that doesn't land exactly on any atom
    delta = 0.37  # Chosen to not align with grid
    source_atoms = target_atoms + delta

    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    # Check validity
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(batch_size), atol=1e-5
    ), "Projected probs should sum to 1"
    assert torch.all(projected >= 0), "Projected probs should be non-negative"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist"
    assert not torch.allclose(
        logits.grad, torch.zeros_like(logits.grad)
    ), "Gradient should be non-zero"

    print(f"✅ No same bounds test passed")
    return True


def test_edge_case_mixed_same_bounds():
    """Test when some batch items have same_bounds, others don't"""
    print("\n" + "=" * 60)
    print("TEST: Mixed Same Bounds")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 4
    num_atoms = 7
    target_atoms = torch.linspace(-3.0, 3.0, num_atoms)

    # Different shifts: some zero (same_bounds), some not
    delta = torch.tensor([[0.0], [0.5], [0.0], [1.0]])
    source_atoms = target_atoms.unsqueeze(0) + delta

    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    # Check validity for all batch items
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(batch_size), atol=1e-5
    ), "All projected probs should sum to 1"
    assert torch.all(projected >= 0), "All projected probs should be non-negative"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist"

    # Check that ALL batch items have gradients
    grad_norms_per_batch = logits.grad.abs().sum(dim=1)
    assert torch.all(
        grad_norms_per_batch > 1e-6
    ), "All batch items should have non-negligible gradients"

    print(f"✅ Mixed same bounds test passed")
    print(f"   Gradient norms per batch: {grad_norms_per_batch.tolist()}")
    return True


def test_edge_case_extreme_shift():
    """Test with extreme shift outside target range"""
    print("\n" + "=" * 60)
    print("TEST: Extreme Shift (Clipping)")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 2
    num_atoms = 10
    v_min, v_max = -5.0, 5.0
    target_atoms = torch.linspace(v_min, v_max, num_atoms)

    # Extreme shift that pushes atoms outside target range
    delta = 100.0  # Way outside [-5, 5]
    source_atoms = target_atoms + delta

    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    # Should clamp and project all mass to v_max
    # (all atoms above v_max)
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(batch_size), atol=1e-5
    ), "Projected probs should sum to 1"

    # Most mass should be at the upper end
    upper_half_mass = projected[:, num_atoms // 2 :].sum(dim=1)
    assert torch.all(
        upper_half_mass > 0.9
    ), "Most mass should be in upper half for large positive shift"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist even with extreme shift"

    print(f"✅ Extreme shift test passed")
    print(f"   Upper half mass: {upper_half_mass.tolist()}")
    return True


def test_edge_case_batch_size_one():
    """Test with batch size = 1"""
    print("\n" + "=" * 60)
    print("TEST: Batch Size = 1")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 1
    num_atoms = 7
    target_atoms = torch.linspace(-3.0, 3.0, num_atoms)
    source_atoms = target_atoms + 0.5

    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    assert projected.shape == (batch_size, num_atoms), "Shape should be correct"
    assert torch.allclose(projected.sum(), torch.tensor(1.0), atol=1e-5), "Probs should sum to 1"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist"

    print(f"✅ Batch size = 1 test passed")
    return True


def test_edge_case_large_batch():
    """Test with large batch size"""
    print("\n" + "=" * 60)
    print("TEST: Large Batch Size")
    print("=" * 60)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 128  # Typical training batch size
    num_atoms = 51  # C51 standard
    v_min, v_max = -10.0, 10.0
    target_atoms = torch.linspace(v_min, v_max, num_atoms)

    # Random shifts
    torch.manual_seed(42)
    delta = torch.randn(batch_size, 1) * 2.0  # Random shifts
    source_atoms = target_atoms.unsqueeze(0) + delta

    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    projected = algo._project_categorical_distribution(
        probs=probs, source_atoms=source_atoms, target_atoms=target_atoms
    )

    # Check all batch items are valid
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(batch_size), atol=1e-5
    ), "All batch items should sum to 1"
    assert torch.all(projected >= 0), "All probabilities should be non-negative"

    # Test gradients
    loss = projected.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient should exist"

    # All batch items should have gradients
    grad_norms = logits.grad.abs().sum(dim=1)
    assert torch.all(grad_norms > 1e-8), "All batch items should have gradients"

    print(f"✅ Large batch test passed (batch_size={batch_size})")
    print(f"   Min gradient norm: {grad_norms.min().item():.2e}")
    print(f"   Max gradient norm: {grad_norms.max().item():.2e}")
    return True


def main():
    if not TORCH_AVAILABLE:
        return 0

    print("=" * 80)
    print("EDGE CASE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Single atom", test_edge_case_single_atom),
        ("All same bounds", test_edge_case_all_same_bounds),
        ("No same bounds", test_edge_case_no_same_bounds),
        ("Mixed same bounds", test_edge_case_mixed_same_bounds),
        ("Extreme shift", test_edge_case_extreme_shift),
        ("Batch size = 1", test_edge_case_batch_size_one),
        ("Large batch", test_edge_case_large_batch),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("EDGE CASE TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL EDGE CASES PASSED!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠️  SOME EDGE CASES FAILED!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
