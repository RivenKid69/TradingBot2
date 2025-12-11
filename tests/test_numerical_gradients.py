#!/usr/bin/env python3
"""
Numerical gradient test using finite differences.
Verifies that autograd gradients match numerical gradients.
"""

import pytest
torch = pytest.importorskip("torch")
import sys

from distributional_ppo import DistributionalPPO


def numerical_gradient(func, input_tensor, eps=1e-4):
    """
    Compute numerical gradient using finite differences.

    Args:
        func: Function that takes tensor and returns scalar
        input_tensor: Input tensor (will be detached)
        eps: Finite difference epsilon

    Returns:
        Numerical gradient tensor
    """
    input_flat = input_tensor.detach().clone().view(-1)
    grad_numerical = torch.zeros_like(input_flat)

    for i in range(len(input_flat)):
        # f(x + eps)
        input_plus = input_flat.clone()
        input_plus[i] += eps
        output_plus = func(input_plus.view_as(input_tensor))

        # f(x - eps)
        input_minus = input_flat.clone()
        input_minus[i] -= eps
        output_minus = func(input_minus.view_as(input_tensor))

        # Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
        grad_numerical[i] = (output_plus - output_minus) / (2 * eps)

    return grad_numerical.view_as(input_tensor)


def test_projection_gradient_numerical():
    """
    Test that autograd gradients match numerical gradients.
    This is the GOLD STANDARD test for gradient correctness.
    """
    print("\n" + "="*80)
    print("NUMERICAL GRADIENT TEST")
    print("="*80)
    print("Comparing autograd vs finite differences")

    algo = DistributionalPPO.__new__(DistributionalPPO)

    # Test parameters
    batch_size = 3
    num_atoms = 7
    target_atoms = torch.linspace(-3.0, 3.0, num_atoms)

    # Create input with requires_grad
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs_input = torch.softmax(logits, dim=1)

    # Create shift that triggers same_bounds (this is the critical case!)
    delta = torch.tensor([[0.0], [1.0], [-1.0]])  # Different shifts per batch
    source_atoms = target_atoms.unsqueeze(0) + delta

    print(f"\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num atoms: {num_atoms}")
    print(f"  Deltas: {delta.squeeze().tolist()}")

    # Define loss function
    def loss_func(logits_flat):
        """Loss function for numerical gradient computation"""
        logits_shaped = logits_flat.view(batch_size, num_atoms)
        probs = torch.softmax(logits_shaped, dim=1)

        # Project
        projected = algo._project_categorical_distribution(
            probs=probs,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
        )

        # Simple loss: sum of probabilities at center atom
        center_idx = num_atoms // 2
        loss = projected[:, center_idx].sum()
        return loss

    # ================================================================
    # Compute autograd gradient
    # ================================================================
    print("\n1. Computing autograd gradient...")
    logits_autograd = logits.detach().clone().requires_grad_(True)
    loss_autograd = loss_func(logits_autograd)
    loss_autograd.backward()
    grad_autograd = logits_autograd.grad.clone()

    print(f"   Loss value: {loss_autograd.item():.6f}")
    print(f"   Gradient norm: {grad_autograd.norm().item():.6f}")
    print(f"   Gradient mean: {grad_autograd.mean().item():.6f}")

    # ================================================================
    # Compute numerical gradient
    # ================================================================
    print("\n2. Computing numerical gradient (finite differences)...")
    print("   This may take a moment...")
    grad_numerical = numerical_gradient(loss_func, logits.detach(), eps=1e-4)

    print(f"   Gradient norm: {grad_numerical.norm().item():.6f}")
    print(f"   Gradient mean: {grad_numerical.mean().item():.6f}")

    # ================================================================
    # Compare gradients
    # ================================================================
    print("\n3. Comparing gradients...")

    # Compute differences
    abs_diff = torch.abs(grad_autograd - grad_numerical)
    rel_diff = abs_diff / (torch.abs(grad_numerical) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\n   Max absolute difference: {max_abs_diff:.6e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"   Max relative difference: {max_rel_diff:.6%}")
    print(f"   Mean relative difference: {mean_rel_diff:.6%}")

    # ================================================================
    # Verdict
    # ================================================================
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    # Thresholds
    abs_tol = 1e-3  # Absolute tolerance
    rel_tol = 0.01  # 1% relative tolerance

    abs_pass = max_abs_diff < abs_tol
    rel_pass = max_rel_diff < rel_tol

    if abs_pass and rel_pass:
        print("✅ PASSED: Gradients match within tolerances!")
        print(f"   Absolute difference < {abs_tol:.1e}: ✅")
        print(f"   Relative difference < {rel_tol:.1%}: ✅")
        print("\n🎉 Gradient flow is CORRECT!")
        return True
    else:
        print("❌ FAILED: Gradients do NOT match!")
        if not abs_pass:
            print(f"   Absolute difference >= {abs_tol:.1e}: ❌ ({max_abs_diff:.2e})")
        if not rel_pass:
            print(f"   Relative difference >= {rel_tol:.1%}: ❌ ({max_rel_diff:.1%})")

        print("\n⚠️  Gradient flow may be BROKEN!")
        print("   Possible causes:")
        print("   - Non-differentiable operations")
        print("   - Detached tensors")
        print("   - .item() calls on gradient-carrying values")
        return False


def test_projection_gradient_same_bounds_specific():
    """
    Specific test for same_bounds case (the bug location).
    """
    print("\n" + "="*80)
    print("SAME_BOUNDS SPECIFIC TEST")
    print("="*80)

    algo = DistributionalPPO.__new__(DistributionalPPO)

    # Setup that guarantees same_bounds
    batch_size = 2
    num_atoms = 5
    target_atoms = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # No shift for first batch - will trigger same_bounds for ALL atoms
    # Small shift for second batch - will trigger same_bounds for some atoms
    delta = torch.tensor([[0.0], [0.0]])
    source_atoms = target_atoms.unsqueeze(0) + delta

    print(f"\nSetup: ALL atoms have same_bounds (zero shift)")
    print(f"  This exercises the exact buggy code path!")

    # Create input
    torch.manual_seed(123)
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    # Project
    projected = algo._project_categorical_distribution(
        probs=probs,
        source_atoms=source_atoms,
        target_atoms=target_atoms,
    )

    # Loss: just sum all projected probabilities
    loss = projected.sum()

    # Backward
    loss.backward()

    # Check gradients exist and are non-zero
    assert logits.grad is not None, "Gradient should exist"
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad)), \
        "Gradient should not be all zeros"

    print(f"✅ Gradients exist and are non-zero")
    print(f"   Gradient norm: {logits.grad.norm().item():.6f}")
    print(f"   All batch items have gradients: {(logits.grad.abs().sum(dim=1) > 1e-6).all().item()}")

    return True


def main():
    if not TORCH_AVAILABLE:
        return 0

    print("="*80)
    print("COMPREHENSIVE GRADIENT FLOW VALIDATION")
    print("="*80)

    tests = []

    try:
        result1 = test_projection_gradient_same_bounds_specific()
        tests.append(("Same bounds specific test", result1))
    except Exception as e:
        print(f"\n❌ Same bounds test FAILED with exception:")
        print(f"   {type(e).__name__}: {e}")
        tests.append(("Same bounds specific test", False))

    try:
        result2 = test_projection_gradient_numerical()
        tests.append(("Numerical gradient test", result2))
    except Exception as e:
        print(f"\n❌ Numerical test FAILED with exception:")
        print(f"   {type(e).__name__}: {e}")
        tests.append(("Numerical gradient test", False))

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in tests)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - Gradient flow is CORRECT!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED - Review implementation!")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
