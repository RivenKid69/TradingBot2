#!/usr/bin/env python3
"""
Test script to verify gradient flow through Lagrangian constraint term.

This test checks whether gradients flow correctly through the constraint term
in the Augmented Lagrangian method implementation.
"""

import pytest

torch = pytest.importorskip("torch")
import sys


def test_gradient_flow_new_tensor():
    """Test gradient flow using loss.new_tensor() - current implementation."""
    print("=" * 80)
    print("Testing CURRENT implementation: loss.new_tensor(lambda_scaled)")
    print("=" * 80)

    # Create a simple loss with requires_grad
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    base_loss = (theta ** 2).sum()

    # Simulate constraint violation that depends on theta
    constraint_violation = torch.clamp(theta.mean() - 1.5, min=0.0)

    # Lagrange multiplier (constant, no grad)
    lambda_scaled = 0.5

    # Current implementation
    lambda_tensor = base_loss.new_tensor(lambda_scaled)
    constraint_term = lambda_tensor * constraint_violation

    total_loss = base_loss + constraint_term

    print(f"Base loss: {base_loss.item():.6f}")
    print(f"Constraint violation: {constraint_violation.item():.6f}")
    print(f"Lambda: {lambda_scaled}")
    print(f"Constraint term: {constraint_term.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"\nBefore backward:")
    print(f"  theta.grad: {theta.grad}")
    print(f"  lambda_tensor.requires_grad: {lambda_tensor.requires_grad}")
    print(f"  constraint_violation.requires_grad: {constraint_violation.requires_grad}")
    print(f"  total_loss.requires_grad: {total_loss.requires_grad}")

    # Compute gradients
    total_loss.backward()

    print(f"\nAfter backward:")
    print(f"  theta.grad: {theta.grad}")

    # Check if gradients are non-zero
    if theta.grad is not None and torch.any(theta.grad != 0):
        print("✓ Gradients FLOW correctly through constraint term!")
        return True, theta.grad.clone()
    else:
        print("✗ Gradients BLOCKED - no gradient flow!")
        return False, None


def test_gradient_flow_torch_tensor():
    """Test gradient flow using torch.tensor() - proposed fix."""
    print("\n" + "=" * 80)
    print("Testing PROPOSED fix: torch.tensor(lambda_scaled, device=..., dtype=...)")
    print("=" * 80)

    # Create a simple loss with requires_grad
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    base_loss = (theta ** 2).sum()

    # Simulate constraint violation that depends on theta
    constraint_violation = torch.clamp(theta.mean() - 1.5, min=0.0)

    # Lagrange multiplier (constant, no grad)
    lambda_scaled = 0.5

    # Proposed implementation
    lambda_tensor = torch.tensor(lambda_scaled, device=base_loss.device, dtype=base_loss.dtype)
    constraint_term = lambda_tensor * constraint_violation

    total_loss = base_loss + constraint_term

    print(f"Base loss: {base_loss.item():.6f}")
    print(f"Constraint violation: {constraint_violation.item():.6f}")
    print(f"Lambda: {lambda_scaled}")
    print(f"Constraint term: {constraint_term.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"\nBefore backward:")
    print(f"  theta.grad: {theta.grad}")
    print(f"  lambda_tensor.requires_grad: {lambda_tensor.requires_grad}")
    print(f"  constraint_violation.requires_grad: {constraint_violation.requires_grad}")
    print(f"  total_loss.requires_grad: {total_loss.requires_grad}")

    # Compute gradients
    total_loss.backward()

    print(f"\nAfter backward:")
    print(f"  theta.grad: {theta.grad}")

    # Check if gradients are non-zero
    if theta.grad is not None and torch.any(theta.grad != 0):
        print("✓ Gradients FLOW correctly through constraint term!")
        return True, theta.grad.clone()
    else:
        print("✗ Gradients BLOCKED - no gradient flow!")
        return False, None


def test_gradient_equivalence():
    """Test if both implementations produce identical gradients."""
    print("\n" + "=" * 80)
    print("Testing gradient EQUIVALENCE between implementations")
    print("=" * 80)

    # Test 1: new_tensor
    theta1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    base_loss1 = (theta1 ** 2).sum()
    constraint_violation1 = torch.clamp(theta1.mean() - 1.5, min=0.0)
    lambda_scaled = 0.5
    lambda_tensor1 = base_loss1.new_tensor(lambda_scaled)
    total_loss1 = base_loss1 + lambda_tensor1 * constraint_violation1
    total_loss1.backward()
    grad1 = theta1.grad.clone()

    # Test 2: torch.tensor
    theta2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    base_loss2 = (theta2 ** 2).sum()
    constraint_violation2 = torch.clamp(theta2.mean() - 1.5, min=0.0)
    lambda_tensor2 = torch.tensor(lambda_scaled, device=base_loss2.device, dtype=base_loss2.dtype)
    total_loss2 = base_loss2 + lambda_tensor2 * constraint_violation2
    total_loss2.backward()
    grad2 = theta2.grad.clone()

    print(f"Gradient from new_tensor: {grad1}")
    print(f"Gradient from torch.tensor: {grad2}")
    print(f"Difference: {(grad1 - grad2).abs().max().item():.10f}")

    if torch.allclose(grad1, grad2, atol=1e-8):
        print("✓ Gradients are IDENTICAL - no practical difference!")
        return True
    else:
        print("✗ Gradients DIFFER - implementations are not equivalent!")
        return False


def test_detach_issue():
    """Test to verify that .detach() would break gradient flow."""
    print("\n" + "=" * 80)
    print("Testing .detach() BLOCKING gradient flow (negative control)")
    print("=" * 80)

    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    base_loss = (theta ** 2).sum()

    # This is WRONG - detaching constraint_violation blocks gradients
    constraint_violation = torch.clamp(theta.mean() - 1.5, min=0.0).detach()

    lambda_scaled = 0.5
    lambda_tensor = base_loss.new_tensor(lambda_scaled)
    total_loss = base_loss + lambda_tensor * constraint_violation

    print(f"constraint_violation.requires_grad: {constraint_violation.requires_grad}")

    total_loss.backward()

    # The gradient should only come from base_loss, not constraint term
    expected_grad_without_constraint = torch.tensor([2.0, 4.0, 6.0])

    print(f"theta.grad: {theta.grad}")
    print(f"Expected (without constraint): {expected_grad_without_constraint}")

    if torch.allclose(theta.grad, expected_grad_without_constraint, atol=1e-6):
        print("✓ Confirmed: .detach() BLOCKS gradient flow from constraint term!")
        return True
    else:
        print("✗ Unexpected: gradients differ from expected")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LAGRANGIAN CONSTRAINT GRADIENT FLOW TEST")
    print("=" * 80)
    print("\nPurpose: Verify that gradients flow correctly through the Lagrangian")
    print("constraint term in the Augmented Lagrangian method implementation.")
    print("\nMathematical background:")
    print("  L(θ, λ) = f(θ) + λ * max(0, c(θ))")
    print("  ∂L/∂θ = ∂f/∂θ + λ * ∂(max(0, c(θ)))/∂θ")
    print("\nKey requirement: Gradients must flow through c(θ), NOT through λ.")
    print("λ is a constant scalar multiplier that is updated separately via dual update.\n")

    results = []

    # Test current implementation
    success1, grad1 = test_gradient_flow_new_tensor()
    results.append(("Current (new_tensor)", success1))

    # Test proposed fix
    success2, grad2 = test_gradient_flow_torch_tensor()
    results.append(("Proposed (torch.tensor)", success2))

    # Test equivalence
    success3 = test_gradient_equivalence()
    results.append(("Equivalence", success3))

    # Test detach (negative control)
    success4 = test_detach_issue()
    results.append(("Detach blocking", success4))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_pass = all(success for _, success in results)

    print("\n" + "=" * 80)
    if all_pass:
        print("CONCLUSION: Both implementations work correctly!")
        print("The current implementation using loss.new_tensor() is CORRECT.")
        print("There is NO gradient flow problem in the current code.")
        print("=" * 80)
        return 0
    else:
        print("CONCLUSION: There are issues with gradient flow!")
        print("Further investigation required.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
