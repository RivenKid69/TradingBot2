"""
Standalone test for gradient flow through categorical distribution projection.
This doesn't require pytest.
"""

import pytest

torch = pytest.importorskip("torch")
import sys

from distributional_ppo import DistributionalPPO


def test_gradient_flow_simple():
    """Test basic gradient flow through projection."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Gradient Flow")
    print("=" * 80)

    algo = DistributionalPPO.__new__(DistributionalPPO)

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

    # Compute a simple loss
    target_probs = torch.ones(batch_size, num_atoms) / num_atoms
    log_probs = torch.log(projected_probs.clamp(min=1e-8))
    loss = -(log_probs * target_probs).sum()

    # Backpropagate
    loss.backward()

    # Check gradients
    if logits.grad is None:
        print("❌ FAILED: No gradient exists on input logits")
        return False

    if torch.allclose(logits.grad, torch.zeros_like(logits.grad)):
        print("❌ FAILED: Gradient is all zeros (gradient flow is BROKEN)")
        print(f"   Gradient: {logits.grad}")
        return False

    grad_norm = logits.grad.norm()
    print(f"✓ PASSED: Gradient exists and is non-zero")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Gradient mean: {logits.grad.mean():.6f}")
    print(f"  Gradient std: {logits.grad.std():.6f}")
    return True


def test_gradient_flow_with_same_bounds():
    """Test gradient flow when same_bounds correction is triggered."""
    print("\n" + "=" * 80)
    print("TEST 2: Gradient Flow with Same Bounds Correction")
    print("=" * 80)
    print("This test specifically checks the suspected bug location!")

    algo = DistributionalPPO.__new__(DistributionalPPO)

    batch_size = 2
    num_atoms = 7
    target_atoms = torch.linspace(-3.0, 3.0, num_atoms)

    # Create input probabilities that require gradients
    logits = torch.randn(batch_size, num_atoms, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    # Create delta that will cause some atoms to land exactly on target atoms
    # This triggers the same_bounds correction path (lines 2704-2757)
    delta = torch.tensor([[0.0], [1.0]])  # First batch: no shift, second: shift
    source_atoms = target_atoms.unsqueeze(0) + delta

    print(f"\nInput setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Delta values: {delta.squeeze().tolist()}")

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

    # Check gradients
    if logits.grad is None:
        print("❌ FAILED: No gradient exists")
        return False

    # Check each batch item separately
    all_passed = True
    for i in range(batch_size):
        row_grad = logits.grad[i]
        if torch.allclose(row_grad, torch.zeros_like(row_grad)):
            print(f"❌ FAILED: Batch item {i} has zero gradient (BROKEN)")
            print(f"   This is the same_bounds correction bug!")
            all_passed = False
        else:
            grad_norm = row_grad.norm()
            print(f"✓ Batch item {i}: gradient norm = {grad_norm:.6f}")

    if all_passed:
        print(f"\n✓ PASSED: All batch items have gradients")
        return True
    else:
        print(f"\n❌ FAILED: Gradient flow is broken for same_bounds case")
        return False


def test_gradient_flow_end_to_end():
    """Test gradient flow in VF clipping scenario."""
    print("\n" + "=" * 80)
    print("TEST 3: End-to-End VF Clipping Scenario")
    print("=" * 80)
    print("This mimics the exact usage in distributional_ppo.py:8788-8819")

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

    mean_clipped = torch.clamp(mean_pred, min=mean_old - clip_range, max=mean_old + clip_range)

    print(f"\nPredicted means: {mean_pred.squeeze().tolist()}")
    print(f"Clipped means: {mean_clipped.squeeze().tolist()}")

    # Compute delta and shift atoms
    delta_norm = mean_clipped - mean_pred
    atoms_shifted = atoms_original.unsqueeze(0) + delta_norm

    # Project
    pred_probs_clipped = algo._project_categorical_distribution(
        probs=pred_probs,
        source_atoms=atoms_shifted,
        target_atoms=atoms_original,
    )

    # Compute loss
    target_probs = torch.ones(batch_size, num_atoms) / num_atoms
    log_predictions_clipped = torch.log(pred_probs_clipped.clamp(min=1e-8))
    critic_loss_clipped = -(log_predictions_clipped * target_probs).sum(dim=1)
    loss = critic_loss_clipped.mean()

    # Backpropagate
    loss.backward()

    # Check gradients
    if logits.grad is None:
        print("❌ FAILED: No gradient exists on network output")
        print("   This means VF clipping cannot train the network!")
        return False

    if torch.allclose(logits.grad, torch.zeros_like(logits.grad)):
        print("❌ FAILED: Gradient is all zeros")
        print("   This is the CRITICAL BUG - VF clipping breaks training!")
        return False

    # Check all batch items
    all_passed = True
    for i in range(batch_size):
        row_grad = logits.grad[i]
        if not torch.any(torch.abs(row_grad) > 1e-6):
            print(f"❌ Batch item {i} has negligible gradients")
            all_passed = False
        else:
            grad_norm = row_grad.norm()
            print(f"✓ Batch item {i}: gradient norm = {grad_norm:.6f}")

    if all_passed:
        grad_norm = logits.grad.norm()
        print(f"\n✓ PASSED: Gradients flow correctly")
        print(f"  Total gradient norm: {grad_norm:.6f}")
        print(f"  Gradient mean: {logits.grad.mean():.6f}")
        print(f"  Gradient std: {logits.grad.std():.6f}")
        return True
    else:
        print(f"\n❌ FAILED: Some gradients are broken")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRADIENT FLOW TEST SUITE")
    print("Testing categorical distribution projection")
    print("=" * 80)

    results = []
    results.append(("Basic gradient flow", test_gradient_flow_simple()))
    results.append(("Same bounds correction", test_gradient_flow_with_same_bounds()))
    results.append(("End-to-end VF clipping", test_gradient_flow_end_to_end()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Gradient flow is working correctly!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED - Gradient flow bug confirmed!")
        print("    The categorical projection breaks gradient flow.")
        print("    This prevents proper training with VF clipping enabled.")
        sys.exit(1)
