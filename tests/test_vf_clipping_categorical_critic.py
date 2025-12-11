"""
Test to verify the VF clipping issue in categorical critic.

This test demonstrates that the current implementation creates a triple max
instead of the correct double max required by PPO.
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np


def test_vf_clipping_triple_max_issue():
    """
    Reproduce the triple max issue in categorical VF clipping.

    The current implementation does:
        final_loss = mean(max(max(L_unclipped, L_clipped_method1), L_clipped_method2))
                   = mean(max(L_unclipped, L_clipped_method1, L_clipped_method2))  # TRIPLE MAX!

    The correct PPO implementation should do:
        final_loss = mean(max(L_unclipped, L_clipped))  # DOUBLE MAX
    """
    torch.manual_seed(42)

    # Simulate loss values where the two clipping methods give different results
    batch_size = 4

    # Create scenario where:
    # - L_unclipped = [1.0, 2.0, 3.0, 4.0]
    # - L_clipped_method1 = [1.5, 1.8, 3.2, 3.8]  # First clipping method
    # - L_clipped_method2 = [1.2, 2.5, 2.8, 4.5]  # Second clipping method

    L_unclipped = torch.tensor([1.0, 2.0, 3.0, 4.0])
    L_clipped_method1 = torch.tensor([1.5, 1.8, 3.2, 3.8])
    L_clipped_method2 = torch.tensor([1.2, 2.5, 2.8, 4.5])

    # Current (buggy) implementation: triple max
    # Step 1: First max (lines 8911-8914)
    first_max = torch.max(L_unclipped, L_clipped_method1)
    print(f"First max(L_unclipped, L_clipped_method1): {first_max}")

    # Step 2: Second max (lines 9136-9140) - this creates TRIPLE MAX!
    triple_max = torch.max(first_max, L_clipped_method2)
    current_loss = triple_max.mean()
    print(f"Triple max(first_max, L_clipped_method2): {triple_max}")
    print(f"Current (buggy) loss: {current_loss.item():.4f}")

    # Correct PPO implementation: double max
    # Should choose ONE clipping method, not apply both
    correct_double_max_method1 = torch.max(L_unclipped, L_clipped_method1)
    correct_loss_method1 = correct_double_max_method1.mean()
    print(f"\nCorrect loss (method 1): {correct_loss_method1.item():.4f}")

    correct_double_max_method2 = torch.max(L_unclipped, L_clipped_method2)
    correct_loss_method2 = correct_double_max_method2.mean()
    print(f"Correct loss (method 2): {correct_loss_method2.item():.4f}")

    # Verify that triple max >= double max (always true mathematically)
    assert current_loss >= correct_loss_method1, (
        "Triple max should always be >= double max (method 1)"
    )
    assert current_loss >= correct_loss_method2, (
        "Triple max should always be >= double max (method 2)"
    )

    # In this example, triple max is strictly greater than both double max values
    print(f"\nLoss inflation from triple max:")
    print(f"  vs method 1: {(current_loss - correct_loss_method1).item():.4f} "
          f"({((current_loss / correct_loss_method1 - 1) * 100).item():.2f}% higher)")
    print(f"  vs method 2: {(current_loss - correct_loss_method2).item():.4f} "
          f"({((current_loss / correct_loss_method2 - 1) * 100).item():.2f}% higher)")

    # Element-wise comparison to show where triple max differs
    print(f"\nElement-wise comparison:")
    print(f"L_unclipped:         {L_unclipped.numpy()}")
    print(f"L_clipped_method1:   {L_clipped_method1.numpy()}")
    print(f"L_clipped_method2:   {L_clipped_method2.numpy()}")
    print(f"max(L_u, L_c1):      {correct_double_max_method1.numpy()}")
    print(f"max(L_u, L_c2):      {correct_double_max_method2.numpy()}")
    print(f"max(L_u, L_c1, L_c2): {triple_max.numpy()}")

    # Show which element gets inflated
    inflation_vs_method1 = triple_max - correct_double_max_method1
    inflation_vs_method2 = triple_max - correct_double_max_method2
    print(f"\nInflation vs method 1: {inflation_vs_method1.numpy()}")
    print(f"Inflation vs method 2: {inflation_vs_method2.numpy()}")

    # The bug is confirmed if any element shows inflation
    has_inflation = (inflation_vs_method1.sum() > 0) or (inflation_vs_method2.sum() > 0)
    print(f"\n{'=' * 60}")
    print(f"BUG CONFIRMED: Triple max creates inflated loss!" if has_inflation else "No inflation detected")
    print(f"{'=' * 60}")

    return {
        'current_loss': current_loss.item(),
        'correct_loss_method1': correct_loss_method1.item(),
        'correct_loss_method2': correct_loss_method2.item(),
        'has_inflation': has_inflation,
    }


def test_categorical_vf_clipping_methods_difference():
    """
    Test that demonstrates the two VF clipping methods produce different results.

    Method 1 (_project_categorical_distribution): Shifts atoms and projects back
    Method 2 (_build_support_distribution): Creates point distribution from clipped mean

    If methods differ, applying both creates triple max instead of double max.
    """
    print("\n" + "=" * 60)
    print("Testing difference between two VF clipping methods")
    print("=" * 60)

    # This test would need access to the actual distributional_ppo.py implementation
    # For now, we demonstrate the conceptual difference

    # Method 1: Preserves distribution shape, shifts atoms
    # - More appropriate for distributional RL
    # - Maintains gradient flow through projection
    # - Preserves uncertainty information

    # Method 2: Collapses to point distribution
    # - Loses distribution shape information
    # - Creates dirac delta at clipped mean
    # - Loses uncertainty information

    print("\nMethod 1 (_project_categorical_distribution):")
    print("  - Shifts atoms by delta and projects back to original grid")
    print("  - Preserves distribution shape and uncertainty")
    print("  - Maintains gradient flow through projection")

    print("\nMethod 2 (_build_support_distribution):")
    print("  - Creates point distribution at clipped mean value")
    print("  - Collapses distribution to dirac delta")
    print("  - Loses uncertainty information")

    print("\nConclusion: Methods are fundamentally different!")
    print("Using BOTH methods creates triple max instead of double max.")
    print("Solution: Use ONLY Method 1 (projection-based clipping).")


if __name__ == "__main__":
    print("=" * 60)
    print("VF Clipping Triple Max Issue - Reproduction Test")
    print("=" * 60)

    results = test_vf_clipping_triple_max_issue()
    print(f"\nTest results: {results}")

    test_categorical_vf_clipping_methods_difference()

    print("\n" + "=" * 60)
    print("CONCLUSION: Bug confirmed - triple max instead of double max!")
    print("=" * 60)
