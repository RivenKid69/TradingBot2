"""
Verification script for Twin Critics loss aggregation issue.

Problem: Current implementation averages losses BEFORE applying max(L_uc, L_c),
but mathematically correct approach should apply max to EACH critic independently.

Current: max((L_uc1 + L_uc2)/2, (L_c1 + L_c2)/2)
Correct: (max(L_uc1, L_c1) + max(L_uc2, L_c2))/2

This script tests whether these formulas are equivalent.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


def current_implementation(loss_c1_unclipped, loss_c2_unclipped, loss_c1_clipped, loss_c2_clipped):
    """
    Current implementation in distributional_ppo.py

    Lines 10673-10688:
        clipped_loss_avg = (loss_c1_clipped + loss_c2_clipped) / 2.0
        loss_unclipped_avg = (loss_c1_unclipped + loss_c2_unclipped) / 2.0
        critic_loss = torch.mean(torch.max(loss_unclipped_avg, clipped_loss_avg))
    """
    clipped_loss_avg = (loss_c1_clipped + loss_c2_clipped) / 2.0
    loss_unclipped_avg = (loss_c1_unclipped + loss_c2_unclipped) / 2.0

    # Element-wise max, then mean
    critic_loss = torch.mean(torch.max(loss_unclipped_avg, clipped_loss_avg))
    return critic_loss


def mathematically_correct_implementation(
    loss_c1_unclipped, loss_c2_unclipped, loss_c1_clipped, loss_c2_clipped
):
    """
    Mathematically correct implementation (claimed by user).

    Apply max to EACH critic independently, then average.
    """
    loss_c1_final = torch.max(loss_c1_unclipped, loss_c1_clipped)
    loss_c2_final = torch.max(loss_c2_unclipped, loss_c2_clipped)
    critic_loss = torch.mean((loss_c1_final + loss_c2_final) / 2.0)
    return critic_loss


def test_case_1_both_clipping():
    """
    Case 1: Both critics require clipping (L_uc > L_c for both)
    Expected: Both implementations should be equal
    """
    print("\n=== Test Case 1: Both critics require clipping ===")

    # Both critics have larger unclipped loss
    loss_c1_uc = torch.tensor([10.0, 15.0, 20.0])
    loss_c1_c = torch.tensor([5.0, 8.0, 12.0])
    loss_c2_uc = torch.tensor([12.0, 18.0, 22.0])
    loss_c2_c = torch.tensor([6.0, 9.0, 13.0])

    current = current_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)
    correct = mathematically_correct_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)

    print(f"Current implementation:  {current:.4f}")
    print(f"Correct implementation:  {correct:.4f}")
    print(f"Difference:              {abs(current - correct):.6f}")
    print(f"Are they equal?          {torch.allclose(current, correct)}")

    return torch.allclose(current, correct)


def test_case_2_no_clipping():
    """
    Case 2: Neither critic requires clipping (L_uc < L_c for both)
    Expected: Both implementations should be equal
    """
    print("\n=== Test Case 2: No clipping needed ===")

    # Both critics have smaller unclipped loss
    loss_c1_uc = torch.tensor([5.0, 8.0, 12.0])
    loss_c1_c = torch.tensor([10.0, 15.0, 20.0])
    loss_c2_uc = torch.tensor([6.0, 9.0, 13.0])
    loss_c2_c = torch.tensor([12.0, 18.0, 22.0])

    current = current_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)
    correct = mathematically_correct_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)

    print(f"Current implementation:  {current:.4f}")
    print(f"Correct implementation:  {correct:.4f}")
    print(f"Difference:              {abs(current - correct):.6f}")
    print(f"Are they equal?          {torch.allclose(current, correct)}")

    return torch.allclose(current, correct)


def test_case_3_mixed():
    """
    Case 3: One critic requires clipping, other doesn't (CRITICAL TEST)
    Expected: Implementations should DIFFER

    This is the case where the bug manifests.
    """
    print("\n=== Test Case 3: Mixed (ONE CRITIC CLIPS, OTHER DOESN'T) ===")

    # Critic 1: Unclipped > Clipped (needs clipping)
    # Critic 2: Unclipped < Clipped (doesn't need clipping)
    loss_c1_uc = torch.tensor([10.0])
    loss_c1_c = torch.tensor([5.0])
    loss_c2_uc = torch.tensor([5.0])
    loss_c2_c = torch.tensor([10.0])

    current = current_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)
    correct = mathematically_correct_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)

    print(
        f"Critic 1: Unclipped={loss_c1_uc.item():.1f}, Clipped={loss_c1_c.item():.1f} -> max={max(loss_c1_uc.item(), loss_c1_c.item()):.1f}"
    )
    print(
        f"Critic 2: Unclipped={loss_c2_uc.item():.1f}, Clipped={loss_c2_c.item():.1f} -> max={max(loss_c2_uc.item(), loss_c2_c.item()):.1f}"
    )
    print()
    print(f"Current implementation:  {current:.4f}")
    print(f"  -> max(avg_uc, avg_c) = max((10+5)/2, (5+10)/2) = max(7.5, 7.5) = 7.5")
    print()
    print(f"Correct implementation:  {correct:.4f}")
    print(f"  -> avg(max_c1, max_c2) = ((max(10,5) + max(5,10))/2 = (10 + 10)/2 = 10.0")
    print()
    print(f"Difference:              {abs(current - correct):.6f}")
    print(f"Relative error:          {(abs(current - correct) / correct * 100):.2f}%")
    print(f"Are they equal?          {torch.allclose(current, correct)}")

    return torch.allclose(current, correct)


def test_case_4_batch_mixed():
    """
    Case 4: Batch with mixed clipping requirements
    Expected: Implementations should DIFFER
    """
    print("\n=== Test Case 4: Batch with mixed clipping ===")

    # Batch of 3 samples with different clipping patterns
    # Sample 0: Both clip
    # Sample 1: Neither clips
    # Sample 2: Mixed (C1 clips, C2 doesn't)
    loss_c1_uc = torch.tensor([10.0, 5.0, 10.0])
    loss_c1_c = torch.tensor([5.0, 10.0, 5.0])
    loss_c2_uc = torch.tensor([12.0, 6.0, 5.0])
    loss_c2_c = torch.tensor([6.0, 12.0, 10.0])

    current = current_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)
    correct = mathematically_correct_implementation(loss_c1_uc, loss_c2_uc, loss_c1_c, loss_c2_c)

    print(f"Current implementation:  {current:.4f}")
    print(f"Correct implementation:  {correct:.4f}")
    print(f"Difference:              {abs(current - correct):.6f}")
    print(f"Relative error:          {(abs(current - correct) / correct * 100):.2f}%")
    print(f"Are they equal?          {torch.allclose(current, correct)}")

    return torch.allclose(current, correct)


def analyze_ppo_semantics():
    """
    Analyze what PPO VF clipping is supposed to do.

    PPO VF clipping prevents VALUE FUNCTION from making too large updates.
    The standard formula is:
        L_VF = max((V - V_target)^2, (V_clip - V_target)^2)

    This takes the MAXIMUM of unclipped and clipped loss to be pessimistic
    about value function updates.

    For Twin Critics, each critic should be clipped INDEPENDENTLY:
        L_C1 = max((V1 - target)^2, (V1_clip - target)^2)
        L_C2 = max((V2 - target)^2, (V2_clip - target)^2)
        L_total = (L_C1 + L_C2) / 2

    NOT:
        L_total = max(avg(V1, V2) - target)^2, avg(V1_clip, V2_clip) - target)^2)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: PPO VF Clipping Semantics")
    print("=" * 80)
    print()
    print("PPO VF clipping prevents value function from making too large updates.")
    print("Standard formula: L_VF = max((V - V_target)^2, (V_clip - V_target)^2)")
    print()
    print("For Twin Critics, the correct approach is:")
    print("  1. Apply VF clipping to EACH critic independently")
    print("  2. Compute loss for each critic: L_C1 = max(...), L_C2 = max(...)")
    print("  3. Average the losses: L_total = (L_C1 + L_C2) / 2")
    print()
    print("Why is this important?")
    print("  - Independence: Twin Critics are independent networks")
    print("  - Different updates: C1 may need clipping while C2 doesn't (or vice versa)")
    print("  - Averaging before max() loses this independence")
    print()
    print("Research support:")
    print("  - TD3 (Fujimoto et al. 2018): Independent critics with separate updates")
    print("  - SAC (Haarnoja et al. 2018): Twin Q-functions with independent optimization")
    print("  - PPO (Schulman et al. 2017): VF clipping via element-wise max")
    print()


def main():
    print("=" * 80)
    print("Twin Critics Loss Aggregation Verification")
    print("=" * 80)

    analyze_ppo_semantics()

    # Run test cases
    results = []
    results.append(("Both clipping", test_case_1_both_clipping()))
    results.append(("No clipping", test_case_2_no_clipping()))
    results.append(("Mixed (CRITICAL)", test_case_3_mixed()))
    results.append(("Batch mixed", test_case_4_batch_mixed()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for name, passed in results:
        status = "[PASS] Equal" if passed else "[FAIL] Different"
        print(f"{name:25s}: {status}")

    print()

    # Conclusion
    if results[2][1]:  # Mixed case (critical test)
        print("[OK] CONCLUSION: Implementations are EQUIVALENT (no bug)")
        print("     Current implementation is mathematically correct.")
    else:
        print("[BUG] CONCLUSION: BUG CONFIRMED!")
        print("      Current implementation DIFFERS from correct implementation.")
        print("      Mixed cases (one critic clips, other doesn't) produce WRONG results.")
        print()
        print("IMPACT:")
        print("  - Underestimates loss when critics have mixed clipping requirements")
        print("  - Can lead to insufficient value function updates")
        print("  - Reduces effectiveness of Twin Critics (defeats the purpose)")
        print()
        print("RECOMMENDATION:")
        print("  - Fix: Apply max() to each critic independently, then average")
        print("  - Expected improvement: Better value estimates, more stable training")


if __name__ == "__main__":
    main()
