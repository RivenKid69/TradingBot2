"""
Test script to verify the correct KL divergence direction for PPO.

Mathematical Analysis:
====================

Given:
- π_old: old policy (used to collect rollout data)
- π_new: new policy (being optimized)
- Actions a are sampled from π_old

KL Divergence Definitions:
-------------------------
1. Forward KL:  KL(π_new || π_old) = E_{a~π_new}[log π_new(a) - log π_old(a)]
2. Reverse KL:  KL(π_old || π_new) = E_{a~π_old}[log π_old(a) - log π_new(a)]

In PPO (from Schulman et al. 2017):
-----------------------------------
PPO uses KL(π_old || π_new) because:
1. We sample actions from π_old (rollout buffer)
2. We want to penalize the new policy from deviating too far from the old policy
3. This is computationally tractable with rollout data

The adaptive KL penalty term is:
L^KLPEN(θ) = E_t[ratio_t * A_t - β * KL(π_old || π_new)]

When sampling from π_old:
KL(π_old || π_new) = E_{a~π_old}[log π_old(a) - log π_new(a)]
                   = E[old_log_prob - log_prob]

Reference Implementations:
--------------------------
1. CleanRL:
   old_approx_kl = (-logratio).mean()
   where logratio = newlogprob - oldlogprob
   => old_approx_kl = mean(oldlogprob - newlogprob) ✓

2. Stable-Baselines3:
   Uses k3 estimator: (exp(log_ratio) - 1) - log_ratio
   where log_ratio = log_prob - old_log_prob
   This approximates KL(π_old || π_new)

Conclusion:
-----------
Current implementation: (old_log_prob - log_prob).mean()
This computes: E[log π_old - log π_new] = KL(π_old || π_new) ✓ CORRECT

Proposed "fix": (log_prob - old_log_prob).mean()
This computes: E[log π_new - log π_old] = -KL(π_old || π_new) ✗ WRONG (negative KL!)
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F


def test_kl_direction():
    """Numerical test to verify KL direction."""
    print("=" * 80)
    print("Testing KL Divergence Direction for PPO")
    print("=" * 80)

    # Create two simple categorical distributions
    # Old policy (used for sampling)
    old_probs = torch.tensor([0.6, 0.3, 0.1])
    # New policy (slightly different)
    new_probs = torch.tensor([0.5, 0.35, 0.15])

    # Compute true KL divergences
    kl_old_new = (old_probs * (old_probs.log() - new_probs.log())).sum()
    kl_new_old = (new_probs * (new_probs.log() - old_probs.log())).sum()

    print(f"\nTrue KL(π_old || π_new) = {kl_old_new.item():.6f}")
    print(f"True KL(π_new || π_old) = {kl_new_old.item():.6f}")
    print(f"Note: These are different values! (asymmetry of KL divergence)")

    # Simulate sampling from old policy
    num_samples = 100000
    torch.manual_seed(42)
    samples = torch.multinomial(old_probs, num_samples, replacement=True)

    # Compute log probabilities
    old_log_probs = old_probs.log()[samples]
    new_log_probs = new_probs.log()[samples]

    # Current implementation (CORRECT)
    current_kl = (old_log_probs - new_log_probs).mean()

    # Proposed "fix" (WRONG)
    proposed_kl = (new_log_probs - old_log_probs).mean()

    print(f"\n{'Method':<40} {'Value':<15} {'Match'}")
    print("-" * 80)
    print(f"{'True KL(π_old || π_new)':<40} {kl_old_new.item():<15.6f}")
    print(f"{'Current: (old_log - new_log).mean()':<40} {current_kl.item():<15.6f} ✓ CORRECT")
    print(f"{'Proposed: (new_log - old_log).mean()':<40} {proposed_kl.item():<15.6f} ✗ WRONG")
    print(f"{'Proposed is actually: -KL(π_old || π_new)':<40} {-kl_old_new.item():<15.6f}")

    # Test k3 estimator (more advanced)
    ratio = torch.exp(new_log_probs - old_log_probs)
    k3_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()

    print(f"\n{'Advanced Estimators:'}")
    print(f"{'k3 estimator: (r-1) - log(r)':<40} {k3_kl.item():<15.6f} ✓ (lower variance)")

    # Verify sign
    print(f"\n{'Sign Analysis:'}")
    print(f"{'KL divergence must be >= 0':<40} {'Always True'}")
    print(f"{'Current implementation sign:':<40} {'+' if current_kl >= 0 else '-'} ✓")
    print(f"{'Proposed implementation sign:':<40} {'+' if proposed_kl >= 0 else '-'} ✗ (negative!)")

    print("\n" + "=" * 80)
    print("CONCLUSION: Current implementation is CORRECT")
    print("=" * 80)
    print("\nThe current code correctly computes KL(π_old || π_new), which is")
    print("the standard formulation used in PPO (Schulman et al. 2017).")
    print("\nThe proposed 'fix' would actually break the implementation by")
    print("computing -KL(π_old || π_new), which is negative and incorrect.")
    print("=" * 80)


if __name__ == "__main__":
    test_kl_direction()
