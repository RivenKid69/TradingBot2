"""
Tests to verify the UPGD utility normalization bias-correction fix.
Ensures that at step 1 (where bias correction has the strongest effect),
high-utility and low-utility parameters receive different scaling factors
instead of all clamping to the same value.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from optimizers import UPGD, AdaptiveUPGD, UPGDW


def test_upgd_bias_correction_first_step():
    """Verify that UPGD correctly differentiates updates based on utility on step 1."""
    # Create two parameter tensors
    p1 = nn.Parameter(torch.tensor([1.0]))
    p2 = nn.Parameter(torch.tensor([2.0]))

    # Initialize UPGD
    # lr=1.0, beta_utility=0.999, sigma=0.0, weight_decay=0.0
    optimizer = UPGD([p1, p2], lr=1.0, beta_utility=0.999, sigma=0.0, weight_decay=0.0)

    # Set gradients such that p1 has higher utility than p2
    # utility = -grad * param
    # p1: grad = -1.0, param = 1.0 -> utility = 1.0 (high)
    # p2: grad = 1.0, param = 2.0 -> utility = -2.0 (low)
    p1.grad = torch.tensor([-1.0])
    p2.grad = torch.tensor([1.0])

    # Step 1
    optimizer.step()

    # After step:
    # Update for p1: p1_new = p1_old - lr * grad1 * (1 - scaled_utility1)
    #               = 1.0 - 1.0 * (-1.0) * (1 - scaled_utility1)
    #               = 1.0 + (1 - scaled_utility1)
    # Effective scaling factor for p1 = (p1_new - p1_old) / (-grad1) = p1_new - 1.0
    #
    # Update for p2: p2_new = p2_old - lr * grad2 * (1 - scaled_utility2)
    #               = 2.0 - 1.0 * 1.0 * (1 - scaled_utility2)
    #               = 2.0 - (1 - scaled_utility2)
    # Effective scaling factor for p2 = (p2_old - p2_new) / grad2 = 2.0 - p2_new

    scale_p1 = p1.item() - 1.0
    scale_p2 = 2.0 - p2.item()

    print(f"\nUPGD Step 1 Verification:")
    print(f"  High Utility Param (p1) Effective Scale: {scale_p1:.6f} (Expected ~0.27)")
    print(f"  Low Utility Param (p2) Effective Scale: {scale_p2:.6f} (Expected ~0.73)")
    print(f"  Ratio (Low / High): {scale_p2 / scale_p1:.2f}x (Expected ~2.7x)")

    # Assert that p2 (low utility) updated significantly more than p1 (high utility)
    # Before the fix, both got normalized to 0.0 (clamped) and updated by the same factor (0.73).
    # Thus, the ratio was exactly 1.0. With the fix, the ratio is ~2.7.
    assert (
        scale_p2 > scale_p1
    ), "Low utility parameter should receive larger updates than high utility parameter"
    assert scale_p2 / scale_p1 > 2.0, f"Expected update ratio > 2.0, got {scale_p2 / scale_p1:.2f}x"


def test_adaptive_upgd_bias_correction_first_step():
    """Verify that AdaptiveUPGD correctly differentiates updates based on utility on step 1."""
    # Create two parameter tensors
    p1 = nn.Parameter(torch.tensor([1.0]))
    p2 = nn.Parameter(torch.tensor([2.0]))

    # Initialize AdaptiveUPGD
    # Disable moments and noise for clean test of utility scaling
    # beta1=0.0 (no momentum), beta2=0.0 (second moment = grad^2 -> sqrt(v) = |grad| -> grad/sqrt(v) = sign(grad))
    optimizer = AdaptiveUPGD(
        [p1, p2], lr=1.0, beta_utility=0.999, sigma=0.0, weight_decay=0.0, beta1=0.0, beta2=0.0
    )

    # p1: grad = -1.0, param = 1.0 -> utility = 1.0 (high)
    # p2: grad = 1.0, param = 2.0 -> utility = -2.0 (low)
    p1.grad = torch.tensor([-1.0])
    p2.grad = torch.tensor([1.0])

    # Step 1
    optimizer.step()

    # For p1: grad=-1.0 -> sign(grad) = -1.0 -> update = -1.0 * (1 - scaled_utility1)
    # For p2: grad=1.0 -> sign(grad) = 1.0 -> update = 1.0 * (1 - scaled_utility2)
    # Effective scale:
    scale_p1 = p1.item() - 1.0
    scale_p2 = 2.0 - p2.item()

    print(f"\nAdaptiveUPGD Step 1 Verification:")
    print(f"  High Utility Scale: {scale_p1:.6f}")
    print(f"  Low Utility Scale: {scale_p2:.6f}")
    print(f"  Ratio: {scale_p2 / scale_p1:.2f}x")

    assert scale_p2 > scale_p1
    assert scale_p2 / scale_p1 > 2.0


def test_upgdw_bias_correction_first_step():
    """Verify that UPGDW correctly differentiates updates based on utility on step 1."""
    # Create two parameter tensors
    p1 = nn.Parameter(torch.tensor([1.0]))
    p2 = nn.Parameter(torch.tensor([2.0]))

    # Initialize UPGDW
    # betas=(0.0, 0.999) -> beta1=0.0 (no momentum), beta2=0.999 (utility decay & second moment)
    # eps=1.0 so that denominator is sqrt(grad^2) + 1.0.
    # Since grad1=-1.0, denom1 = sqrt(0.001) + 1.0 = 1.0316
    # exp_avg_corrected = -1.0
    # adaptive_grad = -1.0 / 1.0316 = -0.969
    optimizer = UPGDW([p1, p2], lr=1.0, betas=(0.0, 0.999), eps=1.0, sigma=0.0, weight_decay=0.0)

    # p1: grad = -1.0, param = 1.0 -> utility = 1.0 (high)
    # p2: grad = 1.0, param = 2.0 -> utility = -2.0 (low)
    p1.grad = torch.tensor([-1.0])
    p2.grad = torch.tensor([1.0])

    # Step 1
    optimizer.step()

    # Effective updates (approx):
    scale_p1 = (p1.item() - 1.0) / 0.969  # Normalize by adaptive grad magnitude
    scale_p2 = (2.0 - p2.item()) / (1.0 / (torch.tensor([0.001]).sqrt().item() + 1.0))

    print(f"\nUPGDW Step 1 Verification:")
    print(f"  High Utility Scale: {scale_p1:.6f}")
    print(f"  Low Utility Scale: {scale_p2:.6f}")
    print(f"  Ratio: {scale_p2 / scale_p1:.2f}x")

    assert scale_p2 > scale_p1
    assert scale_p2 / scale_p1 > 2.0
