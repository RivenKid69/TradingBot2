"""
Test suite for advantage normalization at group level during gradient accumulation.

This test ensures that when using gradient accumulation (multiple microbatches per
optimization step), advantage normalization happens at the GROUP level (across all
microbatches in the accumulation group) rather than per-microbatch.

Per-microbatch normalization would destroy relative importance between microbatches,
making all microbatches appear equally important regardless of their actual advantage
magnitudes. This is mathematically incorrect for gradient accumulation.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from typing import Optional, NamedTuple


class MockRolloutData(NamedTuple):
    """Mock rollout data for testing."""

    advantages: torch.Tensor
    mask: Optional[torch.Tensor] = None


def normalize_advantages_group_level(
    advantages_list: list[torch.Tensor], masks_list: list[Optional[torch.Tensor]]
) -> tuple[list[torch.Tensor], float, float]:
    """
    Normalize advantages at group level (mimics the production code logic).

    Returns:
        Tuple of (normalized_advantages_list, group_mean, group_std)
    """
    # Collect all advantages with masks applied
    all_advantages_flat = []
    for adv, mask in zip(advantages_list, masks_list):
        adv_flat = adv.reshape(-1)
        if mask is not None:
            mask_view = mask.reshape(-1)
            if mask_view.dtype == torch.bool:
                mask_float = mask_view.to(dtype=torch.float32)
            else:
                mask_float = mask_view.to(dtype=torch.float32)
            valid_mask = mask_float > 0
            all_advantages_flat.append(adv_flat[valid_mask])
        else:
            all_advantages_flat.append(adv_flat)

    # Compute group-level statistics
    if all_advantages_flat:
        group_advantages_concat = torch.cat(all_advantages_flat, dim=0)
        if group_advantages_concat.numel() > 0:
            group_mean = group_advantages_concat.mean()
            group_std = group_advantages_concat.std(unbiased=False)
            group_std_clamped = torch.clamp(group_std, min=1e-8)
        else:
            group_mean = torch.zeros(())
            group_std = torch.ones(())
            group_std_clamped = torch.ones(())
    else:
        group_mean = torch.zeros(())
        group_std = torch.ones(())
        group_std_clamped = torch.ones(())

    # Normalize using group-level statistics
    normalized_list = []
    for adv, mask in zip(advantages_list, masks_list):
        adv_flat = adv.reshape(-1)
        if mask is not None:
            mask_view = mask.reshape(-1)
            if mask_view.dtype == torch.bool:
                mask_float = mask_view.to(dtype=torch.float32)
            else:
                mask_float = mask_view.to(dtype=torch.float32)
            valid_mask = mask_float > 0

            normalized_flat = adv_flat.new_zeros(adv_flat.shape)
            normalized_flat[valid_mask] = (adv_flat[valid_mask] - group_mean) / group_std_clamped
            normalized_list.append(normalized_flat.view_as(adv))
        else:
            normalized = (adv - group_mean) / group_std_clamped
            normalized_list.append(normalized)

    return normalized_list, float(group_mean.item()), float(group_std.item())


def test_group_level_normalization_preserves_relative_importance():
    """
    Test that group-level normalization preserves relative importance between microbatches.

    This is the critical test demonstrating the problem with per-microbatch normalization:
    - Microbatch 1: [100, 120, 110] (very profitable)
    - Microbatch 2: [10, 12, 11] (moderate)
    - Microbatch 3: [1, 1.2, 1.1] (weak)
    - Microbatch 4: [-5, -6, -5.5] (unprofitable)

    With per-microbatch normalization, all would become [-1.22, 1.22, 0.00] (same!),
    destroying the 100x difference in actual importance.

    With group-level normalization, relative importance is preserved.
    """
    # Create microbatches with dramatically different advantage scales
    microbatch_1 = torch.tensor([100.0, 120.0, 110.0])
    microbatch_2 = torch.tensor([10.0, 12.0, 11.0])
    microbatch_3 = torch.tensor([1.0, 1.2, 1.1])
    microbatch_4 = torch.tensor([-5.0, -6.0, -5.5])

    advantages_list = [microbatch_1, microbatch_2, microbatch_3, microbatch_4]
    masks_list = [None, None, None, None]

    # Normalize at group level
    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Check that group statistics are computed correctly
    all_values = torch.cat([adv.reshape(-1) for adv in advantages_list])
    expected_mean = all_values.mean().item()
    expected_std = all_values.std(unbiased=False).item()

    assert group_mean == pytest.approx(expected_mean, abs=1e-5)
    assert group_std == pytest.approx(expected_std, abs=1e-5)

    # Check that relative importance is preserved
    # Microbatch 1 (most profitable) should have highest normalized values
    # Microbatch 4 (unprofitable) should have lowest normalized values
    mean_norm_1 = normalized_list[0].mean().item()
    mean_norm_2 = normalized_list[1].mean().item()
    mean_norm_3 = normalized_list[2].mean().item()
    mean_norm_4 = normalized_list[3].mean().item()

    # Assert ordering is preserved
    assert mean_norm_1 > mean_norm_2 > mean_norm_3 > mean_norm_4

    # Check that the difference is substantial (not collapsed to near-zero)
    assert abs(mean_norm_1 - mean_norm_4) > 1.0  # At least 1 std apart

    # Verify normalized values have correct mean and std overall
    all_normalized = torch.cat([norm.reshape(-1) for norm in normalized_list])
    assert all_normalized.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert all_normalized.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_group_level_normalization_with_masks():
    """
    Test that group-level normalization correctly handles masks.

    Masks should be applied when computing group statistics, so that only
    valid (unmasked) advantages contribute to mean/std calculation.
    """
    # Create microbatches where some values are masked
    microbatch_1 = torch.tensor([100.0, 999.0, 110.0])  # 999.0 will be masked
    microbatch_2 = torch.tensor([10.0, 12.0, 888.0])  # 888.0 will be masked

    # Masks: 1.0 = valid, 0.0 = masked
    mask_1 = torch.tensor([1.0, 0.0, 1.0])  # Mask out middle value
    mask_2 = torch.tensor([1.0, 1.0, 0.0])  # Mask out last value

    advantages_list = [microbatch_1, microbatch_2]
    masks_list = [mask_1, mask_2]

    # Normalize at group level
    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Check that group statistics exclude masked values
    valid_values = torch.tensor([100.0, 110.0, 10.0, 12.0])  # Only unmasked values
    expected_mean = valid_values.mean().item()
    expected_std = valid_values.std(unbiased=False).item()

    assert group_mean == pytest.approx(expected_mean, abs=1e-5)
    assert group_std == pytest.approx(expected_std, abs=1e-5)

    # Check that masked positions are zeroed out
    assert normalized_list[0][1].item() == pytest.approx(0.0, abs=1e-6)  # Masked value
    assert normalized_list[1][2].item() == pytest.approx(0.0, abs=1e-6)  # Masked value

    # Check that unmasked positions are properly normalized
    assert normalized_list[0][0].item() != pytest.approx(0.0, abs=0.1)  # Valid value
    assert normalized_list[0][2].item() != pytest.approx(0.0, abs=0.1)  # Valid value


def test_group_level_normalization_single_microbatch():
    """
    Test that group-level normalization works correctly with a single microbatch.

    This is a degenerate case where group-level and per-microbatch normalization
    should be equivalent.
    """
    microbatch = torch.tensor([10.0, 20.0, 15.0, 25.0])
    advantages_list = [microbatch]
    masks_list = [None]

    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Check statistics
    expected_mean = microbatch.mean().item()
    expected_std = microbatch.std(unbiased=False).item()

    assert group_mean == pytest.approx(expected_mean, abs=1e-5)
    assert group_std == pytest.approx(expected_std, abs=1e-5)

    # Check normalization
    normalized = normalized_list[0]
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_group_level_normalization_edge_case_all_zeros():
    """
    Test that group-level normalization handles edge case where all advantages are zero.
    """
    microbatch_1 = torch.zeros(3)
    microbatch_2 = torch.zeros(3)

    advantages_list = [microbatch_1, microbatch_2]
    masks_list = [None, None]

    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Mean should be 0, std should be clamped to min value
    assert group_mean == pytest.approx(0.0, abs=1e-6)
    # Since std is 0, it gets clamped to 1e-8 in the actual code
    # Normalized values should all be 0 (since (0 - 0) / epsilon = 0)
    assert torch.all(normalized_list[0] == 0.0)
    assert torch.all(normalized_list[1] == 0.0)


def test_group_level_normalization_edge_case_all_same():
    """
    Test that group-level normalization handles edge case where all advantages are the same.
    """
    value = 42.0
    microbatch_1 = torch.full((3,), value)
    microbatch_2 = torch.full((3,), value)

    advantages_list = [microbatch_1, microbatch_2]
    masks_list = [None, None]

    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Mean should be the constant value, std should be 0
    assert group_mean == pytest.approx(value, abs=1e-6)
    assert group_std == pytest.approx(0.0, abs=1e-6)

    # Normalized values should all be 0 (since (value - value) / epsilon = 0)
    assert torch.all(torch.abs(normalized_list[0]) < 1e-5)
    assert torch.all(torch.abs(normalized_list[1]) < 1e-5)


def test_group_level_normalization_boolean_masks():
    """
    Test that group-level normalization correctly handles boolean masks.
    """
    microbatch_1 = torch.tensor([100.0, 999.0, 110.0])
    microbatch_2 = torch.tensor([10.0, 12.0, 888.0])

    # Boolean masks
    mask_1 = torch.tensor([True, False, True])
    mask_2 = torch.tensor([True, True, False])

    advantages_list = [microbatch_1, microbatch_2]
    masks_list = [mask_1, mask_2]

    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Check that group statistics exclude masked values
    valid_values = torch.tensor([100.0, 110.0, 10.0, 12.0])
    expected_mean = valid_values.mean().item()
    expected_std = valid_values.std(unbiased=False).item()

    assert group_mean == pytest.approx(expected_mean, abs=1e-5)
    assert group_std == pytest.approx(expected_std, abs=1e-5)


def test_group_level_vs_per_microbatch_difference():
    """
    Explicit test showing the mathematical difference between group-level and
    per-microbatch normalization.

    This test quantifies how much information is lost with per-microbatch normalization.
    """
    # Create the example from the problem description
    microbatch_1 = torch.tensor([100.0, 120.0, 110.0])
    microbatch_2 = torch.tensor([10.0, 12.0, 11.0])
    microbatch_3 = torch.tensor([1.0, 1.2, 1.1])
    microbatch_4 = torch.tensor([-5.0, -6.0, -5.5])

    advantages_list = [microbatch_1, microbatch_2, microbatch_3, microbatch_4]
    masks_list = [None, None, None, None]

    # GROUP-LEVEL normalization (correct)
    group_normalized, _, _ = normalize_advantages_group_level(advantages_list, masks_list)

    # PER-MICROBATCH normalization (incorrect for gradient accumulation)
    per_micro_normalized = []
    for adv in advantages_list:
        mean = adv.mean()
        std = torch.clamp(adv.std(unbiased=False), min=1e-8)
        per_micro_normalized.append((adv - mean) / std)

    # Compare the results
    # Per-microbatch: all microbatches will have nearly identical normalized patterns
    per_micro_means = [norm.mean().item() for norm in per_micro_normalized]
    per_micro_stds = [norm.std(unbiased=False).item() for norm in per_micro_normalized]

    # All should have mean ≈ 0 and std ≈ 1
    for mean in per_micro_means:
        assert mean == pytest.approx(0.0, abs=1e-5)
    for std in per_micro_stds:
        assert std == pytest.approx(1.0, abs=1e-5)

    # Group-level: means should vary significantly between microbatches
    group_means = [norm.mean().item() for norm in group_normalized]

    # The range of means should be substantial with group-level normalization
    group_mean_range = max(group_means) - min(group_means)
    assert group_mean_range > 1.5  # At least 1.5 std units difference

    # But with per-microbatch, the range would be near zero
    per_micro_mean_range = max(per_micro_means) - min(per_micro_means)
    assert per_micro_mean_range < 1e-4  # Essentially zero

    # This demonstrates the information loss with per-microbatch normalization
    information_loss_ratio = group_mean_range / (per_micro_mean_range + 1e-10)
    assert information_loss_ratio > 1000  # At least 1000x information loss!


def test_group_level_normalization_multidimensional():
    """
    Test that group-level normalization works with multi-dimensional advantage tensors.
    """
    # Create 2D advantage tensors (e.g., [batch, time])
    microbatch_1 = torch.tensor([[100.0, 120.0], [110.0, 105.0]])
    microbatch_2 = torch.tensor([[10.0, 12.0], [11.0, 9.0]])

    advantages_list = [microbatch_1, microbatch_2]
    masks_list = [None, None]

    normalized_list, group_mean, group_std = normalize_advantages_group_level(
        advantages_list, masks_list
    )

    # Check that tensors are flattened for statistics
    all_flat = torch.cat([microbatch_1.reshape(-1), microbatch_2.reshape(-1)])
    expected_mean = all_flat.mean().item()
    expected_std = all_flat.std(unbiased=False).item()

    assert group_mean == pytest.approx(expected_mean, abs=1e-5)
    assert group_std == pytest.approx(expected_std, abs=1e-5)

    # Check that output shape matches input shape
    assert normalized_list[0].shape == microbatch_1.shape
    assert normalized_list[1].shape == microbatch_2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
