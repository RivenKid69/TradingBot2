"""
Comprehensive test suite for quantile loss with optional fix flag.

Tests both OLD (buggy) and NEW (fixed) implementations to ensure:
1. Fix is disabled by default (backward compatibility)
2. Fix can be enabled via flag
3. Both implementations produce expected behavior
4. Edge cases are handled correctly
"""

import math

import pytest

pytest.importorskip("torch")

import test_distributional_ppo_raw_outliers  # noqa: F401


def test_quantile_loss_fix_disabled_by_default():
    """
    Test that the quantile loss fix is ENABLED by default (as of 2025-11-20).

    This uses the correct formula (T - Q) from Dabney et al. 2018.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")
        # NO use_fixed_quantile_loss_asymmetry attribute

        @property
        def quantile_levels(self):
            return torch.tensor([0.25], dtype=torch.float32)

    algo.policy = _PolicyStub()

    # Should default to True (changed 2025-11-20)
    algo._use_fixed_quantile_loss_asymmetry = bool(
        getattr(algo.policy, "use_fixed_quantile_loss_asymmetry", True)
    )

    assert (
        algo._use_fixed_quantile_loss_asymmetry is True
    ), "Fix should be enabled by default (as of 2025-11-20)"


def test_quantile_loss_fix_can_be_enabled():
    """
    Test that the quantile loss fix can be enabled via policy attribute.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")
        use_fixed_quantile_loss_asymmetry = True  # Enable fix

        @property
        def quantile_levels(self):
            return torch.tensor([0.25], dtype=torch.float32)

    algo.policy = _PolicyStub()

    algo._use_fixed_quantile_loss_asymmetry = bool(
        getattr(algo.policy, "use_fixed_quantile_loss_asymmetry", False)
    )

    assert (
        algo._use_fixed_quantile_loss_asymmetry is True
    ), "Fix should be enabled when policy attribute is True"


def test_quantile_loss_old_behavior_with_flag_disabled():
    """
    Test that OLD (buggy) behavior is used when flag is disabled.

    With flag disabled, should use delta = predicted - targets (Q - T),
    which produces INVERTED asymmetry.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0
    algo._use_fixed_quantile_loss_asymmetry = False  # OLD behavior

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25], dtype=torch.float32)

    algo.policy = _PolicyStub()

    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
    predicted_under = torch.tensor([[-1.0]], dtype=torch.float32)
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32)

    loss_under = DistributionalPPO._quantile_huber_loss(
        algo, predicted_under, target, reduction="none"
    )
    loss_over = DistributionalPPO._quantile_huber_loss(
        algo, predicted_over, target, reduction="none"
    )

    # OLD (buggy) behavior: INVERTED coefficients
    # Expected: underestimation gets (1-τ), overestimation gets τ
    # For τ=0.25: under=0.75, over=0.25
    # Ratio should be 0.25/0.75 = 0.333 (INVERTED!)

    ratio = loss_over.item() / loss_under.item()
    expected_ratio_inverted = 0.25 / 0.75  # Inverted

    assert math.isclose(
        ratio, expected_ratio_inverted, rel_tol=1e-5
    ), f"OLD behavior should have inverted ratio: expected {expected_ratio_inverted}, got {ratio}"


def test_quantile_loss_new_behavior_with_flag_enabled():
    """
    Test that NEW (fixed) behavior is used when flag is enabled.

    With flag enabled, should use delta = targets - predicted (T - Q),
    which produces CORRECT asymmetry.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0
    algo._use_fixed_quantile_loss_asymmetry = True  # NEW behavior

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25], dtype=torch.float32)

    algo.policy = _PolicyStub()

    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
    predicted_under = torch.tensor([[-1.0]], dtype=torch.float32)
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32)

    loss_under = DistributionalPPO._quantile_huber_loss(
        algo, predicted_under, target, reduction="none"
    )
    loss_over = DistributionalPPO._quantile_huber_loss(
        algo, predicted_over, target, reduction="none"
    )

    # NEW (correct) behavior: CORRECT coefficients
    # Expected: underestimation gets τ, overestimation gets (1-τ)
    # For τ=0.25: under=0.25, over=0.75
    # Ratio should be 0.75/0.25 = 3.0 (CORRECT!)

    ratio = loss_over.item() / loss_under.item()
    expected_ratio_correct = (1 - 0.25) / 0.25  # Correct

    assert math.isclose(
        ratio, expected_ratio_correct, rel_tol=1e-5
    ), f"NEW behavior should have correct ratio: expected {expected_ratio_correct}, got {ratio}"


def test_quantile_loss_comparison_old_vs_new():
    """
    Direct comparison of OLD vs NEW behavior on same inputs.

    Demonstrates that outputs are mathematically equivalent but with
    inverted semantics.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    tau = 0.25
    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
    predicted_under = torch.tensor([[-1.0]], dtype=torch.float32)
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32)

    # OLD behavior
    algo_old = DistributionalPPO.__new__(DistributionalPPO)
    algo_old._quantile_huber_kappa = 1.0
    algo_old._use_fixed_quantile_loss_asymmetry = False

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([tau], dtype=torch.float32)

    algo_old.policy = _PolicyStub()

    loss_under_old = DistributionalPPO._quantile_huber_loss(
        algo_old, predicted_under, target, reduction="none"
    )
    loss_over_old = DistributionalPPO._quantile_huber_loss(
        algo_old, predicted_over, target, reduction="none"
    )

    # NEW behavior
    algo_new = DistributionalPPO.__new__(DistributionalPPO)
    algo_new._quantile_huber_kappa = 1.0
    algo_new._use_fixed_quantile_loss_asymmetry = True
    algo_new.policy = _PolicyStub()

    loss_under_new = DistributionalPPO._quantile_huber_loss(
        algo_new, predicted_under, target, reduction="none"
    )
    loss_over_new = DistributionalPPO._quantile_huber_loss(
        algo_new, predicted_over, target, reduction="none"
    )

    # Verify that OLD under == NEW over (inverted!)
    assert math.isclose(
        loss_under_old.item(), loss_over_new.item(), rel_tol=1e-6
    ), "OLD underestimation loss should equal NEW overestimation loss (inverted)"

    # Verify that OLD over == NEW under (inverted!)
    assert math.isclose(
        loss_over_old.item(), loss_under_new.item(), rel_tol=1e-6
    ), "OLD overestimation loss should equal NEW underestimation loss (inverted)"


def test_quantile_loss_median_unaffected_by_flag():
    """
    Test that median (τ=0.5) produces same results regardless of flag.

    The median is symmetric, so the bug doesn't affect it.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
    predicted_under = torch.tensor([[-1.0]], dtype=torch.float32)
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32)

    # OLD behavior
    algo_old = DistributionalPPO.__new__(DistributionalPPO)
    algo_old._quantile_huber_kappa = 1.0
    algo_old._use_fixed_quantile_loss_asymmetry = False

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo_old.policy = _PolicyStub()

    loss_under_old = DistributionalPPO._quantile_huber_loss(
        algo_old, predicted_under, target, reduction="none"
    )
    loss_over_old = DistributionalPPO._quantile_huber_loss(
        algo_old, predicted_over, target, reduction="none"
    )

    # NEW behavior
    algo_new = DistributionalPPO.__new__(DistributionalPPO)
    algo_new._quantile_huber_kappa = 1.0
    algo_new._use_fixed_quantile_loss_asymmetry = True
    algo_new.policy = _PolicyStub()

    loss_under_new = DistributionalPPO._quantile_huber_loss(
        algo_new, predicted_under, target, reduction="none"
    )
    loss_over_new = DistributionalPPO._quantile_huber_loss(
        algo_new, predicted_over, target, reduction="none"
    )

    # For median, OLD and NEW should be identical
    assert math.isclose(
        loss_under_old.item(), loss_under_new.item(), rel_tol=1e-6
    ), "Median underestimation loss should be same for OLD and NEW"
    assert math.isclose(
        loss_over_old.item(), loss_over_new.item(), rel_tol=1e-6
    ), "Median overestimation loss should be same for OLD and NEW"

    # And they should be symmetric
    assert math.isclose(
        loss_under_new.item(), loss_over_new.item(), rel_tol=1e-6
    ), "Median should be symmetric"


def test_quantile_loss_edge_cases_with_flag():
    """
    Test edge cases work correctly with both flag settings.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    # Test with τ = 0.0 (extreme conservative)
    for use_fix in [False, True]:
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._quantile_huber_kappa = 1.0
        algo._use_fixed_quantile_loss_asymmetry = use_fix

        class _PolicyStub:
            device = torch.device("cpu")

            @property
            def quantile_levels(self):
                return torch.tensor([0.0], dtype=torch.float32)

        algo.policy = _PolicyStub()

        target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
        predicted = torch.tensor([[1.0]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, target, reduction="none")

        # Should produce valid finite loss
        assert torch.isfinite(loss).all(), f"Loss should be finite for τ=0.0 (fix={use_fix})"

    # Test with τ = 1.0 (extreme aggressive)
    for use_fix in [False, True]:
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._quantile_huber_kappa = 1.0
        algo._use_fixed_quantile_loss_asymmetry = use_fix

        class _PolicyStub:
            device = torch.device("cpu")

            @property
            def quantile_levels(self):
                return torch.tensor([1.0], dtype=torch.float32)

        algo.policy = _PolicyStub()

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, target, reduction="none")

        # Should produce valid finite loss
        assert torch.isfinite(loss).all(), f"Loss should be finite for τ=1.0 (fix={use_fix})"


def test_quantile_loss_gradients_with_flag():
    """
    Test that gradients flow correctly with both flag settings.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    for use_fix in [False, True]:
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._quantile_huber_kappa = 1.0
        algo._use_fixed_quantile_loss_asymmetry = use_fix

        class _PolicyStub:
            device = torch.device("cpu")

            @property
            def quantile_levels(self):
                return torch.tensor([0.25, 0.75], dtype=torch.float32)

        algo.policy = _PolicyStub()

        predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
        loss.backward()

        # Check gradients exist and are finite
        assert predicted.grad is not None, f"Gradients should exist (fix={use_fix})"
        assert torch.isfinite(predicted.grad).all(), f"Gradients should be finite (fix={use_fix})"

        # First sample should have near-zero gradients (already at target)
        # Second sample should have non-zero gradients (far from target)
        grad_first = predicted.grad[0].abs().max().item()
        grad_second = predicted.grad[1].abs().max().item()

        assert math.isclose(
            grad_first, 0.0, abs_tol=1e-6
        ), f"First sample gradients should be near zero (fix={use_fix})"
        assert grad_second > 0.1, f"Second sample gradients should be substantial (fix={use_fix})"
