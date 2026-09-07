"""
DEEP COMPREHENSIVE TEST SUITE for Quantile Huber Loss

This test suite provides 100% coverage of edge cases, numerical stability,
gradient flow, and integration scenarios for the quantile Huber loss fix.

Goal: Ensure absolute correctness of the formula:
    ρ^κ_τ(u) = |τ - I{u<0}| · L^κ_H(u)

where L^κ_H(u) = { 0.5 * u²,              if |u| <= κ
                 { κ * (|u| - 0.5 * κ),   if |u| > κ

References:
- Dabney et al., 2018: "Distributional RL with Quantile Regression"
- Google Dopamine implementation
- Stable-Baselines3-Contrib implementation
"""

import math
import sys

import pytest

pytest.importorskip("torch")

import test_distributional_ppo_raw_outliers  # noqa: F401


# ==============================================================================
# PART 1: EXTREME EDGE CASES
# ==============================================================================


def test_quantile_huber_loss_extreme_kappa_values() -> None:
    """
    Test with extreme kappa values to ensure numerical stability.

    Edge cases:
    - Very small kappa (near zero)
    - Very large kappa
    - Minimum allowed kappa (1e-6)
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test various extreme kappa values
    test_cases = [
        (1e-6, "minimum kappa"),
        (1e-5, "very small kappa"),
        (1e-3, "small kappa"),
        (1.0, "standard kappa"),
        (10.0, "large kappa"),
        (100.0, "very large kappa"),
        (1000.0, "extreme kappa"),
    ]

    for kappa, description in test_cases:
        algo._quantile_huber_kappa = kappa

        # Small error (quadratic region)
        error_small = kappa * 0.1
        predicted = torch.tensor([[0.0]], dtype=torch.float32)
        targets = torch.tensor([[error_small]], dtype=torch.float32)

        loss_small = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        # Loss should be finite and non-negative
        assert torch.isfinite(loss_small).all(), f"{description}: loss is not finite"
        assert loss_small >= 0, f"{description}: loss is negative"

        # In quadratic region: loss = 0.5 * |tau - I| * error²
        expected_small = 0.5 * 0.5 * error_small**2
        assert math.isclose(
            loss_small.item(), expected_small, rel_tol=1e-4
        ), f"{description} (quadratic): Expected {expected_small:.8f}, got {loss_small.item():.8f}"

        # Large error (linear region)
        error_large = kappa * 3.0
        targets_large = torch.tensor([[error_large]], dtype=torch.float32)

        loss_large = DistributionalPPO._quantile_huber_loss(algo, predicted, targets_large)

        assert torch.isfinite(loss_large).all(), f"{description}: loss is not finite"
        assert loss_large >= 0, f"{description}: loss is negative"

        # In linear region: loss = |tau - I| * kappa * (|error| - 0.5 * kappa)
        expected_large = 0.5 * kappa * (abs(error_large) - 0.5 * kappa)
        assert math.isclose(
            loss_large.item(), expected_large, rel_tol=1e-4
        ), f"{description} (linear): Expected {expected_large:.8f}, got {loss_large.item():.8f}"


def test_quantile_huber_loss_extreme_errors() -> None:
    """Test with extremely large and small prediction errors."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    extreme_errors = [
        (1e-8, "tiny error"),
        (1e-4, "small error"),
        (1e2, "large error"),
        (1e4, "very large error"),
        (1e6, "extreme error"),
    ]

    for error, description in extreme_errors:
        predicted = torch.tensor([[0.0]], dtype=torch.float32)
        targets = torch.tensor([[error]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        # Loss should always be finite and non-negative
        assert torch.isfinite(loss).all(), f"{description}: loss is not finite"
        assert loss >= 0, f"{description}: loss is negative"


def test_quantile_huber_loss_zero_error() -> None:
    """Test that zero error gives zero loss."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Perfect predictions
    predicted = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([[5.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    assert math.isclose(loss.item(), 0.0, abs_tol=1e-7), f"Expected 0.0, got {loss.item():.10f}"


def test_quantile_huber_loss_negative_targets() -> None:
    """Test with negative target values."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Negative target
    predicted = torch.tensor([[0.0]], dtype=torch.float32)
    targets = torch.tensor([[-5.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    # delta = predicted - target = 0 - (-5) = 5 (positive, linear region)
    # indicator = I{delta < 0} = 0
    # |tau - indicator| = |0.5 - 0| = 0.5
    # huber = kappa * (|delta| - 0.5 * kappa) = 1 * (5 - 0.5) = 4.5
    # loss = 0.5 * 4.5 = 2.25
    expected = 0.5 * 1.0 * (5.0 - 0.5 * 1.0)

    assert math.isclose(
        loss.item(), expected, rel_tol=1e-5
    ), f"Expected {expected:.6f}, got {loss.item():.6f}"


# ==============================================================================
# PART 2: QUANTILE LEVEL CONFIGURATIONS
# ==============================================================================


def test_quantile_huber_loss_various_quantile_configurations() -> None:
    """Test with different quantile level configurations."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    quantile_configs = [
        ([0.5], "single median"),
        ([0.1, 0.9], "tail quantiles"),
        ([0.25, 0.5, 0.75], "quartiles"),
        ([0.1, 0.3, 0.5, 0.7, 0.9], "five quantiles"),
        ([0.05, 0.25, 0.5, 0.75, 0.95], "quintiles with tails"),
        (list(torch.linspace(0.1, 0.9, 9)), "nine uniform quantiles"),
        ([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99], "seven with extreme tails"),
    ]

    for levels, description in quantile_configs:

        class PolicyStub:
            device = torch.device("cpu")

            @property
            def quantile_levels(self):
                return torch.tensor(levels, dtype=torch.float32)

        algo.policy = PolicyStub()

        num_quantiles = len(levels)

        # Create predictions and targets
        predicted = torch.zeros((3, num_quantiles), dtype=torch.float32)
        targets = torch.tensor([[1.0], [0.0], [-1.0]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        # Loss should be finite and non-negative
        assert torch.isfinite(loss).all(), f"{description}: loss is not finite"
        assert loss >= 0, f"{description}: loss is negative"

        # Loss should be reasonable (not too large)
        assert loss < 100.0, f"{description}: loss is unexpectedly large ({loss.item():.2f})"


def test_quantile_huber_loss_extreme_quantile_levels() -> None:
    """Test with extreme quantile levels (near 0 and 1)."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    # Very conservative (near 0) and optimistic (near 1) quantiles
    extreme_configs = [
        ([0.001, 0.999], "extreme tails"),
        ([0.01, 0.5, 0.99], "wide range"),
        ([0.05], "5th percentile only"),
        ([0.95], "95th percentile only"),
    ]

    for levels, description in extreme_configs:

        class PolicyStub:
            device = torch.device("cpu")

            @property
            def quantile_levels(self):
                return torch.tensor(levels, dtype=torch.float32)

        algo.policy = PolicyStub()

        num_quantiles = len(levels)
        predicted = torch.zeros((2, num_quantiles), dtype=torch.float32)
        targets = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        assert torch.isfinite(loss).all(), f"{description}: loss is not finite"
        assert loss >= 0, f"{description}: loss is negative"


# ==============================================================================
# PART 3: GRADIENT FLOW AND BACKPROPAGATION
# ==============================================================================


def test_quantile_huber_loss_gradient_flow_quadratic_region() -> None:
    """Test gradient flow in quadratic region."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Small error (quadratic region)
    predicted = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[0.5]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # Gradients should exist
    assert predicted.grad is not None

    # Gradients should be finite
    assert torch.isfinite(predicted.grad).all()

    # Gradients should not be all zero (we have non-zero error)
    assert predicted.grad.abs().sum() > 0

    # In quadratic region: ∂L/∂predicted = |τ - I| * delta
    # delta = predicted - target
    # For each quantile, gradient should be proportional to delta
    deltas = predicted.detach() - targets

    # Check gradient magnitudes are reasonable
    assert predicted.grad.abs().max() < 10.0, "Gradient too large"


def test_quantile_huber_loss_gradient_flow_linear_region() -> None:
    """Test gradient flow in linear region."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Large error (linear region)
    predicted = torch.tensor([[5.0, 6.0, 7.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[0.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # Gradients should exist and be finite
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()

    # In linear region: ∂L/∂predicted = |τ - I| * kappa * sign(delta)
    # All deltas are positive (predicted > target)
    # Gradients should all be positive
    assert (predicted.grad > 0).all(), "All gradients should be positive"

    # Gradient magnitude should be proportional to |tau - I|
    tau = torch.tensor([0.1, 0.5, 0.9])
    # delta > 0, so I = 0
    expected_weights = tau  # |tau - 0|

    # Gradients should be proportional to these weights
    grad_ratios = predicted.grad[0] / expected_weights

    # All ratios should be approximately equal (all in linear region with same kappa)
    assert torch.allclose(grad_ratios, grad_ratios.mean(), rtol=1e-4)


def test_quantile_huber_loss_second_order_gradients() -> None:
    """Test second-order gradients (for advanced optimizers)."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    predicted = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    # Compute first-order gradient
    (grad,) = torch.autograd.grad(loss, predicted, create_graph=True)

    # Compute second-order gradient
    grad_sum = grad.sum()
    (grad2,) = torch.autograd.grad(grad_sum, predicted)

    # Second-order gradient should exist and be finite
    assert grad2 is not None
    assert torch.isfinite(grad2).all()

    # In quadratic region, second derivative should be positive
    # (convex function)
    assert grad2.item() > 0


def test_quantile_huber_loss_gradient_accumulation() -> None:
    """Test that gradients accumulate correctly over multiple batches."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    predicted = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    # First batch
    targets1 = torch.tensor([[1.0]], dtype=torch.float32)
    loss1 = DistributionalPPO._quantile_huber_loss(algo, predicted, targets1)
    loss1.backward()

    grad1 = predicted.grad.clone()

    # Second batch (accumulate)
    targets2 = torch.tensor([[2.0]], dtype=torch.float32)
    loss2 = DistributionalPPO._quantile_huber_loss(algo, predicted, targets2)
    loss2.backward()

    grad_accumulated = predicted.grad

    # Compute expected gradient (sum of individual gradients)
    predicted_clean = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    targets_combined = torch.tensor([[2.0]], dtype=torch.float32)
    loss_combined = DistributionalPPO._quantile_huber_loss(algo, predicted_clean, targets_combined)
    loss_combined.backward()
    grad2_only = predicted_clean.grad

    # Accumulated gradient should equal grad1 + grad2
    expected_accumulated = grad1 + grad2_only

    assert torch.allclose(grad_accumulated, expected_accumulated, rtol=1e-5)


# ==============================================================================
# PART 4: NUMERICAL STABILITY
# ==============================================================================


def test_quantile_huber_loss_numerical_stability_mixed_precision() -> None:
    """Test numerical stability with different dtypes."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test with float32
    predicted_f32 = torch.tensor([[0.5]], dtype=torch.float32)
    targets_f32 = torch.tensor([[1.0]], dtype=torch.float32)
    loss_f32 = DistributionalPPO._quantile_huber_loss(algo, predicted_f32, targets_f32)

    # Test with float64
    predicted_f64 = torch.tensor([[0.5]], dtype=torch.float64)
    targets_f64 = torch.tensor([[1.0]], dtype=torch.float64)
    loss_f64 = DistributionalPPO._quantile_huber_loss(algo, predicted_f64, targets_f64)

    # Results should be very close
    assert math.isclose(loss_f32.item(), loss_f64.item(), rel_tol=1e-5)


def test_quantile_huber_loss_batch_invariance() -> None:
    """Test that loss is batch-size invariant (returns mean)."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Single sample
    predicted_single = torch.tensor([[0.5]], dtype=torch.float32)
    targets_single = torch.tensor([[1.0]], dtype=torch.float32)
    loss_single = DistributionalPPO._quantile_huber_loss(algo, predicted_single, targets_single)

    # Replicate the same sample 10 times
    predicted_batch = predicted_single.repeat(10, 1)
    targets_batch = targets_single.repeat(10, 1)
    loss_batch = DistributionalPPO._quantile_huber_loss(algo, predicted_batch, targets_batch)

    # Loss should be the same (mean over batch)
    assert math.isclose(loss_single.item(), loss_batch.item(), rel_tol=1e-6)


def test_quantile_huber_loss_large_batch_stability() -> None:
    """Test with large batch sizes."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)

    algo.policy = PolicyStub()

    batch_sizes = [1, 10, 100, 1000, 10000]

    for batch_size in batch_sizes:
        predicted = torch.randn(batch_size, 3, dtype=torch.float32)
        targets = torch.randn(batch_size, 1, dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        # Loss should be finite
        assert torch.isfinite(loss).all(), f"Loss not finite for batch_size={batch_size}"

        # Loss should be non-negative
        assert loss >= 0, f"Loss negative for batch_size={batch_size}"

        # Loss should be reasonable
        assert loss < 1000.0, f"Loss too large for batch_size={batch_size}: {loss.item():.2f}"


# ==============================================================================
# PART 5: INTEGRATION WITH OTHER COMPONENTS
# ==============================================================================


def test_quantile_huber_loss_with_various_target_shapes() -> None:
    """Test that target reshaping works correctly."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    predicted = torch.tensor([[0.5]], dtype=torch.float32)

    # Different target shapes (should all be equivalent)
    target_shapes = [
        torch.tensor([1.0]),  # [1]
        torch.tensor([[1.0]]),  # [1, 1]
        torch.tensor([[[1.0]]]),  # [1, 1, 1]
    ]

    losses = []
    for targets in target_shapes:
        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
        losses.append(loss.item())

    # All losses should be identical
    for i, loss_val in enumerate(losses[1:], 1):
        assert math.isclose(
            losses[0], loss_val, rel_tol=1e-6
        ), f"Loss {i} differs: {losses[0]:.8f} vs {loss_val:.8f}"


def test_quantile_huber_loss_indicator_correctness() -> None:
    """Deep test of indicator function I{delta < 0}."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            # Use extreme quantile to amplify the effect
            return torch.tensor([0.01], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test case 1: delta > 0 (predicted > target)
    # indicator should be 0
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32)
    targets_over = torch.tensor([[0.0]], dtype=torch.float32)
    loss_over = DistributionalPPO._quantile_huber_loss(algo, predicted_over, targets_over)

    # delta = 1.0 - 0.0 = 1.0 > 0, so indicator = 0
    # |tau - indicator| = |0.01 - 0| = 0.01
    # huber (linear): 1.0 * (1.0 - 0.5) = 0.5
    # loss = 0.01 * 0.5 = 0.005
    expected_over = 0.01 * 1.0 * (1.0 - 0.5 * 1.0)
    assert math.isclose(loss_over.item(), expected_over, rel_tol=1e-5)

    # Test case 2: delta < 0 (predicted < target)
    # indicator should be 1
    predicted_under = torch.tensor([[0.0]], dtype=torch.float32)
    targets_under = torch.tensor([[1.0]], dtype=torch.float32)
    loss_under = DistributionalPPO._quantile_huber_loss(algo, predicted_under, targets_under)

    # delta = 0.0 - 1.0 = -1.0 < 0, so indicator = 1
    # |tau - indicator| = |0.01 - 1| = 0.99
    # huber (linear): 1.0 * (1.0 - 0.5) = 0.5
    # loss = 0.99 * 0.5 = 0.495
    expected_under = 0.99 * 1.0 * (1.0 - 0.5 * 1.0)
    assert math.isclose(loss_under.item(), expected_under, rel_tol=1e-5)

    # Ratio should be 0.99 / 0.01 = 99
    ratio = loss_under.item() / loss_over.item()
    assert math.isclose(ratio, 99.0, rel_tol=0.01)


def test_quantile_huber_loss_symmetry_check() -> None:
    """
    Test symmetry properties.

    For tau=0.5, loss should be symmetric for equal-magnitude errors.
    For tau≠0.5, loss should be asymmetric.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    # Test 1: tau = 0.5 (median) - should be symmetric
    class PolicyStubMedian:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStubMedian()

    error_magnitude = 0.3

    # Positive error
    predicted_pos = torch.tensor([[error_magnitude]], dtype=torch.float32)
    targets_pos = torch.tensor([[0.0]], dtype=torch.float32)
    loss_pos = DistributionalPPO._quantile_huber_loss(algo, predicted_pos, targets_pos)

    # Negative error (same magnitude)
    predicted_neg = torch.tensor([[-error_magnitude]], dtype=torch.float32)
    targets_neg = torch.tensor([[0.0]], dtype=torch.float32)
    loss_neg = DistributionalPPO._quantile_huber_loss(algo, predicted_neg, targets_neg)

    # For median, should be symmetric
    assert math.isclose(
        loss_pos.item(), loss_neg.item(), rel_tol=1e-5
    ), f"Median should be symmetric: {loss_pos.item():.6f} vs {loss_neg.item():.6f}"

    # Test 2: tau = 0.1 - should be asymmetric
    class PolicyStubAsym:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1], dtype=torch.float32)

    algo.policy = PolicyStubAsym()

    loss_pos_asym = DistributionalPPO._quantile_huber_loss(algo, predicted_pos, targets_pos)
    loss_neg_asym = DistributionalPPO._quantile_huber_loss(algo, predicted_neg, targets_neg)

    # For tau=0.1, should be asymmetric
    assert not math.isclose(
        loss_pos_asym.item(), loss_neg_asym.item(), rel_tol=0.1
    ), f"Should be asymmetric for tau=0.1: {loss_pos_asym.item():.6f} vs {loss_neg_asym.item():.6f}"


# ==============================================================================
# PART 6: CORRECTNESS VERIFICATION AGAINST REFERENCE
# ==============================================================================


def test_quantile_huber_loss_manual_calculation_verification() -> None:
    """
    Manually calculate expected loss and compare with implementation.

    This is the ultimate correctness test.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test case with known values
    predicted = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([[1.5]], dtype=torch.float32)

    # Manual calculation:
    # tau = [0.25, 0.5, 0.75]
    # delta = predicted - target = [-1.5, -0.5, 0.5]
    # indicator = I{delta < 0} = [1, 1, 0]
    # |tau - indicator| = [|0.25-1|, |0.5-1|, |0.75-0|] = [0.75, 0.5, 0.75]

    # For delta = -1.5: |delta| = 1.5 > kappa, linear region
    #   huber = kappa * (|delta| - 0.5*kappa) = 1.0 * (1.5 - 0.5) = 1.0
    #   loss_component = 0.75 * 1.0 = 0.75

    # For delta = -0.5: |delta| = 0.5 <= kappa, quadratic region
    #   huber = 0.5 * delta² = 0.5 * 0.25 = 0.125
    #   loss_component = 0.5 * 0.125 = 0.0625

    # For delta = 0.5: |delta| = 0.5 <= kappa, quadratic region
    #   huber = 0.5 * delta² = 0.5 * 0.25 = 0.125
    #   loss_component = 0.75 * 0.125 = 0.09375

    # Total loss = mean([0.75, 0.0625, 0.09375]) = 0.90625 / 3 = 0.302083...
    expected_loss = (0.75 + 0.0625 + 0.09375) / 3.0

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    assert math.isclose(
        loss.item(), expected_loss, rel_tol=1e-5
    ), f"Expected {expected_loss:.8f}, got {loss.item():.8f}"


def test_quantile_huber_loss_no_kappa_division_comprehensive() -> None:
    """
    Comprehensive test ensuring NO division by kappa.

    This is the CRITICAL regression test for the fix.
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test across multiple kappa values and error magnitudes
    test_cases = [
        # (kappa, error, region)
        (0.5, 0.1, "quadratic"),
        (0.5, 1.0, "linear"),
        (1.0, 0.5, "quadratic"),
        (1.0, 2.0, "linear"),
        (2.0, 0.5, "quadratic"),
        (2.0, 5.0, "linear"),
        (5.0, 1.0, "quadratic"),
        (5.0, 10.0, "linear"),
    ]

    for kappa, error, region in test_cases:
        algo._quantile_huber_kappa = kappa

        predicted = torch.tensor([[0.0]], dtype=torch.float32)
        targets = torch.tensor([[error]], dtype=torch.float32)

        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

        # Calculate expected loss WITHOUT division by kappa
        delta = abs(error)
        tau = 0.5
        indicator = 1.0 if (-error < 0) else 0.0
        weight = abs(tau - indicator)

        if delta <= kappa:
            # Quadratic region
            huber = 0.5 * error**2
        else:
            # Linear region
            huber = kappa * (delta - 0.5 * kappa)

        expected = weight * huber

        assert math.isclose(loss.item(), expected, rel_tol=1e-5), (
            f"κ={kappa}, error={error}, {region}: "
            f"Expected {expected:.8f}, got {loss.item():.8f}"
        )

        # CRITICAL: If division by kappa were present, we would get:
        # wrong_expected = weight * huber / kappa
        # Verify we DON'T get this wrong value
        wrong_expected = weight * huber / kappa

        if kappa != 1.0:
            # For kappa != 1, the wrong formula should give a different result
            assert not math.isclose(loss.item(), wrong_expected, rel_tol=1e-3), (
                f"κ={kappa}: Loss matches WRONG formula (with division by kappa)! "
                f"This suggests the bug is still present."
            )


def test_quantile_huber_loss_matches_dopamine_implementation() -> None:
    """
    Test that our implementation matches Google Dopamine's formula.

    Reference: dopamine/jax/agents/quantile/quantile_agent.py
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)

    algo.policy = PolicyStub()

    # Test with various inputs
    predicted = torch.tensor([[0.5, 1.0, 1.5], [-0.5, 0.0, 0.5]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    # Manually implement Dopamine's formula for comparison
    tau = torch.tensor([0.1, 0.5, 0.9])
    kappa = 1.0

    total_loss = 0.0
    num_samples = 0

    for sample_idx in range(predicted.shape[0]):
        target = targets[sample_idx, 0]
        for quant_idx in range(predicted.shape[1]):
            pred = predicted[sample_idx, quant_idx]

            # Bellman error (called 'u' in formula)
            bellman_error = target - pred

            # Huber loss
            abs_error = abs(bellman_error)
            if abs_error <= kappa:
                huber = 0.5 * bellman_error**2
            else:
                huber = kappa * (abs_error - 0.5 * kappa)

            # Quantile weighting
            indicator = 1.0 if bellman_error < 0 else 0.0
            tau_diff = abs(tau[quant_idx].item() - indicator)

            # Quantile Huber loss
            quantile_loss = tau_diff * huber

            total_loss += quantile_loss
            num_samples += 1

    expected_loss = total_loss / num_samples

    assert math.isclose(
        loss.item(), expected_loss, rel_tol=1e-5
    ), f"Expected {expected_loss:.8f}, got {loss.item():.8f}"


if __name__ == "__main__":
    # Allow running tests directly for quick verification
    print("Run with pytest for full test suite:")
    print("  pytest tests/test_quantile_huber_deep.py -v")
