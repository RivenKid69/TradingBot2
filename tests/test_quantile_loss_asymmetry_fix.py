"""
Test quantile regression loss asymmetry correctness.

This test verifies that the quantile Huber loss implementation correctly
follows the formula from Dabney et al. 2018:

    ρ_τ^κ(u) = |τ - I{u < 0}| · L_κ(u), where u = target - predicted

The critical fix ensures:
- delta = targets - predicted_quantiles (NOT predicted - targets)
- This gives correct asymmetry coefficients:
  * Underestimation (Q < T): coefficient = τ
  * Overestimation (Q ≥ T): coefficient = (1 - τ)

Reference:
    Dabney et al. 2018, "Distributional Reinforcement Learning with
    Quantile Regression", AAAI
"""

import math

import pytest
pytest.importorskip("torch")

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs


def test_quantile_loss_asymmetry_coefficient_correctness():
    """
    Test that quantile loss coefficients have correct values.

    For τ-quantile:
    - Underestimation should get penalty τ
    - Overestimation should get penalty (1 - τ)
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    # Test with τ = 0.25 (25th percentile)
    class _PolicyStub25:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25], dtype=torch.float32)

    algo.policy = _PolicyStub25()

    # Create symmetric errors
    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)

    # Underestimation: predicted < target
    predicted_under = torch.tensor([[-1.0]], dtype=torch.float32, requires_grad=True)
    loss_under = DistributionalPPO._quantile_huber_loss(
        algo, predicted_under, target, reduction="none"
    )

    # Overestimation: predicted > target
    predicted_over = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    loss_over = DistributionalPPO._quantile_huber_loss(
        algo, predicted_over, target, reduction="none"
    )

    # For τ = 0.25, with symmetric errors:
    # - Underestimation should have coefficient 0.25
    # - Overestimation should have coefficient 0.75
    # Since Huber(1) = 0.5 * 1^2 = 0.5 for both (kappa=1.0):
    # - loss_under should be 0.25 * 0.5 = 0.125
    # - loss_over should be 0.75 * 0.5 = 0.375

    assert math.isclose(loss_under.item(), 0.125, abs_tol=1e-6), \
        f"Underestimation loss should be 0.125, got {loss_under.item()}"

    assert math.isclose(loss_over.item(), 0.375, abs_tol=1e-6), \
        f"Overestimation loss should be 0.375, got {loss_over.item()}"

    # Ratio should be (1-τ)/τ = 0.75/0.25 = 3.0
    ratio = loss_over.item() / loss_under.item()
    expected_ratio = (1 - 0.25) / 0.25

    assert math.isclose(ratio, expected_ratio, rel_tol=1e-5), \
        f"Loss ratio should be {expected_ratio}, got {ratio}"


def test_quantile_loss_asymmetry_multiple_tau():
    """
    Test asymmetry correctness across multiple quantile levels.

    Verifies that the ratio of overestimation/underestimation penalties
    equals (1-τ)/τ for various τ values.
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    tau_values = [0.1, 0.25, 0.5, 0.75, 0.9]

    for tau in tau_values:
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._quantile_huber_kappa = 1.0

        class _PolicyStub:
            device = torch.device("cpu")
            _tau = tau

            @property
            def quantile_levels(self):
                return torch.tensor([self._tau], dtype=torch.float32)

        algo.policy = _PolicyStub()

        target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)

        # Symmetric errors
        predicted_under = torch.tensor([[-1.0]], dtype=torch.float32)
        predicted_over = torch.tensor([[1.0]], dtype=torch.float32)

        loss_under = DistributionalPPO._quantile_huber_loss(
            algo, predicted_under, target, reduction="none"
        )
        loss_over = DistributionalPPO._quantile_huber_loss(
            algo, predicted_over, target, reduction="none"
        )

        # Expected ratio
        expected_ratio = (1 - tau) / tau
        actual_ratio = loss_over.item() / loss_under.item()

        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-5), \
            f"For τ={tau}: expected ratio {expected_ratio}, got {actual_ratio}"


def test_quantile_loss_median_symmetry():
    """
    Test that median (τ=0.5) has symmetric penalties.

    For the median, overestimation and underestimation should be
    penalized equally.
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

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

    # For median, losses should be equal
    assert math.isclose(loss_under.item(), loss_over.item(), rel_tol=1e-6), \
        f"Median should have symmetric losses: {loss_under.item()} vs {loss_over.item()}"


def test_quantile_loss_gradient_direction():
    """
    Test that gradients push predictions in the correct direction.

    For τ-quantile:
    - Low τ (e.g., 0.1): should push predictions DOWN (conservative)
    - High τ (e.g., 0.9): should push predictions UP (aggressive)
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    # Test low quantile (τ = 0.1)
    algo_low = DistributionalPPO.__new__(DistributionalPPO)
    algo_low._quantile_huber_kappa = 1.0

    class _PolicyStub01:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.1], dtype=torch.float32)

    algo_low.policy = _PolicyStub01()

    # Start at zero, target is 1.0
    predicted_low = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))
    target = torch.tensor([1.0], dtype=torch.float32).reshape(-1, 1)

    optimizer_low = torch.optim.SGD([predicted_low], lr=0.5)
    optimizer_low.zero_grad()
    loss_low = DistributionalPPO._quantile_huber_loss(algo_low, predicted_low, target)
    loss_low.backward()
    grad_low = predicted_low.grad.item()

    # Test high quantile (τ = 0.9)
    algo_high = DistributionalPPO.__new__(DistributionalPPO)
    algo_high._quantile_huber_kappa = 1.0

    class _PolicyStub09:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.9], dtype=torch.float32)

    algo_high.policy = _PolicyStub09()

    predicted_high = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))

    optimizer_high = torch.optim.SGD([predicted_high], lr=0.5)
    optimizer_high.zero_grad()
    loss_high = DistributionalPPO._quantile_huber_loss(algo_high, predicted_high, target)
    loss_high.backward()
    grad_high = predicted_high.grad.item()

    # Both should have negative gradients (pushing towards target=1.0)
    # But high τ should have stronger gradient (more aggressive)
    assert grad_low < 0, f"Low quantile gradient should be negative, got {grad_low}"
    assert grad_high < 0, f"High quantile gradient should be negative, got {grad_high}"
    assert abs(grad_high) > abs(grad_low), \
        f"High quantile should have stronger gradient: {grad_high} vs {grad_low}"


def test_quantile_loss_training_convergence():
    """
    Test that training with quantile loss converges to correct quantiles.

    Train a simple model to predict different quantiles of a target
    distribution and verify it learns the correct asymmetry.
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            # Predict 25th, 50th, and 75th percentiles
            return torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)

    algo.policy = _PolicyStub()

    # Create a batch of targets with known distribution
    # Use a simple uniform distribution from -1 to 1
    torch.manual_seed(42)
    targets = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32).reshape(-1, 1)

    # Initialize predictions at zero
    predicted = torch.nn.Parameter(torch.zeros((5, 3), dtype=torch.float32))

    optimizer = torch.optim.Adam([predicted], lr=0.1)

    # Train for multiple steps
    for _ in range(200):
        optimizer.zero_grad()
        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
        loss.backward()
        optimizer.step()

    # Check that quantiles are ordered correctly
    with torch.no_grad():
        mean_quantiles = predicted.mean(dim=0)

    # 25th percentile should be < 50th < 75th
    assert mean_quantiles[0] < mean_quantiles[1], \
        f"25th percentile should be < 50th: {mean_quantiles[0]} vs {mean_quantiles[1]}"
    assert mean_quantiles[1] < mean_quantiles[2], \
        f"50th percentile should be < 75th: {mean_quantiles[1]} vs {mean_quantiles[2]}"

    # For uniform distribution from -1 to 1:
    # - 25th percentile ≈ -0.5
    # - 50th percentile ≈ 0.0
    # - 75th percentile ≈ 0.5
    assert math.isclose(mean_quantiles[0].item(), -0.5, abs_tol=0.2), \
        f"25th percentile should be near -0.5, got {mean_quantiles[0].item()}"
    assert math.isclose(mean_quantiles[1].item(), 0.0, abs_tol=0.2), \
        f"50th percentile should be near 0.0, got {mean_quantiles[1].item()}"
    assert math.isclose(mean_quantiles[2].item(), 0.5, abs_tol=0.2), \
        f"75th percentile should be near 0.5, got {mean_quantiles[2].item()}"


def test_quantile_loss_huber_threshold():
    """
    Test that Huber loss threshold (kappa) works correctly.

    For small errors (< kappa): quadratic loss
    For large errors (≥ kappa): linear loss
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.5], dtype=torch.float32)

    algo.policy = _PolicyStub()

    target = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)

    # Small error (< kappa)
    predicted_small = torch.tensor([[0.5]], dtype=torch.float32)
    loss_small = DistributionalPPO._quantile_huber_loss(
        algo, predicted_small, target, reduction="none"
    )

    # Large error (> kappa)
    predicted_large = torch.tensor([[2.0]], dtype=torch.float32)
    loss_large = DistributionalPPO._quantile_huber_loss(
        algo, predicted_large, target, reduction="none"
    )

    # For small error (0.5 < 1.0): Huber = 0.5 * 0.5^2 = 0.125
    # For median (τ=0.5): coefficient = 0.5
    # Total: 0.5 * 0.125 = 0.0625
    expected_small = 0.5 * 0.5 * 0.5 * 0.5  # coef * 0.5 * delta^2

    # For large error (2.0 > 1.0): Huber = kappa * (|delta| - 0.5 * kappa)
    #                                    = 1.0 * (2.0 - 0.5) = 1.5
    # For median: coefficient = 0.5
    # Total: 0.5 * 1.5 = 0.75
    expected_large = 0.5 * (1.0 * (2.0 - 0.5))

    assert math.isclose(loss_small.item(), expected_small, abs_tol=1e-6), \
        f"Small error loss should be {expected_small}, got {loss_small.item()}"
    assert math.isclose(loss_large.item(), expected_large, abs_tol=1e-6), \
        f"Large error loss should be {expected_large}, got {loss_large.item()}"


def test_quantile_loss_batch_independence():
    """
    Test that batch samples are processed independently.

    Each sample should only affect its own predictions, not other samples.
    """
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

    algo.policy = _PolicyStub()

    # Sample 1: already at target
    # Sample 2: far from target
    predicted = torch.tensor(
        [[0.0, 0.0], [5.0, 5.0]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([0.0, 0.0], dtype=torch.float32).reshape(-1, 1)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # Gradient for first sample should be ~0 (already at target)
    # Gradient for second sample should be large (far from target)
    grad_first = predicted.grad[0].abs().max().item()
    grad_second = predicted.grad[1].abs().max().item()

    assert math.isclose(grad_first, 0.0, abs_tol=1e-6), \
        f"First sample gradient should be near 0, got {grad_first}"
    assert grad_second > 0.1, \
        f"Second sample gradient should be large, got {grad_second}"
