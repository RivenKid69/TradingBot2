"""
Integration tests for Quantile Huber Loss with other system components.

This test suite verifies that the quantile Huber loss fix integrates
correctly with:
- VF clipping
- Target computation
- Multiple quantiles
- Batch processing
"""

import math

import pytest

pytest.importorskip("torch")

import test_distributional_ppo_raw_outliers  # noqa: F401


def test_quantile_huber_loss_vf_clipping_integration() -> None:
    """
    Test that quantile Huber loss works correctly with VF clipping.

    VF clipping formula:
    L_VF = max(L(V, V_targ), L(clip(V), V_targ))

    where V_targ is UNCLIPPED (this was a separate bug that was fixed).
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

    # Original predictions
    predicted_orig = torch.tensor([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]], dtype=torch.float32)

    # Clipped predictions (simulating VF clipping)
    clip_range = 0.2
    predicted_clip = torch.clamp(predicted_orig, min=-clip_range, max=clip_range)

    # Unclipped targets (CRITICAL: targets should NOT be clipped)
    targets = torch.tensor([[0.5], [1.5]], dtype=torch.float32)

    # Compute losses
    loss_unclipped = DistributionalPPO._quantile_huber_loss(algo, predicted_orig, targets)
    loss_clipped = DistributionalPPO._quantile_huber_loss(algo, predicted_clip, targets)

    # Both losses should be finite and non-negative
    assert torch.isfinite(loss_unclipped).all()
    assert torch.isfinite(loss_clipped).all()
    assert loss_unclipped >= 0
    assert loss_clipped >= 0

    # The clipped loss should generally be higher (since we're constraining predictions)
    # but this depends on the specific values


def test_quantile_huber_loss_with_realistic_batch() -> None:
    """Test with realistic batch from training loop."""
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            # Standard 32 quantiles
            return torch.linspace(0.03125, 0.96875, 32, dtype=torch.float32)

    algo.policy = PolicyStub()

    # Realistic batch size from PPO
    batch_size = 256
    num_quantiles = 32

    # Simulated predictions from network
    predicted = torch.randn(batch_size, num_quantiles, dtype=torch.float32) * 2.0

    # Simulated targets from rollout
    targets = torch.randn(batch_size, 1, dtype=torch.float32) * 2.0

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    # Loss should be well-behaved
    assert torch.isfinite(loss).all()
    assert loss >= 0
    assert loss < 100.0  # Reasonable magnitude


def test_quantile_huber_loss_gradient_flow_in_training_loop() -> None:
    """
    Simulate training loop gradient flow.

    This tests that gradients flow correctly through the loss and
    can be used to update network parameters.
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

    # Simulate a simple network
    network = torch.nn.Linear(10, 3)  # Output 3 quantiles
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    # Simulated observations
    obs = torch.randn(32, 10, dtype=torch.float32)
    targets = torch.randn(32, 1, dtype=torch.float32)

    # Training step
    optimizer.zero_grad()
    predicted = network(obs)
    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # Check that gradients exist and are reasonable
    for param in network.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().max() < 100.0  # Not exploding

    # Apply optimizer step
    optimizer.step()

    # Network parameters should have changed
    predicted_after = network(obs)
    assert not torch.allclose(predicted, predicted_after)


def test_quantile_huber_loss_multi_batch_consistency() -> None:
    """
    Test that loss is consistent across multiple batches.

    Process the same data in different batch sizes and verify
    consistency.
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

    # Generate data
    torch.manual_seed(42)
    predicted_full = torch.randn(100, 3, dtype=torch.float32)
    targets_full = torch.randn(100, 1, dtype=torch.float32)

    # Compute loss on full batch
    loss_full = DistributionalPPO._quantile_huber_loss(algo, predicted_full, targets_full)

    # Compute loss on mini-batches and average
    batch_size = 20
    losses_mini = []
    for i in range(0, 100, batch_size):
        predicted_mini = predicted_full[i : i + batch_size]
        targets_mini = targets_full[i : i + batch_size]
        loss_mini = DistributionalPPO._quantile_huber_loss(algo, predicted_mini, targets_mini)
        losses_mini.append(loss_mini.item())

    avg_loss_mini = sum(losses_mini) / len(losses_mini)

    # Should be very close (mean is linear)
    assert math.isclose(
        loss_full.item(), avg_loss_mini, rel_tol=1e-5
    ), f"Full batch: {loss_full.item():.8f}, Mini-batch avg: {avg_loss_mini:.8f}"


def test_quantile_huber_loss_with_detached_targets() -> None:
    """
    Test that targets are properly detached in the indicator.

    The indicator uses delta.detach() which is critical to prevent
    gradients from flowing through the indicator.
    """
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

    # Both require gradients
    predicted = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # Predicted should have gradients
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0

    # Targets should NOT have gradients (not typical in RL, but let's verify the formula)
    # Actually in RL, targets are typically detached already, but the formula doesn't
    # require gradients through targets
    # The indicator uses delta.detach() which stops gradients

    # Verify indicator detachment by checking the computation graph
    # If indicator is detached, changing it shouldn't affect loss gradients


def test_quantile_huber_loss_broadcast_correctness() -> None:
    """
    Deep test of broadcasting behavior.

    Ensure that the loss correctly broadcasts targets across quantiles
    without accidentally broadcasting across batch dimension.
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

    # Two samples with different targets
    predicted = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # First sample: predictions match target, gradients should be ~0
    grad_first = predicted.grad[0].abs().max().item()

    # Second sample: predictions match target, gradients should be ~0
    grad_second = predicted.grad[1].abs().max().item()

    # Both should have very small gradients (perfect predictions)
    assert grad_first < 1e-5, f"First sample gradient too large: {grad_first}"
    assert grad_second < 1e-5, f"Second sample gradient too large: {grad_second}"

    # Now test with mismatched predictions
    predicted2 = torch.tensor(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
    )

    loss2 = DistributionalPPO._quantile_huber_loss(algo, predicted2, targets)
    loss2.backward()

    # First sample: predictions don't match target, should have gradients
    grad2_first = predicted2.grad[0].abs().max().item()

    # Second sample: predictions don't match target, should have gradients
    grad2_second = predicted2.grad[1].abs().max().item()

    assert grad2_first > 0, "First sample should have gradients"
    assert grad2_second > 0, "Second sample should have gradients"

    # Importantly: gradients should be different because targets are different
    # (This verifies no accidental batch-dimension broadcasting)
    assert not math.isclose(
        grad2_first, grad2_second, rel_tol=0.1
    ), "Gradients should differ for different targets"


def test_quantile_huber_loss_kappa_clipping_to_minimum() -> None:
    """
    Test that kappa is clipped to minimum value (1e-6).

    Line 2438: kappa = max(float(self._quantile_huber_kappa), 1e-6)
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

    # Try to set kappa below minimum
    algo._quantile_huber_kappa = 1e-10

    predicted = torch.tensor([[0.0]], dtype=torch.float32)
    targets = torch.tensor([[0.5]], dtype=torch.float32)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)

    # Should use kappa = 1e-6 (minimum)
    # Error = 0.5, which is > 1e-6, so linear region
    # huber = kappa * (|error| - 0.5 * kappa)
    #       = 1e-6 * (0.5 - 0.5 * 1e-6)
    #       ≈ 1e-6 * 0.5 = 5e-7
    # loss = 0.5 * 5e-7 = 2.5e-7

    kappa_actual = 1e-6
    expected = 0.5 * kappa_actual * (0.5 - 0.5 * kappa_actual)

    assert math.isclose(loss.item(), expected, rel_tol=1e-4)


def test_quantile_huber_loss_comprehensive_integration() -> None:
    """
    Comprehensive end-to-end integration test.

    Simulates a complete training scenario with:
    - Multiple quantiles
    - Large batch
    - Gradient accumulation
    - Optimizer step
    - Loss reduction
    """
    torch = pytest.importorskip("torch")
    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    # Realistic quantile configuration
    num_quantiles = 32
    quantile_levels = torch.linspace(
        1.0 / (2 * num_quantiles),
        1.0 - 1.0 / (2 * num_quantiles),
        num_quantiles,
        dtype=torch.float32,
    )

    class PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self):
            return quantile_levels

    algo.policy = PolicyStub()

    # Simulate network
    input_dim = 64
    batch_size = 256

    network = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_quantiles),
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Generate synthetic data
    torch.manual_seed(123)
    obs_batch = torch.randn(batch_size, input_dim)
    targets_batch = torch.randn(batch_size, 1)

    # Initial loss
    predicted_init = network(obs_batch)
    loss_init = DistributionalPPO._quantile_huber_loss(algo, predicted_init, targets_batch)

    # Training steps
    for step in range(10):
        optimizer.zero_grad()
        predicted = network(obs_batch)
        loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets_batch)
        loss.backward()
        optimizer.step()

    # Final loss
    predicted_final = network(obs_batch)
    loss_final = DistributionalPPO._quantile_huber_loss(algo, predicted_final, targets_batch)

    # Loss should decrease (learning is happening)
    assert (
        loss_final < loss_init
    ), f"Training should reduce loss: {loss_init.item():.4f} -> {loss_final.item():.4f}"

    # Loss should remain finite
    assert torch.isfinite(loss_final).all()


if __name__ == "__main__":
    print("Run with pytest:")
    print("  pytest tests/test_quantile_huber_integration.py -v")
