"""
Quick test script for UPGD optimizers without pytest dependency.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from optimizers import UPGD, AdaptiveUPGD, UPGDW


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear(x)))


def test_upgd_basic():
    """Test basic UPGD functionality."""
    print("Testing UPGD basic functionality...")
    model = SimpleModel()
    optimizer = UPGD(model.parameters(), lr=0.01)

    # Generate loss and gradients
    x = torch.randn(8, 10)
    y = torch.randn(8, 2)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Optimization step
    optimizer.step()

    # Check parameters changed
    changed = False
    for p_init, p_curr in zip(initial_params, model.parameters()):
        if not torch.allclose(p_init, p_curr, atol=1e-6):
            changed = True
            break

    assert changed, "Parameters should have changed after optimization step"

    # Check state was created
    for p in model.parameters():
        assert p in optimizer.state, "Parameter should have optimizer state"
        state = optimizer.state[p]
        assert "step" in state
        assert "avg_utility" in state
        assert state["step"] == 1

    print("✓ UPGD basic test passed")


def test_adaptive_upgd():
    """Test AdaptiveUPGD functionality."""
    print("Testing AdaptiveUPGD...")
    model = SimpleModel()
    optimizer = AdaptiveUPGD(model.parameters(), lr=0.01)

    # Run optimization
    for _ in range(3):
        optimizer.zero_grad()
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Check state includes moments
    for p in model.parameters():
        if p in optimizer.state:
            state = optimizer.state[p]
            assert "first_moment" in state, "Should have first moment"
            assert "sec_moment" in state, "Should have second moment"
            assert "avg_utility" in state, "Should have utility tracking"

    print("✓ AdaptiveUPGD test passed")


def test_upgdw():
    """Test UPGDW functionality."""
    print("Testing UPGDW...")
    model = SimpleModel()
    optimizer = UPGDW(model.parameters(), lr=0.01, weight_decay=0.01)

    # Check initialization
    group = optimizer.param_groups[0]
    assert group["weight_decay"] == 0.01, "Weight decay should be set"
    assert "betas" in group, "Should have betas parameter"

    # Run optimization
    optimizer.zero_grad()
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Check state
    params_with_state = sum(1 for p in model.parameters() if p in optimizer.state)
    assert params_with_state > 0, "Should have created optimizer state"

    print("✓ UPGDW test passed")


def test_utility_computation():
    """Test utility computation."""
    print("Testing utility computation...")
    model = nn.Linear(3, 2, bias=False)
    optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0)  # No EMA

    # Set specific values
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    # Create known gradients
    model.zero_grad()
    model.weight.grad = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    optimizer.step()

    # Expected utility = -grad * param
    expected_utility = torch.tensor([[-0.1, -0.4, -0.9], [-1.6, -2.5, -3.6]])

    state = optimizer.state[model.weight]
    assert torch.allclose(
        state["avg_utility"], expected_utility, atol=1e-5
    ), "Utility should be -grad * param"

    print("✓ Utility computation test passed")


def test_weight_decay():
    """Test weight decay."""
    print("Testing weight decay...")
    model = nn.Linear(5, 3, bias=False)
    optimizer = UPGD(model.parameters(), lr=0.01, weight_decay=0.1)

    with torch.no_grad():
        model.weight.copy_(torch.ones_like(model.weight))

    # Small gradients
    model.zero_grad()
    model.weight.grad = torch.ones_like(model.weight) * 0.01

    optimizer.step()

    # Weight decay should have been applied
    # Parameters should have changed
    assert not torch.allclose(
        model.weight, torch.ones_like(model.weight)
    ), "Parameters should change with weight decay"

    print("✓ Weight decay test passed")


def test_state_persistence():
    """Test state persistence across steps."""
    print("Testing state persistence...")
    model = nn.Linear(3, 2, bias=False)
    optimizer = UPGD(model.parameters(), lr=0.01)

    steps = 5
    for i in range(1, steps + 1):
        model.zero_grad()
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()

        state = optimizer.state[model.weight]
        assert state["step"] == i, f"Step count should be {i}"

    # Utility should have been updated
    final_utility = optimizer.state[model.weight]["avg_utility"]
    assert not torch.allclose(
        final_utility, torch.zeros_like(final_utility)
    ), "Utility should have been updated"

    print("✓ State persistence test passed")


def test_numerical_stability():
    """Test numerical stability."""
    print("Testing numerical stability...")
    model = nn.Linear(3, 2, bias=False)
    optimizer = UPGD(model.parameters(), lr=0.001)

    # Very large gradients
    model.zero_grad()
    model.weight.grad = torch.ones_like(model.weight) * 1e6

    optimizer.step()

    # Should not produce NaN or Inf
    assert torch.all(
        torch.isfinite(model.weight)
    ), "Parameters should remain finite even with large gradients"

    print("✓ Numerical stability test passed")


def test_integration_training_loop():
    """Test in realistic training loop."""
    print("Testing integration with training loop...")
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    optimizer = UPGD(model.parameters(), lr=0.01)

    # Generate synthetic data
    X_train = torch.randn(100, 20)
    y_train = torch.randn(100, 10)

    initial_loss = None
    final_loss = None

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_train)
        loss = ((output - y_train) ** 2).mean()

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if epoch == 9:
            final_loss = loss.item()

    # Loss should decrease
    assert (
        final_loss < initial_loss
    ), f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    print(f"✓ Training loop test passed (loss: {initial_loss:.4f} -> {final_loss:.4f})")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running UPGD Optimizer Tests")
    print("=" * 60)

    tests = [
        test_upgd_basic,
        test_adaptive_upgd,
        test_upgdw,
        test_utility_computation,
        test_weight_decay,
        test_state_persistence,
        test_numerical_stability,
        test_integration_training_loop,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
