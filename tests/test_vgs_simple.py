"""
Simple standalone test for VarianceGradientScaler without external dependencies.
"""

import pytest
import sys

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, "/home/user/ai-quant-platform")
from variance_gradient_scaler import VarianceGradientScaler


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_vgs_initialization():
    """Test VGS initialization."""
    print("Test 1: VGS Initialization...")
    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters())

    assert scaler.enabled is True
    assert scaler.beta == 0.99
    assert scaler.alpha == 0.1
    assert scaler._step_count == 0
    print("✓ Initialization test passed")


def test_vgs_gradient_statistics():
    """Test gradient statistics computation."""
    print("\nTest 2: Gradient Statistics...")
    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters())

    # Create gradients
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    stats = scaler.compute_gradient_statistics()

    assert stats["grad_norm"] > 0.0
    assert stats["grad_mean"] > 0.0
    assert stats["num_params"] > 0
    print(f"✓ Gradient statistics test passed")
    print(f"  - Grad norm: {stats['grad_norm']:.6f}")
    print(f"  - Grad mean: {stats['grad_mean']:.6f}")


def test_vgs_ema_updates():
    """Test EMA updates."""
    print("\nTest 3: EMA Updates...")
    torch.manual_seed(42)
    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters(), beta=0.9)

    # First update
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    scaler.update_statistics()
    assert scaler._grad_mean_ema is not None
    first_mean = scaler._grad_mean_ema

    # Second update
    model.zero_grad()
    x2 = torch.randn(4, 10)
    y_true2 = torch.randn(4, 5)
    y_pred2 = model(x2)
    loss2 = nn.functional.mse_loss(y_pred2, y_true2)
    loss2.backward()

    scaler.update_statistics()
    second_mean = scaler._grad_mean_ema

    assert second_mean != first_mean
    print("✓ EMA updates test passed")
    print(f"  - First EMA: {first_mean:.6f}")
    print(f"  - Second EMA: {second_mean:.6f}")


def test_vgs_scaling():
    """Test gradient scaling."""
    print("\nTest 4: Gradient Scaling...")
    torch.manual_seed(42)
    model = SimpleModel()
    scaler = VarianceGradientScaler(
        model.parameters(),
        warmup_steps=0,  # No warmup
        alpha=0.5,
    )

    # Build statistics
    for _ in range(10):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        scaler.update_statistics()
        scaler._step_count += 1

    # New gradients
    model.zero_grad()
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    # Save original gradients
    original_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

    # Apply scaling
    scaling_factor = scaler.scale_gradients()

    # Check scaling was applied
    new_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

    print("✓ Gradient scaling test passed")
    print(f"  - Scaling factor: {scaling_factor:.6f}")
    print(f"  - Original grad norm: {sum(n**2 for n in original_norms)**0.5:.6f}")
    print(f"  - Scaled grad norm: {sum(n**2 for n in new_norms)**0.5:.6f}")

    assert 0.0 < scaling_factor <= 1.0


def test_vgs_warmup():
    """Test warmup behavior."""
    print("\nTest 5: Warmup Behavior...")
    model = SimpleModel()
    warmup_steps = 10
    scaler = VarianceGradientScaler(model.parameters(), warmup_steps=warmup_steps)

    # During warmup, scaling should be 1.0
    for step in range(warmup_steps):
        scaler._step_count = step
        scaling_factor = scaler.get_scaling_factor()
        assert scaling_factor == 1.0

    print("✓ Warmup test passed")


def test_vgs_state_persistence():
    """Test state dict save/load."""
    print("\nTest 6: State Persistence...")
    model = SimpleModel()
    scaler1 = VarianceGradientScaler(model.parameters(), beta=0.95, alpha=0.3)

    # Build some state
    for _ in range(5):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        scaler1.update_statistics()
        scaler1._step_count += 1

    state = scaler1.state_dict()

    # Load into new scaler
    scaler2 = VarianceGradientScaler(model.parameters())
    scaler2.load_state_dict(state)

    assert scaler2.beta == scaler1.beta
    assert scaler2.alpha == scaler1.alpha
    assert scaler2._step_count == scaler1._step_count
    assert scaler2._grad_mean_ema == scaler1._grad_mean_ema

    print("✓ State persistence test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Variance Gradient Scaler Tests")
    print("=" * 60)

    try:
        test_vgs_initialization()
        test_vgs_gradient_statistics()
        test_vgs_ema_updates()
        test_vgs_scaling()
        test_vgs_warmup()
        test_vgs_state_persistence()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
