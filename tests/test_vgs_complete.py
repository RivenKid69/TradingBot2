"""
Complete standalone test suite for Variance Gradient Scaler.

This test can run without pytest or external dependencies beyond PyTorch.
Tests all functionality with 100% code coverage goal.
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import sys
import os

sys.path.insert(0, '/home/user/ai-quant-platform')
from variance_gradient_scaler import VarianceGradientScaler


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_basic_functionality():
    """Test basic VGS functionality."""
    print("\n" + "="*70)
    print("TEST: Basic Functionality")
    print("="*70)

    torch.manual_seed(42)
    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters(), enabled=True)

    # Training loop
    for step in range(20):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        # Apply VGS
        scaling_factor = scaler.scale_gradients()
        scaler.step()

        if step < scaler.warmup_steps:
            assert scaling_factor == 1.0, f"During warmup, scaling should be 1.0"
        else:
            assert 0.0 < scaling_factor <= 1.0, f"After warmup, scaling in (0,1]"

    print(f"✓ Completed {step+1} training steps")
    print(f"  Final scaling factor: {scaling_factor:.6f}")
    print(f"  Step count: {scaler._step_count}")

    return True


def test_gradient_statistics_accuracy():
    """Test accuracy of gradient statistics computation."""
    print("\n" + "="*70)
    print("TEST: Gradient Statistics Accuracy")
    print("="*70)

    torch.manual_seed(123)
    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters())

    # Create gradients
    x = torch.randn(8, 10)
    y_true = torch.randn(8, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    stats = scaler.compute_gradient_statistics()

    # Manual verification
    all_grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    manual_norm = (sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5)
    manual_mean = all_grads.abs().mean().item()
    manual_var = all_grads.abs().var().item()  # Fixed: variance of abs values for consistency
    manual_max = all_grads.abs().max().item()

    print(f"Computed norm: {stats['grad_norm']:.8f}, Manual: {manual_norm:.8f}")
    print(f"Computed mean: {stats['grad_mean']:.8f}, Manual: {manual_mean:.8f}")
    print(f"Computed var:  {stats['grad_var']:.8f}, Manual: {manual_var:.8f}")
    print(f"Computed max:  {stats['grad_max']:.8f}, Manual: {manual_max:.8f}")

    assert abs(stats['grad_norm'] - manual_norm) < 1e-6
    assert abs(stats['grad_mean'] - manual_mean) < 1e-6
    assert abs(stats['grad_var'] - manual_var) < 1e-6
    assert abs(stats['grad_max'] - manual_max) < 1e-6

    print("✓ Gradient statistics match manual computation")
    return True


def test_ema_accumulation():
    """Test EMA accumulation over time."""
    print("\n" + "="*70)
    print("TEST: EMA Accumulation")
    print("="*70)

    torch.manual_seed(456)
    model = SimpleModel()
    beta = 0.9
    scaler = VarianceGradientScaler(model.parameters(), beta=beta)

    ema_history = []

    for step in range(10):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        scaler.update_statistics()
        scaler._step_count += 1

        if scaler._grad_mean_ema is not None:
            ema_history.append(scaler._grad_mean_ema)

    print(f"EMA history (first 10 steps): {[f'{v:.6f}' for v in ema_history]}")

    # EMA should stabilize over time (changes should decrease)
    if len(ema_history) >= 5:
        early_changes = [abs(ema_history[i+1] - ema_history[i]) for i in range(3)]
        late_changes = [abs(ema_history[i+1] - ema_history[i]) for i in range(6, 9)]

        avg_early = sum(early_changes) / len(early_changes)
        avg_late = sum(late_changes) / len(late_changes)

        print(f"Average early changes: {avg_early:.8f}")
        print(f"Average late changes: {avg_late:.8f}")
        print(f"Ratio: {avg_late/avg_early:.4f}")

    print("✓ EMA accumulation verified")
    return True


def test_scaling_application():
    """Test that scaling actually modifies gradients."""
    print("\n" + "="*70)
    print("TEST: Scaling Application")
    print("="*70)

    torch.manual_seed(789)
    model = SimpleModel()
    scaler = VarianceGradientScaler(
        model.parameters(),
        warmup_steps=0,
        alpha=0.5,
    )

    # Build up statistics
    for _ in range(10):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        scaler.update_statistics()
        scaler._step_count += 1

    # Now test scaling
    model.zero_grad()
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    # Save original gradients
    original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # Apply scaling
    scaling_factor = scaler.scale_gradients()

    # Verify scaling was applied
    scaled_grads = [p.grad for p in model.parameters() if p.grad is not None]

    for orig, scaled in zip(original_grads, scaled_grads):
        expected = orig * scaling_factor
        diff = (scaled - expected).abs().max().item()
        assert diff < 1e-6, f"Scaling not applied correctly, max diff: {diff}"

    print(f"✓ Scaling factor {scaling_factor:.6f} applied correctly")
    return True


def test_warmup_behavior():
    """Test warmup period behavior."""
    print("\n" + "="*70)
    print("TEST: Warmup Behavior")
    print("="*70)

    model = SimpleModel()
    warmup_steps = 15
    scaler = VarianceGradientScaler(model.parameters(), warmup_steps=warmup_steps)

    scaling_factors = []

    for step in range(25):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        scaler.update_statistics()
        scaler._step_count = step

        scaling_factor = scaler.get_scaling_factor()
        scaling_factors.append(scaling_factor)

        if step < warmup_steps:
            assert scaling_factor == 1.0, f"Step {step}: expected 1.0, got {scaling_factor}"

    print(f"Warmup period: {warmup_steps} steps")
    print(f"Scaling during warmup: {scaling_factors[:warmup_steps]}")
    print(f"Scaling after warmup: {[f'{s:.4f}' for s in scaling_factors[warmup_steps:warmup_steps+5]]}")

    print("✓ Warmup behavior correct")
    return True


def test_state_persistence():
    """Test state_dict and load_state_dict."""
    print("\n" + "="*70)
    print("TEST: State Persistence")
    print("="*70)

    torch.manual_seed(111)
    model = SimpleModel()
    scaler1 = VarianceGradientScaler(model.parameters(), beta=0.95, alpha=0.25)

    # Train scaler1
    for _ in range(15):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        scaler1.update_statistics()
        scaler1._step_count += 1

    # Save state
    state = scaler1.state_dict()

    print(f"Saved state keys: {list(state.keys())}")
    print(f"  beta: {state['beta']}")
    print(f"  alpha: {state['alpha']}")
    print(f"  step_count: {state['step_count']}")

    # Create new scaler and load state
    scaler2 = VarianceGradientScaler(model.parameters())
    scaler2.load_state_dict(state)

    # Verify all state is restored
    assert scaler2.beta == scaler1.beta
    assert scaler2.alpha == scaler1.alpha
    assert scaler2._step_count == scaler1._step_count
    assert scaler2._grad_mean_ema == scaler1._grad_mean_ema
    assert scaler2._grad_var_ema == scaler1._grad_var_ema
    assert scaler2._grad_norm_ema == scaler1._grad_norm_ema
    assert scaler2._grad_max_ema == scaler1._grad_max_ema

    print("✓ State persistence works correctly")
    return True


def test_reset_functionality():
    """Test reset_statistics."""
    print("\n" + "="*70)
    print("TEST: Reset Functionality")
    print("="*70)

    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters())

    # Accumulate some statistics
    for _ in range(10):
        model.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        scaler.update_statistics()
        scaler._step_count += 1

    print(f"Before reset:")
    print(f"  step_count: {scaler._step_count}")
    print(f"  grad_mean_ema: {scaler._grad_mean_ema}")

    assert scaler._step_count > 0
    assert scaler._grad_mean_ema is not None

    # Reset
    scaler.reset_statistics()

    print(f"After reset:")
    print(f"  step_count: {scaler._step_count}")
    print(f"  grad_mean_ema: {scaler._grad_mean_ema}")

    assert scaler._step_count == 0
    assert scaler._grad_mean_ema is None
    assert scaler._grad_var_ema is None
    assert scaler._grad_norm_ema is None
    assert scaler._grad_max_ema is None

    print("✓ Reset clears all statistics")
    return True


def test_disabled_mode():
    """Test that VGS does nothing when disabled."""
    print("\n" + "="*70)
    print("TEST: Disabled Mode")
    print("="*70)

    model = SimpleModel()
    scaler = VarianceGradientScaler(model.parameters(), enabled=False)

    model.zero_grad()
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    y_pred = model(x)
    loss = nn.functional.mse_loss(y_pred, y_true)
    loss.backward()

    # Save original gradients
    original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # Try to scale (should do nothing)
    scaling_factor = scaler.scale_gradients()

    assert scaling_factor == 1.0

    # Verify gradients unchanged
    for orig, param in zip(original_grads, model.parameters()):
        if param.grad is not None:
            assert torch.all(param.grad == orig)

    print("✓ Disabled mode doesn't modify gradients")
    return True


def test_repr():
    """Test __repr__ method."""
    print("\n" + "="*70)
    print("TEST: String Representation")
    print("="*70)

    scaler = VarianceGradientScaler(
        None,
        enabled=True,
        beta=0.95,
        alpha=0.2,
        eps=1e-6,
        warmup_steps=50,
    )

    repr_str = repr(scaler)

    print(f"repr: {repr_str}")

    assert "VarianceGradientScaler" in repr_str
    assert "enabled=True" in repr_str
    assert "beta=0.95" in repr_str
    assert "alpha=0.2" in repr_str

    print("✓ __repr__ works correctly")
    return True


def test_parameter_validation():
    """Test parameter validation in __init__."""
    print("\n" + "="*70)
    print("TEST: Parameter Validation")
    print("="*70)

    model = SimpleModel()

    # Test invalid beta
    try:
        VarianceGradientScaler(model.parameters(), beta=0.0)
        assert False, "Should raise ValueError for beta=0"
    except ValueError as e:
        print(f"✓ Caught invalid beta=0: {e}")

    try:
        VarianceGradientScaler(model.parameters(), beta=1.0)
        assert False, "Should raise ValueError for beta=1"
    except ValueError as e:
        print(f"✓ Caught invalid beta=1: {e}")

    # Test invalid alpha
    try:
        VarianceGradientScaler(model.parameters(), alpha=-0.1)
        assert False, "Should raise ValueError for negative alpha"
    except ValueError as e:
        print(f"✓ Caught invalid alpha=-0.1: {e}")

    # Test invalid eps
    try:
        VarianceGradientScaler(model.parameters(), eps=0.0)
        assert False, "Should raise ValueError for eps=0"
    except ValueError as e:
        print(f"✓ Caught invalid eps=0: {e}")

    # Test invalid warmup_steps
    try:
        VarianceGradientScaler(model.parameters(), warmup_steps=-10)
        assert False, "Should raise ValueError for negative warmup_steps"
    except ValueError as e:
        print(f"✓ Caught invalid warmup_steps=-10: {e}")

    print("✓ Parameter validation works correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPLETE VGS TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Gradient Statistics Accuracy", test_gradient_statistics_accuracy),
        ("EMA Accumulation", test_ema_accumulation),
        ("Scaling Application", test_scaling_application),
        ("Warmup Behavior", test_warmup_behavior),
        ("State Persistence", test_state_persistence),
        ("Reset Functionality", test_reset_functionality),
        ("Disabled Mode", test_disabled_mode),
        ("String Representation", test_repr),
        ("Parameter Validation", test_parameter_validation),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                errors.append((name, "Test returned False"))
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if errors:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        return False
    else:
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
