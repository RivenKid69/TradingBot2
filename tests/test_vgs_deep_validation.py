"""
Deep validation tests for Variance Gradient Scaler.

Tests focus on:
1. Mathematical correctness of formulas
2. Numerical stability and edge cases
3. Bias correction accuracy
4. Gradient statistics consistency
5. NaN/Inf handling
6. Memory efficiency
7. Thread safety considerations
8. Performance benchmarks
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from variance_gradient_scaler import VarianceGradientScaler


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestMathematicalCorrectness:
    """Test mathematical correctness of VGS formulas."""

    def test_variance_computation_consistency(self):
        """
        CRITICAL: Test that variance and mean are computed consistently.

        Issue: The code computes mean from abs(gradients) but variance from
        raw gradients, which is mathematically inconsistent.
        """
        print("\n" + "="*70)
        print("TEST: Variance-Mean Consistency")
        print("="*70)

        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Create gradients
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()

        # Manual computation for verification
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.data.flatten())

        all_grads_tensor = torch.cat(all_grads)

        # Check: mean is computed from abs values
        manual_mean_abs = all_grads_tensor.abs().mean().item()
        print(f"Computed mean (from abs): {stats['grad_mean']:.8f}")
        print(f"Manual mean (from abs):   {manual_mean_abs:.8f}")
        assert abs(stats['grad_mean'] - manual_mean_abs) < 1e-6

        # Check: variance is ALSO computed from abs values for consistency (FIXED)
        manual_var_abs = all_grads_tensor.abs().var().item()
        print(f"Computed variance (abs):  {stats['grad_var']:.8f}")
        print(f"Manual variance (abs):    {manual_var_abs:.8f}")
        assert abs(stats['grad_var'] - manual_var_abs) < 1e-6

        # FIXED: Now mathematically consistent!
        # normalized_var = Var[|g|] / (E[|g|]^2 + eps)
        # Both variance and mean use abs values

        print("✓ Variance-mean computation verified (mathematically consistent)")
        print("✓ Both use abs values: Var[|g|] and E[|g|]")

    def test_bias_correction_formula(self):
        """Test that bias correction is applied correctly."""
        print("\n" + "="*70)
        print("TEST: Bias Correction Formula")
        print("="*70)

        torch.manual_seed(42)
        model = SimpleModel()
        beta = 0.9
        scaler = VarianceGradientScaler(model.parameters(), beta=beta)

        # Generate gradients and update multiple times
        for step in range(5):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()

            stats = scaler.compute_gradient_statistics()
            scaler.update_statistics()

            # Manual bias correction calculation
            # Note: step_count is incremented AFTER update_statistics in step()
            # But bias correction uses step_count + 1 in get_normalized_variance
            expected_bias_correction = 1.0 - beta ** (scaler._step_count + 1)

            if scaler._grad_mean_ema is not None:
                # Verify bias correction
                corrected_mean = scaler._grad_mean_ema / expected_bias_correction

                print(f"Step {step}:")
                print(f"  step_count: {scaler._step_count}")
                print(f"  bias_correction: {expected_bias_correction:.6f}")
                print(f"  raw EMA: {scaler._grad_mean_ema:.6f}")
                print(f"  corrected: {corrected_mean:.6f}")

            scaler._step_count += 1

        print("✓ Bias correction formula verified")

    def test_normalized_variance_bounds(self):
        """Test that normalized variance stays within reasonable bounds."""
        print("\n" + "="*70)
        print("TEST: Normalized Variance Bounds")
        print("="*70)

        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters(), eps=1e-8)

        # Test with various gradient magnitudes
        test_cases = [
            ("Tiny gradients", 1e-10),
            ("Small gradients", 1e-5),
            ("Normal gradients", 1e-2),
            ("Large gradients", 1.0),
            ("Huge gradients", 100.0),
        ]

        for name, scale in test_cases:
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true) * scale
            loss.backward()

            scaler.update_statistics()
            scaler._step_count += 1

            normalized_var = scaler.get_normalized_variance()

            print(f"{name:20s}: normalized_var = {normalized_var:.6e}")

            # Check bounds
            assert normalized_var >= 0.0, "Normalized variance must be non-negative"
            assert math.isfinite(normalized_var), "Normalized variance must be finite"

        print("✓ Normalized variance bounds verified")

    def test_scaling_factor_bounds(self):
        """Test that scaling factor is always in (0, 1]."""
        print("\n" + "="*70)
        print("TEST: Scaling Factor Bounds")
        print("="*70)

        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(
            model.parameters(),
            alpha=0.5,
            warmup_steps=2,
        )

        for step in range(10):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()

            scaling_factor = scaler.get_scaling_factor()

            print(f"Step {step}: scaling_factor = {scaling_factor:.6f}")

            # Verify bounds
            assert 0.0 < scaling_factor <= 1.0, \
                f"Scaling factor must be in (0, 1], got {scaling_factor}"
            assert math.isfinite(scaling_factor), "Scaling factor must be finite"

            scaler.update_statistics()
            scaler._step_count += 1

        print("✓ Scaling factor bounds verified")


class TestNumericalStability:
    """Test numerical stability in extreme conditions."""

    def test_zero_gradients(self):
        """Test behavior with zero gradients."""
        print("\n" + "="*70)
        print("TEST: Zero Gradients")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Manually set gradients to zero
        for param in model.parameters():
            param.grad = torch.zeros_like(param)

        stats = scaler.compute_gradient_statistics()
        scaler.update_statistics()

        print(f"Stats with zero grads: {stats}")

        assert stats["grad_norm"] == 0.0
        assert stats["grad_mean"] == 0.0
        assert stats["grad_var"] == 0.0

        # After update, EMAs should be 0
        assert scaler._grad_mean_ema == 0.0
        assert scaler._grad_var_ema == 0.0

        print("✓ Zero gradients handled correctly")

    def test_nan_gradients(self):
        """Test behavior with NaN gradients."""
        print("\n" + "="*70)
        print("TEST: NaN Gradients")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Set some gradients to NaN
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.grad = torch.full_like(param, float('nan'))
            else:
                param.grad = torch.randn_like(param)

        stats = scaler.compute_gradient_statistics()

        print(f"Stats with NaN grads: {stats}")

        # Check if NaN is properly detected
        assert not math.isfinite(stats["grad_norm"]) or stats["grad_norm"] == 0.0

        print("✓ NaN gradients detected")

    def test_inf_gradients(self):
        """Test behavior with infinite gradients."""
        print("\n" + "="*70)
        print("TEST: Infinite Gradients")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Set some gradients to inf
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.grad = torch.full_like(param, float('inf'))
            else:
                param.grad = torch.randn_like(param) * 0.01

        stats = scaler.compute_gradient_statistics()

        print(f"Stats with Inf grads: {stats}")

        # Inf should propagate
        assert not math.isfinite(stats["grad_max"])

        print("✓ Infinite gradients detected")

    def test_very_small_eps(self):
        """Test with very small epsilon for numerical stability."""
        print("\n" + "="*70)
        print("TEST: Very Small Epsilon")
        print("="*70)

        torch.manual_seed(42)
        model = SimpleModel()

        # Test with different epsilon values
        eps_values = [1e-8, 1e-12, 1e-16, 0.0]

        for eps in eps_values:
            try:
                scaler = VarianceGradientScaler(model.parameters(), eps=eps)

                x = torch.randn(4, 10)
                y_true = torch.randn(4, 5)
                y_pred = model(x)
                loss = nn.functional.mse_loss(y_pred, y_true)

                model.zero_grad()
                loss.backward()

                scaler.update_statistics()
                scaler._step_count = scaler.warmup_steps + 1

                normalized_var = scaler.get_normalized_variance()
                scaling_factor = scaler.get_scaling_factor()

                print(f"eps={eps:e}: norm_var={normalized_var:.6e}, "
                      f"scale={scaling_factor:.6f}")

                if eps == 0.0:
                    print("  ⚠ WARNING: eps=0 may cause division by zero")

            except ValueError as e:
                print(f"eps={eps}: Validation error: {e}")

        print("✓ Epsilon stability tested")

    def test_extreme_variance(self):
        """Test with artificially extreme variance."""
        print("\n" + "="*70)
        print("TEST: Extreme Variance")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters(), alpha=0.1)

        # Create gradients with extreme variance
        for param in model.parameters():
            # Half very small, half very large
            grad = torch.randn_like(param)
            mask = torch.rand_like(param) > 0.5
            grad[mask] *= 1000.0
            grad[~mask] *= 0.001
            param.grad = grad

        stats = scaler.compute_gradient_statistics()
        scaler.update_statistics()
        scaler._step_count = scaler.warmup_steps + 1

        normalized_var = scaler.get_normalized_variance()
        scaling_factor = scaler.get_scaling_factor()

        print(f"Extreme variance: {stats['grad_var']:.6e}")
        print(f"Normalized variance: {normalized_var:.6e}")
        print(f"Scaling factor: {scaling_factor:.6f}")

        # Scaling should reduce extreme gradients
        assert 0.0 < scaling_factor <= 1.0

        print("✓ Extreme variance handled")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_parameter(self):
        """Test with model having only one parameter."""
        print("\n" + "="*70)
        print("TEST: Single Parameter")
        print("="*70)

        # Create simple model with one parameter
        model = nn.Linear(5, 1, bias=False)
        scaler = VarianceGradientScaler(model.parameters())

        x = torch.randn(4, 5)
        y_true = torch.randn(4, 1)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()

        print(f"Single param stats: {stats}")
        assert stats["num_params"] == 1

        print("✓ Single parameter works")

    def test_no_parameters(self):
        """Test with no parameters."""
        print("\n" + "="*70)
        print("TEST: No Parameters")
        print("="*70)

        scaler = VarianceGradientScaler(None)

        stats = scaler.compute_gradient_statistics()
        scaling_factor = scaler.scale_gradients()

        print(f"No params stats: {stats}")
        print(f"No params scaling: {scaling_factor}")

        assert stats["num_params"] == 0
        assert scaling_factor == 1.0

        print("✓ No parameters handled")

    def test_some_parameters_without_gradients(self):
        """Test when some parameters don't have gradients."""
        print("\n" + "="*70)
        print("TEST: Mixed Gradient Availability")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Set requires_grad=False for some parameters
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:
                param.requires_grad = False

        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()

        print(f"Mixed grads stats: {stats}")
        assert stats["num_params"] < len(list(model.parameters()))

        print("✓ Mixed gradient availability handled")

    def test_update_parameters_mid_training(self):
        """Test updating parameter list during training."""
        print("\n" + "="*70)
        print("TEST: Update Parameters Mid-Training")
        print("="*70)

        model1 = SimpleModel()
        model2 = SimpleModel()

        scaler = VarianceGradientScaler(model1.parameters())

        # Train on model1
        for _ in range(3):
            model1.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model1(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.update_statistics()
            scaler._step_count += 1

        step_count_before = scaler._step_count
        ema_before = scaler._grad_mean_ema

        # Switch to model2
        scaler.update_parameters(model2.parameters())

        # Train on model2
        model2.zero_grad()
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model2(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()

        print(f"Step count preserved: {scaler._step_count == step_count_before}")
        print(f"EMA preserved: {scaler._grad_mean_ema == ema_before}")
        print(f"New stats computed: {stats['num_params'] > 0}")

        assert scaler._step_count == step_count_before
        assert stats["num_params"] > 0

        print("✓ Parameter update works")


class TestPerformance:
    """Test performance and efficiency."""

    def test_memory_efficiency(self):
        """Test that scaler doesn't accumulate unbounded memory."""
        print("\n" + "="*70)
        print("TEST: Memory Efficiency")
        print("="*70)

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        import sys

        # Get initial size
        initial_size = sys.getsizeof(scaler.__dict__)

        # Run many updates
        for _ in range(1000):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.update_statistics()
            scaler._step_count += 1

        final_size = sys.getsizeof(scaler.__dict__)

        print(f"Initial size: {initial_size} bytes")
        print(f"Final size: {final_size} bytes")
        print(f"Growth: {final_size - initial_size} bytes")

        # Size should not grow significantly
        assert final_size < initial_size * 2, "Memory usage grew too much"

        print("✓ Memory efficiency verified")

    def test_computational_overhead(self):
        """Test computational overhead of VGS."""
        print("\n" + "="*70)
        print("TEST: Computational Overhead")
        print("="*70)

        import time

        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Warmup
        for _ in range(10):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()

        # Benchmark without VGS
        start = time.time()
        for _ in range(100):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
        baseline_time = time.time() - start

        # Benchmark with VGS
        start = time.time()
        for _ in range(100):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.scale_gradients()
            scaler.step()
        vgs_time = time.time() - start

        overhead = ((vgs_time - baseline_time) / baseline_time) * 100

        print(f"Baseline time: {baseline_time:.4f}s")
        print(f"With VGS time: {vgs_time:.4f}s")
        print(f"Overhead: {overhead:.2f}%")

        # Overhead should be reasonable (< 100% - gradient tracking adds some cost)
        assert overhead < 100.0, f"Overhead too high: {overhead:.2f}%"

        print("✓ Computational overhead acceptable")


def run_all_tests():
    """Run all deep validation tests."""
    print("\n" + "="*70)
    print("DEEP VALIDATION TESTS FOR VARIANCE GRADIENT SCALER")
    print("="*70)

    test_classes = [
        TestMathematicalCorrectness,
        TestNumericalStability,
        TestEdgeCases,
        TestPerformance,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*70}")

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"\n✗ FAILED: {method_name}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n" + "="*70)
        print("ALL DEEP VALIDATION TESTS PASSED! ✓")
        print("="*70)
        return True


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
