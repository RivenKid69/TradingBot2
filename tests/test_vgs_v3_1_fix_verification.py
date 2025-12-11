"""
Regression tests for VGS v3.1 critical fix (E[g^2] computation).

CRITICAL FIX in v3.1 (2025-11-23):
Previous versions (v1.x-v3.0) INCORRECTLY computed E[(E[g])^2] (square of mean)
instead of E[g^2] (mean of squares). This caused variance to be UNDERESTIMATED
by factor of N (parameter size), making VGS ineffective for large parameters!

v3.1 now CORRECTLY computes E[g^2] = mean(g^2).

These tests ensure the bug does NOT regress.
"""

import pytest
torch = pytest.importorskip("torch")
import pytest
import numpy as np
from variance_gradient_scaler import VarianceGradientScaler


class TestVGSv31Fix:
    """Test VGS v3.1 critical fix for E[g^2] computation."""

    def test_mean_of_squares_not_square_of_mean(self):
        """REGRESSION TEST: Ensure we compute mean(g^2), NOT (mean(g))^2."""
        # Create a simple parameter
        param = torch.nn.Parameter(torch.randn(100))

        # Create VGS
        vgs = VarianceGradientScaler(
            parameters=[param],
            enabled=True,
            beta=0.9,
            warmup_steps=0
        )

        # Set a gradient with known properties
        # Gradient: [2.0, -2.0, 2.0, -2.0, ...]
        # mean(grad) = 0.0 (balanced positive/negative)
        # mean(grad^2) = 4.0 (all squares are 4.0)
        grad = torch.zeros(100)
        grad[::2] = 2.0   # Even indices
        grad[1::2] = -2.0  # Odd indices
        param.grad = grad

        # Compute what SHOULD be tracked (v3.1 CORRECT)
        expected_grad_mean = grad.mean().item()  # Should be ~0.0
        expected_grad_sq = (grad ** 2).mean().item()  # Should be 4.0

        # Compute what was WRONGLY tracked (v3.0 BUG)
        wrong_grad_sq = expected_grad_mean ** 2  # Would be ~0.0 (BUG!)

        # Update statistics
        vgs.update_statistics()

        # Access internal state (after first update, before EMA kicks in fully)
        # Since beta=0.9, after first update:
        # ema = beta * 0 + (1-beta) * current = 0.1 * current
        actual_mean_ema = vgs._param_grad_mean_ema[0].item() / 0.1  # Undo EMA scaling
        actual_sq_ema = vgs._param_grad_sq_ema[0].item() / 0.1

        # CRITICAL ASSERTION: Should use mean(g^2), NOT (mean(g))^2
        assert abs(actual_sq_ema - expected_grad_sq) < 1e-4, \
            f"E[g^2] should be {expected_grad_sq} (mean of squares), " \
            f"got {actual_sq_ema}. BUG: using (mean(g))^2 = {wrong_grad_sq}?"

        assert abs(actual_sq_ema - wrong_grad_sq) > 1.0, \
            f"E[g^2] should NOT equal (mean(g))^2! Got {actual_sq_ema} ≈ {wrong_grad_sq}"

        print(f"[OK] V3.1 FIX VERIFIED:")
        print(f"  E[g] = {actual_mean_ema:.6f} (expected ~0.0)")
        print(f"  E[g^2] = {actual_sq_ema:.2f} (expected 4.0 - mean of squares)")
        print(f"  WRONG (v3.0): (E[g])^2 = {wrong_grad_sq:.6f} (square of mean)")
        print(f"  Difference: {abs(actual_sq_ema - wrong_grad_sq):.2f}")

    def test_variance_underestimation_fixed_for_large_params(self):
        """REGRESSION TEST: Variance should NOT be underestimated for large params."""
        # Large sparse parameter (common in deep learning)
        N = 10000
        param = torch.nn.Parameter(torch.zeros(N))

        vgs = VarianceGradientScaler(
            parameters=[param],
            enabled=True,
            beta=0.9,
            warmup_steps=0
        )

        # Sparse gradient: one large value, rest zero
        grad = torch.zeros(N)
        grad[0] = 10.0
        param.grad = grad

        # CORRECT (v3.1): E[g^2] = mean(g^2) = 100.0 / 10000 = 0.01
        expected_grad_sq_correct = (grad ** 2).mean().item()

        # WRONG (v3.0): E[g^2] = (mean(g))^2 = (0.001)^2 = 0.000001
        grad_mean = grad.mean().item()
        wrong_grad_sq = grad_mean ** 2

        # Ratio: correct / wrong = 0.01 / 0.000001 = 10,000
        expected_ratio = N

        # Update and check
        vgs.update_statistics()

        actual_sq_ema = vgs._param_grad_sq_ema[0].item() / 0.1  # Undo EMA

        # CRITICAL ASSERTION: Should NOT be underestimated by factor of N
        actual_ratio = actual_sq_ema / (wrong_grad_sq + 1e-20)

        assert abs(actual_sq_ema - expected_grad_sq_correct) < 1e-6, \
            f"E[g^2] should be {expected_grad_sq_correct}, got {actual_sq_ema}"

        assert actual_ratio > 100, \
            f"V3.1 fix should give variance {N}x larger than v3.0 bug! " \
            f"Got ratio {actual_ratio:.1f}, expected ~{expected_ratio}"

        print(f"[OK] LARGE PARAMETER FIX VERIFIED:")
        print(f"  Parameter size N = {N}")
        print(f"  E[g^2] (CORRECT v3.1) = {actual_sq_ema:.6f}")
        print(f"  (E[g])^2 (WRONG v3.0) = {wrong_grad_sq:.10f}")
        print(f"  Ratio (improvement) = {actual_ratio:.0f}x")
        print(f"  Expected ratio = {expected_ratio}x")

    def test_variance_tracking_over_time_correct(self):
        """REGRESSION TEST: Variance over time should use correct E[g^2]."""
        param = torch.nn.Parameter(torch.randn(100))

        vgs = VarianceGradientScaler(
            parameters=[param],
            enabled=True,
            beta=0.9,
            warmup_steps=0
        )

        # Simulate gradients over time with known statistics
        # All gradients have mean=1.0, but variance of elements varies
        torch.manual_seed(42)

        grad_means = []
        grad_sqs = []

        for step in range(50):
            # Heterogeneous gradient with mean ≈ 1.0
            grad = torch.randn(100) * 5.0 + 1.0  # High spatial variance
            # Ensure mean is exactly 1.0
            grad = grad - grad.mean() + 1.0

            param.grad = grad

            # Track what SHOULD be tracked
            grad_means.append(grad.mean().item())
            grad_sqs.append((grad ** 2).mean().item())  # CORRECT: mean of squares

            vgs.update_statistics()
            vgs._step_count += 1

        # Compute expected EMA values
        beta = 0.9
        expected_mean_ema = 0.0
        expected_sq_ema = 0.0

        for gm, gs in zip(grad_means, grad_sqs):
            expected_mean_ema = beta * expected_mean_ema + (1 - beta) * gm
            expected_sq_ema = beta * expected_sq_ema + (1 - beta) * gs

        # Check actual values
        actual_mean_ema = vgs._param_grad_mean_ema[0].item()
        actual_sq_ema = vgs._param_grad_sq_ema[0].item()

        # Should match
        assert abs(actual_mean_ema - expected_mean_ema) < 1e-4, \
            f"E[mean(g)] mismatch: expected {expected_mean_ema}, got {actual_mean_ema}"

        assert abs(actual_sq_ema - expected_sq_ema) < 1e-3, \
            f"E[mean(g^2)] mismatch: expected {expected_sq_ema}, got {actual_sq_ema}"

        # Compute variance
        bias_correction = 1.0 - beta ** 50
        variance = (actual_sq_ema / bias_correction) - (actual_mean_ema / bias_correction) ** 2

        # Variance should be positive and significant (heterogeneous gradients)
        assert variance > 1.0, \
            f"Variance should be significant for heterogeneous gradients, got {variance}"

        print(f"[OK] TEMPORAL VARIANCE CORRECT:")
        print(f"  E[mean(g)] = {actual_mean_ema/bias_correction:.4f}")
        print(f"  E[mean(g^2)] = {actual_sq_ema/bias_correction:.2f}")
        print(f"  Var[g] = E[mean(g^2)] - E[mean(g)]^2 = {variance:.2f}")

    def test_state_dict_version_3_1(self):
        """REGRESSION TEST: state_dict should have version 3.1."""
        param = torch.nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler(parameters=[param])

        state = vgs.state_dict()

        assert state["vgs_version"] == "3.1", \
            f"VGS version should be 3.1, got {state['vgs_version']}"

        print(f"[OK] STATE_DICT VERSION: {state['vgs_version']}")

    def test_migration_warning_from_v3_0(self):
        """REGRESSION TEST: Loading v3.0 checkpoint should warn about fix."""
        param = torch.nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler(parameters=[param])

        # Create fake v3.0 state dict
        old_state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "param_grad_mean_ema": torch.zeros(1),
            "param_grad_sq_ema": torch.zeros(1),  # Was WRONG in v3.0!
            "param_numel": torch.tensor([100.0]),
            "vgs_version": "3.0",  # Old version
        }

        # Should warn about migration
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs.load_state_dict(old_state)

            # Should have warning
            assert len(w) >= 1, "Should warn about v3.0 -> v3.1 migration"
            warning_text = str(w[0].message)
            assert "CRITICAL" in warning_text, "Warning should mention CRITICAL fix"
            assert "mean of squares" in warning_text or "square of mean" in warning_text, \
                "Warning should explain the square of mean vs mean of squares issue"

        # Statistics should be reset
        assert vgs._param_grad_mean_ema is None, "Should reset statistics on migration"
        assert vgs._param_grad_sq_ema is None, "Should reset statistics on migration"

        print(f"[OK] MIGRATION WARNING VERIFIED")
        print(f"  Warning message includes E[g^2] fix explanation")
        print(f"  Statistics correctly reset to None")

    def test_no_migration_warning_from_v3_1(self):
        """REGRESSION TEST: Loading v3.1 checkpoint should NOT warn."""
        param = torch.nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler(parameters=[param])

        # Create v3.1 state dict
        new_state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "param_grad_mean_ema": torch.randn(1),
            "param_grad_sq_ema": torch.randn(1).abs(),  # CORRECT in v3.1
            "param_numel": torch.tensor([100.0]),
            "vgs_version": "3.1",  # Current version
        }

        # Should NOT warn
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs.load_state_dict(new_state)

            # Should have NO warnings (or only unrelated ones)
            vgs_warnings = [warning for warning in w if "VGS" in str(warning.message)]
            assert len(vgs_warnings) == 0, \
                f"Should NOT warn when loading v3.1 checkpoint, got {len(vgs_warnings)} warnings"

        # Statistics should be preserved
        assert vgs._param_grad_mean_ema is not None, "Should preserve statistics"
        assert vgs._param_grad_sq_ema is not None, "Should preserve statistics"

        print(f"[OK] NO MIGRATION WARNING for v3.1 -> v3.1")

    def test_formula_correctness_mathematical(self):
        """MATHEMATICAL TEST: Verify Var[X] = E[X^2] - E[X]^2 formula."""
        # Known distribution
        torch.manual_seed(42)
        samples = torch.randn(10000)  # N(0, 1)

        # Method 1: Direct variance
        var_direct = samples.var().item()

        # Method 2: E[X^2] - E[X]^2 formula (what VGS should use)
        mean_x = samples.mean().item()
        mean_x_sq = (samples ** 2).mean().item()  # CORRECT: mean of squares
        var_formula = mean_x_sq - mean_x ** 2

        # Method 3: WRONG - (mean(X))^2 instead of mean(X^2)
        wrong_sq = mean_x ** 2

        # Verify formula is correct
        assert abs(var_direct - var_formula) < 0.01, \
            f"E[X^2] - E[X]^2 should equal torch.var(), got {var_formula} vs {var_direct}"

        # Verify we're NOT using the wrong formula
        assert abs(var_formula - wrong_sq) > 0.5, \
            f"E[X^2] should NOT equal E[X]^2, got {mean_x_sq} vs {wrong_sq}"

        print(f"[OK] MATHEMATICAL FORMULA VERIFIED:")
        print(f"  Var[X] (direct) = {var_direct:.6f}")
        print(f"  Var[X] (E[X^2] - E[X]^2) = {var_formula:.6f}")
        print(f"  E[X] = {mean_x:.6f}")
        print(f"  E[X^2] = {mean_x_sq:.6f}")
        print(f"  E[X]^2 (WRONG!) = {wrong_sq:.6f}")


def test_all_vgs_v31_regression_tests():
    """Run all V3.1 regression tests."""
    print("\n" + "=" * 80)
    print("VGS V3.1 REGRESSION TESTS - CRITICAL FIX VERIFICATION")
    print("=" * 80)

    test_class = TestVGSv31Fix()

    print("\n1. Testing mean of squares vs square of mean...")
    test_class.test_mean_of_squares_not_square_of_mean()

    print("\n2. Testing fix for large parameter underestimation...")
    test_class.test_variance_underestimation_fixed_for_large_params()

    print("\n3. Testing variance tracking over time...")
    test_class.test_variance_tracking_over_time_correct()

    print("\n4. Testing state_dict version...")
    test_class.test_state_dict_version_3_1()

    print("\n5. Testing migration warning from v3.0...")
    test_class.test_migration_warning_from_v3_0()

    print("\n6. Testing no migration warning for v3.1...")
    test_class.test_no_migration_warning_from_v3_1()

    print("\n7. Testing mathematical formula correctness...")
    test_class.test_formula_correctness_mathematical()

    print("\n" + "=" * 80)
    print("ALL V3.1 REGRESSION TESTS PASSED!")
    print("=" * 80)
    print("\nVERIFICATION SUMMARY:")
    print("  [OK] E[g^2] computed as mean(g^2), NOT (mean(g))^2")
    print("  [OK] Large parameter underestimation fixed (10,000x improvement)")
    print("  [OK] Temporal variance tracking uses correct formula")
    print("  [OK] State dict version = 3.1")
    print("  [OK] Migration from v3.0 triggers warning and resets statistics")
    print("  [OK] Loading v3.1 checkpoints works without warnings")
    print("  [OK] Mathematical formula Var[X] = E[X^2] - E[X]^2 verified")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_all_vgs_v31_regression_tests()
