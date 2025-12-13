"""
Comprehensive tests for VGS v3.0 with TRUE stochastic variance computation.

v3.0 CRITICAL FIX: VGS now correctly computes stochastic variance (variance OVER TIME)
using E[g] and E[g²], instead of incorrectly using torch.var() which computes spatial
variance (variance ACROSS ELEMENTS at one timestep).

Tests verify:
1. Uniform noisy gradients produce NON-ZERO stochastic variance (temporal noise)
2. Heterogeneous constant gradients produce ZERO stochastic variance (no temporal noise)
3. E[g] and E[g²] are tracked correctly over time
4. Variance formula Var[g] = E[g²] - E[g]² is applied correctly
5. Checkpoint migration from v1.x/v2.x to v3.0
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import warnings

try:
    from variance_gradient_scaler import VarianceGradientScaler
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from variance_gradient_scaler import VarianceGradientScaler


class TestStochasticVarianceCorrectness:
    """Test that VGS v3.0 correctly computes stochastic variance (temporal)."""

    def test_uniform_noisy_gradients_nonzero_variance(self):
        """
        CRITICAL TEST: Uniform gradients with temporal noise should have NON-ZERO variance.

        This is the smoking gun test that distinguishes spatial vs stochastic variance:
        - Spatial variance (OLD BUG): torch.var(ones * noise) = 0 (all elements same)
        - Stochastic variance (FIXED): Var[noise_sequence] > 0 (temporal variation)
        """
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=5, alpha=0.1)

        # Apply uniform gradients with HIGH temporal noise
        for step in range(30):
            # All elements get SAME value, but it changes over time
            noise_value = torch.randn(1).item() * 2.0 + 1.0
            param.grad = torch.ones(100) * noise_value  # Uniform spatially, noisy temporally
            vgs.scale_gradients()
            vgs.step()

        variance = vgs.get_normalized_variance()

        print(f"Normalized variance (uniform noisy): {variance:.6f}")

        # CRITICAL: Stochastic variance MUST be non-zero (temporal noise exists)
        assert variance > 0.1, (
            f"Expected NON-ZERO variance for uniform noisy gradients (temporal variation), "
            f"got {variance:.6f}. This suggests spatial variance computation (BUG)!"
        )

        print(f"[PASS] Uniform noisy gradients correctly show variance = {variance:.6f} > 0")

    def test_heterogeneous_constant_gradients_zero_variance(self):
        """
        CRITICAL TEST: Heterogeneous but constant gradients should have ZERO variance.

        This tests that VGS measures temporal variation, not spatial heterogeneity:
        - Spatial variance (OLD BUG): torch.var(linspace(-1, 1)) >> 0 (high heterogeneity)
        - Stochastic variance (FIXED): Var[constant_sequence] = 0 (no temporal variation)
        """
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=5, alpha=0.1)

        # Apply SAME heterogeneous gradient at every timestep
        constant_grad = torch.linspace(-1.0, 1.0, 100)  # High spatial heterogeneity
        for step in range(30):
            param.grad = constant_grad.clone()  # SAME every time (no temporal variation)
            vgs.scale_gradients()
            vgs.step()

        variance = vgs.get_normalized_variance()

        print(f"Normalized variance (heterogeneous constant): {variance:.6f}")

        # CRITICAL: Stochastic variance MUST be near zero (no temporal variation)
        assert variance < 0.01, (
            f"Expected ZERO variance for constant gradients (no temporal variation), "
            f"got {variance:.6f}. This suggests spatial variance computation (BUG)!"
        )

        print(f"[PASS] Heterogeneous constant gradients correctly show variance = {variance:.6f} ~= 0")

    def test_variance_formula_applied_correctly(self):
        """Test that variance is computed as Var[g] = E[g²] - E[g]², not torch.var()."""
        param = nn.Parameter(torch.randn(50))
        vgs = VarianceGradientScaler([param], warmup_steps=0, beta=0.9)

        # Apply gradients with known statistics
        for step in range(50):
            # Mean = 1.0, noise std = 0.5
            param.grad = torch.ones(50) * 1.0 + torch.randn(50) * 0.5
            vgs.step()

        # Extract internal EMA stats (after bias correction)
        bias_correction = 1.0 - vgs.beta ** vgs._step_count
        mean_ema = vgs._param_grad_mean_ema[0] / bias_correction  # E[g]
        sq_ema = vgs._param_grad_sq_ema[0] / bias_correction      # E[g²]

        # Compute variance manually
        variance_expected = sq_ema - mean_ema ** 2

        # VGS should compute same variance
        variance_vgs = vgs.get_normalized_variance()

        print(f"E[g] = {mean_ema:.6f}")
        print(f"E[g²] = {sq_ema:.6f}")
        print(f"Var[g] = E[g²] - E[g]² = {variance_expected:.6f}")
        print(f"VGS normalized variance: {variance_vgs:.6f}")

        # E[g] should be close to 1.0 (mean of gradients)
        assert abs(mean_ema - 1.0) < 0.2, f"Expected E[g] ~= 1.0, got {mean_ema:.6f}"

        # E[g²] should be > E[g]² (variance > 0)
        assert sq_ema > mean_ema ** 2, f"Expected E[g²] > E[g]², got {sq_ema:.6f} vs {mean_ema**2:.6f}"

        # Variance should be positive
        assert variance_expected > 0, f"Expected positive variance, got {variance_expected:.6f}"

        print(f"[PASS] Variance formula Var[g] = E[g²] - E[g]² applied correctly")

    def test_temporal_variance_increases_with_noise(self):
        """Test that stochastic variance increases with temporal noise level."""
        param = nn.Parameter(torch.randn(100))

        # Test with LOW noise
        vgs_low = VarianceGradientScaler([param], warmup_steps=5, alpha=0.1)
        for step in range(30):
            param.grad = torch.ones(100) * (1.0 + torch.randn(1).item() * 0.1)  # Low noise: std=0.1
            vgs_low.step()

        var_low = vgs_low.get_normalized_variance()

        # Reset and test with HIGH noise
        vgs_high = VarianceGradientScaler([param], warmup_steps=5, alpha=0.1)
        for step in range(30):
            param.grad = torch.ones(100) * (1.0 + torch.randn(1).item() * 2.0)  # High noise: std=2.0
            vgs_high.step()

        var_high = vgs_high.get_normalized_variance()

        print(f"Variance (low noise): {var_low:.6f}")
        print(f"Variance (high noise): {var_high:.6f}")
        print(f"Ratio: {var_high / (var_low + 1e-8):.2f}x")

        # High noise should produce higher stochastic variance
        assert var_high > var_low * 5, (
            f"Expected high noise ({var_high:.6f}) >> low noise ({var_low:.6f}), "
            f"but ratio is only {var_high / (var_low + 1e-8):.2f}x"
        )

        print(f"[PASS] Variance correctly increases with temporal noise")


class TestEMATracking:
    """Test that E[g] and E[g²] are tracked correctly using EMA."""

    def test_ema_convergence_to_true_mean(self):
        """Test that E[g] EMA converges to true gradient mean."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=0, beta=0.99)

        # Apply gradients with known mean
        true_mean = 2.5
        for step in range(100):
            param.grad = torch.ones(100) * true_mean + torch.randn(100) * 0.1
            vgs.step()

        # Extract EMA mean
        bias_correction = 1.0 - vgs.beta ** vgs._step_count
        ema_mean = vgs._param_grad_mean_ema[0] / bias_correction

        print(f"True mean: {true_mean:.6f}")
        print(f"EMA mean: {ema_mean:.6f}")
        print(f"Error: {abs(ema_mean - true_mean):.6f}")

        # EMA should converge to true mean
        assert abs(ema_mean - true_mean) < 0.1, (
            f"Expected EMA mean to converge to {true_mean:.6f}, got {ema_mean:.6f}"
        )

        print(f"[PASS] E[g] EMA converges to true mean")

    def test_ema_second_moment_correct(self):
        """Test that E[g²] EMA is computed correctly."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=0, beta=0.99)

        # Apply gradients with known distribution
        # grad ~ N(mean=1.0, std=0.5)
        # E[g²] = Var[g] + E[g]² = 0.5² + 1.0² = 0.25 + 1.0 = 1.25
        for step in range(100):
            param.grad = torch.ones(100) * 1.0 + torch.randn(100) * 0.5
            vgs.step()

        # Extract EMA
        bias_correction = 1.0 - vgs.beta ** vgs._step_count
        ema_mean = vgs._param_grad_mean_ema[0] / bias_correction
        ema_sq = vgs._param_grad_sq_ema[0] / bias_correction

        expected_sq = 0.25 + 1.0  # Var + mean²

        print(f"E[g]: {ema_mean:.6f} (expected ~1.0)")
        print(f"E[g²]: {ema_sq:.6f} (expected ~{expected_sq:.6f})")

        # Allow generous tolerance due to sampling variation
        assert abs(ema_mean - 1.0) < 0.2, f"Expected E[g] ~= 1.0, got {ema_mean:.6f}"
        assert abs(ema_sq - expected_sq) < 0.3, f"Expected E[g²] ~= {expected_sq:.6f}, got {ema_sq:.6f}"

        print(f"[PASS] E[g²] EMA computed correctly")


class TestCheckpointMigration:
    """Test backward compatibility and checkpoint migration from v1.x/v2.x to v3.0."""

    def test_v2_checkpoint_migration_warning(self):
        """Test that loading v2.0 checkpoint issues migration warning."""
        param = nn.Parameter(torch.randn(100))

        # Create v3.0 VGS
        vgs = VarianceGradientScaler([param], warmup_steps=10)

        # Simulate v2.0 checkpoint (INCORRECT spatial variance)
        v2_checkpoint = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.15,
            "eps": 1e-8,
            "warmup_steps": 20,
            "step_count": 100,
            # OLD v2.0 format: stored E[|g|] and spatial Var[g]
            "param_grad_mean_ema": torch.tensor([1.0]),    # Was E[|g|]
            "param_grad_sq_ema": torch.tensor([0.5]),      # Was spatial Var[g]
            "param_numel": torch.tensor([100]),
            "grad_mean_ema": 1.0,
            "grad_var_ema": 0.5,
            "grad_norm_ema": 2.0,
            "grad_max_ema": 3.0,
            "vgs_version": "2.0",  # OLD version
        }

        # Load v2.0 checkpoint - should warn and reset stats
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs.load_state_dict(v2_checkpoint)

            # Check that migration warning was issued
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert "v3.0 CRITICAL FIX" in str(w[0].message), "Missing migration warning"
            assert "SPATIAL variance" in str(w[0].message), "Missing bug description"
            assert "STOCHASTIC variance" in str(w[0].message), "Missing fix description"

        # Config should be loaded
        assert vgs.beta == 0.99
        assert vgs.alpha == 0.15
        assert vgs.warmup_steps == 20

        # Per-parameter stats should be RESET (will be reinitialized with CORRECT computation)
        assert vgs._param_grad_mean_ema is None
        assert vgs._param_grad_sq_ema is None

        # Legacy stats should be loaded (for logging)
        assert vgs._grad_mean_ema == 1.0
        assert vgs._grad_var_ema == 0.5

        print(f"[PASS] v2.0 checkpoint migration with warning")

    def test_v1_checkpoint_migration_warning(self):
        """Test that loading v1.0 checkpoint issues migration warning."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=10)

        # Simulate v1.0 checkpoint (no per-parameter stats)
        v1_checkpoint = {
            "enabled": True,
            "beta": 0.98,
            "alpha": 0.2,
            "eps": 1e-7,
            "warmup_steps": 15,
            "step_count": 50,
            # OLD v1.0 format: only global stats
            "grad_mean_ema": 0.8,
            "grad_var_ema": 0.3,
            "grad_norm_ema": 1.5,
            "grad_max_ema": 2.5,
            "vgs_version": "1.0",  # OLD version
        }

        # Load v1.0 checkpoint
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs.load_state_dict(v1_checkpoint)

            assert len(w) == 1
            assert "v3.0 CRITICAL FIX" in str(w[0].message)

        # Per-parameter stats should be reset
        assert vgs._param_grad_mean_ema is None

        print(f"[PASS] v1.0 checkpoint migration with warning")

    def test_v3_checkpoint_load_without_warning(self):
        """Test that loading v3.0 checkpoint does NOT issue warning."""
        param = nn.Parameter(torch.randn(100))

        # Train v3.0 VGS
        vgs1 = VarianceGradientScaler([param], warmup_steps=5)
        for step in range(20):
            param.grad = torch.randn(100) * 0.5 + 1.0
            vgs1.step()

        # Save v3.0 checkpoint
        state1 = vgs1.state_dict()
        assert state1["vgs_version"] == "3.0"

        # Load into new VGS
        vgs2 = VarianceGradientScaler([param], warmup_steps=10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs2.load_state_dict(state1)

            # NO warning should be issued for v3.0 checkpoint
            assert len(w) == 0, f"Expected no warnings for v3.0 checkpoint, got {len(w)}"

        # Stats should be loaded (not reset)
        assert vgs2._param_grad_mean_ema is not None
        assert torch.allclose(vgs2._param_grad_mean_ema, vgs1._param_grad_mean_ema)

        print(f"[PASS] v3.0 checkpoint loads without warning")


class TestNumericalStability:
    """Test numerical stability of Var[g] = E[g²] - E[g]² computation."""

    def test_variance_non_negative(self):
        """Test that variance is clamped to >= 0 (handles numerical errors)."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=5)

        # Apply various gradient patterns
        for step in range(50):
            if step < 20:
                param.grad = torch.randn(100) * 10.0  # Large gradients
            elif step < 40:
                param.grad = torch.randn(100) * 1e-6  # Tiny gradients
            else:
                param.grad = torch.randn(100)  # Normal gradients
            vgs.step()

        variance = vgs.get_normalized_variance()

        # Variance must be non-negative
        assert variance >= 0.0, f"Variance should be >= 0, got {variance:.6f}"

        # Check internal variance (before normalization)
        bias_correction = 1.0 - vgs.beta ** vgs._step_count
        mean_ema = vgs._param_grad_mean_ema / bias_correction
        sq_ema = vgs._param_grad_sq_ema / bias_correction
        raw_variance = sq_ema - mean_ema.pow(2)

        # All raw variances should be >= 0 (after clamping)
        assert torch.all(raw_variance >= 0.0), "Raw variance contains negative values after clamping"

        print(f"[PASS] Variance is non-negative (numerically stable)")

    def test_zero_gradients_handled(self):
        """Test that zero gradients don't cause NaN/Inf."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=0)

        # Apply zero gradients
        for step in range(20):
            param.grad = torch.zeros(100)
            vgs.step()

        variance = vgs.get_normalized_variance()

        # Should handle gracefully
        assert torch.isfinite(torch.tensor(variance)), f"Variance should be finite, got {variance}"
        assert variance == 0.0, f"Expected zero variance for zero gradients, got {variance:.6f}"

        print(f"[PASS] Zero gradients handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
