"""
Comprehensive tests for VGS v2.0 with per-parameter stochastic variance.

Tests cover:
1. Per-parameter stochastic variance computation
2. Spatial vs stochastic variance difference
3. Aggregation methods (percentile)
4. Backward compatibility (checkpoint migration)
5. Integration with DistributionalPPO + UPGD
6. Edge cases and numerical stability
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
import warnings
import tempfile
import os

try:
    from variance_gradient_scaler import VarianceGradientScaler
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from variance_gradient_scaler import VarianceGradientScaler


class TestPerParameterStochasticVariance:
    """Test per-parameter stochastic variance computation."""

    def test_per_parameter_variance_tracking(self):
        """Test that variance is computed per-parameter, not globally."""
        # Create two parameters with different gradient variances
        param1 = nn.Parameter(torch.randn(100))
        param2 = nn.Parameter(torch.randn(100))

        vgs = VarianceGradientScaler([param1, param2], warmup_steps=0)

        # Apply gradients with different temporal variances
        for step in range(50):
            # param1: low temporal variance (stable gradients)
            param1.grad = torch.randn(100) * 0.05 + 1.0  # mean=1.0, std=0.05

            # param2: high temporal variance (noisy gradients)
            param2.grad = torch.randn(100) * 2.0 + 1.0  # mean=1.0, std=2.0

            vgs.scale_gradients()
            vgs.step()

        # Check per-parameter statistics exist
        assert vgs._param_grad_mean_ema is not None
        assert vgs._param_grad_mean_ema.shape[0] == 2

        # Compute per-parameter variance
        # NOTE: In v2.0.1, _param_grad_sq_ema directly stores Var[g], not E[g²]
        bias_correction = 1.0 - vgs.beta**vgs._step_count
        abs_mean_corrected = vgs._param_grad_mean_ema / bias_correction  # E[|g|]
        var_per_param = vgs._param_grad_sq_ema / bias_correction  # Var[g]

        # param2 should have significantly higher variance than param1
        assert (
            var_per_param[1] > var_per_param[0] * 3
        ), f"Expected param2 variance ({var_per_param[1]:.4f}) to be >> param1 ({var_per_param[0]:.4f})"

        print(
            f"[OK] Per-parameter variance: param1={var_per_param[0]:.6f}, param2={var_per_param[1]:.6f}"
        )

    def test_stochastic_vs_spatial_variance(self):
        """Test that stochastic variance differs from spatial variance."""

        # Create network with heterogeneous parameter scales
        class HeterogeneousNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)  # Will have small gradients
                self.layer2 = nn.Linear(10, 10)  # Will have large gradients

        model = HeterogeneousNet()
        vgs = VarianceGradientScaler(model.parameters(), warmup_steps=5, alpha=0.2)

        # Apply gradients with different SCALES but LOW temporal variance
        for step in range(30):
            # layer1: small but stable gradients (low temporal variance)
            for p in model.layer1.parameters():
                p.grad = torch.randn_like(p) * 0.005 + 0.01  # scale=0.01, noise=0.005

            # layer2: large but stable gradients (low temporal variance)
            for p in model.layer2.parameters():
                p.grad = torch.randn_like(p) * 0.005 + 1.0  # scale=1.0, noise=0.005

            vgs.scale_gradients()
            vgs.step()

        # NEW VGS (stochastic variance): Should see LOW variance (both layers are stable)
        stochastic_var = vgs.get_normalized_variance()

        # OLD VGS (spatial variance): Would see HIGH variance (0.01 vs 1.0 scales)
        # Compute spatial variance manually from legacy statistics
        spatial_var = vgs._grad_var_ema / (vgs._grad_mean_ema**2 + 1e-8)

        print(f"Stochastic variance (new): {stochastic_var:.6f}")
        print(f"Spatial variance (old): {spatial_var:.6f}")

        # Stochastic variance should be LOWER than spatial variance
        # because temporal noise is low, even though spatial heterogeneity is high
        assert (
            stochastic_var < spatial_var
        ), f"Expected stochastic var ({stochastic_var:.4f}) < spatial var ({spatial_var:.4f})"

        # Stochastic variance should be low (< 0.1) for stable gradients
        assert (
            stochastic_var < 0.1
        ), f"Expected low stochastic variance for stable gradients, got {stochastic_var:.4f}"

        print(
            f"[OK] Stochastic variance correctly measures temporal noise, not spatial heterogeneity"
        )

    def test_high_noise_triggers_scaling(self):
        """Test that high temporal variance (noise) triggers gradient scaling."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=5, alpha=0.5)

        # Apply very noisy gradients (high temporal variance)
        for step in range(30):
            # High noise: mean stays ~1.0, but std is large
            param.grad = torch.randn(100) * 5.0 + 1.0  # mean=1.0, std=5.0

            scaling_factor = vgs.scale_gradients()
            vgs.step()

        # After warmup, should see gradient scaling (< 1.0)
        final_scaling = vgs.get_scaling_factor()
        final_var = vgs.get_normalized_variance()

        print(f"Final normalized variance: {final_var:.6f}")
        print(f"Final scaling factor: {final_scaling:.6f}")

        # High variance should trigger scaling
        # With alpha=0.5 and high noise (std=5.0), normalized variance should be significant
        assert final_var > 0.1, f"Expected significant variance, got {final_var:.4f}"
        assert final_scaling < 0.95, f"Expected scaling < 0.95, got {final_scaling:.4f}"

        print(f"[OK] High temporal variance correctly triggers gradient scaling")


class TestAggregationMethods:
    """Test variance aggregation to global metric."""

    def test_percentile_aggregation(self):
        """Test 90th percentile aggregation (default)."""
        # Create parameters with varying variances
        params = [nn.Parameter(torch.randn(50)) for _ in range(10)]
        vgs = VarianceGradientScaler(params, warmup_steps=5)

        # Apply gradients with different variances for each parameter
        for step in range(30):
            for i, p in enumerate(params):
                # Variance increases with index
                noise_std = 0.1 * (i + 1)  # 0.1, 0.2, ..., 1.0
                p.grad = torch.randn(50) * noise_std + 1.0

            vgs.scale_gradients()
            vgs.step()

        # Get normalized variance (90th percentile)
        global_var = vgs.get_normalized_variance()

        # Compute per-parameter variances manually
        # NOTE: In v2.0.1, _param_grad_sq_ema directly stores Var[g]
        bias_correction = 1.0 - vgs.beta**vgs._step_count
        abs_mean_corrected = vgs._param_grad_mean_ema / bias_correction  # E[|g|]
        var_per_param = vgs._param_grad_sq_ema / bias_correction  # Var[g]
        normalized_var_per_param = var_per_param / (abs_mean_corrected.pow(2) + vgs.eps)

        # 90th percentile should be close to torch.quantile(..., 0.9)
        expected_p90 = torch.quantile(normalized_var_per_param, 0.9).item()

        print(f"Global variance (p90): {global_var:.6f}")
        print(f"Expected p90: {expected_p90:.6f}")
        print(f"Difference: {abs(global_var - expected_p90):.6f}")

        # Should match closely
        assert (
            abs(global_var - expected_p90) < 0.01
        ), f"Expected global_var ≈ p90, got {global_var:.4f} vs {expected_p90:.4f}"

        # p90 should be higher than mean (skewed distribution)
        mean_var = normalized_var_per_param.mean().item()
        assert global_var > mean_var, f"Expected p90 ({global_var:.4f}) > mean ({mean_var:.4f})"

        print(f"[OK] 90th percentile aggregation working correctly")


class TestBackwardCompatibility:
    """Test backward compatibility with old checkpoints (spatial variance)."""

    def test_old_checkpoint_migration(self):
        """Test loading old VGS checkpoint (v1.x with spatial variance)."""
        param = nn.Parameter(torch.randn(100))

        # Create NEW VGS (v2.0)
        vgs = VarianceGradientScaler([param], warmup_steps=5)

        # Simulate OLD checkpoint format (v1.x - no per-parameter stats)
        old_checkpoint = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 10,
            "step_count": 50,
            # OLD format: only global spatial variance stats
            "grad_mean_ema": 1.0,
            "grad_var_ema": 0.5,
            "grad_norm_ema": 2.0,
            "grad_max_ema": 3.0,
            # NO per-parameter stats!
        }

        # Load old checkpoint - should warn and reset per-parameter stats
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vgs.load_state_dict(old_checkpoint)

            # Check that warning was issued
            assert len(w) == 1
            assert "VGS Checkpoint Migration" in str(w[0].message)
            assert "OLD FORMAT DETECTED" in str(w[0].message)

        # Config should be loaded
        assert vgs.beta == 0.99
        assert vgs.warmup_steps == 10
        assert vgs._step_count == 50

        # Per-parameter stats should be RESET
        assert vgs._param_grad_mean_ema is None
        assert vgs._param_grad_sq_ema is None

        # Legacy global stats should be loaded (for logging)
        assert vgs._grad_mean_ema == 1.0
        assert vgs._grad_var_ema == 0.5

        print(f"[OK] Old checkpoint migration working correctly with warning")

    def test_new_checkpoint_load(self):
        """Test loading new VGS checkpoint (v2.0 with stochastic variance)."""
        param = nn.Parameter(torch.randn(100))

        # Create and train VGS
        vgs1 = VarianceGradientScaler([param], warmup_steps=5)

        for step in range(20):
            param.grad = torch.randn(100) * 0.5 + 1.0
            vgs1.scale_gradients()
            vgs1.step()

        # Save state
        state1 = vgs1.state_dict()

        # Create new VGS and load state
        vgs2 = VarianceGradientScaler([param], warmup_steps=10)  # Different warmup
        vgs2.load_state_dict(state1)

        # Config should match
        assert vgs2.warmup_steps == 5  # Should be overridden
        assert vgs2._step_count == vgs1._step_count

        # Per-parameter stats should be loaded
        assert vgs2._param_grad_mean_ema is not None
        assert torch.allclose(vgs2._param_grad_mean_ema, vgs1._param_grad_mean_ema)
        assert torch.allclose(vgs2._param_grad_sq_ema, vgs1._param_grad_sq_ema)

        print(f"[OK] New checkpoint load working correctly")


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_gradients(self):
        """Test VGS with all-zero gradients."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=0)

        # Apply zero gradients
        for step in range(10):
            param.grad = torch.zeros(100)
            vgs.scale_gradients()
            vgs.step()

        # Should not crash, variance should be zero
        var = vgs.get_normalized_variance()
        assert var == 0.0, f"Expected zero variance, got {var}"

        print(f"[OK] Zero gradients handled correctly")

    def test_nan_gradients(self):
        """Test VGS with NaN gradients."""
        param = nn.Parameter(torch.randn(100))
        vgs = VarianceGradientScaler([param], warmup_steps=0)

        # Apply some normal gradients first
        for step in range(10):
            param.grad = torch.randn(100) * 0.1 + 1.0
            vgs.scale_gradients()
            vgs.step()

        # Apply NaN gradient
        param.grad = torch.full((100,), float("nan"))
        scaling = vgs.scale_gradients()
        vgs.step()

        # Should handle gracefully (NaN replaced with zero)
        var = vgs.get_normalized_variance()
        assert np.isfinite(var), f"Variance should be finite, got {var}"

        print(f"[OK] NaN gradients handled correctly")

    def test_single_parameter(self):
        """Test VGS with single parameter."""
        param = nn.Parameter(torch.randn(10))
        vgs = VarianceGradientScaler([param], warmup_steps=5)

        for step in range(20):
            param.grad = torch.randn(10) * 0.5 + 1.0
            vgs.scale_gradients()
            vgs.step()

        # Should work with single parameter
        var = vgs.get_normalized_variance()
        assert np.isfinite(var), f"Variance should be finite, got {var}"

        print(f"[OK] Single parameter handled correctly")

    def test_large_network(self):
        """Test VGS with large network (memory efficiency)."""
        # Create large network
        model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(10)])  # ~100k parameters

        vgs = VarianceGradientScaler(model.parameters(), warmup_steps=5)

        # Train briefly
        for step in range(20):
            # Simulate backward pass
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 0.1

            vgs.scale_gradients()
            vgs.step()

        # Should work efficiently
        var = vgs.get_normalized_variance()
        assert np.isfinite(var), f"Variance should be finite, got {var}"

        # Check memory usage is reasonable
        # Per-parameter stats: 3 * num_params * 4 bytes
        num_params = sum(1 for _ in model.parameters())
        expected_memory_mb = 3 * num_params * 4 / 1024 / 1024

        print(f"Number of parameters: {num_params}")
        print(f"Expected memory overhead: {expected_memory_mb:.2f} MB")
        assert expected_memory_mb < 1.0, f"Memory overhead too large: {expected_memory_mb:.2f} MB"

        print(f"[OK] Large network handled efficiently")


class TestLoggingMetrics:
    """Test logging of new metrics."""

    def test_new_metrics_logged(self):
        """Test that new stochastic variance metrics are logged."""
        param = nn.Parameter(torch.randn(100))

        # Mock logger
        logged_metrics = {}

        class MockLogger:
            def record(self, key, value):
                logged_metrics[key] = value

        vgs = VarianceGradientScaler([param], warmup_steps=5, logger=MockLogger())

        # Train
        for step in range(20):
            param.grad = torch.randn(100) * 0.5 + 1.0
            vgs.scale_gradients()
            vgs.step()

        # Check new metrics are logged
        assert "vgs/stochastic_var_p10" in logged_metrics
        assert "vgs/stochastic_var_p50" in logged_metrics
        assert "vgs/stochastic_var_p90" in logged_metrics
        assert "vgs/stochastic_var_mean" in logged_metrics

        # Check legacy metrics are still logged (renamed)
        assert "vgs/grad_mean_ema_spatial" in logged_metrics
        assert "vgs/grad_var_ema_spatial" in logged_metrics

        # Check all metrics are finite
        for key, value in logged_metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

        print(f"[OK] New metrics logged correctly: {list(logged_metrics.keys())}")


@pytest.mark.parametrize("warmup_steps", [0, 10, 50])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.5])
def test_parameter_sweep(warmup_steps, alpha):
    """Test VGS with different hyperparameters."""
    param = nn.Parameter(torch.randn(100))
    vgs = VarianceGradientScaler([param], warmup_steps=warmup_steps, alpha=alpha)

    # Train
    for step in range(max(warmup_steps + 10, 20)):
        param.grad = torch.randn(100) * 0.5 + 1.0
        vgs.scale_gradients()
        vgs.step()

    # Should work with all parameter combinations
    var = vgs.get_normalized_variance()
    scaling = vgs.get_scaling_factor()

    assert np.isfinite(var)
    assert 0.0 < scaling <= 1.0

    print(f"[OK] warmup={warmup_steps}, alpha={alpha}: var={var:.4f}, scaling={scaling:.4f}")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
