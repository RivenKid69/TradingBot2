"""
Unit tests for VarianceGradientScaler.

Tests cover:
- Initialization and parameter validation
- Gradient statistics computation
- Variance normalization
- Scaling factor calculation
- Gradient scaling application
- State persistence (state_dict/load_state_dict)
- EMA updates
- Warmup behavior
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn


try:
    from variance_gradient_scaler import VarianceGradientScaler
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from variance_gradient_scaler import VarianceGradientScaler


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestVarianceGradientScalerInit:
    """Test initialization and parameter validation."""

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        assert scaler.enabled is True
        assert scaler.beta == 0.99
        assert scaler.alpha == 0.1
        assert scaler.eps == 1e-8
        assert scaler.warmup_steps == 100
        assert scaler._step_count == 0

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(
            model.parameters(),
            enabled=False,
            beta=0.95,
            alpha=0.2,
            eps=1e-6,
            warmup_steps=50,
        )

        assert scaler.enabled is False
        assert scaler.beta == 0.95
        assert scaler.alpha == 0.2
        assert scaler.eps == 1e-6
        assert scaler.warmup_steps == 50

    def test_init_invalid_beta(self) -> None:
        """Test that invalid beta values raise ValueError."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="beta must be in"):
            VarianceGradientScaler(model.parameters(), beta=0.0)

        with pytest.raises(ValueError, match="beta must be in"):
            VarianceGradientScaler(model.parameters(), beta=1.0)

        with pytest.raises(ValueError, match="beta must be in"):
            VarianceGradientScaler(model.parameters(), beta=-0.5)

    def test_init_invalid_alpha(self) -> None:
        """Test that invalid alpha values raise ValueError."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="alpha must be non-negative"):
            VarianceGradientScaler(model.parameters(), alpha=-0.1)

    def test_init_invalid_eps(self) -> None:
        """Test that invalid eps values raise ValueError."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="eps must be positive"):
            VarianceGradientScaler(model.parameters(), eps=0.0)

        with pytest.raises(ValueError, match="eps must be positive"):
            VarianceGradientScaler(model.parameters(), eps=-1e-8)

    def test_init_invalid_warmup_steps(self) -> None:
        """Test that invalid warmup_steps values raise ValueError."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            VarianceGradientScaler(model.parameters(), warmup_steps=-10)


class TestVarianceGradientScalerStatistics:
    """Test gradient statistics computation."""

    def test_compute_gradient_statistics_no_gradients(self) -> None:
        """Test statistics computation when no gradients are present."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        stats = scaler.compute_gradient_statistics()

        assert stats["grad_norm"] == 0.0
        assert stats["grad_mean"] == 0.0
        assert stats["grad_var"] == 0.0
        assert stats["grad_max"] == 0.0
        assert stats["num_params"] == 0

    def test_compute_gradient_statistics_with_gradients(self) -> None:
        """Test statistics computation with actual gradients."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Create dummy input and compute gradients
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()

        assert stats["grad_norm"] > 0.0
        assert stats["grad_mean"] > 0.0
        assert stats["grad_var"] >= 0.0
        assert stats["grad_max"] > 0.0
        assert stats["num_params"] > 0

    def test_gradient_statistics_deterministic(self) -> None:
        """Test that statistics are deterministic for fixed gradients."""
        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats1 = scaler.compute_gradient_statistics()
        stats2 = scaler.compute_gradient_statistics()

        assert stats1["grad_norm"] == stats2["grad_norm"]
        assert stats1["grad_mean"] == stats2["grad_mean"]
        assert stats1["grad_var"] == stats2["grad_var"]
        assert stats1["grad_max"] == stats2["grad_max"]


class TestVarianceGradientScalerEMA:
    """Test EMA updates."""

    def test_ema_initialization(self) -> None:
        """Test that EMA is initialized on first update."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        assert scaler._grad_mean_ema is None
        assert scaler._grad_var_ema is None
        assert scaler._grad_norm_ema is None
        assert scaler._grad_max_ema is None

        # Compute gradients
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        scaler.update_statistics()

        assert scaler._grad_mean_ema is not None
        assert scaler._grad_var_ema is not None
        assert scaler._grad_norm_ema is not None
        assert scaler._grad_max_ema is not None
        assert scaler._grad_mean_ema > 0.0

    def test_ema_updates_correctly(self) -> None:
        """Test that EMA updates follow exponential moving average formula."""
        torch.manual_seed(42)
        model = SimpleModel()
        beta = 0.9
        scaler = VarianceGradientScaler(model.parameters(), beta=beta)

        # First update
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        scaler.update_statistics()
        first_mean = scaler._grad_mean_ema

        # Second update with new gradients
        model.zero_grad()
        x2 = torch.randn(4, 10)
        y_true2 = torch.randn(4, 5)
        y_pred2 = model(x2)
        loss2 = nn.functional.mse_loss(y_pred2, y_true2)
        loss2.backward()

        stats_before_update = scaler.compute_gradient_statistics()
        scaler.update_statistics()
        second_mean = scaler._grad_mean_ema

        # Verify EMA formula: new_ema = beta * old_ema + (1 - beta) * new_value
        expected_mean = beta * first_mean + (1 - beta) * stats_before_update["grad_mean"]
        assert abs(second_mean - expected_mean) < 1e-6


class TestVarianceGradientScalerNormalizedVariance:
    """Test normalized variance computation."""

    def test_normalized_variance_zero_when_no_statistics(self) -> None:
        """Test that normalized variance is 0 when no statistics are available."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        normalized_var = scaler.get_normalized_variance()
        assert normalized_var == 0.0

    def test_normalized_variance_positive_with_gradients(self) -> None:
        """Test that normalized variance is positive with actual gradients."""
        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Generate multiple gradient updates to accumulate statistics
        for _ in range(5):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.update_statistics()
            scaler._step_count += 1

        normalized_var = scaler.get_normalized_variance()
        assert normalized_var >= 0.0
        assert normalized_var < 1e10  # Reasonable range


class TestVarianceGradientScalerScaling:
    """Test gradient scaling."""

    def test_scaling_factor_one_during_warmup(self) -> None:
        """Test that scaling factor is 1.0 during warmup."""
        model = SimpleModel()
        warmup_steps = 10
        scaler = VarianceGradientScaler(model.parameters(), warmup_steps=warmup_steps)

        # Simulate warmup period
        for step in range(warmup_steps):
            scaler._step_count = step
            scaling_factor = scaler.get_scaling_factor()
            assert scaling_factor == 1.0

    def test_scaling_factor_less_than_one_after_warmup(self) -> None:
        """Test that scaling factor can be < 1.0 after warmup with high variance."""
        torch.manual_seed(42)
        model = SimpleModel()
        warmup_steps = 2
        alpha = 0.5  # Higher alpha for more aggressive scaling
        scaler = VarianceGradientScaler(
            model.parameters(),
            warmup_steps=warmup_steps,
            alpha=alpha,
        )

        # Generate gradients during warmup
        for _ in range(warmup_steps + 5):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.update_statistics()
            scaler._step_count += 1

        scaling_factor = scaler.get_scaling_factor()
        # With variance present, scaling factor should be <= 1.0
        assert 0.0 < scaling_factor <= 1.0

    def test_scale_gradients_modifies_gradients(self) -> None:
        """Test that scale_gradients actually modifies gradients."""
        torch.manual_seed(42)
        model = SimpleModel()
        scaler = VarianceGradientScaler(
            model.parameters(),
            warmup_steps=0,  # No warmup
            alpha=0.5,  # Significant scaling
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

        # New gradients
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

        # Verify gradients were scaled
        for original_grad, param in zip(original_grads, model.parameters()):
            if param.grad is not None:
                expected_grad = original_grad * scaling_factor
                torch.testing.assert_close(param.grad, expected_grad, rtol=1e-5, atol=1e-7)

    def test_scale_gradients_disabled(self) -> None:
        """Test that scale_gradients does nothing when disabled."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters(), enabled=False)

        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        scaling_factor = scaler.scale_gradients()

        assert scaling_factor == 1.0
        for original_grad, param in zip(original_grads, model.parameters()):
            if param.grad is not None:
                torch.testing.assert_close(param.grad, original_grad)


class TestVarianceGradientScalerStatePersistence:
    """Test state_dict and load_state_dict."""

    def test_state_dict_contains_expected_keys(self) -> None:
        """Test that state_dict contains all expected keys."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        state = scaler.state_dict()

        expected_keys = {
            "enabled",
            "beta",
            "alpha",
            "eps",
            "warmup_steps",
            "step_count",
            "grad_mean_ema",
            "grad_var_ema",
            "grad_norm_ema",
            "grad_max_ema",
        }
        assert set(state.keys()) == expected_keys

    def test_load_state_dict_restores_state(self) -> None:
        """Test that load_state_dict correctly restores state."""
        model = SimpleModel()
        scaler1 = VarianceGradientScaler(model.parameters(), beta=0.95, alpha=0.3)

        # Build up some state
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

        # Create new scaler and load state
        scaler2 = VarianceGradientScaler(model.parameters())
        scaler2.load_state_dict(state)

        assert scaler2.beta == scaler1.beta
        assert scaler2.alpha == scaler1.alpha
        assert scaler2._step_count == scaler1._step_count
        assert scaler2._grad_mean_ema == scaler1._grad_mean_ema
        assert scaler2._grad_var_ema == scaler1._grad_var_ema
        assert scaler2._grad_norm_ema == scaler1._grad_norm_ema
        assert scaler2._grad_max_ema == scaler1._grad_max_ema

    def test_reset_statistics_clears_state(self) -> None:
        """Test that reset_statistics clears all accumulated state."""
        model = SimpleModel()
        scaler = VarianceGradientScaler(model.parameters())

        # Build up state
        for _ in range(5):
            model.zero_grad()
            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            scaler.update_statistics()
            scaler._step_count += 1

        assert scaler._step_count > 0
        assert scaler._grad_mean_ema is not None

        scaler.reset_statistics()

        assert scaler._step_count == 0
        assert scaler._grad_mean_ema is None
        assert scaler._grad_var_ema is None
        assert scaler._grad_norm_ema is None
        assert scaler._grad_max_ema is None


class TestVarianceGradientScalerIntegration:
    """Integration tests simulating real training scenarios."""

    def test_full_training_loop(self) -> None:
        """Test VGS in a complete training loop."""
        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = VarianceGradientScaler(
            model.parameters(),
            enabled=True,
            warmup_steps=5,
            alpha=0.1,
        )

        # Training loop
        for step in range(20):
            optimizer.zero_grad()

            x = torch.randn(4, 10)
            y_true = torch.randn(4, 5)
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()

            # Apply VGS
            scaling_factor = scaler.scale_gradients()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scaler.step()

            # Verify state consistency
            assert scaler._step_count == step + 1
            if step >= scaler.warmup_steps:
                # After warmup, scaling may occur
                assert 0.0 < scaling_factor <= 1.0

    def test_update_parameters_after_init(self) -> None:
        """Test updating parameter list after initialization."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        scaler = VarianceGradientScaler(model1.parameters())

        # Update to new model
        scaler.update_parameters(model2.parameters())

        # Verify it works with new model
        x = torch.randn(4, 10)
        y_true = torch.randn(4, 5)
        y_pred = model2(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()

        stats = scaler.compute_gradient_statistics()
        assert stats["num_params"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
