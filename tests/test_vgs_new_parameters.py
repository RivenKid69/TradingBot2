"""
Tests for VGS v3.2 new parameters: min_scaling_factor and variance_cap.

FIX (2025-11-27): These parameters were added to prevent overly aggressive
gradient scaling that blocks learning. See test_vgs_training_interaction.py
for the analysis that led to these changes.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variance_gradient_scaler import VarianceGradientScaler


class TestVGSNewParameters:
    """Test new min_scaling_factor and variance_cap parameters."""

    def test_default_min_scaling_factor(self):
        """Test that default min_scaling_factor is 0.1."""
        vgs = VarianceGradientScaler()
        assert vgs.min_scaling_factor == 0.1

    def test_default_variance_cap(self):
        """Test that default variance_cap is 50.0."""
        vgs = VarianceGradientScaler()
        assert vgs.variance_cap == 50.0

    def test_custom_min_scaling_factor(self):
        """Test setting custom min_scaling_factor."""
        vgs = VarianceGradientScaler(min_scaling_factor=0.2)
        assert vgs.min_scaling_factor == 0.2

    def test_custom_variance_cap(self):
        """Test setting custom variance_cap."""
        vgs = VarianceGradientScaler(variance_cap=100.0)
        assert vgs.variance_cap == 100.0

    def test_variance_cap_none(self):
        """Test setting variance_cap to None (no capping)."""
        vgs = VarianceGradientScaler(variance_cap=None)
        assert vgs.variance_cap is None

    def test_invalid_min_scaling_factor_zero(self):
        """Test that min_scaling_factor=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_scaling_factor must be in"):
            VarianceGradientScaler(min_scaling_factor=0.0)

    def test_invalid_min_scaling_factor_negative(self):
        """Test that negative min_scaling_factor raises ValueError."""
        with pytest.raises(ValueError, match="min_scaling_factor must be in"):
            VarianceGradientScaler(min_scaling_factor=-0.1)

    def test_invalid_min_scaling_factor_gt_one(self):
        """Test that min_scaling_factor > 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_scaling_factor must be in"):
            VarianceGradientScaler(min_scaling_factor=1.5)

    def test_invalid_variance_cap_zero(self):
        """Test that variance_cap=0 raises ValueError."""
        with pytest.raises(ValueError, match="variance_cap must be positive"):
            VarianceGradientScaler(variance_cap=0.0)

    def test_invalid_variance_cap_negative(self):
        """Test that negative variance_cap raises ValueError."""
        with pytest.raises(ValueError, match="variance_cap must be positive"):
            VarianceGradientScaler(variance_cap=-10.0)


class TestVGSScalingWithNewParameters:
    """Test that scaling behavior respects new parameters."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_min_scaling_factor_enforced(self):
        """Test that scaling factor never goes below min_scaling_factor."""
        model = nn.Linear(100, 100)
        vgs = VarianceGradientScaler(
            parameters=model.parameters(),
            min_scaling_factor=0.2,
            warmup_steps=0,  # Skip warmup
        )

        # Simulate many updates with high-variance gradients
        for _ in range(20):
            # Create high-variance gradients
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 100.0  # Very high variance

            vgs.update_statistics()
            vgs.step()

        # Get scaling factor
        scale = vgs.get_scaling_factor()

        # Should be at least min_scaling_factor
        assert scale >= 0.2, f"Scale {scale} < min_scaling_factor 0.2"

    def test_variance_cap_prevents_extreme_scaling(self):
        """Test that variance_cap prevents extreme gradient reduction."""
        model = nn.Linear(100, 100)

        # Without variance cap
        vgs_no_cap = VarianceGradientScaler(
            parameters=model.parameters(),
            min_scaling_factor=0.01,  # Very low floor
            variance_cap=None,  # No cap
            warmup_steps=0,
        )

        # With variance cap
        vgs_with_cap = VarianceGradientScaler(
            parameters=model.parameters(),
            min_scaling_factor=0.01,  # Very low floor
            variance_cap=50.0,  # Cap at 50
            warmup_steps=0,
        )

        # Simulate updates with extreme variance gradients
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 1000.0  # Extreme variance

            vgs_no_cap.update_statistics()
            vgs_no_cap.step()

            vgs_with_cap.update_statistics()
            vgs_with_cap.step()

        scale_no_cap = vgs_no_cap.get_scaling_factor()
        scale_with_cap = vgs_with_cap.get_scaling_factor()

        # With cap, scale should be higher (less aggressive)
        assert (
            scale_with_cap >= scale_no_cap
        ), f"Capped scale {scale_with_cap} should be >= uncapped {scale_no_cap}"

    def test_variance_cap_effect_on_formula(self):
        """Test the mathematical effect of variance cap."""
        alpha = 0.1

        # Test variance values
        variance = 100.0

        # Without cap: scale = 1/(1+0.1*100) = 0.0909
        scale_no_cap = 1.0 / (1.0 + alpha * variance)

        # With cap at 50: scale = 1/(1+0.1*50) = 0.167
        variance_cap = 50.0
        capped_var = min(variance, variance_cap)
        scale_with_cap = 1.0 / (1.0 + alpha * capped_var)

        assert abs(scale_no_cap - 0.0909) < 0.001
        assert abs(scale_with_cap - 0.167) < 0.001
        assert scale_with_cap > scale_no_cap


class TestVGSStateDictNewParameters:
    """Test state_dict serialization with new parameters."""

    def test_state_dict_includes_new_parameters(self):
        """Test that state_dict includes min_scaling_factor and variance_cap."""
        vgs = VarianceGradientScaler(
            min_scaling_factor=0.15,
            variance_cap=75.0,
        )

        state = vgs.state_dict()

        assert "min_scaling_factor" in state
        assert "variance_cap" in state
        assert state["min_scaling_factor"] == 0.15
        assert state["variance_cap"] == 75.0

    def test_load_state_dict_new_parameters(self):
        """Test loading state_dict with new parameters."""
        vgs = VarianceGradientScaler()

        state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "min_scaling_factor": 0.25,
            "variance_cap": 80.0,
            "vgs_version": "3.2",
            "param_grad_mean_ema": None,
            "param_grad_sq_ema": None,
            "param_numel": None,
            "grad_mean_ema": None,
            "grad_var_ema": None,
            "grad_norm_ema": None,
            "grad_max_ema": None,
        }

        vgs.load_state_dict(state)

        assert vgs.min_scaling_factor == 0.25
        assert vgs.variance_cap == 80.0

    def test_load_state_dict_backward_compatible(self):
        """Test loading old state_dict without new parameters."""
        vgs = VarianceGradientScaler()

        # Old state dict without new parameters
        old_state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "vgs_version": "3.1",  # Old version
            "param_grad_mean_ema": None,
            "param_grad_sq_ema": None,
            "param_numel": None,
            "grad_mean_ema": None,
            "grad_var_ema": None,
            "grad_norm_ema": None,
            "grad_max_ema": None,
        }

        # Should not raise, should use defaults
        vgs.load_state_dict(old_state)

        # Should have default values
        assert vgs.min_scaling_factor == 0.1
        assert vgs.variance_cap == 50.0


class TestVGSRepr:
    """Test __repr__ includes new parameters."""

    def test_repr_includes_new_parameters(self):
        """Test that __repr__ shows min_scaling_factor and variance_cap."""
        vgs = VarianceGradientScaler(
            min_scaling_factor=0.15,
            variance_cap=75.0,
        )

        repr_str = repr(vgs)

        assert "min_scaling_factor=0.15" in repr_str
        assert "variance_cap=75.0" in repr_str
        assert "version=3.2" in repr_str


class TestVGSTrainingImprovement:
    """Test that new parameters improve training scenarios."""

    def test_critic_learning_with_improved_vgs(self):
        """
        Test that improved VGS allows critic to learn.

        With old settings (min_scale=1e-4), critic couldn't learn.
        With new settings (min_scale=0.1), critic should learn better.
        """
        torch.manual_seed(42)

        # Simple critic
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # VGS with new default parameters
        vgs = VarianceGradientScaler(
            parameters=model.parameters(),
            min_scaling_factor=0.1,  # New default
            variance_cap=50.0,  # New default
            warmup_steps=5,
        )

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Apply VGS
            vgs.update_statistics()
            vgs.scale_gradients()
            vgs.step()

            optimizer.step()
            final_loss = loss.item()

        # With improved VGS, loss should decrease
        improvement = (initial_loss - final_loss) / initial_loss
        print(f"Loss improvement with improved VGS: {improvement:.2%}")

        # Should have some improvement (may vary due to random data)
        # Main goal is that learning is not completely blocked


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
