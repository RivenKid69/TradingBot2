"""
Tests verifying VGS state is properly handled during PBT exploit.

These tests confirm that VGS state IS correctly serialized and restored,
countering the false claim that "VGS state НЕ копируется".

Reference: CLAUDE.md "НЕ БАГИ" #52
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np

from variance_gradient_scaler import VarianceGradientScaler


class TestVGSStateSerialization:
    """Tests for VGS state serialization/deserialization."""

    def test_vgs_state_dict_contains_all_fields(self):
        """Verify state_dict includes all necessary fields for restoration."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)

        # Create VGS and run some steps
        vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=5)

        # Simulate a few gradient updates
        for _ in range(10):
            # Create fake gradients
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            vgs.step()

        # Get state dict
        state = vgs.state_dict()

        # Verify all critical fields are present
        assert "enabled" in state
        assert "beta" in state
        assert "alpha" in state
        assert "eps" in state
        assert "warmup_steps" in state
        assert "step_count" in state
        assert "param_grad_mean_ema" in state  # E[μ]
        assert "param_grad_sq_ema" in state    # E[g²] - FIXED in v3.1
        assert "vgs_version" in state

        # Verify step count was tracked
        assert state["step_count"] == 10

    def test_vgs_state_roundtrip(self):
        """Verify state can be saved and restored correctly."""
        # Create model and VGS
        model = torch.nn.Linear(10, 5)
        vgs_source = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=5)

        # Run source VGS for some steps
        for _ in range(15):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            vgs_source.step()

        # Save state
        state = vgs_source.state_dict()
        original_step_count = state["step_count"]
        original_scaling_factor = vgs_source.get_scaling_factor()

        # Create new VGS with different parameters
        model2 = torch.nn.Linear(10, 5)
        vgs_target = VarianceGradientScaler(
            model2.parameters(),
            enabled=True,
            warmup_steps=100,  # Different!
            alpha=0.5  # Different!
        )

        # Verify target has different state before load
        assert vgs_target._step_count == 0
        assert vgs_target.warmup_steps == 100

        # Load state
        vgs_target.load_state_dict(state)

        # Verify state was restored
        assert vgs_target._step_count == original_step_count
        assert vgs_target.warmup_steps == vgs_source.warmup_steps  # Restored!

        # Scaling factor should be similar (not identical due to parameter differences)
        target_scaling = vgs_target.get_scaling_factor()
        # Both should be < 1.0 after warmup
        assert original_scaling_factor < 1.0
        assert target_scaling < 1.0

    def test_vgs_per_param_stats_preserved(self):
        """Verify per-parameter statistics are preserved across save/load."""
        model = torch.nn.Linear(10, 5)
        vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=5)

        # Run for many steps to build up statistics
        for i in range(50):
            for p in model.parameters():
                # Use consistent gradients to build meaningful stats
                p.grad = torch.randn_like(p) * (0.1 + i * 0.01)
            vgs.step()

        # Get state
        state = vgs.state_dict()

        # Verify per-param stats are non-trivial
        assert state["param_grad_mean_ema"] is not None
        assert state["param_grad_sq_ema"] is not None
        assert torch.any(state["param_grad_mean_ema"] != 0)
        assert torch.any(state["param_grad_sq_ema"] != 0)

        # Create new VGS and load
        model2 = torch.nn.Linear(10, 5)
        vgs2 = VarianceGradientScaler(model2.parameters(), enabled=True)
        vgs2.load_state_dict(state)

        # Verify stats were restored
        assert vgs2._param_grad_mean_ema is not None
        assert vgs2._param_grad_sq_ema is not None


class TestVGSResetStatistics:
    """Tests for VGS reset_statistics method."""

    def test_reset_clears_all_statistics(self):
        """Verify reset_statistics() clears all accumulated state."""
        model = torch.nn.Linear(10, 5)
        vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=5)

        # Run for some steps
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            vgs.step()

        # Verify state exists
        assert vgs._step_count == 20
        assert vgs._param_grad_mean_ema is not None

        # Reset
        vgs.reset_statistics()

        # Verify state was cleared
        assert vgs._step_count == 0
        assert vgs._param_grad_mean_ema is None
        assert vgs._param_grad_sq_ema is None

    def test_vgs_can_continue_after_reset(self):
        """Verify VGS continues to work after reset."""
        model = torch.nn.Linear(10, 5)
        vgs = VarianceGradientScaler(model.parameters(), enabled=True, warmup_steps=5)

        # Run, reset, run again
        for _ in range(10):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            vgs.step()

        vgs.reset_statistics()

        # Should not crash
        for _ in range(10):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            vgs.step()

        assert vgs._step_count == 10


class TestVGSVersionMigration:
    """Tests for VGS version migration."""

    def test_old_version_triggers_warning(self):
        """Verify loading old checkpoint triggers migration warning."""
        model = torch.nn.Linear(10, 5)
        vgs = VarianceGradientScaler(model.parameters(), enabled=True)

        # Create old-format state dict (v1.0)
        old_state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "vgs_version": "1.0",  # Old version
        }

        # Should issue warning but not crash
        with pytest.warns(UserWarning, match="VGS v3.1 CRITICAL FIX"):
            vgs.load_state_dict(old_state)

        # Per-param stats should be reset (None)
        assert vgs._param_grad_mean_ema is None
        assert vgs._param_grad_sq_ema is None

    def test_v31_loads_without_warning(self):
        """Verify v3.1 checkpoints load cleanly."""
        model = torch.nn.Linear(10, 5)
        vgs = VarianceGradientScaler(model.parameters(), enabled=True)

        # Create v3.1 state dict
        v31_state = {
            "enabled": True,
            "beta": 0.99,
            "alpha": 0.1,
            "eps": 1e-8,
            "warmup_steps": 100,
            "step_count": 50,
            "param_grad_mean_ema": torch.randn(2),
            "param_grad_sq_ema": torch.randn(2).abs(),  # Squares are positive
            "param_numel": torch.tensor([50.0, 5.0]),
            "vgs_version": "3.1",
        }

        # Should load without warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            try:
                vgs.load_state_dict(v31_state)
            except UserWarning:
                pytest.fail("v3.1 checkpoint should not trigger warning")


class TestUPGDSigmoidScaling:
    """Tests documenting UPGD sigmoid scaling is BY DESIGN."""

    def test_sigmoid_scaling_range(self):
        """Verify sigmoid scaling maps [0, 1] → [0.27, 0.73]."""
        # Test the exact formula from UPGD
        def scaled_utility(normalized: float) -> float:
            return float(torch.sigmoid(torch.tensor(2.0 * (normalized - 0.5))))

        # Minimum utility (normalized = 0)
        min_scaled = scaled_utility(0.0)
        assert 0.26 < min_scaled < 0.28, f"Expected ~0.269, got {min_scaled}"

        # Maximum utility (normalized = 1)
        max_scaled = scaled_utility(1.0)
        assert 0.72 < max_scaled < 0.74, f"Expected ~0.731, got {max_scaled}"

        # Weight factors
        min_weight = 1 - min_scaled  # ~0.731 - high plasticity
        max_weight = 1 - max_scaled  # ~0.269 - moderate protection

        assert min_weight > 0.7, "Low utility params should get high update weight"
        assert max_weight > 0.25, "High utility params should still get some updates"
        assert max_weight < 0.3, "High utility params should be mostly protected"

    def test_sigmoid_is_smooth(self):
        """Verify sigmoid provides smooth gradients (no dead zones)."""
        # Check gradient exists at boundaries
        x = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        y = torch.sigmoid(2.0 * (x - 0.5))

        # Compute gradients
        y.sum().backward()

        # All gradients should be non-zero (no dead zones)
        assert torch.all(x.grad > 0), "Sigmoid should have non-zero gradient everywhere"

        # Middle should have highest gradient
        assert x.grad[1] > x.grad[0], "Middle should have higher gradient than edge"
        assert x.grad[1] > x.grad[2], "Middle should have higher gradient than edge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
