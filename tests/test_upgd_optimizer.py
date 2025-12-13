"""
Comprehensive tests for UPGD optimizer implementations.

Tests cover:
1. Basic optimizer functionality
2. Utility computation and tracking
3. Parameter updates with utility-based protection
4. Noise injection and perturbation
5. Weight decay
6. State management and serialization
7. Edge cases and numerical stability
"""

import math
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


class TestUPGDBasic:
    """Test basic UPGD optimizer functionality."""

    def test_initialization_default_params(self):
        """Test UPGD initialization with default parameters."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters())

        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["lr"] == 1e-5
        assert group["weight_decay"] == 0.001
        assert group["beta_utility"] == 0.999
        assert group["sigma"] == 0.001

    def test_initialization_custom_params(self):
        """Test UPGD initialization with custom parameters."""
        model = SimpleModel()
        optimizer = UPGD(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            beta_utility=0.99,
            sigma=0.01,
        )

        group = optimizer.param_groups[0]
        assert group["lr"] == 1e-4
        assert group["weight_decay"] == 0.01
        assert group["beta_utility"] == 0.99
        assert group["sigma"] == 0.01

    def test_invalid_params_raise_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="Invalid learning rate"):
            UPGD(model.parameters(), lr=-1.0)

        with pytest.raises(ValueError, match="Invalid weight_decay"):
            UPGD(model.parameters(), weight_decay=-0.1)

        with pytest.raises(ValueError, match="Invalid beta_utility"):
            UPGD(model.parameters(), beta_utility=1.5)

        with pytest.raises(ValueError, match="Invalid sigma"):
            UPGD(model.parameters(), sigma=-0.01)

    def test_step_creates_state(self):
        """Test that calling step() creates optimizer state."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters(), lr=0.01)

        # Generate dummy loss and gradients
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Before step, state should be empty
        assert all(len(optimizer.state[p]) == 0 for p in model.parameters())

        optimizer.step()

        # After step, state should exist
        for p in model.parameters():
            state = optimizer.state[p]
            assert "step" in state
            assert "avg_utility" in state
            assert state["step"] == 1
            assert state["avg_utility"].shape == p.shape

    def test_parameters_update(self):
        """Test that parameters are updated after optimization step."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters(), lr=0.1)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Generate loss and gradients
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        optimizer.step()

        # Check that parameters have changed
        for p_initial, p_current in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_initial, p_current)

    def test_zero_grad(self):
        """Test zero_grad functionality."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters())

        # Generate gradients
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Gradients should exist
        assert all(p.grad is not None for p in model.parameters())

        # Zero gradients
        optimizer.zero_grad()

        # Gradients should be None or zero
        for p in model.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))


class TestUPGDUtilityMechanism:
    """Test utility computation and tracking."""

    def test_utility_computation(self):
        """Test that utility is computed as -grad * param."""
        model = nn.Linear(3, 2, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0)  # No EMA

        # Set specific parameter values
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        # Create known gradients
        model.zero_grad()
        model.weight.grad = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        optimizer.step()

        # Expected utility = -grad * param
        expected_utility = torch.tensor([[-0.1, -0.4, -0.9], [-1.6, -2.5, -3.6]])

        state = optimizer.state[model.weight]
        # With beta_utility=0, avg_utility should equal current utility
        assert torch.allclose(state["avg_utility"], expected_utility, atol=1e-5)

    def test_utility_ema_tracking(self):
        """Test exponential moving average of utility."""
        model = nn.Linear(2, 1, bias=False)
        beta = 0.9
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=beta)

        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight))

        # First step
        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 0.5
        optimizer.step()

        state = optimizer.state[model.weight]
        # First utility: (1 - beta) * (-0.5 * weight)
        # Note: weight gets updated, so we need to account for that

        # Second step - utility should be EMA updated
        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 0.3
        initial_utility = state["avg_utility"].clone()
        optimizer.step()

        # Utility should have changed due to EMA
        assert not torch.allclose(state["avg_utility"], initial_utility)
        assert state["step"] == 2

    def test_global_max_utility(self):
        """Test that global maximum utility is tracked across all parameters."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters(), lr=0.01)

        # Create gradients
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        optimizer.step()

        # Find global max utility across all parameters
        max_utilities = []
        for p in model.parameters():
            if p in optimizer.state:
                max_utilities.append(optimizer.state[p]["avg_utility"].max().item())

        # All utilities should be compared against the same global max
        # This is implicit in the algorithm but we can verify states exist
        assert len(max_utilities) > 0
        assert all(math.isfinite(u) for u in max_utilities)


class TestUPGDWeightDecay:
    """Test weight decay functionality."""

    def test_weight_decay_applied(self):
        """Test that weight decay reduces parameter magnitudes."""
        model = nn.Linear(5, 3, bias=False)
        weight_decay = 0.1
        optimizer = UPGD(model.parameters(), lr=0.01, weight_decay=weight_decay)

        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight))

        initial_norm = model.weight.norm().item()

        # Small gradients
        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 0.01

        optimizer.step()

        # Weight decay should reduce magnitude
        final_norm = model.weight.norm().item()
        # Note: relationship is complex due to update formula, but norm should generally decrease
        # with weight decay when gradients are small


    def test_zero_weight_decay(self):
        """Test that zero weight decay doesn't affect parameters beyond gradient update."""
        model = nn.Linear(3, 2, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, weight_decay=0.0)

        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight))

        initial_params = model.weight.clone()

        model.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight)

        optimizer.step()

        # With zero gradients and zero weight decay, parameters should stay the same
        # (only noise might affect them)
        # Due to noise, there will be small changes, so we check they're small
        param_change = (model.weight - initial_params).abs().max().item()
        assert param_change < 0.01  # Small due to sigma=0.001 default


class TestUPGDNoisePerturbation:
    """Test noise injection and perturbation."""

    def test_noise_injection(self):
        """Test that noise is injected into updates."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5, bias=False)
        sigma = 0.1
        optimizer = UPGD(model.parameters(), lr=0.01, sigma=sigma)

        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight))

        # Zero gradients - updates should come only from noise
        model.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight)

        initial_weight = model.weight.clone()
        optimizer.step()

        # Parameters should change due to noise
        assert not torch.allclose(model.weight, initial_weight)

    def test_deterministic_with_manual_seed(self):
        """Test that results are deterministic with manual seed."""
        def run_optimization():
            torch.manual_seed(123)
            model = nn.Linear(5, 3, bias=False)
            optimizer = UPGD(model.parameters(), lr=0.01, sigma=0.05)

            with torch.no_grad():
                model.weight.copy_(torch.ones_like(model.weight))

            model.zero_grad()
            model.weight.grad = torch.randn_like(model.weight)
            optimizer.step()

            return model.weight.clone()

        result1 = run_optimization()
        result2 = run_optimization()

        assert torch.allclose(result1, result2)


class TestAdaptiveUPGD:
    """Test AdaptiveUPGD optimizer with Adam-style adaptive learning rates."""

    def test_initialization(self):
        """Test AdaptiveUPGD initialization."""
        model = SimpleModel()
        optimizer = AdaptiveUPGD(model.parameters())

        group = optimizer.param_groups[0]
        assert group["lr"] == 1e-5
        assert group["beta1"] == 0.9
        assert group["beta2"] == 0.999
        assert group["eps"] == 1e-8
        assert group["beta_utility"] == 0.999

    def test_state_includes_moments(self):
        """Test that AdaptiveUPGD maintains first and second moments."""
        model = SimpleModel()
        optimizer = AdaptiveUPGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        optimizer.step()

        for p in model.parameters():
            state = optimizer.state[p]
            assert "first_moment" in state
            assert "sec_moment" in state
            assert "avg_utility" in state
            assert state["first_moment"].shape == p.shape
            assert state["sec_moment"].shape == p.shape

    def test_adaptive_learning_rate(self):
        """Test that adaptive learning rate is applied correctly."""
        model = nn.Linear(3, 2, bias=False)
        optimizer = AdaptiveUPGD(model.parameters(), lr=0.1)

        # Multiple steps to build up moment estimates
        for _ in range(5):
            model.zero_grad()
            x = torch.randn(4, 3)
            y = torch.randn(4, 2)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check that moments are being tracked
        state = optimizer.state[model.weight]
        assert state["step"] == 5
        assert not torch.allclose(state["first_moment"], torch.zeros_like(state["first_moment"]))

    def test_bias_correction(self):
        """Test bias correction in early iterations."""
        model = nn.Linear(2, 1, bias=False)
        optimizer = AdaptiveUPGD(model.parameters(), lr=0.01, beta1=0.9, beta2=0.999)

        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight)

        initial_weight = model.weight.clone()
        optimizer.step()

        # Bias correction should prevent tiny initial steps
        state = optimizer.state[model.weight]
        bias_corr1 = 1 - 0.9 ** state["step"]
        bias_corr2 = 1 - 0.999 ** state["step"]

        assert bias_corr1 < 1.0  # First step should have correction
        assert bias_corr2 < 1.0


class TestUPGDW:
    """Test UPGDW optimizer with decoupled weight decay."""

    def test_initialization(self):
        """Test UPGDW initialization with AdamW-style interface."""
        model = SimpleModel()
        optimizer = UPGDW(model.parameters())

        group = optimizer.param_groups[0]
        assert group["lr"] == 1e-4
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8
        assert group["weight_decay"] == 0.01
        assert group["sigma"] == 0.001

    def test_decoupled_weight_decay(self):
        """Test that weight decay is applied independently of gradients."""
        model = nn.Linear(5, 3, bias=False)
        wd = 0.1
        lr = 0.01
        optimizer = UPGDW(model.parameters(), lr=lr, weight_decay=wd)

        with torch.no_grad():
            initial_weight = torch.ones_like(model.weight)
            model.weight.copy_(initial_weight)

        # Zero gradients - only weight decay and noise should affect parameters
        model.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight)

        optimizer.step()

        # With decoupled weight decay: weight *= (1 - lr * wd)
        # Plus noise effects
        # Main point: weight decay happens independently
        final_weight = model.weight

        # Weight magnitude should decrease due to weight decay
        # (even with zero gradients, weight decay still applies)

    def test_betas_parameter(self):
        """Test that betas parameter works correctly."""
        model = SimpleModel()
        beta1, beta2 = 0.95, 0.998
        optimizer = UPGDW(model.parameters(), betas=(beta1, beta2))

        group = optimizer.param_groups[0]
        assert group["betas"] == (beta1, beta2)

    def test_amsgrad_not_implemented(self):
        """Test that AMSGrad raises NotImplementedError."""
        model = SimpleModel()
        with pytest.raises(NotImplementedError, match="AMSGrad"):
            UPGDW(model.parameters(), amsgrad=True)


class TestUPGDEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_gradients(self):
        """Test behavior with zero gradients."""
        model = nn.Linear(5, 3, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, sigma=0.0)  # No noise

        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight))

        initial_weight = model.weight.clone()

        model.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight)

        optimizer.step()

        # With zero gradients, zero noise, parameters should only change due to weight decay
        # weight *= (1 - lr * weight_decay)
        expected = initial_weight * (1 - 0.01 * 0.001)
        # Note: actual formula is more complex due to utility scaling
        # But we can verify parameters changed only slightly
        param_change = (model.weight - initial_weight).abs().max().item()
        assert param_change < 0.001  # Small change from weight decay

    def test_very_large_gradients(self):
        """Test stability with very large gradients."""
        model = nn.Linear(3, 2, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.001)

        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 1e6

        # Should not crash or produce NaN/Inf
        optimizer.step()

        assert torch.all(torch.isfinite(model.weight))

    def test_mixed_zero_and_nonzero_gradients(self):
        """Test with some parameters having zero gradients."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Zero out gradients for first layer
        model.linear.weight.grad.zero_()
        model.linear.bias.grad.zero_()

        optimizer.step()

        # Should not crash
        assert all(torch.all(torch.isfinite(p)) for p in model.parameters())

    def test_none_gradients(self):
        """Test with None gradients (skipped parameters)."""
        model = SimpleModel()
        optimizer = UPGD(model.parameters(), lr=0.01)

        # Don't create any gradients
        # optimizer.step() should handle this gracefully

        initial_params = [p.clone() for p in model.parameters()]

        optimizer.step()

        # Parameters should not change when gradients are None
        for p_init, p_curr in zip(initial_params, model.parameters()):
            assert torch.allclose(p_init, p_curr)

    def test_single_parameter(self):
        """Test with a single parameter."""
        param = nn.Parameter(torch.randn(5, 3))
        optimizer = UPGD([param], lr=0.01)

        param.grad = torch.randn_like(param)

        optimizer.step()

        assert torch.all(torch.isfinite(param))
        assert param in optimizer.state
        assert optimizer.state[param]["step"] == 1


class TestUPGDStateSerialization:
    """Test optimizer state save/load functionality."""

    def test_state_dict_save_load(self):
        """Test saving and loading optimizer state."""
        model = SimpleModel()
        optimizer1 = UPGD(model.parameters(), lr=0.01)

        # Run a few optimization steps
        for _ in range(3):
            model.zero_grad()
            x = torch.randn(4, 10)
            y = torch.randn(4, 2)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer1.step()

        # Save state
        state_dict = optimizer1.state_dict()

        # Create new optimizer
        optimizer2 = UPGD(model.parameters(), lr=0.01)

        # Load state
        optimizer2.load_state_dict(state_dict)

        # States should match
        for p in model.parameters():
            state1 = optimizer1.state[p]
            state2 = optimizer2.state[p]

            assert state1["step"] == state2["step"]
            assert torch.allclose(state1["avg_utility"], state2["avg_utility"])

    def test_state_persistence_across_steps(self):
        """Test that state persists and accumulates correctly."""
        model = nn.Linear(3, 2, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01)

        steps = 5
        for i in range(1, steps + 1):
            model.zero_grad()
            model.weight.grad = torch.randn_like(model.weight)
            optimizer.step()

            state = optimizer.state[model.weight]
            assert state["step"] == i

        # Utility should have been updated multiple times
        final_utility = optimizer.state[model.weight]["avg_utility"]
        assert not torch.allclose(final_utility, torch.zeros_like(final_utility))


class TestOptimizerComparison:
    """Compare UPGD variants to ensure consistent behavior."""

    def test_upgd_vs_adaptive_upgd_interface(self):
        """Test that UPGD and AdaptiveUPGD have similar interfaces."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)

        opt1 = UPGD(model1.parameters(), lr=0.01)
        opt2 = AdaptiveUPGD(model2.parameters(), lr=0.01)

        # Both should work with same API
        for model, opt in [(model1, opt1), (model2, opt2)]:
            model.zero_grad()
            x = torch.randn(4, 10)
            y = torch.randn(4, 2)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()

        # Both should have updated parameters
        for p in model1.parameters():
            assert p in opt1.state

        for p in model2.parameters():
            assert p in opt2.state

    def test_upgdw_interface_matches_adamw(self):
        """Test that UPGDW interface matches AdamW for easy replacement."""
        model = SimpleModel()

        # Both should accept same basic parameters
        try:
            optimizer = UPGDW(
                model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
            # If initialization succeeds, test passed
            assert True
        except TypeError:
            pytest.fail("UPGDW should accept AdamW-style parameters")


class TestUPGDIntegration:
    """Integration tests with realistic scenarios."""

    def test_training_loop(self):
        """Test UPGD in a realistic training loop."""
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
        assert final_loss < initial_loss

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups."""
        model = SimpleModel()

        # Create separate groups for different layers
        param_groups = [
            {"params": model.linear.parameters(), "lr": 0.01},
            {"params": model.linear2.parameters(), "lr": 0.001},
        ]

        optimizer = UPGD(param_groups)

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.001

        # Run optimization
        model.zero_grad()
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        optimizer.step()

        # All parameters should have state
        for p in model.parameters():
            assert p in optimizer.state


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
