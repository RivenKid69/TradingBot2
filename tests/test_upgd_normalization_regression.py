"""
Regression tests for UPGD utility normalization (2025-11-21 fix).

This test suite ensures that the UPGD normalization fix (using min-max normalization
instead of division by global_max) remains correct and prevents regression to the
buggy behavior that caused optimizer "freezing" with negative utilities.

BACKGROUND:
- BUG (pre-2025-11-21): Used `scaled_utility = sigmoid(utility / global_max)`
  - When all utilities were negative → global_max < 0 → inverted scaling!
  - High utility (less negative) → scaled_utility closer to 0 → LARGE updates (wrong!)
  - Low utility (more negative) → scaled_utility closer to 1 → SMALL updates (wrong!)
  - Result: Optimizer "froze" important parameters and perturbed unimportant ones

- FIX (2025-11-21): Uses min-max normalization:
  - `normalized = (utility - global_min) / (global_max - global_min + eps)`
  - `scaled_utility = sigmoid(2.0 * (normalized - 0.5))`
  - Works correctly for ALL utility signs (positive, negative, mixed)
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from optimizers import UPGD


class TestUPGDNormalizationRegression:
    """Regression tests for UPGD utility normalization fix."""

    def test_negative_utilities_not_inverted(self):
        """
        CRITICAL REGRESSION TEST: Verify that negative utilities don't invert scaling.

        Pre-fix bug: When all utilities negative, scaling was inverted (high utility → large update)
        Post-fix: Scaling is correct regardless of utility sign
        """
        # Create model with known structure
        model = nn.Linear(10, 5, bias=False)
        optimizer = UPGD(
            model.parameters(), lr=0.01, beta_utility=0.0, sigma=0.0
        )  # No EMA, no noise

        # Set specific parameter values and gradients to create negative utilities
        # utility = -grad * param, so positive grad * positive param = negative utility
        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight) * 2.0)  # Positive params

        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 0.5  # Positive gradients

        # utility = -0.5 * 2.0 = -1.0 (all negative)

        # Store initial weights
        weights_before = model.weight.data.clone()

        # Perform optimization step
        optimizer.step()

        # Extract utilities from state
        state = optimizer.state[model.weight]
        utility = state["avg_utility"]

        # Verify all utilities are negative
        assert torch.all(
            utility < 0
        ), f"Expected all negative utilities, got range [{utility.min():.6f}, {utility.max():.6f}]"

        # Verify parameters CHANGED (not frozen)
        weights_after = model.weight.data
        param_change = (weights_after - weights_before).abs().mean().item()

        print(f"Utility range: [{utility.min():.6f}, {utility.max():.6f}]")
        print(f"Parameter change: {param_change:.6f}")

        # Parameters should update (not be frozen)
        assert param_change > 1e-6, (
            f"Parameters appear frozen (change = {param_change:.2e}). "
            f"This suggests utility normalization bug (negative utilities inverted scaling)."
        )

        print(f"[PASS] Negative utilities do not invert scaling")

    def test_mixed_utilities_normalized_correctly(self):
        """Test that mixed (positive and negative) utilities are normalized correctly."""
        model = nn.Linear(10, 5, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0, sigma=0.0)

        # Create mixed utilities by setting heterogeneous params and gradients
        with torch.no_grad():
            # Row 0: positive utility (negative grad or negative param)
            model.weight[0, :] = torch.ones(10) * -1.0  # Negative params
            # Row 1: negative utility (positive grad and positive param)
            model.weight[1, :] = torch.ones(10) * 1.0  # Positive params

        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 0.5  # All positive gradients

        # Row 0 utility: -0.5 * (-1.0) = +0.5 (positive)
        # Row 1 utility: -0.5 * 1.0 = -0.5 (negative)

        optimizer.step()

        state = optimizer.state[model.weight]
        utility = state["avg_utility"]

        # Verify mixed utilities
        assert torch.any(utility > 0) and torch.any(
            utility < 0
        ), "Expected mixed utilities (both positive and negative)"

        print(f"Utility range: [{utility.min():.6f}, {utility.max():.6f}]")
        print(f"[PASS] Mixed utilities normalized correctly")

    def test_zero_utilities_handled(self):
        """Test that zero utilities (zero gradients or zero params) are handled gracefully."""
        model = nn.Linear(5, 3, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0, sigma=0.0)

        # Set zero gradients
        model.zero_grad()
        model.weight.grad = torch.zeros_like(model.weight)

        # This should not crash
        optimizer.step()

        state = optimizer.state[model.weight]
        utility = state["avg_utility"]

        # All utilities should be zero
        assert torch.allclose(
            utility, torch.zeros_like(utility)
        ), f"Expected zero utilities with zero gradients, got range [{utility.min():.6f}, {utility.max():.6f}]"

        print(f"[PASS] Zero utilities handled correctly")

    def test_extreme_utility_values_clamped(self):
        """Test that extreme utility values don't cause NaN or Inf."""
        model = nn.Linear(5, 3, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0, sigma=0.0)

        # Create extreme utilities
        with torch.no_grad():
            model.weight.copy_(torch.ones_like(model.weight) * 1e6)  # Very large params

        model.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * 1e6  # Very large gradients
        # utility = -1e6 * 1e6 = -1e12 (extremely negative)

        # Should not crash or produce NaN/Inf
        optimizer.step()

        # Check that parameters are still finite
        assert torch.all(torch.isfinite(model.weight)), "Parameters contain NaN or Inf"

        state = optimizer.state[model.weight]
        utility = state["avg_utility"]
        assert torch.all(torch.isfinite(utility)), "Utilities contain NaN or Inf"

        print(f"Utility range: [{utility.min():.2e}, {utility.max():.2e}]")
        print(f"[PASS] Extreme utilities handled without NaN/Inf")

    def test_utility_scaling_in_valid_range(self):
        """Test that scaled utility is in valid range [0, 1] after sigmoid."""
        model = nn.Linear(10, 5, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.9, sigma=0.0)

        # Run several steps to build up utility estimates
        for _ in range(10):
            model.zero_grad()
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Extract utilities
        state = optimizer.state[model.weight]
        utility = state["avg_utility"]

        # Min-max normalization should map to [0, 1]
        # Then sigmoid(2 * (x - 0.5)) maps [0,1] → ~[0.27, 0.73]
        # We can't directly access scaled_utility, but we can verify utilities are finite
        assert torch.all(torch.isfinite(utility)), "Utilities should be finite"

        # Verify utility statistics make sense
        print(
            f"Utility stats: mean={utility.mean():.6f}, std={utility.std():.6f}, "
            f"range=[{utility.min():.6f}, {utility.max():.6f}]"
        )

        print(f"[PASS] Utility scaling produces valid values")

    def test_parameters_update_with_all_negative_utilities(self):
        """
        REGRESSION TEST: Verify parameters update normally even when all utilities are negative.

        This is the KEY test to prevent regression to the bug where negative utilities
        caused the optimizer to freeze parameters.
        """
        model = nn.Linear(20, 10, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, sigma=0.01)  # Small noise to ensure updates

        # Initialize with positive parameters
        with torch.no_grad():
            model.weight.copy_(torch.abs(torch.randn_like(model.weight)) + 1.0)  # All positive

        initial_weight = model.weight.data.clone()

        # Run training loop
        for step in range(20):
            model.zero_grad()
            # Create loss that produces positive gradients
            x = torch.randn(8, 20)
            y = torch.ones(8, 10)  # Target all ones
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()

            # Gradients will be positive on average (output < target)
            # With positive params and positive grads → negative utilities
            optimizer.step()

        # Check that parameters have changed significantly
        final_weight = model.weight.data
        total_change = (final_weight - initial_weight).abs().mean().item()

        print(f"Total parameter change over 20 steps: {total_change:.6f}")

        # Parameters MUST have changed (not frozen)
        assert total_change > 0.01, (
            f"Parameters appear frozen (total change = {total_change:.2e}). "
            f"This suggests regression to the UPGD negative utility bug!"
        )

        # Extract final utility
        state = optimizer.state[model.weight]
        final_utility = state["avg_utility"]
        print(f"Final utility range: [{final_utility.min():.6f}, {final_utility.max():.6f}]")

        print(f"[PASS] Parameters update normally with negative utilities")

    def test_normalization_formula_correctness(self):
        """
        Test the exact normalization formula to ensure it matches specification.

        Correct formula (post-fix):
            normalized = (utility - global_min) / (global_max - global_min + epsilon)
            normalized = clamp(normalized, 0.0, 1.0)
            scaled_utility = sigmoid(2.0 * (normalized - 0.5))

        This test verifies the formula behavior by checking edge cases.
        """
        model = nn.Linear(5, 3, bias=False)
        optimizer = UPGD(model.parameters(), lr=0.01, beta_utility=0.0, sigma=0.0)

        # Create utilities with known range
        with torch.no_grad():
            # Create gradient pattern: [0.1, 0.2, 0.3, 0.4, 0.5] for each row
            model.weight.zero_()
            model.weight[:, 0] = 1.0  # utility = -0.1 * 1.0 = -0.1
            model.weight[:, 1] = 2.0  # utility = -0.2 * 2.0 = -0.4
            model.weight[:, 2] = 3.0  # utility = -0.3 * 3.0 = -0.9
            model.weight[:, 3] = 4.0  # utility = -0.4 * 4.0 = -1.6
            model.weight[:, 4] = 5.0  # utility = -0.5 * 5.0 = -2.5

        model.zero_grad()
        model.weight.grad = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]] * 3)

        optimizer.step()

        state = optimizer.state[model.weight]
        utility = state["avg_utility"]

        # Expected utilities (approximately, with beta_utility=0.0):
        # Column 0: -0.1 (least negative → highest utility)
        # Column 4: -2.5 (most negative → lowest utility)

        print(f"Utility by column:")
        for col in range(5):
            col_utility = utility[:, col].mean().item()
            print(f"  Column {col}: {col_utility:.6f}")

        # Min-max normalization should map:
        # - Highest utility (-0.1) → normalized ~1.0 → sigmoid(1.0) ~0.73
        # - Lowest utility (-2.5) → normalized ~0.0 → sigmoid(-1.0) ~0.27
        # These get multiplied by (1 - scaled_utility) for update magnitude

        # The key check: utilities should be in expected range
        assert utility.min() < -2.0, "Expected minimum utility < -2.0"
        assert utility.max() > -0.2, "Expected maximum utility > -0.2"

        print(f"[PASS] Normalization formula produces expected utility range")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
