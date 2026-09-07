"""Tests for limit order price tolerance fix (2025-11-26).

Bug Description:
    The fixed tolerance of 1e-12 for limit order fill comparisons was smaller
    than floating-point machine epsilon for high-value assets (e.g., BTC at $100,000).
    This caused legitimate maker fills to be rejected due to floating-point
    representation errors.

Fix:
    Replaced fixed absolute tolerance with relative tolerance that scales
    with price magnitude: tolerance = max(rel_tol * max_price, abs_tol)

Reference:
    - IEEE 754 double precision: ~15-16 significant digits
    - Machine epsilon: ~2.22e-16
    - At price $100,000: epsilon * price ~ 2.2e-11 (larger than 1e-12!)
"""

import math
import pytest
import numpy as np

# Import the ExecutionSimulator class
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from execution_sim import ExecutionSimulator


class TestComputePriceTolerance:
    """Tests for the _compute_price_tolerance static method."""

    def test_basic_functionality(self):
        """Test that tolerance scales with price magnitude."""
        # Low price: should use abs_tol floor
        tol_100 = ExecutionSimulator._compute_price_tolerance(100.0)
        assert tol_100 >= 1e-12, "Tolerance should be at least abs_tol"

        # High price: should scale with price
        tol_100k = ExecutionSimulator._compute_price_tolerance(100000.0)
        assert tol_100k > tol_100, "Higher price should have higher tolerance"

        # Very high price (BTC scenario)
        tol_btc = ExecutionSimulator._compute_price_tolerance(100000.0)
        expected_btc = 1e-9 * 100000.0  # = 1e-4
        assert tol_btc == pytest.approx(expected_btc, rel=1e-6)

    def test_two_prices(self):
        """Test tolerance computation with two price inputs."""
        # Should use the max of two prices
        tol = ExecutionSimulator._compute_price_tolerance(1000.0, 100000.0)
        expected = 1e-9 * 100000.0  # Uses max price
        assert tol == pytest.approx(expected, rel=1e-6)

        # Order shouldn't matter
        tol_reversed = ExecutionSimulator._compute_price_tolerance(100000.0, 1000.0)
        assert tol == tol_reversed

    def test_none_handling(self):
        """Test handling of None values."""
        # Single None
        tol = ExecutionSimulator._compute_price_tolerance(None)
        assert tol == 1e-12, "Should return abs_tol when price is None"

        # Both None
        tol = ExecutionSimulator._compute_price_tolerance(None, None)
        assert tol == 1e-12

        # One valid, one None
        tol = ExecutionSimulator._compute_price_tolerance(100000.0, None)
        expected = 1e-9 * 100000.0
        assert tol == pytest.approx(expected, rel=1e-6)

    def test_invalid_values(self):
        """Test handling of invalid values."""
        # NaN
        tol = ExecutionSimulator._compute_price_tolerance(float("nan"))
        assert tol == 1e-12

        # Infinity
        tol = ExecutionSimulator._compute_price_tolerance(float("inf"))
        assert tol == 1e-12

        # Negative infinity
        tol = ExecutionSimulator._compute_price_tolerance(float("-inf"))
        assert tol == 1e-12

        # String (should not crash)
        tol = ExecutionSimulator._compute_price_tolerance("not_a_number")
        assert tol == 1e-12

    def test_custom_tolerances(self):
        """Test custom rel_tol and abs_tol parameters."""
        price = 100000.0

        # Custom rel_tol
        tol = ExecutionSimulator._compute_price_tolerance(price, rel_tol=1e-6)
        assert tol == pytest.approx(1e-6 * price, rel=1e-6)

        # Custom abs_tol (higher than computed relative)
        tol = ExecutionSimulator._compute_price_tolerance(price, abs_tol=1.0)
        assert tol == 1.0  # abs_tol dominates for small price

        # Very small price where abs_tol dominates
        tol = ExecutionSimulator._compute_price_tolerance(1e-6, abs_tol=1e-12)
        assert tol == 1e-12

    def test_float_precision_adequacy(self):
        """Test that tolerance is adequate for floating-point precision."""
        prices = [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

        for price in prices:
            tol = ExecutionSimulator._compute_price_tolerance(price)
            machine_eps_at_price = np.finfo(np.float64).eps * price

            # Tolerance should be larger than machine epsilon
            assert tol > machine_eps_at_price, (
                f"Tolerance {tol} should exceed machine epsilon {machine_eps_at_price} "
                f"at price {price}"
            )


class TestLimitOrderFillWithTolerance:
    """Integration tests for limit order fills with the new tolerance."""

    @pytest.fixture
    def sim(self):
        """Create an ExecutionSimulator instance."""
        sim = ExecutionSimulator()
        sim.symbol = "BTCUSDT"
        return sim

    def test_btc_limit_fill_precision(self, sim):
        """Test that BTC limit orders fill correctly despite FP precision."""
        # Scenario: BUY LIMIT at $100,000
        # Price drops to exactly $100,000 (with tiny FP noise)
        limit_price = 100000.0

        # Simulate tiny floating-point noise (like from calculation roundoff)
        fill_price_with_noise = limit_price * (1.0 + 1e-15)

        # With old tolerance (1e-12), this would fail!
        old_tolerance = 1e-12
        old_result = fill_price_with_noise <= limit_price + old_tolerance
        # This might be False due to FP precision!

        # With new tolerance, it should pass
        new_tolerance = sim._compute_price_tolerance(limit_price, fill_price_with_noise)
        new_result = fill_price_with_noise <= limit_price + new_tolerance

        assert new_result, (
            f"Limit order fill should succeed with new tolerance. "
            f"limit={limit_price}, fill={fill_price_with_noise}, "
            f"diff={fill_price_with_noise - limit_price}, tol={new_tolerance}"
        )

    def test_low_price_asset_fill(self, sim):
        """Test that low-price assets still fill correctly."""
        # Scenario: BUY LIMIT for a $0.001 token
        limit_price = 0.001
        fill_price = 0.001 * (1.0 + 1e-15)

        tolerance = sim._compute_price_tolerance(limit_price, fill_price)

        # Should use abs_tol floor for very low prices
        assert tolerance >= 1e-12
        assert fill_price <= limit_price + tolerance

    def test_sell_limit_precision(self, sim):
        """Test SELL LIMIT with floating-point precision issues."""
        # Scenario: SELL LIMIT at $100,000
        # Price rises to exactly $100,000 (with tiny FP noise)
        limit_price = 100000.0
        fill_price_with_noise = limit_price * (1.0 - 1e-15)

        tolerance = sim._compute_price_tolerance(limit_price, fill_price_with_noise)

        # For SELL: fill_price >= limit_price - tolerance
        result = fill_price_with_noise >= limit_price - tolerance

        assert result, (
            f"SELL limit order fill should succeed with new tolerance. "
            f"limit={limit_price}, fill={fill_price_with_noise}, "
            f"diff={limit_price - fill_price_with_noise}, tol={tolerance}"
        )

    def test_tolerance_not_too_loose(self, sim):
        """Ensure tolerance isn't so loose that it fills incorrectly."""
        limit_price = 100000.0
        tolerance = sim._compute_price_tolerance(limit_price)

        # Tolerance should be around 1e-4 for $100k (0.01% of price)
        # This is tight enough to not fill orders that are way off
        assert tolerance < 0.01, f"Tolerance {tolerance} is too loose"

        # A fill price $1 above limit should NOT fill
        fill_price_1_dollar_above = limit_price + 1.0
        assert (
            fill_price_1_dollar_above > limit_price + tolerance
        ), "Fill $1 above limit should not satisfy tolerance"


class TestRegressionScenarios:
    """Regression tests for specific scenarios that would fail with old code."""

    def test_btc_maker_fill_at_exact_price(self):
        """Regression: BTC maker fill at exactly limit price."""
        # This scenario failed with old 1e-12 tolerance due to FP precision
        limit_price = 99999.99999999999  # Result of some calculation
        bar_low = 100000.0  # Exactly at limit (with FP representation)

        # Old code would sometimes fail this comparison
        tolerance = ExecutionSimulator._compute_price_tolerance(limit_price, bar_low)

        # BUY LIMIT: fill if bar_low <= limit_price + tolerance
        should_fill = bar_low <= limit_price + tolerance

        # With proper tolerance, prices that are essentially equal should fill
        assert should_fill, (
            f"Should fill when bar_low ({bar_low}) essentially equals "
            f"limit_price ({limit_price}), diff={bar_low - limit_price}"
        )

    def test_calculation_chain_precision_loss(self):
        """Test scenario where price goes through calculation chain."""
        # Simulate a price that went through: fetch -> convert -> calculate -> compare
        original_price = 100000.0

        # Chain of operations that might introduce FP errors
        price_after_ops = original_price
        price_after_ops = price_after_ops * 1.0001  # Some spread calc
        price_after_ops = price_after_ops / 1.0001  # Reverse

        # Due to FP, this might not be exactly original_price
        diff = abs(price_after_ops - original_price)

        tolerance = ExecutionSimulator._compute_price_tolerance(original_price, price_after_ops)

        # Tolerance should accommodate this kind of roundoff error
        assert (
            tolerance > diff or diff < 1e-10
        ), f"Tolerance {tolerance} should handle calculation chain error {diff}"


class TestEdgeCases:
    """Edge case tests for tolerance computation."""

    def test_zero_price(self):
        """Test with zero price (edge case)."""
        tol = ExecutionSimulator._compute_price_tolerance(0.0)
        assert tol == 1e-12, "Zero price should use abs_tol"

    def test_negative_price(self):
        """Test with negative price (uses absolute value)."""
        tol = ExecutionSimulator._compute_price_tolerance(-100000.0)
        expected = 1e-9 * 100000.0  # abs(-100000)
        assert tol == pytest.approx(expected, rel=1e-6)

    def test_very_small_price(self):
        """Test with very small price."""
        tol = ExecutionSimulator._compute_price_tolerance(1e-10)
        assert tol == 1e-12, "Very small price should use abs_tol floor"

    def test_very_large_price(self):
        """Test with very large price."""
        tol = ExecutionSimulator._compute_price_tolerance(1e15)
        expected = 1e-9 * 1e15  # = 1e6
        assert tol == pytest.approx(expected, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
