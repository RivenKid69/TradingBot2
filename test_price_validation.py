"""
Tests for price validation including spike detection.

This test module validates the _validate_critical_price function in mediator.py
which includes:
1. Basic validation (NaN, Inf, <= 0)
2. Spike detection (abnormal price changes)

Best practices tested:
- "Best Practices for Ensuring Financial Data Accuracy" (Paystand)
- "Investment Model Validation" (CFA Institute)
- "Training ML Models with Financial Data" (EODHD)
- "Anomaly Detection in Financial Time Series" (Journal of Finance)
"""

import math
import pytest
import sys
from pathlib import Path

# Add project root to path
base = Path(__file__).resolve().parent
if str(base) not in sys.path:
    sys.path.append(str(base))

from mediator import Mediator


class TestPriceValidationBasic:
    """Test basic price validation (NaN, Inf, <= 0)."""

    def test_valid_price(self):
        """Valid positive price should pass."""
        result = Mediator._validate_critical_price(100.0, "test_price")
        assert result == 100.0

    def test_none_price_raises(self):
        """None price should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid test_price: None"):
            Mediator._validate_critical_price(None, "test_price")

    def test_nan_price_raises(self):
        """NaN price should raise ValueError."""
        with pytest.raises(ValueError, match="NaN \\(Not a Number\\)"):
            Mediator._validate_critical_price(float('nan'), "test_price")

    def test_inf_price_raises(self):
        """Infinity price should raise ValueError."""
        with pytest.raises(ValueError, match="positive infinity"):
            Mediator._validate_critical_price(float('inf'), "test_price")

    def test_neg_inf_price_raises(self):
        """Negative infinity price should raise ValueError."""
        with pytest.raises(ValueError, match="negative infinity"):
            Mediator._validate_critical_price(float('-inf'), "test_price")

    def test_zero_price_raises(self):
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            Mediator._validate_critical_price(0.0, "test_price")

    def test_negative_price_raises(self):
        """Negative price should raise ValueError."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            Mediator._validate_critical_price(-100.0, "test_price")

    def test_very_small_positive_price(self):
        """Very small but positive price should pass."""
        result = Mediator._validate_critical_price(0.0001, "test_price")
        assert result == 0.0001


class TestPriceSpikeDetection:
    """Test price spike detection functionality."""

    def test_no_spike_within_threshold(self):
        """Price change within threshold should pass."""
        # 30% increase (within 50% threshold)
        prev_price = 100.0
        current_price = 130.0
        result = Mediator._validate_critical_price(
            current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
        )
        assert result == current_price

    def test_no_spike_decrease_within_threshold(self):
        """Price decrease within threshold should pass."""
        # 30% decrease (within 50% threshold)
        prev_price = 100.0
        current_price = 70.0
        result = Mediator._validate_critical_price(
            current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
        )
        assert result == current_price

    def test_spike_increase_exceeds_threshold(self):
        """Price increase exceeding threshold should raise ValueError."""
        # 60% increase (exceeds 50% threshold)
        prev_price = 100.0
        current_price = 160.0
        with pytest.raises(ValueError, match="price spike detected.*60\\.00%.*increase"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_spike_decrease_exceeds_threshold(self):
        """Price decrease exceeding threshold should raise ValueError."""
        # 60% decrease (exceeds 50% threshold)
        prev_price = 100.0
        current_price = 40.0
        with pytest.raises(ValueError, match="price spike detected.*60\\.00%.*decrease"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_flash_crash_detected(self):
        """Simulated flash crash (90% drop) should be detected."""
        prev_price = 50000.0  # BTC at $50k
        current_price = 5000.0  # Drops to $5k (90% crash)
        with pytest.raises(ValueError, match="price spike detected.*90\\.00%.*decrease"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_flash_pump_detected(self):
        """Simulated flash pump (100% increase) should be detected."""
        prev_price = 50000.0  # BTC at $50k
        current_price = 100000.0  # Jumps to $100k (100% pump)
        with pytest.raises(ValueError, match="price spike detected.*100\\.00%.*increase"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_exactly_at_threshold(self):
        """Price change exactly at threshold should pass."""
        # Exactly 50% increase (at threshold)
        prev_price = 100.0
        current_price = 150.0
        result = Mediator._validate_critical_price(
            current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
        )
        assert result == current_price

    def test_barely_over_threshold(self):
        """Price change barely over threshold should raise."""
        # 50.01% increase (just over threshold)
        prev_price = 100.0
        current_price = 150.01
        with pytest.raises(ValueError, match="price spike detected"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_custom_threshold(self):
        """Custom spike threshold should be respected."""
        # 15% increase with 10% threshold
        prev_price = 100.0
        current_price = 115.0
        with pytest.raises(ValueError, match="price spike detected.*15\\.00%"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.1
            )

    def test_no_prev_price_skips_spike_check(self):
        """When prev_price is None, spike detection should be skipped."""
        # Large price should pass without prev_price comparison
        result = Mediator._validate_critical_price(
            1000000.0, "test_price", prev_price=None
        )
        assert result == 1000000.0

    def test_zero_prev_price_skips_spike_check(self):
        """When prev_price is 0, spike detection should be skipped."""
        # Should not raise even with large current price
        result = Mediator._validate_critical_price(
            1000000.0, "test_price", prev_price=0.0
        )
        assert result == 1000000.0


class TestPriceValidationEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_string_to_float_conversion(self):
        """String representing valid number should be converted."""
        result = Mediator._validate_critical_price("100.5", "test_price")
        assert result == 100.5

    def test_invalid_string_raises(self):
        """String that can't be converted to float should raise."""
        with pytest.raises(ValueError, match="cannot convert str to float"):
            Mediator._validate_critical_price("not_a_number", "test_price")

    def test_very_large_price_no_spike(self):
        """Very large price without previous reference should pass."""
        # BTC at $1M (hypothetical future price)
        result = Mediator._validate_critical_price(1000000.0, "test_price")
        assert result == 1000000.0

    def test_very_small_price_change(self):
        """Very small price change should pass."""
        # 0.01% change
        prev_price = 100.0
        current_price = 100.01
        result = Mediator._validate_critical_price(
            current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
        )
        assert result == current_price

    def test_error_message_includes_param_name(self):
        """Error message should include the parameter name."""
        with pytest.raises(ValueError, match="Invalid mark_price"):
            Mediator._validate_critical_price(None, "mark_price")

    def test_error_message_includes_diagnostic_info(self):
        """Spike error should include diagnostic information."""
        prev_price = 100.0
        current_price = 200.0
        with pytest.raises(
            ValueError,
            match=(
                r"Previous price: 100\.00.*"
                r"Current price: 200\.00.*"
                r"Maximum allowed change: 50\.0%"
            )
        ):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )


class TestRealWorldScenarios:
    """Test real-world cryptocurrency scenarios."""

    def test_normal_4h_volatility(self):
        """Normal 4h crypto volatility (5-15%) should pass."""
        scenarios = [
            (50000.0, 52500.0),  # 5% increase
            (50000.0, 55000.0),  # 10% increase
            (50000.0, 57500.0),  # 15% increase
            (50000.0, 47500.0),  # 5% decrease
            (50000.0, 45000.0),  # 10% decrease
            (50000.0, 42500.0),  # 15% decrease
        ]

        for prev, curr in scenarios:
            result = Mediator._validate_critical_price(
                curr, "test_price", prev_price=prev, max_spike_pct=0.5
            )
            assert result == curr

    def test_extreme_but_acceptable_4h_volatility(self):
        """Extreme but within threshold 4h volatility (20-40%) should pass."""
        scenarios = [
            (50000.0, 60000.0),  # 20% increase
            (50000.0, 65000.0),  # 30% increase
            (50000.0, 70000.0),  # 40% increase
            (50000.0, 40000.0),  # 20% decrease
            (50000.0, 35000.0),  # 30% decrease
            (50000.0, 30000.0),  # 40% decrease
        ]

        for prev, curr in scenarios:
            result = Mediator._validate_critical_price(
                curr, "test_price", prev_price=prev, max_spike_pct=0.5
            )
            assert result == curr

    def test_historical_flash_crash_binance_2021(self):
        """Binance flash crash scenario (87% drop) should be rejected."""
        # BTC flash crashed from ~$60k to ~$8k on Binance (May 2021)
        prev_price = 60000.0
        current_price = 8000.0  # 87% drop
        with pytest.raises(ValueError, match="price spike detected"):
            Mediator._validate_critical_price(
                current_price, "test_price", prev_price=prev_price, max_spike_pct=0.5
            )

    def test_stablecoin_minor_fluctuation(self):
        """Stablecoin minor fluctuation should pass."""
        # USDT fluctuates by 0.1-0.5%
        scenarios = [
            (1.0, 1.001),    # 0.1% increase
            (1.0, 1.005),    # 0.5% increase
            (1.0, 0.999),    # 0.1% decrease
            (1.0, 0.995),    # 0.5% decrease
        ]

        for prev, curr in scenarios:
            result = Mediator._validate_critical_price(
                curr, "test_price", prev_price=prev, max_spike_pct=0.5
            )
            assert result == curr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
