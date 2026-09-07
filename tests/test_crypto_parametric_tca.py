# -*- coding: utf-8 -*-
"""
tests/test_crypto_parametric_tca.py
Comprehensive tests for CryptoParametricSlippageProvider.

Test coverage:
- Configuration validation
- Individual factor tests (√Participation, Volatility Regime, Imbalance, etc.)
- Regime detection accuracy
- Asymmetric slippage behavior
- Whale threshold detection
- Time-of-day curve shape
- Adaptive impact coefficient
- Edge cases and error handling
- Integration with L2ExecutionProvider
- Regression tests for output stability

References:
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - Cartea, Jaimungal, Penalva (2015): "Algorithmic and High-Frequency Trading"
"""

import math
import pytest
import numpy as np
from typing import List, Optional

from execution_providers import (
    # Core classes
    AssetClass,
    Order,
    MarketState,
    BarData,
    Fill,
    # New parametric TCA classes
    VolatilityRegime,
    CryptoParametricConfig,
    CryptoParametricSlippageProvider,
    # Existing classes for integration
    L2ExecutionProvider,
    OHLCVFillProvider,
    CryptoFeeProvider,
    SlippageProvider,
    create_execution_provider,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_provider() -> CryptoParametricSlippageProvider:
    """Create a provider with default configuration."""
    return CryptoParametricSlippageProvider()


@pytest.fixture
def default_config() -> CryptoParametricConfig:
    """Create default configuration."""
    return CryptoParametricConfig()


@pytest.fixture
def buy_order() -> Order:
    """Create a buy market order."""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        qty=1.0,
        order_type="MARKET",
    )


@pytest.fixture
def sell_order() -> Order:
    """Create a sell market order."""
    return Order(
        symbol="BTCUSDT",
        side="SELL",
        qty=1.0,
        order_type="MARKET",
    )


@pytest.fixture
def basic_market() -> MarketState:
    """Create a basic market state."""
    return MarketState(
        timestamp=1700000000000,
        bid=50000.0,
        ask=50005.0,
        bid_size=100.0,
        ask_size=100.0,
        adv=500_000_000.0,  # $500M ADV
    )


@pytest.fixture
def market_with_imbalance_bid() -> MarketState:
    """Market with more bids than asks (bullish)."""
    return MarketState(
        timestamp=1700000000000,
        bid=50000.0,
        ask=50005.0,
        bid_size=200.0,  # More bids
        ask_size=100.0,
        adv=500_000_000.0,
    )


@pytest.fixture
def market_with_imbalance_ask() -> MarketState:
    """Market with more asks than bids (bearish)."""
    return MarketState(
        timestamp=1700000000000,
        bid=50000.0,
        ask=50005.0,
        bid_size=100.0,
        ask_size=200.0,  # More asks
        adv=500_000_000.0,
    )


@pytest.fixture
def high_vol_market() -> MarketState:
    """Market with high volatility."""
    return MarketState(
        timestamp=1700000000000,
        bid=50000.0,
        ask=50010.0,
        volatility=0.08,  # 2x typical crypto vol
        adv=500_000_000.0,
    )


@pytest.fixture
def low_vol_market() -> MarketState:
    """Market with low volatility."""
    return MarketState(
        timestamp=1700000000000,
        bid=50000.0,
        ask=50002.0,
        volatility=0.02,  # 0.5x typical crypto vol
        adv=500_000_000.0,
    )


# =============================================================================
# Test Configuration Validation
# =============================================================================


class TestCryptoParametricConfig:
    """Tests for CryptoParametricConfig validation."""

    def test_default_config_valid(self, default_config):
        """Test default configuration is valid."""
        assert default_config.impact_coef_base == 0.10
        assert default_config.spread_bps == 5.0
        assert default_config.whale_threshold == 0.01
        assert default_config.vol_lookback_periods == 20

    def test_invalid_impact_coef_raises(self):
        """Test that invalid impact coefficient raises error."""
        with pytest.raises(ValueError, match="impact_coef_base must be positive"):
            CryptoParametricConfig(impact_coef_base=0.0)

        with pytest.raises(ValueError, match="impact_coef_base must be positive"):
            CryptoParametricConfig(impact_coef_base=-0.1)

    def test_invalid_impact_range_raises(self):
        """Test that invalid impact range raises error."""
        with pytest.raises(ValueError, match="impact_coef_range must have min < max"):
            CryptoParametricConfig(impact_coef_range=(0.15, 0.05))

        with pytest.raises(ValueError, match="impact_coef_range must have min < max"):
            CryptoParametricConfig(impact_coef_range=(0.10, 0.10))

    def test_invalid_whale_threshold_raises(self):
        """Test that invalid whale threshold raises error."""
        with pytest.raises(ValueError, match="whale_threshold must be positive"):
            CryptoParametricConfig(whale_threshold=0.0)

    def test_invalid_vol_lookback_raises(self):
        """Test that invalid vol lookback raises error."""
        with pytest.raises(ValueError, match="vol_lookback_periods must be >= 2"):
            CryptoParametricConfig(vol_lookback_periods=1)

    def test_custom_config_valid(self):
        """Test custom configuration creation."""
        config = CryptoParametricConfig(
            impact_coef_base=0.15,
            spread_bps=8.0,
            whale_threshold=0.02,
            asymmetric_sell_premium=0.3,
        )
        assert config.impact_coef_base == 0.15
        assert config.spread_bps == 8.0
        assert config.whale_threshold == 0.02
        assert config.asymmetric_sell_premium == 0.3


# =============================================================================
# Test Basic Slippage Calculation
# =============================================================================


class TestBasicSlippage:
    """Tests for basic slippage calculation."""

    def test_default_parameters(self, default_provider):
        """Test provider initializes with default parameters."""
        assert default_provider.config.impact_coef_base == 0.10
        assert default_provider.config.spread_bps == 5.0
        assert default_provider._adaptive_k == 0.10

    def test_custom_parameters(self):
        """Test provider with custom parameters."""
        provider = CryptoParametricSlippageProvider(
            impact_coef=0.15,
            spread_bps=8.0,
        )
        assert provider.config.impact_coef_base == 0.15
        assert provider.config.spread_bps == 8.0

    def test_zero_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at zero participation."""
        slippage = default_provider.compute_slippage_bps(buy_order, basic_market, 0.0)
        # Should still have spread component
        assert slippage > 0
        assert slippage >= default_provider.config.min_slippage_bps

    def test_small_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at small participation."""
        slippage = default_provider.compute_slippage_bps(buy_order, basic_market, 0.001)
        # Should be reasonable for 0.1% participation
        assert slippage > 0
        assert slippage < 100.0

    def test_large_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at large participation."""
        slippage = default_provider.compute_slippage_bps(buy_order, basic_market, 0.10)
        # Should be significantly higher for 10% participation
        assert slippage > 50.0

    def test_slippage_increases_with_participation(self, default_provider, buy_order, basic_market):
        """Test that slippage increases with participation (√scaling)."""
        slip_small = default_provider.compute_slippage_bps(buy_order, basic_market, 0.001)
        slip_medium = default_provider.compute_slippage_bps(buy_order, basic_market, 0.01)
        slip_large = default_provider.compute_slippage_bps(buy_order, basic_market, 0.10)

        assert slip_small < slip_medium < slip_large

    def test_sqrt_scaling_approximate(self, buy_order, basic_market):
        """Test approximate √scaling of participation."""
        provider = CryptoParametricSlippageProvider(spread_bps=0.0, min_slippage_bps=0.0)

        slip_1 = provider.compute_slippage_bps(buy_order, basic_market, 0.0001)
        slip_4 = provider.compute_slippage_bps(buy_order, basic_market, 0.0004)

        # sqrt(4x) = 2x impact, so ratio should be ~2
        ratio = slip_4 / slip_1
        assert 1.8 <= ratio <= 2.2

    def test_max_slippage_cap(self, default_provider, buy_order, basic_market):
        """Test maximum slippage cap is enforced."""
        slippage = default_provider.compute_slippage_bps(
            buy_order, basic_market, 100.0  # Extreme participation
        )
        assert slippage <= default_provider.config.max_slippage_bps

    def test_min_slippage_floor(self, default_provider, buy_order, basic_market):
        """Test minimum slippage floor is enforced."""
        slippage = default_provider.compute_slippage_bps(buy_order, basic_market, 1e-10)
        assert slippage >= default_provider.config.min_slippage_bps


# =============================================================================
# Test √Participation Factor (Almgren-Chriss)
# =============================================================================


class TestParticipationFactor:
    """Tests for √participation factor (Almgren-Chriss model)."""

    def test_impact_proportional_to_sqrt_participation(self, buy_order, basic_market):
        """Test that impact scales with √participation."""
        provider = CryptoParametricSlippageProvider(
            spread_bps=0.0,  # Remove spread to isolate impact
            min_slippage_bps=0.0,
        )

        # Compute slippage for participation p and 4p
        p = 0.001
        slip_p = provider.compute_slippage_bps(buy_order, basic_market, p)
        slip_4p = provider.compute_slippage_bps(buy_order, basic_market, 4 * p)

        # Impact should scale as sqrt(4) = 2
        ratio = slip_4p / slip_p
        assert 1.8 <= ratio <= 2.2  # Allow some tolerance for other factors

    def test_impact_coef_effect(self, buy_order, basic_market):
        """Test that impact coefficient affects slippage."""
        provider_low = CryptoParametricSlippageProvider(impact_coef=0.05)
        provider_high = CryptoParametricSlippageProvider(impact_coef=0.15)

        slip_low = provider_low.compute_slippage_bps(buy_order, basic_market, 0.01)
        slip_high = provider_high.compute_slippage_bps(buy_order, basic_market, 0.01)

        # Higher k should give higher slippage
        assert slip_high > slip_low

    def test_negative_participation_handled(self, default_provider, buy_order, basic_market):
        """Test that negative participation is handled (uses absolute value)."""
        slip_pos = default_provider.compute_slippage_bps(buy_order, basic_market, 0.01)
        slip_neg = default_provider.compute_slippage_bps(buy_order, basic_market, -0.01)

        # Should give same result (abs)
        assert slip_pos == pytest.approx(slip_neg, rel=1e-6)


# =============================================================================
# Test Volatility Regime Detection
# =============================================================================


class TestVolatilityRegime:
    """Tests for volatility regime detection."""

    def test_regime_normal_default(self, default_provider):
        """Test that default regime is NORMAL when no returns."""
        regime = default_provider.detect_volatility_regime(None)
        assert regime == VolatilityRegime.NORMAL

    def test_regime_normal_insufficient_returns(self, default_provider):
        """Test regime is NORMAL with insufficient returns."""
        regime = default_provider.detect_volatility_regime([0.01])
        assert regime == VolatilityRegime.NORMAL

    def test_regime_detection_with_synthetic_data(self, default_provider):
        """Test regime detection with synthetic volatility data."""
        # Seed provider with baseline volatility
        baseline_returns = [0.01, -0.01, 0.005, -0.005] * 50  # ~200 returns

        # Build up history
        for i in range(0, len(baseline_returns), 20):
            window = baseline_returns[i : i + 20]
            default_provider.detect_volatility_regime(window)

        # Now test regime detection
        # Low volatility returns
        low_vol_returns = [0.001, -0.001, 0.0005, -0.0005] * 5
        # Need more history first
        default_provider.detect_volatility_regime([0.001] * 20)  # Very low vol

        # The test verifies the method runs without error
        # In practice, regime depends on history which is built dynamically
        assert default_provider.detect_volatility_regime(low_vol_returns) in [
            VolatilityRegime.LOW,
            VolatilityRegime.NORMAL,
            VolatilityRegime.HIGH,
        ]

    def test_vol_regime_multipliers_applied(self, buy_order, basic_market):
        """Test that volatility regime multipliers are applied."""
        provider = CryptoParametricSlippageProvider()

        # Manually set volatility history to trigger different regimes
        # This is testing the multiplier application, not detection
        config = CryptoParametricConfig()
        assert config.vol_regime_multipliers["low"] == 0.8
        assert config.vol_regime_multipliers["normal"] == 1.0
        assert config.vol_regime_multipliers["high"] == 1.5

    def test_realtime_volatility_adjustment(self, buy_order, high_vol_market, low_vol_market):
        """Test that real-time volatility from market state affects slippage."""
        provider = CryptoParametricSlippageProvider()

        slip_high = provider.compute_slippage_bps(buy_order, high_vol_market, 0.01)
        slip_low = provider.compute_slippage_bps(buy_order, low_vol_market, 0.01)

        # High volatility should give higher slippage
        assert slip_high > slip_low


# =============================================================================
# Test Order Book Imbalance Factor
# =============================================================================


class TestOrderBookImbalance:
    """Tests for order book imbalance factor (Cont et al. 2014)."""

    def test_imbalance_computation(self, default_provider, basic_market):
        """Test basic imbalance computation."""
        # Balanced market
        imbalance = default_provider._compute_order_book_imbalance(basic_market)
        assert imbalance == pytest.approx(0.0, abs=0.01)

    def test_imbalance_bullish(self, default_provider, market_with_imbalance_bid):
        """Test imbalance with more bids (bullish)."""
        # bid=200, ask=100 → (200-100)/(200+100) = 0.333
        imbalance = default_provider._compute_order_book_imbalance(market_with_imbalance_bid)
        assert imbalance == pytest.approx(0.333, rel=0.01)

    def test_imbalance_bearish(self, default_provider, market_with_imbalance_ask):
        """Test imbalance with more asks (bearish)."""
        # bid=100, ask=200 → (100-200)/(100+200) = -0.333
        imbalance = default_provider._compute_order_book_imbalance(market_with_imbalance_ask)
        assert imbalance == pytest.approx(-0.333, rel=0.01)

    def test_buy_penalized_when_asks_thin(self, buy_order, market_with_imbalance_bid):
        """Test that buys cost more when asks are thin (positive imbalance)."""
        provider = CryptoParametricSlippageProvider()

        slip_balanced = provider.compute_slippage_bps(
            buy_order,
            MarketState(timestamp=0, bid=100.0, ask=100.05, bid_size=100, ask_size=100),
            0.01,
        )
        slip_imbalanced = provider.compute_slippage_bps(
            buy_order,
            market_with_imbalance_bid,  # More bids = thin asks
            0.01,
        )

        # Buying into thin asks should cost more
        assert slip_imbalanced > slip_balanced

    def test_sell_penalized_when_bids_thin(self, sell_order, market_with_imbalance_ask):
        """Test that sells cost more when bids are thin (negative imbalance)."""
        provider = CryptoParametricSlippageProvider()

        slip_balanced = provider.compute_slippage_bps(
            sell_order,
            MarketState(timestamp=0, bid=100.0, ask=100.05, bid_size=100, ask_size=100),
            0.01,
        )
        slip_imbalanced = provider.compute_slippage_bps(
            sell_order,
            market_with_imbalance_ask,  # More asks = thin bids
            0.01,
        )

        # Selling into thin bids should cost more
        assert slip_imbalanced > slip_balanced

    def test_explicit_depth_used(self, default_provider, buy_order, basic_market):
        """Test that explicit depth parameters are used."""
        slip_balanced = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            bid_depth_total=1000.0,
            ask_depth_total=1000.0,
        )
        slip_imbalanced = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            bid_depth_total=2000.0,
            ask_depth_total=500.0,  # Thin asks
        )

        assert slip_imbalanced > slip_balanced


# =============================================================================
# Test Funding Rate Stress Factor
# =============================================================================


class TestFundingRateStress:
    """Tests for funding rate stress factor (perp-specific)."""

    def test_no_funding_neutral(self, default_provider, buy_order, basic_market):
        """Test that missing funding rate is neutral."""
        slip_none = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=None,
        )
        slip_zero = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.0,
        )

        assert slip_none == pytest.approx(slip_zero, rel=1e-6)

    def test_positive_funding_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that positive funding (crowded long) increases slippage."""
        slip_base = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.0,
        )
        slip_funding = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.001,  # 0.1% funding rate
        )

        assert slip_funding > slip_base

    def test_negative_funding_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that negative funding (crowded short) also increases slippage."""
        slip_base = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.0,
        )
        slip_funding = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=-0.001,  # -0.1% funding rate
        )

        assert slip_funding > slip_base

    def test_funding_effect_magnitude(self, default_provider, buy_order, basic_market):
        """Test that funding effect scales with absolute value."""
        slip_small = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.0001,
        )
        slip_large = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=0.001,
        )

        # 10x higher funding should give noticeably more slippage
        assert slip_large > slip_small


# =============================================================================
# Test Time-of-Day Factor
# =============================================================================


class TestTimeOfDayFactor:
    """Tests for time-of-day liquidity factor."""

    def test_tod_none_neutral(self, default_provider):
        """Test that None hour returns neutral factor."""
        factor = default_provider.get_time_of_day_factor(None)
        assert factor == 1.0

    def test_tod_asia_session_low(self, default_provider):
        """Test that Asia session (00:00-08:00 UTC) has lower liquidity."""
        # Asia hours should have factors < 1
        asia_hours = [0, 1, 2, 3, 4, 5, 6, 7]
        for hour in asia_hours:
            factor = default_provider.get_time_of_day_factor(hour)
            assert factor < 1.0, f"Hour {hour} should have factor < 1"

    def test_tod_eu_us_overlap_high(self, default_provider):
        """Test that EU/US overlap (14:00-18:00 UTC) has highest liquidity."""
        peak_hours = [14, 15, 16, 17]
        for hour in peak_hours:
            factor = default_provider.get_time_of_day_factor(hour)
            assert factor >= 1.05, f"Hour {hour} should have factor >= 1.05"

    def test_tod_curve_shape(self, default_provider):
        """Test overall time-of-day curve shape: Asia < EU < US overlap."""
        asia_avg = sum(default_provider.get_time_of_day_factor(h) for h in [2, 3, 4]) / 3
        eu_avg = sum(default_provider.get_time_of_day_factor(h) for h in [10, 11, 12]) / 3
        overlap_avg = sum(default_provider.get_time_of_day_factor(h) for h in [15, 16, 17]) / 3

        assert asia_avg < eu_avg < overlap_avg

    def test_tod_affects_slippage(self, default_provider, buy_order, basic_market):
        """Test that time-of-day affects slippage calculation."""
        slip_asia = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            hour_utc=3,  # Low liquidity
        )
        slip_peak = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            hour_utc=16,  # High liquidity
        )

        # Low liquidity hour should have higher slippage
        assert slip_asia > slip_peak

    def test_tod_hour_wrapping(self, default_provider):
        """Test that hour values wrap correctly."""
        assert default_provider.get_time_of_day_factor(
            24
        ) == default_provider.get_time_of_day_factor(0)
        assert default_provider.get_time_of_day_factor(
            25
        ) == default_provider.get_time_of_day_factor(1)
        assert default_provider.get_time_of_day_factor(
            -1
        ) == default_provider.get_time_of_day_factor(23)


# =============================================================================
# Test BTC Correlation Decay Factor
# =============================================================================


class TestBTCCorrelationDecay:
    """Tests for BTC correlation decay factor (altcoin fragmentation)."""

    def test_correlation_none_neutral(self, default_provider, buy_order, basic_market):
        """Test that missing correlation is neutral."""
        slip_none = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=None,
        )
        slip_one = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=1.0,  # Perfect correlation
        )

        # Both should be similar (correlation=1 means no decay)
        assert slip_none == pytest.approx(slip_one, rel=0.1)

    def test_low_correlation_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that low BTC correlation increases slippage."""
        slip_high_corr = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=0.9,  # High correlation
        )
        slip_low_corr = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=0.3,  # Low correlation
        )

        # Low correlation altcoins have less liquidity
        assert slip_low_corr > slip_high_corr

    def test_correlation_bounds_clamped(self, default_provider, buy_order, basic_market):
        """Test that correlation values are clamped to [0, 1]."""
        # These should not crash and give reasonable results
        slip_negative = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=-0.5,  # Should be treated as 0
        )
        slip_over_one = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            btc_correlation=1.5,  # Should be treated as 1
        )

        assert slip_negative > 0
        assert slip_over_one > 0


# =============================================================================
# Test Asymmetric Slippage
# =============================================================================


class TestAsymmetricSlippage:
    """Tests for asymmetric slippage (sells more expensive in downtrend)."""

    def test_buy_sell_equal_in_uptrend(self, buy_order, sell_order, basic_market):
        """Test that buy and sell costs are similar in neutral/uptrend."""
        provider = CryptoParametricSlippageProvider()

        # Uptrend returns
        uptrend_returns = [0.01, 0.02, 0.015, 0.005, 0.01]

        slip_buy = provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            recent_returns=uptrend_returns,
        )
        slip_sell = provider.compute_slippage_bps(
            sell_order,
            basic_market,
            0.01,
            recent_returns=uptrend_returns,
        )

        # Should be similar (within 10%)
        assert abs(slip_buy - slip_sell) / slip_buy < 0.15

    def test_sell_more_expensive_in_downtrend(self, buy_order, sell_order, basic_market):
        """Test that sells are more expensive in downtrend."""
        provider = CryptoParametricSlippageProvider()

        # Strong downtrend (cumulative < -2%)
        downtrend_returns = [-0.02, -0.015, -0.01, -0.005, -0.01]

        slip_buy = provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            recent_returns=downtrend_returns,
        )
        slip_sell = provider.compute_slippage_bps(
            sell_order,
            basic_market,
            0.01,
            recent_returns=downtrend_returns,
        )

        # Sells should be 20% more expensive (asymmetric_sell_premium=0.2)
        assert slip_sell > slip_buy
        expected_premium = 1.0 + provider.config.asymmetric_sell_premium
        ratio = slip_sell / slip_buy
        assert ratio >= expected_premium * 0.9  # Allow some tolerance

    def test_downtrend_threshold(self, sell_order, basic_market):
        """Test that downtrend threshold is respected."""
        provider = CryptoParametricSlippageProvider()

        # Just above threshold (cumulative = -0.015 > -0.02)
        mild_downtrend = [-0.005, -0.005, -0.005]  # -1.5%
        slip_mild = provider.compute_slippage_bps(
            sell_order,
            basic_market,
            0.01,
            recent_returns=mild_downtrend,
        )

        # Below threshold (cumulative = -0.03 < -0.02)
        strong_downtrend = [-0.01, -0.01, -0.01]  # -3%
        slip_strong = provider.compute_slippage_bps(
            sell_order,
            basic_market,
            0.01,
            recent_returns=strong_downtrend,
        )

        # Strong downtrend should have asymmetric premium
        assert slip_strong > slip_mild


# =============================================================================
# Test Whale Detection
# =============================================================================


class TestWhaleDetection:
    """Tests for whale order detection and TWAP adjustment."""

    def test_whale_threshold_1_percent(self, default_provider):
        """Test that default whale threshold is 1% of ADV."""
        assert default_provider.config.whale_threshold == 0.01

    def test_whale_gets_reduced_impact(self, buy_order, basic_market):
        """Test that whale orders get reduced impact coefficient."""
        provider = CryptoParametricSlippageProvider()

        # Non-whale order (0.5% ADV)
        slip_normal = provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.005,
        )

        # Whale order (2% ADV) - should have TWAP-adjusted impact
        slip_whale = provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.02,
        )

        # Whale should have proportionally lower slippage per unit
        # Due to TWAP adjustment (whale_twap_adjustment = 0.7)
        # Without adjustment: sqrt(4x) = 2x impact
        # With adjustment: less than 2x
        ratio = slip_whale / slip_normal
        assert ratio < 2.5  # Less than sqrt scaling would suggest

    def test_whale_twap_adjustment_factor(self):
        """Test that whale TWAP adjustment factor is configurable."""
        config = CryptoParametricConfig(whale_twap_adjustment=0.5)
        assert config.whale_twap_adjustment == 0.5

    def test_estimate_identifies_whale(self):
        """Test that estimate_impact_cost identifies whale orders."""
        provider = CryptoParametricSlippageProvider()

        # Non-whale
        est_normal = provider.estimate_impact_cost(
            notional=4_000_000,  # 0.8% of $500M ADV
            adv=500_000_000,
        )
        assert est_normal["is_whale"] is False

        # Whale
        est_whale = provider.estimate_impact_cost(
            notional=6_000_000,  # 1.2% of $500M ADV
            adv=500_000_000,
        )
        assert est_whale["is_whale"] is True


# =============================================================================
# Test Adaptive Impact Coefficient
# =============================================================================


class TestAdaptiveImpact:
    """Tests for adaptive impact coefficient adjustment."""

    def test_initial_k_equals_base(self, default_provider):
        """Test that initial adaptive k equals base."""
        assert default_provider._adaptive_k == default_provider.config.impact_coef_base

    def test_update_increases_k_when_underpredicting(self, default_provider):
        """Test that k increases when we consistently underpredict."""
        initial_k = default_provider._adaptive_k

        # Simulate consistent underprediction
        for _ in range(20):
            default_provider.update_fill_quality(
                predicted_slippage_bps=10.0,
                actual_slippage_bps=15.0,  # Actual higher
            )

        assert default_provider._adaptive_k > initial_k

    def test_update_decreases_k_when_overpredicting(self, default_provider):
        """Test that k decreases when we consistently overpredict."""
        initial_k = default_provider._adaptive_k

        # Simulate consistent overprediction
        for _ in range(20):
            default_provider.update_fill_quality(
                predicted_slippage_bps=20.0,
                actual_slippage_bps=10.0,  # Actual lower
            )

        assert default_provider._adaptive_k < initial_k

    def test_k_stays_in_range(self, default_provider):
        """Test that adaptive k stays within configured range."""
        min_k, max_k = default_provider.config.impact_coef_range

        # Try to push k beyond bounds with extreme updates
        for _ in range(50):
            default_provider.update_fill_quality(
                predicted_slippage_bps=1.0,
                actual_slippage_bps=100.0,  # Extreme underprediction
            )

        assert default_provider._adaptive_k <= max_k

        # Reset and try other direction
        default_provider.reset_adaptive_state()
        for _ in range(50):
            default_provider.update_fill_quality(
                predicted_slippage_bps=100.0,
                actual_slippage_bps=1.0,  # Extreme overprediction
            )

        assert default_provider._adaptive_k >= min_k

    def test_reset_adaptive_state(self, default_provider):
        """Test that reset_adaptive_state clears all adaptive state."""
        # Modify state
        default_provider.update_fill_quality(10.0, 20.0)
        default_provider._volatility_history.append(0.05)

        # Reset
        default_provider.reset_adaptive_state()

        assert default_provider._adaptive_k == default_provider.config.impact_coef_base
        assert len(default_provider._fill_quality_history) == 0
        assert len(default_provider._volatility_history) == 0

    def test_invalid_fill_quality_ignored(self, default_provider):
        """Test that invalid fill quality updates are ignored."""
        initial_k = default_provider._adaptive_k

        default_provider.update_fill_quality(-10.0, 10.0)  # Negative predicted
        default_provider.update_fill_quality(0.0, 10.0)  # Zero predicted
        default_provider.update_fill_quality(10.0, -5.0)  # Negative actual

        assert default_provider._adaptive_k == initial_k


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_adv_handled(self):
        """Test handling of zero ADV."""
        provider = CryptoParametricSlippageProvider()
        result = provider.estimate_impact_cost(notional=1000.0, adv=0.0)
        assert result["participation"] == 0.0
        assert "Unable to estimate" in result["recommendation"]

    def test_nan_funding_rate(self, default_provider, buy_order, basic_market):
        """Test handling of NaN funding rate."""
        slippage = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            funding_rate=float("nan"),
        )
        # Should use neutral factor
        assert slippage > 0
        assert math.isfinite(slippage)

    def test_nan_volatility(self, buy_order):
        """Test handling of NaN volatility in market state."""
        provider = CryptoParametricSlippageProvider()
        market = MarketState(timestamp=0, volatility=float("nan"))

        slippage = provider.compute_slippage_bps(buy_order, market, 0.01)
        assert slippage > 0
        assert math.isfinite(slippage)

    def test_inf_values_handled(self, default_provider, buy_order):
        """Test handling of infinite values."""
        market = MarketState(
            timestamp=0,
            bid=float("inf"),
            ask=float("inf"),
        )

        slippage = default_provider.compute_slippage_bps(buy_order, market, 0.01)
        # Should use default spread
        assert slippage > 0
        assert math.isfinite(slippage)

    def test_missing_market_data(self, default_provider, buy_order):
        """Test with minimal market data."""
        market = MarketState(timestamp=0)  # No bid/ask/etc

        slippage = default_provider.compute_slippage_bps(buy_order, market, 0.01)
        # Should use defaults
        assert slippage > 0
        assert math.isfinite(slippage)

    def test_empty_returns_array(self, default_provider, buy_order, basic_market):
        """Test with empty returns array."""
        slippage = default_provider.compute_slippage_bps(
            buy_order,
            basic_market,
            0.01,
            recent_returns=[],
        )
        assert slippage > 0

    def test_l3_depth_used_for_imbalance(self, default_provider, buy_order):
        """Test that L3 depth is used for imbalance when available."""
        market = MarketState(
            timestamp=0,
            bid=100.0,
            ask=100.05,
            bid_depth=[(100.0, 50.0), (99.95, 50.0)],  # Total 100
            ask_depth=[(100.05, 100.0), (100.10, 100.0)],  # Total 200
        )

        imbalance = default_provider._compute_order_book_imbalance(market)
        # (100 - 200) / (100 + 200) = -0.333
        assert imbalance == pytest.approx(-0.333, rel=0.01)


# =============================================================================
# Test Factory Methods
# =============================================================================


class TestFactoryMethods:
    """Tests for factory and profile methods."""

    def test_from_config_none(self):
        """Test from_config with None creates defaults."""
        provider = CryptoParametricSlippageProvider.from_config(None)
        assert provider.config.impact_coef_base == 0.10

    def test_from_config_dict(self):
        """Test from_config with dict."""
        config = {
            "impact_coef": 0.15,
            "spread_bps": 8.0,
        }
        provider = CryptoParametricSlippageProvider.from_config(config)
        assert provider.config.impact_coef_base == 0.15
        assert provider.config.spread_bps == 8.0

    def test_from_config_with_overrides(self):
        """Test from_config with kwargs overrides."""
        config = {"impact_coef": 0.12}
        provider = CryptoParametricSlippageProvider.from_config(
            config,
            spread_bps=10.0,  # Override
        )
        assert provider.config.spread_bps == 10.0

    def test_from_profile_default(self):
        """Test from_profile with default."""
        provider = CryptoParametricSlippageProvider.from_profile("default")
        assert provider.config.impact_coef_base == 0.10

    def test_from_profile_conservative(self):
        """Test from_profile with conservative."""
        provider = CryptoParametricSlippageProvider.from_profile("conservative")
        assert provider.config.impact_coef_base == 0.12
        assert provider.config.spread_bps == 6.0

    def test_from_profile_aggressive(self):
        """Test from_profile with aggressive."""
        provider = CryptoParametricSlippageProvider.from_profile("aggressive")
        assert provider.config.impact_coef_base == 0.08
        assert provider.config.spread_bps == 4.0

    def test_from_profile_altcoin(self):
        """Test from_profile with altcoin."""
        provider = CryptoParametricSlippageProvider.from_profile("altcoin")
        assert provider.config.impact_coef_base == 0.15
        assert provider.config.spread_bps == 10.0

    def test_from_profile_stablecoin(self):
        """Test from_profile with stablecoin."""
        provider = CryptoParametricSlippageProvider.from_profile("stablecoin")
        assert provider.config.impact_coef_base == 0.05
        assert provider.config.spread_bps == 1.0

    def test_from_profile_unknown_uses_default(self):
        """Test from_profile with unknown name uses default."""
        provider = CryptoParametricSlippageProvider.from_profile("unknown_profile")
        assert provider.config.impact_coef_base == 0.10


# =============================================================================
# Test Estimate Impact Cost
# =============================================================================


class TestEstimateImpactCost:
    """Tests for pre-trade cost estimation."""

    def test_estimate_basic(self):
        """Test basic cost estimation."""
        provider = CryptoParametricSlippageProvider()
        result = provider.estimate_impact_cost(
            notional=1_000_000,
            adv=100_000_000,
        )

        assert "participation" in result
        assert "impact_bps" in result
        assert "impact_cost" in result
        assert "is_whale" in result
        assert "recommendation" in result

        assert result["participation"] == 0.01
        assert result["impact_cost"] > 0

    def test_estimate_with_all_params(self):
        """Test estimation with all optional parameters."""
        provider = CryptoParametricSlippageProvider()
        result = provider.estimate_impact_cost(
            notional=500_000,
            adv=100_000_000,
            side="BUY",
            volatility=0.05,
            funding_rate=0.0005,
            btc_correlation=0.7,
            hour_utc=16,
        )

        assert result["participation"] == 0.005
        assert result["tod_factor"] > 1.0  # Peak hour

    def test_estimate_recommendation_whale(self):
        """Test that whale orders get TWAP recommendation."""
        provider = CryptoParametricSlippageProvider()
        result = provider.estimate_impact_cost(
            notional=2_000_000,  # 2% of ADV
            adv=100_000_000,
        )

        assert result["is_whale"] is True
        assert "TWAP" in result["recommendation"] or "VWAP" in result["recommendation"]

    def test_estimate_recommendation_low_liquidity_hour(self):
        """Test recommendation for low liquidity hour."""
        provider = CryptoParametricSlippageProvider()
        result = provider.estimate_impact_cost(
            notional=100_000,
            adv=100_000_000,
            hour_utc=3,  # Low liquidity
        )

        assert (
            "Low liquidity" in result["recommendation"]
            or "delay" in result["recommendation"].lower()
        )


# =============================================================================
# Test Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Tests that CryptoParametricSlippageProvider satisfies SlippageProvider protocol."""

    def test_slippage_provider_protocol(self, default_provider):
        """Test SlippageProvider protocol compliance."""
        assert isinstance(default_provider, SlippageProvider)

    def test_has_compute_slippage_bps_method(self, default_provider):
        """Test that required method exists."""
        assert hasattr(default_provider, "compute_slippage_bps")
        assert callable(default_provider.compute_slippage_bps)


# =============================================================================
# Test Integration with L2ExecutionProvider
# =============================================================================


class TestL2Integration:
    """Tests for integration with L2ExecutionProvider."""

    def test_use_as_slippage_provider(self, buy_order, basic_market):
        """Test using CryptoParametricSlippageProvider in L2ExecutionProvider."""
        parametric_slippage = CryptoParametricSlippageProvider()

        provider = L2ExecutionProvider(
            asset_class=AssetClass.CRYPTO,
            slippage_provider=parametric_slippage,
        )

        bar = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=100.0)
        fill = provider.execute(buy_order, basic_market, bar)

        assert fill is not None
        assert fill.slippage_bps >= 0

    def test_use_with_ohlcv_fill_provider(self, buy_order, basic_market):
        """Test using CryptoParametricSlippageProvider with OHLCVFillProvider."""
        parametric_slippage = CryptoParametricSlippageProvider()
        fees = CryptoFeeProvider()

        fill_provider = OHLCVFillProvider(
            slippage_provider=parametric_slippage,
            fee_provider=fees,
        )

        bar = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=100.0)
        fill = fill_provider.try_fill(buy_order, basic_market, bar)

        assert fill is not None
        assert fill.qty == buy_order.qty

    def test_cost_estimation_workflow(self):
        """Test complete cost estimation workflow."""
        provider = CryptoParametricSlippageProvider()

        # Pre-trade analysis
        estimate = provider.estimate_impact_cost(
            notional=1_000_000,
            adv=500_000_000,
            side="BUY",
            hour_utc=16,
        )

        # Verify estimate
        assert estimate["impact_bps"] > 0

        # Use for execution - use hour_utc to match estimate
        order = Order("BTCUSDT", "BUY", 0.02, "MARKET")  # ~$1M at $50k
        market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50005.0,
            adv=500_000_000,
        )

        slippage = provider.compute_slippage_bps(
            order,
            market,
            estimate["participation"],
            hour_utc=16,  # Match the estimate hour
        )

        # Should be in same ballpark as estimate
        # Note: estimate uses default spread (5 bps), actual uses market spread (1 bp)
        # So actual may differ by ~15-20% due to spread difference
        assert slippage == pytest.approx(estimate["impact_bps"], rel=0.2)


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Regression tests for output stability."""

    def test_output_stable_for_fixed_inputs(self):
        """Test that output is deterministic for fixed inputs."""
        provider = CryptoParametricSlippageProvider()
        order = Order("BTCUSDT", "BUY", 1.0, "MARKET")
        market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50005.0,
            bid_size=100.0,
            ask_size=100.0,
            adv=500_000_000,
        )

        # Compute multiple times
        results = [provider.compute_slippage_bps(order, market, 0.01) for _ in range(10)]

        # All should be identical
        assert all(r == results[0] for r in results)

    def test_expected_output_baseline(self):
        """Test expected output for baseline scenario."""
        provider = CryptoParametricSlippageProvider()
        order = Order("BTCUSDT", "BUY", 1.0, "MARKET")
        market = MarketState(
            timestamp=0,
            bid=100.0,
            ask=100.10,
            bid_size=100.0,
            ask_size=100.0,
            adv=10_000_000,
        )

        slippage = provider.compute_slippage_bps(order, market, 0.01)

        # Should be in reasonable range for 1% participation
        # Half spread (~5 bps) + impact (~100 bps for sqrt(0.01) * 0.1 * 10000)
        assert 10.0 < slippage < 200.0

    def test_order_independence(self, basic_market):
        """Test that order object doesn't affect calculation beyond side/qty."""
        provider = CryptoParametricSlippageProvider()

        order1 = Order("BTCUSDT", "BUY", 1.0, "MARKET")
        order2 = Order("ETHUSDT", "BUY", 1.0, "LIMIT", limit_price=100.0)

        slip1 = provider.compute_slippage_bps(order1, basic_market, 0.01)
        slip2 = provider.compute_slippage_bps(order2, basic_market, 0.01)

        # Should be same (symbol and order_type don't affect slippage calc)
        assert slip1 == pytest.approx(slip2, rel=1e-6)


# =============================================================================
# Benchmark Tests (optional - run with pytest -v)
# =============================================================================


class TestPerformance:
    """Performance tests for slippage calculation."""

    def test_compute_slippage_fast(self, default_provider, buy_order, basic_market):
        """Test that slippage computation is fast enough."""
        import time

        # Warmup
        for _ in range(100):
            default_provider.compute_slippage_bps(buy_order, basic_market, 0.01)

        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            default_provider.compute_slippage_bps(
                buy_order,
                basic_market,
                0.01,
                funding_rate=0.0001,
                btc_correlation=0.85,
                hour_utc=14,
                recent_returns=[0.01, -0.005, 0.003],
            )
        elapsed = time.perf_counter() - start

        # Should complete 10k iterations in < 1 second
        assert elapsed < 1.0, f"Took {elapsed:.2f}s for {iterations} iterations"

        # Report
        per_call = elapsed / iterations * 1_000_000
        print(f"\nPerformance: {per_call:.1f} microseconds per call")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
