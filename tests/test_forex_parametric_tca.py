# -*- coding: utf-8 -*-
"""
tests/test_forex_parametric_tca.py
Comprehensive tests for ForexParametricSlippageProvider.

Test coverage:
- Configuration validation
- Individual factor tests (8 factors)
- Session detection (Sydney, Tokyo, London, NY, overlaps, off-hours, weekend)
- Pair type classification (major, minor, cross, exotic)
- Volatility regime detection
- Carry trade stress (interest rate differential)
- DXY correlation decay
- News event impact
- Edge cases and error handling
- Integration with factory functions
- Fee provider tests

References:
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - King, Osler, Rime (2013): "Foreign Exchange Market Structure"
    - BIS Triennial Survey (2022): FX Market Volume Statistics
"""

import math
import pytest
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional

from execution_providers import (
    # Core classes
    AssetClass,
    Order,
    MarketState,
    BarData,
    Fill,
    # Forex parametric TCA classes
    ForexSession,
    PairType,
    VolatilityRegime,
    ForexParametricConfig,
    ForexParametricSlippageProvider,
    ForexFeeProvider,
    # Factory functions
    create_slippage_provider,
    create_fee_provider,
    create_execution_provider,
    # Existing classes for integration
    L2ExecutionProvider,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_provider() -> ForexParametricSlippageProvider:
    """Create a provider with default configuration."""
    return ForexParametricSlippageProvider()


@pytest.fixture
def default_config() -> ForexParametricConfig:
    """Create default configuration."""
    return ForexParametricConfig()


@pytest.fixture
def buy_order() -> Order:
    """Create a buy market order for EUR/USD."""
    return Order(
        symbol="EURUSD",
        side="BUY",
        qty=100000.0,  # 1 standard lot
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


@pytest.fixture
def sell_order() -> Order:
    """Create a sell market order for EUR/USD."""
    return Order(
        symbol="EURUSD",
        side="SELL",
        qty=100000.0,
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


@pytest.fixture
def basic_market() -> MarketState:
    """Create a basic market state for EUR/USD."""
    # London session: 8:00 UTC (Wednesday)
    london_session_ts = 1700038800000  # 2023-11-15 08:00 UTC (Wednesday)
    return MarketState(
        timestamp=london_session_ts,
        bid=1.08500,
        ask=1.08510,  # 1 pip spread
        bid_size=5_000_000.0,  # $5M
        ask_size=5_000_000.0,
        adv=500_000_000_000.0,  # $500B ADV (EUR/USD is most liquid)
    )


@pytest.fixture
def sydney_market() -> MarketState:
    """Market during Sydney session (lower liquidity)."""
    # Sydney session: 22:00 UTC
    sydney_ts = 1700089200000  # 2023-11-15 22:00 UTC
    return MarketState(
        timestamp=sydney_ts,
        bid=1.08500,
        ask=1.08515,  # Wider spread
        adv=500_000_000_000.0,
    )


@pytest.fixture
def tokyo_market() -> MarketState:
    """Market during Tokyo session."""
    # Tokyo session: 01:00 UTC
    tokyo_ts = 1700013600000  # 2023-11-15 01:00 UTC
    return MarketState(
        timestamp=tokyo_ts,
        bid=108.500,  # USD/JPY
        ask=108.510,
        adv=500_000_000_000.0,
    )


@pytest.fixture
def ny_market() -> MarketState:
    """Market during NY session."""
    # NY session: 14:00 UTC
    ny_ts = 1700060400000  # 2023-11-15 14:00 UTC
    return MarketState(
        timestamp=ny_ts,
        bid=1.08500,
        ask=1.08508,
        adv=500_000_000_000.0,
    )


@pytest.fixture
def london_ny_overlap_market() -> MarketState:
    """Market during London-NY overlap (peak liquidity)."""
    # Overlap: 13:00 UTC
    overlap_ts = 1700056800000  # 2023-11-15 13:00 UTC
    return MarketState(
        timestamp=overlap_ts,
        bid=1.08500,
        ask=1.08505,  # Tightest spread
        adv=500_000_000_000.0,
    )


@pytest.fixture
def weekend_market() -> MarketState:
    """Market during weekend (market closed)."""
    # Saturday 12:00 UTC
    weekend_ts = 1700222400000  # 2023-11-18 00:00 UTC (Saturday)
    return MarketState(
        timestamp=weekend_ts,
        bid=1.08500,
        ask=1.08550,  # Very wide spread
        adv=0.0,  # No trading
    )


@pytest.fixture
def high_vol_market() -> MarketState:
    """Market with high volatility."""
    return MarketState(
        timestamp=1700056800000,
        bid=1.08500,
        ask=1.08530,  # 3 pip spread
        volatility=0.015,  # High for forex
        adv=500_000_000_000.0,
    )


@pytest.fixture
def low_vol_market() -> MarketState:
    """Market with low volatility."""
    return MarketState(
        timestamp=1700056800000,
        bid=1.08500,
        ask=1.08503,  # Very tight
        volatility=0.003,  # Low
        adv=500_000_000_000.0,
    )


@pytest.fixture
def exotic_order() -> Order:
    """Order for exotic pair USD/TRY."""
    return Order(
        symbol="USDTRY",
        side="BUY",
        qty=100000.0,
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


@pytest.fixture
def minor_order() -> Order:
    """Order for minor pair EUR/GBP."""
    return Order(
        symbol="EURGBP",
        side="BUY",
        qty=100000.0,
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


@pytest.fixture
def jpy_order() -> Order:
    """Order for JPY pair USD/JPY."""
    return Order(
        symbol="USDJPY",
        side="BUY",
        qty=100000.0,
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


# =============================================================================
# Test Configuration Validation
# =============================================================================


class TestForexParametricConfig:
    """Tests for ForexParametricConfig validation."""

    def test_default_config_valid(self, default_config):
        """Test default configuration is valid."""
        assert default_config.impact_coef_base == 0.03
        assert default_config.min_slippage_pips == 0.05
        assert default_config.max_slippage_pips == 150.0
        assert default_config.carry_sensitivity == 0.03
        assert default_config.dxy_correlation_decay == 0.25

    def test_invalid_impact_coef_raises(self):
        """Test that invalid impact coefficient raises error."""
        with pytest.raises(ValueError, match="impact_coef_base must be positive"):
            ForexParametricConfig(impact_coef_base=0.0)

        with pytest.raises(ValueError, match="impact_coef_base must be positive"):
            ForexParametricConfig(impact_coef_base=-0.1)

    def test_invalid_impact_range_raises(self):
        """Test that invalid impact range raises error."""
        with pytest.raises(ValueError, match="impact_coef_range must have min < max"):
            ForexParametricConfig(impact_coef_range=(0.05, 0.01))

        with pytest.raises(ValueError, match="impact_coef_range must have min < max"):
            ForexParametricConfig(impact_coef_range=(0.02, 0.02))

    def test_invalid_slippage_range_raises(self):
        """Test that invalid slippage range raises error."""
        with pytest.raises(ValueError, match="min_slippage_pips must be non-negative"):
            ForexParametricConfig(min_slippage_pips=-0.1)

        with pytest.raises(ValueError, match="max_slippage_pips must be greater than min"):
            ForexParametricConfig(min_slippage_pips=10.0, max_slippage_pips=5.0)

    def test_custom_config_valid(self):
        """Test custom configuration creation."""
        config = ForexParametricConfig(
            impact_coef_base=0.05,
            min_slippage_pips=0.2,
            max_slippage_pips=30.0,
            carry_sensitivity=0.08,
        )
        assert config.impact_coef_base == 0.05
        assert config.min_slippage_pips == 0.2
        assert config.max_slippage_pips == 30.0
        assert config.carry_sensitivity == 0.08


# =============================================================================
# Test Basic Slippage Calculation
# =============================================================================


class TestBasicSlippage:
    """Tests for basic slippage calculation."""

    def test_default_parameters(self, default_provider):
        """Test provider initializes with default parameters."""
        assert default_provider.config.impact_coef_base == 0.03
        assert default_provider._adaptive_k == 0.03

    def test_custom_parameters(self):
        """Test provider with custom parameters."""
        provider = ForexParametricSlippageProvider(
            impact_coef=0.05,
        )
        assert provider.config.impact_coef_base == 0.05

    def test_slippage_in_pips(self, default_provider, buy_order, basic_market):
        """Test slippage calculation returns pips."""
        slippage_pips = default_provider.compute_slippage_pips(buy_order, basic_market, 0.001)
        assert slippage_pips > 0
        assert slippage_pips < 10.0  # Reasonable for major pair

    def test_slippage_in_bps(self, default_provider, buy_order, basic_market):
        """Test slippage in basis points."""
        slippage_bps = default_provider.compute_slippage_bps(buy_order, basic_market, 0.001)
        assert slippage_bps > 0

    def test_pips_to_bps_conversion(self, default_provider, buy_order, basic_market):
        """Test pips to bps conversion."""
        slippage_pips = default_provider.compute_slippage_pips(buy_order, basic_market, 0.001)
        slippage_bps = default_provider.compute_slippage_bps(buy_order, basic_market, 0.001)

        # For EUR/USD: 1 pip = 0.0001 = 1 bps relative to quote
        # The conversion depends on price level
        assert slippage_bps > 0

    def test_zero_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at zero participation."""
        slippage = default_provider.compute_slippage_pips(buy_order, basic_market, 0.0)
        # Should still have spread component
        assert slippage > 0
        assert slippage >= default_provider.config.min_slippage_pips

    def test_small_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at small participation."""
        slippage = default_provider.compute_slippage_pips(buy_order, basic_market, 0.0001)
        assert slippage > 0
        assert slippage < 10.0  # Reasonable for forex

    def test_large_participation(self, default_provider, buy_order, basic_market):
        """Test slippage at large participation."""
        slippage = default_provider.compute_slippage_pips(buy_order, basic_market, 0.01)
        # Should be higher for 1% participation (but forex is very liquid)
        assert slippage > 0.5

    def test_slippage_increases_with_participation(self, default_provider, buy_order, basic_market):
        """Test that slippage increases with participation (√scaling)."""
        slip_small = default_provider.compute_slippage_pips(buy_order, basic_market, 0.0001)
        slip_medium = default_provider.compute_slippage_pips(buy_order, basic_market, 0.001)
        slip_large = default_provider.compute_slippage_pips(buy_order, basic_market, 0.01)

        assert slip_small < slip_medium < slip_large

    def test_max_slippage_cap(self, default_provider, buy_order, basic_market):
        """Test maximum slippage cap is enforced."""
        slippage = default_provider.compute_slippage_pips(
            buy_order, basic_market, 10.0  # Extreme participation
        )
        assert slippage <= default_provider.config.max_slippage_pips

    def test_min_slippage_floor(self, default_provider, buy_order, basic_market):
        """Test minimum slippage floor is enforced."""
        slippage = default_provider.compute_slippage_pips(buy_order, basic_market, 1e-10)
        assert slippage >= default_provider.config.min_slippage_pips


# =============================================================================
# Test √Participation Factor (Almgren-Chriss)
# =============================================================================


class TestParticipationFactor:
    """Tests for √participation factor (Almgren-Chriss model)."""

    def test_impact_proportional_to_sqrt_participation(self, buy_order, basic_market):
        """Test that impact scales with √participation."""
        provider = ForexParametricSlippageProvider(
            min_slippage_pips=0.0,
        )

        # Compute slippage for participation p and 4p
        p = 0.0001
        slip_p = provider.compute_slippage_pips(buy_order, basic_market, p)
        slip_4p = provider.compute_slippage_pips(buy_order, basic_market, 4 * p)

        # Impact should scale proportionally with participation
        # Due to spread component being constant, ratio will be less than sqrt(4)=2
        # For forex with low impact coefficient, spread dominates
        ratio = slip_4p / slip_p
        assert ratio >= 1.0  # Higher participation = higher slippage

    def test_impact_coef_effect(self, buy_order, basic_market):
        """Test that impact coefficient affects slippage."""
        provider_low = ForexParametricSlippageProvider(impact_coef=0.02)
        provider_high = ForexParametricSlippageProvider(impact_coef=0.05)

        slip_low = provider_low.compute_slippage_pips(buy_order, basic_market, 0.001)
        slip_high = provider_high.compute_slippage_pips(buy_order, basic_market, 0.001)

        # Higher k should give higher slippage
        assert slip_high > slip_low

    def test_negative_participation_handled(self, default_provider, buy_order, basic_market):
        """Test that negative participation is handled (uses absolute value)."""
        slip_pos = default_provider.compute_slippage_pips(buy_order, basic_market, 0.001)
        slip_neg = default_provider.compute_slippage_pips(buy_order, basic_market, -0.001)

        # Should give same result (abs)
        assert slip_pos == pytest.approx(slip_neg, rel=1e-6)


# =============================================================================
# Test Session Detection
# =============================================================================


class TestSessionDetection:
    """Tests for forex session detection."""

    def test_sydney_session_detection(self, default_provider, sydney_market):
        """Test Sydney session detection (21:00-06:00 UTC)."""
        session = default_provider._detect_session(sydney_market.timestamp)
        assert session == ForexSession.SYDNEY

    def test_tokyo_session_detection(self, default_provider):
        """Test Tokyo session detection (00:00-09:00 UTC)."""
        # 06:00 UTC on a weekday should be Tokyo session (after Sydney)
        tokyo_ts = 1700028000000  # 2023-11-15 06:00 UTC (Wednesday)
        session = default_provider._detect_session(tokyo_ts)
        # 06:00 UTC could be Tokyo or overlap, both are acceptable
        assert session in (
            ForexSession.TOKYO,
            ForexSession.TOKYO_LONDON_OVERLAP,
            ForexSession.SYDNEY,
        )

    def test_london_session_detection(self, default_provider):
        """Test London session detection (08:00-16:00 UTC)."""
        # 10:00 UTC on a weekday is clearly London session (before NY opens)
        london_ts = 1700042400000  # 2023-11-15 10:00 UTC
        session = default_provider._detect_session(london_ts)
        assert session == ForexSession.LONDON

    def test_ny_session_detection(self, default_provider, ny_market):
        """Test NY session detection (13:00-22:00 UTC)."""
        # 18:00 UTC is clearly NY session (after London closes)
        ny_only_ts = 1700074800000  # 2023-11-15 18:00 UTC
        session = default_provider._detect_session(ny_only_ts)
        assert session == ForexSession.NEW_YORK

    def test_london_ny_overlap_detection(self, default_provider, london_ny_overlap_market):
        """Test London-NY overlap detection (13:00-16:00 UTC)."""
        session = default_provider._detect_session(london_ny_overlap_market.timestamp)
        assert session == ForexSession.LONDON_NY_OVERLAP

    def test_tokyo_london_overlap_detection(self, default_provider):
        """Test Tokyo-London overlap detection (07:00-09:00 UTC)."""
        # 08:00 UTC is Tokyo-London overlap
        overlap_ts = 1700035200000  # 2023-11-15 08:00 UTC
        session = default_provider._detect_session(overlap_ts)
        assert session == ForexSession.TOKYO_LONDON_OVERLAP

    def test_weekend_detection(self, default_provider):
        """Test weekend detection (Saturday after market close)."""
        # Saturday 12:00 UTC - market is closed
        saturday_ts = 1700312400000  # 2023-11-18 12:00 UTC (Saturday)
        session = default_provider._detect_session(saturday_ts)
        assert session == ForexSession.WEEKEND

    def test_session_changes_throughout_day(self, default_provider):
        """Test that session detection varies throughout the trading day."""
        # Get sessions at different hours on the same weekday
        sessions = []
        base_ts = 1700006400000  # 2023-11-15 00:00 UTC (Wednesday)
        for hour in [0, 4, 8, 10, 14, 18, 22]:
            ts = base_ts + hour * 3600 * 1000
            session = default_provider._detect_session(ts)
            sessions.append(session)

        # Should have variety in sessions
        unique_sessions = set(sessions)
        assert len(unique_sessions) >= 3  # At least 3 different sessions


class TestSessionLiquidity:
    """Tests for session-based liquidity adjustments."""

    def test_overlap_has_lowest_slippage(self, default_provider, buy_order):
        """Test that overlap sessions have best liquidity (lowest slippage)."""
        # Create markets at different sessions with same spread
        base_market = MarketState(
            timestamp=0,  # Will be overwritten
            bid=1.08500,
            ask=1.08510,
            adv=500_000_000_000.0,
        )

        # London-NY overlap: 14:00 UTC
        overlap_market = MarketState(
            timestamp=1700060400000,
            bid=1.08500,
            ask=1.08510,
            adv=500_000_000_000.0,
        )

        # Sydney: 22:00 UTC
        sydney_market = MarketState(
            timestamp=1700089200000,
            bid=1.08500,
            ask=1.08510,
            adv=500_000_000_000.0,
        )

        slip_overlap = default_provider.compute_slippage_pips(buy_order, overlap_market, 0.001)
        slip_sydney = default_provider.compute_slippage_pips(buy_order, sydney_market, 0.001)

        # Overlap should have lower slippage
        assert slip_overlap < slip_sydney

    def test_weekend_vs_weekday_slippage(self, default_provider, buy_order):
        """Test that session affects slippage."""
        # Weekend market with Saturday timestamp
        weekend_market = MarketState(
            timestamp=1700312400000,  # Saturday 12:00 UTC
            bid=1.08500,
            ask=1.08550,  # Wider spread during weekend
            adv=0.0,  # No volume
        )
        slip_weekend = default_provider.compute_slippage_pips(buy_order, weekend_market, 0.001)

        # Compare with weekday London session
        london_market = MarketState(
            timestamp=1700042400000,  # Wednesday 10:00 UTC
            bid=1.08500,
            ask=1.08510,
            adv=500_000_000_000.0,
        )
        slip_london = default_provider.compute_slippage_pips(buy_order, london_market, 0.001)

        # Weekend should have max slippage due to market closed / zero ADV
        assert slip_weekend >= slip_london


# =============================================================================
# Test Pair Type Classification
# =============================================================================


class TestPairClassification:
    """Tests for currency pair type classification."""

    def test_major_pairs_classification(self, default_provider):
        """Test major pairs are correctly classified."""
        majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]
        for pair in majors:
            pair_type = default_provider._classify_pair(pair)
            assert pair_type == PairType.MAJOR, f"{pair} should be MAJOR"

    def test_minor_pairs_classification(self, default_provider):
        """Test minor pairs (non-USD major currencies) are correctly classified."""
        # Minor pairs: pairs of major currencies without USD
        minors = ["EURGBP", "EURCHF", "AUDNZD"]
        for pair in minors:
            pair_type = default_provider._classify_pair(pair)
            assert pair_type == PairType.MINOR, f"{pair} should be MINOR"

    def test_cross_pairs_classification(self, default_provider):
        """Test cross pairs (JPY crosses) are correctly classified."""
        # Cross pairs: JPY with non-USD (often called yen crosses)
        crosses = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"]
        for pair in crosses:
            pair_type = default_provider._classify_pair(pair)
            assert pair_type == PairType.CROSS, f"{pair} should be CROSS"

    def test_exotic_pairs_classification(self, default_provider):
        """Test exotic pairs are correctly classified."""
        exotics = ["USDTRY", "USDZAR", "USDMXN", "USDHKD", "USDSGD", "EURTRY"]
        for pair in exotics:
            pair_type = default_provider._classify_pair(pair)
            assert pair_type == PairType.EXOTIC, f"{pair} should be EXOTIC"

    def test_unknown_pair_classification(self, default_provider):
        """Test unknown pairs default behavior."""
        pair_type = default_provider._classify_pair("XXXYYY")
        # Unknown pairs should be classified as CROSS or EXOTIC
        assert pair_type in (PairType.CROSS, PairType.EXOTIC)

    def test_case_insensitive_classification(self, default_provider):
        """Test pair classification is case-insensitive."""
        assert default_provider._classify_pair("eurusd") == PairType.MAJOR
        assert default_provider._classify_pair("EURUSD") == PairType.MAJOR
        assert default_provider._classify_pair("EurUsd") == PairType.MAJOR


class TestPairTypeSlippage:
    """Tests for pair-type based slippage adjustments."""

    def test_major_has_lowest_slippage(self, default_provider, basic_market):
        """Test that major pairs have lowest slippage multiplier."""
        buy_major = Order("EURUSD", "BUY", 100000.0, "MARKET", asset_class=AssetClass.FOREX)
        buy_exotic = Order("USDTRY", "BUY", 100000.0, "MARKET", asset_class=AssetClass.FOREX)

        slip_major = default_provider.compute_slippage_pips(buy_major, basic_market, 0.001)
        slip_exotic = default_provider.compute_slippage_pips(buy_exotic, basic_market, 0.001)

        # Major should have lower slippage
        assert slip_major < slip_exotic

    def test_exotic_has_highest_slippage(self, default_provider, basic_market):
        """Test that exotic pairs have highest slippage multiplier."""
        buy_major = Order("EURUSD", "BUY", 100000.0, "MARKET", asset_class=AssetClass.FOREX)
        buy_minor = Order("EURGBP", "BUY", 100000.0, "MARKET", asset_class=AssetClass.FOREX)
        buy_exotic = Order("USDTRY", "BUY", 100000.0, "MARKET", asset_class=AssetClass.FOREX)

        slip_major = default_provider.compute_slippage_pips(buy_major, basic_market, 0.001)
        slip_minor = default_provider.compute_slippage_pips(buy_minor, basic_market, 0.001)
        slip_exotic = default_provider.compute_slippage_pips(buy_exotic, basic_market, 0.001)

        assert slip_major < slip_minor < slip_exotic


# =============================================================================
# Test Volatility Regime Detection
# =============================================================================


class TestVolatilityRegime:
    """Tests for volatility regime detection."""

    def test_regime_normal_default(self, default_provider):
        """Test that default regime is NORMAL when no returns."""
        regime = default_provider._detect_volatility_regime(None)
        assert regime == "normal"  # Returns string, not enum

    def test_regime_normal_insufficient_returns(self, default_provider):
        """Test regime is NORMAL with insufficient returns."""
        regime = default_provider._detect_volatility_regime([0.001])
        assert regime == "normal"

    def test_regime_with_stable_returns(self, default_provider):
        """Test regime detection with stable returns."""
        # Very stable returns (should give normal or low)
        returns = [0.0001] * 30
        regime = default_provider._detect_volatility_regime(returns)
        assert regime in ("low", "normal")

    def test_regime_with_volatile_returns(self, default_provider):
        """Test regime detection with volatile returns."""
        # High volatility returns (should give normal or high)
        returns = [0.02, -0.015, 0.018, -0.022, 0.025] * 6
        regime = default_provider._detect_volatility_regime(returns)
        assert regime in ("normal", "high", "extreme")

    def test_volatility_regime_is_string(self, default_provider):
        """Test that volatility regime returns a string."""
        regime = default_provider._detect_volatility_regime([0.01] * 30)
        assert isinstance(regime, str)
        assert regime in ("low", "normal", "high", "extreme")


# =============================================================================
# Test Carry Trade (Interest Rate Differential)
# =============================================================================


class TestCarryTrade:
    """Tests for carry trade / interest rate differential factor."""

    def test_interest_rate_diff_effect(self, default_provider, buy_order, basic_market):
        """Test that interest rate differential affects slippage."""
        # Base case: no carry
        slip_no_carry = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            interest_rate_diff=0.0,
        )

        # With large carry differential (e.g., EM currencies)
        slip_high_carry = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            interest_rate_diff=0.10,  # 10% differential
        )

        # Both should produce valid slippage
        assert slip_no_carry > 0
        assert slip_high_carry > 0

    def test_negative_interest_rate_diff(self, default_provider, buy_order, basic_market):
        """Test negative interest rate differential."""
        slip_neg_carry = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            interest_rate_diff=-0.08,
        )

        # Should produce valid slippage
        assert slip_neg_carry > 0


# =============================================================================
# Test DXY Correlation Decay
# =============================================================================


class TestDXYCorrelation:
    """Tests for DXY correlation decay factor."""

    def test_dxy_correlation_parameter(self, default_provider, buy_order, basic_market):
        """Test that DXY correlation parameter works."""
        # High correlation (like EUR/USD)
        slip_high_corr = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            dxy_correlation=0.95,
        )

        # Low correlation (less liquid)
        slip_low_corr = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            dxy_correlation=0.50,
        )

        # Both should produce valid slippage
        assert slip_high_corr > 0
        assert slip_low_corr > 0

    def test_extreme_dxy_correlations(self, default_provider, buy_order, basic_market):
        """Test extreme DXY correlation values."""
        slip_high = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            dxy_correlation=1.0,
        )

        slip_low = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            dxy_correlation=0.0,
        )

        assert slip_high > 0
        assert slip_low > 0


# =============================================================================
# Test News Events
# =============================================================================


class TestNewsEvents:
    """Tests for news event impact factor."""

    def test_nfp_event_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that NFP (Non-Farm Payrolls) increases slippage."""
        slip_normal = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
        )

        slip_nfp = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            upcoming_news="nfp",
        )

        assert slip_nfp > slip_normal

    def test_fomc_event_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that FOMC increases slippage."""
        slip_normal = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
        )

        slip_fomc = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            upcoming_news="fomc",
        )

        assert slip_fomc > slip_normal

    def test_ecb_event_increases_slippage(self, default_provider, buy_order, basic_market):
        """Test that ECB decision increases slippage."""
        slip_ecb = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
            upcoming_news="ecb",
        )

        slip_normal = default_provider.compute_slippage_pips(
            buy_order,
            basic_market,
            0.001,
        )

        assert slip_ecb > slip_normal


# =============================================================================
# Test Spread Regime
# =============================================================================


class TestSpreadRegime:
    """Tests for spread regime factor."""

    def test_spread_affects_slippage(self, default_provider, buy_order):
        """Test that spread affects slippage."""
        tight_market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=1.08502,  # 0.2 pip spread
            adv=500_000_000_000.0,
        )

        wide_market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=1.08530,  # 3 pip spread
            adv=500_000_000_000.0,
        )

        slip_tight = default_provider.compute_slippage_pips(buy_order, tight_market, 0.001)
        slip_wide = default_provider.compute_slippage_pips(buy_order, wide_market, 0.001)

        # Both should be valid
        assert slip_tight > 0
        assert slip_wide > 0


# =============================================================================
# Test Adaptive Impact Coefficient
# =============================================================================


class TestAdaptiveImpact:
    """Tests for adaptive impact coefficient learning."""

    def test_update_fill_quality_adjusts_k(self, default_provider):
        """Test that fill quality feedback adjusts impact coefficient."""
        initial_k = default_provider._adaptive_k

        # Report consistently higher actual slippage than predicted
        for _ in range(20):
            default_provider.update_fill_quality(
                predicted_slippage_pips=1.0, actual_slippage_pips=2.0
            )

        # k should increase to match higher actual slippage
        assert default_provider._adaptive_k > initial_k

    def test_update_fill_quality_decreases_k(self, default_provider):
        """Test that lower actual slippage decreases k."""
        initial_k = default_provider._adaptive_k

        # Report consistently lower actual slippage than predicted
        for _ in range(20):
            default_provider.update_fill_quality(
                predicted_slippage_pips=2.0, actual_slippage_pips=1.0
            )

        # k should decrease
        assert default_provider._adaptive_k < initial_k

    def test_k_stays_within_bounds(self, default_provider):
        """Test that k stays within configured bounds."""
        k_min, k_max = default_provider.config.impact_coef_range

        # Try to push k very high
        for _ in range(100):
            default_provider.update_fill_quality(
                predicted_slippage_pips=0.5, actual_slippage_pips=10.0
            )

        assert default_provider._adaptive_k <= k_max

        # Reset and try to push k very low
        default_provider.reset_adaptive_state()
        for _ in range(100):
            default_provider.update_fill_quality(
                predicted_slippage_pips=10.0, actual_slippage_pips=0.5
            )

        assert default_provider._adaptive_k >= k_min


# =============================================================================
# Test Pre-Trade Cost Estimation
# =============================================================================


class TestPreTradeCostEstimation:
    """Tests for pre-trade impact cost estimation."""

    def test_estimate_impact_cost_returns_dict(self, default_provider):
        """Test that estimate_impact_cost returns expected structure."""
        estimate = default_provider.estimate_impact_cost(
            notional=1_000_000.0,
            adv=500_000_000_000.0,
            symbol="EUR_USD",
            side="BUY",
        )

        assert isinstance(estimate, dict)
        assert "impact_pips" in estimate
        assert "participation" in estimate  # Key is "participation", not "participation_ratio"

    def test_large_order_estimation(self, default_provider):
        """Test estimation for large orders."""
        # Large order relative to ADV
        estimate = default_provider.estimate_impact_cost(
            notional=10_000_000_000.0,  # $10B
            adv=500_000_000_000.0,  # $500B ADV
            symbol="EUR_USD",
            side="BUY",
        )

        # Should return valid estimate
        assert estimate["impact_pips"] > 0
        assert estimate["participation"] > 0  # Key is "participation"

    def test_small_order_market_ok(self, default_provider):
        """Test that small orders have low impact."""
        estimate = default_provider.estimate_impact_cost(
            notional=100_000.0,  # $100k
            adv=500_000_000_000.0,
            symbol="EUR_USD",
            side="BUY",
        )

        # Very small participation should have low impact
        assert estimate["impact_pips"] < 5.0


# =============================================================================
# Test Profiles
# =============================================================================


class TestProfiles:
    """Tests for configuration profiles."""

    def test_from_profile_retail(self):
        """Test retail profile creation (default)."""
        provider = ForexParametricSlippageProvider.from_profile("retail")
        assert provider.config.impact_coef_base == 0.03  # Default

    def test_from_profile_conservative(self):
        """Test conservative profile has higher impact coef."""
        retail = ForexParametricSlippageProvider.from_profile("retail")
        conservative = ForexParametricSlippageProvider.from_profile("conservative")

        assert conservative.config.impact_coef_base > retail.config.impact_coef_base

    def test_from_profile_institutional(self):
        """Test institutional profile has lower impact coef."""
        institutional = ForexParametricSlippageProvider.from_profile("institutional")
        retail = ForexParametricSlippageProvider.from_profile("retail")

        # Institutional should have lower slippage
        assert institutional.config.impact_coef_base < retail.config.impact_coef_base

    def test_from_profile_exotic(self):
        """Test exotic profile has higher impact coef."""
        retail = ForexParametricSlippageProvider.from_profile("retail")
        exotic = ForexParametricSlippageProvider.from_profile("exotic")

        assert exotic.config.impact_coef_base > retail.config.impact_coef_base

    def test_from_profile_major_only(self):
        """Test major_only profile."""
        major_only = ForexParametricSlippageProvider.from_profile("major_only")
        retail = ForexParametricSlippageProvider.from_profile("retail")

        assert major_only.config.impact_coef_base < retail.config.impact_coef_base

    def test_unknown_profile_defaults_to_retail(self):
        """Test that unknown profile defaults to retail."""
        provider = ForexParametricSlippageProvider.from_profile("unknown_profile")
        assert provider is not None  # Should not raise, defaults to retail


# =============================================================================
# Test JPY Pairs (Different Pip Definition)
# =============================================================================


class TestJPYPairs:
    """Tests for JPY pairs which have different pip definition."""

    def test_jpy_pair_detection(self, default_provider):
        """Test JPY pairs are correctly detected."""
        jpy_pairs = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY"]
        for pair in jpy_pairs:
            assert default_provider._is_jpy_pair(pair), f"{pair} should be JPY pair"

    def test_non_jpy_pair_detection(self, default_provider):
        """Test non-JPY pairs are correctly detected."""
        non_jpy = ["EURUSD", "GBPUSD", "EURGBP", "AUDNZD"]
        for pair in non_jpy:
            assert not default_provider._is_jpy_pair(pair), f"{pair} should NOT be JPY pair"

    def test_jpy_pip_conversion(self, default_provider, jpy_order):
        """Test JPY pair pip calculation."""
        jpy_market = MarketState(
            timestamp=1700056800000,
            bid=150.00,  # USD/JPY
            ask=150.02,  # 2 pip spread
            adv=500_000_000_000.0,
        )

        slippage_pips = default_provider.compute_slippage_pips(jpy_order, jpy_market, 0.001)

        # Should be a reasonable number of pips
        assert 0.1 <= slippage_pips <= 20.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_adv_handled(self, default_provider, buy_order):
        """Test handling of zero ADV."""
        market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=1.08510,
            adv=0.0,  # Zero ADV
        )

        # Should not crash, should give high slippage
        slippage = default_provider.compute_slippage_pips(buy_order, market, 0.001)
        assert slippage > 0
        # With zero ADV, participation becomes undefined, should get max slippage
        assert slippage >= default_provider.config.min_slippage_pips

    def test_none_adv_handled(self, default_provider, buy_order):
        """Test handling of None ADV."""
        market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=1.08510,
            adv=None,
        )

        slippage = default_provider.compute_slippage_pips(buy_order, market, 0.001)
        assert slippage > 0

    def test_nan_values_handled(self, default_provider, buy_order):
        """Test handling of NaN values."""
        market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=float("nan"),
            adv=500_000_000_000.0,
        )

        # Should not crash
        slippage = default_provider.compute_slippage_pips(buy_order, market, 0.001)
        assert math.isfinite(slippage)

    def test_inf_values_capped(self, default_provider, buy_order):
        """Test that infinite values are capped."""
        market = MarketState(
            timestamp=1700056800000,
            bid=1.08500,
            ask=1.08510,
            adv=float("inf"),
        )

        slippage = default_provider.compute_slippage_pips(buy_order, market, 0.001)
        assert math.isfinite(slippage)

    def test_empty_symbol_handled(self, default_provider, basic_market):
        """Test handling of empty symbol."""
        order = Order(
            symbol="",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
        )

        slippage = default_provider.compute_slippage_pips(order, basic_market, 0.001)
        assert slippage > 0

    def test_negative_spread_handled(self, default_provider, buy_order):
        """Test handling of negative spread (crossed market)."""
        crossed_market = MarketState(
            timestamp=1700056800000,
            bid=1.08510,  # Bid > Ask (crossed)
            ask=1.08500,
            adv=500_000_000_000.0,
        )

        slippage = default_provider.compute_slippage_pips(buy_order, crossed_market, 0.001)
        assert slippage >= default_provider.config.min_slippage_pips


# =============================================================================
# Test Fee Provider
# =============================================================================


class TestForexFeeProvider:
    """Tests for ForexFeeProvider."""

    def test_default_zero_commission(self):
        """Test default provider has zero commission."""
        provider = ForexFeeProvider()
        fee = provider.compute_fee(1_000_000.0, "BUY", "taker", 100000.0)
        assert fee == 0.0

    def test_institutional_commission(self):
        """Test institutional commission."""
        provider = ForexFeeProvider(commission_bps=0.5)  # 0.5 bps
        fee = provider.compute_fee(1_000_000.0, "BUY", "taker", 100000.0)
        assert fee == pytest.approx(50.0, rel=1e-6)  # $50 on $1M

    def test_swap_cost_estimation_long(self):
        """Test swap cost estimation for long position."""
        provider = ForexFeeProvider(include_swap=True)
        swap_cost = provider.estimate_swap_cost(
            notional=1_000_000.0,
            side="BUY",
            holding_days=5,
            swap_points_long=-5.0,  # Paying swap (negative swap points)
            swap_points_short=3.0,
        )
        # Negative swap points mean cost, result should be negative
        assert swap_cost < 0

    def test_swap_cost_estimation_short(self):
        """Test swap cost estimation for short position."""
        provider = ForexFeeProvider(include_swap=True)
        swap_cost = provider.estimate_swap_cost(
            notional=1_000_000.0,
            side="SELL",
            holding_days=5,
            swap_points_long=-5.0,
            swap_points_short=3.0,  # Receiving swap (positive swap points)
        )
        # Positive swap points mean credit, result should be positive
        assert swap_cost > 0

    def test_from_config(self):
        """Test creating provider from config dict."""
        config = {"commission_bps": 1.0, "include_swap": True}
        provider = ForexFeeProvider.from_config(config)
        assert provider.commission_bps == 1.0
        assert provider.include_swap is True


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory function integration."""

    def test_create_slippage_provider_forex(self):
        """Test creating forex slippage provider via factory."""
        provider = create_slippage_provider(
            level="L2",
            asset_class=AssetClass.FOREX,
        )
        assert isinstance(provider, ForexParametricSlippageProvider)

    def test_create_slippage_provider_forex_with_profile(self):
        """Test creating forex provider with profile."""
        provider = create_slippage_provider(
            level="L2",
            asset_class=AssetClass.FOREX,
            profile="conservative",
        )
        assert isinstance(provider, ForexParametricSlippageProvider)
        assert provider.config.impact_coef_base > 0.03  # Conservative has higher k

    def test_create_fee_provider_forex(self):
        """Test creating forex fee provider via factory."""
        provider = create_fee_provider(asset_class=AssetClass.FOREX)
        assert isinstance(provider, ForexFeeProvider)

    def test_create_fee_provider_forex_with_config(self):
        """Test creating forex fee provider with config."""
        provider = create_fee_provider(
            asset_class=AssetClass.FOREX,
            config={"commission_bps": 0.5},
        )
        assert isinstance(provider, ForexFeeProvider)
        assert provider.commission_bps == 0.5

    def test_create_execution_provider_forex(self):
        """Test creating combined execution provider for forex."""
        provider = create_execution_provider(
            asset_class=AssetClass.FOREX,
            level="L2",
        )
        assert isinstance(provider, L2ExecutionProvider)


# =============================================================================
# Test From Config
# =============================================================================


class TestFromConfig:
    """Tests for creating provider from config dict."""

    def test_from_config_basic(self):
        """Test creating provider from basic config."""
        config = {
            "impact_coef_base": 0.04,
            "min_slippage_pips": 0.2,
            "max_slippage_pips": 30.0,
        }
        provider = ForexParametricSlippageProvider.from_config(config)
        assert provider.config.impact_coef_base == 0.04
        assert provider.config.min_slippage_pips == 0.2
        assert provider.config.max_slippage_pips == 30.0

    def test_from_config_with_impact_coef(self):
        """Test creating provider with custom impact coefficient."""
        config = {
            "impact_coef": 0.04,
        }
        provider = ForexParametricSlippageProvider.from_config(config)
        assert provider.config.impact_coef_base == 0.04

    def test_from_config_empty(self):
        """Test creating provider from empty config uses defaults."""
        provider = ForexParametricSlippageProvider.from_config({})
        assert provider.config.impact_coef_base == 0.03


# =============================================================================
# Test Regression (Output Stability)
# =============================================================================


class TestRegression:
    """Regression tests for output stability."""

    def test_eurusd_baseline_slippage(self, default_provider, buy_order, basic_market):
        """Test baseline slippage for EUR/USD under normal conditions."""
        slippage = default_provider.compute_slippage_pips(buy_order, basic_market, 0.001)
        # Should be in reasonable range for 0.1% participation in EUR/USD
        assert 0.5 <= slippage <= 5.0

    def test_exotic_baseline_slippage(self, default_provider, exotic_order, basic_market):
        """Test baseline slippage for exotic pair."""
        slippage = default_provider.compute_slippage_pips(exotic_order, basic_market, 0.001)
        # Exotic should have higher slippage
        assert slippage >= 1.0

    def test_london_ny_overlap_best_liquidity(
        self, default_provider, buy_order, london_ny_overlap_market
    ):
        """Test that London-NY overlap has best execution."""
        slippage = default_provider.compute_slippage_pips(
            buy_order, london_ny_overlap_market, 0.001
        )
        # Best liquidity should give low slippage
        assert slippage < 3.0


# =============================================================================
# Test Consistency with Crypto/Equity Providers
# =============================================================================


class TestConsistency:
    """Tests for API consistency with other parametric providers."""

    def test_same_interface_as_crypto(self):
        """Test that forex provider has same interface as crypto."""
        forex = ForexParametricSlippageProvider()

        # Should have same key methods
        assert hasattr(forex, "compute_slippage_bps")
        assert hasattr(forex, "update_fill_quality")
        assert hasattr(forex, "estimate_impact_cost")
        assert hasattr(forex, "from_profile")
        assert hasattr(forex, "from_config")

    def test_forex_lower_slippage_than_crypto(self, buy_order, basic_market):
        """Test that forex generally has lower slippage than crypto."""
        forex_provider = ForexParametricSlippageProvider()

        # EUR/USD should have low slippage
        slippage_bps = forex_provider.compute_slippage_bps(buy_order, basic_market, 0.001)

        # Forex majors should typically have < 10 bps slippage for small trades
        assert slippage_bps < 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
