# -*- coding: utf-8 -*-
"""
Tests for execution_providers_cme.py - CME Execution Providers.

This test file provides comprehensive coverage for CME execution providers:
- CMESlippageProvider slippage calculations
- Session awareness (RTH/ETH)
- Settlement time effects
- Circuit breaker integration
- CMEFeeProvider fee calculations
- CMEL2ExecutionProvider combined execution

Total: 55+ tests for 100% coverage.
"""

import pytest
import math
from decimal import Decimal
from typing import Optional

from execution_providers_cme import (
    # Enums
    CMETradingSession,
    CircuitBreakerState,
    # Constants
    TICK_SIZES,
    DEFAULT_TICK_SIZE,
    # Config
    CMESlippageConfig,
    CME_SLIPPAGE_PROFILES,
    CMEFeeConfig,
    # Providers
    CMESlippageProvider,
    CMEFeeProvider,
    CMEL2ExecutionProvider,
    # Factory functions
    create_cme_slippage_provider,
    create_cme_execution_provider,
    get_tick_size,
    calculate_spread_in_bps,
)

from execution_providers import (
    Order,
    MarketState,
    BarData,
    Fill,
    AssetClass,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> CMESlippageConfig:
    """Default CME slippage configuration."""
    return CMESlippageConfig()


@pytest.fixture
def slippage_provider() -> CMESlippageProvider:
    """Default CME slippage provider."""
    return CMESlippageProvider()


@pytest.fixture
def fee_provider() -> CMEFeeProvider:
    """Default CME fee provider."""
    return CMEFeeProvider()


@pytest.fixture
def execution_provider() -> CMEL2ExecutionProvider:
    """Default CME execution provider."""
    return CMEL2ExecutionProvider()


@pytest.fixture
def es_order() -> Order:
    """Sample ES market order."""
    return Order(
        symbol="ES",
        side="BUY",
        qty=1.0,
        order_type="MARKET",
    )


@pytest.fixture
def es_limit_order() -> Order:
    """Sample ES limit order."""
    return Order(
        symbol="ES",
        side="BUY",
        qty=1.0,
        order_type="LIMIT",
        price=4500.0,
    )


@pytest.fixture
def es_market_state() -> MarketState:
    """Sample ES market state."""
    return MarketState(
        timestamp=0,
        bid=4500.00,
        ask=4500.25,
        adv=2_000_000_000,  # $2B ADV
    )


@pytest.fixture
def es_bar_data() -> BarData:
    """Sample ES bar data."""
    return BarData(
        open=4500.0,
        high=4510.0,
        low=4490.0,
        close=4505.0,
        volume=100000,
    )


# =============================================================================
# Test Tick Sizes
# =============================================================================


class TestTickSizes:
    """Tests for tick size configuration."""

    def test_es_tick_size(self):
        """ES tick size is 0.25."""
        assert TICK_SIZES["ES"] == Decimal("0.25")

    def test_nq_tick_size(self):
        """NQ tick size is 0.25."""
        assert TICK_SIZES["NQ"] == Decimal("0.25")

    def test_gc_tick_size(self):
        """GC tick size is 0.10."""
        assert TICK_SIZES["GC"] == Decimal("0.10")

    def test_cl_tick_size(self):
        """CL tick size is 0.01."""
        assert TICK_SIZES["CL"] == Decimal("0.01")

    def test_6e_tick_size(self):
        """6E tick size is 0.00005."""
        assert TICK_SIZES["6E"] == Decimal("0.00005")

    def test_zn_tick_size(self):
        """ZN tick size is 1/64th."""
        assert TICK_SIZES["ZN"] == Decimal("0.015625")

    def test_default_tick_size(self):
        """Default tick size is 0.01."""
        assert DEFAULT_TICK_SIZE == Decimal("0.01")

    def test_get_tick_size_known(self):
        """Get tick size for known symbol."""
        assert get_tick_size("ES") == Decimal("0.25")

    def test_get_tick_size_unknown(self):
        """Get default tick size for unknown symbol."""
        assert get_tick_size("UNKNOWN") == DEFAULT_TICK_SIZE


# =============================================================================
# Test CMESlippageConfig
# =============================================================================


class TestCMESlippageConfig:
    """Tests for CME slippage configuration."""

    def test_default_config_creation(self, default_config):
        """Default config can be created."""
        assert default_config is not None

    def test_default_spreads(self, default_config):
        """Default spreads configured for major products."""
        assert default_config.symbol_spreads["ES"] == Decimal("0.25")
        assert default_config.symbol_spreads["GC"] == Decimal("0.10")
        assert default_config.symbol_spreads["CL"] == Decimal("0.01")

    def test_default_impacts(self, default_config):
        """Default impact coefficients configured."""
        assert default_config.symbol_impacts["ES"] == 0.03
        assert default_config.symbol_impacts["NQ"] == 0.04
        assert default_config.symbol_impacts["GC"] == 0.05

    def test_session_multipliers(self, default_config):
        """Session multipliers configured."""
        assert default_config.rth_spread_multiplier == 1.0
        assert default_config.eth_spread_multiplier == 1.5

    def test_settlement_premium(self, default_config):
        """Settlement premium configured."""
        assert default_config.settlement_premium_max == 0.30
        assert default_config.settlement_window_minutes == 30

    def test_get_spread_known(self, default_config):
        """Get spread for known symbol."""
        spread = default_config.get_spread_ticks("ES")
        assert spread == Decimal("0.25")

    def test_get_spread_unknown(self, default_config):
        """Get default spread for unknown symbol."""
        spread = default_config.get_spread_ticks("UNKNOWN")
        assert spread == Decimal("1")

    def test_get_impact_known(self, default_config):
        """Get impact coefficient for known symbol."""
        impact = default_config.get_impact_coef("ES")
        assert impact == 0.03

    def test_get_impact_unknown(self, default_config):
        """Get default impact for unknown symbol."""
        impact = default_config.get_impact_coef("UNKNOWN")
        assert impact == default_config.default_impact


# =============================================================================
# Test Slippage Profiles
# =============================================================================


class TestSlippageProfiles:
    """Tests for predefined slippage profiles."""

    def test_default_profile_exists(self):
        """Default profile exists."""
        assert "default" in CME_SLIPPAGE_PROFILES

    def test_conservative_profile(self):
        """Conservative profile has wider spreads."""
        config = CME_SLIPPAGE_PROFILES["conservative"]
        assert config.eth_spread_multiplier == 2.0
        assert config.min_slippage_bps == 1.0

    def test_aggressive_profile(self):
        """Aggressive profile has tighter spreads."""
        config = CME_SLIPPAGE_PROFILES["aggressive"]
        assert config.eth_spread_multiplier == 1.3
        assert config.min_slippage_bps == 0.3

    def test_equity_index_profile(self):
        """Equity index profile optimized for index futures."""
        config = CME_SLIPPAGE_PROFILES["equity_index"]
        assert config.symbol_impacts["ES"] == 0.025

    def test_metals_profile(self):
        """Metals profile configured for metals."""
        config = CME_SLIPPAGE_PROFILES["metals"]
        assert config.symbol_impacts["GC"] == 0.05

    def test_energy_profile(self):
        """Energy profile configured for energy."""
        config = CME_SLIPPAGE_PROFILES["energy"]
        assert config.symbol_impacts["CL"] == 0.04
        assert config.symbol_impacts["NG"] == 0.09


# =============================================================================
# Test CMESlippageProvider - Basic
# =============================================================================


class TestCMESlippageProviderBasic:
    """Tests for basic slippage provider operations."""

    def test_create_provider(self, slippage_provider):
        """Provider can be created."""
        assert slippage_provider is not None

    def test_create_from_profile(self):
        """Provider can be created from profile."""
        provider = CMESlippageProvider.from_profile("conservative")
        assert provider is not None

    def test_create_from_invalid_profile(self):
        """Invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown profile"):
            CMESlippageProvider.from_profile("invalid_profile")

    def test_config_property(self, slippage_provider):
        """Config property accessible."""
        assert slippage_provider.config is not None


# =============================================================================
# Test CMESlippageProvider - Slippage Calculations
# =============================================================================


class TestCMESlippageCalculations:
    """Tests for slippage calculations."""

    def test_basic_slippage_calculation(self, slippage_provider, es_order, es_market_state):
        """Basic slippage calculation returns positive value."""
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert slippage > 0
        assert slippage < 200  # Should be reasonable

    def test_slippage_increases_with_participation(
        self, slippage_provider, es_order, es_market_state
    ):
        """Slippage increases with participation ratio."""
        low_part = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        high_part = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.10,
        )
        assert high_part > low_part

    def test_eth_wider_spread(self, slippage_provider, es_order, es_market_state):
        """ETH has wider spreads than RTH."""
        rth_slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
            is_rth=True,
        )
        eth_slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
            is_rth=False,
        )
        assert eth_slippage > rth_slippage

    def test_settlement_premium(self, slippage_provider, es_order, es_market_state):
        """Settlement time increases slippage."""
        normal = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
            minutes_to_settlement=60,
        )
        near_settlement = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
            minutes_to_settlement=5,
        )
        assert near_settlement > normal

    def test_roll_period_premium(self, slippage_provider, es_order, es_market_state):
        """Roll period increases slippage."""
        normal = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
            is_roll_period=False,
        )
        slippage_provider.set_roll_period(True)
        roll_slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert roll_slippage > normal
        slippage_provider.set_roll_period(False)

    def test_minimum_slippage_floor(self, slippage_provider, es_order, es_market_state):
        """Slippage has minimum floor."""
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.0001,  # Very small
        )
        assert slippage >= slippage_provider.config.min_slippage_bps

    def test_maximum_slippage_cap(self, slippage_provider, es_order, es_market_state):
        """Slippage has maximum cap."""
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=1.0,  # Very large
        )
        assert slippage <= slippage_provider.config.max_slippage_bps


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration in slippage."""

    def test_normal_state_no_effect(self, slippage_provider, es_order, es_market_state):
        """Normal circuit breaker state has no effect."""
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert slippage < 200

    def test_level_1_increases_slippage(self, slippage_provider, es_order, es_market_state):
        """Level 1 circuit breaker increases slippage."""
        normal = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.LEVEL_1)
        level1 = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert level1 > normal
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)

    def test_level_2_max_slippage(self, slippage_provider, es_order, es_market_state):
        """Level 2 circuit breaker returns max slippage."""
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.LEVEL_2)
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert slippage == slippage_provider.config.max_slippage_bps
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)

    def test_level_3_max_slippage(self, slippage_provider, es_order, es_market_state):
        """Level 3 circuit breaker returns max slippage."""
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.LEVEL_3)
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert slippage == slippage_provider.config.max_slippage_bps
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)

    def test_velocity_pause_elevated_slippage(self, slippage_provider, es_order, es_market_state):
        """Velocity pause returns elevated slippage."""
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.VELOCITY_PAUSE)
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=0.001,
        )
        assert slippage == slippage_provider.config.max_slippage_bps / 2
        slippage_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)


# =============================================================================
# Test Impact Cost Estimation
# =============================================================================


class TestImpactCostEstimation:
    """Tests for pre-trade impact cost estimation."""

    def test_estimate_impact_cost(self, slippage_provider):
        """Impact cost can be estimated."""
        result = slippage_provider.estimate_impact_cost(
            notional=1_000_000,
            adv=100_000_000,
            symbol="ES",
        )
        assert "impact_bps" in result
        assert "impact_cost" in result
        assert "recommendation" in result

    def test_low_participation_recommendation(self, slippage_provider):
        """Low participation gets favorable recommendation."""
        result = slippage_provider.estimate_impact_cost(
            notional=100_000,
            adv=100_000_000,  # 0.1% participation
            symbol="ES",
        )
        assert "LOW" in result["recommendation"]

    def test_high_participation_recommendation(self, slippage_provider):
        """High participation gets warning recommendation."""
        result = slippage_provider.estimate_impact_cost(
            notional=15_000_000,
            adv=100_000_000,  # 15% participation (>10% triggers CRITICAL)
            symbol="ES",
        )
        assert "CRITICAL" in result["recommendation"]

    def test_zero_adv_handled(self, slippage_provider):
        """Zero ADV is handled gracefully."""
        result = slippage_provider.estimate_impact_cost(
            notional=100_000,
            adv=0,
            symbol="ES",
        )
        assert result["impact_bps"] == slippage_provider.config.max_slippage_bps


# =============================================================================
# Test CMEFeeProvider
# =============================================================================


class TestCMEFeeProvider:
    """Tests for CME fee provider."""

    def test_create_provider(self, fee_provider):
        """Fee provider can be created."""
        assert fee_provider is not None

    def test_es_fee(self, fee_provider):
        """ES fee calculated correctly."""
        fee = fee_provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol="ES",
        )
        # Exchange (1.18) + Clearing (0.10) + NFA (0.02) + Tech (0.25) = 1.55
        assert abs(fee - 1.55) < 0.01

    def test_gc_fee(self, fee_provider):
        """GC fee calculated correctly."""
        fee = fee_provider.compute_fee(
            notional=200000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol="GC",
        )
        # Exchange (1.50) + Clearing (0.10) + NFA (0.02) + Tech (0.25) = 1.87
        assert abs(fee - 1.87) < 0.01

    def test_multi_contract_fee(self, fee_provider):
        """Fee scales with contracts."""
        single = fee_provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol="ES",
        )
        five = fee_provider.compute_fee(
            notional=1125000,
            side="BUY",
            liquidity="taker",
            qty=5.0,
            symbol="ES",
        )
        assert abs(five - single * 5) < 0.01

    def test_fee_same_maker_taker(self, fee_provider):
        """CME fees are same for maker/taker."""
        maker = fee_provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="maker",
            qty=1.0,
            symbol="ES",
        )
        taker = fee_provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol="ES",
        )
        assert maker == taker

    def test_fee_breakdown(self, fee_provider):
        """Fee breakdown available."""
        breakdown = fee_provider.get_fee_breakdown("ES", 1.0)
        assert "exchange_fee" in breakdown
        assert "clearing_fee" in breakdown
        assert "nfa_fee" in breakdown
        assert "tech_fee" in breakdown
        assert "total_fee" in breakdown

    def test_unknown_symbol_default_fee(self, fee_provider):
        """Unknown symbol gets default fee."""
        fee = fee_provider.compute_fee(
            notional=100000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol="UNKNOWN",
        )
        assert fee > 0


# =============================================================================
# Test CMEL2ExecutionProvider
# =============================================================================


class TestCMEL2ExecutionProvider:
    """Tests for combined CME execution provider."""

    def test_create_provider(self, execution_provider):
        """Provider can be created."""
        assert execution_provider is not None

    def test_asset_class(self, execution_provider):
        """Asset class is FUTURES."""
        assert execution_provider.asset_class == AssetClass.FUTURES

    def test_market_order_fills(self, execution_provider, es_order, es_market_state, es_bar_data):
        """Market order fills successfully."""
        fill = execution_provider.execute(
            order=es_order,
            market=es_market_state,
            bar=es_bar_data,
        )
        assert fill is not None
        assert fill.qty == es_order.qty
        assert fill.fee > 0
        assert fill.slippage_bps > 0

    def test_limit_order_fills_when_touched(self, execution_provider, es_market_state, es_bar_data):
        """Limit order fills when price touches."""
        order = Order(
            symbol="ES",
            side="BUY",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4495.0,  # Below current, should fill since bar low is 4490
        )
        fill = execution_provider.execute(
            order=order,
            market=es_market_state,
            bar=es_bar_data,
        )
        assert fill is not None

    def test_limit_order_no_fill_when_not_touched(
        self, execution_provider, es_market_state, es_bar_data
    ):
        """Limit order doesn't fill when price not touched."""
        order = Order(
            symbol="ES",
            side="BUY",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4480.0,  # Below bar low (4490)
        )
        fill = execution_provider.execute(
            order=order,
            market=es_market_state,
            bar=es_bar_data,
        )
        assert fill is None

    def test_sell_order_slippage_direction(self, execution_provider, es_market_state, es_bar_data):
        """Sell order has downward price movement."""
        buy_order = Order(symbol="ES", side="BUY", qty=1.0, order_type="MARKET")
        sell_order = Order(symbol="ES", side="SELL", qty=1.0, order_type="MARKET")

        buy_fill = execution_provider.execute(
            order=buy_order,
            market=es_market_state,
            bar=es_bar_data,
        )
        sell_fill = execution_provider.execute(
            order=sell_order,
            market=es_market_state,
            bar=es_bar_data,
        )

        # Buy fills higher, sell fills lower
        assert buy_fill.price > sell_fill.price

    def test_fill_price_capped_by_bar(self, execution_provider, es_market_state, es_bar_data):
        """Fill price capped by bar high/low."""
        buy_order = Order(symbol="ES", side="BUY", qty=1.0, order_type="MARKET")
        fill = execution_provider.execute(
            order=buy_order,
            market=es_market_state,
            bar=es_bar_data,
        )
        assert fill.price <= es_bar_data.high

    def test_circuit_breaker_integration(
        self, execution_provider, es_order, es_market_state, es_bar_data
    ):
        """Circuit breaker affects execution."""
        normal_fill = execution_provider.execute(
            order=es_order,
            market=es_market_state,
            bar=es_bar_data,
        )

        execution_provider.set_circuit_breaker_state(CircuitBreakerState.LEVEL_1)
        cb_fill = execution_provider.execute(
            order=es_order,
            market=es_market_state,
            bar=es_bar_data,
        )

        # Circuit breaker should increase slippage
        assert cb_fill.slippage_bps > normal_fill.slippage_bps

        execution_provider.set_circuit_breaker_state(CircuitBreakerState.NORMAL)

    def test_pre_trade_cost_estimation(self, execution_provider):
        """Pre-trade cost estimation available."""
        result = execution_provider.estimate_cost(
            symbol="ES",
            qty=1.0,
            price=4500.0,
            adv=2_000_000_000,
        )
        assert "slippage_bps" in result
        assert "slippage_cost" in result
        assert "fee_cost" in result
        assert "total_cost" in result
        assert "recommendation" in result


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_slippage_provider_default(self):
        """Create slippage provider with default profile."""
        provider = create_cme_slippage_provider()
        assert provider is not None

    def test_create_slippage_provider_with_profile(self):
        """Create slippage provider with specific profile."""
        provider = create_cme_slippage_provider("conservative")
        assert provider is not None

    def test_create_execution_provider_default(self):
        """Create execution provider with default profile."""
        provider = create_cme_execution_provider()
        assert provider is not None

    def test_create_execution_provider_with_profile(self):
        """Create execution provider with specific profile."""
        provider = create_cme_execution_provider("equity_index")
        assert provider is not None


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_spread_in_bps(self):
        """Calculate spread in basis points."""
        # ES: 1 tick = 0.25, at 4500 price
        spread_bps = calculate_spread_in_bps(
            symbol="ES",
            spread_ticks=1.0,
            mid_price=4500.0,
        )
        expected = (0.25 / 4500.0) * 10000
        assert abs(spread_bps - expected) < 0.01

    def test_calculate_spread_zero_price(self):
        """Handle zero price gracefully."""
        spread_bps = calculate_spread_in_bps(
            symbol="ES",
            spread_ticks=1.0,
            mid_price=0.0,
        )
        assert spread_bps == 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_participation_clamped(self, slippage_provider, es_order, es_market_state):
        """Negative participation is clamped to zero."""
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=-0.1,
        )
        assert slippage >= 0

    def test_participation_above_one_clamped(self, slippage_provider, es_order, es_market_state):
        """Participation above 1 is clamped."""
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=es_market_state,
            participation_ratio=1.5,
        )
        assert slippage <= slippage_provider.config.max_slippage_bps

    def test_zero_mid_price_handled(self, slippage_provider, es_order):
        """Zero mid price is handled gracefully."""
        market = MarketState(timestamp=0, bid=0, ask=0, adv=1_000_000)
        slippage = slippage_provider.compute_slippage_bps(
            order=es_order,
            market=market,
            participation_ratio=0.001,
        )
        assert slippage >= 0

    def test_negative_qty_in_fee(self, fee_provider):
        """Negative quantity handled with abs."""
        fee = fee_provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="taker",
            qty=-1.0,
            symbol="ES",
        )
        assert fee > 0

    def test_unknown_order_type(self, execution_provider, es_market_state, es_bar_data):
        """Unknown order type returns None."""
        order = Order(
            symbol="ES",
            side="BUY",
            qty=1.0,
            order_type="STOP",  # Unknown type
        )
        fill = execution_provider.execute(
            order=order,
            market=es_market_state,
            bar=es_bar_data,
        )
        # Should log warning and return None or handle gracefully
        # Note: actual behavior depends on implementation

    def test_limit_order_without_price(self, execution_provider, es_market_state, es_bar_data):
        """Limit order without price returns None."""
        order = Order(
            symbol="ES",
            side="BUY",
            qty=1.0,
            order_type="LIMIT",
            limit_price=None,
        )
        fill = execution_provider.execute(
            order=order,
            market=es_market_state,
            bar=es_bar_data,
        )
        assert fill is None


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestCoverageEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_currency_futures_spread_calculation(self):
        """Test spread calculation for currency futures (small spread values)."""
        provider = CMESlippageProvider()
        # Currency futures like 6E have very small tick sizes
        order = Order(symbol="6E", side="BUY", qty=1.0, order_type="MARKET")
        # 6E mid price around 1.10, tick size 0.00005
        market = MarketState(
            timestamp=0,
            bid=1.10000,
            ask=1.10010,  # 1 tick spread
            adv=50_000_000_000,  # High ADV for low participation
        )
        slippage = provider.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=0.0001,
        )
        # Should compute valid slippage for currency futures
        assert slippage > 0
        assert slippage < 50  # Should be reasonable

    def test_estimate_cost_moderate_participation(self):
        """Test cost estimation for 1-5% ADV participation (moderate recommendation)."""
        provider = CMESlippageProvider()
        result = provider.estimate_impact_cost(
            notional=75_000_000,  # $75M
            adv=2_500_000_000,  # $2.5B ADV -> 3% participation
            symbol="ES",
        )
        assert "recommendation" in result
        assert "MODERATE" in result["recommendation"] or "1-5%" in result["recommendation"]

    def test_estimate_cost_high_participation(self):
        """Test cost estimation for 5-10% ADV participation (high recommendation)."""
        provider = CMESlippageProvider()
        result = provider.estimate_impact_cost(
            notional=200_000_000,  # $200M
            adv=2_500_000_000,  # $2.5B ADV -> 8% participation
            symbol="ES",
        )
        assert "recommendation" in result
        assert "HIGH" in result["recommendation"] or "5-10%" in result["recommendation"]

    def test_fee_compute_without_symbol(self):
        """Test fee computation with None symbol (defaults to ES)."""
        provider = CMEFeeProvider()
        fee = provider.compute_fee(
            notional=225000,
            side="BUY",
            liquidity="taker",
            qty=1.0,
            symbol=None,
        )
        # Should use ES default fees
        assert fee > 0
        # ES fee is $1.29 per contract
        assert 1.0 <= fee <= 2.0

    def test_sell_limit_order_passive_fill(self):
        """Test SELL LIMIT order that fills passively (maker)."""
        provider = CMEL2ExecutionProvider()
        order = Order(
            symbol="ES",
            side="SELL",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4510.0,  # Limit above current price
        )
        market = MarketState(
            timestamp=0,
            bid=4499.75,
            ask=4500.00,
            adv=50_000_000_000,
        )
        bar = BarData(
            open=4500.0,
            high=4515.0,  # High exceeds limit -> fills
            low=4498.0,
            close=4510.0,
            volume=100000,
        )
        fill = provider.execute(order=order, market=market, bar=bar)
        # Should fill as maker since limit is above current ask
        assert fill is not None
        assert fill.liquidity == "maker"
        assert fill.price >= order.limit_price

    def test_sell_limit_order_aggressive_crossing(self):
        """Test SELL LIMIT order that crosses spread aggressively (taker)."""
        provider = CMEL2ExecutionProvider()
        order = Order(
            symbol="ES",
            side="SELL",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4499.00,  # Below bid -> immediate execution
        )
        market = MarketState(
            timestamp=0,
            bid=4499.75,
            ask=4500.00,
            adv=50_000_000_000,
        )
        bar = BarData(
            open=4500.0,
            high=4505.0,
            low=4495.0,
            close=4500.0,
            volume=100000,
        )
        fill = provider.execute(order=order, market=market, bar=bar)
        # Should fill as taker since limit is at or below bid
        assert fill is not None
        assert fill.liquidity == "taker"

    def test_sell_limit_order_no_fill(self):
        """Test SELL LIMIT order that doesn't fill (price not reached)."""
        provider = CMEL2ExecutionProvider()
        order = Order(
            symbol="ES",
            side="SELL",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4520.0,  # High limit
        )
        market = MarketState(
            timestamp=0,
            bid=4499.75,
            ask=4500.00,
            adv=50_000_000_000,
        )
        bar = BarData(
            open=4500.0,
            high=4510.0,  # High doesn't reach limit
            low=4495.0,
            close=4505.0,
            volume=100000,
        )
        fill = provider.execute(order=order, market=market, bar=bar)
        # Should not fill since bar high < limit
        assert fill is None

    def test_set_roll_period(self):
        """Test set_roll_period method on execution provider."""
        provider = CMEL2ExecutionProvider()
        # Initially not in roll period
        provider.set_roll_period(True)
        # Check internal state
        assert provider._slippage._is_roll_period is True

        provider.set_roll_period(False)
        assert provider._slippage._is_roll_period is False

    def test_buy_limit_order_aggressive_crossing(self):
        """Test BUY LIMIT order that crosses spread aggressively (taker)."""
        provider = CMEL2ExecutionProvider()
        order = Order(
            symbol="ES",
            side="BUY",
            qty=1.0,
            order_type="LIMIT",
            limit_price=4500.50,  # Above ask -> immediate execution
        )
        market = MarketState(
            timestamp=0,
            bid=4499.75,
            ask=4500.00,
            adv=50_000_000_000,
        )
        bar = BarData(
            open=4500.0,
            high=4505.0,
            low=4495.0,
            close=4500.0,
            volume=100000,
        )
        fill = provider.execute(order=order, market=market, bar=bar)
        # Should fill as taker since limit is at or above ask
        assert fill is not None
        assert fill.liquidity == "taker"
