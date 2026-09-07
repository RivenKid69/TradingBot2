# -*- coding: utf-8 -*-
"""
tests/test_execution_providers.py
Comprehensive tests for execution_providers.py

Test coverage:
- Data classes (MarketState, Order, Fill, BarData)
- Protocol implementations
- L2 providers (Statistical slippage, OHLCV fills, fees)
- L3 stubs
- Factory functions
- Backward compatibility
- Edge cases and error handling
"""

import math
import pytest

pytest.importorskip("sortedcontainers")
from typing import Optional

from execution_providers import (
    # Enums
    AssetClass,
    OrderSide,
    OrderType,
    LiquidityRole,
    # Data classes
    MarketState,
    Order,
    Fill,
    BarData,
    # Protocols
    SlippageProvider,
    FillProvider,
    FeeProvider,
    ExecutionProvider,
    # L2 Implementations
    StatisticalSlippageProvider,
    OHLCVFillProvider,
    ZeroFeeProvider,
    CryptoFeeProvider,
    EquityFeeProvider,
    L2ExecutionProvider,
    # L3 Stubs
    LOBSlippageProvider,
    LOBFillProvider,
    # Factory functions
    create_slippage_provider,
    create_fee_provider,
    create_fill_provider,
    create_execution_provider,
    # Backward compatibility
    wrap_legacy_slippage_config,
    wrap_legacy_fees_model,
)

# L3 providers (Stage 7)
from execution_providers_l3 import (
    L3SlippageProvider,
    L3FillProvider,
    L3ExecutionProvider,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_market_state() -> MarketState:
    """Create a basic market state for testing."""
    return MarketState(
        timestamp=1700000000000,
        bid=100.0,
        ask=100.10,
        bid_size=1000.0,
        ask_size=1000.0,
        last_price=100.05,
    )


@pytest.fixture
def equity_market_state() -> MarketState:
    """Create market state for equity testing."""
    return MarketState(
        timestamp=1700000000000,
        bid=150.00,
        ask=150.02,
        bid_size=500.0,
        ask_size=500.0,
        last_price=150.01,
        adv=10_000_000.0,  # $10M ADV
    )


@pytest.fixture
def basic_bar() -> BarData:
    """Create a basic bar for testing."""
    return BarData(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10000.0,
        timestamp=1700000000000,
        timeframe_ms=3600000,
    )


@pytest.fixture
def buy_market_order() -> Order:
    """Create a buy market order."""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        qty=1.0,
        order_type="MARKET",
    )


@pytest.fixture
def sell_market_order() -> Order:
    """Create a sell market order."""
    return Order(
        symbol="BTCUSDT",
        side="SELL",
        qty=1.0,
        order_type="MARKET",
    )


@pytest.fixture
def buy_limit_order() -> Order:
    """Create a buy limit order."""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        qty=1.0,
        order_type="LIMIT",
        limit_price=99.5,
    )


@pytest.fixture
def sell_limit_order() -> Order:
    """Create a sell limit order."""
    return Order(
        symbol="BTCUSDT",
        side="SELL",
        qty=1.0,
        order_type="LIMIT",
        limit_price=100.5,
    )


@pytest.fixture
def equity_order() -> Order:
    """Create an equity order."""
    return Order(
        symbol="AAPL",
        side="BUY",
        qty=100.0,
        order_type="MARKET",
        asset_class=AssetClass.EQUITY,
    )


# =============================================================================
# Test Data Classes
# =============================================================================


class TestMarketState:
    """Tests for MarketState data class."""

    def test_basic_creation(self):
        """Test basic MarketState creation."""
        state = MarketState(timestamp=1000, bid=100.0, ask=100.10)
        assert state.timestamp == 1000
        assert state.bid == 100.0
        assert state.ask == 100.10

    def test_get_mid_price_from_bid_ask(self, basic_market_state):
        """Test mid-price calculation from bid/ask."""
        mid = basic_market_state.get_mid_price()
        assert mid == pytest.approx(100.05, rel=1e-6)

    def test_get_mid_price_explicit(self):
        """Test mid-price when explicitly provided."""
        state = MarketState(timestamp=0, mid_price=99.99)
        assert state.get_mid_price() == 99.99

    def test_get_mid_price_fallback_to_last(self):
        """Test mid-price fallback to last_price."""
        state = MarketState(timestamp=0, last_price=100.0)
        assert state.get_mid_price() == 100.0

    def test_get_mid_price_none(self):
        """Test mid-price returns None when no data."""
        state = MarketState(timestamp=0)
        assert state.get_mid_price() is None

    def test_get_spread_bps(self, basic_market_state):
        """Test spread calculation in basis points."""
        spread = basic_market_state.get_spread_bps()
        # Spread = (100.10 - 100.0) / 100.05 * 10000 ≈ 9.995 bps
        assert spread == pytest.approx(9.995, rel=1e-2)

    def test_get_spread_bps_explicit(self):
        """Test spread when explicitly provided."""
        state = MarketState(timestamp=0, spread_bps=5.0)
        assert state.get_spread_bps() == 5.0

    def test_get_spread_bps_none(self):
        """Test spread returns None when no data."""
        state = MarketState(timestamp=0)
        assert state.get_spread_bps() is None

    def test_get_reference_price_buy(self, basic_market_state):
        """Test reference price for buy (should be ask)."""
        ref = basic_market_state.get_reference_price("BUY")
        assert ref == 100.10

    def test_get_reference_price_sell(self, basic_market_state):
        """Test reference price for sell (should be bid)."""
        ref = basic_market_state.get_reference_price("SELL")
        assert ref == 100.0

    def test_get_reference_price_enum(self, basic_market_state):
        """Test reference price with enum side."""
        ref = basic_market_state.get_reference_price(OrderSide.BUY)
        assert ref == 100.10


class TestOrder:
    """Tests for Order data class."""

    def test_basic_creation(self):
        """Test basic Order creation."""
        order = Order(symbol="BTCUSDT", side="BUY", qty=1.0, order_type="MARKET")
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.qty == 1.0
        assert order.order_type == "MARKET"

    def test_is_buy(self):
        """Test is_buy property."""
        buy = Order(symbol="X", side="BUY", qty=1.0, order_type="MARKET")
        sell = Order(symbol="X", side="SELL", qty=1.0, order_type="MARKET")
        assert buy.is_buy is True
        assert sell.is_buy is False

    def test_get_notional(self):
        """Test notional calculation."""
        order = Order(symbol="X", side="BUY", qty=10.0, order_type="MARKET")
        assert order.get_notional(100.0) == 1000.0

    def test_get_notional_explicit(self):
        """Test notional when explicitly provided."""
        order = Order(symbol="X", side="BUY", qty=10.0, order_type="MARKET", notional=999.0)
        assert order.get_notional(100.0) == 999.0

    def test_default_values(self):
        """Test default values."""
        order = Order(symbol="X", side="BUY", qty=1.0, order_type="MARKET")
        assert order.asset_class == AssetClass.CRYPTO
        assert order.time_in_force == "GTC"
        assert order.limit_price is None


class TestFill:
    """Tests for Fill data class."""

    def test_basic_creation(self):
        """Test basic Fill creation."""
        fill = Fill(
            price=100.0,
            qty=10.0,
            fee=0.5,
            slippage_bps=5.0,
            liquidity="taker",
        )
        assert fill.price == 100.0
        assert fill.qty == 10.0
        assert fill.fee == 0.5
        assert fill.slippage_bps == 5.0
        assert fill.liquidity == "taker"

    def test_notional_computed(self):
        """Test notional is computed if not provided."""
        fill = Fill(price=100.0, qty=10.0, fee=0.0, slippage_bps=0.0, liquidity="taker")
        assert fill.notional == 1000.0

    def test_total_cost(self):
        """Test total cost calculation."""
        fill = Fill(
            price=100.0,
            qty=10.0,
            fee=1.0,
            slippage_bps=10.0,  # 10 bps = 0.1%
            liquidity="taker",
        )
        # Slippage cost = 1000 * 10 / 10000 = 1.0
        # Total = 1.0 + 1.0 = 2.0
        assert fill.total_cost == pytest.approx(2.0, rel=1e-6)

    def test_total_cost_bps(self):
        """Test total cost in basis points."""
        fill = Fill(
            price=100.0,
            qty=10.0,
            fee=1.0,
            slippage_bps=10.0,
            liquidity="taker",
        )
        # Total cost = 2.0, notional = 1000
        # Total bps = 2.0 / 1000 * 10000 = 20 bps
        assert fill.total_cost_bps == pytest.approx(20.0, rel=1e-6)


class TestBarData:
    """Tests for BarData data class."""

    def test_basic_creation(self, basic_bar):
        """Test basic BarData creation."""
        assert basic_bar.open == 100.0
        assert basic_bar.high == 101.0
        assert basic_bar.low == 99.0
        assert basic_bar.close == 100.5

    def test_contains_price(self, basic_bar):
        """Test price containment check."""
        assert basic_bar.contains_price(100.0) is True
        assert basic_bar.contains_price(99.0) is True
        assert basic_bar.contains_price(101.0) is True
        assert basic_bar.contains_price(98.0) is False
        assert basic_bar.contains_price(102.0) is False

    def test_contains_price_with_tolerance(self, basic_bar):
        """Test price containment with tolerance."""
        assert basic_bar.contains_price(98.5, tolerance=0.5) is True
        assert basic_bar.contains_price(101.5, tolerance=0.5) is True

    def test_typical_price(self, basic_bar):
        """Test typical price calculation."""
        # (101 + 99 + 100.5) / 3 = 100.167
        assert basic_bar.typical_price == pytest.approx(100.167, rel=1e-3)

    def test_bar_range(self, basic_bar):
        """Test bar range calculation."""
        assert basic_bar.bar_range == 2.0  # 101 - 99


# =============================================================================
# Test L2 Slippage Provider
# =============================================================================


class TestStatisticalSlippageProvider:
    """Tests for StatisticalSlippageProvider."""

    def test_default_parameters(self):
        """Test default parameter values."""
        provider = StatisticalSlippageProvider()
        assert provider.impact_coef == 0.1
        assert provider.spread_bps == 5.0
        assert provider.volatility_scale == 1.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        provider = StatisticalSlippageProvider(
            impact_coef=0.2,
            spread_bps=10.0,
            volatility_scale=1.5,
        )
        assert provider.impact_coef == 0.2
        assert provider.spread_bps == 10.0
        assert provider.volatility_scale == 1.5

    def test_zero_participation(self, buy_market_order, basic_market_state):
        """Test slippage at zero participation."""
        provider = StatisticalSlippageProvider()
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.0)
        # Should still have half spread component
        assert slippage > 0

    def test_small_participation(self, buy_market_order, basic_market_state):
        """Test slippage at small participation."""
        provider = StatisticalSlippageProvider()
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.001)
        # Half spread (~5 bps) + impact = k * sqrt(0.001) * 10000 ≈ 316 bps
        # But with default volatility_scale=1.0 and actual spread, expect ~30-40 bps
        assert slippage >= 4.0
        assert slippage <= 100.0  # Reasonable upper bound for 0.1% participation

    def test_large_participation(self, buy_market_order, basic_market_state):
        """Test slippage at large participation."""
        provider = StatisticalSlippageProvider()
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.10)
        # Should be significantly higher
        assert slippage > 100.0

    def test_slippage_increases_with_participation(self, buy_market_order, basic_market_state):
        """Test that slippage increases with participation."""
        provider = StatisticalSlippageProvider()

        slippage_small = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.001)
        slippage_medium = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.01)
        slippage_large = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.10)

        assert slippage_small < slippage_medium < slippage_large

    def test_sqrt_scaling(self, buy_market_order, basic_market_state):
        """Test sqrt scaling of market impact."""
        provider = StatisticalSlippageProvider(spread_bps=0.0)  # Remove spread

        slip_1 = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.01)
        slip_4 = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.04)

        # sqrt(4) / sqrt(1) = 2, so slippage should roughly double
        assert slip_4 / slip_1 == pytest.approx(2.0, rel=0.1)

    def test_max_slippage_cap(self, buy_market_order, basic_market_state):
        """Test maximum slippage cap."""
        provider = StatisticalSlippageProvider(max_slippage_bps=50.0)
        slippage = provider.compute_slippage_bps(
            buy_market_order, basic_market_state, 10.0  # Very large participation
        )
        assert slippage <= 50.0

    def test_min_slippage_floor(self, buy_market_order, basic_market_state):
        """Test minimum slippage floor."""
        provider = StatisticalSlippageProvider(min_slippage_bps=10.0)
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.0)
        assert slippage >= 10.0

    def test_volatility_adjustment(self, buy_market_order):
        """Test volatility adjustment."""
        low_vol = MarketState(timestamp=0, volatility=0.5)
        high_vol = MarketState(timestamp=0, volatility=2.0)

        provider = StatisticalSlippageProvider()

        slip_low = provider.compute_slippage_bps(buy_market_order, low_vol, 0.01)
        slip_high = provider.compute_slippage_bps(buy_market_order, high_vol, 0.01)

        assert slip_high > slip_low

    def test_estimate_impact_cost(self):
        """Test impact cost estimation."""
        provider = StatisticalSlippageProvider()
        result = provider.estimate_impact_cost(
            notional=100_000.0,
            adv=10_000_000.0,
            volatility=0.02,
        )

        assert "participation" in result
        assert "impact_bps" in result
        assert "impact_cost" in result
        assert result["participation"] == 0.01


# =============================================================================
# Test L2 Fee Providers
# =============================================================================


class TestZeroFeeProvider:
    """Tests for ZeroFeeProvider."""

    def test_always_zero(self):
        """Test that fees are always zero."""
        provider = ZeroFeeProvider()
        assert provider.compute_fee(1000.0, "BUY", "taker", 10.0) == 0.0
        assert provider.compute_fee(1000.0, "SELL", "maker", 10.0) == 0.0


class TestCryptoFeeProvider:
    """Tests for CryptoFeeProvider."""

    def test_default_rates(self):
        """Test default fee rates."""
        provider = CryptoFeeProvider()
        assert provider.maker_bps == 2.0
        assert provider.taker_bps == 4.0

    def test_maker_fee(self):
        """Test maker fee calculation."""
        provider = CryptoFeeProvider(maker_bps=2.0)
        fee = provider.compute_fee(10000.0, "BUY", "maker", 10.0)
        # 10000 * 2 / 10000 = 2.0
        assert fee == pytest.approx(2.0, rel=1e-6)

    def test_taker_fee(self):
        """Test taker fee calculation."""
        provider = CryptoFeeProvider(taker_bps=4.0)
        fee = provider.compute_fee(10000.0, "BUY", "taker", 10.0)
        # 10000 * 4 / 10000 = 4.0
        assert fee == pytest.approx(4.0, rel=1e-6)

    def test_discount(self):
        """Test discount application."""
        provider = CryptoFeeProvider(taker_bps=4.0, discount_rate=0.75, use_discount=True)
        fee = provider.compute_fee(10000.0, "BUY", "taker", 10.0)
        # 10000 * 4 * 0.75 / 10000 = 3.0
        assert fee == pytest.approx(3.0, rel=1e-6)

    def test_side_independent(self):
        """Test that crypto fees are side-independent."""
        provider = CryptoFeeProvider()
        buy_fee = provider.compute_fee(10000.0, "BUY", "taker", 10.0)
        sell_fee = provider.compute_fee(10000.0, "SELL", "taker", 10.0)
        assert buy_fee == sell_fee


class TestEquityFeeProvider:
    """Tests for EquityFeeProvider."""

    def test_buy_free(self):
        """Test that buys are commission-free."""
        provider = EquityFeeProvider()
        fee = provider.compute_fee(15000.0, "BUY", "taker", 100.0)
        assert fee == 0.0

    def test_sell_regulatory_fees(self):
        """Test regulatory fees on sells."""
        provider = EquityFeeProvider()
        # $15000 notional, 100 shares
        fee = provider.compute_fee(15000.0, "SELL", "taker", 100.0)

        # SEC fee: 15000 * 0.0000278 ≈ 0.417
        # TAF fee: 100 * 0.000166 = 0.0166
        # Total ≈ 0.434
        assert fee > 0
        assert fee < 1.0

    def test_taf_max_cap(self):
        """Test TAF fee maximum cap."""
        provider = EquityFeeProvider()
        # Very large trade
        fee = provider.compute_fee(1_000_000.0, "SELL", "taker", 100_000.0)

        # TAF should be capped at 8.30
        breakdown = provider.estimate_regulatory_breakdown(1_000_000.0, 100_000.0)
        assert breakdown["taf_fee"] == 8.30

    def test_regulatory_disabled(self):
        """Test disabling regulatory fees."""
        provider = EquityFeeProvider(include_regulatory=False)
        fee = provider.compute_fee(15000.0, "SELL", "taker", 100.0)
        assert fee == 0.0

    def test_estimate_breakdown(self):
        """Test regulatory fee breakdown."""
        provider = EquityFeeProvider()
        breakdown = provider.estimate_regulatory_breakdown(15000.0, 100.0)

        assert "sec_fee" in breakdown
        assert "taf_fee" in breakdown
        assert "total" in breakdown
        assert breakdown["total"] == breakdown["sec_fee"] + breakdown["taf_fee"]


# =============================================================================
# Test L2 Fill Provider
# =============================================================================


class TestOHLCVFillProvider:
    """Tests for OHLCVFillProvider."""

    def test_market_order_fills(self, buy_market_order, basic_market_state, basic_bar):
        """Test market order always fills."""
        provider = OHLCVFillProvider()
        fill = provider.try_fill(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.qty == buy_market_order.qty
        assert fill.liquidity == "taker"

    def test_limit_buy_fills_when_touched(self, basic_market_state, basic_bar):
        """Test limit buy fills when bar low touches price."""
        order = Order(
            symbol="TEST",
            side="BUY",
            qty=10.0,
            order_type="LIMIT",
            limit_price=99.5,  # Within bar range (low=99)
        )
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.liquidity == "maker"

    def test_limit_buy_no_fill_when_not_touched(self, basic_market_state, basic_bar):
        """Test limit buy doesn't fill when not touched."""
        order = Order(
            symbol="TEST",
            side="BUY",
            qty=10.0,
            order_type="LIMIT",
            limit_price=98.0,  # Below bar low (99)
        )
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is None

    def test_limit_sell_fills_when_touched(self, basic_market_state, basic_bar):
        """Test limit sell fills when bar high touches price."""
        order = Order(
            symbol="TEST",
            side="SELL",
            qty=10.0,
            order_type="LIMIT",
            limit_price=100.5,  # Within bar range (high=101)
        )
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.liquidity == "maker"

    def test_limit_sell_no_fill_when_not_touched(self, basic_market_state, basic_bar):
        """Test limit sell doesn't fill when not touched."""
        order = Order(
            symbol="TEST",
            side="SELL",
            qty=10.0,
            order_type="LIMIT",
            limit_price=102.0,  # Above bar high (101)
        )
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is None

    def test_limit_crosses_spread_taker(self, basic_market_state, basic_bar):
        """Test limit order crossing spread fills as taker."""
        # Buy limit above ask = immediate taker fill
        order = Order(
            symbol="TEST",
            side="BUY",
            qty=10.0,
            order_type="LIMIT",
            limit_price=100.20,  # Above ask (100.10)
        )
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.liquidity == "taker"

    def test_fill_includes_fees(self, buy_market_order, basic_market_state, basic_bar):
        """Test fill includes fee calculation."""
        fee_provider = CryptoFeeProvider(taker_bps=10.0)
        provider = OHLCVFillProvider(fee_provider=fee_provider)
        fill = provider.try_fill(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.fee > 0

    def test_fill_includes_slippage(self, buy_market_order, basic_market_state, basic_bar):
        """Test fill includes slippage."""
        provider = OHLCVFillProvider()
        fill = provider.try_fill(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.slippage_bps >= 0

    def test_zero_qty_no_fill(self, basic_market_state, basic_bar):
        """Test zero quantity order doesn't fill."""
        order = Order(symbol="TEST", side="BUY", qty=0.0, order_type="MARKET")
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)

        assert fill is None


# =============================================================================
# Test L2 Execution Provider
# =============================================================================


class TestL2ExecutionProvider:
    """Tests for L2ExecutionProvider."""

    def test_crypto_defaults(self):
        """Test crypto defaults are applied."""
        provider = L2ExecutionProvider(asset_class=AssetClass.CRYPTO)
        assert provider.asset_class == AssetClass.CRYPTO
        assert isinstance(provider.fees, CryptoFeeProvider)

    def test_equity_defaults(self):
        """Test equity defaults are applied."""
        provider = L2ExecutionProvider(asset_class=AssetClass.EQUITY)
        assert provider.asset_class == AssetClass.EQUITY
        assert isinstance(provider.fees, EquityFeeProvider)

    def test_execute_crypto(self, buy_market_order, basic_market_state, basic_bar):
        """Test executing crypto order."""
        provider = L2ExecutionProvider(asset_class=AssetClass.CRYPTO)
        fill = provider.execute(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.qty == buy_market_order.qty

    def test_execute_equity(self, equity_order, equity_market_state, basic_bar):
        """Test executing equity order."""
        provider = L2ExecutionProvider(asset_class=AssetClass.EQUITY)
        bar = BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000.0)
        fill = provider.execute(equity_order, equity_market_state, bar)

        assert fill is not None
        assert fill.fee == 0.0  # Buy is commission-free

    def test_estimate_execution_cost(self):
        """Test pre-trade cost estimation."""
        provider = L2ExecutionProvider(asset_class=AssetClass.CRYPTO)
        estimate = provider.estimate_execution_cost(
            notional=100_000.0,
            adv=10_000_000.0,
            side="BUY",
        )

        assert "participation" in estimate
        assert "slippage_bps" in estimate
        assert "fee" in estimate
        assert "total_cost" in estimate
        assert estimate["participation"] == 0.01

    def test_custom_providers(self, buy_market_order, basic_market_state, basic_bar):
        """Test custom slippage and fee providers."""
        custom_slippage = StatisticalSlippageProvider(impact_coef=0.5)
        custom_fees = CryptoFeeProvider(taker_bps=10.0)

        provider = L2ExecutionProvider(
            asset_class=AssetClass.CRYPTO,
            slippage_provider=custom_slippage,
            fee_provider=custom_fees,
        )

        fill = provider.execute(buy_market_order, basic_market_state, basic_bar)
        assert fill is not None


# =============================================================================
# Test L3 Stubs
# =============================================================================


class TestLOBSlippageProvider:
    """Tests for LOBSlippageProvider stub."""

    def test_stub_warning(self, caplog):
        """Test stub logs warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            provider = LOBSlippageProvider()
        assert "stub" in caplog.text.lower()

    def test_fallback_to_spread(self, buy_market_order, basic_market_state):
        """Test fallback when no LOB data."""
        provider = LOBSlippageProvider()
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 0.01)
        assert slippage > 0


class TestLOBFillProvider:
    """Tests for LOBFillProvider stub."""

    def test_stub_warning(self, caplog):
        """Test stub logs warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            provider = LOBFillProvider()
        assert "stub" in caplog.text.lower()

    def test_fallback_to_ohlcv(self, buy_market_order, basic_market_state, basic_bar):
        """Test fallback to OHLCV fill."""
        provider = LOBFillProvider()
        fill = provider.try_fill(buy_market_order, basic_market_state, basic_bar)
        assert fill is not None


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_slippage_provider_l2(self):
        """Test creating L2 slippage provider."""
        provider = create_slippage_provider("L2")
        assert isinstance(provider, StatisticalSlippageProvider)

    def test_create_slippage_provider_l3(self):
        """Test creating L3 slippage provider."""
        provider = create_slippage_provider("L3")
        assert isinstance(provider, L3SlippageProvider)

    def test_create_slippage_provider_equity_defaults(self):
        """Test equity defaults for slippage provider."""
        provider = create_slippage_provider("L2", AssetClass.EQUITY)
        assert isinstance(provider, StatisticalSlippageProvider)
        # Equity has tighter spreads
        assert provider.spread_bps == 2.0

    def test_create_fee_provider_crypto(self):
        """Test creating crypto fee provider."""
        provider = create_fee_provider(AssetClass.CRYPTO)
        assert isinstance(provider, CryptoFeeProvider)

    def test_create_fee_provider_equity(self):
        """Test creating equity fee provider."""
        provider = create_fee_provider(AssetClass.EQUITY)
        assert isinstance(provider, EquityFeeProvider)

    def test_create_fill_provider_l2(self):
        """Test creating L2 fill provider."""
        provider = create_fill_provider("L2")
        assert isinstance(provider, OHLCVFillProvider)

    def test_create_fill_provider_l3(self):
        """Test creating L3 fill provider."""
        provider = create_fill_provider("L3")
        assert isinstance(provider, L3FillProvider)

    def test_create_execution_provider(self):
        """Test creating execution provider."""
        provider = create_execution_provider(AssetClass.CRYPTO)
        assert isinstance(provider, L2ExecutionProvider)


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility functions."""

    def test_wrap_legacy_slippage_config_none(self):
        """Test wrapping None config."""
        provider = wrap_legacy_slippage_config(None)
        assert isinstance(provider, StatisticalSlippageProvider)

    def test_wrap_legacy_slippage_config_dict(self):
        """Test wrapping dict config."""
        config = {"k": 0.2, "default_spread_bps": 8.0}
        provider = wrap_legacy_slippage_config(config)
        assert provider.impact_coef == 0.2
        assert provider.spread_bps == 8.0

    def test_wrap_legacy_slippage_config_object(self):
        """Test wrapping object config."""

        class LegacyConfig:
            k = 0.15
            default_spread_bps = 6.0

        provider = wrap_legacy_slippage_config(LegacyConfig())
        assert provider.impact_coef == 0.15
        assert provider.spread_bps == 6.0

    def test_wrap_legacy_fees_model_none(self):
        """Test wrapping None fees model."""
        provider = wrap_legacy_fees_model(None)
        assert isinstance(provider, CryptoFeeProvider)

    def test_wrap_legacy_fees_model_dict(self):
        """Test wrapping dict fees model."""
        model = {"maker_rate_bps": 1.0, "taker_rate_bps": 3.0}
        provider = wrap_legacy_fees_model(model)
        assert provider.maker_bps == 1.0
        assert provider.taker_bps == 3.0


# =============================================================================
# Test Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Tests that implementations satisfy protocols."""

    def test_slippage_provider_protocol(self):
        """Test SlippageProvider protocol compliance."""
        provider = StatisticalSlippageProvider()
        assert isinstance(provider, SlippageProvider)

    def test_fee_provider_protocol_crypto(self):
        """Test CryptoFeeProvider protocol compliance."""
        provider = CryptoFeeProvider()
        assert isinstance(provider, FeeProvider)

    def test_fee_provider_protocol_equity(self):
        """Test EquityFeeProvider protocol compliance."""
        provider = EquityFeeProvider()
        assert isinstance(provider, FeeProvider)

    def test_fill_provider_protocol(self):
        """Test FillProvider protocol compliance."""
        provider = OHLCVFillProvider()
        assert isinstance(provider, FillProvider)

    def test_execution_provider_protocol(self):
        """Test ExecutionProvider protocol compliance."""
        provider = L2ExecutionProvider()
        assert isinstance(provider, ExecutionProvider)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_prices_handling(self):
        """Test handling of NaN prices."""
        state = MarketState(timestamp=0, bid=float("nan"), ask=float("nan"))
        assert state.get_mid_price() is None
        assert state.get_spread_bps() is None

    def test_inf_prices_handling(self):
        """Test handling of infinite prices."""
        state = MarketState(timestamp=0, bid=float("inf"), ask=float("inf"))
        assert state.get_mid_price() is None

    def test_negative_participation(self, buy_market_order, basic_market_state):
        """Test negative participation is handled."""
        provider = StatisticalSlippageProvider()
        # Should use absolute value
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, -0.01)
        assert slippage > 0
        assert math.isfinite(slippage)

    def test_very_large_participation(self, buy_market_order, basic_market_state):
        """Test very large participation is capped."""
        provider = StatisticalSlippageProvider(max_slippage_bps=100.0)
        slippage = provider.compute_slippage_bps(buy_market_order, basic_market_state, 100.0)
        assert slippage <= 100.0

    def test_empty_symbol(self, basic_market_state, basic_bar):
        """Test order with empty symbol."""
        order = Order(symbol="", side="BUY", qty=1.0, order_type="MARKET")
        provider = OHLCVFillProvider()
        fill = provider.try_fill(order, basic_market_state, basic_bar)
        # Should still work - symbol is just metadata
        assert fill is not None

    def test_missing_market_data(self, buy_market_order, basic_bar):
        """Test fill with minimal market data."""
        state = MarketState(timestamp=0)  # No bid/ask
        provider = OHLCVFillProvider()
        fill = provider.try_fill(buy_market_order, state, basic_bar)
        # Market order should still fill using bar open
        assert fill is not None

    def test_zero_volume_bar(self, buy_market_order, basic_market_state):
        """Test fill with zero volume bar."""
        bar = BarData(open=100.0, high=101.0, low=99.0, close=100.5, volume=0.0)
        provider = OHLCVFillProvider()
        fill = provider.try_fill(buy_market_order, basic_market_state, bar)
        assert fill is not None

    def test_inverted_bar(self, buy_market_order, basic_market_state):
        """Test bar where low > high (invalid data)."""
        bar = BarData(open=100.0, high=99.0, low=101.0, close=100.5)  # Inverted
        provider = OHLCVFillProvider()
        # Should handle gracefully
        fill = provider.try_fill(buy_market_order, basic_market_state, bar)
        # May or may not fill, but shouldn't crash


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_crypto_workflow(self):
        """Test full crypto execution workflow."""
        # Setup
        provider = create_execution_provider(AssetClass.CRYPTO)
        market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50010.0,
            adv=500_000_000.0,
        )
        bar = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)
        order = Order(symbol="BTCUSDT", side="BUY", qty=0.1, order_type="MARKET")

        # Execute
        fill = provider.execute(order, market, bar)

        # Verify
        assert fill is not None
        assert fill.qty == 0.1
        assert fill.fee > 0  # Crypto has maker/taker fees
        assert fill.slippage_bps >= 0

    def test_full_equity_workflow(self):
        """Test full equity execution workflow."""
        # Setup
        provider = create_execution_provider(AssetClass.EQUITY)
        market = MarketState(
            timestamp=1700000000000,
            bid=150.00,
            ask=150.02,
            adv=10_000_000.0,
        )
        bar = BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000.0)

        # Test buy (no fees)
        buy_order = Order(symbol="AAPL", side="BUY", qty=100.0, order_type="MARKET")
        buy_fill = provider.execute(buy_order, market, bar)
        assert buy_fill is not None
        assert buy_fill.fee == 0.0

        # Test sell (has regulatory fees)
        sell_order = Order(symbol="AAPL", side="SELL", qty=100.0, order_type="MARKET")
        sell_fill = provider.execute(sell_order, market, bar)
        assert sell_fill is not None
        assert sell_fill.fee > 0

    def test_limit_order_workflow(self):
        """Test limit order execution workflow."""
        provider = create_execution_provider(AssetClass.CRYPTO)
        market = MarketState(timestamp=0, bid=100.0, ask=100.10)

        # Limit order that should fill
        bar_fills = BarData(open=100.0, high=101.0, low=99.0, close=100.5)
        order = Order(symbol="TEST", side="BUY", qty=1.0, order_type="LIMIT", limit_price=99.5)
        fill = provider.execute(order, market, bar_fills)
        assert fill is not None
        assert fill.liquidity == "maker"

        # Limit order that shouldn't fill
        bar_no_fill = BarData(open=100.0, high=101.0, low=100.0, close=100.5)
        fill_none = provider.execute(order, market, bar_no_fill)
        assert fill_none is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
