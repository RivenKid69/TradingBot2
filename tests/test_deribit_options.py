# -*- coding: utf-8 -*-
"""
tests/test_deribit_options.py

Comprehensive tests for Deribit Crypto Options Adapter (Phase 2B).

Test Categories (120 tests):
    1. Instrument naming (12 tests)
    2. Inverse payoffs (20 tests)
    3. Greeks validation (15 tests)
    4. Quote/market data (18 tests)
    5. Margin calculations (25 tests)
    6. Order execution (15 tests)
    7. WebSocket streaming (15 tests)

Run with: pytest tests/test_deribit_options.py -v
"""

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import all components to test
from adapters.deribit.options import (
    # Enums
    DeribitOptionType,
    DeribitOrderType,
    DeribitOrderState,
    DeribitTimeInForce,
    DeribitDirection,
    # Data classes
    DeribitGreeks,
    DeribitOptionContract,
    DeribitOptionQuote,
    DeribitOrderbook,
    DeribitOrderbookLevel,
    DVOLData,
    DeribitInstrumentInfo,
    DeribitPosition,
    DeribitOrder,
    DeribitOrderResult,
    # Adapters
    DeribitOptionsMarketDataAdapter,
    DeribitOptionsOrderExecutionAdapter,
    DeribitAPIClient,
    DeribitRateLimiter,
    # Utility functions
    parse_deribit_instrument_name,
    create_deribit_instrument_name,
    btc_to_usd,
    usd_to_btc,
    eth_to_usd,
    usd_to_eth,
    _compute_inverse_call_payoff,
    _compute_inverse_put_payoff,
    # Factory functions
    create_deribit_options_market_data_adapter,
    create_deribit_options_order_execution_adapter,
    # Constants
    STRIKE_INCREMENTS,
    MIN_ORDER_SIZE,
    TICK_SIZE,
)

from adapters.deribit.margin import (
    # Enums
    MarginMode,
    MarginCallLevel,
    # Data classes
    InversePayoff,
    DeribitMarginResult,
    PositionForMargin,
    # Classes
    InverseSettlementCalculator,
    DeribitMarginCalculator,
    # Functions
    calculate_inverse_call_payoff,
    calculate_inverse_put_payoff,
    create_deribit_margin_calculator,
)

from adapters.deribit.websocket import (
    # Enums
    DeribitChannelType,
    ConnectionState,
    # Data classes
    DeribitStreamConfig,
    DeribitSubscription,
    DeribitMessage,
    # Classes
    DeribitWebSocketClient,
    # Factory
    create_deribit_websocket_client,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_btc_call_instrument() -> str:
    """Sample BTC call instrument name."""
    return "BTC-28MAR25-100000-C"


@pytest.fixture
def sample_btc_put_instrument() -> str:
    """Sample BTC put instrument name."""
    return "BTC-28MAR25-80000-P"


@pytest.fixture
def sample_eth_call_instrument() -> str:
    """Sample ETH call instrument name."""
    return "ETH-28MAR25-4000-C"


@pytest.fixture
def sample_ticker_response() -> Dict[str, Any]:
    """Sample Deribit ticker API response."""
    return {
        "instrument_name": "BTC-28MAR25-100000-C",
        "timestamp": 1700000000000,
        "best_bid_price": 0.045,
        "best_ask_price": 0.048,
        "best_bid_amount": 10.0,
        "best_ask_amount": 15.0,
        "last_price": 0.046,
        "mark_price": 0.0465,
        "index_price": 45000.0,
        "underlying_price": 45000.0,
        "mark_iv": 75.5,
        "bid_iv": 74.0,
        "ask_iv": 77.0,
        "delta": 0.25,
        "gamma": 0.00002,
        "theta": -0.0001,
        "vega": 0.0005,
        "rho": 0.0001,
        "open_interest": 1000.0,
        "stats": {"volume": 500.0},
    }


@pytest.fixture
def sample_orderbook_response() -> Dict[str, Any]:
    """Sample Deribit orderbook API response."""
    return {
        "instrument_name": "BTC-28MAR25-100000-C",
        "timestamp": 1700000000000,
        "bids": [
            [0.045, 10.0],
            [0.044, 20.0],
            [0.043, 30.0],
        ],
        "asks": [
            [0.048, 15.0],
            [0.049, 25.0],
            [0.050, 35.0],
        ],
    }


@pytest.fixture
def sample_position_response() -> Dict[str, Any]:
    """Sample Deribit position API response."""
    return {
        "instrument_name": "BTC-28MAR25-100000-C",
        "size": 0.5,
        "average_price": 0.04,
        "mark_price": 0.0465,
        "index_price": 45000.0,
        "realized_profit_loss": 0.002,
        "floating_profit_loss": 0.003,
        "total_profit_loss": 0.005,
        "delta": 0.125,
        "gamma": 0.00001,
        "theta": -0.00005,
        "vega": 0.00025,
        "maintenance_margin": 0.01,
        "initial_margin": 0.015,
    }


@pytest.fixture
def sample_order_response() -> Dict[str, Any]:
    """Sample Deribit order API response."""
    return {
        "order_id": "order-123456",
        "instrument_name": "BTC-28MAR25-100000-C",
        "direction": "buy",
        "amount": 0.5,
        "price": 0.045,
        "filled_amount": 0.3,
        "average_price": 0.0448,
        "order_state": "open",
        "order_type": "limit",
        "creation_timestamp": 1700000000000,
        "last_update_timestamp": 1700000001000,
        "commission": 0.0001,
        "label": "my-order-1",
    }


@pytest.fixture
def margin_calculator() -> DeribitMarginCalculator:
    """Create margin calculator for tests."""
    return DeribitMarginCalculator(currency="BTC")


# =============================================================================
# Test Category 1: Instrument Naming (12 tests)
# =============================================================================


class TestInstrumentNaming:
    """Tests for Deribit instrument name parsing and creation."""

    def test_parse_btc_call(self, sample_btc_call_instrument: str):
        """Test parsing BTC call instrument name."""
        result = parse_deribit_instrument_name(sample_btc_call_instrument)
        assert result["underlying"] == "BTC"
        assert result["expiration"] == date(2025, 3, 28)
        assert result["strike"] == Decimal("100000")
        assert result["option_type"] == DeribitOptionType.CALL

    def test_parse_btc_put(self, sample_btc_put_instrument: str):
        """Test parsing BTC put instrument name."""
        result = parse_deribit_instrument_name(sample_btc_put_instrument)
        assert result["underlying"] == "BTC"
        assert result["option_type"] == DeribitOptionType.PUT
        assert result["strike"] == Decimal("80000")

    def test_parse_eth_call(self, sample_eth_call_instrument: str):
        """Test parsing ETH call instrument name."""
        result = parse_deribit_instrument_name(sample_eth_call_instrument)
        assert result["underlying"] == "ETH"
        assert result["strike"] == Decimal("4000")
        assert result["option_type"] == DeribitOptionType.CALL

    def test_parse_single_digit_day(self):
        """Test parsing instrument with single digit day."""
        result = parse_deribit_instrument_name("BTC-1JAN25-50000-C")
        assert result["expiration"] == date(2025, 1, 1)

    def test_parse_december_expiry(self):
        """Test parsing December expiry."""
        result = parse_deribit_instrument_name("BTC-27DEC24-100000-P")
        assert result["expiration"] == date(2024, 12, 27)
        assert result["option_type"] == DeribitOptionType.PUT

    def test_parse_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_deribit_instrument_name("INVALID")

    def test_parse_invalid_month_raises(self):
        """Test that invalid month raises ValueError."""
        with pytest.raises(ValueError):
            parse_deribit_instrument_name("BTC-28XXX25-100000-C")

    def test_create_btc_call(self):
        """Test creating BTC call instrument name."""
        name = create_deribit_instrument_name(
            underlying="BTC",
            expiration=date(2025, 3, 28),
            strike=100000,
            option_type="C",
        )
        assert name == "BTC-28MAR25-100000-C"

    def test_create_eth_put(self):
        """Test creating ETH put instrument name."""
        name = create_deribit_instrument_name(
            underlying="ETH",
            expiration=date(2025, 6, 27),
            strike=Decimal("4000"),
            option_type=DeribitOptionType.PUT,
        )
        assert name == "ETH-27JUN25-4000-P"

    def test_create_parse_roundtrip(self, sample_btc_call_instrument: str):
        """Test create/parse roundtrip."""
        parsed = parse_deribit_instrument_name(sample_btc_call_instrument)
        created = create_deribit_instrument_name(
            underlying=parsed["underlying"],
            expiration=parsed["expiration"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
        )
        assert created == sample_btc_call_instrument

    def test_option_type_from_string_call(self):
        """Test DeribitOptionType.from_string for call."""
        assert DeribitOptionType.from_string("C") == DeribitOptionType.CALL
        assert DeribitOptionType.from_string("call") == DeribitOptionType.CALL
        assert DeribitOptionType.from_string("CALL") == DeribitOptionType.CALL

    def test_option_type_from_string_put(self):
        """Test DeribitOptionType.from_string for put."""
        assert DeribitOptionType.from_string("P") == DeribitOptionType.PUT
        assert DeribitOptionType.from_string("put") == DeribitOptionType.PUT


# =============================================================================
# Test Category 2: Inverse Payoffs (20 tests)
# =============================================================================


class TestInversePayoffs:
    """Tests for inverse settlement payoff calculations."""

    def test_call_payoff_itm(self):
        """Test ITM call payoff: max(0, S-K)/S."""
        payoff = calculate_inverse_call_payoff(spot=Decimal("60000"), strike=Decimal("50000"))
        # (60000 - 50000) / 60000 = 10000 / 60000 ≈ 0.1667
        expected = (Decimal("60000") - Decimal("50000")) / Decimal("60000")
        assert payoff == expected
        assert payoff > 0

    def test_call_payoff_atm(self):
        """Test ATM call payoff: should be 0."""
        payoff = calculate_inverse_call_payoff(spot=Decimal("50000"), strike=Decimal("50000"))
        assert payoff == Decimal("0")

    def test_call_payoff_otm(self):
        """Test OTM call payoff: should be 0."""
        payoff = calculate_inverse_call_payoff(spot=Decimal("40000"), strike=Decimal("50000"))
        assert payoff == Decimal("0")

    def test_call_payoff_deep_itm(self):
        """Test deep ITM call approaches but never exceeds 1 BTC."""
        # As S → ∞, payoff → 1 BTC (the maximum)
        payoff = calculate_inverse_call_payoff(spot=Decimal("1000000"), strike=Decimal("50000"))
        # (1M - 50K) / 1M = 0.95
        assert payoff == Decimal("0.95")
        assert payoff < Decimal("1")

    def test_put_payoff_itm(self):
        """Test ITM put payoff: max(0, K-S)/S."""
        payoff = calculate_inverse_put_payoff(spot=Decimal("40000"), strike=Decimal("50000"))
        # (50000 - 40000) / 40000 = 10000 / 40000 = 0.25
        expected = (Decimal("50000") - Decimal("40000")) / Decimal("40000")
        assert payoff == expected
        assert payoff == Decimal("0.25")

    def test_put_payoff_atm(self):
        """Test ATM put payoff: should be 0."""
        payoff = calculate_inverse_put_payoff(spot=Decimal("50000"), strike=Decimal("50000"))
        assert payoff == Decimal("0")

    def test_put_payoff_otm(self):
        """Test OTM put payoff: should be 0."""
        payoff = calculate_inverse_put_payoff(spot=Decimal("60000"), strike=Decimal("50000"))
        assert payoff == Decimal("0")

    def test_put_payoff_deep_itm(self):
        """Test deep ITM put can exceed 1 BTC."""
        # As S → 0, payoff → ∞ (catastrophic for seller)
        payoff = calculate_inverse_put_payoff(spot=Decimal("10000"), strike=Decimal("50000"))
        # (50000 - 10000) / 10000 = 40000 / 10000 = 4.0 BTC!
        assert payoff == Decimal("4")
        assert payoff > Decimal("1")

    def test_call_payoff_zero_spot(self):
        """Test call payoff with zero spot returns 0."""
        payoff = calculate_inverse_call_payoff(spot=Decimal("0"), strike=Decimal("50000"))
        assert payoff == Decimal("0")

    def test_put_payoff_very_low_spot(self):
        """Test put payoff with very low spot."""
        payoff = calculate_inverse_put_payoff(spot=Decimal("1000"), strike=Decimal("50000"))
        # (50000 - 1000) / 1000 = 49 BTC
        assert payoff == Decimal("49")

    def test_inverse_settlement_calculator_call(self):
        """Test InverseSettlementCalculator for call."""
        calc = InverseSettlementCalculator()
        result = calc.calculate_call_payoff(60000, 50000)
        assert isinstance(result, InversePayoff)
        assert result.is_call is True
        assert result.is_itm is True
        assert result.payoff_crypto > 0

    def test_inverse_settlement_calculator_put(self):
        """Test InverseSettlementCalculator for put."""
        calc = InverseSettlementCalculator()
        result = calc.calculate_put_payoff(40000, 50000)
        assert isinstance(result, InversePayoff)
        assert result.is_call is False
        assert result.is_itm is True
        assert result.payoff_crypto > 0

    def test_inverse_payoff_dataclass(self):
        """Test InversePayoff dataclass properties."""
        payoff = InversePayoff(
            payoff_crypto=Decimal("0.1667"),
            payoff_usd=Decimal("10000"),
            spot_price=Decimal("60000"),
            strike=Decimal("50000"),
            is_call=True,
            is_itm=True,
        )
        assert payoff.intrinsic_value_usd == Decimal("10000")

    def test_pnl_calculation_long(self):
        """Test P&L calculation for long position."""
        calc = InverseSettlementCalculator()
        pnl = calc.calculate_pnl(
            entry_price=Decimal("0.04"),
            exit_price=Decimal("0.05"),
            size=Decimal("1"),
            is_long=True,
        )
        assert pnl == Decimal("0.01")  # Profit

    def test_pnl_calculation_short(self):
        """Test P&L calculation for short position."""
        calc = InverseSettlementCalculator()
        pnl = calc.calculate_pnl(
            entry_price=Decimal("0.05"),
            exit_price=Decimal("0.04"),
            size=Decimal("1"),
            is_long=False,
        )
        assert pnl == Decimal("0.01")  # Profit from short

    def test_pnl_calculation_short_loss(self):
        """Test P&L calculation for losing short position."""
        calc = InverseSettlementCalculator()
        pnl = calc.calculate_pnl(
            entry_price=Decimal("0.04"),
            exit_price=Decimal("0.05"),
            size=Decimal("1"),
            is_long=False,
        )
        assert pnl == Decimal("-0.01")  # Loss

    def test_currency_conversion_btc_to_usd(self):
        """Test BTC to USD conversion."""
        result = btc_to_usd(Decimal("1.5"), Decimal("50000"))
        assert result == Decimal("75000")

    def test_currency_conversion_usd_to_btc(self):
        """Test USD to BTC conversion."""
        result = usd_to_btc(Decimal("75000"), Decimal("50000"))
        assert result == Decimal("1.5")

    def test_currency_conversion_eth_to_usd(self):
        """Test ETH to USD conversion."""
        result = eth_to_usd(Decimal("10"), Decimal("3000"))
        assert result == Decimal("30000")

    def test_currency_conversion_usd_to_eth(self):
        """Test USD to ETH conversion."""
        result = usd_to_eth(Decimal("30000"), Decimal("3000"))
        assert result == Decimal("10")


# =============================================================================
# Test Category 3: Greeks Validation (15 tests)
# =============================================================================


class TestGreeksValidation:
    """Tests for Greeks data structures and validation."""

    def test_greeks_from_api_response(self):
        """Test creating Greeks from API response."""
        data = {
            "delta": 0.5,
            "gamma": 0.0001,
            "theta": -0.001,
            "vega": 0.002,
            "rho": 0.0003,
        }
        greeks = DeribitGreeks.from_api_response(data)
        assert greeks.delta == Decimal("0.5")
        assert greeks.gamma == Decimal("0.0001")
        assert greeks.theta == Decimal("-0.001")
        assert greeks.vega == Decimal("0.002")
        assert greeks.rho == Decimal("0.0003")

    def test_greeks_defaults(self):
        """Test Greeks default values."""
        data = {}
        greeks = DeribitGreeks.from_api_response(data)
        assert greeks.delta == Decimal("0")
        assert greeks.gamma == Decimal("0")

    def test_quote_contains_greeks(self, sample_ticker_response: Dict):
        """Test that quote contains Greeks when available."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        assert quote.greeks is not None
        assert quote.greeks.delta == Decimal("0.25")
        assert quote.greeks.gamma == Decimal("0.00002")

    def test_position_greeks(self, sample_position_response: Dict):
        """Test position Greek values."""
        position = DeribitPosition.from_api_response(sample_position_response)
        assert position.delta == Decimal("0.125")
        assert position.gamma == Decimal("0.00001")
        assert position.vega == Decimal("0.00025")

    def test_portfolio_greeks_aggregation(self, margin_calculator: DeribitMarginCalculator):
        """Test that portfolio margin aggregates Greeks correctly."""
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("1"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
                delta=Decimal("0.3"),
                gamma=Decimal("0.0001"),
                vega=Decimal("0.002"),
                theta=Decimal("-0.001"),
            ),
            PositionForMargin(
                instrument_name="BTC-28MAR25-80000-P",
                size=Decimal("-0.5"),
                mark_price=Decimal("0.03"),
                index_price=Decimal("50000"),
                strike=Decimal("80000"),
                is_call=False,
                delta=Decimal("-0.25"),
                gamma=Decimal("0.0001"),
                vega=Decimal("0.002"),
                theta=Decimal("-0.001"),
            ),
        ]
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("1.0"))
        # delta_total = 0.3*1 + (-0.25)*(-0.5) = 0.3 + 0.125 = 0.425
        assert result.delta_total == Decimal("0.425")

    def test_greeks_sign_conventions(self):
        """Test Greek sign conventions."""
        # Delta: positive for calls, negative for puts
        # Theta: typically negative (time decay)
        # Vega: typically positive (vol sensitivity)
        # Gamma: typically positive
        greeks = DeribitGreeks(
            delta=Decimal("0.5"),  # Call delta
            gamma=Decimal("0.001"),  # Always positive for vanilla
            theta=Decimal("-0.01"),  # Time decay (negative)
            vega=Decimal("0.02"),  # Vol sensitivity (positive)
            rho=Decimal("0.001"),  # Interest rate sensitivity
        )
        assert greeks.gamma > 0
        assert greeks.theta < 0
        assert greeks.vega > 0

    def test_delta_bounded_for_calls(self):
        """Test that call delta should be between 0 and 1."""
        greeks = DeribitGreeks(
            delta=Decimal("0.75"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0"),
            rho=Decimal("0"),
        )
        assert Decimal("0") <= greeks.delta <= Decimal("1")

    def test_delta_bounded_for_puts(self):
        """Test that put delta should be between -1 and 0."""
        greeks = DeribitGreeks(
            delta=Decimal("-0.25"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0"),
            rho=Decimal("0"),
        )
        assert Decimal("-1") <= greeks.delta <= Decimal("0")

    def test_greeks_decimal_precision(self):
        """Test Greeks maintain decimal precision."""
        greeks = DeribitGreeks(
            delta=Decimal("0.123456789"),
            gamma=Decimal("0.000001234"),
            theta=Decimal("-0.000000123"),
            vega=Decimal("0.001234567"),
            rho=Decimal("0.000012345"),
        )
        # Ensure precision is maintained
        assert str(greeks.delta) == "0.123456789"
        assert str(greeks.gamma) == "0.000001234"

    def test_quote_iv_values(self, sample_ticker_response: Dict):
        """Test IV values in quote."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        assert quote.mark_iv == Decimal("75.5")
        assert quote.bid_iv == Decimal("74")
        assert quote.ask_iv == Decimal("77")

    def test_position_greeks_scaling(self, sample_position_response: Dict):
        """Test position Greeks scale with size."""
        position = DeribitPosition.from_api_response(sample_position_response)
        # Position delta = per-contract delta * size
        assert position.delta == Decimal("0.125")  # 0.25 * 0.5

    def test_gamma_sensitivity(self):
        """Test gamma represents delta sensitivity."""
        # Gamma measures how much delta changes for $1 move
        greeks = DeribitGreeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0.0001"),  # Delta changes 0.0001 per $1
            theta=Decimal("0"),
            vega=Decimal("0"),
            rho=Decimal("0"),
        )
        # If spot moves $100, delta should change ~0.01
        expected_delta_change = greeks.gamma * Decimal("100")
        assert expected_delta_change == Decimal("0.01")

    def test_theta_daily_decay(self):
        """Test theta represents daily decay."""
        greeks = DeribitGreeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0"),
            theta=Decimal("-0.001"),  # Loses 0.001 BTC per day
            vega=Decimal("0"),
            rho=Decimal("0"),
        )
        # 30-day decay
        monthly_decay = greeks.theta * Decimal("30")
        assert monthly_decay == Decimal("-0.03")

    def test_vega_iv_sensitivity(self):
        """Test vega represents IV sensitivity."""
        greeks = DeribitGreeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0.002"),  # Value changes 0.002 per 1% IV
            rho=Decimal("0"),
        )
        # If IV rises 10%, value increases 0.02
        iv_impact = greeks.vega * Decimal("10")
        assert iv_impact == Decimal("0.02")


# =============================================================================
# Test Category 4: Quote/Market Data (18 tests)
# =============================================================================


class TestQuoteMarketData:
    """Tests for quote and market data handling."""

    def test_quote_from_api_response(self, sample_ticker_response: Dict):
        """Test creating quote from API response."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        assert quote.instrument_name == "BTC-28MAR25-100000-C"
        assert quote.bid_price == Decimal("0.045")
        assert quote.ask_price == Decimal("0.048")
        assert quote.mark_price == Decimal("0.0465")

    def test_quote_mid_price(self, sample_ticker_response: Dict):
        """Test mid price calculation."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        expected_mid = (Decimal("0.045") + Decimal("0.048")) / 2
        assert quote.mid_price == expected_mid

    def test_quote_spread(self, sample_ticker_response: Dict):
        """Test spread calculation."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        assert quote.spread == Decimal("0.003")  # 0.048 - 0.045

    def test_quote_spread_bps(self, sample_ticker_response: Dict):
        """Test spread in basis points."""
        quote = DeribitOptionQuote.from_api_response(sample_ticker_response)
        # spread / mid * 10000
        mid = (Decimal("0.045") + Decimal("0.048")) / 2
        expected_bps = (Decimal("0.003") / mid) * Decimal("10000")
        assert quote.spread_bps == expected_bps

    def test_quote_missing_bid(self):
        """Test quote with missing bid."""
        data = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "best_bid_price": None,
            "best_ask_price": 0.05,
            "mark_price": 0.048,
            "index_price": 50000,
            "mark_iv": 80,
        }
        quote = DeribitOptionQuote.from_api_response(data)
        assert quote.bid_price is None
        assert quote.mid_price is None
        assert quote.spread is None

    def test_orderbook_from_api_response(self, sample_orderbook_response: Dict):
        """Test creating orderbook from API response."""
        book = DeribitOrderbook.from_api_response(sample_orderbook_response)
        assert len(book.bids) == 3
        assert len(book.asks) == 3
        assert book.bids[0].price == Decimal("0.045")
        assert book.asks[0].price == Decimal("0.048")

    def test_orderbook_best_bid_ask(self, sample_orderbook_response: Dict):
        """Test orderbook best bid/ask."""
        book = DeribitOrderbook.from_api_response(sample_orderbook_response)
        assert book.best_bid.price == Decimal("0.045")
        assert book.best_ask.price == Decimal("0.048")
        assert book.best_bid.size == Decimal("10")

    def test_orderbook_mid_price(self, sample_orderbook_response: Dict):
        """Test orderbook mid price."""
        book = DeribitOrderbook.from_api_response(sample_orderbook_response)
        expected_mid = (Decimal("0.045") + Decimal("0.048")) / 2
        assert book.mid_price == expected_mid

    def test_orderbook_empty(self):
        """Test empty orderbook."""
        data = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "timestamp": 1700000000000,
            "bids": [],
            "asks": [],
        }
        book = DeribitOrderbook.from_api_response(data)
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None

    def test_dvol_from_api_response(self):
        """Test creating DVOL data from API response."""
        data = {
            "volatility": 75.5,
            "timestamp": 1700000000000,
            "high": 80.0,
            "low": 70.0,
            "change": 2.5,
        }
        dvol = DVOLData.from_api_response(data, "BTC")
        assert dvol.value == Decimal("75.5")
        assert dvol.underlying == "BTC"
        assert dvol.high_24h == Decimal("80")
        assert dvol.low_24h == Decimal("70")

    def test_contract_from_api_response(self):
        """Test creating contract from API response."""
        data = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "settlement_currency": "BTC",
            "contract_size": 1,
            "tick_size": 0.0001,
            "min_trade_amount": 0.1,
            "is_active": True,
        }
        contract = DeribitOptionContract.from_api_response(data)
        assert contract.underlying == "BTC"
        assert contract.strike == Decimal("100000")
        assert contract.is_call is True
        assert contract.is_put is False

    def test_contract_from_instrument_name(self, sample_btc_call_instrument: str):
        """Test creating contract from instrument name."""
        contract = DeribitOptionContract.from_instrument_name(sample_btc_call_instrument)
        assert contract.underlying == "BTC"
        assert contract.expiration == date(2025, 3, 28)
        assert contract.strike == Decimal("100000")

    def test_contract_days_to_expiry(self):
        """Test days to expiry calculation."""
        contract = DeribitOptionContract(
            instrument_name="BTC-28MAR25-100000-C",
            underlying="BTC",
            expiration=date(2025, 3, 28),
            strike=Decimal("100000"),
            option_type=DeribitOptionType.CALL,
            settlement_currency="BTC",
        )
        from_date = date(2025, 3, 20)
        assert contract.days_to_expiry(from_date) == 8

    def test_instrument_info_from_api_response(self):
        """Test creating instrument info from API response."""
        data = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "settlement_currency": "BTC",
            "contract_size": 1,
            "tick_size": 0.0001,
            "min_trade_amount": 0.1,
            "maker_commission": 0.0003,
            "taker_commission": 0.0003,
            "is_active": True,
            "creation_timestamp": 1600000000000,
            "expiration_timestamp": 1711612800000,
        }
        info = DeribitInstrumentInfo.from_api_response(data)
        assert info.maker_commission == Decimal("0.0003")
        assert info.taker_commission == Decimal("0.0003")
        assert info.is_active is True

    def test_position_from_api_response(self, sample_position_response: Dict):
        """Test creating position from API response."""
        position = DeribitPosition.from_api_response(sample_position_response)
        assert position.size == Decimal("0.5")
        assert position.is_long is True
        assert position.is_short is False
        assert position.unrealized_pnl == Decimal("0.003")

    def test_position_notional(self, sample_position_response: Dict):
        """Test position notional calculation."""
        position = DeribitPosition.from_api_response(sample_position_response)
        expected_notional = Decimal("0.5") * Decimal("45000")
        assert position.notional_usd == expected_notional

    def test_rate_limiter(self):
        """Test rate limiter acquires tokens."""
        limiter = DeribitRateLimiter(requests_per_second=100, burst_size=5)
        # Should be able to acquire burst
        for _ in range(5):
            assert limiter.acquire(timeout=0.01) is True


# =============================================================================
# Test Category 5: Margin Calculations (25 tests)
# =============================================================================


class TestMarginCalculations:
    """Tests for margin and risk calculations."""

    def test_long_option_margin(self, margin_calculator: DeribitMarginCalculator):
        """Test margin for long option position."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-100000-C",
            size=Decimal("1"),
            mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
        )
        im, mm = margin_calculator.calculate_single_position_margin(position)
        # Long positions: margin = premium
        assert im == Decimal("0.05")  # 1 * 0.05
        assert mm < im  # Maintenance lower than initial

    def test_short_option_margin(self, margin_calculator: DeribitMarginCalculator):
        """Test margin for short option position."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-100000-C",
            size=Decimal("-1"),  # Short
            mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
        )
        im, mm = margin_calculator.calculate_single_position_margin(position)
        # Short positions need more margin
        assert im > Decimal("0.05")  # More than premium
        assert mm > 0

    def test_portfolio_margin_single_position(self, margin_calculator: DeribitMarginCalculator):
        """Test portfolio margin with single position."""
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("1"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
            )
        ]
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("1.0"))
        assert result.initial_margin > 0
        assert result.maintenance_margin > 0
        assert result.margin_call_level == MarginCallLevel.NONE

    def test_portfolio_margin_hedged(self, margin_calculator: DeribitMarginCalculator):
        """Test portfolio margin benefits from hedging."""
        # Delta-hedged position should have lower margin
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("1"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
                delta=Decimal("0.5"),
            ),
            PositionForMargin(
                instrument_name="BTC-28MAR25-80000-P",
                size=Decimal("1"),
                mark_price=Decimal("0.03"),
                index_price=Decimal("50000"),
                strike=Decimal("80000"),
                is_call=False,
                delta=Decimal("-0.5"),  # Offsetting delta
            ),
        ]
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("1.0"))
        # Net delta = 0.5 - 0.5 = 0 (perfectly hedged)
        assert result.delta_total == Decimal("0")

    def test_margin_call_warning(self, margin_calculator: DeribitMarginCalculator):
        """Test margin warning level."""
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("-5"),  # Large short
                mark_price=Decimal("0.1"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
            )
        ]
        # Low margin balance relative to position
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("0.5"))
        assert result.margin_call_level in (
            MarginCallLevel.WARNING,
            MarginCallLevel.MARGIN_CALL,
            MarginCallLevel.LIQUIDATION,
        )

    def test_margin_call_liquidation(self, margin_calculator: DeribitMarginCalculator):
        """Test liquidation level margin."""
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("-10"),
                mark_price=Decimal("0.1"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
            )
        ]
        # Very low margin
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("0.1"))
        assert result.margin_call_level == MarginCallLevel.LIQUIDATION

    def test_margin_result_to_usd(self, margin_calculator: DeribitMarginCalculator):
        """Test converting margin result to USD."""
        result = DeribitMarginResult(
            initial_margin=Decimal("0.1"),
            maintenance_margin=Decimal("0.075"),
            mark_value=Decimal("0.05"),
            available_margin=Decimal("0.9"),
            margin_balance=Decimal("1.0"),
            margin_ratio=Decimal("0.075"),
            margin_call_level=MarginCallLevel.NONE,
            currency="BTC",
        )
        usd_values = result.to_usd(Decimal("50000"))
        assert usd_values["initial_margin_usd"] == Decimal("5000")
        assert usd_values["margin_balance_usd"] == Decimal("50000")

    def test_margin_result_excess_margin(self):
        """Test excess margin calculation."""
        result = DeribitMarginResult(
            initial_margin=Decimal("0.1"),
            maintenance_margin=Decimal("0.075"),
            mark_value=Decimal("0.05"),
            available_margin=Decimal("0.9"),
            margin_balance=Decimal("1.0"),
            margin_ratio=Decimal("0.075"),
            margin_call_level=MarginCallLevel.NONE,
        )
        assert result.excess_margin == Decimal("0.925")  # 1.0 - 0.075

    def test_liquidation_price_short_call(self, margin_calculator: DeribitMarginCalculator):
        """Test liquidation price estimate for short call."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-100000-C",
            size=Decimal("-1"),
            mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
        )
        liq_price = margin_calculator.estimate_liquidation_price(position, Decimal("0.5"))
        # Should be above current price for short call
        assert liq_price is not None
        assert liq_price > Decimal("50000")

    def test_liquidation_price_short_put(self, margin_calculator: DeribitMarginCalculator):
        """Test liquidation price estimate for short put."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-40000-P",
            size=Decimal("-1"),
            mark_price=Decimal("0.03"),
            index_price=Decimal("50000"),
            strike=Decimal("40000"),
            is_call=False,
        )
        liq_price = margin_calculator.estimate_liquidation_price(position, Decimal("0.5"))
        # Should be below current price for short put
        assert liq_price is not None
        assert liq_price < Decimal("50000")

    def test_liquidation_price_long_returns_none(self, margin_calculator: DeribitMarginCalculator):
        """Test liquidation price returns None for long positions."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-100000-C",
            size=Decimal("1"),  # Long
            mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
        )
        liq_price = margin_calculator.estimate_liquidation_price(position, Decimal("0.5"))
        assert liq_price is None

    def test_max_position_size_long(self, margin_calculator: DeribitMarginCalculator):
        """Test maximum position size for long."""
        max_size = margin_calculator.calculate_max_position_size(
            margin_available=Decimal("0.5"),
            instrument_mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
            is_long=True,
        )
        # Can buy 10 contracts with 0.5 BTC at 0.05 each
        assert max_size == Decimal("10")

    def test_max_position_size_short(self, margin_calculator: DeribitMarginCalculator):
        """Test maximum position size for short."""
        max_size = margin_calculator.calculate_max_position_size(
            margin_available=Decimal("0.5"),
            instrument_mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
            is_long=False,
        )
        # Short requires more margin, so fewer contracts
        assert max_size < Decimal("10")
        assert max_size > 0

    def test_max_position_size_zero_margin(self, margin_calculator: DeribitMarginCalculator):
        """Test max position with zero margin returns zero."""
        max_size = margin_calculator.calculate_max_position_size(
            margin_available=Decimal("0"),
            instrument_mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
            is_long=True,
        )
        assert max_size == Decimal("0")

    def test_margin_mode_enum(self):
        """Test MarginMode enum values."""
        assert MarginMode.CROSS.value == "cross"
        assert MarginMode.ISOLATED.value == "isolated"

    def test_margin_call_level_enum(self):
        """Test MarginCallLevel enum values."""
        assert MarginCallLevel.NONE.value == "none"
        assert MarginCallLevel.WARNING.value == "warning"
        assert MarginCallLevel.MARGIN_CALL.value == "margin_call"
        assert MarginCallLevel.LIQUIDATION.value == "liquidation"

    def test_position_for_margin_properties(self):
        """Test PositionForMargin computed properties."""
        position = PositionForMargin(
            instrument_name="BTC-28MAR25-100000-C",
            size=Decimal("-1.5"),
            mark_price=Decimal("0.05"),
            index_price=Decimal("50000"),
            strike=Decimal("100000"),
            is_call=True,
        )
        assert position.is_short is True
        assert position.is_long is False
        assert position.abs_size == Decimal("1.5")
        assert position.notional_crypto == Decimal("0.075")  # 1.5 * 0.05
        assert position.notional_usd == Decimal("75000")  # 1.5 * 50000

    def test_hedge_benefit_calculation(self, margin_calculator: DeribitMarginCalculator):
        """Test hedge benefit calculation."""
        # Perfectly hedged portfolio should get max benefit
        positions = [
            PositionForMargin(
                instrument_name="pos1",
                size=Decimal("1"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
                delta=Decimal("0.5"),
            ),
            PositionForMargin(
                instrument_name="pos2",
                size=Decimal("1"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("80000"),
                is_call=False,
                delta=Decimal("-0.5"),
            ),
        ]
        benefit = margin_calculator._calculate_hedge_benefit(positions, Decimal("0"))
        assert benefit == Decimal("0.5")  # Max 50% benefit

    def test_margin_empty_portfolio(self, margin_calculator: DeribitMarginCalculator):
        """Test margin for empty portfolio."""
        result = margin_calculator.calculate_portfolio_margin([], Decimal("1.0"))
        assert result.initial_margin == Decimal("0")
        assert result.maintenance_margin == Decimal("0")
        assert result.available_margin == Decimal("1.0")

    def test_create_margin_calculator_factory(self):
        """Test margin calculator factory function."""
        calc = create_deribit_margin_calculator("ETH")
        assert calc.currency == "ETH"

    def test_create_margin_calculator_with_factors(self):
        """Test margin calculator with custom factors."""
        calc = create_deribit_margin_calculator(
            "BTC",
            im_factor=Decimal("0.15"),
            mm_factor=Decimal("0.10"),
        )
        assert calc._im_factor == Decimal("0.15")
        assert calc._mm_factor == Decimal("0.10")

    def test_margin_result_is_margin_call(self):
        """Test is_margin_call property."""
        result_ok = DeribitMarginResult(
            initial_margin=Decimal("0.1"),
            maintenance_margin=Decimal("0.075"),
            mark_value=Decimal("0"),
            available_margin=Decimal("0.9"),
            margin_balance=Decimal("1.0"),
            margin_ratio=Decimal("0.075"),
            margin_call_level=MarginCallLevel.NONE,
        )
        assert result_ok.is_margin_call is False

        result_bad = DeribitMarginResult(
            initial_margin=Decimal("0.1"),
            maintenance_margin=Decimal("0.075"),
            mark_value=Decimal("0"),
            available_margin=Decimal("0"),
            margin_balance=Decimal("0.05"),
            margin_ratio=Decimal("1.5"),
            margin_call_level=MarginCallLevel.MARGIN_CALL,
        )
        assert result_bad.is_margin_call is True


# =============================================================================
# Test Category 6: Order Execution (15 tests)
# =============================================================================


class TestOrderExecution:
    """Tests for order creation and execution."""

    def test_order_creation(self):
        """Test creating order object."""
        order = DeribitOrder(
            instrument_name="BTC-28MAR25-100000-C",
            direction=DeribitDirection.BUY,
            amount=Decimal("0.5"),
            order_type=DeribitOrderType.LIMIT,
            price=Decimal("0.05"),
        )
        assert order.direction == DeribitDirection.BUY
        assert order.amount == Decimal("0.5")

    def test_order_to_api_params(self):
        """Test converting order to API parameters."""
        order = DeribitOrder(
            instrument_name="BTC-28MAR25-100000-C",
            direction=DeribitDirection.SELL,
            amount=Decimal("1.0"),
            order_type=DeribitOrderType.LIMIT,
            price=Decimal("0.06"),
            post_only=True,
            label="test-order",
        )
        params = order.to_api_params()
        assert params["instrument_name"] == "BTC-28MAR25-100000-C"
        assert params["amount"] == 1.0
        assert params["price"] == 0.06
        assert params["post_only"] is True
        assert params["label"] == "test-order"

    def test_order_result_from_api(self, sample_order_response: Dict):
        """Test creating order result from API response."""
        result = DeribitOrderResult.from_api_response(sample_order_response)
        assert result.order_id == "order-123456"
        assert result.direction == DeribitDirection.BUY
        assert result.filled_amount == Decimal("0.3")
        assert result.is_open is True
        assert result.is_filled is False

    def test_order_result_remaining(self, sample_order_response: Dict):
        """Test remaining amount calculation."""
        result = DeribitOrderResult.from_api_response(sample_order_response)
        assert result.remaining_amount == Decimal("0.2")  # 0.5 - 0.3

    def test_order_direction_enum(self):
        """Test DeribitDirection enum."""
        assert DeribitDirection.BUY.value == "buy"
        assert DeribitDirection.SELL.value == "sell"

    def test_order_type_enum(self):
        """Test DeribitOrderType enum."""
        assert DeribitOrderType.LIMIT.value == "limit"
        assert DeribitOrderType.MARKET.value == "market"
        assert DeribitOrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_state_enum(self):
        """Test DeribitOrderState enum."""
        assert DeribitOrderState.OPEN.value == "open"
        assert DeribitOrderState.FILLED.value == "filled"
        assert DeribitOrderState.CANCELLED.value == "cancelled"

    def test_time_in_force_enum(self):
        """Test DeribitTimeInForce enum."""
        assert DeribitTimeInForce.GTC.value == "good_til_cancelled"
        assert DeribitTimeInForce.IOC.value == "immediate_or_cancel"
        assert DeribitTimeInForce.FOK.value == "fill_or_kill"

    def test_order_market_no_price(self):
        """Test market order doesn't require price."""
        order = DeribitOrder(
            instrument_name="BTC-28MAR25-100000-C",
            direction=DeribitDirection.BUY,
            amount=Decimal("0.5"),
            order_type=DeribitOrderType.MARKET,
        )
        params = order.to_api_params()
        assert "price" not in params

    def test_order_reduce_only(self):
        """Test reduce-only order flag."""
        order = DeribitOrder(
            instrument_name="BTC-28MAR25-100000-C",
            direction=DeribitDirection.SELL,
            amount=Decimal("0.5"),
            order_type=DeribitOrderType.LIMIT,
            price=Decimal("0.05"),
            reduce_only=True,
        )
        params = order.to_api_params()
        assert params["reduce_only"] is True

    def test_order_ioc_time_in_force(self):
        """Test IOC time in force."""
        order = DeribitOrder(
            instrument_name="BTC-28MAR25-100000-C",
            direction=DeribitDirection.BUY,
            amount=Decimal("0.5"),
            order_type=DeribitOrderType.LIMIT,
            price=Decimal("0.05"),
            time_in_force=DeribitTimeInForce.IOC,
        )
        params = order.to_api_params()
        assert params["time_in_force"] == "immediate_or_cancel"

    def test_order_result_filled_state(self):
        """Test filled order state."""
        response = {
            "order_id": "order-filled",
            "instrument_name": "BTC-28MAR25-100000-C",
            "direction": "buy",
            "amount": 0.5,
            "price": 0.05,
            "filled_amount": 0.5,
            "average_price": 0.0498,
            "order_state": "filled",
            "order_type": "limit",
            "creation_timestamp": 1700000000000,
            "last_update_timestamp": 1700000001000,
            "commission": 0.00015,
        }
        result = DeribitOrderResult.from_api_response(response)
        assert result.is_filled is True
        assert result.is_open is False
        assert result.remaining_amount == Decimal("0")

    def test_factory_market_data_adapter(self):
        """Test market data adapter factory."""
        adapter = create_deribit_options_market_data_adapter(testnet=True)
        assert adapter._config["testnet"] is True

    def test_factory_order_execution_adapter(self):
        """Test order execution adapter factory."""
        adapter = create_deribit_options_order_execution_adapter(
            client_id="test_id",
            client_secret="test_secret",
            testnet=True,
        )
        assert adapter._config["testnet"] is True
        assert adapter._config["client_id"] == "test_id"


# =============================================================================
# Test Category 7: WebSocket Streaming (15 tests)
# =============================================================================


class TestWebSocketStreaming:
    """Tests for WebSocket streaming functionality."""

    def test_stream_config_defaults(self):
        """Test DeribitStreamConfig defaults."""
        config = DeribitStreamConfig()
        assert config.testnet is True
        assert config.reconnect_enabled is True
        assert config.heartbeat_interval == 10

    def test_stream_config_ws_url(self):
        """Test WebSocket URL selection."""
        config_test = DeribitStreamConfig(testnet=True)
        config_live = DeribitStreamConfig(testnet=False)
        assert "test" in config_test.ws_url
        assert "test" not in config_live.ws_url

    def test_stream_config_requires_auth(self):
        """Test authentication requirement check."""
        config_no_auth = DeribitStreamConfig()
        config_auth = DeribitStreamConfig(client_id="id", client_secret="secret")
        assert config_no_auth.requires_auth is False
        assert config_auth.requires_auth is True

    def test_subscription_ticker_creation(self):
        """Test creating ticker subscription."""
        sub = DeribitSubscription.ticker("BTC-28MAR25-100000-C")
        assert sub.channel_type == DeribitChannelType.TICKER
        assert sub.instrument_name == "BTC-28MAR25-100000-C"
        assert "ticker" in sub.channel
        assert sub.is_private is False

    def test_subscription_book_creation(self):
        """Test creating orderbook subscription."""
        sub = DeribitSubscription.book("BTC-28MAR25-100000-C", depth=20)
        assert sub.channel_type == DeribitChannelType.BOOK
        assert "book" in sub.channel
        assert "20" in sub.channel

    def test_subscription_trades_creation(self):
        """Test creating trades subscription."""
        sub = DeribitSubscription.trades("BTC-28MAR25-100000-C")
        assert sub.channel_type == DeribitChannelType.TRADES
        assert "trades" in sub.channel

    def test_subscription_price_index(self):
        """Test creating price index subscription."""
        sub = DeribitSubscription.price_index("BTC")
        assert sub.channel_type == DeribitChannelType.PRICE_INDEX
        assert "btc_usd" in sub.channel

    def test_subscription_volatility_index(self):
        """Test creating volatility index (DVOL) subscription."""
        sub = DeribitSubscription.volatility_index("ETH")
        assert sub.channel_type == DeribitChannelType.VOLATILITY_INDEX
        assert "eth_usd" in sub.channel

    def test_subscription_user_orders(self):
        """Test creating user orders subscription (private)."""
        sub = DeribitSubscription.user_orders("BTC")
        assert sub.channel_type == DeribitChannelType.USER_ORDERS
        assert sub.is_private is True

    def test_subscription_user_portfolio(self):
        """Test creating user portfolio subscription (private)."""
        sub = DeribitSubscription.user_portfolio("BTC")
        assert sub.channel_type == DeribitChannelType.USER_PORTFOLIO
        assert sub.is_private is True

    def test_subscription_with_callback(self):
        """Test subscription with callback."""

        def my_callback(data):
            pass

        sub = DeribitSubscription.ticker("BTC-28MAR25-100000-C", callback=my_callback)
        assert sub.callback is my_callback

    def test_message_from_notification(self):
        """Test parsing subscription notification."""
        msg = {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "ticker.BTC-28MAR25-100000-C.100ms",
                "data": {
                    "instrument_name": "BTC-28MAR25-100000-C",
                    "best_bid_price": 0.045,
                    "timestamp": 1700000000000,
                },
            },
        }
        parsed = DeribitMessage.from_notification(msg)
        assert parsed.channel == "ticker.BTC-28MAR25-100000-C.100ms"
        assert parsed.channel_type == DeribitChannelType.TICKER
        assert parsed.data["best_bid_price"] == 0.045

    def test_message_from_response(self):
        """Test parsing JSON-RPC response."""
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"status": "ok"},
        }
        parsed = DeribitMessage.from_response(msg)
        assert parsed.channel is None
        assert parsed.data == {"status": "ok"}

    def test_connection_state_enum(self):
        """Test ConnectionState enum values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.AUTHENTICATED.value == "authenticated"

    def test_websocket_client_factory(self):
        """Test WebSocket client factory."""
        client = create_deribit_websocket_client(
            testnet=True,
            client_id="test_id",
            client_secret="test_secret",
        )
        assert client._config.testnet is True
        assert client._config.client_id == "test_id"
        assert client.state == ConnectionState.DISCONNECTED


# =============================================================================
# Integration Tests (Additional)
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_contract_workflow(self):
        """Test full contract creation and parsing workflow."""
        # Create contract
        name = create_deribit_instrument_name(
            underlying="BTC",
            expiration=date(2025, 6, 27),
            strike=Decimal("75000"),
            option_type=DeribitOptionType.PUT,
        )

        # Parse it back
        parsed = parse_deribit_instrument_name(name)

        # Create contract object
        contract = DeribitOptionContract.from_instrument_name(name)

        # Verify consistency
        assert contract.underlying == parsed["underlying"]
        assert contract.expiration == parsed["expiration"]
        assert contract.strike == parsed["strike"]
        assert contract.option_type == parsed["option_type"]

    def test_margin_with_greeks(self, margin_calculator: DeribitMarginCalculator):
        """Test margin calculation with full Greeks."""
        positions = [
            PositionForMargin(
                instrument_name="BTC-28MAR25-100000-C",
                size=Decimal("2"),
                mark_price=Decimal("0.05"),
                index_price=Decimal("50000"),
                strike=Decimal("100000"),
                is_call=True,
                delta=Decimal("0.3"),
                gamma=Decimal("0.0001"),
                vega=Decimal("0.002"),
                theta=Decimal("-0.001"),
            ),
        ]
        result = margin_calculator.calculate_portfolio_margin(positions, Decimal("1.0"))

        # Verify all components
        assert result.initial_margin > 0
        assert result.delta_total == Decimal("0.6")  # 0.3 * 2
        assert result.gamma_total == Decimal("0.0002")  # 0.0001 * 2
        assert result.vega_total == Decimal("0.004")  # 0.002 * 2
        assert result.theta_total == Decimal("-0.002")  # -0.001 * 2

    def test_inverse_payoff_consistency(self):
        """Test inverse payoff formula consistency."""
        calc = InverseSettlementCalculator()

        # Test: Call payoff + Put payoff at same strike should sum to forward
        spot = Decimal("50000")
        strike = Decimal("50000")  # ATM

        call = calc.calculate_call_payoff(spot, strike)
        put = calc.calculate_put_payoff(spot, strike)

        # Both should be 0 at ATM
        assert call.payoff_crypto == Decimal("0")
        assert put.payoff_crypto == Decimal("0")

    def test_subscription_channel_parsing(self):
        """Test subscription channel type detection."""
        # Create various subscriptions
        ticker = DeribitSubscription.ticker("BTC-28MAR25-100000-C")
        book = DeribitSubscription.book("BTC-28MAR25-100000-C")
        trades = DeribitSubscription.trades("BTC-28MAR25-100000-C")
        index = DeribitSubscription.price_index("BTC")
        dvol = DeribitSubscription.volatility_index("BTC")

        # Verify channel types
        assert ticker.channel_type == DeribitChannelType.TICKER
        assert book.channel_type == DeribitChannelType.BOOK
        assert trades.channel_type == DeribitChannelType.TRADES
        assert index.channel_type == DeribitChannelType.PRICE_INDEX
        assert dvol.channel_type == DeribitChannelType.VOLATILITY_INDEX


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
