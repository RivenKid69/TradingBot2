# -*- coding: utf-8 -*-
"""
tests/test_adapters_base.py
Tests for adapter base classes and registry.

Tests:
1. Model serialization/deserialization
2. Registry registration and factory functions
3. Adapter interface contracts
4. Configuration loading
"""

import pytest
from decimal import Decimal
from typing import Iterator, List, Optional, Sequence
from unittest.mock import MagicMock, patch

# Import models
from adapters.models import (
    MarketType,
    ExchangeVendor,
    FeeStructure,
    SessionType,
    ExchangeRule,
    TradingSession,
    MarketCalendar,
    FeeSchedule,
    AccountInfo,
    SymbolInfo,
    US_EQUITY_SESSIONS,
    CRYPTO_CONTINUOUS_SESSION,
    create_us_equity_calendar,
    create_crypto_calendar,
)

# Import base classes
from adapters.base import (
    BaseAdapter,
    MarketDataAdapter,
    FeeAdapter,
    TradingHoursAdapter,
    OrderExecutionAdapter,
    ExchangeInfoAdapter,
    OrderResult,
)

# Import registry
from adapters.registry import (
    AdapterType,
    AdapterRegistry,
    get_registry,
    register,
    create_market_data_adapter,
    create_fee_adapter,
    create_trading_hours_adapter,
)

from core_models import Bar, Tick, Side, Liquidity


# =========================
# Model Tests
# =========================


class TestMarketType:
    """Test MarketType enum."""

    def test_crypto_types(self):
        """Test is_crypto property."""
        assert MarketType.CRYPTO_SPOT.is_crypto is True
        assert MarketType.CRYPTO_FUTURES.is_crypto is True
        assert MarketType.CRYPTO_PERP.is_crypto is True
        assert MarketType.EQUITY.is_crypto is False

    def test_equity_types(self):
        """Test is_equity property."""
        assert MarketType.EQUITY.is_equity is True
        assert MarketType.EQUITY_OPTIONS.is_equity is True
        assert MarketType.CRYPTO_SPOT.is_equity is False

    def test_has_trading_hours(self):
        """Test has_trading_hours property."""
        assert MarketType.EQUITY.has_trading_hours is True
        assert MarketType.CRYPTO_SPOT.has_trading_hours is False


class TestExchangeVendor:
    """Test ExchangeVendor enum."""

    def test_market_type_property(self):
        """Test default market type for vendors."""
        assert ExchangeVendor.BINANCE.market_type == MarketType.CRYPTO_SPOT
        assert ExchangeVendor.ALPACA.market_type == MarketType.EQUITY


class TestExchangeRule:
    """Test ExchangeRule dataclass."""

    def test_creation(self):
        """Test basic creation."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
        )
        assert rule.symbol == "BTCUSDT"
        assert rule.tick_size == Decimal("0.01")
        assert rule.step_size == Decimal("0.001")
        assert rule.min_notional == Decimal("10")

    def test_quantize_price(self):
        """Test price quantization."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
        )
        assert rule.quantize_price(Decimal("100.456")) == Decimal("100.45")
        assert rule.quantize_price(Decimal("100.001")) == Decimal("100.00")

    def test_quantize_qty(self):
        """Test quantity quantization."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.01"),
            min_notional=Decimal("10"),
        )
        assert rule.quantize_qty(Decimal("1.2345")) == Decimal("1.23")

    def test_validate_order_success(self):
        """Test order validation - success case."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            min_qty=Decimal("0.001"),
        )
        is_valid, error = rule.validate_order(Decimal("100"), Decimal("1"))
        assert is_valid is True
        assert error is None

    def test_validate_order_min_notional(self):
        """Test order validation - below min notional."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
        )
        is_valid, error = rule.validate_order(Decimal("1"), Decimal("0.001"))
        assert is_valid is False
        assert "below minimum" in error.lower()

    def test_serialization(self):
        """Test to_dict and from_dict."""
        rule = ExchangeRule(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            market_type=MarketType.CRYPTO_SPOT,
        )
        d = rule.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["tick_size"] == "0.01"

        restored = ExchangeRule.from_dict(d)
        assert restored.symbol == rule.symbol
        assert restored.tick_size == rule.tick_size


class TestTradingSession:
    """Test TradingSession dataclass."""

    def test_creation(self):
        """Test basic creation."""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_minutes=570,  # 9:30 AM
            end_minutes=960,  # 4:00 PM
            timezone="America/New_York",
        )
        assert session.session_type == SessionType.REGULAR
        assert session.start_time_str == "09:30"
        assert session.end_time_str == "16:00"

    def test_duration(self):
        """Test duration calculation."""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_minutes=570,
            end_minutes=960,
            timezone="America/New_York",
        )
        assert session.duration_minutes == 390  # 6.5 hours


class TestFeeSchedule:
    """Test FeeSchedule dataclass."""

    def test_percentage_fee(self):
        """Test percentage fee calculation."""
        schedule = FeeSchedule(
            structure=FeeStructure.PERCENTAGE,
            maker_rate=10.0,  # 10 bps
            taker_rate=10.0,
        )
        fee = schedule.compute_fee(
            notional=10000,
            qty=1,
            is_maker=False,
        )
        assert fee == 10.0  # 10000 * 10/10000

    def test_per_share_fee(self):
        """Test per-share fee calculation."""
        schedule = FeeSchedule(
            structure=FeeStructure.PER_SHARE,
            maker_rate=0.005,
            taker_rate=0.005,
        )
        fee = schedule.compute_fee(
            notional=10000,
            qty=100,
            is_maker=False,
        )
        assert fee == 0.5  # 100 * 0.005


class TestPredefindSessions:
    """Test predefined sessions and calendars."""

    def test_us_equity_sessions(self):
        """Test US equity session definitions."""
        assert len(US_EQUITY_SESSIONS) == 3

        # Find regular session
        regular = next(s for s in US_EQUITY_SESSIONS if s.session_type == SessionType.REGULAR)
        assert regular.start_time_str == "09:30"
        assert regular.end_time_str == "16:00"

    def test_crypto_session(self):
        """Test crypto continuous session."""
        assert CRYPTO_CONTINUOUS_SESSION.session_type == SessionType.CONTINUOUS
        assert CRYPTO_CONTINUOUS_SESSION.days_of_week == (0, 1, 2, 3, 4, 5, 6)

    def test_create_us_calendar(self):
        """Test US equity calendar creation."""
        calendar = create_us_equity_calendar()
        assert calendar.market_type == MarketType.EQUITY
        assert not calendar.is_24_7

    def test_create_crypto_calendar(self):
        """Test crypto calendar creation."""
        calendar = create_crypto_calendar()
        assert calendar.market_type == MarketType.CRYPTO_SPOT
        assert calendar.is_24_7


# =========================
# Registry Tests
# =========================


class TestAdapterRegistry:
    """Test adapter registry."""

    def test_singleton(self):
        """Test registry is singleton."""
        reg1 = AdapterRegistry()
        reg2 = AdapterRegistry()
        assert reg1 is reg2

    def test_global_registry(self):
        """Test get_registry function."""
        reg = get_registry()
        assert isinstance(reg, AdapterRegistry)


class TestMockAdapter(MarketDataAdapter):
    """Mock adapter for testing."""

    def __init__(self, vendor=ExchangeVendor.UNKNOWN, config=None):
        super().__init__(vendor, config)
        self._bars = []

    def get_bars(self, symbol, timeframe, *, limit=500, start_ts=None, end_ts=None):
        return self._bars[:limit]

    def get_latest_bar(self, symbol, timeframe):
        return self._bars[-1] if self._bars else None

    def get_tick(self, symbol):
        return None

    def stream_bars(self, symbols, interval_ms):
        yield from self._bars

    def stream_ticks(self, symbols):
        return iter([])


class TestAdapterRegistration:
    """Test adapter registration and creation."""

    def test_register_adapter(self):
        """Test registering an adapter."""
        registry = get_registry()

        # Register mock adapter
        registry.register(
            vendor=ExchangeVendor.UNKNOWN,
            adapter_type=AdapterType.MARKET_DATA,
            adapter_class=TestMockAdapter,
            description="Test adapter",
        )

        # Verify registration
        reg = registry.get_registration(ExchangeVendor.UNKNOWN, AdapterType.MARKET_DATA)
        assert reg is not None
        assert reg.adapter_class == TestMockAdapter

        # Clean up
        registry.unregister(ExchangeVendor.UNKNOWN, AdapterType.MARKET_DATA)

    def test_create_adapter(self):
        """Test creating adapter from registry."""
        registry = get_registry()

        # Register
        registry.register(
            vendor=ExchangeVendor.UNKNOWN,
            adapter_type=AdapterType.MARKET_DATA,
            adapter_class=TestMockAdapter,
        )

        # Create
        adapter = registry.create_adapter(
            ExchangeVendor.UNKNOWN,
            AdapterType.MARKET_DATA,
            config={"test": True},
        )

        assert isinstance(adapter, TestMockAdapter)
        assert adapter.config.get("test") is True

        # Clean up
        registry.unregister(ExchangeVendor.UNKNOWN)


# =========================
# Base Adapter Tests
# =========================


class TestBaseAdapter:
    """Test BaseAdapter functionality."""

    def test_properties(self):
        """Test basic properties."""
        adapter = TestMockAdapter(
            vendor=ExchangeVendor.BINANCE,
            config={"key": "value"},
        )

        assert adapter.vendor == ExchangeVendor.BINANCE
        assert adapter.config == {"key": "value"}
        assert adapter.is_connected is False

    def test_connect_disconnect(self):
        """Test connect/disconnect lifecycle."""
        adapter = TestMockAdapter()

        assert adapter.is_connected is False

        result = adapter.connect()
        assert result is True
        assert adapter.is_connected is True

        adapter.disconnect()
        assert adapter.is_connected is False

    def test_context_manager(self):
        """Test context manager protocol."""
        with TestMockAdapter() as adapter:
            assert adapter.is_connected is True

        assert adapter.is_connected is False


# =========================
# Order Result Tests
# =========================


class TestOrderResult:
    """Test OrderResult dataclass."""

    def test_success_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="123",
            client_order_id="client-123",
            status="filled",
            filled_qty=Decimal("1.5"),
            filled_price=Decimal("100.50"),
        )

        assert result.success is True
        assert result.order_id == "123"
        assert result.filled_qty == Decimal("1.5")
        assert result.error_code is None

    def test_failure_result(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            error_code="INSUFFICIENT_FUNDS",
            error_message="Not enough balance",
        )

        assert result.success is False
        assert result.error_code == "INSUFFICIENT_FUNDS"
        assert result.order_id is None


# =========================
# Integration Tests
# =========================


class TestBinanceAdapterIntegration:
    """Integration tests for Binance adapters (mocked)."""

    @pytest.fixture
    def mock_binance_client(self):
        """Create mock Binance client."""
        mock = MagicMock()
        mock.get_klines.return_value = [
            [
                1700000000000,
                "100.0",
                "101.0",
                "99.0",
                "100.5",
                "1000.0",
                0,
                "100500.0",
                500,
                "500.0",
            ],
        ]
        mock.get_book_ticker.return_value = (Decimal("100.0"), Decimal("100.1"))
        mock.get_last_price.return_value = Decimal("100.05")
        return mock

    def test_binance_market_data_import(self):
        """Test that Binance market data adapter can be imported."""
        try:
            from adapters.binance import BinanceMarketDataAdapter

            assert BinanceMarketDataAdapter is not None
        except ImportError as e:
            pytest.skip(f"Binance adapter not available: {e}")

    def test_binance_fee_adapter_import(self):
        """Test that Binance fee adapter can be imported."""
        try:
            from adapters.binance import BinanceFeeAdapter

            assert BinanceFeeAdapter is not None
        except ImportError as e:
            pytest.skip(f"Binance adapter not available: {e}")


class TestAlpacaAdapterIntegration:
    """Integration tests for Alpaca adapters."""

    def test_alpaca_trading_hours(self):
        """Test Alpaca trading hours adapter."""
        try:
            from adapters.alpaca import AlpacaTradingHoursAdapter

            adapter = AlpacaTradingHoursAdapter(
                config={
                    "allow_extended_hours": True,
                }
            )

            # Crypto-style timestamps for testing
            import time

            ts = int(time.time() * 1000)

            # These should not raise
            calendar = adapter.get_calendar()
            assert calendar.market_type == MarketType.EQUITY

        except ImportError as e:
            pytest.skip(f"Alpaca adapter not available: {e}")

    def test_alpaca_fee_computation(self):
        """Test Alpaca fee adapter - commission free."""
        try:
            from adapters.alpaca import AlpacaFeeAdapter

            adapter = AlpacaFeeAdapter(
                config={
                    "include_regulatory_fees": False,
                }
            )

            # Buy should be free
            fee = adapter.compute_fee(
                notional=10000,
                side=Side.BUY,
                liquidity="taker",
            )
            assert fee == 0.0

        except ImportError as e:
            pytest.skip(f"Alpaca adapter not available: {e}")


# =========================
# Configuration Tests
# =========================


class TestAdapterConfig:
    """Test adapter configuration."""

    def test_binance_config(self):
        """Test Binance configuration."""
        try:
            from adapters.config import BinanceConfig, ExchangeConfig

            config = BinanceConfig(
                timeout=30,
                use_bnb_discount=True,
            )
            adapter_config = config.to_adapter_config()

            assert adapter_config["timeout"] == 30
            assert adapter_config["use_bnb_discount"] is True

        except ImportError as e:
            pytest.skip(f"Config module not available: {e}")

    def test_exchange_config_vendor_selection(self):
        """Test exchange config vendor selection."""
        try:
            from adapters.config import ExchangeConfig

            config = ExchangeConfig(vendor="binance")
            assert config.exchange_vendor == ExchangeVendor.BINANCE

            config = ExchangeConfig(vendor="alpaca")
            assert config.exchange_vendor == ExchangeVendor.ALPACA

        except ImportError as e:
            pytest.skip(f"Config module not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
