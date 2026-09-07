# -*- coding: utf-8 -*-
"""
tests/test_alpaca_adapters.py
Unit tests for Alpaca exchange adapters.

These tests verify the Alpaca adapter implementations work correctly
without requiring actual API credentials (using mocks where needed).
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from typing import Any, Dict


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def alpaca_config() -> Dict[str, Any]:
    """Basic Alpaca configuration for testing."""
    return {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "paper": True,
        "feed": "iex",
        "extended_hours": False,
        "allow_extended_hours": True,
        "include_regulatory_fees": True,
    }


# =============================================================================
# Test Registry and Models
# =============================================================================


class TestAlpacaModels:
    """Tests for Alpaca-related data models."""

    def test_exchange_vendor_alpaca_exists(self):
        """Test that ALPACA vendor is defined in ExchangeVendor enum."""
        from adapters.models import ExchangeVendor

        assert hasattr(ExchangeVendor, "ALPACA")
        assert ExchangeVendor.ALPACA.value == "alpaca"

    def test_market_type_equity_exists(self):
        """Test that EQUITY market type is defined."""
        from adapters.models import MarketType

        assert hasattr(MarketType, "EQUITY")

    def test_us_equity_sessions_defined(self):
        """Test that US equity trading sessions are defined."""
        from adapters.models import US_EQUITY_SESSIONS

        assert len(US_EQUITY_SESSIONS) >= 3  # Pre-market, Regular, After-hours

    def test_create_us_equity_calendar(self):
        """Test US equity calendar creation."""
        from adapters.models import create_us_equity_calendar, ExchangeVendor

        calendar = create_us_equity_calendar(ExchangeVendor.ALPACA)
        assert calendar is not None
        assert calendar.timezone == "America/New_York"


class TestAlpacaRegistry:
    """Tests for Alpaca adapter registration."""

    def test_alpaca_adapters_registered(self):
        """Test that Alpaca adapters are registered in the registry."""
        from adapters.registry import get_registry, AdapterType
        from adapters.models import ExchangeVendor

        # Import to trigger registration
        import adapters.alpaca  # noqa: F401

        registry = get_registry()

        # Check that all adapter types are registered
        for adapter_type in [
            AdapterType.MARKET_DATA,
            AdapterType.FEE,
            AdapterType.TRADING_HOURS,
            AdapterType.EXCHANGE_INFO,
            AdapterType.ORDER_EXECUTION,
        ]:
            reg = registry.get_registration(ExchangeVendor.ALPACA, adapter_type)
            assert reg is not None, f"{adapter_type} not registered for Alpaca"

    def test_create_market_data_adapter_from_registry(self, alpaca_config):
        """Test creating market data adapter via registry."""
        from adapters.registry import create_market_data_adapter

        # This will fail without SDK, but should not raise import error
        try:
            adapter = create_market_data_adapter("alpaca", alpaca_config)
            assert adapter is not None
        except ImportError as e:
            # Expected if alpaca-py not installed
            assert "alpaca" in str(e).lower()


# =============================================================================
# Test Fee Adapter
# =============================================================================


class TestAlpacaFeeAdapter:
    """Tests for Alpaca fee computation."""

    def test_fee_adapter_init(self, alpaca_config):
        """Test fee adapter initialization."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )
        assert adapter is not None
        assert adapter.vendor == ExchangeVendor.ALPACA

    def test_commission_free_buys(self, alpaca_config):
        """Test that buy orders have zero commission."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor
        from core_models import Side

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )

        # Buy should be free
        fee = adapter.compute_fee(
            notional=10000.0,
            side=Side.BUY,
            liquidity="taker",
            qty=100,
        )
        assert fee == 0.0

    def test_regulatory_fees_on_sells(self, alpaca_config):
        """Test that sells have small regulatory fees."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor
        from core_models import Side

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )

        # Sell should have SEC + TAF fees
        fee = adapter.compute_fee(
            notional=10000.0,
            side=Side.SELL,
            liquidity="taker",
            qty=100,
            price=100.0,
        )

        # Fee should be positive but very small
        assert fee > 0
        assert fee < 1.0  # Less than $1 for $10k trade

    def test_regulatory_fees_disabled(self, alpaca_config):
        """Test that regulatory fees can be disabled."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor
        from core_models import Side

        config = {**alpaca_config, "include_regulatory_fees": False}
        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        fee = adapter.compute_fee(
            notional=10000.0,
            side=Side.SELL,
            liquidity="taker",
            qty=100,
        )
        assert fee == 0.0

    def test_get_fee_schedule(self, alpaca_config):
        """Test getting fee schedule."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor, FeeStructure

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )

        schedule = adapter.get_fee_schedule()
        assert schedule.structure == FeeStructure.FLAT
        assert schedule.maker_rate == 0.0
        assert schedule.taker_rate == 0.0

    def test_effective_rates_zero(self, alpaca_config):
        """Test that effective rates are zero."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )

        maker, taker = adapter.get_effective_rates()
        assert maker == 0.0
        assert taker == 0.0

    def test_options_fee(self, alpaca_config):
        """Test options trading fee calculation."""
        from adapters.alpaca.fees import AlpacaFeeAdapter
        from adapters.models import ExchangeVendor

        adapter = AlpacaFeeAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=alpaca_config,
        )

        fee = adapter.compute_options_fee(contracts=10, opening=True)
        assert fee == 6.5  # $0.65 per contract


# =============================================================================
# Test Trading Hours Adapter
# =============================================================================


class TestAlpacaTradingHoursAdapter:
    """Tests for Alpaca trading hours."""

    def test_trading_hours_init(self, alpaca_config):
        """Test trading hours adapter initialization."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import ExchangeVendor

        # Don't try to load from API (no credentials)
        config = {**alpaca_config, "use_alpaca_calendar": False}

        adapter = AlpacaTradingHoursAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )
        assert adapter is not None

    def test_market_closed_weekend(self, alpaca_config):
        """Test that market is closed on weekends."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import ExchangeVendor
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = {**alpaca_config, "use_alpaca_calendar": False}
        adapter = AlpacaTradingHoursAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        # Saturday noon
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        ts = int(saturday.timestamp() * 1000)

        assert adapter.is_market_open(ts) is False

    def test_market_open_regular_hours(self, alpaca_config):
        """Test market open during regular hours."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import ExchangeVendor, SessionType
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = {**alpaca_config, "use_alpaca_calendar": False}
        adapter = AlpacaTradingHoursAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        # Monday 10:00 AM ET
        monday = datetime(2024, 1, 8, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        ts = int(monday.timestamp() * 1000)

        assert adapter.is_market_open(ts, session_type=SessionType.REGULAR) is True

    def test_pre_market_hours(self, alpaca_config):
        """Test pre-market hours detection."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import ExchangeVendor, SessionType
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = {**alpaca_config, "use_alpaca_calendar": False}
        adapter = AlpacaTradingHoursAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        # Monday 5:00 AM ET (pre-market)
        monday = datetime(2024, 1, 8, 5, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        ts = int(monday.timestamp() * 1000)

        assert adapter.is_market_open(ts, session_type=SessionType.PRE_MARKET) is True
        assert adapter.is_market_open(ts, session_type=SessionType.REGULAR) is False

    def test_get_calendar(self, alpaca_config):
        """Test getting market calendar."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import ExchangeVendor

        config = {**alpaca_config, "use_alpaca_calendar": False}
        adapter = AlpacaTradingHoursAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        calendar = adapter.get_calendar()
        assert calendar is not None
        assert calendar.timezone == "America/New_York"
        assert len(calendar.sessions) >= 3


# =============================================================================
# Test Exchange Info Adapter
# =============================================================================


class TestAlpacaExchangeInfoAdapter:
    """Tests for Alpaca exchange info."""

    def test_exchange_info_init(self, alpaca_config):
        """Test exchange info adapter initialization."""
        from adapters.alpaca.exchange_info import AlpacaExchangeInfoAdapter
        from adapters.models import ExchangeVendor

        # Don't auto-refresh (would require API)
        config = {**alpaca_config, "auto_refresh": False}

        adapter = AlpacaExchangeInfoAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )
        assert adapter is not None

    def test_get_symbols_empty_cache(self, alpaca_config):
        """Test getting symbols with empty cache."""
        from adapters.alpaca.exchange_info import AlpacaExchangeInfoAdapter
        from adapters.models import ExchangeVendor

        config = {**alpaca_config, "auto_refresh": False}
        adapter = AlpacaExchangeInfoAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        # Should return empty list without API call
        symbols = adapter.get_symbols()
        assert symbols == []

    def test_search_symbols(self, alpaca_config):
        """Test symbol search functionality."""
        from adapters.alpaca.exchange_info import AlpacaExchangeInfoAdapter
        from adapters.models import ExchangeVendor

        config = {**alpaca_config, "auto_refresh": False}
        adapter = AlpacaExchangeInfoAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=config,
        )

        # Should return empty without data
        matches = adapter.search_symbols("AAPL")
        assert matches == []


# =============================================================================
# Test Configuration
# =============================================================================


class TestAlpacaConfig:
    """Tests for Alpaca configuration."""

    def test_alpaca_config_creation(self):
        """Test creating Alpaca config."""
        from adapters.config import AlpacaConfig

        config = AlpacaConfig(
            api_key="${ALPACA_API_KEY}",
            api_secret="${ALPACA_API_SECRET}",
            paper=True,
            feed="iex",
        )
        assert config.paper is True
        assert config.feed == "iex"

    def test_exchange_config_with_alpaca(self):
        """Test ExchangeConfig with Alpaca vendor."""
        from adapters.config import ExchangeConfig

        config = ExchangeConfig(
            vendor="alpaca",
            market_type="EQUITY",
        )
        assert config.vendor == "alpaca"
        assert config.exchange_vendor.value == "alpaca"

    def test_exchange_config_from_dict(self):
        """Test creating ExchangeConfig from dict."""
        from adapters.config import ExchangeConfig

        data = {
            "vendor": "alpaca",
            "market_type": "EQUITY",
            "alpaca": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "paper": True,
            },
        }
        config = ExchangeConfig.from_dict(data)
        assert config.vendor == "alpaca"
        assert config.alpaca.paper is True


# =============================================================================
# Test Bar Source Wrapper
# =============================================================================


class TestAlpacaBarSource:
    """Tests for AlpacaBarSource wrapper."""

    def test_timeframe_parsing(self):
        """Test timeframe string parsing."""
        from impl_alpaca_public import timeframe_to_ms

        assert timeframe_to_ms("1m") == 60_000
        assert timeframe_to_ms("1h") == 3_600_000
        assert timeframe_to_ms("4h") == 14_400_000
        assert timeframe_to_ms("1d") == 86_400_000

    def test_timeframe_validation(self):
        """Test timeframe validation."""
        from impl_alpaca_public import ensure_timeframe

        # Valid timeframes should not raise
        ensure_timeframe("1m")
        ensure_timeframe("1h")
        ensure_timeframe("1d")

        # Invalid should raise
        with pytest.raises(ValueError):
            ensure_timeframe("invalid")

    def test_bar_source_init(self, alpaca_config):
        """Test AlpacaBarSource initialization."""
        from impl_alpaca_public import AlpacaBarSource, AlpacaWSConfig

        cfg = AlpacaWSConfig(
            api_key="test",
            api_secret="test",
            paper=True,
            feed="iex",
        )

        source = AlpacaBarSource(timeframe="1h", cfg=cfg)
        assert source._interval_ms == 3_600_000

    def test_bar_source_timeframe_mismatch(self, alpaca_config):
        """Test that timeframe mismatch raises error."""
        from impl_alpaca_public import AlpacaBarSource, AlpacaWSConfig

        cfg = AlpacaWSConfig(api_key="test", api_secret="test")
        source = AlpacaBarSource(timeframe="1h", cfg=cfg)

        with pytest.raises(ValueError, match="mismatch"):
            list(source.stream_bars(["AAPL"], interval_ms=60_000))

    def test_bar_source_no_symbols(self, alpaca_config):
        """Test that empty symbols list raises error."""
        from impl_alpaca_public import AlpacaBarSource, AlpacaWSConfig

        cfg = AlpacaWSConfig(api_key="test", api_secret="test")
        source = AlpacaBarSource(timeframe="1h", cfg=cfg)

        with pytest.raises(ValueError, match="No symbols"):
            list(source.stream_bars([], interval_ms=3_600_000))


# =============================================================================
# Test DI Integration
# =============================================================================


class TestDIIntegration:
    """Tests for DI registry integration."""

    def test_build_exchange_adapters_with_alpaca(self):
        """Test building exchange adapters from config."""
        from di_registry import _build_exchange_adapters

        container = {}
        config = {
            "vendor": "alpaca",
            "market_type": "EQUITY",
            "alpaca": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "paper": True,
            },
        }

        _build_exchange_adapters(config, container)

        # Should have exchange_config and vendor
        assert "exchange_config" in container
        assert "exchange_vendor" in container
        assert container["exchange_vendor"] == "alpaca"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
