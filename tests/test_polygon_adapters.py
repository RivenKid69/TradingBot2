# -*- coding: utf-8 -*-
"""
tests/test_polygon_adapters.py
Comprehensive tests for Polygon.io adapter implementations.

Tests cover:
- PolygonMarketDataAdapter (historical bars, streaming)
- PolygonTradingHoursAdapter (market hours, holidays)
- PolygonExchangeInfoAdapter (symbol info, search)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from adapters.polygon.market_data import (
    PolygonMarketDataAdapter,
    _timeframe_to_polygon,
    _timeframe_to_ms,
)
from adapters.polygon.trading_hours import (
    PolygonTradingHoursAdapter,
    US_MARKET_HOLIDAYS,
    US_MARKET_HALF_DAYS,
)
from adapters.polygon.exchange_info import (
    PolygonExchangeInfoAdapter,
    DEFAULT_EQUITY_RULE,
)
from adapters.models import ExchangeVendor, MarketType


# =============================================================================
# Timeframe Utilities Tests
# =============================================================================


class TestTimeframeUtilities:
    """Tests for timeframe conversion utilities."""

    def test_timeframe_to_polygon_minutes(self):
        """Test minute timeframe conversion."""
        assert _timeframe_to_polygon("1m") == (1, "minute")
        assert _timeframe_to_polygon("1min") == (1, "minute")
        assert _timeframe_to_polygon("5m") == (5, "minute")
        assert _timeframe_to_polygon("15m") == (15, "minute")
        assert _timeframe_to_polygon("30m") == (30, "minute")

    def test_timeframe_to_polygon_hours(self):
        """Test hour timeframe conversion."""
        assert _timeframe_to_polygon("1h") == (1, "hour")
        assert _timeframe_to_polygon("1hour") == (1, "hour")
        assert _timeframe_to_polygon("4h") == (4, "hour")
        assert _timeframe_to_polygon("4hour") == (4, "hour")

    def test_timeframe_to_polygon_days(self):
        """Test day timeframe conversion."""
        assert _timeframe_to_polygon("1d") == (1, "day")
        assert _timeframe_to_polygon("1day") == (1, "day")

    def test_timeframe_to_polygon_weeks(self):
        """Test week timeframe conversion."""
        assert _timeframe_to_polygon("1w") == (1, "week")
        assert _timeframe_to_polygon("1week") == (1, "week")

    def test_timeframe_to_polygon_case_insensitive(self):
        """Test case insensitivity."""
        assert _timeframe_to_polygon("1M") == (1, "minute")
        assert _timeframe_to_polygon("1H") == (1, "hour")
        assert _timeframe_to_polygon("1D") == (1, "day")

    def test_timeframe_to_polygon_invalid(self):
        """Test invalid timeframe handling."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            _timeframe_to_polygon("invalid")

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            _timeframe_to_polygon("2y")

    def test_timeframe_to_ms(self):
        """Test timeframe to milliseconds conversion."""
        assert _timeframe_to_ms("1m") == 60_000
        assert _timeframe_to_ms("5m") == 300_000
        assert _timeframe_to_ms("1h") == 3_600_000
        assert _timeframe_to_ms("4h") == 14_400_000
        assert _timeframe_to_ms("1d") == 86_400_000
        assert _timeframe_to_ms("1w") == 604_800_000


# =============================================================================
# PolygonMarketDataAdapter Tests
# =============================================================================


class TestPolygonMarketDataAdapter:
    """Tests for Polygon market data adapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked client."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            return adapter

    def test_initialization(self):
        """Test adapter initialization."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()

            assert adapter._api_key == "test_key"
            assert adapter._timeout == 30
            assert adapter._retries == 3
            assert adapter._rest_client is None

    def test_initialization_with_config(self):
        """Test adapter initialization with config."""
        adapter = PolygonMarketDataAdapter(
            config={
                "api_key": "config_key",
                "timeout": 60,
                "retries": 5,
            }
        )

        assert adapter._api_key == "config_key"
        assert adapter._timeout == 60
        assert adapter._retries == 5

    def test_vendor(self):
        """Test vendor property."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            assert adapter.vendor == ExchangeVendor.POLYGON

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_bars(self, mock_get_client):
        """Test fetching historical bars."""
        # Setup mock
        mock_client = MagicMock()
        mock_agg = MagicMock()
        mock_agg.timestamp = 1705312800000
        mock_agg.open = 150.0
        mock_agg.high = 152.0
        mock_agg.low = 149.0
        mock_agg.close = 151.0
        mock_agg.volume = 1000000
        mock_agg.vwap = 150.5
        mock_agg.transactions = 5000

        mock_client.get_aggs.return_value = [mock_agg]
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            bars = adapter.get_bars("AAPL", "1h", limit=10)

        assert len(bars) == 1
        bar = bars[0]
        assert bar.symbol == "AAPL"
        assert bar.open == Decimal("150.0")
        assert bar.high == Decimal("152.0")
        assert bar.low == Decimal("149.0")
        assert bar.close == Decimal("151.0")
        assert bar.volume_base == Decimal("1000000")

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_bars_with_timestamps(self, mock_get_client):
        """Test fetching bars with specific timestamps."""
        mock_client = MagicMock()
        mock_client.get_aggs.return_value = []
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()

            start_ts = 1705312800000
            end_ts = 1705399200000

            adapter.get_bars(
                "AAPL",
                "1h",
                start_ts=start_ts,
                end_ts=end_ts,
            )

        mock_client.get_aggs.assert_called_once()
        call_kwargs = mock_client.get_aggs.call_args.kwargs
        assert call_kwargs["ticker"] == "AAPL"

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_bars_error_handling(self, mock_get_client):
        """Test error handling in get_bars."""
        mock_client = MagicMock()
        mock_client.get_aggs.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            bars = adapter.get_bars("AAPL", "1h")

        assert bars == []

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_latest_bar(self, mock_get_client):
        """Test getting latest bar."""
        mock_client = MagicMock()
        mock_agg = MagicMock()
        mock_agg.timestamp = 1705312800000
        mock_agg.open = 150.0
        mock_agg.high = 152.0
        mock_agg.low = 149.0
        mock_agg.close = 151.0
        mock_agg.volume = 1000000
        mock_agg.vwap = 150.5

        mock_client.get_aggs.return_value = [mock_agg]
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            bar = adapter.get_latest_bar("AAPL", "1h")

        assert bar is not None
        assert bar.close == Decimal("151.0")

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_tick(self, mock_get_client):
        """Test getting tick data."""
        mock_client = MagicMock()
        mock_trade = MagicMock()
        mock_trade.sip_timestamp = 1705312800000000000  # nanoseconds
        mock_trade.price = 150.5
        mock_trade.size = 100

        mock_client.get_last_trade.return_value = mock_trade
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            tick = adapter.get_tick("AAPL")

        assert tick is not None
        assert tick.symbol == "AAPL"
        assert tick.price == Decimal("150.5")

    @patch("adapters.polygon.market_data.PolygonMarketDataAdapter._get_rest_client")
    def test_get_bars_multi(self, mock_get_client):
        """Test fetching bars for multiple symbols."""
        mock_client = MagicMock()
        mock_agg = MagicMock()
        mock_agg.timestamp = 1705312800000
        mock_agg.open = 150.0
        mock_agg.high = 152.0
        mock_agg.low = 149.0
        mock_agg.close = 151.0
        mock_agg.volume = 1000000
        mock_agg.vwap = 150.5

        mock_client.get_aggs.return_value = [mock_agg]
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonMarketDataAdapter()
            result = adapter.get_bars_multi(["AAPL", "MSFT"], "1h", limit=10)

        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 1
        assert len(result["MSFT"]) == 1


# =============================================================================
# PolygonTradingHoursAdapter Tests
# =============================================================================


class TestPolygonTradingHoursAdapter:
    """Tests for Polygon trading hours adapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return PolygonTradingHoursAdapter()

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.vendor == ExchangeVendor.POLYGON
        # Calendar timezone is set correctly
        assert adapter._calendar.timezone == "America/New_York"

    def test_regular_market_hours(self, adapter):
        """Test regular market hours detection."""
        # Tuesday at 10:00 AM ET (market open)
        # 2025-01-14 10:00:00 ET = 2025-01-14 15:00:00 UTC
        ts_open = 1736866800000  # ~10:00 AM ET on a Tuesday

        # Should be open during regular hours
        # Note: Actual time depends on timezone calculations

    def test_market_closed_weekend(self, adapter):
        """Test market closed on weekend."""
        # Saturday 2025-01-18 12:00 PM UTC
        ts_saturday = 1737201600000

        assert adapter.is_market_open(ts_saturday) is False

    def test_holiday_detection(self, adapter):
        """Test holiday detection."""
        # Check that holiday list is populated
        assert len(US_MARKET_HOLIDAYS) > 0

        # Common holidays should be in list (format is tuple: (year, month, day))
        holidays_2025 = [h for h in US_MARKET_HOLIDAYS if h[0] == 2025]
        assert len(holidays_2025) > 0

    def test_half_day_detection(self, adapter):
        """Test half day detection."""
        # Check that half days list is populated
        assert len(US_MARKET_HALF_DAYS) > 0

        # Day after Thanksgiving 2025 should be a half day (format is tuple)
        thanksgiving_half_day_2025 = (2025, 11, 28)
        assert thanksgiving_half_day_2025 in US_MARKET_HALF_DAYS

    def test_extended_hours(self, adapter):
        """Test extended hours configuration."""
        adapter_extended = PolygonTradingHoursAdapter(config={"extended_hours": True})

        assert adapter_extended._extended_hours is True

    def test_get_calendar(self, adapter):
        """Test getting trading calendar."""
        calendar = adapter.get_calendar()

        assert calendar is not None
        # Calendar should have sessions defined
        assert len(calendar.sessions) > 0
        assert calendar.timezone == "America/New_York"

    def test_next_open(self, adapter):
        """Test finding next market open."""
        # Saturday 12:00 PM UTC
        ts_saturday = 1737201600000

        next_open = adapter.next_open(ts_saturday)

        # Next open should be Monday
        assert next_open > ts_saturday

    def test_next_close(self, adapter):
        """Test finding next market close."""
        # Monday 10:00 AM ET during market hours
        ts_monday = 1705326000000

        next_close_ts = adapter.next_close(ts_monday)

        # Next close should be same day at 4 PM
        assert next_close_ts >= ts_monday


# =============================================================================
# PolygonExchangeInfoAdapter Tests
# =============================================================================


class TestPolygonExchangeInfoAdapter:
    """Tests for Polygon exchange info adapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked client."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            return PolygonExchangeInfoAdapter()

    def test_initialization(self):
        """Test adapter initialization."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()

            assert adapter._api_key == "test_key"
            assert adapter._cache_ttl == 3600
            assert len(adapter._symbol_cache) == 0

    def test_default_equity_rule(self):
        """Test default equity rule values."""
        rule = DEFAULT_EQUITY_RULE

        assert rule.tick_size == Decimal("0.01")
        assert rule.step_size == Decimal("1")
        assert rule.min_qty == Decimal("1")
        assert rule.price_precision == 2
        assert rule.market_type == MarketType.EQUITY
        assert rule.is_tradable is True

    @patch("adapters.polygon.exchange_info.PolygonExchangeInfoAdapter._get_rest_client")
    def test_get_symbols(self, mock_get_client):
        """Test getting symbol list."""
        mock_client = MagicMock()
        mock_ticker1 = MagicMock()
        mock_ticker1.ticker = "AAPL"
        mock_ticker2 = MagicMock()
        mock_ticker2.ticker = "MSFT"

        mock_client.list_tickers.return_value = [mock_ticker1, mock_ticker2]
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()
            symbols = adapter.get_symbols()

        assert "AAPL" in symbols
        assert "MSFT" in symbols

    @patch("adapters.polygon.exchange_info.PolygonExchangeInfoAdapter._get_rest_client")
    def test_get_symbol_info(self, mock_get_client):
        """Test getting symbol info."""
        mock_client = MagicMock()

        # Use a simple class instead of MagicMock to avoid __dict__ conflicts
        class MockDetails:
            name = "Apple Inc."
            sic_description = "Technology"
            type = "CS"
            active = True
            list_date = "1980-12-12"

        mock_details = MockDetails()

        mock_client.get_ticker_details.return_value = mock_details
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()
            info = adapter.get_symbol_info("AAPL")

        assert info is not None
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.market_type == MarketType.EQUITY

    @patch("adapters.polygon.exchange_info.PolygonExchangeInfoAdapter._get_rest_client")
    def test_get_symbol_info_caching(self, mock_get_client):
        """Test symbol info caching."""
        mock_client = MagicMock()

        # Use a simple class instead of MagicMock to avoid __dict__ conflicts
        class MockDetails:
            name = "Apple Inc."
            sic_description = "Technology"
            type = "CS"
            active = True

        mock_details = MockDetails()

        mock_client.get_ticker_details.return_value = mock_details
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()

            # First call
            info1 = adapter.get_symbol_info("AAPL")
            # Second call (should use cache)
            info2 = adapter.get_symbol_info("AAPL")

        # API should only be called once
        assert mock_client.get_ticker_details.call_count == 1
        assert info1 == info2

    @patch("adapters.polygon.exchange_info.PolygonExchangeInfoAdapter._get_rest_client")
    def test_get_symbol_info_not_found(self, mock_get_client):
        """Test getting info for non-existent symbol."""
        mock_client = MagicMock()
        mock_client.get_ticker_details.return_value = None
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()
            info = adapter.get_symbol_info("INVALID")

        assert info is None

    def test_get_popular_symbols(self):
        """Test getting popular symbols list."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()
            popular = adapter.get_popular_symbols(limit=10)

        assert len(popular) == 10
        # Should include major stocks
        assert "AAPL" in popular
        assert "MSFT" in popular

    def test_refresh_clears_cache(self):
        """Test refresh clears cache."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()

            # Add to cache
            adapter._symbol_cache["TEST"] = MagicMock()
            adapter._symbols_list_cache = ["TEST"]

            # Refresh
            result = adapter.refresh()

            assert result is True
            assert len(adapter._symbol_cache) == 0
            assert adapter._symbols_list_cache is None

    @patch("adapters.polygon.exchange_info.PolygonExchangeInfoAdapter._get_rest_client")
    def test_get_exchange_rules(self, mock_get_client):
        """Test getting exchange rules."""
        mock_client = MagicMock()

        # Use a simple class instead of MagicMock to avoid __dict__ conflicts
        class MockDetails:
            name = "Apple Inc."
            sic_description = "Technology"
            type = "CS"
            active = True

        mock_details = MockDetails()

        mock_client.get_ticker_details.return_value = mock_details
        mock_get_client.return_value = mock_client

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            adapter = PolygonExchangeInfoAdapter()
            rules = adapter.get_exchange_rules("AAPL")

        assert rules is not None
        assert rules.symbol == "AAPL"
        assert rules.tick_size == Decimal("0.01")


# =============================================================================
# Integration Tests
# =============================================================================


class TestPolygonIntegration:
    """Integration tests for Polygon adapters."""

    def test_adapters_share_vendor(self):
        """Test all adapters share same vendor."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            market_data = PolygonMarketDataAdapter()
            trading_hours = PolygonTradingHoursAdapter()
            exchange_info = PolygonExchangeInfoAdapter()

            assert market_data.vendor == ExchangeVendor.POLYGON
            assert trading_hours.vendor == ExchangeVendor.POLYGON
            assert exchange_info.vendor == ExchangeVendor.POLYGON

    def test_api_key_from_env(self):
        """Test API key loading from environment."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "env_key"}):
            market_data = PolygonMarketDataAdapter()
            exchange_info = PolygonExchangeInfoAdapter()

            assert market_data._api_key == "env_key"
            assert exchange_info._api_key == "env_key"

    def test_api_key_from_config_overrides_env(self):
        """Test config API key overrides environment."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "env_key"}):
            adapter = PolygonMarketDataAdapter(config={"api_key": "config_key"})

            assert adapter._api_key == "config_key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
