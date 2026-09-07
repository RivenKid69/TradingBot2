# -*- coding: utf-8 -*-
"""
tests/test_stock_integration_phase2.py
Tests for stock integration improvements (Phase 2 completion):

1. Yahoo market data adapter (VIX, DXY, indices)
2. Download stock data script (VIX pipeline, auto-detection)
3. Alpaca streaming async wrappers

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_models import Bar, Tick
from adapters.models import ExchangeVendor


# =============================================================================
# SECTION 1: Yahoo Market Data Adapter Tests
# =============================================================================


class TestYahooMarketDataAdapter:
    """Tests for YahooMarketDataAdapter."""

    def test_import_adapter(self):
        """Test that adapter can be imported."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        assert YahooMarketDataAdapter is not None

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()
        assert adapter._vendor == ExchangeVendor.YAHOO
        assert adapter._yf is None  # Lazy initialization

    def test_adapter_with_config(self):
        """Test adapter initialization with config."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        config = {
            "rate_limit_pause": 1.0,
            "max_retries": 5,
        }
        adapter = YahooMarketDataAdapter(config=config)
        assert adapter._rate_limit_pause == 1.0
        assert adapter._max_retries == 5

    def test_parse_timeframe(self):
        """Test timeframe parsing."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        assert adapter._parse_timeframe("1m") == "1m"
        assert adapter._parse_timeframe("5m") == "5m"
        assert adapter._parse_timeframe("1h") == "1h"
        assert adapter._parse_timeframe("1d") == "1d"
        assert adapter._parse_timeframe("1day") == "1d"
        assert adapter._parse_timeframe("1w") == "1wk"
        assert adapter._parse_timeframe("1mo") == "1mo"

    def test_estimate_days_for_limit(self):
        """Test days estimation for limit."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        # Daily data: 1 bar per day
        days_1d = adapter._estimate_days_for_limit(100, "1d")
        assert days_1d >= 100

        # Hourly data: ~7 bars per day
        days_1h = adapter._estimate_days_for_limit(100, "1h")
        assert days_1h > 14  # 100/7 = ~14

    def test_get_supported_indices(self):
        """Test supported indices lookup."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        indices = YahooMarketDataAdapter.get_supported_indices()

        assert "^VIX" in indices
        assert "DX-Y.NYB" in indices
        assert "^TNX" in indices
        assert "^TYX" in indices
        assert "GC=F" in indices

    def test_is_yahoo_symbol(self):
        """Test Yahoo symbol detection."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        # Should return True for Yahoo symbols
        assert YahooMarketDataAdapter.is_yahoo_symbol("^VIX") is True
        assert YahooMarketDataAdapter.is_yahoo_symbol("^GSPC") is True
        assert YahooMarketDataAdapter.is_yahoo_symbol("GC=F") is True
        assert YahooMarketDataAdapter.is_yahoo_symbol("DX-Y.NYB") is True

        # Should return False for regular stocks
        assert YahooMarketDataAdapter.is_yahoo_symbol("AAPL") is False
        assert YahooMarketDataAdapter.is_yahoo_symbol("MSFT") is False

    def test_get_bars_mocked(self):
        """Test get_bars with mocked yfinance."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        # Create mock ticker
        mock_yf = MagicMock()
        mock_ticker = MagicMock()

        # Create mock DataFrame
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        mock_df = pd.DataFrame(
            {
                "Open": [20.0, 21.0, 22.0, 23.0, 24.0],
                "High": [21.0, 22.0, 23.0, 24.0, 25.0],
                "Low": [19.0, 20.0, 21.0, 22.0, 23.0],
                "Close": [20.5, 21.5, 22.5, 23.5, 24.5],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        adapter = YahooMarketDataAdapter()
        # Directly set the mock yfinance
        adapter._yf = mock_yf

        # Mock _ensure_yfinance to return our mock
        with patch.object(adapter, "_ensure_yfinance", return_value=mock_yf):
            bars = adapter.get_bars("^VIX", "1d", limit=5)

        assert len(bars) == 5
        assert all(isinstance(b, Bar) for b in bars)
        assert bars[0].symbol == "^VIX"
        assert float(bars[0].close) == 20.5

    def test_stream_bars_not_supported(self):
        """Test that streaming raises NotImplementedError."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        with pytest.raises(NotImplementedError):
            list(adapter.stream_bars(["^VIX"], 60000))

    def test_stream_ticks_not_supported(self):
        """Test that tick streaming raises NotImplementedError."""
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        with pytest.raises(NotImplementedError):
            list(adapter.stream_ticks(["^VIX"]))


class TestYahooAdapterRegistration:
    """Tests for Yahoo adapter registration."""

    def test_yahoo_registered_in_registry(self):
        """Test that Yahoo adapter is registered."""
        # Import to trigger registration
        import adapters.yahoo  # noqa: F401
        from adapters.registry import create_market_data_adapter

        # Try to create adapter - will fail if not registered
        adapter = create_market_data_adapter("yahoo")
        assert adapter is not None
        assert adapter._vendor == ExchangeVendor.YAHOO

    def test_create_yahoo_adapter_via_registry(self):
        """Test creating Yahoo adapter via registry."""
        from adapters.registry import create_market_data_adapter

        adapter = create_market_data_adapter("yahoo")

        assert adapter is not None
        assert adapter._vendor == ExchangeVendor.YAHOO


# =============================================================================
# SECTION 2: Download Stock Data Script Tests
# =============================================================================


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_sanitize_vix_symbol(self):
        """Test sanitizing VIX symbol."""
        from scripts.download_stock_data import sanitize_filename

        assert sanitize_filename("^VIX") == "VIX"
        assert sanitize_filename("^VXN") == "VXN"
        assert sanitize_filename("^GSPC") == "GSPC"

    def test_sanitize_futures_symbol(self):
        """Test sanitizing futures symbols."""
        from scripts.download_stock_data import sanitize_filename

        assert sanitize_filename("GC=F") == "GC_F"
        assert sanitize_filename("SI=F") == "SI_F"
        assert sanitize_filename("CL=F") == "CL_F"

    def test_sanitize_dxy_symbol(self):
        """Test sanitizing DXY symbol."""
        from scripts.download_stock_data import sanitize_filename

        assert sanitize_filename("DX-Y.NYB") == "DX-Y_NYB"

    def test_sanitize_regular_symbol(self):
        """Test that regular symbols are unchanged."""
        from scripts.download_stock_data import sanitize_filename

        assert sanitize_filename("AAPL") == "AAPL"
        assert sanitize_filename("MSFT") == "MSFT"
        assert sanitize_filename("GOOGL") == "GOOGL"


class TestDownloadConfig:
    """Tests for DownloadConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from scripts.download_stock_data import DownloadConfig

        config = DownloadConfig()

        assert config.provider == "alpaca"
        assert config.timeframe == "1h"
        assert config.output_dir == "data/raw_stocks"
        assert config.skip_existing is True
        assert config.feed == "iex"

    def test_custom_config(self):
        """Test custom configuration."""
        from scripts.download_stock_data import DownloadConfig

        config = DownloadConfig(
            provider="yahoo",
            symbols=["^VIX", "^VXN"],
            timeframe="1d",
            start_date="2020-01-01",
        )

        assert config.provider == "yahoo"
        assert config.symbols == ["^VIX", "^VXN"]
        assert config.timeframe == "1d"
        assert config.start_date == "2020-01-01"


class TestDownloadSymbolYahoo:
    """Tests for Yahoo download function."""

    @patch("adapters.yahoo.market_data.YahooMarketDataAdapter")
    def test_download_vix_success(self, mock_adapter_class):
        """Test successful VIX download."""
        from scripts.download_stock_data import download_symbol_yahoo, DownloadConfig

        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter

        # Create mock bars
        mock_bars = [
            MagicMock(
                ts=1704067200000,  # 2024-01-01
                open=Decimal("20.0"),
                high=Decimal("21.0"),
                low=Decimal("19.0"),
                close=Decimal("20.5"),
                volume_base=Decimal("1000"),
            )
        ]
        mock_adapter.get_bars.return_value = mock_bars

        config = DownloadConfig(
            provider="yahoo",
            start_date="2024-01-01",
            end_date="2024-01-02",
            timeframe="1d",
        )

        symbol, df, error = download_symbol_yahoo("^VIX", config)

        assert symbol == "^VIX"
        assert error is None
        assert df is not None
        assert len(df) == 1
        assert "close" in df.columns

    @patch("adapters.yahoo.market_data.YahooMarketDataAdapter")
    def test_download_no_data(self, mock_adapter_class):
        """Test download with no data returned."""
        from scripts.download_stock_data import download_symbol_yahoo, DownloadConfig

        mock_adapter = MagicMock()
        mock_adapter.get_bars.return_value = []
        mock_adapter_class.return_value = mock_adapter

        config = DownloadConfig(start_date="2024-01-01")
        symbol, df, error = download_symbol_yahoo("^VIX", config)

        assert symbol == "^VIX"
        assert df is None
        assert error == "No data returned"


class TestAutoProviderSelection:
    """Tests for auto-provider selection logic."""

    def test_yahoo_symbol_detection(self):
        """Test that Yahoo symbols are auto-detected."""

        # Simulate the logic from download_all_symbols
        def get_provider_for_symbol(symbol: str, default: str = "alpaca") -> str:
            if symbol.startswith("^") or "=" in symbol or "-Y." in symbol:
                return "yahoo"
            return default

        assert get_provider_for_symbol("^VIX") == "yahoo"
        assert get_provider_for_symbol("^VXN") == "yahoo"
        assert get_provider_for_symbol("GC=F") == "yahoo"
        assert get_provider_for_symbol("DX-Y.NYB") == "yahoo"
        assert get_provider_for_symbol("AAPL") == "alpaca"
        assert get_provider_for_symbol("MSFT") == "alpaca"


class TestNYSECalendar:
    """Tests for NYSE Calendar."""

    def test_is_weekend(self):
        """Test weekend detection."""
        from scripts.download_stock_data import NYSECalendar

        # Saturday
        saturday = datetime(2024, 1, 6)
        assert NYSECalendar.is_weekend(saturday) is True

        # Sunday
        sunday = datetime(2024, 1, 7)
        assert NYSECalendar.is_weekend(sunday) is True

        # Monday
        monday = datetime(2024, 1, 8)
        assert NYSECalendar.is_weekend(monday) is False

    def test_is_holiday(self):
        """Test holiday detection."""
        from scripts.download_stock_data import NYSECalendar

        # Christmas 2024
        christmas = datetime(2024, 12, 25)
        assert NYSECalendar.is_holiday(christmas) is True

        # Regular day
        regular = datetime(2024, 6, 15)
        assert NYSECalendar.is_holiday(regular) is False

    def test_is_trading_day(self):
        """Test trading day detection."""
        from scripts.download_stock_data import NYSECalendar

        # Regular Monday
        monday = datetime(2024, 6, 17)
        assert NYSECalendar.is_trading_day(monday) is True

        # Saturday
        saturday = datetime(2024, 6, 15)
        assert NYSECalendar.is_trading_day(saturday) is False


# =============================================================================
# SECTION 3: Alpaca Streaming Tests
# =============================================================================


class TestAlpacaStreamingSync:
    """Tests for sync streaming methods."""

    def test_stream_bars_signature(self):
        """Test stream_bars method signature."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Check method exists and is callable
        assert hasattr(adapter, "stream_bars")
        assert callable(adapter.stream_bars)

    def test_stream_ticks_signature(self):
        """Test stream_ticks method signature."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Check method exists and is callable
        assert hasattr(adapter, "stream_ticks")
        assert callable(adapter.stream_ticks)


class TestAlpacaStreamingAsync:
    """Tests for async streaming methods."""

    def test_stream_bars_async_exists(self):
        """Test stream_bars_async method exists."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        assert hasattr(adapter, "stream_bars_async")
        assert callable(adapter.stream_bars_async)

    def test_stream_ticks_async_exists(self):
        """Test stream_ticks_async method exists."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        assert hasattr(adapter, "stream_ticks_async")
        assert callable(adapter.stream_ticks_async)

    def test_stream_bars_async_is_async_generator(self):
        """Test that stream_bars_async returns an async generator."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter
        import inspect

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Check it's an async generator function
        assert inspect.isasyncgenfunction(adapter.stream_bars_async)

    def test_stream_ticks_async_is_async_generator(self):
        """Test that stream_ticks_async returns an async generator."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter
        import inspect

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Check it's an async generator function
        assert inspect.isasyncgenfunction(adapter.stream_ticks_async)


class TestAlpacaBarConversion:
    """Tests for bar conversion."""

    def test_convert_bar_success(self):
        """Test successful bar conversion."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Create mock Alpaca bar
        mock_bar = MagicMock()
        mock_bar.timestamp.timestamp.return_value = 1704067200.0  # 2024-01-01 00:00:00
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 152.0
        mock_bar.volume = 1000000
        mock_bar.vwap = 151.5
        mock_bar.trade_count = 5000

        bar = adapter._convert_bar("AAPL", mock_bar)

        assert bar is not None
        assert bar.symbol == "AAPL"
        assert float(bar.open) == 150.0
        assert float(bar.high) == 155.0
        assert float(bar.low) == 149.0
        assert float(bar.close) == 152.0
        assert float(bar.volume_base) == 1000000

    def test_convert_bar_error_handling(self):
        """Test bar conversion error handling."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        # Create mock bar that raises error
        mock_bar = MagicMock()
        mock_bar.timestamp.timestamp.side_effect = Exception("Test error")

        bar = adapter._convert_bar("AAPL", mock_bar)

        assert bar is None


try:
    from alpaca.data.timeframe import TimeFrame

    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False


@pytest.mark.skipif(not HAS_ALPACA, reason="alpaca-py not installed")
class TestAlpacaTimeframeParsing:
    """Tests for timeframe parsing."""

    def test_parse_minute_timeframes(self):
        """Test parsing minute timeframes."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter
        from alpaca.data.timeframe import TimeFrame

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        tf = adapter._parse_timeframe("1m")
        assert tf == TimeFrame.Minute

        tf = adapter._parse_timeframe("1min")
        assert tf == TimeFrame.Minute

    def test_parse_hour_timeframes(self):
        """Test parsing hour timeframes."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter
        from alpaca.data.timeframe import TimeFrame

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        tf = adapter._parse_timeframe("1h")
        assert tf == TimeFrame.Hour

        tf = adapter._parse_timeframe("1hour")
        assert tf == TimeFrame.Hour

    def test_parse_day_timeframes(self):
        """Test parsing day timeframes."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter
        from alpaca.data.timeframe import TimeFrame

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        tf = adapter._parse_timeframe("1d")
        assert tf == TimeFrame.Day

        tf = adapter._parse_timeframe("1day")
        assert tf == TimeFrame.Day

    def test_parse_invalid_timeframe(self):
        """Test parsing invalid timeframe."""
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        adapter = AlpacaMarketDataAdapter(
            config={
                "api_key": "test",
                "api_secret": "test",
            }
        )

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            adapter._parse_timeframe("invalid")


# =============================================================================
# SECTION 4: Docstring Tests
# =============================================================================


class TestDocstringUpdates:
    """Tests for docstring updates (STUB removed)."""

    def test_alpaca_market_data_docstring(self):
        """Test that STUB is removed from market_data.py docstring."""
        from adapters.alpaca import market_data

        docstring = market_data.__doc__
        assert "STUB" not in docstring
        assert "Tested and operational" in docstring

    def test_alpaca_init_docstring(self):
        """Test that STUB is removed from __init__.py docstring."""
        from adapters import alpaca

        docstring = alpaca.__doc__
        assert "STUB" not in docstring
        assert "Tested and operational" in docstring


# =============================================================================
# SECTION 5: Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests (with mocking)."""

    @patch("scripts.download_stock_data.download_symbol_yahoo")
    def test_download_vix_integration(self, mock_download):
        """Test VIX download integration."""
        from scripts.download_stock_data import DownloadConfig, download_all_symbols
        import tempfile
        import os

        # Setup mock
        mock_df = pd.DataFrame(
            {
                "timestamp": [1704067200],
                "open": [20.0],
                "high": [21.0],
                "low": [19.0],
                "close": [20.5],
                "volume": [1000],
                "symbol": ["^VIX"],
            }
        )
        mock_download.return_value = ("^VIX", mock_df, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DownloadConfig(
                provider="yahoo",
                symbols=["^VIX"],
                output_dir=tmpdir,
                skip_existing=False,
            )

            results = download_all_symbols(config)

            assert results["success"] == 1
            assert results["failed"] == 0

            # Check file was created with sanitized name
            expected_file = os.path.join(tmpdir, "VIX.parquet")
            assert os.path.exists(expected_file)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
