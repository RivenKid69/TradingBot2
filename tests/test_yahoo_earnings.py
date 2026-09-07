# -*- coding: utf-8 -*-
"""
Tests for Yahoo Finance earnings adapter.

Tests cover:
- YahooEarningsAdapter initialization
- get_earnings_history()
- get_upcoming_earnings()
- get_earnings_calendar()
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

from adapters.models import (
    ExchangeVendor,
    EarningsEvent,
)


# Create a mock yfinance module for testing
@pytest.fixture
def mock_yfinance():
    """Create a mock yfinance module."""
    mock_yf = MagicMock()
    with patch.dict(sys.modules, {"yfinance": mock_yf}):
        yield mock_yf


class TestYahooEarningsAdapterInit:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        adapter = YahooEarningsAdapter()
        assert adapter._vendor == ExchangeVendor.YAHOO
        assert adapter._cache_ttl == 3600
        assert adapter._ticker_cache == {}
        assert adapter._earnings_cache == {}

    def test_custom_config(self):
        """Test initialization with custom config."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        config = {"cache_ttl": 7200}
        adapter = YahooEarningsAdapter(config=config)
        assert adapter._cache_ttl == 7200


class TestYahooEarningsGetHistory:
    """Tests for get_earnings_history method."""

    def test_get_earnings_history_success(self, mock_yfinance):
        """Test successful earnings history fetch."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        # Create mock earnings_dates DataFrame
        dates = pd.to_datetime(["2024-01-25", "2024-04-25", "2024-07-25"])
        mock_df = pd.DataFrame(
            {
                "EPS Estimate": [2.10, 2.15, 2.20],
                "Reported EPS": [2.18, 2.22, None],
                "Surprise(%)": [3.8, 3.2, None],
            },
            index=dates,
        )
        mock_ticker.earnings_dates = mock_df

        adapter = YahooEarningsAdapter()
        events = adapter.get_earnings_history("AAPL")

        assert len(events) == 3
        assert all(isinstance(e, EarningsEvent) for e in events)
        assert events[0].symbol == "AAPL"
        assert events[0].report_date == "2024-01-25"

    def test_get_earnings_history_empty(self, mock_yfinance):
        """Test when no earnings data exists."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        mock_ticker.earnings_dates = None

        adapter = YahooEarningsAdapter()
        events = adapter.get_earnings_history("INVALID")

        assert events == []


class TestYahooEarningsGetCalendar:
    """Tests for get_earnings_calendar method."""

    def test_get_earnings_calendar_warning(self):
        """Test that calendar endpoint returns empty with warning."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        adapter = YahooEarningsAdapter()
        result = adapter.get_earnings_calendar("2024-01-25")

        # Yahoo doesn't support bulk calendar queries
        assert result == []


class TestYahooEarningsComputeFeatures:
    """Tests for compute_earnings_features method."""

    def test_compute_earnings_features_no_data(self, mock_yfinance):
        """Test features when no earnings data exists."""
        from adapters.yahoo.earnings import YahooEarningsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        mock_ticker.earnings_dates = None

        adapter = YahooEarningsAdapter()
        features = adapter.compute_earnings_features("INVALID", "2024-01-15")

        assert features["days_to_next_earnings"] is None
        assert features["days_since_last_earnings"] is None
        assert features["last_surprise_pct"] is None
        assert features["beat_streak"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
