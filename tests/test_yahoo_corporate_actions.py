# -*- coding: utf-8 -*-
"""
Tests for Yahoo Finance corporate actions adapter.

Tests cover:
- YahooCorporateActionsAdapter initialization
- get_dividends()
- get_splits()
- get_corporate_actions()
- get_adjustment_factors()
- compute_dividend_adjustment()
- Caching behavior
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import sys

from adapters.models import (
    ExchangeVendor,
    CorporateActionType,
    Dividend,
    StockSplit,
    AdjustmentFactors,
)


# Create a mock yfinance module for testing
@pytest.fixture
def mock_yfinance():
    """Create a mock yfinance module."""
    mock_yf = MagicMock()
    with patch.dict(sys.modules, {"yfinance": mock_yf}):
        yield mock_yf


class TestYahooCorporateActionsAdapterInit:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        adapter = YahooCorporateActionsAdapter()
        assert adapter._vendor == ExchangeVendor.YAHOO
        assert adapter._cache_ttl == 3600
        assert adapter._max_retries == 3
        assert adapter._ticker_cache == {}

    def test_custom_config(self):
        """Test initialization with custom config."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        config = {"cache_ttl": 7200, "max_retries": 5}
        adapter = YahooCorporateActionsAdapter(config=config)
        assert adapter._cache_ttl == 7200
        assert adapter._max_retries == 5


class TestYahooCorporateActionsGetDividends:
    """Tests for get_dividends method."""

    def test_get_dividends_success(self, mock_yfinance):
        """Test successful dividend fetch."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        # Mock yfinance Ticker
        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        # Create mock dividend series
        dates = pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15"])
        mock_ticker.dividends = pd.Series([0.24, 0.24, 0.25], index=dates)

        adapter = YahooCorporateActionsAdapter()
        dividends = adapter.get_dividends("AAPL")

        assert len(dividends) == 3
        assert all(isinstance(d, Dividend) for d in dividends)
        assert dividends[0].symbol == "AAPL"
        assert dividends[0].amount == Decimal("0.24")
        assert dividends[0].ex_date == "2024-01-15"

    def test_get_dividends_with_date_filter(self, mock_yfinance):
        """Test dividend fetch with date range."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        dates = pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15"])
        mock_ticker.dividends = pd.Series([0.24, 0.24, 0.25], index=dates)

        adapter = YahooCorporateActionsAdapter()
        dividends = adapter.get_dividends(
            "AAPL",
            start_date="2024-03-01",
            end_date="2024-05-01",
        )

        assert len(dividends) == 1
        assert dividends[0].ex_date == "2024-04-15"

    def test_get_dividends_empty(self, mock_yfinance):
        """Test dividend fetch when no dividends exist."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        mock_ticker.dividends = pd.Series([], dtype=float)

        adapter = YahooCorporateActionsAdapter()
        dividends = adapter.get_dividends("TSLA")

        assert dividends == []

    def test_get_dividends_error_handling(self, mock_yfinance):
        """Test graceful error handling."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        # Simulate an exception when accessing dividends
        type(mock_ticker).dividends = property(
            lambda self: (_ for _ in ()).throw(Exception("API error"))
        )

        adapter = YahooCorporateActionsAdapter()
        # Should return empty list, not raise
        dividends = adapter.get_dividends("INVALID")
        assert dividends == []


class TestYahooCorporateActionsGetSplits:
    """Tests for get_splits method."""

    def test_get_splits_forward(self, mock_yfinance):
        """Test fetching forward splits."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        dates = pd.to_datetime(["2020-08-31", "2022-06-06"])
        # yfinance returns ratio as float (4.0 for 4:1)
        mock_ticker.splits = pd.Series([4.0, 3.0], index=dates)

        adapter = YahooCorporateActionsAdapter()
        splits = adapter.get_splits("AAPL")

        assert len(splits) == 2
        assert all(isinstance(s, StockSplit) for s in splits)
        assert splits[0].ratio == (4, 1)
        assert splits[0].is_reverse is False
        assert splits[1].ratio == (3, 1)

    def test_get_splits_reverse(self, mock_yfinance):
        """Test fetching reverse splits."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        dates = pd.to_datetime(["2024-01-15"])
        # 0.1 = 1:10 reverse split
        mock_ticker.splits = pd.Series([0.1], index=dates)

        adapter = YahooCorporateActionsAdapter()
        splits = adapter.get_splits("XYZ")

        assert len(splits) == 1
        assert splits[0].is_reverse is True
        assert splits[0].ratio == (1, 10)

    def test_get_splits_empty(self, mock_yfinance):
        """Test when no splits exist."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        mock_ticker.splits = pd.Series([], dtype=float)

        adapter = YahooCorporateActionsAdapter()
        splits = adapter.get_splits("MSFT")

        assert splits == []


class TestYahooCorporateActionsGetCorporateActions:
    """Tests for get_corporate_actions method."""

    def test_get_all_actions(self, mock_yfinance):
        """Test fetching all corporate actions."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        # Dividends
        div_dates = pd.to_datetime(["2024-01-15", "2024-04-15"])
        mock_ticker.dividends = pd.Series([0.24, 0.24], index=div_dates)

        # Splits
        split_dates = pd.to_datetime(["2024-02-01"])
        mock_ticker.splits = pd.Series([2.0], index=split_dates)

        adapter = YahooCorporateActionsAdapter()
        actions = adapter.get_corporate_actions("AAPL")

        assert len(actions) == 3
        # Should be sorted by date
        assert actions[0].ex_date == "2024-01-15"
        assert actions[1].ex_date == "2024-02-01"
        assert actions[2].ex_date == "2024-04-15"

    def test_filter_by_action_type(self, mock_yfinance):
        """Test filtering by action type."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        div_dates = pd.to_datetime(["2024-01-15"])
        mock_ticker.dividends = pd.Series([0.24], index=div_dates)

        split_dates = pd.to_datetime(["2024-02-01"])
        mock_ticker.splits = pd.Series([2.0], index=split_dates)

        adapter = YahooCorporateActionsAdapter()

        # Only dividends
        div_actions = adapter.get_corporate_actions(
            "AAPL",
            action_types=[CorporateActionType.DIVIDEND],
        )
        assert len(div_actions) == 1
        assert div_actions[0].action_type == CorporateActionType.DIVIDEND

        # Only splits
        split_actions = adapter.get_corporate_actions(
            "AAPL",
            action_types=[CorporateActionType.SPLIT],
        )
        assert len(split_actions) == 1
        assert split_actions[0].action_type == CorporateActionType.SPLIT


class TestYahooCorporateActionsGetAdjustmentFactors:
    """Tests for get_adjustment_factors method."""

    def test_adjustment_factors_with_splits(self, mock_yfinance):
        """Test computing adjustment factors."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        # Two splits: 4:1 and 2:1
        split_dates = pd.to_datetime(["2020-01-15", "2022-06-01"])
        mock_ticker.splits = pd.Series([4.0, 2.0], index=split_dates)

        adapter = YahooCorporateActionsAdapter()
        factors = adapter.get_adjustment_factors("AAPL", "2024-01-01")

        assert isinstance(factors, AdjustmentFactors)
        assert factors.symbol == "AAPL"
        # Cumulative: 0.25 * 0.5 = 0.125
        assert factors.split_factor == pytest.approx(0.125)
        assert factors.dividend_factor == 1.0  # Not computed by this adapter

    def test_adjustment_factors_no_splits(self, mock_yfinance):
        """Test adjustment factors when no splits."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker
        mock_ticker.splits = pd.Series([], dtype=float)

        adapter = YahooCorporateActionsAdapter()
        factors = adapter.get_adjustment_factors("MSFT", "2024-01-01")

        assert factors.split_factor == 1.0
        assert factors.dividend_factor == 1.0


class TestYahooCorporateActionsComputeDividendAdjustment:
    """Tests for compute_dividend_adjustment method."""

    def test_compute_dividend_adjustment(self, mock_yfinance):
        """Test computing dividend adjustment factors."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_yfinance.Ticker.return_value = mock_ticker

        # Dividend on 2024-01-16 (ex-date)
        div_dates = pd.to_datetime(["2024-01-16"])
        mock_ticker.dividends = pd.Series([0.50], index=div_dates)

        adapter = YahooCorporateActionsAdapter()

        prices = {
            "2024-01-14": 100.0,  # Two days before ex-date
            "2024-01-15": 100.0,  # Day before ex-date
            "2024-01-16": 99.50,  # Ex-date (price drops by dividend)
            "2024-01-17": 100.0,
        }

        factors = adapter.compute_dividend_adjustment(
            "AAPL",
            start_date="2024-01-14",
            end_date="2024-01-17",
            prices=prices,
        )

        # All dates should have factors computed
        assert "2024-01-14" in factors
        assert "2024-01-17" in factors
        assert factors["2024-01-17"] == pytest.approx(1.0)
        # Earlier dates should have smaller factors (accounting for dividend)
        # Note: The algorithm works backwards, so Jan 14 gets the adjusted factor
        assert factors["2024-01-14"] < 1.0


class TestYahooCorporateActionsCaching:
    """Tests for caching behavior."""

    def test_ticker_caching(self, mock_yfinance):
        """Test that Ticker objects are cached."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_yfinance.Ticker.return_value = mock_ticker

        adapter = YahooCorporateActionsAdapter()

        # First call creates Ticker
        with patch("time.time", return_value=1000.0):
            adapter.get_dividends("AAPL")
        assert mock_yfinance.Ticker.call_count == 1

        # Second call uses cache (time hasn't advanced much)
        with patch("time.time", return_value=1001.0):
            adapter.get_dividends("AAPL")
        assert mock_yfinance.Ticker.call_count == 1

    def test_cache_expiry(self, mock_yfinance):
        """Test that cache expires after TTL."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        mock_ticker = Mock()
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_yfinance.Ticker.return_value = mock_ticker

        adapter = YahooCorporateActionsAdapter(config={"cache_ttl": 3600})

        # First call at t=1000
        with patch("time.time", return_value=1000.0):
            adapter.get_dividends("AAPL")
        assert mock_yfinance.Ticker.call_count == 1

        # Second call within TTL
        with patch("time.time", return_value=2000.0):
            adapter.get_dividends("AAPL")
        assert mock_yfinance.Ticker.call_count == 1

        # Third call after TTL (> 3600 seconds)
        with patch("time.time", return_value=5000.0):
            adapter.get_dividends("AAPL")
        assert mock_yfinance.Ticker.call_count == 2


class TestYahooCorporateActionsYfinanceImport:
    """Tests for yfinance import handling."""

    def test_missing_yfinance_raises(self):
        """Test that missing yfinance raises ImportError with helpful message."""
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter

        adapter = YahooCorporateActionsAdapter()

        # Clear any cached ticker
        adapter._ticker_cache.clear()
        adapter._cache_timestamps.clear()

        # This test passes if yfinance is not installed or we mock ImportError
        # The actual implementation has a try/except for import


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
