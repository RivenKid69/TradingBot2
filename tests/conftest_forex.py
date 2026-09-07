# -*- coding: utf-8 -*-
"""
tests/conftest_forex.py
Phase 0: Forex Test Fixtures and Mock Infrastructure

This module provides pytest fixtures for forex-related testing:
1. OANDA API mock responses (VCR-style pattern)
2. Forex data fixtures (sample OHLCV data)
3. Configuration fixtures for different forex scenarios
4. Shared test utilities for forex tests

Usage:
    Import fixtures in test files or add to pytest's conftest.py plugin loading.

References:
    - OANDA v20 API: https://developer.oanda.com/rest-live-v20/introduction/
    - BIS Triennial Survey 2022: https://www.bis.org/statistics/rpfx22.htm
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch
import json


# =============================================================================
# OANDA Configuration Fixtures
# =============================================================================


@pytest.fixture
def oanda_config() -> Dict[str, Any]:
    """Basic OANDA configuration for testing."""
    return {
        "api_key": "test-api-key-12345",
        "account_id": "101-001-12345678-001",
        "practice": True,  # Use demo/practice environment
        "streaming": False,
        "timeout": 30,
    }


@pytest.fixture
def oanda_config_live() -> Dict[str, Any]:
    """Live OANDA configuration (for testing config validation)."""
    return {
        "api_key": "live-api-key-12345",
        "account_id": "001-001-12345678-001",
        "practice": False,  # Live environment
        "streaming": True,
        "timeout": 10,
    }


# =============================================================================
# Mock OANDA API Response Fixtures
# =============================================================================


@dataclass(frozen=True)
class MockOandaPrice:
    """Mock OANDA price quote."""

    instrument: str
    bid: Decimal
    ask: Decimal
    time: str
    tradeable: bool = True

    @property
    def spread_pips(self) -> float:
        """Calculate spread in pips."""
        from adapters.models import get_pip_size

        spread = float(self.ask - self.bid)
        pip_size = get_pip_size(self.instrument)
        return spread / pip_size if pip_size > 0 else 0.0

    def to_oanda_response(self) -> Dict[str, Any]:
        """Convert to OANDA v20 API response format."""
        return {
            "prices": [
                {
                    "asks": [{"price": str(self.ask), "liquidity": 1000000}],
                    "bids": [{"price": str(self.bid), "liquidity": 1000000}],
                    "closeoutAsk": str(self.ask),
                    "closeoutBid": str(self.bid),
                    "instrument": self.instrument,
                    "status": "tradeable" if self.tradeable else "non-tradeable",
                    "time": self.time,
                    "tradeable": self.tradeable,
                }
            ]
        }


@dataclass(frozen=True)
class MockOandaCandle:
    """Mock OANDA candlestick data."""

    instrument: str
    time: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    complete: bool = True

    def to_oanda_response(self) -> Dict[str, Any]:
        """Convert to OANDA v20 API candle format."""
        return {
            "complete": self.complete,
            "mid": {
                "o": str(self.open),
                "h": str(self.high),
                "l": str(self.low),
                "c": str(self.close),
            },
            "time": self.time,
            "volume": self.volume,
        }


@pytest.fixture
def mock_eurusd_price() -> MockOandaPrice:
    """Mock EUR/USD price quote."""
    return MockOandaPrice(
        instrument="EUR_USD",
        bid=Decimal("1.08500"),
        ask=Decimal("1.08515"),
        time="2025-01-15T12:00:00.000000000Z",
        tradeable=True,
    )


@pytest.fixture
def mock_usdjpy_price() -> MockOandaPrice:
    """Mock USD/JPY price quote."""
    return MockOandaPrice(
        instrument="USD_JPY",
        bid=Decimal("150.120"),
        ask=Decimal("150.135"),
        time="2025-01-15T12:00:00.000000000Z",
        tradeable=True,
    )


@pytest.fixture
def mock_gbpusd_price() -> MockOandaPrice:
    """Mock GBP/USD price quote."""
    return MockOandaPrice(
        instrument="GBP_USD",
        bid=Decimal("1.26500"),
        ask=Decimal("1.26520"),
        time="2025-01-15T12:00:00.000000000Z",
        tradeable=True,
    )


@pytest.fixture
def mock_eurusd_candles() -> List[MockOandaCandle]:
    """Mock EUR/USD historical candles (1 hour timeframe)."""
    base_price = Decimal("1.08000")
    candles = []
    for i in range(24):  # 24 hours of data
        hour_str = f"{i:02d}"
        candles.append(
            MockOandaCandle(
                instrument="EUR_USD",
                time=f"2025-01-15T{hour_str}:00:00.000000000Z",
                open=base_price + Decimal(str(i * 0.0001)),
                high=base_price + Decimal(str((i + 1) * 0.0001)),
                low=base_price + Decimal(str((i - 0.5) * 0.0001)) if i > 0 else base_price,
                close=base_price + Decimal(str((i + 0.5) * 0.0001)),
                volume=10000 + (i * 500),
                complete=True,
            )
        )
    return candles


@pytest.fixture
def mock_oanda_instruments_response() -> Dict[str, Any]:
    """Mock OANDA instruments endpoint response."""
    return {
        "instruments": [
            {
                "name": "EUR_USD",
                "type": "CURRENCY",
                "displayName": "EUR/USD",
                "pipLocation": -4,
                "displayPrecision": 5,
                "tradeUnitsPrecision": 0,
                "minimumTradeSize": "1",
                "maximumTrailingStopDistance": "1.00000",
                "minimumTrailingStopDistance": "0.00050",
                "maximumPositionSize": "0",
                "maximumOrderUnits": "100000000",
                "marginRate": "0.02",  # 50:1 leverage
                "guaranteedStopLossOrderMode": "DISABLED",
                "financing": {
                    "longRate": "-0.0173",
                    "shortRate": "0.0115",
                },
            },
            {
                "name": "USD_JPY",
                "type": "CURRENCY",
                "displayName": "USD/JPY",
                "pipLocation": -2,
                "displayPrecision": 3,
                "tradeUnitsPrecision": 0,
                "minimumTradeSize": "1",
                "maximumTrailingStopDistance": "100.000",
                "minimumTrailingStopDistance": "0.050",
                "maximumPositionSize": "0",
                "maximumOrderUnits": "100000000",
                "marginRate": "0.02",
                "guaranteedStopLossOrderMode": "DISABLED",
                "financing": {
                    "longRate": "-0.0285",
                    "shortRate": "0.0195",
                },
            },
            {
                "name": "GBP_USD",
                "type": "CURRENCY",
                "displayName": "GBP/USD",
                "pipLocation": -4,
                "displayPrecision": 5,
                "tradeUnitsPrecision": 0,
                "minimumTradeSize": "1",
                "maximumTrailingStopDistance": "1.00000",
                "minimumTrailingStopDistance": "0.00050",
                "maximumPositionSize": "0",
                "maximumOrderUnits": "100000000",
                "marginRate": "0.03",  # 33:1 leverage
                "guaranteedStopLossOrderMode": "DISABLED",
                "financing": {
                    "longRate": "-0.0145",
                    "shortRate": "0.0095",
                },
            },
            {
                "name": "USD_TRY",
                "type": "CURRENCY",
                "displayName": "USD/TRY",
                "pipLocation": -4,
                "displayPrecision": 5,
                "tradeUnitsPrecision": 0,
                "minimumTradeSize": "1",
                "maximumTrailingStopDistance": "10.00000",
                "minimumTrailingStopDistance": "0.00500",
                "maximumPositionSize": "0",
                "maximumOrderUnits": "10000000",
                "marginRate": "0.10",  # 10:1 leverage for exotic
                "guaranteedStopLossOrderMode": "DISABLED",
                "financing": {
                    "longRate": "-0.4500",
                    "shortRate": "0.3500",
                },
            },
        ]
    }


@pytest.fixture
def mock_oanda_account_response() -> Dict[str, Any]:
    """Mock OANDA account endpoint response."""
    return {
        "account": {
            "id": "101-001-12345678-001",
            "alias": "Test Account",
            "currency": "USD",
            "balance": "100000.0000",
            "createdByUserID": 12345678,
            "createdTime": "2024-01-01T00:00:00.000000000Z",
            "guaranteedStopLossOrderMode": "DISABLED",
            "pl": "1234.5678",
            "resettablePL": "1234.5678",
            "resettablePLTime": "2024-01-01T00:00:00.000000000Z",
            "financing": "-123.4567",
            "commission": "0.0000",
            "dividendAdjustment": "0",
            "guaranteedExecutionFees": "0.0000",
            "openTradeCount": 2,
            "openPositionCount": 2,
            "pendingOrderCount": 0,
            "hedgingEnabled": False,
            "unrealizedPL": "567.8901",
            "NAV": "100567.8901",
            "marginUsed": "2000.0000",
            "marginAvailable": "98567.8901",
            "positionValue": "100000.0000",
            "marginCloseoutUnrealizedPL": "567.8901",
            "marginCloseoutNAV": "100567.8901",
            "marginCloseoutMarginUsed": "2000.0000",
            "marginCloseoutPercent": "0.0099",
            "marginCloseoutPositionValue": "100000.0000",
            "withdrawalLimit": "98567.8901",
            "marginCallMarginUsed": "2000.0000",
            "marginCallPercent": "0.0099",
            "lastTransactionID": "12345",
        }
    }


# =============================================================================
# Forex Data Fixtures
# =============================================================================


@pytest.fixture
def forex_major_pairs() -> List[str]:
    """List of major forex pairs."""
    return [
        "EUR_USD",
        "USD_JPY",
        "GBP_USD",
        "USD_CHF",
        "AUD_USD",
        "USD_CAD",
        "NZD_USD",
    ]


@pytest.fixture
def forex_jpy_crosses() -> List[str]:
    """List of JPY cross pairs."""
    return [
        "EUR_JPY",
        "GBP_JPY",
        "AUD_JPY",
        "CHF_JPY",
        "CAD_JPY",
        "NZD_JPY",
    ]


@pytest.fixture
def forex_exotic_pairs() -> List[str]:
    """List of exotic pairs."""
    return [
        "USD_TRY",
        "USD_ZAR",
        "USD_MXN",
        "USD_SGD",
        "USD_HKD",
        "USD_NOK",
        "USD_SEK",
    ]


@pytest.fixture
def sample_forex_ohlcv_data() -> List[Dict[str, Any]]:
    """Sample OHLCV data for EUR/USD."""
    return [
        {
            "timestamp": "2025-01-15T00:00:00Z",
            "open": 1.08000,
            "high": 1.08150,
            "low": 1.07950,
            "close": 1.08100,
            "volume": 50000,
        },
        {
            "timestamp": "2025-01-15T01:00:00Z",
            "open": 1.08100,
            "high": 1.08200,
            "low": 1.08050,
            "close": 1.08150,
            "volume": 45000,
        },
        {
            "timestamp": "2025-01-15T02:00:00Z",
            "open": 1.08150,
            "high": 1.08250,
            "low": 1.08100,
            "close": 1.08200,
            "volume": 40000,
        },
        {
            "timestamp": "2025-01-15T03:00:00Z",
            "open": 1.08200,
            "high": 1.08300,
            "low": 1.08150,
            "close": 1.08250,
            "volume": 35000,
        },
        {
            "timestamp": "2025-01-15T04:00:00Z",
            "open": 1.08250,
            "high": 1.08350,
            "low": 1.08200,
            "close": 1.08300,
            "volume": 30000,
        },
    ]


@pytest.fixture
def sample_forex_tick_data() -> List[Dict[str, Any]]:
    """Sample tick data for EUR/USD."""
    return [
        {"timestamp": "2025-01-15T12:00:00.000Z", "bid": 1.08500, "ask": 1.08515},
        {"timestamp": "2025-01-15T12:00:00.100Z", "bid": 1.08502, "ask": 1.08517},
        {"timestamp": "2025-01-15T12:00:00.200Z", "bid": 1.08498, "ask": 1.08513},
        {"timestamp": "2025-01-15T12:00:00.300Z", "bid": 1.08505, "ask": 1.08520},
        {"timestamp": "2025-01-15T12:00:00.400Z", "bid": 1.08510, "ask": 1.08525},
    ]


# =============================================================================
# Session and Timing Fixtures
# =============================================================================


@pytest.fixture
def session_test_times() -> List[Tuple[int, int, str]]:
    """Test times for different forex sessions: (hour_utc, day_of_week, expected_session)."""
    return [
        # London/NY overlap (best liquidity)
        (13, 0, "london_ny_overlap"),  # Monday 13:00 UTC
        (14, 1, "london_ny_overlap"),  # Tuesday 14:00 UTC
        (15, 2, "london_ny_overlap"),  # Wednesday 15:00 UTC
        # London session
        (8, 0, "london"),  # Monday 08:00 UTC
        (10, 1, "london"),  # Tuesday 10:00 UTC
        # New York session (after London close)
        (17, 0, "new_york"),  # Monday 17:00 UTC
        (18, 1, "new_york"),  # Tuesday 18:00 UTC
        # Tokyo session
        (2, 0, "tokyo"),  # Monday 02:00 UTC
        (5, 1, "tokyo"),  # Tuesday 05:00 UTC
        # Tokyo/London overlap
        (8, 0, "tokyo_london_overlap"),  # Monday 08:00 UTC
        # Sydney session
        (22, 0, "sydney"),  # Monday 22:00 UTC (next day Sydney)
        (23, 6, "sydney"),  # Sunday 23:00 UTC (Sydney open)
        # Weekend
        (12, 5, "weekend"),  # Saturday 12:00 UTC
        (10, 6, "weekend"),  # Sunday 10:00 UTC (before market open)
    ]


@pytest.fixture
def weekend_times() -> List[Tuple[int, int]]:
    """Times when forex market is closed: (hour_utc, day_of_week)."""
    return [
        (0, 5),  # Saturday 00:00 UTC
        (12, 5),  # Saturday 12:00 UTC
        (23, 5),  # Saturday 23:00 UTC
        (0, 6),  # Sunday 00:00 UTC
        (12, 6),  # Sunday 12:00 UTC
        (20, 6),  # Sunday 20:00 UTC (before open)
    ]


@pytest.fixture
def high_liquidity_times() -> List[Tuple[int, int]]:
    """Times with high liquidity (London/NY overlap): (hour_utc, day_of_week)."""
    return [
        (12, 0),  # Monday 12:00 UTC
        (13, 1),  # Tuesday 13:00 UTC
        (14, 2),  # Wednesday 14:00 UTC
        (15, 3),  # Thursday 15:00 UTC
    ]


# =============================================================================
# Mock API Client Fixtures
# =============================================================================


@pytest.fixture
def mock_oanda_client():
    """Create a mock OANDA API client."""
    client = MagicMock()

    # Mock account methods
    client.account.get.return_value = MagicMock(
        body={"account": {"balance": "100000.00", "NAV": "100500.00"}}
    )

    # Mock pricing methods
    client.pricing.get.return_value = MagicMock(
        body={
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.08500"}],
                    "asks": [{"price": "1.08515"}],
                    "tradeable": True,
                }
            ]
        }
    )

    # Mock candle methods
    client.instrument.candles.return_value = MagicMock(
        body={
            "candles": [
                {
                    "time": "2025-01-15T12:00:00.000000000Z",
                    "mid": {"o": "1.08500", "h": "1.08600", "l": "1.08400", "c": "1.08550"},
                    "volume": 10000,
                    "complete": True,
                }
            ]
        }
    )

    return client


# =============================================================================
# Parametrized Test Data
# =============================================================================

# Currency pair test data: (symbol, expected_category, expected_pip_size)
CURRENCY_PAIR_TEST_DATA = [
    ("EUR_USD", "major", 0.0001),
    ("USD_JPY", "major", 0.01),
    ("GBP_USD", "major", 0.0001),
    ("USD_CHF", "major", 0.0001),
    ("AUD_USD", "major", 0.0001),
    ("USD_CAD", "major", 0.0001),
    ("NZD_USD", "major", 0.0001),
    ("EUR_GBP", "minor", 0.0001),
    ("EUR_CHF", "minor", 0.0001),
    ("EUR_JPY", "cross", 0.01),
    ("GBP_JPY", "cross", 0.01),
    ("AUD_JPY", "cross", 0.01),
    ("USD_TRY", "exotic", 0.0001),
    ("USD_ZAR", "exotic", 0.0001),
    ("USD_MXN", "exotic", 0.0001),
]

# Session timing test data: (hour_utc, day_of_week, expected_liquidity_range)
SESSION_LIQUIDITY_TEST_DATA = [
    (14, 0, (1.2, 1.5)),  # London/NY overlap - highest
    (10, 1, (1.0, 1.2)),  # London - high
    (18, 2, (0.9, 1.15)),  # New York - moderate-high
    (3, 3, (0.6, 0.9)),  # Tokyo - moderate
    (23, 6, (0.5, 0.8)),  # Sydney - lower
    (12, 5, (0.0, 0.0)),  # Weekend - zero
]


def get_currency_pair_test_data():
    """Get currency pair test data for parametrization."""
    return CURRENCY_PAIR_TEST_DATA


def get_session_liquidity_test_data():
    """Get session liquidity test data for parametrization."""
    return SESSION_LIQUIDITY_TEST_DATA
