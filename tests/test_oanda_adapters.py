# -*- coding: utf-8 -*-
"""
tests/test_oanda_adapters.py
Comprehensive tests for OANDA forex adapters.

Phase 2: OANDA Adapter Implementation Tests

Test Categories:
1. Market Data Adapter (~30 tests)
2. Trading Hours Adapter (~35 tests)
3. Fee Adapter (~25 tests)
4. Exchange Info Adapter (~20 tests)
5. Order Execution Adapter (~20 tests)

Total: ~130 tests

Coverage:
- Basic functionality
- Edge cases (DST, weekend, holidays)
- Error handling
- API response parsing
- Session detection
- Spread calculations
- Order flow
"""

from __future__ import annotations

import math
import pytest
import requests
from requests.exceptions import (
    Timeout as RequestsTimeout,
    ConnectionError as RequestsConnectionError,
    HTTPError as RequestsHTTPError,
    RequestException,
)
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, patch
from zoneinfo import ZoneInfo

from core_models import Bar, Tick, Order, Side, OrderType

# Import adapters
from adapters.oanda.market_data import (
    OandaMarketDataAdapter,
    RateLimiter,
    OandaCandleData,
)
from adapters.oanda.trading_hours import (
    OandaTradingHoursAdapter,
)
from adapters.oanda.fees import (
    OandaFeeAdapter,
    RETAIL_SPREADS,
    INSTITUTIONAL_SPREADS,
)
from adapters.oanda.exchange_info import (
    OandaExchangeInfoAdapter,
    MAJOR_PAIRS,
    MINOR_PAIRS,
    CROSS_PAIRS,
    EXOTIC_PAIRS,
)
from adapters.oanda.order_execution import (
    OandaOrderExecutionAdapter,
    OandaOrderType,
    OandaTimeInForce,
)
from adapters.models import (
    ExchangeVendor,
    MarketType,
    ForexSessionType,
)


# =========================
# Fixtures
# =========================


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Standard mock configuration for testing."""
    return {
        "api_key": "test_api_key",
        "account_id": "test_account_id",
        "practice": True,
    }


@pytest.fixture
def market_data_adapter(mock_config) -> OandaMarketDataAdapter:
    """Create OandaMarketDataAdapter instance for testing."""
    return OandaMarketDataAdapter(config=mock_config)


@pytest.fixture
def trading_hours_adapter() -> OandaTradingHoursAdapter:
    """Create OandaTradingHoursAdapter instance for testing."""
    return OandaTradingHoursAdapter()


@pytest.fixture
def fee_adapter() -> OandaFeeAdapter:
    """Create OandaFeeAdapter instance for testing."""
    return OandaFeeAdapter()


@pytest.fixture
def exchange_info_adapter() -> OandaExchangeInfoAdapter:
    """Create OandaExchangeInfoAdapter instance for testing."""
    return OandaExchangeInfoAdapter()


@pytest.fixture
def order_execution_adapter(mock_config) -> OandaOrderExecutionAdapter:
    """Create OandaOrderExecutionAdapter instance for testing."""
    return OandaOrderExecutionAdapter(config=mock_config)


# =========================
# Rate Limiter Tests
# =========================


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init_default_values(self):
        """Test rate limiter initialization with defaults."""
        limiter = RateLimiter()
        assert limiter.rate == 120.0
        assert limiter.burst == 200

    def test_init_custom_values(self):
        """Test rate limiter initialization with custom values."""
        limiter = RateLimiter(rate=60.0, burst=100)
        assert limiter.rate == 60.0
        assert limiter.burst == 100

    def test_available_tokens_initial(self):
        """Test initial token count equals burst."""
        limiter = RateLimiter(burst=200)
        assert limiter.available_tokens >= 199.0  # Allow for timing

    def test_acquire_sync_consumes_token(self):
        """Test that sync acquire consumes a token."""
        limiter = RateLimiter(rate=120.0, burst=200)
        initial = limiter.available_tokens
        limiter.acquire_sync()
        assert limiter.available_tokens < initial

    def test_acquire_sync_no_block_with_tokens(self):
        """Test that acquire_sync doesn't block when tokens available."""
        import time

        limiter = RateLimiter(rate=120.0, burst=200)
        start = time.monotonic()
        limiter.acquire_sync()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be nearly instant


# =========================
# Market Data Adapter Tests
# =========================


class TestOandaMarketDataAdapter:
    """Tests for OandaMarketDataAdapter."""

    def test_init_default_vendor(self, market_data_adapter):
        """Test adapter initializes with OANDA vendor."""
        assert market_data_adapter.vendor == ExchangeVendor.OANDA

    def test_init_practice_environment(self, market_data_adapter):
        """Test adapter uses practice URL by default."""
        assert market_data_adapter._practice is True
        assert "fxpractice" in market_data_adapter._base_url

    def test_init_live_environment(self, mock_config):
        """Test adapter uses live URL when practice=False."""
        mock_config["practice"] = False
        adapter = OandaMarketDataAdapter(config=mock_config)
        assert adapter._practice is False
        assert "fxtrade" in adapter._base_url

    def test_normalize_symbol_forward_slash(self, market_data_adapter):
        """Test symbol normalization with forward slash."""
        assert market_data_adapter._normalize_symbol("EUR/USD") == "EUR_USD"

    def test_normalize_symbol_underscore(self, market_data_adapter):
        """Test symbol normalization with underscore."""
        assert market_data_adapter._normalize_symbol("EUR_USD") == "EUR_USD"

    def test_normalize_symbol_no_separator(self, market_data_adapter):
        """Test symbol normalization without separator."""
        assert market_data_adapter._normalize_symbol("EURUSD") == "EUR_USD"

    def test_normalize_symbol_lowercase(self, market_data_adapter):
        """Test symbol normalization with lowercase."""
        assert market_data_adapter._normalize_symbol("eur/usd") == "EUR_USD"

    def test_parse_timeframe_1h(self, market_data_adapter):
        """Test timeframe parsing for 1 hour."""
        assert market_data_adapter._parse_timeframe("1h") == "H1"

    def test_parse_timeframe_4h(self, market_data_adapter):
        """Test timeframe parsing for 4 hours."""
        assert market_data_adapter._parse_timeframe("4h") == "H4"

    def test_parse_timeframe_1d(self, market_data_adapter):
        """Test timeframe parsing for 1 day."""
        assert market_data_adapter._parse_timeframe("1d") == "D"

    def test_parse_timeframe_native_format(self, market_data_adapter):
        """Test timeframe parsing with native OANDA format."""
        assert market_data_adapter._parse_timeframe("H1") == "H1"
        assert market_data_adapter._parse_timeframe("M15") == "M15"

    def test_parse_timeframe_invalid(self, market_data_adapter):
        """Test timeframe parsing with invalid value."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            market_data_adapter._parse_timeframe("invalid")

    def test_get_pip_size_standard_pair(self, market_data_adapter):
        """Test pip size for standard pair."""
        assert market_data_adapter.get_pip_size("EUR_USD") == 0.0001

    def test_get_pip_size_jpy_pair(self, market_data_adapter):
        """Test pip size for JPY pair."""
        assert market_data_adapter.get_pip_size("USD_JPY") == 0.01

    def test_get_pip_size_jpy_cross(self, market_data_adapter):
        """Test pip size for JPY cross."""
        assert market_data_adapter.get_pip_size("EUR/JPY") == 0.01

    def test_get_headers_with_api_key(self, market_data_adapter):
        """Test header generation with API key."""
        headers = market_data_adapter._get_headers()
        assert "Bearer test_api_key" in headers["Authorization"]
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_no_api_key(self):
        """Test header generation without API key raises error."""
        adapter = OandaMarketDataAdapter(config={"practice": True})
        adapter._api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            adapter._get_headers()

    def test_parse_candle_success(self, market_data_adapter):
        """Test parsing a valid candle response."""
        candle = {
            "time": "1609459200.000000000",  # 2021-01-01 00:00:00 UTC
            "complete": True,
            "volume": 1000,
            "mid": {"o": "1.22345", "h": "1.22500", "l": "1.22200", "c": "1.22400"},
            "bid": {"o": "1.22340", "h": "1.22495", "l": "1.22195", "c": "1.22395"},
            "ask": {"o": "1.22350", "h": "1.22505", "l": "1.22205", "c": "1.22405"},
        }
        bar = market_data_adapter._parse_candle("EUR_USD", candle)

        assert bar is not None
        assert bar.symbol == "EUR_USD"
        assert float(bar.open) == pytest.approx(1.22345)
        assert float(bar.high) == pytest.approx(1.22500)
        assert float(bar.low) == pytest.approx(1.22200)
        assert float(bar.close) == pytest.approx(1.22400)
        assert int(bar.volume_base) == 1000


# =========================
# Trading Hours Adapter Tests
# =========================


class TestOandaTradingHoursAdapter:
    """Tests for OandaTradingHoursAdapter."""

    def test_init_default_vendor(self, trading_hours_adapter):
        """Test adapter initializes with OANDA vendor."""
        assert trading_hours_adapter.vendor == ExchangeVendor.OANDA

    def test_market_open_monday(self, trading_hours_adapter):
        """Test market is open on Monday."""
        # Monday 2024-01-15 10:00:00 UTC
        ts = int(datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is True

    def test_market_open_friday_before_close(self, trading_hours_adapter):
        """Test market is open on Friday before 5pm ET."""
        # Friday 2024-01-19 15:00:00 ET (20:00 UTC in winter)
        et = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 19, 15, 0, 0, tzinfo=et)
        ts = int(dt.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is True

    def test_market_closed_friday_after_close(self, trading_hours_adapter):
        """Test market is closed on Friday after 5pm ET."""
        # Friday 2024-01-19 17:30:00 ET (22:30 UTC in winter)
        et = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 19, 17, 30, 0, tzinfo=et)
        ts = int(dt.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is False

    def test_market_closed_saturday(self, trading_hours_adapter):
        """Test market is closed on Saturday."""
        # Saturday 2024-01-20 12:00:00 UTC
        ts = int(datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is False

    def test_market_closed_sunday_early(self, trading_hours_adapter):
        """Test market is closed on Sunday before 5pm ET."""
        # Sunday 2024-01-21 12:00:00 ET (17:00 UTC in winter)
        et = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 21, 12, 0, 0, tzinfo=et)
        ts = int(dt.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is False

    def test_market_open_sunday_after_open(self, trading_hours_adapter):
        """Test market is open on Sunday after 5pm ET."""
        # Sunday 2024-01-21 17:30:00 ET (22:30 UTC in winter)
        et = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 21, 17, 30, 0, tzinfo=et)
        ts = int(dt.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is True

    def test_session_london_ny_overlap(self, trading_hours_adapter):
        """Test London/NY overlap session detection."""
        # 14:00 UTC is London/NY overlap
        ts = int(datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.LONDON_NY_OVERLAP
        assert liquidity == 1.35
        assert spread == 0.8

    def test_session_london(self, trading_hours_adapter):
        """Test London session detection."""
        # 10:00 UTC is London session
        ts = int(datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.LONDON
        assert liquidity == 1.10
        assert spread == 1.0

    def test_session_tokyo(self, trading_hours_adapter):
        """Test Tokyo session detection."""
        # 03:00 UTC is Tokyo session
        ts = int(datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.TOKYO
        assert liquidity == 0.75
        assert spread == 1.3

    def test_session_sydney(self, trading_hours_adapter):
        """Test Sydney session detection."""
        # 23:00 UTC is Sydney session
        ts = int(datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.SYDNEY
        assert liquidity == 0.65
        assert spread == 1.5

    def test_session_new_york(self, trading_hours_adapter):
        """Test New York session detection."""
        # 18:00 UTC is NY session (after overlap)
        ts = int(datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.NEW_YORK
        assert liquidity == 1.05
        assert spread == 1.0

    def test_session_weekend(self, trading_hours_adapter):
        """Test weekend session detection."""
        # Saturday 12:00 UTC
        ts = int(datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, liquidity, spread = trading_hours_adapter.get_current_session(ts)

        assert session == ForexSessionType.WEEKEND
        assert liquidity == 0.0
        assert spread == float("inf")

    def test_rollover_time_winter(self, trading_hours_adapter):
        """Test rollover time in winter (standard time)."""
        rollover = trading_hours_adapter.get_rollover_time_utc(date(2024, 1, 15))
        # 5pm ET in winter = 22:00 UTC
        assert rollover.hour == 22

    def test_rollover_time_summer(self, trading_hours_adapter):
        """Test rollover time in summer (DST)."""
        rollover = trading_hours_adapter.get_rollover_time_utc(date(2024, 7, 15))
        # 5pm ET in summer = 21:00 UTC
        assert rollover.hour == 21

    def test_is_rollover_time_within_window(self, trading_hours_adapter):
        """Test rollover detection within tolerance window."""
        # Get rollover time
        rollover = trading_hours_adapter.get_rollover_time_utc(date(2024, 1, 15))
        ts = int(rollover.timestamp() * 1000)

        assert trading_hours_adapter.is_rollover_time(ts, tolerance_minutes=5) is True

    def test_is_rollover_time_outside_window(self, trading_hours_adapter):
        """Test rollover detection outside tolerance window."""
        # 1 hour before rollover
        rollover = trading_hours_adapter.get_rollover_time_utc(date(2024, 1, 15))
        ts = int((rollover - timedelta(hours=1)).timestamp() * 1000)

        assert trading_hours_adapter.is_rollover_time(ts, tolerance_minutes=5) is False

    def test_liquidity_factor(self, trading_hours_adapter):
        """Test liquidity factor retrieval."""
        ts = int(datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        liquidity = trading_hours_adapter.get_liquidity_factor(ts)
        assert liquidity == 1.35  # London/NY overlap

    def test_spread_multiplier(self, trading_hours_adapter):
        """Test spread multiplier retrieval."""
        ts = int(datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        spread = trading_hours_adapter.get_spread_multiplier(ts)
        assert spread == 0.8  # London/NY overlap

    def test_session_for_jpy_pair_tokyo(self, trading_hours_adapter):
        """Test adjusted liquidity for JPY pair during Tokyo session."""
        # 03:00 UTC is Tokyo session
        ts = int(datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session, adjusted_liq = trading_hours_adapter.get_session_for_pair("USD_JPY", ts)

        assert session == ForexSessionType.TOKYO
        assert adjusted_liq == 0.75 * 1.2  # 20% boost for JPY in Tokyo

    def test_trading_day_saturday(self, trading_hours_adapter):
        """Test Saturday is not a trading day."""
        assert trading_hours_adapter.is_trading_day(date(2024, 1, 20)) is False  # Saturday

    def test_trading_day_weekday(self, trading_hours_adapter):
        """Test weekdays are trading days."""
        assert trading_hours_adapter.is_trading_day(date(2024, 1, 15)) is True  # Monday

    def test_hours_until_close_weekday(self, trading_hours_adapter):
        """Test hours until close on weekday."""
        et = ZoneInfo("America/New_York")
        # Wednesday 10am ET
        dt = datetime(2024, 1, 17, 10, 0, 0, tzinfo=et)
        ts = int(dt.timestamp() * 1000)

        hours = trading_hours_adapter.get_hours_until_close(ts)
        # Close is Friday 5pm, so about 55 hours
        assert hours > 50
        assert hours < 60


# =========================
# Fee Adapter Tests
# =========================


class TestOandaFeeAdapter:
    """Tests for OandaFeeAdapter."""

    def test_init_default_profile(self, fee_adapter):
        """Test adapter initializes with retail profile."""
        assert fee_adapter.spread_profile == "retail"

    def test_init_institutional_profile(self):
        """Test adapter with institutional profile."""
        adapter = OandaFeeAdapter(config={"spread_profile": "institutional"})
        assert "EUR_USD" in adapter._spreads
        assert adapter._spreads["EUR_USD"] == INSTITUTIONAL_SPREADS["EUR_USD"]

    def test_compute_fee_returns_zero(self, fee_adapter):
        """Test that compute_fee returns zero (forex is spread-based)."""
        fee = fee_adapter.compute_fee(
            notional=100000,
            side=Side.BUY,
            liquidity="taker",
        )
        assert fee == 0.0

    def test_get_fee_schedule(self, fee_adapter):
        """Test fee schedule returns zero rates."""
        schedule = fee_adapter.get_fee_schedule("EUR_USD")
        assert schedule.maker_rate == 0.0
        assert schedule.taker_rate == 0.0
        assert schedule.flat_fee == 0.0

    def test_get_effective_rates(self, fee_adapter):
        """Test effective rates are zero."""
        maker, taker = fee_adapter.get_effective_rates("EUR_USD")
        assert maker == 0.0
        assert taker == 0.0

    def test_get_spread_pips_major(self, fee_adapter):
        """Test spread for major pair."""
        spread = fee_adapter.get_spread_pips("EUR_USD")
        assert spread == RETAIL_SPREADS["EUR_USD"]

    def test_get_spread_pips_cross(self, fee_adapter):
        """Test spread for cross pair."""
        spread = fee_adapter.get_spread_pips("EUR_JPY")
        assert spread == RETAIL_SPREADS["EUR_JPY"]

    def test_get_spread_pips_unknown(self, fee_adapter):
        """Test spread estimation for unknown pair."""
        spread = fee_adapter.get_spread_pips("UNKNOWN_PAIR")
        assert spread > 0  # Should return estimated spread

    def test_get_pip_size_standard(self, fee_adapter):
        """Test pip size for standard pair."""
        assert fee_adapter.get_pip_size("EUR_USD") == 0.0001

    def test_get_pip_size_jpy(self, fee_adapter):
        """Test pip size for JPY pair."""
        assert fee_adapter.get_pip_size("USD_JPY") == 0.01

    def test_compute_swap_long_eurusd(self, fee_adapter):
        """Test swap computation for long EUR/USD."""
        swap = fee_adapter.compute_swap(
            notional=100000,
            symbol="EUR_USD",
            side=Side.BUY,
            days=1,
        )
        # EUR/USD long swap is typically negative (pay)
        assert swap < 0

    def test_compute_swap_short_eurusd(self, fee_adapter):
        """Test swap computation for short EUR/USD."""
        swap = fee_adapter.compute_swap(
            notional=100000,
            symbol="EUR_USD",
            side=Side.SELL,
            days=1,
        )
        # EUR/USD short swap is typically positive (receive)
        assert swap > 0

    def test_compute_swap_multiple_days(self, fee_adapter):
        """Test swap scales with days."""
        swap_1 = fee_adapter.compute_swap(100000, "EUR_USD", Side.BUY, days=1)
        swap_5 = fee_adapter.compute_swap(100000, "EUR_USD", Side.BUY, days=5)
        assert abs(swap_5) == pytest.approx(abs(swap_1) * 5, rel=0.01)

    def test_get_swap_rates(self, fee_adapter):
        """Test swap rates retrieval."""
        long_swap, short_swap = fee_adapter.get_swap_rates("EUR_USD")
        assert long_swap < 0  # Pay to be long EUR
        assert short_swap > 0  # Receive to be short EUR

    def test_get_spread_cost(self, fee_adapter):
        """Test spread cost calculation."""
        cost = fee_adapter.get_spread_cost(
            symbol="EUR_USD",
            notional=100000,
            price=1.10,
        )
        # Should be approximately half the spread in value terms
        assert cost > 0
        assert cost < 100  # Reasonable range for 100k notional

    def test_estimate_total_cost(self, fee_adapter):
        """Test total cost estimation."""
        result = fee_adapter.estimate_total_cost(
            symbol="EUR_USD",
            notional=100000,
            price=1.10,
            side=Side.BUY,
            holding_days=1,
        )
        assert "spread_cost" in result
        assert "swap_cost" in result
        assert "total_cost" in result
        assert result["spread_pips"] == RETAIL_SPREADS["EUR_USD"]

    def test_update_swap_rates(self, fee_adapter):
        """Test swap rates update."""
        fee_adapter.update_swap_rates({"EUR_USD": (-1.0, 0.5)})
        long_swap, short_swap = fee_adapter.get_swap_rates("EUR_USD")
        assert long_swap == -1.0
        assert short_swap == 0.5

    def test_update_spreads(self, fee_adapter):
        """Test spread update."""
        fee_adapter.update_spreads({"EUR_USD": 0.5})
        assert fee_adapter.get_spread_pips("EUR_USD") == 0.5


# =========================
# Exchange Info Adapter Tests
# =========================


class TestOandaExchangeInfoAdapter:
    """Tests for OandaExchangeInfoAdapter."""

    def test_init_static_instruments(self, exchange_info_adapter):
        """Test adapter initializes with static instruments."""
        symbols = exchange_info_adapter.get_tradable_symbols()
        assert len(symbols) > 0
        assert "EUR_USD" in symbols

    def test_get_symbol_info_major(self, exchange_info_adapter):
        """Test symbol info for major pair."""
        info = exchange_info_adapter.get_symbol_info("EUR_USD")
        assert info is not None
        assert info.symbol == "EUR_USD"
        assert info.exchange_rule.base_asset == "EUR"
        assert info.exchange_rule.quote_asset == "USD"
        assert info.market_type == MarketType.FOREX

    def test_get_symbol_info_jpy(self, exchange_info_adapter):
        """Test symbol info for JPY pair."""
        info = exchange_info_adapter.get_symbol_info("USD_JPY")
        assert info is not None
        assert info.exchange_rule.tick_size == Decimal("0.01")

    def test_get_symbol_info_normalized(self, exchange_info_adapter):
        """Test symbol info with different formats."""
        info1 = exchange_info_adapter.get_symbol_info("EUR/USD")
        info2 = exchange_info_adapter.get_symbol_info("EUR_USD")
        assert info1 is not None
        assert info2 is not None
        assert info1.symbol == info2.symbol

    def test_get_exchange_rule_major(self, exchange_info_adapter):
        """Test exchange rule for major pair."""
        rule = exchange_info_adapter.get_exchange_rule("EUR_USD")
        assert rule is not None
        assert rule.is_tradable is True
        assert rule.is_shortable is True
        assert rule.market_type == MarketType.FOREX

    def test_get_pip_size_standard(self, exchange_info_adapter):
        """Test pip size for standard pair."""
        pip = exchange_info_adapter.get_pip_size("EUR_USD")
        assert pip == Decimal("0.0001")

    def test_get_pip_size_jpy(self, exchange_info_adapter):
        """Test pip size for JPY pair."""
        pip = exchange_info_adapter.get_pip_size("USD_JPY")
        assert pip == Decimal("0.01")

    def test_get_margin_rate_major(self, exchange_info_adapter):
        """Test margin rate for major pair."""
        margin = exchange_info_adapter.get_margin_rate("EUR_USD")
        assert margin == 0.02  # 50:1 leverage

    def test_get_margin_rate_exotic(self, exchange_info_adapter):
        """Test margin rate for exotic pair."""
        margin = exchange_info_adapter.get_margin_rate("USD_TRY")
        assert margin == 0.10  # 10:1 leverage

    def test_get_max_leverage_major(self, exchange_info_adapter):
        """Test max leverage for major pair."""
        leverage = exchange_info_adapter.get_max_leverage("EUR_USD")
        assert leverage == 50.0

    def test_classify_pair_major(self, exchange_info_adapter):
        """Test pair classification for major."""
        assert exchange_info_adapter.classify_pair("EUR_USD") == "major"
        assert exchange_info_adapter.classify_pair("GBP_USD") == "major"

    def test_classify_pair_minor(self, exchange_info_adapter):
        """Test pair classification for minor."""
        assert exchange_info_adapter.classify_pair("EUR_GBP") == "minor"

    def test_classify_pair_cross(self, exchange_info_adapter):
        """Test pair classification for cross."""
        assert exchange_info_adapter.classify_pair("EUR_JPY") == "cross"

    def test_classify_pair_exotic(self, exchange_info_adapter):
        """Test pair classification for exotic."""
        assert exchange_info_adapter.classify_pair("USD_TRY") == "exotic"

    def test_is_tradable_known_pair(self, exchange_info_adapter):
        """Test tradable check for known pair."""
        assert exchange_info_adapter.is_tradable("EUR_USD") is True

    def test_is_tradable_unknown_pair(self, exchange_info_adapter):
        """Test tradable check for unknown pair."""
        assert exchange_info_adapter.is_tradable("XXX_YYY") is False

    def test_get_all_majors(self, exchange_info_adapter):
        """Test getting all major pairs."""
        majors = exchange_info_adapter.get_all_majors()
        assert "EUR_USD" in majors
        assert "USD_JPY" in majors
        assert len(majors) == len(MAJOR_PAIRS)

    def test_validate_order_valid(self, exchange_info_adapter):
        """Test order validation for valid order."""
        result = exchange_info_adapter.validate_order(
            symbol="EUR_USD",
            qty=10000,
            price=1.12345,
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_order_qty_too_small(self, exchange_info_adapter):
        """Test order validation for quantity too small."""
        result = exchange_info_adapter.validate_order(
            symbol="EUR_USD",
            qty=0.1,
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# =========================
# Order Execution Adapter Tests
# =========================


class TestOandaOrderExecutionAdapter:
    """Tests for OandaOrderExecutionAdapter."""

    def test_init_default_vendor(self, order_execution_adapter):
        """Test adapter initializes with OANDA vendor."""
        assert order_execution_adapter.vendor == ExchangeVendor.OANDA

    def test_init_practice_environment(self, order_execution_adapter):
        """Test adapter uses practice URL by default."""
        assert order_execution_adapter._practice is True
        assert "fxpractice" in order_execution_adapter._base_url

    def test_normalize_symbol(self, order_execution_adapter):
        """Test symbol normalization."""
        assert order_execution_adapter._normalize_symbol("EUR/USD") == "EUR_USD"

    def test_build_order_request_market_buy(self, order_execution_adapter):
        """Test building market buy order request."""
        order = Order(
            ts=1609459200000,
            symbol="EUR_USD",
            side=Side.BUY,
            quantity=Decimal("10000"),
            order_type=OrderType.MARKET,
        )
        request = order_execution_adapter._build_order_request(order, "EUR_USD")

        assert request["order"]["instrument"] == "EUR_USD"
        assert request["order"]["type"] == "MARKET"
        assert int(request["order"]["units"]) > 0  # Positive for buy

    def test_build_order_request_market_sell(self, order_execution_adapter):
        """Test building market sell order request."""
        order = Order(
            ts=1609459200000,
            symbol="EUR_USD",
            side=Side.SELL,
            quantity=Decimal("10000"),
            order_type=OrderType.MARKET,
        )
        request = order_execution_adapter._build_order_request(order, "EUR_USD")

        assert int(request["order"]["units"]) < 0  # Negative for sell

    def test_build_order_request_limit(self, order_execution_adapter):
        """Test building limit order request."""
        order = Order(
            ts=1609459200000,
            symbol="EUR_USD",
            side=Side.BUY,
            quantity=Decimal("10000"),
            order_type=OrderType.LIMIT,
            price=Decimal("1.10000"),
        )
        request = order_execution_adapter._build_order_request(order, "EUR_USD")

        assert request["order"]["type"] == "LIMIT"
        assert request["order"]["price"] == "1.10000"
        assert request["order"]["timeInForce"] == "GTC"

    def test_parse_order_response_filled(self, order_execution_adapter):
        """Test parsing filled order response."""
        response = {
            "orderFillTransaction": {
                "orderID": "123",
                "units": "10000",
                "price": "1.10500",
                "commission": "0.0",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client123")

        assert result.success is True
        assert result.order_id == "123"
        assert result.status == "FILLED"
        assert result.filled_qty == Decimal("10000")
        assert result.filled_price == Decimal("1.10500")

    def test_parse_order_response_pending(self, order_execution_adapter):
        """Test parsing pending order response."""
        response = {
            "orderCreateTransaction": {
                "id": "456",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client456")

        assert result.success is True
        assert result.order_id == "456"
        assert result.status == "PENDING"

    def test_parse_order_response_rejected(self, order_execution_adapter):
        """Test parsing rejected order response."""
        response = {
            "orderRejectTransaction": {
                "rejectReason": "INSUFFICIENT_MARGIN",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client789")

        assert result.success is False
        assert result.error_code == "INSUFFICIENT_MARGIN"

    def test_oanda_order_types_enum(self):
        """Test OANDA order types enum values."""
        assert OandaOrderType.MARKET.value == "MARKET"
        assert OandaOrderType.LIMIT.value == "LIMIT"
        assert OandaOrderType.STOP.value == "STOP"
        assert OandaOrderType.TAKE_PROFIT.value == "TAKE_PROFIT"
        assert OandaOrderType.STOP_LOSS.value == "STOP_LOSS"

    def test_oanda_time_in_force_enum(self):
        """Test OANDA time-in-force enum values."""
        assert OandaTimeInForce.GTC.value == "GTC"
        assert OandaTimeInForce.FOK.value == "FOK"
        assert OandaTimeInForce.IOC.value == "IOC"


# =========================
# Integration Tests
# =========================


class TestOandaAdapterIntegration:
    """Integration tests for OANDA adapters."""

    def test_adapter_registration(self):
        """Test that adapters are registered in the registry."""
        from adapters.registry import get_registry, AdapterType

        # Trigger lazy loading by importing oanda package
        import adapters.oanda  # noqa: F401

        registry = get_registry()

        # Check OANDA adapters are registered (may need to trigger lazy loading first)
        reg = registry.get_registration(ExchangeVendor.OANDA, AdapterType.MARKET_DATA)
        assert reg is not None, "OANDA market data adapter not registered"

    def test_create_adapter_from_registry(self):
        """Test creating adapter via registry."""
        from adapters.registry import create_market_data_adapter

        adapter = create_market_data_adapter(
            "oanda",
            {
                "api_key": "test_key",
                "account_id": "test_account",
                "practice": True,
            },
        )

        assert adapter is not None
        assert isinstance(adapter, OandaMarketDataAdapter)

    def test_all_forex_pairs_classified(self, exchange_info_adapter):
        """Test that all defined pairs can be classified."""
        all_pairs = MAJOR_PAIRS | MINOR_PAIRS | CROSS_PAIRS | EXOTIC_PAIRS

        for pair in all_pairs:
            classification = exchange_info_adapter.classify_pair(pair)
            assert classification in ("major", "minor", "cross", "exotic")

    def test_session_coverage_24h(self, trading_hours_adapter):
        """Test that every hour maps to a session (when market open)."""
        # Test a Monday
        base_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        for hour in range(24):
            dt = base_date.replace(hour=hour)
            ts = int(dt.timestamp() * 1000)
            session, liq, spread = trading_hours_adapter.get_current_session(ts)

            # Should get a valid session
            assert session is not None
            assert liq > 0
            assert spread > 0

    def test_spread_consistency(self, fee_adapter, exchange_info_adapter):
        """Test spread and pip size consistency."""
        for pair in ["EUR_USD", "USD_JPY", "GBP_USD"]:
            spread_pips = fee_adapter.get_spread_pips(pair)
            pip_size = float(exchange_info_adapter.get_pip_size(pair))

            # Spread in price terms
            spread_price = spread_pips * pip_size

            # Should be reasonable
            if "JPY" in pair:
                assert 0.001 < spread_price < 1.0  # JPY pairs
            else:
                assert 0.00001 < spread_price < 0.01  # Standard pairs


# =========================
# Edge Case Tests
# =========================


class TestOandaEdgeCases:
    """Edge case tests for OANDA adapters."""

    def test_dst_transition_spring(self, trading_hours_adapter):
        """Test session detection during spring DST transition."""
        # March 10, 2024 - US DST starts (clocks spring forward at 2am)
        # Note: Forex market opens at 5pm ET on Sunday
        et = ZoneInfo("America/New_York")

        # Test after market opens (6pm ET on Sunday)
        after_open = datetime(2024, 3, 10, 18, 0, 0, tzinfo=et)
        ts_after_open = int(after_open.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts_after_open) is True

        # Before 5pm ET, market is closed even on DST transition day
        before_open = datetime(2024, 3, 10, 12, 0, 0, tzinfo=et)
        ts_before_open = int(before_open.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts_before_open) is False

    def test_dst_transition_fall(self, trading_hours_adapter):
        """Test session detection during fall DST transition."""
        # November 3, 2024 - US DST ends (clocks fall back at 2am)
        # Note: Forex market opens at 5pm ET on Sunday
        et = ZoneInfo("America/New_York")

        # Test after market opens (6pm ET on Sunday)
        after_open = datetime(2024, 11, 3, 18, 0, 0, tzinfo=et)
        ts_after_open = int(after_open.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts_after_open) is True

        # Before 5pm ET, market is closed even on DST transition day
        before_open = datetime(2024, 11, 3, 12, 0, 0, tzinfo=et)
        ts_before_open = int(before_open.timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts_before_open) is False

    def test_new_years_eve(self, trading_hours_adapter):
        """Test market behavior around New Year."""
        # Dec 31, 2024 is Tuesday - market should be open
        ts = int(datetime(2024, 12, 31, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert trading_hours_adapter.is_market_open(ts) is True

    def test_exotic_pair_high_spread(self, fee_adapter):
        """Test that exotic pairs have higher spreads."""
        major_spread = fee_adapter.get_spread_pips("EUR_USD")
        exotic_spread = fee_adapter.get_spread_pips("USD_TRY")

        assert exotic_spread > major_spread * 10  # Much wider

    def test_symbol_normalization_edge_cases(self, market_data_adapter):
        """Test symbol normalization edge cases."""
        # Various formats
        assert market_data_adapter._normalize_symbol("eurusd") == "EUR_USD"
        assert market_data_adapter._normalize_symbol("EUR-USD") == "EUR_USD"
        assert market_data_adapter._normalize_symbol("EUR/usd") == "EUR_USD"

    def test_zero_notional_fee(self, fee_adapter):
        """Test fee calculation with zero notional."""
        fee = fee_adapter.compute_fee(
            notional=0,
            side=Side.BUY,
            liquidity="taker",
        )
        assert fee == 0.0

    def test_empty_positions_response(self, order_execution_adapter):
        """Test handling of empty positions response."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {"positions": []}
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            positions = order_execution_adapter.get_positions()
            assert positions == {}


# =========================
# Backward Compatibility Tests
# =========================


class TestOandaBackwardCompatibility:
    """Backward compatibility tests."""

    def test_forex_market_type_exists(self):
        """Test FOREX market type is available."""
        assert MarketType.FOREX.value == "FOREX"

    def test_oanda_vendor_exists(self):
        """Test OANDA vendor is available."""
        assert ExchangeVendor.OANDA.value == "oanda"

    def test_vendor_market_type(self):
        """Test OANDA vendor returns FOREX market type."""
        assert ExchangeVendor.OANDA.market_type == MarketType.FOREX

    def test_vendor_is_forex(self):
        """Test OANDA vendor is_forex property."""
        assert ExchangeVendor.OANDA.is_forex is True

    def test_session_type_enum_values(self):
        """Test ForexSessionType enum values."""
        assert ForexSessionType.LONDON.value == "london"
        assert ForexSessionType.NEW_YORK.value == "new_york"
        assert ForexSessionType.TOKYO.value == "tokyo"
        assert ForexSessionType.SYDNEY.value == "sydney"


# =========================
# API Error Handling Tests
# =========================


class TestOandaApiErrorHandling:
    """Tests for OANDA API error handling."""

    def test_market_data_network_timeout(self, market_data_adapter):
        """Test handling of network timeout."""
        with patch.object(market_data_adapter, "_ensure_session") as mock_session:
            mock_session.return_value.get.side_effect = RequestsTimeout("Connection timed out")

            # Adapter wraps exceptions in ConnectionError
            with pytest.raises(ConnectionError) as exc_info:
                market_data_adapter.get_bars("EUR_USD", "H1", limit=10)
            assert "Connection timed out" in str(exc_info.value)

    def test_market_data_connection_error(self, market_data_adapter):
        """Test handling of connection error."""
        with patch.object(market_data_adapter, "_ensure_session") as mock_session:
            mock_session.return_value.get.side_effect = RequestsConnectionError(
                "Failed to establish connection"
            )

            # Adapter wraps exceptions in ConnectionError
            with pytest.raises(ConnectionError) as exc_info:
                market_data_adapter.get_bars("EUR_USD", "H1", limit=10)
            assert "Failed to establish connection" in str(exc_info.value)

    def test_order_execution_rate_limit_exceeded(self, order_execution_adapter):
        """Test handling of rate limit exceeded (HTTP 429)."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = RequestsHTTPError("429 Too Many Requests")
            mock_session.return_value.post.return_value = mock_response

            order = Order(
                ts=1609459200000,
                symbol="EUR_USD",
                side=Side.BUY,
                quantity=Decimal("10000"),
                order_type=OrderType.MARKET,
            )
            result = order_execution_adapter.submit_order(order)

            assert result.success is False
            assert "429" in result.error_message or "API_ERROR" in result.error_code

    def test_order_execution_server_error(self, order_execution_adapter):
        """Test handling of server error (HTTP 500)."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = RequestsHTTPError(
                "500 Internal Server Error"
            )
            mock_session.return_value.post.return_value = mock_response

            order = Order(
                ts=1609459200000,
                symbol="EUR_USD",
                side=Side.BUY,
                quantity=Decimal("10000"),
                order_type=OrderType.MARKET,
            )
            result = order_execution_adapter.submit_order(order)

            assert result.success is False

    def test_order_execution_api_maintenance(self, order_execution_adapter):
        """Test handling of API maintenance (HTTP 503)."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_response.raise_for_status.side_effect = RequestsHTTPError(
                "503 Service Unavailable"
            )
            mock_session.return_value.post.return_value = mock_response

            order = Order(
                ts=1609459200000,
                symbol="EUR_USD",
                side=Side.BUY,
                quantity=Decimal("10000"),
                order_type=OrderType.MARKET,
            )
            result = order_execution_adapter.submit_order(order)

            assert result.success is False

    def test_market_data_invalid_json_response(self, market_data_adapter):
        """Test handling of invalid JSON response."""
        with patch.object(market_data_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_session.return_value.get.return_value = mock_response

            # Adapter catches ValueError and raises ConnectionError
            with pytest.raises(ConnectionError):
                market_data_adapter.get_bars("EUR_USD", "H1", limit=10)

    def test_order_execution_unauthorized(self, order_execution_adapter):
        """Test handling of unauthorized request (HTTP 401)."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = RequestsHTTPError("401 Unauthorized")
            mock_session.return_value.post.return_value = mock_response

            order = Order(
                ts=1609459200000,
                symbol="EUR_USD",
                side=Side.BUY,
                quantity=Decimal("10000"),
                order_type=OrderType.MARKET,
            )
            result = order_execution_adapter.submit_order(order)

            assert result.success is False

    def test_get_positions_with_api_error(self, order_execution_adapter):
        """Test get_positions handles API errors gracefully."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_session.return_value.get.side_effect = RequestException("Network error")

            positions = order_execution_adapter.get_positions()
            assert positions == {}

    def test_get_account_info_with_error(self, order_execution_adapter):
        """Test get_account_info handles errors gracefully."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_session.return_value.get.side_effect = RequestException("Network error")

            account_info = order_execution_adapter.get_account_info()
            # AccountInfo uses cash_balance, not balance
            assert account_info.cash_balance is None or account_info.cash_balance == Decimal("0")


# =========================
# Async Streaming Tests
# =========================


class TestOandaAsyncStreaming:
    """Tests for OANDA async streaming functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_async_acquire(self):
        """Test async rate limiter token acquisition."""
        limiter = RateLimiter(rate=120.0, burst=200)

        # Should not block with available tokens
        import asyncio

        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_rate_limiter_token_replenishment(self):
        """Test that tokens replenish over time."""
        limiter = RateLimiter(rate=120.0, burst=10)

        # Consume all tokens
        for _ in range(10):
            await limiter.acquire()

        # Wait for partial replenishment
        import asyncio

        await asyncio.sleep(0.1)  # Should add ~12 tokens (120 * 0.1)

        # Should be able to acquire again without long wait
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.05  # Should be quick after replenishment

    @pytest.mark.asyncio
    async def test_market_data_streaming_mock(self, market_data_adapter):
        """Test that streaming method signatures are correct."""
        # The adapter should have stream_ticks_async method
        assert hasattr(market_data_adapter, "stream_ticks_async")
        assert callable(market_data_adapter.stream_ticks_async)

        # Check that stream_bars_async also exists
        assert hasattr(market_data_adapter, "stream_bars_async")
        assert callable(market_data_adapter.stream_bars_async)

    def test_market_data_has_streaming_methods(self, market_data_adapter):
        """Test that market data adapter has streaming method signatures."""
        # Check that async streaming methods exist
        assert hasattr(market_data_adapter, "stream_ticks_async")
        assert hasattr(market_data_adapter, "stream_bars_async")
        assert hasattr(market_data_adapter, "stream_ticks")
        assert hasattr(market_data_adapter, "stream_bars")

    def test_rate_limiter_available_tokens_property(self):
        """Test rate limiter available_tokens property."""
        limiter = RateLimiter(rate=120.0, burst=200)

        initial_tokens = limiter.available_tokens
        assert initial_tokens >= 199.0  # Should be near burst limit

        limiter.acquire_sync()
        after_acquire = limiter.available_tokens

        assert after_acquire < initial_tokens


# =========================
# Last-Look Rejection Tests
# =========================


class TestOandaLastLookRejection:
    """Tests for OANDA last-look order rejection handling."""

    def test_parse_last_look_rejection(self, order_execution_adapter):
        """Test parsing of last-look rejection response."""
        response = {
            "orderCancelTransaction": {
                "orderID": "999",
                "reason": "MARKET_ORDER_FILL_TIMEOUT",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client999")

        assert result.success is False
        assert result.error_code == "LAST_LOOK_REJECTED"
        assert (
            "MARKET_ORDER_FILL_TIMEOUT" in result.error_message
            or "cancelled" in result.error_message.lower()
        )

    def test_parse_insufficient_liquidity(self, order_execution_adapter):
        """Test parsing of insufficient liquidity rejection."""
        response = {
            "orderRejectTransaction": {
                "rejectReason": "INSUFFICIENT_LIQUIDITY",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client_liq")

        assert result.success is False
        assert result.error_code == "INSUFFICIENT_LIQUIDITY"

    def test_parse_market_halted(self, order_execution_adapter):
        """Test parsing of market halted rejection."""
        response = {
            "orderRejectTransaction": {
                "rejectReason": "MARKET_HALTED",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client_halt")

        assert result.success is False
        assert result.error_code == "MARKET_HALTED"

    def test_parse_price_bound_violation(self, order_execution_adapter):
        """Test parsing of price bound violation."""
        response = {
            "orderRejectTransaction": {
                "rejectReason": "BOUNDS_VIOLATION",
            }
        }
        result = order_execution_adapter._parse_order_response(response, "client_bounds")

        assert result.success is False
        assert result.error_code == "BOUNDS_VIOLATION"


# =========================
# Position Management Tests
# =========================


class TestOandaPositionManagement:
    """Tests for OANDA position management."""

    def test_get_positions_with_long_position(self, order_execution_adapter):
        """Test parsing positions with long position."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "positions": [
                    {
                        "instrument": "EUR_USD",
                        "long": {
                            "units": "100000",
                            "averagePrice": "1.10500",
                            "unrealizedPL": "250.00",
                        },
                        "short": {
                            "units": "0",
                            "averagePrice": "0.0",
                            "unrealizedPL": "0.0",
                        },
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            positions = order_execution_adapter.get_positions()

            assert "EUR_USD" in positions
            pos = positions["EUR_USD"]
            # Position qty is positive for long
            assert pos.qty == Decimal("100000")
            assert pos.qty > 0  # Long position
            assert pos.avg_entry_price == Decimal("1.10500")
            assert pos.meta.get("side") == "BUY"

    def test_get_positions_with_short_position(self, order_execution_adapter):
        """Test parsing positions with short position."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "positions": [
                    {
                        "instrument": "GBP_USD",
                        "long": {
                            "units": "0",
                            "averagePrice": "0.0",
                            "unrealizedPL": "0.0",
                        },
                        "short": {
                            "units": "-50000",
                            "averagePrice": "1.25000",
                            "unrealizedPL": "-125.00",
                        },
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            positions = order_execution_adapter.get_positions()

            assert "GBP_USD" in positions
            pos = positions["GBP_USD"]
            # Position qty is negative for short
            assert pos.qty == Decimal("-50000")
            assert pos.qty < 0  # Short position
            assert pos.meta.get("side") == "SELL"

    def test_get_positions_with_hedged_position(self, order_execution_adapter):
        """Test parsing hedged positions (net zero)."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "positions": [
                    {
                        "instrument": "USD_JPY",
                        "long": {
                            "units": "50000",
                            "averagePrice": "140.00",
                            "unrealizedPL": "100.00",
                        },
                        "short": {
                            "units": "-50000",
                            "averagePrice": "140.50",
                            "unrealizedPL": "-50.00",
                        },
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            positions = order_execution_adapter.get_positions()

            # Net position is zero, should not appear in results
            assert "USD_JPY" not in positions

    def test_close_position_success(self, order_execution_adapter):
        """Test successful position close."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "longOrderFillTransaction": {
                    "orderID": "12345",
                    "units": "-100000",
                    "price": "1.10600",
                }
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.put.return_value = mock_response

            result = order_execution_adapter.close_position("EUR_USD")

            assert result.success is True
            assert result.status == "FILLED"

    def test_close_position_no_position(self, order_execution_adapter):
        """Test closing non-existent position."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status = Mock()
            mock_session.return_value.put.return_value = mock_response

            result = order_execution_adapter.close_position("EUR_USD")

            assert result.success is False
            assert result.error_code == "NO_FILL"


# =========================
# Account Info Tests
# =========================


class TestOandaAccountInfo:
    """Tests for OANDA account information."""

    def test_get_account_info_success(self, order_execution_adapter):
        """Test successful account info retrieval."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "account": {
                    "id": "101-001-12345-001",
                    "currency": "USD",
                    "balance": "100000.00",
                    "marginAvailable": "95000.00",
                    "marginUsed": "5000.00",
                    "unrealizedPL": "1500.00",
                }
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            account = order_execution_adapter.get_account_info()

            assert account.account_id == "101-001-12345-001"
            assert account.cash_balance == Decimal("100000.00")
            assert account.buying_power == Decimal("95000.00")
            assert account.margin_enabled is True
            assert account.raw_data.get("currency") == "USD"
            assert account.raw_data.get("margin_used") == "5000.00"

    def test_get_account_info_with_leverage(self, order_execution_adapter):
        """Test account info with high margin usage."""
        with patch.object(order_execution_adapter, "_ensure_session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "account": {
                    "id": "test_account",
                    "currency": "EUR",
                    "balance": "50000.00",
                    "marginAvailable": "10000.00",
                    "marginUsed": "40000.00",
                    "unrealizedPL": "-500.00",
                }
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.get.return_value = mock_response

            account = order_execution_adapter.get_account_info()

            # margin_used and unrealized_pnl are in raw_data now
            assert account.raw_data.get("margin_used") == "40000.00"
            assert account.raw_data.get("unrealized_pnl") == "-500.00"
            assert account.buying_power == Decimal("10000.00")
            assert account.cash_balance == Decimal("50000.00")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
