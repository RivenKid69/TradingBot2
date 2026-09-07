# -*- coding: utf-8 -*-
"""
tests/test_stock_trading_hours.py
Tests for trading hours enforcement in stock trading (Phase 4.6).

Test coverage:
- Alpaca trading hours adapter
- Polygon trading hours adapter
- Binance 24/7 trading hours
- Regular vs extended hours
- US market holidays
- Half-days (early close)
"""

import pytest
from datetime import datetime, date, timedelta
from typing import Optional
from unittest.mock import Mock, patch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def alpaca_adapter():
    """Create Alpaca trading hours adapter without API credentials."""
    from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter

    return AlpacaTradingHoursAdapter(config={"use_alpaca_calendar": False})


@pytest.fixture
def polygon_adapter():
    """Create Polygon trading hours adapter without API credentials."""
    from adapters.polygon.trading_hours import PolygonTradingHoursAdapter

    return PolygonTradingHoursAdapter(config={"api_key": ""})


@pytest.fixture
def binance_adapter():
    """Create Binance trading hours adapter."""
    from adapters.binance.trading_hours import BinanceTradingHoursAdapter

    return BinanceTradingHoursAdapter()


def ts_from_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    """Create timestamp from ET time components."""
    try:
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        dt = datetime(year, month, day, hour, minute, tzinfo=et)
        return int(dt.timestamp() * 1000)
    except ImportError:
        # Fallback: approximate offset
        dt = datetime(year, month, day, hour, minute)
        return int((dt.timestamp() + 5 * 3600) * 1000)


# =============================================================================
# Test Alpaca Trading Hours
# =============================================================================


class TestAlpacaTradingHours:
    """Tests for Alpaca trading hours adapter."""

    def test_regular_hours_open(self, alpaca_adapter):
        """Test market is open during regular hours (9:30-16:00 ET)."""
        # Tuesday at 10:00 AM ET (definitely open)
        ts = ts_from_et(2024, 11, 26, 10, 0)
        assert alpaca_adapter.is_market_open(ts) is True

    def test_regular_hours_close(self, alpaca_adapter):
        """Test market is closed outside regular hours."""
        # Tuesday at 8:00 AM ET (before pre-market)
        ts = ts_from_et(2024, 11, 26, 3, 0)
        assert alpaca_adapter.is_market_open(ts) is False

    def test_pre_market_hours(self, alpaca_adapter):
        """Test pre-market hours (4:00-9:30 AM ET)."""
        from adapters.models import SessionType

        # Tuesday at 5:00 AM ET (pre-market)
        ts = ts_from_et(2024, 11, 26, 5, 0)

        # Regular session should be closed
        assert alpaca_adapter.is_market_open(ts, session_type=SessionType.REGULAR) is False

        # Pre-market should be open
        assert alpaca_adapter.is_market_open(ts, session_type=SessionType.PRE_MARKET) is True

    def test_after_hours(self, alpaca_adapter):
        """Test after-hours trading (4:00-8:00 PM ET)."""
        from adapters.models import SessionType

        # Tuesday at 6:00 PM ET (after hours)
        ts = ts_from_et(2024, 11, 26, 18, 0)

        # Regular session should be closed
        assert alpaca_adapter.is_market_open(ts, session_type=SessionType.REGULAR) is False

        # After hours should be open
        assert alpaca_adapter.is_market_open(ts, session_type=SessionType.AFTER_HOURS) is True

    def test_weekend_closed(self, alpaca_adapter):
        """Test market is closed on weekends."""
        # Saturday at 10:00 AM ET
        ts = ts_from_et(2024, 11, 23, 10, 0)
        assert alpaca_adapter.is_market_open(ts) is False

        # Sunday at 10:00 AM ET
        ts = ts_from_et(2024, 11, 24, 10, 0)
        assert alpaca_adapter.is_market_open(ts) is False

    def test_extended_hours_disabled(self):
        """Test extended hours can be disabled."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter

        adapter = AlpacaTradingHoursAdapter(
            config={
                "use_alpaca_calendar": False,
                "allow_extended_hours": False,
            }
        )

        # Tuesday at 5:00 AM ET (pre-market)
        ts = ts_from_et(2024, 11, 26, 5, 0)

        # Should be closed when extended hours disabled
        assert adapter.is_market_open(ts) is False

        # Same time during regular hours should be open
        ts_regular = ts_from_et(2024, 11, 26, 10, 0)
        assert adapter.is_market_open(ts_regular) is True

    def test_next_open_from_closed(self, alpaca_adapter):
        """Test getting next open time from closed state."""
        # Sunday at 10:00 AM ET (closed)
        ts = ts_from_et(2024, 11, 24, 10, 0)

        next_open = alpaca_adapter.next_open(ts)

        # Should be Monday pre-market (4:00 AM) or regular (9:30 AM)
        assert next_open > ts

    def test_next_close_from_open(self, alpaca_adapter):
        """Test getting next close time from open state."""
        # Tuesday at 10:00 AM ET (open)
        ts = ts_from_et(2024, 11, 26, 10, 0)

        next_close = alpaca_adapter.next_close(ts)

        # Should be same day at 4:00 PM ET (regular) or 8:00 PM ET (after hours)
        assert next_close > ts

    def test_get_calendar(self, alpaca_adapter):
        """Test getting market calendar."""
        calendar = alpaca_adapter.get_calendar()

        assert calendar is not None
        assert calendar.timezone == "America/New_York"
        assert len(calendar.sessions) > 0

    def test_get_market_hours(self, alpaca_adapter):
        """Test getting market hours for a specific date."""
        hours = alpaca_adapter.get_market_hours(date(2024, 11, 26))

        assert hours["is_trading_day"] is True
        assert hours["regular"]["open"] == "09:30"
        assert hours["regular"]["close"] == "16:00"
        assert hours["timezone"] == "America/New_York"


# =============================================================================
# Test Polygon Trading Hours
# =============================================================================


class TestPolygonTradingHours:
    """Tests for Polygon trading hours adapter."""

    def test_regular_hours_open(self, polygon_adapter):
        """Test market is open during regular hours."""
        # Tuesday at 10:00 AM ET
        ts = ts_from_et(2024, 11, 26, 10, 0)
        assert polygon_adapter.is_market_open(ts) is True

    def test_regular_hours_closed_night(self, polygon_adapter):
        """Test market is closed at night."""
        # Tuesday at 2:00 AM ET
        ts = ts_from_et(2024, 11, 26, 2, 0)
        assert polygon_adapter.is_market_open(ts) is False

    def test_holiday_closed(self, polygon_adapter):
        """Test market is closed on holidays."""
        # Christmas Day 2024
        ts = ts_from_et(2024, 12, 25, 10, 0)
        assert polygon_adapter.is_holiday(ts) is True
        assert polygon_adapter.is_market_open(ts) is False

        # Thanksgiving 2024
        ts = ts_from_et(2024, 11, 28, 10, 0)
        assert polygon_adapter.is_holiday(ts) is True

    def test_get_calendar_has_holidays(self, polygon_adapter):
        """Test calendar includes holidays."""
        calendar = polygon_adapter.get_calendar()

        # Check for known holidays
        assert len(calendar.holidays) > 0

        # Check Christmas 2024 is in holidays
        christmas_2024 = (2024, 12, 25)
        assert christmas_2024 in calendar.holidays

    def test_get_calendar_has_half_days(self, polygon_adapter):
        """Test calendar includes half-days."""
        calendar = polygon_adapter.get_calendar()

        # Check for known half days
        assert len(calendar.half_days) > 0

        # Christmas Eve 2024 is typically a half day
        xmas_eve_2024 = (2024, 12, 24)
        assert xmas_eve_2024 in calendar.half_days


# =============================================================================
# Test Binance 24/7 Trading Hours
# =============================================================================


class TestBinanceTradingHours:
    """Tests for Binance (crypto) trading hours adapter."""

    def test_always_open(self, binance_adapter):
        """Test crypto market is always open."""
        # Test various times
        test_times = [
            ts_from_et(2024, 11, 26, 10, 0),  # Tuesday morning
            ts_from_et(2024, 11, 23, 3, 0),  # Saturday night
            ts_from_et(2024, 12, 25, 12, 0),  # Christmas Day
            ts_from_et(2024, 1, 1, 0, 0),  # New Year's Day
        ]

        for ts in test_times:
            assert binance_adapter.is_market_open(ts) is True

    def test_no_holidays(self, binance_adapter):
        """Test crypto has no holidays."""
        # Check Christmas
        ts = ts_from_et(2024, 12, 25, 10, 0)
        assert binance_adapter.is_holiday(ts) is False

        # Check New Year
        ts = ts_from_et(2024, 1, 1, 10, 0)
        assert binance_adapter.is_holiday(ts) is False

    def test_next_open_returns_current(self, binance_adapter):
        """Test next_open returns current time (already open)."""
        ts = ts_from_et(2024, 11, 26, 10, 0)
        assert binance_adapter.next_open(ts) == ts

    def test_next_close_far_future(self, binance_adapter):
        """Test next_close returns far future (never closes)."""
        ts = ts_from_et(2024, 11, 26, 10, 0)
        next_close = binance_adapter.next_close(ts)

        # Should be at least 1 year in the future
        one_year_ms = 365 * 24 * 60 * 60 * 1000
        assert next_close > ts + one_year_ms

    def test_time_to_close_none(self, binance_adapter):
        """Test time_to_close returns None (never closes)."""
        ts = ts_from_et(2024, 11, 26, 10, 0)
        assert binance_adapter.time_to_close(ts) is None

    def test_supports_extended_hours(self, binance_adapter):
        """Test crypto supports extended hours (24/7)."""
        assert binance_adapter.supports_extended_hours() is True

    def test_get_calendar(self, binance_adapter):
        """Test getting crypto calendar."""
        calendar = binance_adapter.get_calendar()

        assert calendar is not None
        assert len(calendar.holidays) == 0  # No holidays
        assert len(calendar.sessions) > 0


# =============================================================================
# Test Asset Class Comparison
# =============================================================================


class TestAssetClassComparison:
    """Compare trading hours between asset classes."""

    def test_equity_vs_crypto_weekend(self, alpaca_adapter, binance_adapter):
        """Test equity is closed on weekends while crypto is open."""
        # Saturday at noon
        ts = ts_from_et(2024, 11, 23, 12, 0)

        # Equity should be closed
        assert alpaca_adapter.is_market_open(ts) is False

        # Crypto should be open
        assert binance_adapter.is_market_open(ts) is True

    def test_equity_vs_crypto_holiday(self, alpaca_adapter, binance_adapter):
        """Test equity is closed on holidays while crypto is open."""
        from datetime import date

        # Add Christmas to the adapter's holiday cache (since we don't have API credentials)
        christmas = date(2024, 12, 25)
        alpaca_adapter._holidays_cache.add(christmas)

        # Christmas at noon
        ts = ts_from_et(2024, 12, 25, 12, 0)

        # Equity should be closed (holiday)
        assert alpaca_adapter.is_market_open(ts) is False

        # Crypto should be open
        assert binance_adapter.is_market_open(ts) is True

    def test_equity_regular_hours_only(self):
        """Test equity with only regular hours (no extended)."""
        from adapters.alpaca.trading_hours import AlpacaTradingHoursAdapter
        from adapters.models import SessionType

        # Equity adapter without extended hours
        adapter = AlpacaTradingHoursAdapter(
            config={
                "use_alpaca_calendar": False,
                "allow_extended_hours": False,
            }
        )

        # Pre-market time (5 AM)
        ts = ts_from_et(2024, 11, 26, 5, 0)

        # Should be closed when extended hours disabled
        assert adapter.is_market_open(ts) is False

        # But pre-market session specifically should still detect it
        assert adapter.is_market_open(ts, session_type=SessionType.PRE_MARKET) is True


# =============================================================================
# Test Trading Hours Enforcement in Config
# =============================================================================


class TestTradingHoursConfig:
    """Test trading hours configuration in configs."""

    def test_crypto_session_calendar(self):
        """Test crypto uses 24/7 calendar."""
        import yaml

        config_path = "configs/config_train.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert config.get("asset_class") == "crypto"
            assert config.get("env", {}).get("session", {}).get("calendar") == "crypto_24x7"
        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_equity_session_calendar(self):
        """Test equity uses US market calendar."""
        import yaml

        config_path = "configs/config_train_stocks.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert config.get("asset_class") == "equity"
            assert config.get("env", {}).get("session", {}).get("calendar") == "us_equity"
        except FileNotFoundError:
            pytest.skip("Config file not found")


# =============================================================================
# Test Session Type Enum
# =============================================================================


class TestSessionType:
    """Test SessionType enum values."""

    def test_session_types_exist(self):
        """Test all expected session types exist."""
        from adapters.models import SessionType

        assert hasattr(SessionType, "REGULAR")
        assert hasattr(SessionType, "PRE_MARKET")
        assert hasattr(SessionType, "AFTER_HOURS")
        assert hasattr(SessionType, "EXTENDED")

    def test_session_type_values(self):
        """Test session type values are strings."""
        from adapters.models import SessionType

        assert isinstance(SessionType.REGULAR.value, str)
        assert isinstance(SessionType.PRE_MARKET.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
