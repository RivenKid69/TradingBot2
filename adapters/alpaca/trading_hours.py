# -*- coding: utf-8 -*-
"""
adapters/alpaca/trading_hours.py
Alpaca trading hours adapter for US equity markets.

Handles NYSE/NASDAQ trading schedule:
- Pre-market: 4:00 AM - 9:30 AM ET
- Regular: 9:30 AM - 4:00 PM ET
- After-hours: 4:00 PM - 8:00 PM ET

Also handles market holidays and half-days.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Any, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from adapters.base import TradingHoursAdapter
from adapters.models import (
    ExchangeVendor,
    MarketCalendar,
    MarketType,
    SessionType,
    TradingSession,
    US_EQUITY_SESSIONS,
    create_us_equity_calendar,
)

logger = logging.getLogger(__name__)

# US Eastern timezone
ET = ZoneInfo("America/New_York")


class AlpacaTradingHoursAdapter(TradingHoursAdapter):
    """
    Alpaca trading hours adapter for US equity markets.

    Implements NYSE/NASDAQ trading schedule with proper handling of:
    - Regular trading hours (9:30 AM - 4:00 PM ET)
    - Extended hours (pre-market and after-hours)
    - Market holidays
    - Half-days (early close at 1:00 PM ET)

    Configuration:
        allow_extended_hours: Whether to consider extended hours as "open" (default: True)
        use_alpaca_calendar: Use Alpaca API for calendar (default: True if connected)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        self._allow_extended = self._config.get("allow_extended_hours", True)
        self._calendar = create_us_equity_calendar(vendor)

        # Cache for Alpaca calendar data
        self._alpaca_calendar: Optional[List[dict]] = None
        self._holidays_cache: set[date] = set()
        self._half_days_cache: set[date] = set()

        # Try to load calendar from Alpaca if credentials available
        if self._config.get("use_alpaca_calendar", True):
            self._try_load_alpaca_calendar()

    def _try_load_alpaca_calendar(self) -> None:
        """Try to load trading calendar from Alpaca API."""
        api_key = self._config.get("api_key")
        api_secret = self._config.get("api_secret")

        if not api_key or not api_secret:
            logger.debug("Alpaca credentials not provided, using default calendar")
            return

        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key=api_key, secret_key=api_secret)

            # Get calendar for next year
            today = date.today()
            end_date = today + timedelta(days=365)

            calendar = client.get_calendar(
                start=today.isoformat(),
                end=end_date.isoformat(),
            )

            self._alpaca_calendar = []
            for day in calendar:
                day_date = day.date
                # Standard close time is 16:00 ET
                # Half days close at 13:00 ET
                if day.close.hour < 16:
                    self._half_days_cache.add(day_date)

                self._alpaca_calendar.append(
                    {
                        "date": day_date,
                        "open": day.open,
                        "close": day.close,
                    }
                )

            logger.info(f"Loaded Alpaca calendar: {len(self._alpaca_calendar)} trading days")

        except Exception as e:
            logger.warning(f"Failed to load Alpaca calendar: {e}")

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """
        Check if market is open at given timestamp.

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Specific session to check:
                - None: Any tradable session
                - REGULAR: Only regular hours (9:30-16:00)
                - PRE_MARKET: Pre-market (4:00-9:30)
                - AFTER_HOURS: After hours (16:00-20:00)
                - EXTENDED: Any extended hours

        Returns:
            True if market is open
        """
        dt = datetime.fromtimestamp(ts / 1000, tz=ET)

        # Check if it's a trading day
        if not self._is_trading_day(dt.date()):
            return False

        # Get current time in minutes from midnight
        current_minutes = dt.hour * 60 + dt.minute

        # Check specific session type
        if session_type == SessionType.REGULAR:
            return self._in_regular_hours(current_minutes, dt.date())

        elif session_type == SessionType.PRE_MARKET:
            return self._in_pre_market(current_minutes)

        elif session_type == SessionType.AFTER_HOURS:
            return self._in_after_hours(current_minutes, dt.date())

        elif session_type == SessionType.EXTENDED:
            return self._in_pre_market(current_minutes) or self._in_after_hours(
                current_minutes, dt.date()
            )

        # None = any tradable session
        if self._allow_extended:
            return (
                self._in_pre_market(current_minutes)
                or self._in_regular_hours(current_minutes, dt.date())
                or self._in_after_hours(current_minutes, dt.date())
            )
        else:
            return self._in_regular_hours(current_minutes, dt.date())

    def _is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day."""
        # Weekend check
        if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Holiday check
        if d in self._holidays_cache:
            return False

        # Check Alpaca calendar if available
        if self._alpaca_calendar:
            return any(day["date"] == d for day in self._alpaca_calendar)

        return True

    def _in_pre_market(self, minutes: int) -> bool:
        """Check if time is in pre-market hours."""
        # 4:00 AM - 9:30 AM ET = 240 - 570 minutes
        return 240 <= minutes < 570

    def _in_regular_hours(self, minutes: int, d: date) -> bool:
        """Check if time is in regular trading hours."""
        # 9:30 AM - 4:00 PM ET = 570 - 960 minutes
        # Half days close at 1:00 PM = 780 minutes
        close_minutes = 780 if d in self._half_days_cache else 960
        return 570 <= minutes < close_minutes

    def _in_after_hours(self, minutes: int, d: date) -> bool:
        """Check if time is in after-hours."""
        # 4:00 PM - 8:00 PM ET = 960 - 1200 minutes
        # No after-hours on half days
        if d in self._half_days_cache:
            return False
        return 960 <= minutes < 1200

    def next_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market open.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Session type to wait for

        Returns:
            Timestamp of next open in milliseconds
        """
        dt = datetime.fromtimestamp(ts / 1000, tz=ET)

        # If already open, return current ts
        if self.is_market_open(ts, session_type=session_type):
            return ts

        # Find next trading day
        target_date = dt.date()
        for _ in range(10):  # Max 10 days forward (holidays)
            if not self._is_trading_day(target_date) or target_date == dt.date():
                target_date += timedelta(days=1)
                continue
            break

        # Determine open time based on session type
        if session_type == SessionType.PRE_MARKET:
            open_time = datetime(
                target_date.year, target_date.month, target_date.day, 4, 0, 0, tzinfo=ET
            )
        elif session_type == SessionType.REGULAR:
            open_time = datetime(
                target_date.year, target_date.month, target_date.day, 9, 30, 0, tzinfo=ET
            )
        elif session_type in (SessionType.AFTER_HOURS, SessionType.EXTENDED):
            # After hours starts after regular close
            open_time = datetime(
                target_date.year, target_date.month, target_date.day, 16, 0, 0, tzinfo=ET
            )
        else:
            # Default: pre-market if extended allowed, else regular
            if self._allow_extended:
                open_time = datetime(
                    target_date.year, target_date.month, target_date.day, 4, 0, 0, tzinfo=ET
                )
            else:
                open_time = datetime(
                    target_date.year, target_date.month, target_date.day, 9, 30, 0, tzinfo=ET
                )

        return int(open_time.timestamp() * 1000)

    def next_close(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market close.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Session type

        Returns:
            Timestamp of next close in milliseconds
        """
        dt = datetime.fromtimestamp(ts / 1000, tz=ET)
        target_date = dt.date()

        # If not a trading day, find next trading day
        if not self._is_trading_day(target_date):
            for _ in range(10):
                target_date += timedelta(days=1)
                if self._is_trading_day(target_date):
                    break

        # Determine close time
        if session_type == SessionType.PRE_MARKET:
            close_time = datetime(
                target_date.year, target_date.month, target_date.day, 9, 30, 0, tzinfo=ET
            )
        elif session_type == SessionType.REGULAR:
            close_hour = 13 if target_date in self._half_days_cache else 16
            close_time = datetime(
                target_date.year, target_date.month, target_date.day, close_hour, 0, 0, tzinfo=ET
            )
        elif session_type in (SessionType.AFTER_HOURS, SessionType.EXTENDED):
            close_time = datetime(
                target_date.year, target_date.month, target_date.day, 20, 0, 0, tzinfo=ET
            )
        else:
            # Default: end of after-hours if extended allowed, else regular close
            if self._allow_extended:
                close_time = datetime(
                    target_date.year, target_date.month, target_date.day, 20, 0, 0, tzinfo=ET
                )
            else:
                close_hour = 13 if target_date in self._half_days_cache else 16
                close_time = datetime(
                    target_date.year,
                    target_date.month,
                    target_date.day,
                    close_hour,
                    0,
                    0,
                    tzinfo=ET,
                )

        return int(close_time.timestamp() * 1000)

    def get_calendar(self) -> MarketCalendar:
        """Get market calendar."""
        # Update holidays from cache
        holidays = [(d.year, d.month, d.day) for d in self._holidays_cache]
        half_days = [(d.year, d.month, d.day) for d in self._half_days_cache]

        return MarketCalendar(
            vendor=self._vendor,
            market_type=MarketType.EQUITY,
            sessions=list(US_EQUITY_SESSIONS),
            holidays=holidays,
            half_days=half_days,
            timezone="America/New_York",
        )

    def is_holiday(self, ts: int) -> bool:
        """Check if date is a market holiday."""
        dt = datetime.fromtimestamp(ts / 1000, tz=ET)
        d = dt.date()

        # Check cache
        if d in self._holidays_cache:
            return True

        # Check Alpaca calendar if available
        if self._alpaca_calendar:
            return not any(day["date"] == d for day in self._alpaca_calendar)

        # Fallback: just check weekends
        return d.weekday() >= 5

    def is_half_day(self, ts: int) -> bool:
        """Check if date is a half-day (early close)."""
        dt = datetime.fromtimestamp(ts / 1000, tz=ET)
        return dt.date() in self._half_days_cache

    def get_market_hours(
        self,
        d: Optional[date] = None,
    ) -> dict:
        """
        Get market hours for a specific date.

        Args:
            d: Date to check (default: today)

        Returns:
            Dict with session times
        """
        if d is None:
            d = date.today()

        is_half = d in self._half_days_cache
        regular_close = "13:00" if is_half else "16:00"
        after_close = None if is_half else "20:00"

        return {
            "date": d.isoformat(),
            "is_trading_day": self._is_trading_day(d),
            "is_half_day": is_half,
            "pre_market": {"open": "04:00", "close": "09:30"},
            "regular": {"open": "09:30", "close": regular_close},
            "after_hours": {"open": "16:00", "close": after_close} if after_close else None,
            "timezone": "America/New_York",
        }
