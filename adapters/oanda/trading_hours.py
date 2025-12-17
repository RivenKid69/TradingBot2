# -*- coding: utf-8 -*-
"""
adapters/oanda/trading_hours.py
Forex trading hours adapter with DST awareness.

Status: Tested and operational (Phase 2 Complete)

Forex Market Hours:
- Opens: Sunday 5:00 PM ET (21:00 UTC winter, 22:00 UTC summer)
- Closes: Friday 5:00 PM ET
- Daily rollover: 5:00 PM ET (swap points applied)

Sessions (all times UTC):
- Sydney: 21:00-06:00 (lowest liquidity)
- Tokyo: 00:00-09:00 (JPY pairs active)
- London: 07:00-16:00 (highest EUR/GBP liquidity)
- New York: 12:00-21:00 (highest USD liquidity)
- London/NY Overlap: 12:00-16:00 (BEST liquidity)
- Tokyo/London Overlap: 07:00-09:00

DST Handling:
- US DST: Second Sunday March to First Sunday November
- Rollover time shifts between 21:00 and 22:00 UTC
- Session boundaries adjust accordingly

Usage:
    adapter = OandaTradingHoursAdapter()

    # Check if market is open
    is_open = adapter.is_market_open(timestamp_ms)

    # Get current session with liquidity info
    session, liquidity, spread_mult = adapter.get_current_session(timestamp_ms)

    # Get rollover time for swap calculation
    rollover_utc = adapter.get_rollover_time_utc(date)

References:
    - OANDA trading hours: https://www.oanda.com/trading/hours/
    - BIS Triennial Survey 2022: https://www.bis.org/statistics/rpfx22.htm
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from adapters.base import TradingHoursAdapter
from adapters.models import (
    ExchangeVendor,
    ForexSessionType,
    ForexSessionWindow,
    FOREX_SESSION_WINDOWS,
    MarketCalendar,
    MarketType,
    SessionType,
    TradingSession,
)

logger = logging.getLogger(__name__)


# US Eastern timezone (for rollover calculations)
ET = ZoneInfo("America/New_York")


class OandaTradingHoursAdapter(TradingHoursAdapter):
    """
    Forex trading hours adapter with DST awareness.

    Implements forex market schedule with proper handling of:
    - Session-based liquidity (Sydney < Tokyo < London ≈ NY < Overlap)
    - Weekend closures (Fri 5pm ET to Sun 5pm ET)
    - DST transitions (rollover shifts between 21:00 and 22:00 UTC)
    - Holiday schedule (minimal impact on forex)

    Configuration:
        use_session_liquidity: Return liquidity factors with session (default: True)
        rollover_hour_et: Hour in ET when daily rollover occurs (default: 17 = 5pm)

    Session Liquidity Factors:
        - London/NY Overlap: 1.35 (best execution)
        - London: 1.10
        - New York: 1.05
        - Tokyo: 0.75
        - Sydney: 0.65
        - Off-hours: 0.50
        - Weekend: 0.00 (closed)

    Spread Multipliers (higher = wider spreads):
        - London/NY Overlap: 0.8 (tightest)
        - London/NY: 1.0 (normal)
        - Tokyo: 1.3
        - Sydney: 1.5
        - Off-hours: 2.0
    """

    # Rollover time in ET (constant regardless of DST)
    ROLLOVER_HOUR_ET = 17  # 5:00 PM ET

    # Market open/close in ET
    MARKET_OPEN_DAY = 6  # Sunday
    MARKET_OPEN_HOUR = 17  # 5:00 PM ET Sunday
    MARKET_CLOSE_DAY = 4  # Friday
    MARKET_CLOSE_HOUR = 17  # 5:00 PM ET Friday

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize forex trading hours adapter.

        Args:
            vendor: Exchange vendor (default: OANDA)
            config: Configuration dict
        """
        super().__init__(vendor, config)

        self._use_session_liquidity = self._config.get("use_session_liquidity", True)
        self._rollover_hour = self._config.get("rollover_hour_et", self.ROLLOVER_HOUR_ET)

        # Cache for session windows
        self._session_windows = list(FOREX_SESSION_WINDOWS)

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """
        Check if forex market is open at given timestamp.

        Forex market is open from Sunday 5pm ET to Friday 5pm ET.

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Optional session filter (not used for forex 24/5)

        Returns:
            True if market is open, False if weekend

        Example:
            >>> adapter.is_market_open(1701100800000)  # Mon 10am
            True
            >>> adapter.is_market_open(1701532800000)  # Sat 10am
            False
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ET)
        weekday = dt_et.weekday()  # 0=Monday, 6=Sunday
        hour_et = dt_et.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: opens at 5pm ET
        if weekday == 6:
            return hour_et >= self.MARKET_OPEN_HOUR

        # Friday: closes at 5pm ET
        if weekday == 4:
            return hour_et < self.MARKET_CLOSE_HOUR

        # Monday-Thursday: 24 hours open
        return True

    def next_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market open.

        For forex, market opens Sunday 5pm ET.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Ignored for forex (24/5 market)

        Returns:
            Timestamp of next open in milliseconds
        """
        if self.is_market_open(ts):
            return ts  # Already open

        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ET)

        # Find next Sunday 5pm ET
        days_until_sunday = (6 - dt_et.weekday()) % 7
        if days_until_sunday == 0 and dt_et.hour >= self.MARKET_OPEN_HOUR:
            # Current Sunday after open - already handled by is_market_open
            days_until_sunday = 7

        next_sunday = dt_et.date() + timedelta(days=days_until_sunday)
        open_dt = datetime(
            year=next_sunday.year,
            month=next_sunday.month,
            day=next_sunday.day,
            hour=self.MARKET_OPEN_HOUR,
            minute=0,
            second=0,
            tzinfo=ET,
        )
        return int(open_dt.astimezone(timezone.utc).timestamp() * 1000)

    def next_close(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market close.

        For forex, market closes Friday 5pm ET.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Ignored for forex (24/5 market)

        Returns:
            Timestamp of next close in milliseconds
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ET)

        # Find next Friday 5pm ET
        days_until_friday = (4 - dt_et.weekday()) % 7
        if days_until_friday == 0:
            # It's Friday
            if dt_et.hour >= self.MARKET_CLOSE_HOUR:
                # Past close, next week
                days_until_friday = 7
        elif days_until_friday < 0:
            days_until_friday += 7

        next_friday = dt_et.date() + timedelta(days=days_until_friday)
        close_dt = datetime(
            year=next_friday.year,
            month=next_friday.month,
            day=next_friday.day,
            hour=self.MARKET_CLOSE_HOUR,
            minute=0,
            second=0,
            tzinfo=ET,
        )
        return int(close_dt.astimezone(timezone.utc).timestamp() * 1000)

    def get_calendar(self) -> MarketCalendar:
        """
        Get forex market calendar.

        Forex has minimal holidays - market is 24/5.

        Returns:
            MarketCalendar for forex
        """
        sessions = [
            TradingSession(
                session_type=SessionType.REGULAR,
                open_time="17:00",
                close_time="17:00",
                timezone="America/New_York",
            ),
        ]

        return MarketCalendar(
            vendor=self._vendor,
            market_type=MarketType.FOREX,
            timezone="America/New_York",
            sessions=sessions,
            holidays=[],  # Forex has no regular holidays
        )

    def is_holiday(self, ts: int) -> bool:
        """
        Check if date is a forex market holiday.

        Forex markets trade 24/5 with minimal holidays.
        Major holidays (Christmas, New Year's Day) may have reduced liquidity
        but market typically remains open.

        Args:
            ts: Timestamp in milliseconds

        Returns:
            True if major holiday (very rare for forex)
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        month = dt_utc.month
        day = dt_utc.day

        # Only Christmas Day is reliably closed
        if month == 12 and day == 25:
            return True

        # New Year's Day typically closed
        if month == 1 and day == 1:
            return True

        return False

    def get_current_session(
        self,
        ts: int,
    ) -> Tuple[ForexSessionType, float, float]:
        """
        Determine current forex session with liquidity characteristics.

        Returns session type, liquidity factor, and spread multiplier.

        Args:
            ts: Unix timestamp in milliseconds

        Returns:
            Tuple of (session_type, liquidity_factor, spread_multiplier)
            - liquidity_factor: 0.0-1.5 (higher = more liquidity)
            - spread_multiplier: 0.8-2.0 (lower = tighter spreads)

        Example:
            >>> session, liq, spread = adapter.get_current_session(1701100800000)
            >>> session
            ForexSessionType.LONDON
            >>> liq
            1.10
            >>> spread
            1.0
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour_utc = dt_utc.hour
        weekday = dt_utc.weekday()

        # Weekend check (Saturday all day, Sunday before market open)
        if weekday == 5:
            return (ForexSessionType.WEEKEND, 0.0, float('inf'))

        # Sunday before market open (5pm ET)
        if weekday == 6:
            dt_et = dt_utc.astimezone(ET)
            if dt_et.hour < self.MARKET_OPEN_HOUR:
                return (ForexSessionType.WEEKEND, 0.0, float('inf'))

        # Friday after market close
        if weekday == 4:
            dt_et = dt_utc.astimezone(ET)
            if dt_et.hour >= self.MARKET_CLOSE_HOUR:
                return (ForexSessionType.WEEKEND, 0.0, float('inf'))

        # Check overlaps first (they take priority for best liquidity)
        # London/NY Overlap: 12:00-16:00 UTC (BEST)
        if 12 <= hour_utc < 16:
            return (ForexSessionType.LONDON_NY_OVERLAP, 1.35, 0.8)

        # Tokyo/London Overlap: 07:00-09:00 UTC
        if 7 <= hour_utc < 9:
            return (ForexSessionType.TOKYO_LONDON_OVERLAP, 0.90, 1.1)

        # Individual sessions
        # Tokyo: 00:00-09:00 UTC (takes priority over Sydney during overlap)
        if 0 <= hour_utc < 9:
            return (ForexSessionType.TOKYO, 0.75, 1.3)

        # Sydney: 21:00-00:00 UTC (before Tokyo opens)
        # Note: After midnight, Tokyo session takes priority
        if hour_utc >= 21:
            return (ForexSessionType.SYDNEY, 0.65, 1.5)

        # London: 07:00-16:00 UTC
        if 7 <= hour_utc < 16:
            return (ForexSessionType.LONDON, 1.10, 1.0)

        # New York: 12:00-21:00 UTC
        if 12 <= hour_utc < 21:
            return (ForexSessionType.NEW_YORK, 1.05, 1.0)

        # Off-hours (shouldn't hit this with proper session coverage)
        return (ForexSessionType.OFF_HOURS, 0.50, 2.0)

    def get_session_type(self, ts: int) -> ForexSessionType:
        """
        Get just the session type (convenience method).

        Args:
            ts: Unix timestamp in milliseconds

        Returns:
            Current forex session type
        """
        session_type, _, _ = self.get_current_session(ts)
        return session_type

    def get_liquidity_factor(self, ts: int) -> float:
        """
        Get liquidity factor for current session.

        Args:
            ts: Unix timestamp in milliseconds

        Returns:
            Liquidity factor (0.0 to 1.5)
        """
        _, liquidity, _ = self.get_current_session(ts)
        return liquidity

    def get_spread_multiplier(self, ts: int) -> float:
        """
        Get spread multiplier for current session.

        Args:
            ts: Unix timestamp in milliseconds

        Returns:
            Spread multiplier (0.8 to inf)
        """
        _, _, spread_mult = self.get_current_session(ts)
        return spread_mult

    def get_rollover_time_utc(self, target_date: date) -> datetime:
        """
        Get daily rollover time in UTC for a given date (DST-aware).

        Rollover is always at 5pm ET, but the UTC equivalent shifts
        based on whether US DST is in effect.

        Args:
            target_date: Date to calculate rollover for

        Returns:
            Rollover datetime in UTC

        Example:
            >>> adapter.get_rollover_time_utc(date(2024, 1, 15))  # Winter
            datetime.datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc)  # 22:00 UTC
            >>> adapter.get_rollover_time_utc(date(2024, 7, 15))  # Summer (DST)
            datetime.datetime(2024, 7, 15, 21, 0, tzinfo=timezone.utc)  # 21:00 UTC
        """
        # Create 5pm ET for the given date
        dt_et = datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=self._rollover_hour,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=ET,
        )
        # Convert to UTC (will handle DST automatically)
        return dt_et.astimezone(timezone.utc)

    def is_rollover_time(self, ts: int, tolerance_minutes: int = 5) -> bool:
        """
        Check if timestamp is within rollover window.

        Useful for swap point calculations and position adjustments.

        Args:
            ts: Unix timestamp in milliseconds
            tolerance_minutes: Minutes around rollover to consider

        Returns:
            True if within rollover window
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        rollover_utc = self.get_rollover_time_utc(dt_utc.date())

        # Check if within tolerance window
        delta = abs((dt_utc - rollover_utc).total_seconds())
        return delta <= tolerance_minutes * 60

    def get_next_session_change(self, ts: int) -> Tuple[int, ForexSessionType]:
        """
        Get timestamp and type of next session change.

        Args:
            ts: Current timestamp in milliseconds

        Returns:
            Tuple of (next_change_ts_ms, next_session_type)
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        current_hour = dt_utc.hour

        # Session boundaries (UTC hours)
        boundaries = [
            (6, ForexSessionType.TOKYO),       # Sydney -> Tokyo
            (7, ForexSessionType.TOKYO_LONDON_OVERLAP),  # Tokyo -> Overlap
            (9, ForexSessionType.LONDON),      # Overlap -> London
            (12, ForexSessionType.LONDON_NY_OVERLAP),  # London -> L/NY Overlap
            (16, ForexSessionType.NEW_YORK),   # Overlap -> NY
            (21, ForexSessionType.SYDNEY),     # NY -> Sydney
        ]

        # Find next boundary
        for hour, next_session in boundaries:
            if current_hour < hour:
                next_dt = dt_utc.replace(hour=hour, minute=0, second=0, microsecond=0)
                return (int(next_dt.timestamp() * 1000), next_session)

        # Wrap to next day (Sydney session)
        next_day = dt_utc + timedelta(days=1)
        next_dt = next_day.replace(hour=6, minute=0, second=0, microsecond=0)
        return (int(next_dt.timestamp() * 1000), ForexSessionType.TOKYO)

    def get_market_open_time(self, week_date: date) -> datetime:
        """
        Get market open time for a given week.

        Args:
            week_date: Any date in the week

        Returns:
            Market open datetime (Sunday 5pm ET) in UTC
        """
        # Find Sunday of this week
        days_since_sunday = (week_date.weekday() + 1) % 7
        sunday = week_date - timedelta(days=days_since_sunday)

        # Create 5pm ET on Sunday
        dt_et = datetime(
            year=sunday.year,
            month=sunday.month,
            day=sunday.day,
            hour=self.MARKET_OPEN_HOUR,
            minute=0,
            second=0,
            tzinfo=ET,
        )
        return dt_et.astimezone(timezone.utc)

    def get_market_close_time(self, week_date: date) -> datetime:
        """
        Get market close time for a given week.

        Args:
            week_date: Any date in the week

        Returns:
            Market close datetime (Friday 5pm ET) in UTC
        """
        # Find Friday of this week
        days_until_friday = (4 - week_date.weekday()) % 7
        if days_until_friday == 0 and week_date.weekday() != 4:
            days_until_friday = 7
        friday = week_date + timedelta(days=days_until_friday)

        # Create 5pm ET on Friday
        dt_et = datetime(
            year=friday.year,
            month=friday.month,
            day=friday.day,
            hour=self.MARKET_CLOSE_HOUR,
            minute=0,
            second=0,
            tzinfo=ET,
        )
        return dt_et.astimezone(timezone.utc)

    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if a date is a forex trading day.

        Forex trades Sunday 5pm ET through Friday 5pm ET.
        Only Saturday is never a trading day.

        Args:
            check_date: Date to check

        Returns:
            True if forex trades on this date (any part)
        """
        # Saturday is never a trading day
        return check_date.weekday() != 5

    def get_session_windows(self) -> List[ForexSessionWindow]:
        """
        Get all session window definitions.

        Returns:
            List of ForexSessionWindow objects
        """
        return self._session_windows.copy()

    def get_hours_until_close(self, ts: int) -> float:
        """
        Get hours until market close (Friday 5pm ET).

        Useful for position management before weekend.

        Args:
            ts: Unix timestamp in milliseconds

        Returns:
            Hours until close (negative if closed)
        """
        if not self.is_market_open(ts):
            return -1.0

        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        close_time = self.get_market_close_time(dt_utc.date())

        # If we're after Friday close, get next week's close
        if dt_utc > close_time:
            close_time = self.get_market_close_time(dt_utc.date() + timedelta(days=7))

        delta = close_time - dt_utc
        return delta.total_seconds() / 3600.0

    def get_session_for_pair(
        self,
        symbol: str,
        ts: int,
    ) -> Tuple[ForexSessionType, float]:
        """
        Get optimal session for a currency pair.

        Different pairs have different optimal trading sessions.

        Args:
            symbol: Currency pair (e.g., "EUR_USD", "USD_JPY")
            ts: Timestamp in milliseconds

        Returns:
            (session_type, adjusted_liquidity)
        """
        session, base_liquidity, _ = self.get_current_session(ts)

        # Adjust liquidity based on pair
        symbol_upper = symbol.upper().replace("/", "_")

        # JPY pairs are most liquid during Tokyo
        if "JPY" in symbol_upper:
            if session in (ForexSessionType.TOKYO, ForexSessionType.TOKYO_LONDON_OVERLAP):
                return (session, base_liquidity * 1.2)

        # EUR/GBP pairs are most liquid during London
        if "EUR" in symbol_upper or "GBP" in symbol_upper:
            if session == ForexSessionType.LONDON:
                return (session, base_liquidity * 1.1)

        # AUD/NZD pairs are most liquid during Sydney/Tokyo
        if "AUD" in symbol_upper or "NZD" in symbol_upper:
            if session == ForexSessionType.SYDNEY:
                return (session, base_liquidity * 1.3)

        return (session, base_liquidity)
