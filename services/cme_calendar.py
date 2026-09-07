# -*- coding: utf-8 -*-
"""
services/cme_calendar.py
CME trading calendar and hours service.

Provides:
- CME Globex trading hours
- US market holidays
- Settlement time lookup
- Maintenance window handling

CME Globex Hours:
- Regular: Sunday 6:00pm ET - Friday 5:00pm ET
- Daily maintenance: 4:15pm - 4:30pm ET (Mon-Fri)
- Extended maintenance: Sunday - may vary

Reference:
- CME Globex: https://www.cmegroup.com/globex.html
- Holiday Calendar: https://www.cmegroup.com/tools-information/holiday-calendar.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =========================
# Trading Session Types
# =========================


class CMESession(str, Enum):
    """CME trading session types."""

    REGULAR = "regular"  # Normal trading hours
    MAINTENANCE = "maintenance"  # Daily maintenance window
    CLOSED = "closed"  # Market closed (weekends, holidays)


# =========================
# Holiday Calendar
# =========================

# CME Group observed holidays (2024-2026)
# Markets are closed or have early close on these dates
CME_HOLIDAYS: Set[str] = {
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}

# Early close days (typically day before major holidays)
CME_EARLY_CLOSE: Dict[str, time] = {
    # Day before Thanksgiving - closes at 12:15 PM CT (1:15 PM ET)
    "2024-11-27": time(13, 15),
    "2025-11-26": time(13, 15),
    # Christmas Eve - varies by product, typical 12:15 PM CT
    "2024-12-24": time(13, 15),
    "2025-12-24": time(13, 15),
    # New Year's Eve - typical early close
    "2024-12-31": time(13, 15),
    "2025-12-31": time(13, 15),
}


# =========================
# Trading Hours Models
# =========================


@dataclass
class TradingHoursInfo:
    """Trading hours information for a specific time."""

    session: CMESession
    is_trading: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    notes: str = ""


@dataclass
class DaySchedule:
    """Trading schedule for a specific day."""

    date: date
    is_holiday: bool
    is_early_close: bool
    open_time: Optional[time] = None
    close_time: Optional[time] = None
    maintenance_start: Optional[time] = None
    maintenance_end: Optional[time] = None


# =========================
# CME Trading Calendar
# =========================


class CMETradingCalendar:
    """
    CME trading hours and holiday calendar.

    CME Globex Hours (Eastern Time):
    ─────────────────────────────────────────────────────────────────
    | Day       | Open Time        | Close Time       | Notes       |
    |-----------|------------------|------------------|-------------|
    | Sunday    | 6:00 PM ET       | -                | Week opens  |
    | Mon-Thu   | 24 hours         | 24 hours         | With break  |
    | Friday    | -                | 5:00 PM ET       | Week closes |
    | Sat       | Closed           | Closed           |             |
    ─────────────────────────────────────────────────────────────────

    Daily Maintenance Window (Mon-Fri):
    - 4:15 PM ET - 4:30 PM ET (15 minutes)
    - All trading halted during this window

    Usage:
        calendar = CMETradingCalendar()

        # Check if trading
        if calendar.is_trading_hours(datetime.now()):
            submit_order()

        # Get next open/close
        next_open = calendar.get_next_open(datetime.now())
        next_close = calendar.get_next_close(datetime.now())

        # Check for holiday
        if calendar.is_holiday(date.today()):
            print("Market closed for holiday")
    """

    # Globex hours in Eastern Time
    GLOBEX_SUNDAY_OPEN = time(18, 0)  # Sunday 6:00 PM
    GLOBEX_FRIDAY_CLOSE = time(17, 0)  # Friday 5:00 PM

    # Daily maintenance window in Eastern Time
    MAINTENANCE_START = time(16, 15)  # 4:15 PM
    MAINTENANCE_END = time(16, 30)  # 4:30 PM

    # Extended holidays set (can be expanded)
    HOLIDAYS = CME_HOLIDAYS
    EARLY_CLOSE = CME_EARLY_CLOSE

    def __init__(
        self,
        timezone_name: str = "US/Eastern",
    ) -> None:
        """
        Initialize trading calendar.

        Args:
            timezone_name: Timezone for time comparisons (default: US/Eastern)
        """
        self._timezone_name = timezone_name
        # Note: In production, use pytz for proper timezone handling
        # For now, we'll work with UTC and manual offset

    def is_trading_hours(
        self,
        dt: datetime,
        check_maintenance: bool = True,
        check_holidays: bool = True,
    ) -> bool:
        """
        Check if market is open at given datetime.

        Args:
            dt: Datetime to check (should be timezone-aware or UTC)
            check_maintenance: If True, also check maintenance window
            check_holidays: If True, also check holiday calendar

        Returns:
            True if market is open
        """
        # Convert to Eastern Time (simplified: assume UTC input, offset by -5)
        et_dt = self._to_eastern(dt)
        weekday = et_dt.weekday()  # Monday=0, Sunday=6
        t = et_dt.time()
        d = et_dt.date()

        # Check holiday first
        if check_holidays and self.is_holiday(d):
            return False

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open from 6pm
        if weekday == 6:
            return t >= self.GLOBEX_SUNDAY_OPEN

        # Friday: close at 5pm
        if weekday == 4 and t >= self.GLOBEX_FRIDAY_CLOSE:
            return False

        # Check early close
        date_str = d.isoformat()
        if date_str in self.EARLY_CLOSE:
            early_close = self.EARLY_CLOSE[date_str]
            if t >= early_close:
                return False

        # Daily maintenance window (Mon-Fri)
        if check_maintenance and weekday < 5:  # Mon-Fri
            if self.MAINTENANCE_START <= t < self.MAINTENANCE_END:
                return False

        # Otherwise open
        return True

    def get_session(self, dt: datetime) -> CMESession:
        """
        Get current trading session type.

        Args:
            dt: Datetime to check

        Returns:
            CMESession type
        """
        et_dt = self._to_eastern(dt)
        weekday = et_dt.weekday()
        t = et_dt.time()
        d = et_dt.date()

        # Holiday
        if self.is_holiday(d):
            return CMESession.CLOSED

        # Weekend
        if weekday == 5:  # Saturday
            return CMESession.CLOSED
        if weekday == 6 and t < self.GLOBEX_SUNDAY_OPEN:  # Sunday before open
            return CMESession.CLOSED

        # Friday after close
        if weekday == 4 and t >= self.GLOBEX_FRIDAY_CLOSE:
            return CMESession.CLOSED

        # Maintenance window
        if weekday < 5 and self.MAINTENANCE_START <= t < self.MAINTENANCE_END:
            return CMESession.MAINTENANCE

        return CMESession.REGULAR

    def get_trading_hours_info(self, dt: datetime) -> TradingHoursInfo:
        """
        Get comprehensive trading hours information.

        Args:
            dt: Datetime to check

        Returns:
            TradingHoursInfo with session details
        """
        session = self.get_session(dt)
        is_trading = session == CMESession.REGULAR

        next_open = None
        next_close = None
        notes = ""

        if not is_trading:
            next_open = self.get_next_open(dt)
            if session == CMESession.MAINTENANCE:
                notes = "Daily maintenance window"
            elif session == CMESession.CLOSED:
                d = self._to_eastern(dt).date()
                if self.is_holiday(d):
                    notes = "Market holiday"
                else:
                    notes = "Weekend closure"
        else:
            next_close = self.get_next_close(dt)

        return TradingHoursInfo(
            session=session,
            is_trading=is_trading,
            next_open=next_open,
            next_close=next_close,
            notes=notes,
        )

    def get_next_open(self, dt: datetime) -> datetime:
        """
        Get next market open datetime.

        Args:
            dt: Current datetime

        Returns:
            Datetime of next market open
        """
        et_dt = self._to_eastern(dt)
        weekday = et_dt.weekday()
        t = et_dt.time()
        d = et_dt.date()

        # Already trading? Return current time
        if self.is_trading_hours(dt):
            return dt

        # In maintenance window? Return end of maintenance
        if weekday < 5 and self.MAINTENANCE_START <= t < self.MAINTENANCE_END:
            next_open = datetime.combine(d, self.MAINTENANCE_END)
            return self._from_eastern(next_open)

        # After Friday close or on Saturday
        if (weekday == 4 and t >= self.GLOBEX_FRIDAY_CLOSE) or weekday == 5:
            # Next open is Sunday 6pm
            days_until_sunday = (6 - weekday) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7 if weekday == 6 and t >= self.GLOBEX_SUNDAY_OPEN else 0
            next_sunday = d + timedelta(days=days_until_sunday)
            next_open = datetime.combine(next_sunday, self.GLOBEX_SUNDAY_OPEN)
            return self._from_eastern(next_open)

        # Sunday before open
        if weekday == 6 and t < self.GLOBEX_SUNDAY_OPEN:
            next_open = datetime.combine(d, self.GLOBEX_SUNDAY_OPEN)
            return self._from_eastern(next_open)

        # Holiday - find next non-holiday weekday
        if self.is_holiday(d):
            next_d = d + timedelta(days=1)
            while self.is_holiday(next_d) or next_d.weekday() >= 5:
                next_d += timedelta(days=1)
            # Open at midnight (actually continuing from previous session)
            next_open = datetime.combine(next_d, time(0, 0))
            return self._from_eastern(next_open)

        # Fallback
        return dt

    def get_next_close(self, dt: datetime) -> datetime:
        """
        Get next market close datetime.

        Args:
            dt: Current datetime

        Returns:
            Datetime of next market close
        """
        et_dt = self._to_eastern(dt)
        weekday = et_dt.weekday()
        t = et_dt.time()
        d = et_dt.date()

        # Not trading? No close
        if not self.is_trading_hours(dt):
            return dt

        # Check for early close today
        date_str = d.isoformat()
        if date_str in self.EARLY_CLOSE:
            early_close = self.EARLY_CLOSE[date_str]
            if t < early_close:
                next_close = datetime.combine(d, early_close)
                return self._from_eastern(next_close)

        # Maintenance window coming up today
        if weekday < 5 and t < self.MAINTENANCE_START:
            next_close = datetime.combine(d, self.MAINTENANCE_START)
            return self._from_eastern(next_close)

        # Friday close
        if weekday == 4 and t < self.GLOBEX_FRIDAY_CLOSE:
            next_close = datetime.combine(d, self.GLOBEX_FRIDAY_CLOSE)
            return self._from_eastern(next_close)

        # Otherwise, next maintenance or Friday
        days_until_friday = (4 - weekday) % 7
        if days_until_friday == 0 and t >= self.GLOBEX_FRIDAY_CLOSE:
            days_until_friday = 7

        if days_until_friday > 0:
            # Next close is tomorrow's maintenance or Friday
            next_d = d + timedelta(days=1)
            if next_d.weekday() < 5:
                next_close = datetime.combine(next_d, self.MAINTENANCE_START)
                return self._from_eastern(next_close)

        # Next Friday close
        next_friday = d + timedelta(days=days_until_friday)
        next_close = datetime.combine(next_friday, self.GLOBEX_FRIDAY_CLOSE)
        return self._from_eastern(next_close)

    def is_holiday(self, d: date) -> bool:
        """
        Check if date is a market holiday.

        Args:
            d: Date to check

        Returns:
            True if market is closed for holiday
        """
        return d.isoformat() in self.HOLIDAYS

    def is_early_close(self, d: date) -> bool:
        """
        Check if date has early close.

        Args:
            d: Date to check

        Returns:
            True if early close day
        """
        return d.isoformat() in self.EARLY_CLOSE

    def get_day_schedule(self, d: date) -> DaySchedule:
        """
        Get trading schedule for a specific day.

        Args:
            d: Date to get schedule for

        Returns:
            DaySchedule with open/close times
        """
        weekday = d.weekday()
        date_str = d.isoformat()

        is_holiday = self.is_holiday(d)
        is_early = self.is_early_close(d)

        if is_holiday:
            return DaySchedule(
                date=d,
                is_holiday=True,
                is_early_close=False,
            )

        if weekday == 5:  # Saturday
            return DaySchedule(
                date=d,
                is_holiday=False,
                is_early_close=False,
            )

        if weekday == 6:  # Sunday
            return DaySchedule(
                date=d,
                is_holiday=False,
                is_early_close=False,
                open_time=self.GLOBEX_SUNDAY_OPEN,
            )

        if weekday == 4:  # Friday
            close = self.EARLY_CLOSE.get(date_str, self.GLOBEX_FRIDAY_CLOSE)
            return DaySchedule(
                date=d,
                is_holiday=False,
                is_early_close=is_early,
                open_time=time(0, 0),
                close_time=close,
                maintenance_start=self.MAINTENANCE_START,
                maintenance_end=self.MAINTENANCE_END,
            )

        # Mon-Thu
        close = self.EARLY_CLOSE.get(date_str) if is_early else None
        return DaySchedule(
            date=d,
            is_holiday=False,
            is_early_close=is_early,
            open_time=time(0, 0),
            close_time=close or time(23, 59),
            maintenance_start=self.MAINTENANCE_START,
            maintenance_end=self.MAINTENANCE_END,
        )

    def get_holidays_in_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """
        Get all holidays in a date range.

        Args:
            start_date: Start of range
            end_date: End of range

        Returns:
            List of holiday dates
        """
        holidays = []
        for holiday_str in self.HOLIDAYS:
            try:
                holiday_date = date.fromisoformat(holiday_str)
                if start_date <= holiday_date <= end_date:
                    holidays.append(holiday_date)
            except ValueError:
                continue
        return sorted(holidays)

    def count_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> int:
        """
        Count number of trading days in a range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            Number of trading days
        """
        count = 0
        current = start_date

        while current <= end_date:
            if current.weekday() < 5 and not self.is_holiday(current):
                count += 1
            current += timedelta(days=1)

        return count

    def add_trading_days(
        self,
        start_date: date,
        days: int,
    ) -> date:
        """
        Add trading days to a date.

        Args:
            start_date: Starting date
            days: Number of trading days to add

        Returns:
            Resulting date
        """
        current = start_date
        remaining = days

        while remaining > 0:
            current += timedelta(days=1)
            if current.weekday() < 5 and not self.is_holiday(current):
                remaining -= 1

        return current

    # =========================
    # Private Helper Methods
    # =========================

    def _to_eastern(self, dt: datetime) -> datetime:
        """
        Convert datetime to Eastern Time.

        Note: Simplified implementation. In production, use pytz.
        """
        # Assume input is UTC, offset by -5 hours (EST)
        # This doesn't handle DST - production should use pytz
        if dt.tzinfo is None:
            # Assume UTC
            return dt - timedelta(hours=5)
        # Already timezone-aware - would need proper conversion
        return dt

    def _from_eastern(self, dt: datetime) -> datetime:
        """
        Convert Eastern Time to UTC.

        Note: Simplified implementation. In production, use pytz.
        """
        # Add 5 hours to get UTC (EST)
        return dt + timedelta(hours=5)


# =========================
# Convenience Functions
# =========================


def create_cme_calendar() -> CMETradingCalendar:
    """Create a new CME trading calendar."""
    return CMETradingCalendar()


def is_cme_trading_now() -> bool:
    """Check if CME Globex is currently trading."""
    calendar = CMETradingCalendar()
    return calendar.is_trading_hours(datetime.utcnow())


def get_next_cme_open() -> datetime:
    """Get next CME Globex open time."""
    calendar = CMETradingCalendar()
    return calendar.get_next_open(datetime.utcnow())


def get_next_cme_close() -> datetime:
    """Get next CME Globex close time."""
    calendar = CMETradingCalendar()
    return calendar.get_next_close(datetime.utcnow())
