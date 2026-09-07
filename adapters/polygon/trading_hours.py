# -*- coding: utf-8 -*-
"""
adapters/polygon/trading_hours.py
Polygon.io trading hours adapter implementation.

Provides US market schedule information from Polygon.io API:
- Market status (open/closed)
- Next open/close times
- Holiday calendar
- Pre-market and after-hours sessions

API Reference:
    https://polygon.io/docs/stocks/get_v1_marketstatus_now
    https://polygon.io/docs/stocks/get_v1_marketstatus_upcoming
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, List, Mapping, Optional, Tuple

from ..base import TradingHoursAdapter
from ..models import (
    ExchangeVendor,
    MarketCalendar,
    MarketType,
    SessionType,
    TradingSession,
    US_EQUITY_SESSIONS,
    create_us_equity_calendar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# US MARKET HOLIDAYS (2024-2025)
# =============================================================================

# Note: This is a static list. For production, fetch from Polygon API.
US_MARKET_HOLIDAYS: List[Tuple[int, int, int]] = [
    # 2024
    (2024, 1, 1),  # New Year's Day
    (2024, 1, 15),  # MLK Day
    (2024, 2, 19),  # Presidents Day
    (2024, 3, 29),  # Good Friday
    (2024, 5, 27),  # Memorial Day
    (2024, 6, 19),  # Juneteenth
    (2024, 7, 4),  # Independence Day
    (2024, 9, 2),  # Labor Day
    (2024, 11, 28),  # Thanksgiving
    (2024, 12, 25),  # Christmas
    # 2025
    (2025, 1, 1),  # New Year's Day
    (2025, 1, 20),  # MLK Day
    (2025, 2, 17),  # Presidents Day
    (2025, 4, 18),  # Good Friday
    (2025, 5, 26),  # Memorial Day
    (2025, 6, 19),  # Juneteenth
    (2025, 7, 4),  # Independence Day
    (2025, 9, 1),  # Labor Day
    (2025, 11, 27),  # Thanksgiving
    (2025, 12, 25),  # Christmas
]

# Half days (early close at 13:00 ET)
US_MARKET_HALF_DAYS: List[Tuple[int, int, int]] = [
    (2024, 7, 3),  # Day before Independence Day
    (2024, 11, 29),  # Day after Thanksgiving
    (2024, 12, 24),  # Christmas Eve
    (2025, 7, 3),  # Day before Independence Day
    (2025, 11, 28),  # Day after Thanksgiving
    (2025, 12, 24),  # Christmas Eve
]


# =============================================================================
# POLYGON TRADING HOURS ADAPTER
# =============================================================================


class PolygonTradingHoursAdapter(TradingHoursAdapter):
    """
    Trading hours adapter for Polygon.io (US markets).

    Provides:
    - Real-time market status from Polygon API
    - US market schedule (NYSE/NASDAQ)
    - Holiday and half-day handling

    Configuration:
        api_key: Polygon.io API key
        use_cache: Cache market status (default: True)
        cache_ttl: Cache TTL in seconds (default: 60)
        extended_hours: Include pre/after-market (default: True)

    Example:
        adapter = PolygonTradingHoursAdapter(config={"api_key": "..."})

        if adapter.is_market_open(now_ms):
            # Market is open
            pass
        else:
            next_open = adapter.next_open(now_ms)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.POLYGON,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor=vendor, config=config)

        self._api_key = self._config.get("api_key") or os.environ.get("POLYGON_API_KEY", "")
        self._use_cache = bool(self._config.get("use_cache", True))
        self._cache_ttl = float(self._config.get("cache_ttl", 60))
        self._extended_hours = bool(self._config.get("extended_hours", True))

        # Cache
        self._market_status_cache: Optional[dict] = None
        self._cache_timestamp: Optional[datetime] = None

        # Calendar
        self._calendar = self._build_calendar()

        # REST client (lazy)
        self._rest_client: Optional[Any] = None

    def _get_rest_client(self) -> Any:
        """Lazy initialization of REST client."""
        if self._rest_client is None:
            if not self._api_key:
                raise ValueError("Polygon API key required")
            try:
                from polygon import RESTClient

                self._rest_client = RESTClient(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "polygon-api-client not installed. Run: pip install polygon-api-client"
                )
        return self._rest_client

    def _build_calendar(self) -> MarketCalendar:
        """Build US market calendar."""
        return MarketCalendar(
            vendor=ExchangeVendor.POLYGON,
            market_type=MarketType.EQUITY,
            sessions=list(US_EQUITY_SESSIONS),
            holidays=list(US_MARKET_HOLIDAYS),
            half_days=list(US_MARKET_HALF_DAYS),
            timezone="America/New_York",
        )

    # -------------------------------------------------------------------------
    # TradingHoursAdapter Interface
    # -------------------------------------------------------------------------

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """
        Check if market is open at given timestamp.

        Uses Polygon API for real-time status, falling back to schedule.

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Specific session (None = any open session)

        Returns:
            True if market is open
        """
        # Try API first
        try:
            status = self._get_market_status()
            if status:
                market_open = status.get("market") == "open"

                if session_type is None:
                    # Any session counts
                    if self._extended_hours:
                        return market_open or status.get("afterHours") or status.get("earlyHours")
                    return market_open

                elif session_type == SessionType.REGULAR:
                    return market_open

                elif session_type == SessionType.PRE_MARKET:
                    return bool(status.get("earlyHours"))

                elif session_type == SessionType.AFTER_HOURS:
                    return bool(status.get("afterHours"))

                return market_open

        except Exception as e:
            logger.debug(f"Failed to get market status from API: {e}")

        # Fallback to schedule-based calculation
        return self._is_open_by_schedule(ts, session_type)

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
            session_type: Specific session type

        Returns:
            Timestamp of next open in milliseconds
        """
        return self._next_session_boundary(ts, is_open=True, session_type=session_type)

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
            session_type: Specific session type

        Returns:
            Timestamp of next close in milliseconds
        """
        return self._next_session_boundary(ts, is_open=False, session_type=session_type)

    def get_calendar(self) -> MarketCalendar:
        """Get market calendar."""
        return self._calendar

    def is_holiday(self, ts: int) -> bool:
        """Check if date is a market holiday."""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

        # Convert to ET for date comparison
        try:
            import pytz

            et = pytz.timezone("America/New_York")
            dt_et = dt.astimezone(et)
            date_tuple = (dt_et.year, dt_et.month, dt_et.day)
        except ImportError:
            # Fallback without pytz (approximate)
            date_tuple = (dt.year, dt.month, dt.day)

        return date_tuple in self._calendar.holidays

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _get_market_status(self) -> Optional[dict]:
        """Get market status from API with caching."""
        now = datetime.now(timezone.utc)

        # Check cache
        if (
            self._use_cache
            and self._market_status_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl
        ):
            return self._market_status_cache

        # Fetch from API
        try:
            client = self._get_rest_client()
            status = client.get_market_status()

            self._market_status_cache = status.__dict__ if hasattr(status, "__dict__") else {}
            self._cache_timestamp = now

            return self._market_status_cache

        except Exception as e:
            logger.warning(f"Failed to fetch market status: {e}")
            return None

    def _is_open_by_schedule(
        self,
        ts: int,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """Check if market is open based on schedule."""
        try:
            import pytz

            et = pytz.timezone("America/New_York")
            dt = datetime.fromtimestamp(ts / 1000, tz=pytz.UTC).astimezone(et)
        except ImportError:
            # Fallback: assume UTC-5 offset
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            dt = dt - timedelta(hours=5)

        # Check holiday
        date_tuple = (dt.year, dt.month, dt.day)
        if date_tuple in self._calendar.holidays:
            return False

        # Check day of week (0=Monday, 6=Sunday)
        if dt.weekday() >= 5:  # Weekend
            return False

        # Convert time to minutes from midnight
        minutes = dt.hour * 60 + dt.minute

        # Check half day (early close at 13:00 = 780 minutes)
        is_half_day = date_tuple in self._calendar.half_days

        # Check each session
        for session in self._calendar.sessions:
            if session_type is not None and session.session_type != session_type:
                continue

            if dt.weekday() not in session.days_of_week:
                continue

            start = session.start_minutes
            end = session.end_minutes

            # Adjust for half days
            if is_half_day and session.session_type == SessionType.REGULAR:
                end = 13 * 60  # 13:00 ET

            # Handle overnight sessions
            if end > start:
                if start <= minutes < end:
                    return True
            else:
                # Overnight (shouldn't happen for US equities)
                if minutes >= start or minutes < end:
                    return True

        return False

    def _next_session_boundary(
        self,
        ts: int,
        is_open: bool,
        session_type: Optional[SessionType],
    ) -> int:
        """Calculate next open or close time."""
        try:
            import pytz

            et = pytz.timezone("America/New_York")
            dt = datetime.fromtimestamp(ts / 1000, tz=pytz.UTC).astimezone(et)
        except ImportError:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            dt = dt - timedelta(hours=5)

        # Search up to 7 days ahead
        for day_offset in range(8):
            check_date = dt.date() + timedelta(days=day_offset)

            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            # Skip holidays
            date_tuple = (check_date.year, check_date.month, check_date.day)
            if date_tuple in self._calendar.holidays:
                continue

            # Check half day
            is_half_day = date_tuple in self._calendar.half_days

            # Get target session
            target_session = None
            for session in self._calendar.sessions:
                if session_type is not None and session.session_type != session_type:
                    continue
                if session.session_type == SessionType.REGULAR or session_type is None:
                    target_session = session
                    break

            if target_session is None:
                target_session = self._calendar.sessions[1]  # Regular session

            # Calculate boundary time
            if is_open:
                boundary_minutes = target_session.start_minutes
            else:
                boundary_minutes = target_session.end_minutes
                if is_half_day:
                    boundary_minutes = 13 * 60  # 13:00 ET

            boundary_hour = boundary_minutes // 60
            boundary_min = boundary_minutes % 60

            try:
                import pytz

                boundary_dt = et.localize(
                    datetime(
                        check_date.year,
                        check_date.month,
                        check_date.day,
                        boundary_hour,
                        boundary_min,
                    )
                )
                boundary_ts = int(boundary_dt.timestamp() * 1000)
            except:
                # Fallback
                boundary_dt = datetime(
                    check_date.year,
                    check_date.month,
                    check_date.day,
                    boundary_hour,
                    boundary_min,
                    tzinfo=timezone.utc,
                )
                boundary_ts = int((boundary_dt.timestamp() + 5 * 3600) * 1000)

            # Return if boundary is in the future
            if boundary_ts > ts:
                return boundary_ts

        # Fallback: return ts + 1 day
        return ts + 86_400_000
