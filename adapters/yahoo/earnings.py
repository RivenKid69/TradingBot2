# -*- coding: utf-8 -*-
"""
adapters/yahoo/earnings.py
Yahoo Finance adapter for earnings calendar and estimates.

This adapter uses yfinance to fetch:
- Historical earnings dates and actuals
- Upcoming earnings calendar
- EPS estimates and surprises

Important for ML features:
- days_to_earnings: Feature showing proximity to earnings
- earnings_surprise: Historical surprise for regime detection
- post_earnings_drift: Price movement after earnings

References:
    - Post-Earnings Announcement Drift (PEAD)
    - Earnings momentum factor
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from adapters.base import EarningsAdapter
from adapters.models import (
    EarningsEvent,
    ExchangeVendor,
)

logger = logging.getLogger(__name__)


class YahooEarningsAdapter(EarningsAdapter):
    """
    Yahoo Finance adapter for earnings calendar.

    Fetches earnings dates, estimates, and actuals using yfinance.

    Attributes:
        _cache_ttl: Cache TTL in seconds
        _earnings_cache: In-memory cache for earnings data
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.YAHOO,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize Yahoo earnings adapter.

        Args:
            vendor: Exchange vendor (should be YAHOO)
            config: Configuration options:
                - cache_ttl: Cache TTL in seconds (default: 3600)
        """
        super().__init__(vendor, config)

        self._cache_ttl = int(self._config.get("cache_ttl", 3600))

        # In-memory caches
        self._ticker_cache: Dict[str, Any] = {}
        self._earnings_cache: Dict[str, List[EarningsEvent]] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _get_ticker(self, symbol: str) -> Any:
        """Get yfinance Ticker object with caching."""
        import time

        now = time.time()

        # Check cache validity
        if symbol in self._ticker_cache:
            cached_time = self._cache_timestamps.get(symbol, 0)
            if now - cached_time < self._cache_ttl:
                return self._ticker_cache[symbol]

        # Lazy import yfinance
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for Yahoo adapter. "
                "Install with: pip install yfinance>=0.2.0"
            ) from e

        ticker = yf.Ticker(symbol)
        self._ticker_cache[symbol] = ticker
        self._cache_timestamps[symbol] = now

        return ticker

    def get_earnings_history(
        self,
        symbol: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[EarningsEvent]:
        """
        Get historical earnings for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of EarningsEvent objects with actuals filled in
        """
        ticker = self._get_ticker(symbol)

        try:
            # Get earnings history from yfinance
            # earnings_history returns DataFrame with columns:
            # Date, EPS Estimate, Reported EPS, Surprise(%)
            earnings_df = ticker.earnings_dates

            if earnings_df is None or earnings_df.empty:
                logger.debug(f"No earnings data for {symbol}")
                return []

            events: List[EarningsEvent] = []

            for idx, row in earnings_df.iterrows():
                # Handle both datetime and Timestamp indices
                if hasattr(idx, "strftime"):
                    report_date = idx.strftime("%Y-%m-%d")
                else:
                    report_date = str(idx)[:10]

                # Filter by date range
                if start_date and report_date < start_date:
                    continue
                if end_date and report_date > end_date:
                    continue

                # Extract values safely
                eps_estimate = None
                eps_actual = None
                surprise_pct = None

                # Column names vary between yfinance versions
                estimate_col = next((c for c in row.index if "estimate" in c.lower()), None)
                actual_col = next(
                    (c for c in row.index if "reported" in c.lower() or "actual" in c.lower()), None
                )
                surprise_col = next((c for c in row.index if "surprise" in c.lower()), None)

                if estimate_col is not None:
                    val = row.get(estimate_col)
                    if val is not None and not (isinstance(val, float) and val != val):  # NaN check
                        eps_estimate = Decimal(str(round(float(val), 4)))

                if actual_col is not None:
                    val = row.get(actual_col)
                    if val is not None and not (isinstance(val, float) and val != val):
                        eps_actual = Decimal(str(round(float(val), 4)))

                if surprise_col is not None:
                    val = row.get(surprise_col)
                    if val is not None and not (isinstance(val, float) and val != val):
                        surprise_pct = float(val)

                # Determine fiscal quarter from date (approximation)
                month = int(report_date[5:7])
                fiscal_quarter = ((month - 1) // 3) + 1
                fiscal_year = int(report_date[:4])

                events.append(
                    EarningsEvent(
                        symbol=symbol.upper(),
                        report_date=report_date,
                        fiscal_quarter=fiscal_quarter,
                        fiscal_year=fiscal_year,
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        surprise_pct=surprise_pct,
                        is_confirmed=eps_actual is not None,  # Confirmed if we have actuals
                    )
                )

            # Sort by report_date ascending
            events.sort(key=lambda e: e.report_date)

            logger.debug(f"Fetched {len(events)} earnings events for {symbol}")
            return events

        except Exception as e:
            logger.warning(f"Error fetching earnings for {symbol}: {e}")
            return []

    def get_upcoming_earnings(
        self,
        symbols: Sequence[str],
        days_ahead: int = 30,
    ) -> Dict[str, List[EarningsEvent]]:
        """
        Get upcoming earnings dates for multiple symbols.

        Args:
            symbols: List of symbols
            days_ahead: Days to look ahead

        Returns:
            Dict mapping symbol to list of upcoming EarningsEvent
        """
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        result: Dict[str, List[EarningsEvent]] = {}

        for symbol in symbols:
            try:
                # Get earnings that haven't been reported yet
                all_earnings = self.get_earnings_history(
                    symbol,
                    start_date=today,
                    end_date=end_date,
                )

                # Filter to only upcoming (no actuals yet)
                upcoming = [e for e in all_earnings if not e.has_reported]

                if upcoming:
                    result[symbol] = upcoming

            except Exception as e:
                logger.warning(f"Error getting upcoming earnings for {symbol}: {e}")
                continue

        return result

    def get_earnings_calendar(
        self,
        date: str,
    ) -> List[EarningsEvent]:
        """
        Get all earnings scheduled for a specific date.

        Note: This implementation is limited by yfinance capabilities.
        For comprehensive calendar data, consider using a dedicated
        earnings calendar API (e.g., Earnings Whispers, Zacks).

        Args:
            date: Date to check (ISO format)

        Returns:
            List of EarningsEvent for companies reporting that day
        """
        # yfinance doesn't have a direct calendar endpoint
        # This would require fetching from a separate source

        logger.warning(
            "Yahoo Finance adapter doesn't support bulk calendar queries. "
            "Use get_upcoming_earnings() with specific symbols instead."
        )
        return []

    def get_earnings_with_estimates(
        self,
        symbol: str,
        num_quarters: int = 8,
    ) -> List[EarningsEvent]:
        """
        Get earnings with full estimate data for a symbol.

        Fetches recent quarters with estimates, actuals, and surprise.

        Args:
            symbol: Stock symbol
            num_quarters: Number of quarters to fetch (default: 8)

        Returns:
            List of recent EarningsEvent objects
        """
        ticker = self._get_ticker(symbol)

        events: List[EarningsEvent] = []

        try:
            # Try to get quarterly earnings
            quarterly = ticker.quarterly_earnings

            if quarterly is not None and not quarterly.empty:
                for idx, row in quarterly.iterrows():
                    # Index is typically the quarter end date
                    if hasattr(idx, "strftime"):
                        report_date = idx.strftime("%Y-%m-%d")
                    else:
                        report_date = str(idx)[:10]

                    # Extract Revenue and Earnings
                    revenue = None
                    earnings = None

                    if "Revenue" in row.index:
                        val = row["Revenue"]
                        if val is not None and not (isinstance(val, float) and val != val):
                            revenue = Decimal(str(int(val)))

                    if "Earnings" in row.index:
                        val = row["Earnings"]
                        if val is not None and not (isinstance(val, float) and val != val):
                            earnings = Decimal(str(round(float(val), 2)))

                    # Estimate fiscal quarter
                    month = int(report_date[5:7])
                    fiscal_quarter = ((month - 1) // 3) + 1
                    fiscal_year = int(report_date[:4])

                    events.append(
                        EarningsEvent(
                            symbol=symbol.upper(),
                            report_date=report_date,
                            fiscal_quarter=fiscal_quarter,
                            fiscal_year=fiscal_year,
                            eps_actual=earnings,
                            revenue_actual=revenue,
                            is_confirmed=True,
                        )
                    )

            # Sort and limit
            events.sort(key=lambda e: e.report_date, reverse=True)
            events = events[:num_quarters]
            events.reverse()  # Back to ascending order

        except Exception as e:
            logger.warning(f"Error fetching quarterly earnings for {symbol}: {e}")

        return events

    def compute_earnings_features(
        self,
        symbol: str,
        as_of_date: str,
    ) -> Dict[str, Any]:
        """
        Compute earnings-related features for a given date.

        Features computed:
        - days_to_next_earnings: Days until next earnings (0-90, capped)
        - days_since_last_earnings: Days since last earnings
        - last_surprise_pct: EPS surprise from last quarter
        - beat_streak: Consecutive quarters beating estimates

        Args:
            symbol: Stock symbol
            as_of_date: Reference date (ISO format)

        Returns:
            Dict of feature name to value
        """
        features: Dict[str, Any] = {
            "days_to_next_earnings": None,
            "days_since_last_earnings": None,
            "last_surprise_pct": None,
            "beat_streak": 0,
        }

        try:
            # Get historical earnings
            start_date = (datetime.fromisoformat(as_of_date) - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            )
            end_date = (datetime.fromisoformat(as_of_date) + timedelta(days=90)).strftime(
                "%Y-%m-%d"
            )

            earnings = self.get_earnings_history(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )

            if not earnings:
                return features

            as_of_dt = datetime.fromisoformat(as_of_date)

            # Split into past and future
            past_earnings = [
                e for e in earnings if datetime.fromisoformat(e.report_date) <= as_of_dt
            ]
            future_earnings = [
                e for e in earnings if datetime.fromisoformat(e.report_date) > as_of_dt
            ]

            # Days since last earnings
            if past_earnings:
                last = past_earnings[-1]
                last_dt = datetime.fromisoformat(last.report_date)
                features["days_since_last_earnings"] = (as_of_dt - last_dt).days
                features["last_surprise_pct"] = last.surprise_pct

                # Compute beat streak
                streak = 0
                for e in reversed(past_earnings):
                    if e.beat_estimates:
                        streak += 1
                    else:
                        break
                features["beat_streak"] = streak

            # Days to next earnings
            if future_earnings:
                next_e = future_earnings[0]
                next_dt = datetime.fromisoformat(next_e.report_date)
                days_to = (next_dt - as_of_dt).days
                features["days_to_next_earnings"] = min(days_to, 90)  # Cap at 90

        except Exception as e:
            logger.warning(f"Error computing earnings features for {symbol}: {e}")

        return features
