# -*- coding: utf-8 -*-
"""
economic_calendar.py
Economic events calendar for forex trading (Phase 4).

Provides access to economic calendar events that can significantly impact
currency prices. High-impact events (NFP, FOMC, ECB, etc.) often cause
increased volatility and spread widening.

Data Sources (in priority order):
1. OANDA Labs Calendar API (primary for live)
2. ForexFactory (scraping fallback)
3. Cached historical data (for backtesting)

High-Impact Events by Currency:
- USD: NFP, FOMC, CPI, GDP, ISM PMI, Retail Sales
- EUR: ECB, German CPI, Eurozone GDP, PMI
- GBP: BOE, UK CPI, UK GDP, Employment
- JPY: BOJ, Tankan, Japan CPI
- AUD: RBA, Employment, CPI
- CAD: BOC, Employment, CPI
- CHF: SNB, CPI
- NZD: RBNZ, GDP, CPI

Impact Levels (0-3):
- 0: No impact / holiday
- 1: Low impact (minor data)
- 2: Medium impact (secondary data)
- 3: High impact (major data, central bank)

References:
- Andersen et al. (2003): "Micro Effects of Macro Announcements"
- Ehrmann & Fratzscher (2005): "Exchange Rates and Fundamentals"

Author: Sultan Suyunbayev
Date: 2025-11-30
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# High-impact event keywords by currency
HIGH_IMPACT_KEYWORDS: Dict[str, List[str]] = {
    "USD": [
        "Non-Farm Payrolls", "NFP", "FOMC", "Federal Reserve", "Fed Chair",
        "CPI", "Core CPI", "GDP", "ISM Manufacturing", "ISM Services",
        "Retail Sales", "Unemployment Rate", "Treasury", "Powell",
    ],
    "EUR": [
        "ECB", "European Central Bank", "Lagarde", "German CPI", "Eurozone GDP",
        "German PMI", "Eurozone CPI", "German Ifo", "German ZEW",
    ],
    "GBP": [
        "BOE", "Bank of England", "Bailey", "UK CPI", "UK GDP",
        "UK Employment", "UK Retail Sales", "UK PMI",
    ],
    "JPY": [
        "BOJ", "Bank of Japan", "Ueda", "Japan CPI", "Tankan",
        "Japan GDP", "Japan Trade Balance",
    ],
    "AUD": [
        "RBA", "Reserve Bank of Australia", "Australia Employment",
        "Australia CPI", "Australia GDP", "Australia Retail Sales",
    ],
    "CAD": [
        "BOC", "Bank of Canada", "Canada Employment", "Canada CPI",
        "Canada GDP", "Canada Retail Sales", "Ivey PMI",
    ],
    "CHF": [
        "SNB", "Swiss National Bank", "Switzerland CPI",
        "Switzerland GDP", "KOF Leading Indicator",
    ],
    "NZD": [
        "RBNZ", "Reserve Bank of New Zealand", "NZ GDP",
        "NZ CPI", "NZ Employment", "NZ Trade Balance",
    ],
}

# Central bank meeting schedule (approximate, update annually)
CENTRAL_BANK_MEETINGS: Dict[str, int] = {
    "FOMC": 8,      # 8 meetings per year
    "ECB": 8,
    "BOE": 8,
    "BOJ": 8,
    "RBA": 11,      # Monthly except January
    "BOC": 8,
    "SNB": 4,
    "RBNZ": 7,
}

# OANDA Labs Calendar API endpoint
OANDA_CALENDAR_URL = "https://www.oanda.com/labsapi/calendar"

# ForexFactory base URL
FOREXFACTORY_URL = "https://www.forexfactory.com/calendar"

# Default cache directory
DEFAULT_CACHE_DIR = "data/calendar_cache"


# =============================================================================
# ENUMS
# =============================================================================

class ImpactLevel(Enum):
    """Economic event impact level."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class CalendarSource(Enum):
    """Calendar data source."""
    OANDA_LABS = "oanda_labs"
    FOREXFACTORY = "forexfactory"
    INVESTING_COM = "investing_com"
    CACHED = "cached"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EconomicEvent:
    """
    Represents a single economic calendar event.

    Attributes:
        event_id: Unique identifier
        datetime_utc: Event datetime in UTC
        currency: Affected currency code
        event_name: Event description
        impact: Impact level (0-3)
        actual: Actual value (if released)
        forecast: Forecast value
        previous: Previous value
        source: Data source
    """
    event_id: str
    datetime_utc: datetime
    currency: str
    event_name: str
    impact: ImpactLevel = ImpactLevel.LOW
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    source: CalendarSource = CalendarSource.CACHED

    def __post_init__(self):
        """Ensure datetime is UTC."""
        if self.datetime_utc.tzinfo is None:
            self.datetime_utc = self.datetime_utc.replace(tzinfo=timezone.utc)

    @property
    def timestamp_ms(self) -> int:
        """Get timestamp in milliseconds."""
        return int(self.datetime_utc.timestamp() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "datetime": self.datetime_utc.isoformat(),
            "timestamp": self.timestamp_ms,
            "currency": self.currency,
            "event_name": self.event_name,
            "impact": self.impact.value,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EconomicEvent":
        """Create from dictionary."""
        dt = data.get("datetime") or data.get("time")
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        elif isinstance(dt, (int, float)):
            dt = datetime.fromtimestamp(dt / 1000.0, tz=timezone.utc)

        return cls(
            event_id=data.get("event_id", data.get("id", "")),
            datetime_utc=dt,
            currency=data.get("currency", ""),
            event_name=data.get("event_name", data.get("name", data.get("title", ""))),
            impact=ImpactLevel(data.get("impact", 1)),
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
            source=CalendarSource(data.get("source", "cached")),
        )


@dataclass
class CalendarConfig:
    """Configuration for calendar data fetching."""
    source: CalendarSource = CalendarSource.CACHED
    cache_dir: str = DEFAULT_CACHE_DIR
    cache_ttl_hours: float = 24.0
    oanda_api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    retry_count: int = 3
    high_impact_only: bool = False


# =============================================================================
# BASE CALENDAR PROVIDER
# =============================================================================

class CalendarProvider(ABC):
    """Abstract base class for calendar data providers."""

    @abstractmethod
    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """
        Get economic events in date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            currencies: Filter by currencies (None = all)

        Returns:
            List of EconomicEvent objects
        """
        pass

    @abstractmethod
    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """
        Get upcoming events for specified currencies.

        Args:
            currencies: Currency codes to filter
            hours_ahead: Hours to look ahead

        Returns:
            List of upcoming events
        """
        pass


# =============================================================================
# CACHED CALENDAR PROVIDER
# =============================================================================

class CachedCalendarProvider(CalendarProvider):
    """
    Calendar provider using local cached data.

    Ideal for backtesting where historical calendar data is needed.
    """

    def __init__(self, config: Optional[CalendarConfig] = None):
        """Initialize with configuration."""
        self.config = config or CalendarConfig()
        self._cache: Dict[str, List[EconomicEvent]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached calendar data from disk."""
        cache_dir = self.config.cache_dir
        if not os.path.exists(cache_dir):
            return

        for filename in os.listdir(cache_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    events = [EconomicEvent.from_dict(e) for e in data]
                    year_month = filename.replace(".json", "")
                    self._cache[year_month] = events
            except Exception as e:
                logger.warning(f"Failed to load calendar cache {filename}: {e}")

    def save_cache(self) -> None:
        """Save cache to disk."""
        cache_dir = self.config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        for key, events in self._cache.items():
            filepath = os.path.join(cache_dir, f"{key}.json")
            try:
                with open(filepath, "w") as f:
                    json.dump([e.to_dict() for e in events], f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save calendar cache {key}: {e}")

    def add_events(self, events: List[EconomicEvent]) -> None:
        """Add events to cache."""
        for event in events:
            key = event.datetime_utc.strftime("%Y-%m")
            if key not in self._cache:
                self._cache[key] = []
            self._cache[key].append(event)

    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get events from cache."""
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Collect events from relevant months
        all_events: List[EconomicEvent] = []
        current = start_date.replace(day=1)
        while current <= end_date:
            key = current.strftime("%Y-%m")
            if key in self._cache:
                all_events.extend(self._cache[key])
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Filter by date range
        events = [e for e in all_events
                  if start_date <= e.datetime_utc <= end_date]

        # Filter by currency
        if currencies:
            currencies_upper = [c.upper() for c in currencies]
            events = [e for e in events if e.currency.upper() in currencies_upper]

        # Filter by impact if configured
        if self.config.high_impact_only:
            events = [e for e in events if e.impact == ImpactLevel.HIGH]

        return sorted(events, key=lambda e: e.datetime_utc)

    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """Get upcoming events."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        return self.get_events(now, end, currencies)


# =============================================================================
# OANDA LABS CALENDAR PROVIDER
# =============================================================================

class OandaCalendarProvider(CalendarProvider):
    """
    Calendar provider using OANDA Labs API.

    Requires OANDA API key for access.
    Free tier available with rate limits.
    """

    def __init__(self, config: Optional[CalendarConfig] = None):
        """Initialize with configuration."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for OandaCalendarProvider")

        self.config = config or CalendarConfig()
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # Rate limit: 2 req/sec

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        headers = {
            "Accept": "application/json",
        }
        if self.config.oanda_api_key:
            headers["Authorization"] = f"Bearer {self.config.oanda_api_key}"

        url = f"{OANDA_CALENDAR_URL}/{endpoint}"

        for attempt in range(self.config.retry_count):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                self._last_request_time = time.time()

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"OANDA API error: {response.status_code}")
                    return None

            except requests.RequestException as e:
                logger.warning(f"OANDA request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        return None

    def _parse_event(self, data: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Parse OANDA event data."""
        try:
            # OANDA uses epoch seconds
            timestamp = data.get("timestamp", 0)
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Map OANDA impact to our scale
            oanda_impact = data.get("impact", 1)
            if oanda_impact >= 3:
                impact = ImpactLevel.HIGH
            elif oanda_impact >= 2:
                impact = ImpactLevel.MEDIUM
            elif oanda_impact >= 1:
                impact = ImpactLevel.LOW
            else:
                impact = ImpactLevel.NONE

            return EconomicEvent(
                event_id=str(data.get("id", "")),
                datetime_utc=dt,
                currency=data.get("currency", ""),
                event_name=data.get("title", ""),
                impact=impact,
                actual=data.get("actual"),
                forecast=data.get("forecast"),
                previous=data.get("previous"),
                source=CalendarSource.OANDA_LABS,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OANDA event: {e}")
            return None

    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get events from OANDA Labs API."""
        params = {
            "start": int(start_date.timestamp()),
            "end": int(end_date.timestamp()),
        }

        if currencies:
            params["currency"] = ",".join(currencies)

        data = self._make_request("events", params)
        if not data:
            return []

        events = []
        for item in data.get("events", []):
            event = self._parse_event(item)
            if event:
                if self.config.high_impact_only and event.impact != ImpactLevel.HIGH:
                    continue
                events.append(event)

        return sorted(events, key=lambda e: e.datetime_utc)

    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """Get upcoming events."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        return self.get_events(now, end, currencies)


# =============================================================================
# FOREXFACTORY CALENDAR PROVIDER (Scraping)
# =============================================================================

class ForexFactoryProvider(CalendarProvider):
    """
    Calendar provider using ForexFactory web scraping.

    Fallback when OANDA API is unavailable.
    Note: Web scraping may break if site structure changes.
    """

    def __init__(self, config: Optional[CalendarConfig] = None):
        """Initialize with configuration."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for ForexFactoryProvider")
        if not HAS_BS4:
            raise ImportError("beautifulsoup4 library required for ForexFactoryProvider")

        self.config = config or CalendarConfig()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })

    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """
        Get events by scraping ForexFactory.

        Note: This is a simplified implementation.
        ForexFactory structure may change.
        """
        events: List[EconomicEvent] = []

        # ForexFactory uses date format in URL
        current = start_date.date()
        while current <= end_date.date():
            day_events = self._fetch_day(current)
            events.extend(day_events)
            current += timedelta(days=1)

        # Filter by currency
        if currencies:
            currencies_upper = [c.upper() for c in currencies]
            events = [e for e in events if e.currency.upper() in currencies_upper]

        # Filter by date range (exact)
        events = [e for e in events
                  if start_date <= e.datetime_utc <= end_date]

        # Filter by impact
        if self.config.high_impact_only:
            events = [e for e in events if e.impact == ImpactLevel.HIGH]

        return sorted(events, key=lambda e: e.datetime_utc)

    def _fetch_day(self, date: datetime.date) -> List[EconomicEvent]:
        """Fetch events for a single day."""
        # ForexFactory URL format
        date_str = date.strftime("%b%d.%Y").lower()
        url = f"{FOREXFACTORY_URL}?day={date_str}"

        try:
            response = self._session.get(url, timeout=self.config.timeout_seconds)
            if response.status_code != 200:
                return []

            # Parse HTML (simplified - actual parsing depends on current site structure)
            # This is a placeholder - real implementation would parse the table
            return []

        except Exception as e:
            logger.warning(f"ForexFactory fetch failed for {date}: {e}")
            return []

    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """Get upcoming events."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        return self.get_events(now, end, currencies)


# =============================================================================
# ECONOMIC CALENDAR (Main Interface)
# =============================================================================

class EconomicCalendar:
    """
    Main interface for economic calendar data.

    Provides unified access to economic events from multiple sources
    with caching and fallback support.

    Usage:
        calendar = EconomicCalendar(source="oanda_labs")
        events = calendar.get_upcoming_events(["EUR", "USD"], hours_ahead=24)

        for event in events:
            if event.impact == ImpactLevel.HIGH:
                print(f"{event.datetime_utc}: {event.event_name}")
    """

    def __init__(
        self,
        source: Union[str, CalendarSource] = CalendarSource.CACHED,
        config: Optional[CalendarConfig] = None,
    ):
        """
        Initialize economic calendar.

        Args:
            source: Data source (oanda_labs, forexfactory, cached)
            config: Calendar configuration
        """
        if isinstance(source, str):
            source = CalendarSource(source)

        self.config = config or CalendarConfig(source=source)
        self._providers: Dict[CalendarSource, CalendarProvider] = {}
        self._cache = CachedCalendarProvider(self.config)
        self._last_refresh: Optional[datetime] = None

        # Initialize primary provider
        self._init_provider(source)

    def _init_provider(self, source: CalendarSource) -> None:
        """Initialize provider for source."""
        if source == CalendarSource.CACHED:
            self._providers[source] = self._cache
        elif source == CalendarSource.OANDA_LABS:
            try:
                self._providers[source] = OandaCalendarProvider(self.config)
            except ImportError:
                logger.warning("OANDA provider unavailable, falling back to cache")
                self._providers[source] = self._cache
        elif source == CalendarSource.FOREXFACTORY:
            try:
                self._providers[source] = ForexFactoryProvider(self.config)
            except ImportError:
                logger.warning("ForexFactory provider unavailable, falling back to cache")
                self._providers[source] = self._cache

    @property
    def primary_provider(self) -> CalendarProvider:
        """Get primary provider."""
        return self._providers.get(self.config.source, self._cache)

    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """
        Get economic events in date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            currencies: Filter by currencies (None = all)

        Returns:
            List of EconomicEvent objects sorted by datetime
        """
        try:
            events = self.primary_provider.get_events(start_date, end_date, currencies)

            # Update cache
            if self.config.source != CalendarSource.CACHED and events:
                self._cache.add_events(events)

            return events

        except Exception as e:
            logger.warning(f"Primary provider failed: {e}, using cache")
            return self._cache.get_events(start_date, end_date, currencies)

    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[EconomicEvent]:
        """
        Get upcoming high-impact events.

        Args:
            currencies: Currency codes to filter
            hours_ahead: Hours to look ahead

        Returns:
            List of upcoming events sorted by datetime
        """
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        return self.get_events(now, end, currencies)

    def hours_to_next_high_impact(
        self,
        currency: str,
        current_ts: Optional[Union[int, float, datetime]] = None,
    ) -> Tuple[float, str, int]:
        """
        Get hours until next high-impact event.

        Args:
            currency: Currency code
            current_ts: Current timestamp in ms (int/float), datetime, or None for now

        Returns:
            Tuple of (hours, event_name, impact_level)
        """
        if current_ts is None:
            now = datetime.now(timezone.utc)
        elif isinstance(current_ts, datetime):
            now = current_ts if current_ts.tzinfo else current_ts.replace(tzinfo=timezone.utc)
        else:
            # Assume milliseconds timestamp
            now = datetime.fromtimestamp(current_ts / 1000.0, tz=timezone.utc)

        # Look ahead 7 days
        end = now + timedelta(days=7)

        events = self.get_events(now, end, [currency])
        high_impact = [e for e in events if e.impact == ImpactLevel.HIGH]

        if not high_impact:
            return (999.0, "", 0)

        next_event = high_impact[0]
        delta = next_event.datetime_utc - now
        hours = delta.total_seconds() / 3600.0

        return (hours, next_event.event_name, next_event.impact.value)

    def is_in_news_window(
        self,
        currencies: List[str],
        current_ts: Optional[int] = None,
        window_hours: float = 2.0,
    ) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        Check if current time is within news window.

        Args:
            currencies: Currency codes to check
            current_ts: Current timestamp in ms
            window_hours: Hours before/after high-impact event

        Returns:
            Tuple of (is_in_window, closest_event)
        """
        if current_ts is not None:
            now = datetime.fromtimestamp(current_ts / 1000.0, tz=timezone.utc)
        else:
            now = datetime.now(timezone.utc)

        start = now - timedelta(hours=window_hours)
        end = now + timedelta(hours=window_hours)

        events = self.get_events(start, end, currencies)
        high_impact = [e for e in events if e.impact.value >= 2]

        if not high_impact:
            return (False, None)

        # Find closest event
        closest = min(high_impact, key=lambda e: abs((e.datetime_utc - now).total_seconds()))
        return (True, closest)

    def refresh(self) -> None:
        """Refresh cache from primary source."""
        if self.config.source == CalendarSource.CACHED:
            return

        now = datetime.now(timezone.utc)

        # Fetch next 30 days
        end = now + timedelta(days=30)
        events = self.primary_provider.get_events(now, end)

        if events:
            self._cache.add_events(events)
            self._cache.save_cache()
            self._last_refresh = now

    def save_cache(self) -> None:
        """Save cache to disk."""
        self._cache.save_cache()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def classify_event_impact(event_name: str, currency: str) -> ImpactLevel:
    """
    Classify event impact based on name and currency.

    Args:
        event_name: Event description
        currency: Currency code

    Returns:
        ImpactLevel classification
    """
    event_upper = event_name.upper()
    currency_keywords = HIGH_IMPACT_KEYWORDS.get(currency.upper(), [])

    for keyword in currency_keywords:
        if keyword.upper() in event_upper:
            return ImpactLevel.HIGH

    # Check for generic high-impact terms
    high_impact_generic = [
        "CENTRAL BANK", "RATE DECISION", "INTEREST RATE",
        "EMPLOYMENT", "INFLATION", "CPI", "GDP",
    ]
    for keyword in high_impact_generic:
        if keyword in event_upper:
            return ImpactLevel.MEDIUM

    return ImpactLevel.LOW


def create_calendar_from_dataframe(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    currency_col: str = "currency",
    event_col: str = "event",
    impact_col: Optional[str] = "impact",
) -> List[EconomicEvent]:
    """
    Create EconomicEvent list from DataFrame.

    Args:
        df: DataFrame with calendar data
        datetime_col: Column name for datetime
        currency_col: Column name for currency
        event_col: Column name for event description
        impact_col: Column name for impact (optional)

    Returns:
        List of EconomicEvent objects
    """
    events = []

    for idx, row in df.iterrows():
        try:
            dt = row[datetime_col]
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            elif isinstance(dt, (int, float)):
                dt = datetime.fromtimestamp(dt / 1000.0, tz=timezone.utc)

            currency = str(row[currency_col])
            event_name = str(row[event_col])

            if impact_col and impact_col in row:
                impact = ImpactLevel(int(row[impact_col]))
            else:
                impact = classify_event_impact(event_name, currency)

            event = EconomicEvent(
                event_id=f"{currency}_{idx}",
                datetime_utc=dt,
                currency=currency,
                event_name=event_name,
                impact=impact,
                source=CalendarSource.CACHED,
            )
            events.append(event)

        except Exception as e:
            logger.warning(f"Failed to parse row {idx}: {e}")
            continue

    return sorted(events, key=lambda e: e.datetime_utc)
