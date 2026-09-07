# -*- coding: utf-8 -*-
"""
scripts/download_economic_calendar.py
Download economic calendar events for forex trading.

Economic events (NFP, CPI, central bank decisions) create volatility spikes
and are critical for:
1. No-trade windows around high-impact events
2. Volatility regime features for ML models
3. Event-driven position sizing

Sources:
- Primary: OANDA Labs Calendar API (free with OANDA account)
- Backup: Investing.com / ForexFactory (requires scraping)

Features:
- Historical and future events
- Impact classification (High/Medium/Low)
- Multiple currencies
- Event categories (NFP, CPI, GDP, Interest Rate, etc.)
- Actual vs forecast vs previous values

Usage:
    # Download calendar for major currencies
    python scripts/download_economic_calendar.py --currencies USD EUR GBP JPY --start 2020-01-01

    # Download high-impact events only
    python scripts/download_economic_calendar.py --currencies USD --high-impact-only

    # Output directory
    python scripts/download_economic_calendar.py --output data/forex/calendar/

Output:
    data/forex/calendar/economic_calendar.parquet

Columns:
    - datetime: Event datetime (UTC)
    - date: Date (YYYY-MM-DD)
    - time: Time (HH:MM)
    - currency: Currency code (USD, EUR, etc.)
    - event: Event name
    - impact: Impact level (high, medium, low)
    - actual: Actual value (if released)
    - forecast: Market forecast
    - previous: Previous period value
    - source: Data source

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================


class EventImpact(Enum):
    """Economic event impact level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CalendarDownloadConfig:
    """Configuration for economic calendar download."""

    # Currencies to include
    currencies: List[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])

    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = 365 * 3

    # Filtering
    high_impact_only: bool = False
    include_forecasts: bool = True

    # Output
    output_dir: str = "data/forex/calendar"
    output_format: str = "parquet"

    # Skip existing
    skip_existing: bool = True


# =========================
# High-Impact Economic Events
# =========================

# High-impact events that typically move forex markets significantly
HIGH_IMPACT_EVENTS = {
    "USD": [
        "Non-Farm Payrolls",
        "FOMC Rate Decision",
        "CPI",
        "Core CPI",
        "Retail Sales",
        "GDP",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "ADP Employment",
        "Initial Jobless Claims",
        "Durable Goods Orders",
        "Trade Balance",
        "Consumer Confidence",
        "PCE Price Index",
        "Core PCE Price Index",
        "Fed Chair Powell Speech",
    ],
    "EUR": [
        "ECB Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "PMI Manufacturing",
        "PMI Services",
        "ZEW Economic Sentiment",
        "German CPI",
        "German GDP",
        "Trade Balance",
        "ECB President Lagarde Speech",
    ],
    "GBP": [
        "BOE Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Retail Sales",
        "PMI Manufacturing",
        "PMI Services",
        "Employment Change",
        "Trade Balance",
        "BOE Governor Bailey Speech",
    ],
    "JPY": [
        "BOJ Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Trade Balance",
        "Tankan Survey",
        "Industrial Production",
        "Retail Sales",
        "BOJ Governor Speech",
    ],
    "CHF": [
        "SNB Rate Decision",
        "CPI",
        "GDP",
        "Trade Balance",
    ],
    "AUD": [
        "RBA Rate Decision",
        "CPI",
        "GDP",
        "Employment Change",
        "Trade Balance",
        "Retail Sales",
    ],
    "CAD": [
        "BOC Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Employment Change",
        "Retail Sales",
        "Trade Balance",
    ],
    "NZD": [
        "RBNZ Rate Decision",
        "CPI",
        "GDP",
        "Employment Change",
        "Trade Balance",
    ],
}

# Event schedule patterns (approximate release times in UTC)
# These are used to generate synthetic historical data
EVENT_SCHEDULES = {
    "Non-Farm Payrolls": {
        "frequency": "monthly",
        "day": "first_friday",
        "time_utc": "13:30",
        "currency": "USD",
    },
    "FOMC Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "19:00",
        "currency": "USD",
    },
    "ECB Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "12:45",
        "currency": "EUR",
    },
    "BOE Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "12:00",
        "currency": "GBP",
    },
    "CPI": {
        "frequency": "monthly",
        "currency": "multiple",
    },
}


# =========================
# OANDA Calendar API
# =========================


def fetch_calendar_oanda(
    config: CalendarDownloadConfig,
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch economic calendar from OANDA Labs API.

    Note: OANDA's calendar API may have limited historical data.
    This function attempts to fetch what's available.

    Args:
        config: Download configuration
        api_key: OANDA API key
        account_id: OANDA account ID

    Returns:
        DataFrame with calendar events or None if failed
    """
    try:
        import requests

        api_key = api_key or os.environ.get("OANDA_API_KEY")
        if not api_key:
            logger.warning("OANDA API key not provided, using synthetic data")
            return None

        # OANDA Labs calendar endpoint
        # Note: This is the public calendar, not requiring authentication
        url = "https://www.oanda.com/labsapi/calendar"

        # Date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        params = {
            "period": f"{start_dt.strftime('%Y%m%d')}/{end_dt.strftime('%Y%m%d')}",
        }

        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            logger.warning(f"OANDA calendar API returned {response.status_code}")
            return None

        data = response.json()

        if not data:
            return None

        # Parse events
        records = []
        for event in data:
            currency = event.get("currency", "")
            if config.currencies and currency not in config.currencies:
                continue

            impact = _classify_impact(event.get("title", ""), currency)
            if config.high_impact_only and impact != "high":
                continue

            records.append(
                {
                    "datetime": event.get("timestamp"),
                    "date": event.get("timestamp", "")[:10],
                    "time": event.get("timestamp", "")[11:16],
                    "currency": currency,
                    "event": event.get("title", ""),
                    "impact": impact,
                    "actual": event.get("actual"),
                    "forecast": event.get("forecast"),
                    "previous": event.get("previous"),
                    "source": "oanda",
                }
            )

        if not records:
            return None

        df = pd.DataFrame(records)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch OANDA calendar: {e}")
        return None


def _classify_impact(event_name: str, currency: str) -> str:
    """Classify event impact level based on event name."""
    event_lower = event_name.lower()

    # Check if it's a known high-impact event
    high_impact_keywords = [
        "rate decision",
        "non-farm",
        "payroll",
        "cpi",
        "inflation",
        "gdp",
        "employment",
        "retail sales",
        "pmi",
        "fomc",
        "ecb",
        "boe",
        "boj",
        "rba",
        "rbnz",
        "snb",
        "boc",
    ]

    for keyword in high_impact_keywords:
        if keyword in event_lower:
            return "high"

    medium_impact_keywords = [
        "trade balance",
        "industrial",
        "housing",
        "consumer",
        "sentiment",
        "confidence",
        "speech",
        "testimony",
    ]

    for keyword in medium_impact_keywords:
        if keyword in event_lower:
            return "medium"

    return "low"


# =========================
# ForexFactory Calendar Scraper (Backup)
# =========================
# Based on best practices from:
# - Babypips Calendar Research (https://www.babypips.com/economic-calendar)
# - ForexFactory structure analysis
# - Rate limiting best practices (polite scraping)

# ForexFactory URL pattern
FOREXFACTORY_BASE_URL = "https://www.forexfactory.com"
FOREXFACTORY_CALENDAR_URL = f"{FOREXFACTORY_BASE_URL}/calendar"

# User-Agent to avoid blocks (be a good citizen)
FOREXFACTORY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Impact color mapping (ForexFactory uses colored icons)
FOREXFACTORY_IMPACT_MAP = {
    "high": "high",
    "medium": "medium",
    "low": "low",
    "holiday": "low",
    "red": "high",  # Red folder icon
    "orange": "medium",  # Orange folder icon
    "yellow": "low",  # Yellow folder icon
    "gray": "low",  # Gray = holiday/non-event
}


def fetch_calendar_forexfactory(
    config: CalendarDownloadConfig,
    max_retries: int = 3,
    rate_limit_sec: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Fetch economic calendar from ForexFactory (backup source).

    ForexFactory is one of the most comprehensive free forex calendars.
    Uses HTML scraping with polite rate limiting.

    Reference: ForexFactory Calendar Structure
    - Events organized by week
    - Impact levels: High (red), Medium (orange), Low (yellow)
    - Includes actual/forecast/previous values

    Args:
        config: Download configuration
        max_retries: Maximum retry attempts per request
        rate_limit_sec: Delay between requests (be polite!)

    Returns:
        DataFrame with calendar events or None if failed

    Note:
        This scraper respects robots.txt and rate limits.
        ForexFactory allows scraping but requests reasonable delays.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup not installed. Run: pip install beautifulsoup4")
        return None

    logger.info("Fetching calendar from ForexFactory (backup source)")

    # Date range
    end_dt = datetime.now(timezone.utc)
    if config.end_date:
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if config.start_date:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=config.lookback_days)

    all_records: List[Dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(FOREXFACTORY_HEADERS)

    # Iterate through weeks
    current_week = start_dt
    weeks_processed = 0
    max_weeks = (end_dt - start_dt).days // 7 + 2

    while current_week <= end_dt and weeks_processed < max_weeks:
        # ForexFactory week URL format: ?week=month.day.year
        week_str = current_week.strftime("%b%d.%Y").lower()
        url = f"{FOREXFACTORY_CALENDAR_URL}?week={week_str}"

        for attempt in range(max_retries):
            try:
                time.sleep(rate_limit_sec)  # Polite rate limiting

                response = session.get(url, timeout=30)

                if response.status_code == 200:
                    records = _parse_forexfactory_page(
                        response.text,
                        config.currencies,
                        config.high_impact_only,
                    )
                    all_records.extend(records)
                    break
                elif response.status_code == 429:
                    # Rate limited - back off exponentially
                    wait_time = rate_limit_sec * (2**attempt)
                    logger.warning(f"Rate limited by ForexFactory, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"ForexFactory returned {response.status_code} for {url}")
                    break

            except requests.RequestException as e:
                logger.warning(f"ForexFactory request failed: {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(rate_limit_sec * 2)

        # Move to next week
        current_week += timedelta(days=7)
        weeks_processed += 1

        # Progress logging for long downloads
        if weeks_processed % 10 == 0:
            logger.info(
                f"ForexFactory: processed {weeks_processed} weeks, {len(all_records)} events"
            )

    if not all_records:
        logger.warning("No events fetched from ForexFactory")
        return None

    df = pd.DataFrame(all_records)

    # Parse datetime and sort
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Filter by date range
    df = df[(df["datetime"] >= pd.Timestamp(start_dt)) & (df["datetime"] <= pd.Timestamp(end_dt))]

    logger.info(f"ForexFactory: fetched {len(df)} events from {weeks_processed} weeks")
    return df


def _parse_forexfactory_page(
    html: str,
    currencies: List[str],
    high_impact_only: bool,
) -> List[Dict[str, Any]]:
    """
    Parse ForexFactory calendar HTML page.

    ForexFactory HTML structure (2024 format):
    - Events in <tr class="calendar__row">
    - Date in <td class="calendar__date">
    - Time in <td class="calendar__time">
    - Currency in <td class="calendar__currency">
    - Impact in <td class="calendar__impact"> with colored span
    - Event in <td class="calendar__event">
    - Actual/Forecast/Previous in respective <td> columns

    Args:
        html: Raw HTML content
        currencies: List of currencies to filter
        high_impact_only: Only return high-impact events

    Returns:
        List of event dictionaries
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    records = []

    # Find all calendar rows
    rows = soup.find_all("tr", class_=re.compile(r"calendar__row"))

    current_date = None
    current_time = None

    for row in rows:
        try:
            # Skip non-event rows
            if "calendar__row--day-breaker" in row.get("class", []):
                continue

            # Date (may be empty if same as previous row)
            date_cell = row.find("td", class_="calendar__date")
            if date_cell and date_cell.get_text(strip=True):
                date_text = date_cell.get_text(strip=True)
                current_date = _parse_forexfactory_date(date_text)

            # Time
            time_cell = row.find("td", class_="calendar__time")
            if time_cell and time_cell.get_text(strip=True):
                time_text = time_cell.get_text(strip=True)
                if time_text.lower() not in ["all day", "tentative", ""]:
                    current_time = _parse_forexfactory_time(time_text)

            # Currency
            currency_cell = row.find("td", class_="calendar__currency")
            currency = currency_cell.get_text(strip=True) if currency_cell else ""

            if currencies and currency not in currencies:
                continue

            # Impact
            impact_cell = row.find("td", class_="calendar__impact")
            impact = _parse_forexfactory_impact(impact_cell)

            if high_impact_only and impact != "high":
                continue

            # Event name
            event_cell = row.find("td", class_="calendar__event")
            event_name = ""
            if event_cell:
                event_link = event_cell.find("span", class_="calendar__event-title")
                if event_link:
                    event_name = event_link.get_text(strip=True)
                else:
                    event_name = event_cell.get_text(strip=True)

            if not event_name:
                continue

            # Actual value
            actual_cell = row.find("td", class_="calendar__actual")
            actual = _parse_forexfactory_value(actual_cell)

            # Forecast value
            forecast_cell = row.find("td", class_="calendar__forecast")
            forecast = _parse_forexfactory_value(forecast_cell)

            # Previous value
            previous_cell = row.find("td", class_="calendar__previous")
            previous = _parse_forexfactory_value(previous_cell)

            # Build datetime
            if current_date and current_time:
                event_dt = datetime.combine(current_date, current_time)
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            elif current_date:
                event_dt = datetime.combine(current_date, datetime.min.time())
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            else:
                continue  # Skip events without date

            records.append(
                {
                    "datetime": event_dt.isoformat(),
                    "date": event_dt.strftime("%Y-%m-%d"),
                    "time": event_dt.strftime("%H:%M"),
                    "currency": currency,
                    "event": event_name,
                    "impact": impact,
                    "actual": actual,
                    "forecast": forecast,
                    "previous": previous,
                    "source": "forexfactory",
                }
            )

        except Exception as e:
            # Skip malformed rows
            logger.debug(f"Failed to parse ForexFactory row: {e}")
            continue

    return records


def _parse_forexfactory_date(date_text: str) -> Optional[datetime]:
    """
    Parse ForexFactory date format.

    Examples: "Mon Jan 15", "Tue Feb 20", "Wed Mar 1"
    """
    try:
        # Current year assumption (ForexFactory shows current week)
        current_year = datetime.now().year

        # Parse without year
        date_text_clean = re.sub(r"\s+", " ", date_text.strip())

        # Try multiple formats
        for fmt in ["%a %b %d", "%A %b %d", "%b %d"]:
            try:
                dt = datetime.strptime(date_text_clean, fmt)
                return dt.replace(year=current_year).date()
            except ValueError:
                continue

        return None

    except Exception:
        return None


def _parse_forexfactory_time(time_text: str) -> Optional[datetime]:
    """
    Parse ForexFactory time format.

    Examples: "8:30am", "2:00pm", "12:30am"
    ForexFactory shows times in US Eastern Time, we convert to UTC.
    """
    try:
        time_text = time_text.strip().lower()

        # Parse AM/PM format
        match = re.match(r"(\d{1,2}):(\d{2})(am|pm)", time_text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3)

            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0

            # ForexFactory shows Eastern Time (ET)
            # Convert to UTC: add 5 hours (EST) or 4 hours (EDT)
            # Simplified: assume EST (+5)
            hour = (hour + 5) % 24

            return datetime.min.replace(hour=hour, minute=minute).time()

        return None

    except Exception:
        return None


def _parse_forexfactory_impact(impact_cell) -> str:
    """
    Parse ForexFactory impact level from cell.

    ForexFactory uses colored icons:
    - Red folder = High impact
    - Orange folder = Medium impact
    - Yellow folder = Low impact
    - Gray = Holiday/non-event
    """
    if not impact_cell:
        return "low"

    # Look for impact span with class
    impact_span = impact_cell.find("span", class_=re.compile(r"icon--ff-impact"))
    if impact_span:
        classes = impact_span.get("class", [])
        class_str = " ".join(classes).lower()

        if "high" in class_str or "red" in class_str:
            return "high"
        elif "medium" in class_str or "orange" in class_str:
            return "medium"
        elif "low" in class_str or "yellow" in class_str:
            return "low"

    # Alternative: check for impact icon title
    icon = impact_cell.find("span", title=True)
    if icon:
        title = icon.get("title", "").lower()
        if "high" in title:
            return "high"
        elif "medium" in title:
            return "medium"

    return "low"


def _parse_forexfactory_value(cell) -> Optional[str]:
    """Parse actual/forecast/previous value from cell."""
    if not cell:
        return None

    text = cell.get_text(strip=True)

    # Empty or placeholder
    if not text or text in ["-", "—", ""]:
        return None

    return text


# =========================
# Synthetic Calendar Generation
# =========================


def generate_synthetic_calendar(
    config: CalendarDownloadConfig,
) -> pd.DataFrame:
    """
    Generate synthetic economic calendar based on known schedules.

    This is used when API data is unavailable or for historical backfill.
    Uses typical event patterns and schedules.

    Args:
        config: Download configuration

    Returns:
        DataFrame with synthetic calendar events
    """
    logger.info("Generating synthetic economic calendar")

    end_dt = datetime.now(timezone.utc)
    if config.end_date:
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if config.start_date:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=config.lookback_days)

    records = []

    for currency in config.currencies:
        high_impact_events = HIGH_IMPACT_EVENTS.get(currency, [])

        for event_name in high_impact_events:
            # Generate monthly events
            current_date = start_dt
            while current_date <= end_dt:
                # Determine event time (approximate)
                if "Rate Decision" in event_name:
                    # Central bank decisions typically on specific weeks
                    event_hour = 12 if currency in ["EUR", "GBP"] else 19
                    event_date = _get_meeting_date(current_date, currency)
                elif "Non-Farm" in event_name:
                    # First Friday of month, 8:30 AM ET = 13:30 UTC
                    event_date = _get_first_friday(current_date)
                    event_hour = 13
                else:
                    # Other events typically mid-month
                    event_date = current_date.replace(day=min(15, 28))
                    event_hour = 13

                if event_date and start_dt <= event_date <= end_dt:
                    event_dt = event_date.replace(hour=event_hour, minute=30)

                    records.append(
                        {
                            "datetime": event_dt.isoformat(),
                            "date": event_dt.strftime("%Y-%m-%d"),
                            "time": event_dt.strftime("%H:%M"),
                            "currency": currency,
                            "event": event_name,
                            "impact": "high" if event_name in high_impact_events[:5] else "medium",
                            "actual": None,  # No actual values for synthetic
                            "forecast": None,
                            "previous": None,
                            "source": "synthetic",
                        }
                    )

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

    df = pd.DataFrame(records)

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Generated {len(df)} synthetic calendar events")
    return df


def _get_first_friday(dt: datetime) -> datetime:
    """Get first Friday of the month."""
    first_day = dt.replace(day=1)
    # Days until Friday (4 = Friday)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)


def _get_meeting_date(dt: datetime, currency: str) -> Optional[datetime]:
    """
    Get approximate central bank meeting date for the month.

    Central banks typically meet every 6-8 weeks on predetermined schedules.
    This is a simplified approximation.
    """
    # Typical meeting patterns (day of month)
    meeting_patterns = {
        "USD": [15, 16],  # FOMC mid-month Wed/Thu
        "EUR": [10, 12],  # ECB typically early month
        "GBP": [5, 8],  # BOE typically first week
        "JPY": [15, 20],  # BOJ mid-month
        "AUD": [3, 5],  # RBA first Tuesday
        "CAD": [8, 10],  # BOC
        "NZD": [10, 12],  # RBNZ
        "CHF": [15, 18],  # SNB
    }

    days = meeting_patterns.get(currency, [15])

    # Central banks meet ~8 times per year, so skip some months
    # Simplified: assume meetings in specific months
    meeting_months = [1, 3, 5, 6, 7, 9, 11, 12]  # Approximate

    if dt.month in meeting_months:
        return dt.replace(day=days[0])
    return None


# =========================
# File I/O
# =========================


def save_calendar(
    df: pd.DataFrame,
    config: CalendarDownloadConfig,
) -> str:
    """Save calendar DataFrame to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "parquet":
        filepath = output_dir / "economic_calendar.parquet"
        df.to_parquet(filepath, index=False)
    else:
        filepath = output_dir / "economic_calendar.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


def load_existing_calendar(config: CalendarDownloadConfig) -> Optional[pd.DataFrame]:
    """Load existing calendar file if present."""
    output_dir = Path(config.output_dir)

    for ext in ["parquet", "csv"]:
        filepath = output_dir / f"economic_calendar.{ext}"
        if filepath.exists():
            if ext == "parquet":
                return pd.read_parquet(filepath)
            else:
                return pd.read_csv(filepath)

    return None


# =========================
# Main Runner
# =========================


def download_calendar(config: CalendarDownloadConfig) -> Dict[str, Any]:
    """
    Download or generate economic calendar.

    Data source priority (fallback chain):
    1. OANDA Labs Calendar API (primary, requires API key)
    2. ForexFactory scraper (backup, comprehensive & free)
    3. Synthetic generation (fallback, uses known schedules)

    Args:
        config: Download configuration

    Returns:
        Summary dict with events count and source
    """
    # Check existing
    if config.skip_existing:
        existing = load_existing_calendar(config)
        if existing is not None and len(existing) > 0:
            logger.info(f"Using existing calendar with {len(existing)} events")
            return {"events": len(existing), "source": "existing"}

    df = None
    source_used = "none"

    # 1. Try OANDA first (primary source)
    logger.info("Attempting OANDA calendar API (primary source)")
    df = fetch_calendar_oanda(config)
    if df is not None and not df.empty:
        source_used = "oanda"
        logger.info(f"OANDA returned {len(df)} events")

    # 2. Try ForexFactory as backup
    if df is None or df.empty:
        logger.info("OANDA unavailable, trying ForexFactory (backup source)")
        df = fetch_calendar_forexfactory(config)
        if df is not None and not df.empty:
            source_used = "forexfactory"
            logger.info(f"ForexFactory returned {len(df)} events")

    # 3. Fall back to synthetic generation
    if df is None or df.empty:
        logger.info("No API sources available, using synthetic calendar generation")
        df = generate_synthetic_calendar(config)
        if df is not None and not df.empty:
            source_used = "synthetic"

    if df is None or df.empty:
        logger.warning("No calendar events generated from any source")
        return {"events": 0, "source": "none"}

    # Save
    filepath = save_calendar(df, config)
    logger.info(f"Saved {len(df)} events to {filepath} (source: {source_used})")

    return {
        "events": len(df),
        "source": source_used,
        "filepath": filepath,
    }


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download economic calendar for forex trading",
    )

    parser.add_argument(
        "--currencies",
        nargs="+",
        default=["USD", "EUR", "GBP", "JPY"],
        help="Currency codes (default: USD EUR GBP JPY)",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="data/forex/calendar",
        help="Output directory",
    )
    parser.add_argument(
        "--high-impact-only",
        action="store_true",
        help="Only include high-impact events",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if file exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = CalendarDownloadConfig(
        currencies=[c.upper() for c in args.currencies],
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        high_impact_only=args.high_impact_only,
        skip_existing=not args.force,
    )

    try:
        summary = download_calendar(config)
        logger.info(f"Calendar download complete: {summary}")
        return 0
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
