# -*- coding: utf-8 -*-
"""
cot_data_loader.py
CFTC Commitments of Traders (COT) data loader for forex trading (Phase 4).

The COT report shows weekly positioning of large speculators, commercial
hedgers, and small traders in futures markets. Currency futures positioning
can provide insights into sentiment and potential reversals.

Data Source:
- CFTC: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- Historical: Quandl/Nasdaq Data Link (now paid)
- Backup: OpenCOT or cached data

COT Report Types:
- Legacy Report: Commercial vs Non-Commercial
- Disaggregated Report: More granular breakdown
- Traders in Financial Futures (TFF): For currency futures

Currency Futures Contracts:
- CME: EUR/USD, GBP/USD, JPY/USD, CHF/USD, AUD/USD, CAD/USD, NZD/USD
- Note: Contracts are quoted as X/USD except JPY which is USD/JPY

Key Metrics:
- Net Long = Long - Short positions
- Net Long % = Net Long / (Long + Short)
- Change in Net = Current Net - Previous Net
- Z-score = (Current Net - Mean) / Std

References:
- Klitgaard & Weir (2004): "Exchange Rate Changes and Net Positions"
- CFTC Commitments of Traders Explanation

Author: Sultan Suyunbayev
Date: 2025-11-30
Version: 1.0.0
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# CFTC data URLs
CFTC_BASE_URL = "https://www.cftc.gov/files/dea/history"

# COT report type files
COT_FILES = {
    "legacy_futures": "deacot{year}.zip",  # Legacy Futures Only
    "legacy_combined": "dea_com_txt_{year}.zip",  # Legacy Combined
    "disaggregated": "f_year{year}.zip",  # Disaggregated
    "tff": "tff_year{year}.zip",  # Traders in Financial Futures
}

# Currency futures contract codes (CFTC)
# Format: CFTC Market Code -> Currency
CFTC_CURRENCY_CODES = {
    "099741": "EUR",  # Euro FX
    "096742": "GBP",  # British Pound
    "097741": "JPY",  # Japanese Yen
    "092741": "CHF",  # Swiss Franc
    "232741": "AUD",  # Australian Dollar
    "090741": "CAD",  # Canadian Dollar
    "112741": "NZD",  # New Zealand Dollar
    "098662": "BRL",  # Brazilian Real
    "095741": "MXN",  # Mexican Peso
    "096661": "ZAR",  # South African Rand
}

# Reverse mapping
CURRENCY_TO_CFTC = {v: k for k, v in CFTC_CURRENCY_CODES.items()}

# Contract to pair mapping (XXX/USD convention)
CURRENCY_TO_PAIR = {
    "EUR": "EUR_USD",
    "GBP": "GBP_USD",
    "JPY": "USD_JPY",  # Note: JPY is inverted
    "CHF": "USD_CHF",  # Note: CHF is inverted
    "AUD": "AUD_USD",
    "CAD": "USD_CAD",  # Note: CAD is inverted
    "NZD": "NZD_USD",
}

# Pairs where position should be inverted for interpretation
INVERTED_PAIRS = {"USD_JPY", "USD_CHF", "USD_CAD"}

# Default cache directory
DEFAULT_COT_CACHE_DIR = "data/cot_cache"

# Column mappings for different COT report types
LEGACY_COLUMNS = {
    "market_code": "CFTC Market Code in Initials",
    "report_date": "As of Date in Form YYYY-MM-DD",
    "noncomm_long": "Noncommercial Positions-Long (All)",
    "noncomm_short": "Noncommercial Positions-Short (All)",
    "comm_long": "Commercial Positions-Long (All)",
    "comm_short": "Commercial Positions-Short (All)",
    "total_long": "Total Long (All)",
    "total_short": "Total Short (All)",
    "open_interest": "Open Interest (All)",
}

TFF_COLUMNS = {
    "market_code": "CFTC Market Code in Initials",
    "report_date": "As of Date in Form YYYY-MM-DD",
    "dealer_long": "Dealer Longs",
    "dealer_short": "Dealer Shorts",
    "asset_mgr_long": "Asset Manager Longs",
    "asset_mgr_short": "Asset Manager Shorts",
    "lev_money_long": "Leveraged Money Longs",
    "lev_money_short": "Leveraged Money Shorts",
    "open_interest": "Open Interest",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class COTPosition:
    """
    COT positioning data for a single report date.

    Attributes:
        currency: Currency code
        report_date: Report date (Tuesday as of)
        net_long: Net long position (contracts)
        net_long_pct: Net long as percentage of open interest
        change_1w: Change from previous week
        change_1w_pct: Percentage change
        open_interest: Total open interest
        long_positions: Total long positions
        short_positions: Total short positions
    """

    currency: str
    report_date: datetime
    net_long: float
    net_long_pct: float = 0.0
    change_1w: float = 0.0
    change_1w_pct: float = 0.0
    open_interest: float = 0.0
    long_positions: float = 0.0
    short_positions: float = 0.0
    source: str = "cftc"

    def __post_init__(self):
        if self.report_date.tzinfo is None:
            self.report_date = self.report_date.replace(tzinfo=timezone.utc)

    @property
    def timestamp_ms(self) -> int:
        """Get timestamp in milliseconds."""
        return int(self.report_date.timestamp() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "currency": self.currency,
            "report_date": self.report_date.isoformat(),
            "net_long": self.net_long,
            "net_long_pct": self.net_long_pct,
            "change_1w": self.change_1w,
            "change_1w_pct": self.change_1w_pct,
            "open_interest": self.open_interest,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "COTPosition":
        """Create from dictionary."""
        report_date = data.get("report_date")
        if isinstance(report_date, str):
            report_date = datetime.fromisoformat(report_date.replace("Z", "+00:00"))

        return cls(
            currency=data["currency"],
            report_date=report_date,
            net_long=data["net_long"],
            net_long_pct=data.get("net_long_pct", 0.0),
            change_1w=data.get("change_1w", 0.0),
            change_1w_pct=data.get("change_1w_pct", 0.0),
            open_interest=data.get("open_interest", 0.0),
            long_positions=data.get("long_positions", 0.0),
            short_positions=data.get("short_positions", 0.0),
            source=data.get("source", "cftc"),
        )


@dataclass
class COTConfig:
    """Configuration for COT data loading."""

    cache_dir: str = DEFAULT_COT_CACHE_DIR
    report_type: str = "legacy"  # "legacy", "tff", "disaggregated"
    use_noncommercial: bool = True  # Use non-commercial (speculator) positions
    zscore_lookback: int = 52  # Weeks for z-score calculation
    auto_download: bool = False  # Download from CFTC if missing
    timeout_seconds: float = 60.0


# =============================================================================
# COT DATA LOADER
# =============================================================================


class COTDataLoader:
    """
    Load and process CFTC COT data for forex currencies.

    Usage:
        loader = COTDataLoader(config)
        loader.load_data(start_year=2020, end_year=2024)

        # Get positioning for EUR
        eur_pos = loader.get_current_position("EUR")

        # Get DataFrame with all currencies
        df = loader.to_dataframe()
    """

    def __init__(self, config: Optional[COTConfig] = None):
        """Initialize loader."""
        self.config = config or COTConfig()
        self._data: Dict[str, List[COTPosition]] = {}
        self._loaded = False

    def load_data(
        self,
        start_year: int = 2020,
        end_year: Optional[int] = None,
    ) -> None:
        """
        Load COT data for specified years.

        Args:
            start_year: Start year
            end_year: End year (default: current year)
        """
        if end_year is None:
            end_year = datetime.now().year

        # Try to load from cache first
        self._load_from_cache()

        # Download missing years if configured
        if self.config.auto_download and HAS_REQUESTS:
            for year in range(start_year, end_year + 1):
                if not self._has_year_data(year):
                    self._download_year(year)

        self._loaded = True

    def _has_year_data(self, year: int) -> bool:
        """Check if we have data for a specific year."""
        if not self._data:
            return False

        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, tzinfo=timezone.utc)

        for positions in self._data.values():
            for pos in positions:
                if year_start <= pos.report_date <= year_end:
                    return True

        return False

    def _load_from_cache(self) -> None:
        """Load cached COT data."""
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
                    currency = filename.replace(".json", "").upper()
                    positions = [COTPosition.from_dict(p) for p in data]
                    self._data[currency] = sorted(positions, key=lambda p: p.report_date)
            except Exception as e:
                logger.warning(f"Failed to load COT cache {filename}: {e}")

    def save_cache(self) -> None:
        """Save data to cache."""
        cache_dir = self.config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        for currency, positions in self._data.items():
            filepath = os.path.join(cache_dir, f"{currency}.json")
            try:
                with open(filepath, "w") as f:
                    json.dump([p.to_dict() for p in positions], f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save COT cache {currency}: {e}")

    def _download_year(self, year: int) -> None:
        """Download COT data for a year from CFTC."""
        if not HAS_REQUESTS:
            logger.warning("requests library required for COT download")
            return

        # Determine file to download based on report type
        if self.config.report_type == "tff":
            file_template = COT_FILES["tff"]
        else:
            file_template = COT_FILES["legacy_futures"]

        filename = file_template.format(year=year)
        url = f"{CFTC_BASE_URL}/{filename}"

        try:
            logger.info(f"Downloading COT data for {year} from {url}")
            response = requests.get(url, timeout=self.config.timeout_seconds)

            if response.status_code != 200:
                logger.warning(f"Failed to download COT data: {response.status_code}")
                return

            # Parse zip file
            self._parse_cot_zip(response.content, year)

        except Exception as e:
            logger.warning(f"Failed to download COT data for {year}: {e}")

    def _parse_cot_zip(self, content: bytes, year: int) -> None:
        """Parse CFTC zip file and extract currency data."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Find the CSV/TXT file in the archive
                for name in zf.namelist():
                    if name.endswith(".txt") or name.endswith(".csv"):
                        with zf.open(name) as f:
                            text = f.read().decode("utf-8", errors="ignore")
                            self._parse_cot_csv(text)
                        break

        except Exception as e:
            logger.warning(f"Failed to parse COT zip: {e}")

    def _parse_cot_csv(self, content: str) -> None:
        """Parse COT CSV content."""
        try:
            reader = csv.DictReader(io.StringIO(content))

            for row in reader:
                # Check if this is a currency futures row
                market_code = row.get("CFTC Market Code in Initials", "").strip()

                if market_code not in CFTC_CURRENCY_CODES:
                    continue

                currency = CFTC_CURRENCY_CODES[market_code]

                # Parse report date
                date_str = row.get("As of Date in Form YYYY-MM-DD", "")
                if not date_str:
                    continue

                try:
                    report_date = datetime.strptime(date_str, "%Y-%m-%d")
                    report_date = report_date.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                # Parse positions based on report type
                if self.config.use_noncommercial:
                    long_col = "Noncommercial Positions-Long (All)"
                    short_col = "Noncommercial Positions-Short (All)"
                else:
                    long_col = "Commercial Positions-Long (All)"
                    short_col = "Commercial Positions-Short (All)"

                try:
                    long_pos = float(row.get(long_col, "0").replace(",", "") or 0)
                    short_pos = float(row.get(short_col, "0").replace(",", "") or 0)
                    open_interest = float(row.get("Open Interest (All)", "0").replace(",", "") or 0)
                except ValueError:
                    continue

                net_long = long_pos - short_pos

                # Calculate net long percentage
                if open_interest > 0:
                    net_long_pct = net_long / open_interest
                else:
                    net_long_pct = 0.0

                position = COTPosition(
                    currency=currency,
                    report_date=report_date,
                    net_long=net_long,
                    net_long_pct=net_long_pct,
                    open_interest=open_interest,
                    long_positions=long_pos,
                    short_positions=short_pos,
                )

                if currency not in self._data:
                    self._data[currency] = []

                self._data[currency].append(position)

            # Sort by date
            for currency in self._data:
                self._data[currency] = sorted(self._data[currency], key=lambda p: p.report_date)

            # Calculate week-over-week changes
            self._calculate_changes()

        except Exception as e:
            logger.warning(f"Failed to parse COT CSV: {e}")

    def _calculate_changes(self) -> None:
        """Calculate week-over-week changes."""
        for currency, positions in self._data.items():
            for i in range(1, len(positions)):
                prev = positions[i - 1]
                curr = positions[i]

                curr.change_1w = curr.net_long - prev.net_long

                if abs(prev.net_long) > 0:
                    curr.change_1w_pct = curr.change_1w / abs(prev.net_long)
                else:
                    curr.change_1w_pct = 0.0

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load COT data from a DataFrame.

        Expected columns:
        - currency: Currency code
        - report_date: Report date
        - net_long: Net long position
        - open_interest: Open interest (optional)
        """
        for _, row in df.iterrows():
            currency = str(row.get("currency", "")).upper()
            if not currency:
                continue

            report_date = row.get("report_date")
            if isinstance(report_date, str):
                report_date = datetime.fromisoformat(report_date)
            elif isinstance(report_date, pd.Timestamp):
                report_date = report_date.to_pydatetime()

            if report_date.tzinfo is None:
                report_date = report_date.replace(tzinfo=timezone.utc)

            position = COTPosition(
                currency=currency,
                report_date=report_date,
                net_long=float(row.get("net_long", 0)),
                net_long_pct=float(row.get("net_long_pct", 0)),
                open_interest=float(row.get("open_interest", 0)),
                long_positions=float(row.get("long_positions", 0)),
                short_positions=float(row.get("short_positions", 0)),
            )

            if currency not in self._data:
                self._data[currency] = []

            self._data[currency].append(position)

        # Sort and calculate changes
        for currency in self._data:
            self._data[currency] = sorted(self._data[currency], key=lambda p: p.report_date)

        self._calculate_changes()
        self._loaded = True

    def get_current_position(
        self,
        currency: str,
        as_of: Optional[datetime] = None,
    ) -> Optional[COTPosition]:
        """
        Get most recent COT position for a currency.

        Args:
            currency: Currency code (e.g., "EUR")
            as_of: Get position as of this date (None = latest)

        Returns:
            COTPosition or None if unavailable
        """
        currency = currency.upper()
        if currency not in self._data or not self._data[currency]:
            return None

        if as_of is None:
            return self._data[currency][-1]

        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        # Find most recent position before as_of
        positions = [p for p in self._data[currency] if p.report_date <= as_of]
        if not positions:
            return None

        return positions[-1]

    def get_positions(
        self,
        currency: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[COTPosition]:
        """
        Get COT positions for a currency within date range.

        Args:
            currency: Currency code
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of COTPosition objects
        """
        currency = currency.upper()
        if currency not in self._data:
            return []

        positions = self._data[currency]

        if start_date is not None:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            positions = [p for p in positions if p.report_date >= start_date]

        if end_date is not None:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            positions = [p for p in positions if p.report_date <= end_date]

        return positions

    def calculate_zscore(
        self,
        currency: str,
        as_of: Optional[datetime] = None,
        lookback: Optional[int] = None,
    ) -> Tuple[float, bool]:
        """
        Calculate z-score of current COT position.

        Args:
            currency: Currency code
            as_of: Calculate as of this date
            lookback: Number of weeks for lookback (default: from config)

        Returns:
            Tuple of (zscore, is_valid)
        """
        lookback = lookback or self.config.zscore_lookback

        positions = self.get_positions(currency)

        if as_of is not None:
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            positions = [p for p in positions if p.report_date <= as_of]

        if len(positions) < lookback:
            if len(positions) < 10:
                return (0.0, False)
            lookback = len(positions)

        recent = positions[-lookback:]
        net_longs = [p.net_long for p in recent]

        current = net_longs[-1]
        mean_val = np.mean(net_longs)
        std_val = np.std(net_longs)

        if std_val < 1e-8:
            return (0.0, False)

        zscore = (current - mean_val) / std_val
        zscore = max(-3.0, min(3.0, zscore))

        return (zscore, True)

    def to_dataframe(
        self,
        currencies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert all COT data to DataFrame.

        Args:
            currencies: Filter by currencies (None = all)

        Returns:
            DataFrame with columns for each metric
        """
        all_data = []

        for currency, positions in self._data.items():
            if currencies and currency not in currencies:
                continue

            for pos in positions:
                all_data.append(
                    {
                        "currency": currency,
                        "report_date": pos.report_date,
                        "net_long": pos.net_long,
                        "net_long_pct": pos.net_long_pct,
                        "change_1w": pos.change_1w,
                        "open_interest": pos.open_interest,
                    }
                )

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(["currency", "report_date"])

        return df

    def get_cot_dataframe_for_features(
        self,
        currencies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get COT data in format suitable for forex_features.py.

        Returns DataFrame with columns like EUR_NET, EUR_OI, etc.
        """
        df = self.to_dataframe(currencies)

        if df.empty:
            return pd.DataFrame()

        # Pivot to wide format
        pivot = df.pivot(
            index="report_date",
            columns="currency",
            values=["net_long", "net_long_pct", "open_interest"],
        )

        # Flatten column names
        pivot.columns = [f"{curr}_{metric.upper()}" for metric, curr in pivot.columns]

        # Reset index to have report_date as column
        pivot = pivot.reset_index()
        pivot = pivot.rename(columns={"report_date": "timestamp"})

        return pivot

    @property
    def available_currencies(self) -> List[str]:
        """Get list of available currencies."""
        return list(self._data.keys())

    def __len__(self) -> int:
        """Get total number of positions."""
        return sum(len(positions) for positions in self._data.values())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_mock_cot_data(
    currencies: List[str],
    start_date: datetime,
    end_date: datetime,
    seed: Optional[int] = None,
) -> COTDataLoader:
    """
    Create mock COT data for testing.

    Generates realistic-looking COT positioning data with:
    - Mean-reverting behavior
    - Trending periods
    - Weekly frequency (Tuesday reports)

    Args:
        currencies: List of currency codes
        start_date: Start date
        end_date: End date
        seed: Random seed for reproducibility

    Returns:
        COTDataLoader with mock data
    """
    if seed is not None:
        np.random.seed(seed)

    loader = COTDataLoader()

    # Generate weekly dates (COT reports are released on Fridays for Tuesday positions)
    # But report_date is Tuesday
    current = start_date

    # Find first Tuesday
    while current.weekday() != 1:  # 1 = Tuesday
        current += timedelta(days=1)

    for currency in currencies:
        positions = []
        net_long = 0.0
        open_interest = 100000.0

        while current <= end_date:
            # Random walk with mean reversion
            drift = -0.05 * net_long / 100000  # Mean reversion
            shock = np.random.normal(0, 5000)

            net_long = net_long + drift * 10000 + shock
            open_interest = max(50000, open_interest + np.random.normal(0, 2000))

            # Bound net_long
            net_long = max(-100000, min(100000, net_long))

            long_pos = max(0, (open_interest + net_long) / 2)
            short_pos = max(0, (open_interest - net_long) / 2)

            position = COTPosition(
                currency=currency,
                report_date=current,
                net_long=net_long,
                net_long_pct=net_long / open_interest if open_interest > 0 else 0,
                open_interest=open_interest,
                long_positions=long_pos,
                short_positions=short_pos,
            )

            positions.append(position)
            current += timedelta(weeks=1)

        # Reset current for next currency
        current = start_date
        while current.weekday() != 1:
            current += timedelta(days=1)

        loader._data[currency] = positions

    loader._calculate_changes()
    loader._loaded = True

    return loader


def get_cot_for_pair(
    loader: COTDataLoader,
    pair: str,
    as_of: Optional[datetime] = None,
) -> Tuple[float, float, bool]:
    """
    Get COT positioning for a currency pair.

    Handles the convention that some pairs are quoted as USD/XXX.

    Args:
        loader: COTDataLoader instance
        pair: Currency pair (e.g., "EUR_USD")
        as_of: Get position as of date

    Returns:
        Tuple of (net_long_pct, zscore, is_valid)
    """
    # Parse pair
    pair = pair.upper().replace("/", "_")

    if "_" in pair:
        base, quote = pair.split("_")
    else:
        return (0.5, 0.0, False)

    # Determine which currency to look up
    # COT data is for XXX/USD contracts
    if base == "USD":
        # USD is base, look up quote currency
        # Position is INVERTED (long contract = short USD = long quote)
        currency = quote
        invert = True
    else:
        # Standard XXX/USD
        currency = base
        invert = False

    position = loader.get_current_position(currency, as_of)
    if position is None:
        return (0.5, 0.0, False)

    net_pct = position.net_long_pct
    if invert:
        net_pct = -net_pct

    # Normalize to [0, 1]
    net_normalized = (net_pct + 1.0) / 2.0
    net_normalized = max(0.0, min(1.0, net_normalized))

    zscore, valid = loader.calculate_zscore(currency, as_of)
    if invert:
        zscore = -zscore

    return (net_normalized, zscore, valid)


if __name__ == "__main__":
    import argparse
    import logging
    from datetime import datetime

    parser = argparse.ArgumentParser(description="CFTC COT Data Loader CLI")
    parser.add_argument(
        "--symbols", default="", help="Comma-separated symbols to process (e.g. EUR,GBP,JPY)"
    )
    parser.add_argument(
        "--lookback", type=int, default=52, help="Lookback weeks for Z-score calculation"
    )
    parser.add_argument(
        "--report-type",
        default="legacy_combined",
        choices=["legacy_futures", "legacy_combined", "disaggregated", "tff"],
        help="Type of CFTC COT report",
    )
    parser.add_argument("--cache-dir", default="data/cot_cache", help="Path to cache directory")
    args = parser.parse_args()

    # Configure logging to see download progress
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Map report types for COTDataLoader
    # The config expects report_type to be legacy or tff or disaggregated.
    mapped_report_type = "legacy"
    if args.report_type == "tff":
        mapped_report_type = "tff"
    elif args.report_type == "disaggregated":
        mapped_report_type = "disaggregated"

    # Initialize loader
    cfg = COTConfig(
        cache_dir=args.cache_dir,
        report_type=mapped_report_type,
        zscore_lookback=args.lookback,
        auto_download=True,
    )

    loader = COTDataLoader(cfg)

    print(
        f"Запуск загрузки CFTC COT данных... Report Type: {args.report_type}, Cache: {args.cache_dir}"
    )
    # Download and load data from 2020 to current year
    current_year = datetime.now().year
    loader.load_data(start_year=2020, end_year=current_year)
    loader.save_cache()
    print("Готово. COT кэш успешно сохранен.")
