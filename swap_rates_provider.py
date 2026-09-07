# -*- coding: utf-8 -*-
"""
swap_rates_provider.py
Provides swap/financing rates for forex positions (Phase 4).

Swap rates (also called rollover or financing rates) are the cost/benefit
of holding forex positions overnight. They depend on:
1. Interest rate differential between currencies
2. Broker markup
3. Day of week (Wednesday = 3x for weekend rollover)

Data Sources:
- OANDA: /v3/accounts/{id}/instruments (financing field)
- Historical: Cache locally for backtesting
- Fallback: Interest rate differential calculation

Swap Calculation:
- Long swap: (Base rate - Quote rate - Markup) / 365
- Short swap: (Quote rate - Base rate - Markup) / 365
- Wednesday: 3x swap (weekend rollover)

References:
- OANDA Financing Documentation
- "Currency Carry Trade" - Burnside et al. (2011)

Author: Sultan Suyunbayev
Date: 2025-11-30
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

# OANDA API endpoints
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"

# Default broker markup (typical retail broker)
# Expressed as annual percentage points
DEFAULT_MARKUP_BPS = 50  # 0.50% annual = 50 bps

# Days with 3x rollover (typically Wednesday for weekend)
TRIPLE_ROLLOVER_DAYS = [2]  # Wednesday (0=Monday)

# Major currency interest rates (as of late 2024)
# These serve as fallback when live data unavailable
DEFAULT_RATES: Dict[str, float] = {
    "USD": 5.25,  # Federal Funds Rate
    "EUR": 4.00,  # ECB Deposit Rate
    "GBP": 5.25,  # BOE Bank Rate
    "JPY": -0.10,  # BOJ Policy Rate
    "CHF": 1.50,  # SNB Policy Rate
    "AUD": 4.35,  # RBA Cash Rate
    "CAD": 5.00,  # BOC Policy Rate
    "NZD": 5.50,  # RBNZ OCR
}

# Default cache directory
DEFAULT_SWAP_CACHE_DIR = "data/swap_cache"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SwapRates:
    """
    Container for swap rates for a currency pair.

    Attributes:
        symbol: Currency pair (e.g., "EUR_USD")
        long_swap_pips: Swap rate for long positions (pips/lot/day)
        short_swap_pips: Swap rate for short positions (pips/lot/day)
        long_swap_pct: Annualized percentage for long
        short_swap_pct: Annualized percentage for short
        timestamp: When rates were fetched
        source: Data source
    """

    symbol: str
    long_swap_pips: float
    short_swap_pips: float
    long_swap_pct: float = 0.0
    short_swap_pct: float = 0.0
    timestamp: Optional[datetime] = None
    source: str = "cached"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def get_daily_cost(
        self,
        position_size: float,
        is_long: bool,
        pip_value: float = 10.0,
    ) -> float:
        """
        Calculate daily swap cost/credit.

        Args:
            position_size: Position size in lots
            is_long: True for long, False for short
            pip_value: Value per pip (default: $10/lot for major pairs)

        Returns:
            Daily swap cost (negative) or credit (positive) in quote currency
        """
        swap_pips = self.long_swap_pips if is_long else self.short_swap_pips
        return swap_pips * position_size * pip_value

    def get_holding_cost(
        self,
        position_size: float,
        is_long: bool,
        days: int,
        include_weekends: bool = True,
    ) -> float:
        """
        Calculate total swap cost for holding period.

        Args:
            position_size: Position size in lots
            is_long: True for long, False for short
            days: Number of days
            include_weekends: Account for 3x Wednesday rollover

        Returns:
            Total swap cost in quote currency
        """
        daily_cost = self.get_daily_cost(position_size, is_long)

        if include_weekends:
            # Approximate: each week has 1 triple rollover day
            weeks = days // 7
            extra_days = days % 7
            total_days = days + 2 * weeks  # Add 2 for each weekend
        else:
            total_days = days

        return daily_cost * total_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "long_swap_pips": self.long_swap_pips,
            "short_swap_pips": self.short_swap_pips,
            "long_swap_pct": self.long_swap_pct,
            "short_swap_pct": self.short_swap_pct,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwapRates":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return cls(
            symbol=data["symbol"],
            long_swap_pips=data["long_swap_pips"],
            short_swap_pips=data["short_swap_pips"],
            long_swap_pct=data.get("long_swap_pct", 0.0),
            short_swap_pct=data.get("short_swap_pct", 0.0),
            timestamp=timestamp,
            source=data.get("source", "cached"),
        )


@dataclass
class SwapProviderConfig:
    """Configuration for swap rate providers."""

    oanda_api_key: Optional[str] = None
    oanda_account_id: Optional[str] = None
    practice: bool = True
    cache_dir: str = DEFAULT_SWAP_CACHE_DIR
    cache_ttl_hours: float = 24.0
    markup_bps: int = DEFAULT_MARKUP_BPS
    timeout_seconds: float = 30.0


# =============================================================================
# BASE SWAP PROVIDER
# =============================================================================


class SwapRatesProvider(ABC):
    """Abstract base class for swap rate providers."""

    @abstractmethod
    def get_current_swaps(self, symbol: str) -> Optional[SwapRates]:
        """
        Get current swap rates for a symbol.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            SwapRates or None if unavailable
        """
        pass

    @abstractmethod
    def get_historical_swaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get historical swap rates.

        Args:
            symbol: Currency pair
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with columns: timestamp, long_swap, short_swap
        """
        pass


# =============================================================================
# INTEREST RATE BASED PROVIDER
# =============================================================================


class InterestRateSwapProvider(SwapRatesProvider):
    """
    Calculate swap rates from interest rate differentials.

    This is a theoretical approach based on covered interest rate parity.
    Actual broker swaps may differ due to markups and market conditions.

    Formula:
        Long swap = (Base rate - Quote rate - Markup) / 365
        Short swap = (Quote rate - Base rate - Markup) / 365

    Note: Negative swap means you pay; positive means you receive.
    """

    def __init__(
        self,
        config: Optional[SwapProviderConfig] = None,
        rates: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize provider.

        Args:
            config: Provider configuration
            rates: Interest rates by currency (annual %)
        """
        self.config = config or SwapProviderConfig()
        self.rates = rates if rates is not None else DEFAULT_RATES.copy()
        self._cache: Dict[str, SwapRates] = {}

    def update_rates(self, rates: Dict[str, float]) -> None:
        """Update interest rates."""
        self.rates.update(rates)
        # Invalidate cache
        self._cache.clear()

    def get_rate(self, currency: str) -> float:
        """Get interest rate for currency."""
        return self.rates.get(currency.upper(), 0.0)

    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse currency pair into base and quote."""
        symbol = symbol.upper().strip()
        if "_" in symbol:
            parts = symbol.split("_")
            return (parts[0], parts[1])
        elif "/" in symbol:
            parts = symbol.split("/")
            return (parts[0], parts[1])
        elif len(symbol) == 6:
            return (symbol[:3], symbol[3:])
        else:
            raise ValueError(f"Invalid symbol format: {symbol}")

    def calculate_swap_rates(
        self,
        symbol: str,
        pip_size: float = 0.0001,
    ) -> SwapRates:
        """
        Calculate swap rates from interest rate differential.

        Args:
            symbol: Currency pair
            pip_size: Pip size for the pair (0.0001 for most, 0.01 for JPY pairs)

        Returns:
            SwapRates object
        """
        base, quote = self._parse_symbol(symbol)
        base_rate = self.get_rate(base)
        quote_rate = self.get_rate(quote)
        markup_pct = self.config.markup_bps / 100.0

        # Calculate daily swap in percentage terms
        # Long: you receive base rate, pay quote rate (minus markup)
        long_daily_pct = (base_rate - quote_rate - markup_pct) / 365
        short_daily_pct = (quote_rate - base_rate - markup_pct) / 365

        # Convert to pips per lot
        # Assuming 1 lot = 100,000 units
        lot_size = 100000

        # Approximate pip value (simplified)
        # This is approximate; actual pip value depends on pair and account currency
        long_swap_pips = long_daily_pct / 100 * lot_size * pip_size / pip_size
        short_swap_pips = short_daily_pct / 100 * lot_size * pip_size / pip_size

        # Scale to reasonable pip values (typical range: -20 to +20)
        scale_factor = 10.0
        long_swap_pips = long_swap_pips * scale_factor / lot_size
        short_swap_pips = short_swap_pips * scale_factor / lot_size

        return SwapRates(
            symbol=symbol,
            long_swap_pips=long_swap_pips,
            short_swap_pips=short_swap_pips,
            long_swap_pct=long_daily_pct * 365,  # Annualized
            short_swap_pct=short_daily_pct * 365,
            timestamp=datetime.now(timezone.utc),
            source="interest_rate_calc",
        )

    def get_current_swaps(self, symbol: str) -> Optional[SwapRates]:
        """Get current swap rates."""
        try:
            if symbol in self._cache:
                cached = self._cache[symbol]
                # Check TTL
                if cached.timestamp:
                    age = datetime.now(timezone.utc) - cached.timestamp
                    if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                        return cached

            swaps = self.calculate_swap_rates(symbol)
            self._cache[symbol] = swaps
            return swaps

        except Exception as e:
            logger.warning(f"Failed to calculate swaps for {symbol}: {e}")
            return None

    def get_historical_swaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get historical swap rates.

        Note: This is a simplified implementation using current rates.
        For accurate historical swaps, use cached OANDA data.
        """
        swaps = self.get_current_swaps(symbol)
        if swaps is None:
            return pd.DataFrame()

        # Generate daily entries
        dates = pd.date_range(start=start_date, end=end_date, freq="D", tz=timezone.utc)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "long_swap": [swaps.long_swap_pips] * len(dates),
                "short_swap": [swaps.short_swap_pips] * len(dates),
            }
        )


# =============================================================================
# OANDA SWAP PROVIDER
# =============================================================================


class OandaSwapProvider(SwapRatesProvider):
    """
    Get swap rates from OANDA API.

    Uses /v3/accounts/{id}/instruments endpoint which returns
    financing information for each tradeable instrument.
    """

    def __init__(self, config: Optional[SwapProviderConfig] = None):
        """Initialize provider."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for OandaSwapProvider")

        self.config = config or SwapProviderConfig()

        if not self.config.oanda_api_key:
            raise ValueError("OANDA API key required")
        if not self.config.oanda_account_id:
            raise ValueError("OANDA account ID required")

        self._base_url = OANDA_PRACTICE_URL if self.config.practice else OANDA_LIVE_URL
        self._session = requests.Session()
        self._cache: Dict[str, SwapRates] = {}

    def _make_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make OANDA API request."""
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.oanda_api_key}",
            "Accept": "application/json",
        }

        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"OANDA API error: {response.status_code}")
                return None

        except requests.RequestException as e:
            logger.warning(f"OANDA request failed: {e}")
            return None

    def _parse_oanda_symbol(self, symbol: str) -> str:
        """Convert to OANDA symbol format (EUR_USD)."""
        symbol = symbol.upper().strip()
        if "_" in symbol:
            return symbol
        elif "/" in symbol:
            return symbol.replace("/", "_")
        elif len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
        return symbol

    def get_current_swaps(self, symbol: str) -> Optional[SwapRates]:
        """Get current swap rates from OANDA."""
        oanda_symbol = self._parse_oanda_symbol(symbol)

        # Check cache
        if oanda_symbol in self._cache:
            cached = self._cache[oanda_symbol]
            if cached.timestamp:
                age = datetime.now(timezone.utc) - cached.timestamp
                if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                    return cached

        # Fetch from API
        endpoint = f"/v3/accounts/{self.config.oanda_account_id}/instruments"
        data = self._make_request(endpoint)

        if not data:
            return None

        instruments = data.get("instruments", [])

        for inst in instruments:
            if inst.get("name") != oanda_symbol:
                continue

            financing = inst.get("financing", {})
            if not financing:
                continue

            # OANDA provides financing as daily rate (percentage)
            long_rate = float(financing.get("longRate", 0))
            short_rate = float(financing.get("shortRate", 0))

            # Convert to pips (approximate)
            # This depends on pip size and lot size
            pip_size = 0.01 if "JPY" in oanda_symbol else 0.0001

            # Simplified conversion (actual depends on pair)
            long_pips = long_rate * 10000 * pip_size
            short_pips = short_rate * 10000 * pip_size

            swaps = SwapRates(
                symbol=oanda_symbol,
                long_swap_pips=long_pips,
                short_swap_pips=short_pips,
                long_swap_pct=long_rate * 365,
                short_swap_pct=short_rate * 365,
                timestamp=datetime.now(timezone.utc),
                source="oanda",
            )

            self._cache[oanda_symbol] = swaps
            return swaps

        return None

    def get_historical_swaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get historical swap rates.

        Note: OANDA doesn't provide historical financing rates.
        This returns cached current rates or empty DataFrame.
        """
        swaps = self.get_current_swaps(symbol)
        if swaps is None:
            return pd.DataFrame()

        dates = pd.date_range(start=start_date, end=end_date, freq="D", tz=timezone.utc)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "long_swap": [swaps.long_swap_pips] * len(dates),
                "short_swap": [swaps.short_swap_pips] * len(dates),
            }
        )


# =============================================================================
# CACHED SWAP PROVIDER
# =============================================================================


class CachedSwapProvider(SwapRatesProvider):
    """
    Swap provider using local cached data.

    Ideal for backtesting where historical swap data is needed.
    """

    def __init__(self, config: Optional[SwapProviderConfig] = None):
        """Initialize provider."""
        self.config = config or SwapProviderConfig()
        self._cache: Dict[str, List[SwapRates]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached swap data from disk."""
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
                    symbol = filename.replace(".json", "").replace("_", "/")
                    swaps = [SwapRates.from_dict(s) for s in data]
                    self._cache[symbol.replace("/", "_")] = swaps
            except Exception as e:
                logger.warning(f"Failed to load swap cache {filename}: {e}")

    def save_cache(self) -> None:
        """Save cache to disk."""
        cache_dir = self.config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        for symbol, swaps in self._cache.items():
            filename = symbol.replace("/", "_") + ".json"
            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, "w") as f:
                    json.dump([s.to_dict() for s in swaps], f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save swap cache {symbol}: {e}")

    def add_swaps(self, symbol: str, swaps: SwapRates) -> None:
        """Add swap rates to cache."""
        key = symbol.replace("/", "_").upper()
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(swaps)

    def get_current_swaps(self, symbol: str) -> Optional[SwapRates]:
        """Get most recent swap rates from cache."""
        key = symbol.replace("/", "_").upper()
        if key not in self._cache or not self._cache[key]:
            return None
        return self._cache[key][-1]

    def get_historical_swaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get historical swap rates from cache."""
        key = symbol.replace("/", "_").upper()
        if key not in self._cache:
            return pd.DataFrame()

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        swaps = [
            s for s in self._cache[key] if s.timestamp and start_date <= s.timestamp <= end_date
        ]

        if not swaps:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "timestamp": [s.timestamp for s in swaps],
                "long_swap": [s.long_swap_pips for s in swaps],
                "short_swap": [s.short_swap_pips for s in swaps],
            }
        )


# =============================================================================
# UNIFIED SWAP RATES INTERFACE
# =============================================================================


class SwapRatesManager:
    """
    Unified interface for swap rates with fallback support.

    Priority:
    1. OANDA API (if configured)
    2. Interest rate calculation
    3. Cached data

    Usage:
        manager = SwapRatesManager(config)
        swaps = manager.get_swaps("EUR_USD")

        if swaps:
            daily_cost = swaps.get_daily_cost(1.0, is_long=True)
    """

    def __init__(self, config: Optional[SwapProviderConfig] = None):
        """Initialize manager."""
        self.config = config or SwapProviderConfig()
        self._providers: List[SwapRatesProvider] = []

        # Initialize providers in priority order
        if self.config.oanda_api_key and self.config.oanda_account_id:
            try:
                self._providers.append(OandaSwapProvider(self.config))
            except (ImportError, ValueError) as e:
                logger.warning(f"OANDA swap provider unavailable: {e}")

        self._providers.append(InterestRateSwapProvider(self.config))
        self._providers.append(CachedSwapProvider(self.config))

    def get_swaps(self, symbol: str) -> Optional[SwapRates]:
        """
        Get swap rates using available providers.

        Args:
            symbol: Currency pair

        Returns:
            SwapRates or None if unavailable
        """
        for provider in self._providers:
            try:
                swaps = provider.get_current_swaps(symbol)
                if swaps is not None:
                    return swaps
            except Exception as e:
                logger.debug(f"Provider {type(provider).__name__} failed: {e}")
                continue

        return None

    def get_historical_swaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get historical swap rates.

        Args:
            symbol: Currency pair
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with swap history
        """
        for provider in self._providers:
            try:
                df = provider.get_historical_swaps(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.debug(f"Provider {type(provider).__name__} failed: {e}")
                continue

        return pd.DataFrame()

    def update_interest_rates(self, rates: Dict[str, float]) -> None:
        """Update interest rates used for calculation."""
        for provider in self._providers:
            if isinstance(provider, InterestRateSwapProvider):
                provider.update_rates(rates)
                break


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def calculate_carry_cost(
    symbol: str,
    position_size: float,
    is_long: bool,
    holding_days: int,
    swaps: Optional[SwapRates] = None,
    rates: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate total carry/swap cost for a position.

    Args:
        symbol: Currency pair
        position_size: Position size in lots
        is_long: True for long, False for short
        holding_days: Number of days to hold
        swaps: SwapRates object (if None, calculate from rates)
        rates: Interest rates by currency

    Returns:
        Total swap cost in quote currency (negative = cost)
    """
    if swaps is None:
        provider = InterestRateSwapProvider(rates=rates)
        swaps = provider.get_current_swaps(symbol)

    if swaps is None:
        return 0.0

    return swaps.get_holding_cost(position_size, is_long, holding_days)


def get_pip_size(symbol: str) -> float:
    """Get pip size for a currency pair."""
    symbol = symbol.upper()
    # JPY pairs have pip size of 0.01
    if "JPY" in symbol:
        return 0.01
    return 0.0001


def is_triple_rollover_day(date: datetime) -> bool:
    """Check if date is a triple rollover day (typically Wednesday)."""
    return date.weekday() in TRIPLE_ROLLOVER_DAYS
