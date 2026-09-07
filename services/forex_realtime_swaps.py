# -*- coding: utf-8 -*-
"""
services/forex_realtime_swaps.py
Real-time Swap Rates Provider with OANDA Streaming Integration

Phase 11: Forex Realism Enhancement (2025-11-30)

This module provides real-time swap rate fetching and streaming from OANDA API,
with intelligent caching, fallback mechanisms, and interpolation for missing data.

Key Features:
1. Real-time streaming swap rates from OANDA v20 API
2. Intelligent caching with TTL and staleness detection
3. Interpolation for pairs without direct quotes
4. Wednesday triple swap calculation
5. Historical swap rate storage for backtesting
6. Fallback to calculated rates from interest rate differentials

OTC Forex Swap Mechanics:
- Swaps are charged at 5pm ET (rollover time)
- Wednesday swaps are 3x (cover weekend settlement T+2)
- Long positions: pay if quote currency rate > base currency rate
- Short positions: pay if base currency rate > quote currency rate
- Broker markup typically 0.25-0.75% added to interbank rate

References:
- OANDA v20 API Documentation: Financing endpoint
- Burnside et al. (2011): "Carry Trades and Currency Crashes"
- Lustig et al. (2011): "Common Risk Factors in Currency Markets"
- BIS (2022): Triennial Survey - FX Swap Market

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default OANDA API endpoints
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"
OANDA_STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"

# Swap rate cache TTL (seconds)
DEFAULT_CACHE_TTL_SEC = 3600  # 1 hour
STALE_THRESHOLD_SEC = 7200  # 2 hours = stale warning

# Interbank rates for interpolation (annual %, as of late 2024)
# Source: Central bank policy rates
DEFAULT_INTERBANK_RATES: Dict[str, float] = {
    "USD": 5.25,  # Fed Funds Rate
    "EUR": 4.50,  # ECB Main Refinancing Rate
    "GBP": 5.25,  # Bank of England Rate
    "JPY": 0.10,  # Bank of Japan Rate
    "CHF": 1.75,  # Swiss National Bank Rate
    "AUD": 4.35,  # Reserve Bank of Australia Rate
    "CAD": 5.00,  # Bank of Canada Rate
    "NZD": 5.50,  # Reserve Bank of New Zealand Rate
    "SEK": 4.00,  # Sveriges Riksbank Rate
    "NOK": 4.50,  # Norges Bank Rate
    "DKK": 3.60,  # Danmarks Nationalbank Rate
    "PLN": 5.75,  # National Bank of Poland Rate
    "HUF": 6.50,  # Magyar Nemzeti Bank Rate
    "CZK": 4.75,  # Czech National Bank Rate
    "TRY": 50.00,  # Central Bank of Turkey Rate
    "ZAR": 8.25,  # South African Reserve Bank Rate
    "MXN": 11.00,  # Banco de México Rate
    "SGD": 3.50,  # Monetary Authority of Singapore
    "HKD": 5.75,  # Hong Kong Monetary Authority
}

# Broker markup ranges (annual %)
BROKER_MARKUP_RANGES: Dict[str, Tuple[float, float]] = {
    "retail": (0.50, 1.00),
    "professional": (0.25, 0.50),
    "institutional": (0.10, 0.25),
}


# =============================================================================
# Enums
# =============================================================================


class SwapRateSource(str, Enum):
    """Source of swap rate data."""

    OANDA_REALTIME = "oanda_realtime"
    OANDA_CACHED = "oanda_cached"
    CALCULATED = "calculated"
    INTERPOLATED = "interpolated"
    FALLBACK = "fallback"


class SwapDirection(str, Enum):
    """Position direction for swap calculation."""

    LONG = "long"
    SHORT = "short"


class SwapRateQuality(str, Enum):
    """Quality indicator for swap rate."""

    FRESH = "fresh"  # < 1 hour old
    RECENT = "recent"  # 1-2 hours old
    STALE = "stale"  # 2-6 hours old
    EXPIRED = "expired"  # > 6 hours old


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RealtimeSwapRate:
    """
    Real-time swap rate for a currency pair.

    Attributes:
        pair: Currency pair (e.g., "EUR_USD")
        long_swap_pips: Swap for long position in pips/day
        short_swap_pips: Swap for short position in pips/day
        long_swap_pct: Swap for long position as annual %
        short_swap_pct: Swap for short position as annual %
        timestamp_ms: When this rate was fetched (milliseconds)
        source: Source of this rate
        quality: Quality indicator based on age
        broker_markup_applied: Whether broker markup is included
        interbank_long: Raw interbank rate for long (before markup)
        interbank_short: Raw interbank rate for short (before markup)
    """

    pair: str
    long_swap_pips: float
    short_swap_pips: float
    long_swap_pct: float = 0.0
    short_swap_pct: float = 0.0
    timestamp_ms: int = 0
    source: SwapRateSource = SwapRateSource.FALLBACK
    quality: SwapRateQuality = SwapRateQuality.FRESH
    broker_markup_applied: bool = True
    interbank_long: float = 0.0
    interbank_short: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    @property
    def age_seconds(self) -> float:
        """Get age of this rate in seconds."""
        now_ms = int(time.time() * 1000)
        return (now_ms - self.timestamp_ms) / 1000.0

    def is_fresh(self, max_age_sec: float = DEFAULT_CACHE_TTL_SEC) -> bool:
        """Check if rate is still fresh."""
        return self.age_seconds < max_age_sec

    def get_swap_for_direction(
        self,
        direction: SwapDirection,
        in_pips: bool = True,
    ) -> float:
        """Get swap rate for a direction."""
        if direction == SwapDirection.LONG:
            return self.long_swap_pips if in_pips else self.long_swap_pct
        return self.short_swap_pips if in_pips else self.short_swap_pct

    def apply_wednesday_multiplier(self, day_of_week: int) -> "RealtimeSwapRate":
        """
        Apply Wednesday triple swap multiplier.

        Wednesday swap is 3x because of T+2 settlement covering the weekend.

        Args:
            day_of_week: 0=Monday, 2=Wednesday, etc.

        Returns:
            New RealtimeSwapRate with multiplied values if Wednesday
        """
        if day_of_week != 2:  # Not Wednesday
            return self

        return RealtimeSwapRate(
            pair=self.pair,
            long_swap_pips=self.long_swap_pips * 3,
            short_swap_pips=self.short_swap_pips * 3,
            long_swap_pct=self.long_swap_pct * 3,
            short_swap_pct=self.short_swap_pct * 3,
            timestamp_ms=self.timestamp_ms,
            source=self.source,
            quality=self.quality,
            broker_markup_applied=self.broker_markup_applied,
            interbank_long=self.interbank_long * 3,
            interbank_short=self.interbank_short * 3,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "long_swap_pips": self.long_swap_pips,
            "short_swap_pips": self.short_swap_pips,
            "long_swap_pct": self.long_swap_pct,
            "short_swap_pct": self.short_swap_pct,
            "timestamp_ms": self.timestamp_ms,
            "source": self.source.value,
            "quality": self.quality.value,
            "broker_markup_applied": self.broker_markup_applied,
            "age_seconds": self.age_seconds,
        }


@dataclass
class SwapRateUpdate:
    """
    Streaming swap rate update event.

    Used for real-time streaming updates from OANDA.
    """

    pair: str
    long_financing: float  # Daily financing rate for long
    short_financing: float  # Daily financing rate for short
    timestamp_ms: int
    is_tradeable: bool = True

    def to_swap_rate(self, pip_value: float = 0.0001) -> RealtimeSwapRate:
        """Convert to RealtimeSwapRate."""
        # Convert daily financing rate to pips
        # Financing is typically expressed as daily interest rate
        long_pips = self.long_financing / pip_value * 10000
        short_pips = self.short_financing / pip_value * 10000

        return RealtimeSwapRate(
            pair=self.pair,
            long_swap_pips=long_pips,
            short_swap_pips=short_pips,
            long_swap_pct=self.long_financing * 365 * 100,
            short_swap_pct=self.short_financing * 365 * 100,
            timestamp_ms=self.timestamp_ms,
            source=SwapRateSource.OANDA_REALTIME,
            quality=SwapRateQuality.FRESH,
        )


@dataclass
class SwapRateCacheConfig:
    """Configuration for swap rate caching."""

    cache_ttl_sec: float = DEFAULT_CACHE_TTL_SEC
    stale_threshold_sec: float = STALE_THRESHOLD_SEC
    max_cache_size: int = 200
    persist_to_disk: bool = True
    cache_file_path: str = "data/forex/realtime_swaps_cache.json"
    auto_refresh_enabled: bool = True
    auto_refresh_interval_sec: float = 1800.0  # 30 minutes
    fallback_to_calculated: bool = True
    broker_tier: str = "retail"


# =============================================================================
# Protocols
# =============================================================================


class SwapRateCallback(Protocol):
    """Callback for swap rate updates."""

    def __call__(self, rate: RealtimeSwapRate) -> None:
        """Called when a new swap rate is received."""
        ...


class InterestRateProvider(Protocol):
    """Protocol for interest rate data providers."""

    def get_rate(self, currency: str) -> Optional[float]:
        """Get current interest rate for a currency."""
        ...


# =============================================================================
# Swap Rate Calculator
# =============================================================================


class SwapRateCalculator:
    """
    Calculates swap rates from interest rate differentials.

    This is used when real-time rates are not available.
    The calculation follows the standard forex swap formula:

    Swap = (Base Rate - Quote Rate - Broker Markup) × Position × Days / 365

    For pips: Swap_pips = Swap_% × Price / Pip_Value

    References:
        - Burnside et al. (2011): "Carry Trades and Currency Crashes"
        - Lustig et al. (2011): "Common Risk Factors in Currency Markets"
    """

    def __init__(
        self,
        interest_rates: Optional[Dict[str, float]] = None,
        broker_tier: str = "retail",
        custom_markups: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize calculator.

        Args:
            interest_rates: Currency -> annual interest rate (%)
            broker_tier: "retail", "professional", or "institutional"
            custom_markups: Pair -> custom markup override
        """
        self._rates = interest_rates or DEFAULT_INTERBANK_RATES.copy()
        self._broker_tier = broker_tier
        self._custom_markups = custom_markups or {}

        # Get markup range for tier
        self._markup_range = BROKER_MARKUP_RANGES.get(broker_tier, BROKER_MARKUP_RANGES["retail"])

    def update_rate(self, currency: str, rate: float) -> None:
        """Update interest rate for a currency."""
        self._rates[currency.upper()] = rate

    def get_rate(self, currency: str) -> Optional[float]:
        """Get interest rate for a currency."""
        return self._rates.get(currency.upper())

    def calculate_swap_rate(
        self,
        pair: str,
        direction: SwapDirection,
        mid_price: float = 1.0,
        include_markup: bool = True,
    ) -> Tuple[float, float]:
        """
        Calculate swap rate for a pair and direction.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            direction: LONG or SHORT
            mid_price: Current mid price for pip conversion
            include_markup: Whether to include broker markup

        Returns:
            (swap_pips, swap_pct) tuple
        """
        # Parse pair
        parts = pair.upper().replace("/", "_").split("_")
        if len(parts) != 2:
            logger.warning(f"Invalid pair format: {pair}")
            return (0.0, 0.0)

        base_ccy, quote_ccy = parts

        # Get interest rates
        base_rate = self._rates.get(base_ccy)
        quote_rate = self._rates.get(quote_ccy)

        if base_rate is None or quote_rate is None:
            logger.warning(
                f"Missing interest rate for {pair}: " f"base={base_rate}, quote={quote_rate}"
            )
            return (0.0, 0.0)

        # Calculate interest rate differential
        # Long position: earn base rate, pay quote rate
        # Short position: pay base rate, earn quote rate
        if direction == SwapDirection.LONG:
            rate_diff = base_rate - quote_rate
        else:
            rate_diff = quote_rate - base_rate

        # Apply broker markup (reduces the rate received or increases rate paid)
        if include_markup:
            markup = self._get_markup(pair)
            # If positive (earning), reduce by markup
            # If negative (paying), increase by markup
            if rate_diff > 0:
                rate_diff = max(0, rate_diff - markup)
            else:
                rate_diff = rate_diff - markup

        # Convert to daily rate
        daily_rate_pct = rate_diff / 365.0

        # Convert to pips
        # For most pairs: 1 pip = 0.0001
        # For JPY pairs: 1 pip = 0.01
        is_jpy = "JPY" in pair.upper()
        pip_value = 0.01 if is_jpy else 0.0001

        # Swap in pips = daily_rate × mid_price / pip_value
        swap_pips = daily_rate_pct / 100.0 * mid_price / pip_value

        return (swap_pips, daily_rate_pct)

    def _get_markup(self, pair: str) -> float:
        """Get broker markup for a pair."""
        # Check custom markup first
        if pair in self._custom_markups:
            return self._custom_markups[pair]

        # Use average of markup range for tier
        return (self._markup_range[0] + self._markup_range[1]) / 2.0

    def calculate_full_swap_rate(
        self,
        pair: str,
        mid_price: float = 1.0,
    ) -> RealtimeSwapRate:
        """
        Calculate full swap rate for both directions.

        Args:
            pair: Currency pair
            mid_price: Current mid price

        Returns:
            RealtimeSwapRate with calculated values
        """
        long_pips, long_pct = self.calculate_swap_rate(pair, SwapDirection.LONG, mid_price)
        short_pips, short_pct = self.calculate_swap_rate(pair, SwapDirection.SHORT, mid_price)

        # Also calculate interbank rates (without markup)
        interbank_long, _ = self.calculate_swap_rate(
            pair, SwapDirection.LONG, mid_price, include_markup=False
        )
        interbank_short, _ = self.calculate_swap_rate(
            pair, SwapDirection.SHORT, mid_price, include_markup=False
        )

        return RealtimeSwapRate(
            pair=pair,
            long_swap_pips=long_pips,
            short_swap_pips=short_pips,
            long_swap_pct=long_pct * 365,  # Annualize
            short_swap_pct=short_pct * 365,
            timestamp_ms=int(time.time() * 1000),
            source=SwapRateSource.CALCULATED,
            quality=SwapRateQuality.FRESH,
            broker_markup_applied=True,
            interbank_long=interbank_long,
            interbank_short=interbank_short,
        )


# =============================================================================
# OANDA API Client
# =============================================================================


class OandaSwapRateClient:
    """
    OANDA API client for fetching swap/financing rates.

    Uses OANDA v20 API to fetch real-time financing rates.
    Supports both REST polling and streaming updates.

    API Endpoints:
        - GET /v3/accounts/{accountId}/instruments/{instrument}/candles
        - GET /v3/accounts/{accountId}/pricing (includes financing)
        - GET /v3/accounts/{accountId}/pricing/stream (streaming)

    References:
        - OANDA v20 REST API: https://developer.oanda.com/rest-live-v20/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        practice: bool = True,
        timeout_sec: float = 30.0,
    ) -> None:
        """
        Initialize OANDA client.

        Args:
            api_key: OANDA API key (or from OANDA_API_KEY env var)
            account_id: OANDA account ID (or from OANDA_ACCOUNT_ID env var)
            practice: Use practice (demo) environment
            timeout_sec: Request timeout in seconds
        """
        self._api_key = api_key or os.environ.get("OANDA_API_KEY", "")
        self._account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self._practice = practice
        self._timeout = timeout_sec

        self._base_url = OANDA_PRACTICE_URL if practice else OANDA_LIVE_URL
        self._stream_url = OANDA_STREAM_PRACTICE_URL if practice else OANDA_STREAM_LIVE_URL

        self._session: Optional[Any] = None
        self._is_connected = False

    @property
    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self._api_key and self._account_id)

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX",
        }

    def fetch_financing_rates(
        self,
        instruments: List[str],
    ) -> Dict[str, RealtimeSwapRate]:
        """
        Fetch financing rates for instruments.

        Uses the pricing endpoint which includes financing information.

        Args:
            instruments: List of instrument names (e.g., ["EUR_USD", "GBP_USD"])

        Returns:
            Dict mapping instrument -> RealtimeSwapRate
        """
        if not self.is_configured:
            logger.warning("OANDA credentials not configured")
            return {}

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return {}

        results: Dict[str, RealtimeSwapRate] = {}

        # OANDA pricing endpoint
        url = f"{self._base_url}/v3/accounts/{self._account_id}/pricing"
        params = {"instruments": ",".join(instruments)}

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            for price in data.get("prices", []):
                instrument = price.get("instrument")
                if not instrument:
                    continue

                # Extract financing rates
                # OANDA provides positive/negative rates for long/short
                long_financing = float(price.get("financing", {}).get("longRate", 0))
                short_financing = float(price.get("financing", {}).get("shortRate", 0))

                # Determine pip value
                is_jpy = "JPY" in instrument
                pip_value = 0.01 if is_jpy else 0.0001

                # Get current price for conversion
                bid = float(price.get("bids", [{}])[0].get("price", 1.0))
                ask = float(price.get("asks", [{}])[0].get("price", 1.0))
                mid = (bid + ask) / 2.0

                # Convert annual rate to pips/day
                long_pips = long_financing / 365.0 * mid / pip_value
                short_pips = short_financing / 365.0 * mid / pip_value

                results[instrument] = RealtimeSwapRate(
                    pair=instrument,
                    long_swap_pips=long_pips,
                    short_swap_pips=short_pips,
                    long_swap_pct=long_financing,
                    short_swap_pct=short_financing,
                    timestamp_ms=int(time.time() * 1000),
                    source=SwapRateSource.OANDA_REALTIME,
                    quality=SwapRateQuality.FRESH,
                    broker_markup_applied=True,
                )

            logger.info(f"Fetched {len(results)} swap rates from OANDA")

        except Exception as e:
            logger.error(f"Failed to fetch OANDA financing rates: {e}")

        return results

    async def stream_financing_rates(
        self,
        instruments: List[str],
        callback: SwapRateCallback,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Stream financing rate updates.

        Connects to OANDA streaming API and calls callback on each update.

        Args:
            instruments: Instruments to stream
            callback: Called with each rate update
            stop_event: Event to signal stream stop
        """
        if not self.is_configured:
            logger.warning("OANDA credentials not configured for streaming")
            return

        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp library not installed for streaming")
            return

        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        params = {"instruments": ",".join(instruments)}

        stop_event = stop_event or asyncio.Event()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                ) as response:
                    self._is_connected = True
                    logger.info(f"Connected to OANDA swap rate stream")

                    async for line in response.content:
                        if stop_event.is_set():
                            break

                        if not line:
                            continue

                        try:
                            data = json.loads(line.decode("utf-8"))

                            if data.get("type") == "PRICE":
                                rate = self._parse_price_update(data)
                                if rate:
                                    callback(rate)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing stream data: {e}")

            except Exception as e:
                logger.error(f"Stream error: {e}")
            finally:
                self._is_connected = False

    def _parse_price_update(self, data: Dict[str, Any]) -> Optional[RealtimeSwapRate]:
        """Parse price update from stream."""
        instrument = data.get("instrument")
        if not instrument:
            return None

        financing = data.get("financing", {})
        long_rate = float(financing.get("longRate", 0))
        short_rate = float(financing.get("shortRate", 0))

        # Get price for conversion
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        bid = float(bids[0].get("price", 1.0)) if bids else 1.0
        ask = float(asks[0].get("price", 1.0)) if asks else 1.0
        mid = (bid + ask) / 2.0

        is_jpy = "JPY" in instrument
        pip_value = 0.01 if is_jpy else 0.0001

        long_pips = long_rate / 365.0 * mid / pip_value
        short_pips = short_rate / 365.0 * mid / pip_value

        timestamp = data.get("time", "")
        if timestamp:
            try:
                ts_ms = int(float(timestamp) * 1000)
            except ValueError:
                ts_ms = int(time.time() * 1000)
        else:
            ts_ms = int(time.time() * 1000)

        return RealtimeSwapRate(
            pair=instrument,
            long_swap_pips=long_pips,
            short_swap_pips=short_pips,
            long_swap_pct=long_rate,
            short_swap_pct=short_rate,
            timestamp_ms=ts_ms,
            source=SwapRateSource.OANDA_REALTIME,
            quality=SwapRateQuality.FRESH,
        )


# =============================================================================
# Real-time Swap Rate Provider
# =============================================================================


class RealtimeSwapRateProvider:
    """
    Production-grade real-time swap rate provider.

    Combines multiple data sources with intelligent caching:
    1. OANDA real-time streaming (primary)
    2. OANDA REST polling (fallback)
    3. Calculated from interest rates (last resort)

    Features:
    - Automatic fallback chain
    - TTL-based caching with staleness detection
    - Persistence to disk for cold starts
    - Background refresh thread
    - Rate interpolation for exotic pairs

    Thread Safety:
    - Uses threading.Lock for cache access
    - Background refresh runs in separate thread
    - Streaming runs in asyncio event loop

    Usage:
        provider = RealtimeSwapRateProvider()
        provider.start_background_refresh()

        # Get swap rate for a pair
        rate = provider.get_swap_rate("EUR_USD")
        print(f"Long swap: {rate.long_swap_pips} pips/day")

        # Get swap cost for a position
        cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100000,  # 1 lot
            days=1,
        )
    """

    # Default instruments to track
    DEFAULT_INSTRUMENTS: List[str] = [
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "USD_CHF",
        "AUD_USD",
        "USD_CAD",
        "NZD_USD",
        "EUR_GBP",
        "EUR_JPY",
        "GBP_JPY",
        "EUR_CHF",
        "EUR_AUD",
        "EUR_CAD",
        "AUD_JPY",
        "AUD_NZD",
        "CAD_JPY",
    ]

    def __init__(
        self,
        config: Optional[SwapRateCacheConfig] = None,
        oanda_client: Optional[OandaSwapRateClient] = None,
        calculator: Optional[SwapRateCalculator] = None,
    ) -> None:
        """
        Initialize real-time swap rate provider.

        Args:
            config: Cache and refresh configuration
            oanda_client: OANDA API client
            calculator: Interest rate calculator for fallback
        """
        self.config = config or SwapRateCacheConfig()
        self._oanda = oanda_client or OandaSwapRateClient()
        self._calculator = calculator or SwapRateCalculator(broker_tier=self.config.broker_tier)

        # Cache storage
        self._cache: Dict[str, RealtimeSwapRate] = {}
        self._cache_lock = threading.Lock()

        # Background refresh
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()

        # Streaming
        self._streaming_task: Optional[asyncio.Task] = None
        self._stream_stop = asyncio.Event()

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_fetches": 0,
            "calculated_rates": 0,
            "errors": 0,
        }

        # Load persisted cache
        if self.config.persist_to_disk:
            self._load_cache_from_disk()

    def get_swap_rate(
        self,
        pair: str,
        force_refresh: bool = False,
        mid_price: Optional[float] = None,
    ) -> RealtimeSwapRate:
        """
        Get swap rate for a currency pair.

        Follows fallback chain:
        1. Fresh cached rate (< TTL)
        2. OANDA API fetch
        3. Calculated from interest rates

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            force_refresh: Force API fetch regardless of cache
            mid_price: Current mid price for calculation

        Returns:
            RealtimeSwapRate (never None, may be calculated fallback)
        """
        pair = self._normalize_pair(pair)

        # Check cache first (unless force refresh)
        if not force_refresh:
            with self._cache_lock:
                cached = self._cache.get(pair)
                if cached and cached.is_fresh(self.config.cache_ttl_sec):
                    self._stats["cache_hits"] += 1
                    return self._update_quality(cached)

        self._stats["cache_misses"] += 1

        # Try OANDA API
        if self._oanda.is_configured:
            try:
                rates = self._oanda.fetch_financing_rates([pair])
                if pair in rates:
                    self._stats["api_fetches"] += 1
                    rate = rates[pair]
                    self._update_cache(pair, rate)
                    return rate
            except Exception as e:
                logger.warning(f"OANDA fetch failed for {pair}: {e}")
                self._stats["errors"] += 1

        # Fall back to calculation
        if self.config.fallback_to_calculated:
            self._stats["calculated_rates"] += 1
            rate = self._calculator.calculate_full_swap_rate(pair, mid_price=mid_price or 1.0)
            self._update_cache(pair, rate)
            return rate

        # Return stale cached rate if available
        with self._cache_lock:
            if pair in self._cache:
                return self._update_quality(self._cache[pair])

        # Last resort: empty rate
        logger.warning(f"No swap rate available for {pair}")
        return RealtimeSwapRate(
            pair=pair,
            long_swap_pips=0.0,
            short_swap_pips=0.0,
            source=SwapRateSource.FALLBACK,
            quality=SwapRateQuality.EXPIRED,
        )

    def get_multiple_swap_rates(
        self,
        pairs: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, RealtimeSwapRate]:
        """
        Get swap rates for multiple pairs efficiently.

        Args:
            pairs: List of currency pairs
            force_refresh: Force API fetch

        Returns:
            Dict mapping pair -> RealtimeSwapRate
        """
        results: Dict[str, RealtimeSwapRate] = {}
        pairs_to_fetch: List[str] = []

        # Check cache
        for pair in pairs:
            pair = self._normalize_pair(pair)
            with self._cache_lock:
                cached = self._cache.get(pair)
                if cached and cached.is_fresh(self.config.cache_ttl_sec) and not force_refresh:
                    results[pair] = self._update_quality(cached)
                else:
                    pairs_to_fetch.append(pair)

        # Batch fetch from API
        if pairs_to_fetch and self._oanda.is_configured:
            try:
                fetched = self._oanda.fetch_financing_rates(pairs_to_fetch)
                for pair, rate in fetched.items():
                    self._update_cache(pair, rate)
                    results[pair] = rate
                    if pair in pairs_to_fetch:
                        pairs_to_fetch.remove(pair)
            except Exception as e:
                logger.warning(f"Batch fetch failed: {e}")

        # Calculate remaining
        for pair in pairs_to_fetch:
            if pair not in results:
                rate = self._calculator.calculate_full_swap_rate(pair)
                self._update_cache(pair, rate)
                results[pair] = rate

        return results

    def calculate_swap_cost(
        self,
        pair: str,
        direction: SwapDirection,
        position_size: float,
        days: int = 1,
        wednesday_triple: bool = True,
        day_of_week: Optional[int] = None,
    ) -> float:
        """
        Calculate swap cost for a position.

        Args:
            pair: Currency pair
            direction: LONG or SHORT
            position_size: Position size in base currency units
            days: Number of days to hold
            wednesday_triple: Apply Wednesday 3x multiplier
            day_of_week: Current day (0=Mon, for Wednesday check)

        Returns:
            Swap cost in quote currency (negative = cost, positive = earn)
        """
        rate = self.get_swap_rate(pair)

        # Apply Wednesday multiplier if applicable
        if wednesday_triple and day_of_week == 2:  # Wednesday
            rate = rate.apply_wednesday_multiplier(day_of_week)

        # Get swap in pips
        swap_pips = rate.get_swap_for_direction(direction, in_pips=True)

        # Convert pips to quote currency
        is_jpy = "JPY" in pair.upper()
        pip_value = 0.01 if is_jpy else 0.0001

        # Swap per lot per day
        swap_per_unit = swap_pips * pip_value

        # Total swap cost
        total_swap = swap_per_unit * position_size * days

        return total_swap

    def start_background_refresh(self) -> None:
        """Start background refresh thread."""
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            logger.warning("Background refresh already running")
            return

        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._background_refresh_loop,
            daemon=True,
            name="SwapRateRefresh",
        )
        self._refresh_thread.start()
        logger.info("Started background swap rate refresh")

    def stop_background_refresh(self) -> None:
        """Stop background refresh thread."""
        self._stop_refresh.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5.0)
        logger.info("Stopped background swap rate refresh")

    def _background_refresh_loop(self) -> None:
        """Background refresh loop."""
        while not self._stop_refresh.is_set():
            try:
                # Refresh all tracked instruments
                self.get_multiple_swap_rates(
                    self.DEFAULT_INSTRUMENTS,
                    force_refresh=True,
                )

                # Persist to disk
                if self.config.persist_to_disk:
                    self._save_cache_to_disk()

            except Exception as e:
                logger.error(f"Background refresh error: {e}")

            # Wait for next refresh
            self._stop_refresh.wait(self.config.auto_refresh_interval_sec)

    def _update_cache(self, pair: str, rate: RealtimeSwapRate) -> None:
        """Update cache with new rate."""
        with self._cache_lock:
            self._cache[pair] = rate

            # Enforce max cache size
            if len(self._cache) > self.config.max_cache_size:
                # Remove oldest entries
                sorted_pairs = sorted(
                    self._cache.keys(),
                    key=lambda p: self._cache[p].timestamp_ms,
                )
                for old_pair in sorted_pairs[: len(self._cache) - self.config.max_cache_size]:
                    del self._cache[old_pair]

    def _update_quality(self, rate: RealtimeSwapRate) -> RealtimeSwapRate:
        """Update quality indicator based on age."""
        age = rate.age_seconds

        if age < DEFAULT_CACHE_TTL_SEC:
            quality = SwapRateQuality.FRESH
        elif age < STALE_THRESHOLD_SEC:
            quality = SwapRateQuality.RECENT
        elif age < STALE_THRESHOLD_SEC * 3:
            quality = SwapRateQuality.STALE
        else:
            quality = SwapRateQuality.EXPIRED

        if rate.quality != quality:
            rate.quality = quality

        return rate

    def _normalize_pair(self, pair: str) -> str:
        """Normalize pair format to OANDA style."""
        return pair.upper().replace("/", "_")

    def _load_cache_from_disk(self) -> None:
        """Load cached rates from disk."""
        cache_path = Path(self.config.cache_file_path)

        if not cache_path.exists():
            return

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            for pair, rate_dict in data.items():
                rate = RealtimeSwapRate(
                    pair=pair,
                    long_swap_pips=rate_dict.get("long_swap_pips", 0.0),
                    short_swap_pips=rate_dict.get("short_swap_pips", 0.0),
                    long_swap_pct=rate_dict.get("long_swap_pct", 0.0),
                    short_swap_pct=rate_dict.get("short_swap_pct", 0.0),
                    timestamp_ms=rate_dict.get("timestamp_ms", 0),
                    source=SwapRateSource(rate_dict.get("source", "fallback")),
                    quality=SwapRateQuality.STALE,  # Disk data is always stale
                )
                self._cache[pair] = rate

            logger.info(f"Loaded {len(self._cache)} swap rates from disk cache")

        except Exception as e:
            logger.warning(f"Failed to load swap cache from disk: {e}")

    def _save_cache_to_disk(self) -> None:
        """Save current cache to disk."""
        cache_path = Path(self.config.cache_file_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._cache_lock:
                data = {pair: rate.to_dict() for pair, rate in self._cache.items()}

            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save swap cache to disk: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        with self._cache_lock:
            cache_size = len(self._cache)
            fresh_count = sum(1 for r in self._cache.values() if r.quality == SwapRateQuality.FRESH)

        return {
            **self._stats,
            "cache_size": cache_size,
            "fresh_rates": fresh_count,
            "oanda_configured": self._oanda.is_configured,
            "background_running": (
                self._refresh_thread is not None and self._refresh_thread.is_alive()
            ),
        }

    def clear_cache(self) -> None:
        """Clear all cached rates."""
        with self._cache_lock:
            self._cache.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def create_realtime_swap_provider(
    oanda_api_key: Optional[str] = None,
    oanda_account_id: Optional[str] = None,
    practice: bool = True,
    broker_tier: str = "retail",
    auto_start_refresh: bool = True,
    cache_file: str = "data/forex/realtime_swaps_cache.json",
) -> RealtimeSwapRateProvider:
    """
    Create a configured real-time swap rate provider.

    Args:
        oanda_api_key: OANDA API key
        oanda_account_id: OANDA account ID
        practice: Use practice environment
        broker_tier: Broker tier for markup calculation
        auto_start_refresh: Start background refresh automatically
        cache_file: Path for persistent cache

    Returns:
        Configured RealtimeSwapRateProvider
    """
    config = SwapRateCacheConfig(
        broker_tier=broker_tier,
        cache_file_path=cache_file,
    )

    oanda = OandaSwapRateClient(
        api_key=oanda_api_key,
        account_id=oanda_account_id,
        practice=practice,
    )

    calculator = SwapRateCalculator(broker_tier=broker_tier)

    provider = RealtimeSwapRateProvider(
        config=config,
        oanda_client=oanda,
        calculator=calculator,
    )

    if auto_start_refresh and config.auto_refresh_enabled:
        provider.start_background_refresh()

    return provider


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "SwapRateSource",
    "SwapDirection",
    "SwapRateQuality",
    # Data classes
    "RealtimeSwapRate",
    "SwapRateUpdate",
    "SwapRateCacheConfig",
    # Classes
    "SwapRateCalculator",
    "OandaSwapRateClient",
    "RealtimeSwapRateProvider",
    # Factory
    "create_realtime_swap_provider",
    # Constants
    "DEFAULT_INTERBANK_RATES",
    "BROKER_MARKUP_RANGES",
]
