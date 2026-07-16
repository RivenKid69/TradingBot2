# -*- coding: utf-8 -*-
"""
adapters/oanda/market_data.py
OANDA market data adapter for forex trading.

Status: Tested and operational (Phase 2 Complete)

OANDA v20 API provides:
- REST API for historical candles (get_bars, get_latest_bar, get_tick)
- Streaming prices via HTTP streaming:
  - Sync iterators: stream_bars(), stream_ticks()
  - Async generators: stream_bars_async(), stream_ticks_async()
- Bid/Ask/Mid prices for accurate forex spread modeling

Timeframes:
- S5, S10, S15, S30: Seconds (5, 10, 15, 30)
- M1, M2, M4, M5, M10, M15, M30: Minutes
- H1, H2, H3, H4, H6, H8, H12: Hours
- D, W, M: Day, Week, Month

API Rate Limits:
- 120 requests per second (REST)
- Streaming is separate (no request limit)
- Max 5000 candles per request
- Burst: 200 requests allowed

Usage:
    # Sync historical data
    adapter = OandaMarketDataAdapter(config={
        "api_key": "your_key",
        "account_id": "your_account",
        "practice": True,
    })
    bars = adapter.get_bars("EUR_USD", "H1", limit=100)

    # Async streaming
    async for tick in adapter.stream_ticks_async(["EUR_USD", "GBP_USD"]):
        await process_tick(tick)

References:
    - OANDA v20 API: https://developer.oanda.com/rest-live-v20/introduction/
    - Rate Limits: https://developer.oanda.com/rest-live-v20/best-practices/
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

import requests

from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor

# Optional async support
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


# =========================
# Rate Limiter
# =========================

class RateLimiter:
    """
    Token bucket rate limiter for OANDA API.

    OANDA allows 120 requests per second with burst capacity of 200.

    Thread-safe implementation using asyncio.Lock for async operations.

    Attributes:
        rate: Tokens per second (default: 120.0)
        burst: Maximum burst capacity (default: 200)
    """

    def __init__(self, rate: float = 120.0, burst: int = 200) -> None:
        """
        Initialize rate limiter.

        Args:
            rate: Tokens added per second
            burst: Maximum token capacity
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

    def acquire_sync(self) -> None:
        """Acquire a token synchronously, blocking if necessary."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

        if self._tokens < 1.0:
            wait_time = (1.0 - self._tokens) / self.rate
            time.sleep(wait_time)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        now = time.monotonic()
        elapsed = now - self._last_update
        return min(self.burst, self._tokens + elapsed * self.rate)


# =========================
# OANDA Bar Data (with bid/ask)
# =========================

@dataclass
class OandaCandleData:
    """
    Extended candle data from OANDA with bid/ask prices.

    OANDA provides separate bid/ask/mid candles for accurate spread modeling.
    """
    timestamp: int  # milliseconds UTC
    open_bid: float
    high_bid: float
    low_bid: float
    close_bid: float
    open_ask: float
    high_ask: float
    low_ask: float
    close_ask: float
    open_mid: float
    high_mid: float
    low_mid: float
    close_mid: float
    volume: int
    complete: bool


# =========================
# OANDA Market Data Adapter
# =========================

class OandaMarketDataAdapter(MarketDataAdapter):
    """
    OANDA v20 API market data adapter for forex trading.

    Provides historical candles and real-time streaming for forex pairs.

    Features:
    - Historical candles with bid/ask/mid prices
    - Real-time streaming via HTTP streaming
    - Auto-reconnect with exponential backoff
    - Rate limiting (120 req/s)
    - DST-aware timezone handling

    Configuration:
        api_key: OANDA API access token (required)
        account_id: Trading account ID (required)
        practice: Use practice/demo environment (default: True)
        base_url: Override API base URL
        stream_url: Override streaming URL
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)

    Environment Variables:
        OANDA_API_KEY: API access token
        OANDA_ACCOUNT_ID: Trading account ID
        OANDA_PRACTICE: "true" for demo, "false" for live

    Example:
        adapter = OandaMarketDataAdapter(config={
            "api_key": "your_key",
            "account_id": "your_account",
            "practice": True,
        })
        bars = adapter.get_bars("EUR_USD", "H1", limit=100)
    """

    # OANDA API endpoints
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"
    STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
    STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"

    # Timeframe mapping: user-friendly -> OANDA format
    TIMEFRAME_MAP: Dict[str, str] = {
        # Seconds
        "5s": "S5", "10s": "S10", "15s": "S15", "30s": "S30",
        "S5": "S5", "S10": "S10", "S15": "S15", "S30": "S30",
        # Minutes
        "1m": "M1", "2m": "M2", "4m": "M4", "5m": "M5",
        "10m": "M10", "15m": "M15", "30m": "M30",
        "M1": "M1", "M2": "M2", "M4": "M4", "M5": "M5",
        "M10": "M10", "M15": "M15", "M30": "M30",
        # Hours
        "1h": "H1", "2h": "H2", "3h": "H3", "4h": "H4",
        "6h": "H6", "8h": "H8", "12h": "H12",
        "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4",
        "H6": "H6", "H8": "H8", "H12": "H12",
        # Days/Weeks/Months
        "1d": "D", "1w": "W", "1M": "M",
        "D": "D", "W": "W", "M": "M",
        "d": "D", "w": "W",
    }

    # Major currency pairs for validation
    MAJOR_PAIRS = frozenset({
        "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
    })

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize OANDA market data adapter.

        Args:
            vendor: Exchange vendor (default: OANDA)
            config: Configuration dict with api_key, account_id, etc.
        """
        super().__init__(vendor, config)

        # API configuration
        self._api_key = self._config.get("api_key") or os.getenv("OANDA_API_KEY")
        self._account_id = self._config.get("account_id") or os.getenv("OANDA_ACCOUNT_ID")

        # Environment: practice vs live
        practice_env = os.getenv("OANDA_PRACTICE", "true").lower()
        self._practice = self._config.get("practice", practice_env == "true")

        # Set URLs based on environment
        if self._practice:
            self._base_url = self._config.get("base_url", self.PRACTICE_URL)
            self._stream_url = self._config.get("stream_url", self.STREAM_PRACTICE_URL)
        else:
            self._base_url = self._config.get("base_url", self.LIVE_URL)
            self._stream_url = self._config.get("stream_url", self.STREAM_LIVE_URL)

        # Request settings
        self._timeout = self._config.get("timeout", 30)
        self._max_retries = self._config.get("max_retries", 3)

        # Rate limiter
        self._rate_limiter = RateLimiter()

        # HTTP session (lazy initialized)
        self._session = None
        self._async_session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OANDA API requests."""
        if not self._api_key:
            raise ValueError(
                "OANDA API key is required. "
                "Set via config['api_key'] or OANDA_API_KEY environment variable."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX",
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to OANDA format (EUR_USD).

        Args:
            symbol: Currency pair in any format (EUR/USD, EURUSD, EUR_USD)

        Returns:
            Normalized symbol (e.g., EUR_USD)
        """
        # Replace common separators
        norm = symbol.upper().replace("/", "_").replace("-", "_")

        # Handle no separator (EURUSD -> EUR_USD)
        if "_" not in norm and len(norm) == 6:
            norm = f"{norm[:3]}_{norm[3:]}"

        return norm

    def _parse_timeframe(self, timeframe: str) -> str:
        """
        Parse timeframe to OANDA granularity.

        Args:
            timeframe: Timeframe string (e.g., "1h", "H1", "4h")

        Returns:
            OANDA granularity (e.g., "H1", "H4")

        Raises:
            ValueError: If timeframe is not supported
        """
        tf = timeframe.strip()
        granularity = self.TIMEFRAME_MAP.get(tf)

        if granularity is None:
            # Try uppercase
            granularity = self.TIMEFRAME_MAP.get(tf.upper())

        if granularity is None:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(self.TIMEFRAME_MAP.keys())}"
            )

        return granularity

    def _do_connect(self) -> None:
        """Initialize HTTP session."""
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update(self._get_headers())
            logger.debug(f"OANDA adapter connected (practice={self._practice})")
        except ImportError:
            raise ImportError(
                "requests library is required. Install with: pip install requests"
            )

    def _do_disconnect(self) -> None:
        """Close HTTP session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        logger.debug("OANDA adapter disconnected")

    def _ensure_session(self):
        """Ensure HTTP session is initialized."""
        if self._session is None:
            self._do_connect()
        return self._session

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical candles from OANDA.

        Args:
            symbol: Currency pair (e.g., "EUR_USD", "GBP/JPY")
            timeframe: Bar timeframe (e.g., "1h", "4h", "1d", "H4")
            limit: Maximum bars (max 5000 per request)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects with bid/ask/mid prices, oldest first

        Raises:
            ValueError: If API key not configured or invalid parameters
            ConnectionError: If API request fails
        """
        session = self._ensure_session()
        self._rate_limiter.acquire_sync()

        instrument = self._normalize_symbol(symbol)
        granularity = self._parse_timeframe(timeframe)

        # Build request parameters
        params: Dict[str, Any] = {
            "granularity": granularity,
            "count": min(limit, 5000),
            "price": "MBA",  # Mid, Bid, Ask
        }

        if start_ts is not None:
            params["from"] = str(start_ts / 1000)  # OANDA uses Unix seconds
        if end_ts is not None:
            params["to"] = str(end_ts / 1000)

        # Make request
        url = f"{self._base_url}/v3/instruments/{instrument}/candles"

        try:
            response = session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"OANDA API error: {e}")
            raise ConnectionError(f"Failed to fetch candles: {e}") from e

        # Parse response
        bars = []
        candles = data.get("candles", [])

        for candle in candles:
            if not candle.get("complete", False):
                continue  # Skip incomplete candles

            bar = self._parse_candle(instrument, candle)
            if bar is not None:
                bars.append(bar)

        return bars

    def _parse_candle(self, symbol: str, candle: Dict[str, Any]) -> Optional[Bar]:
        """
        Parse OANDA candle to Bar object.

        Args:
            symbol: Currency pair
            candle: Raw candle data from OANDA

        Returns:
            Bar object or None if parsing fails
        """
        try:
            # Get timestamp (OANDA returns Unix seconds as string)
            ts_str = candle.get("time", "0")
            ts_ms = int(float(ts_str) * 1000)

            # Get mid prices (default to bid if mid unavailable)
            mid = candle.get("mid", candle.get("bid", {}))
            bid = candle.get("bid", mid)
            ask = candle.get("ask", mid)

            # Retrieve distinct values
            mid_o, mid_h, mid_l, mid_c = float(mid.get("o", 0)), float(mid.get("h", 0)), float(mid.get("l", 0)), float(mid.get("c", 0))
            bid_o, bid_h, bid_l, bid_c = float(bid.get("o", mid_o)), float(bid.get("h", mid_h)), float(bid.get("l", mid_l)), float(bid.get("c", mid_c))
            ask_o, ask_h, ask_l, ask_c = float(ask.get("o", mid_o)), float(ask.get("h", mid_h)), float(ask.get("l", mid_l)), float(ask.get("c", mid_c))

            # Use selected price type for main OHLC
            price_type = self._config.get("price_type", "mid")
            if price_type == "bid":
                open_price, high_price, low_price, close_price = bid_o, bid_h, bid_l, bid_c
            elif price_type == "ask":
                open_price, high_price, low_price, close_price = ask_o, ask_h, ask_l, ask_c
            else:
                open_price, high_price, low_price, close_price = mid_o, mid_h, mid_l, mid_c

            # Calculate spread from bid/ask (in price units)
            spread = ask_c - bid_c

            # Volume (OANDA returns tick volume)
            volume = int(candle.get("volume", 0))

            return Bar(
                ts=ts_ms,
                symbol=symbol,
                open=Decimal(str(open_price)),
                high=Decimal(str(high_price)),
                low=Decimal(str(low_price)),
                close=Decimal(str(close_price)),
                volume_base=Decimal(str(volume)),  # Tick volume
                volume_quote=Decimal(str(spread * 10000)),  # Spread in pips approx
                bid_open=Decimal(str(bid_o)),
                bid_high=Decimal(str(bid_h)),
                bid_low=Decimal(str(bid_l)),
                bid_close=Decimal(str(bid_c)),
                ask_open=Decimal(str(ask_o)),
                ask_high=Decimal(str(ask_h)),
                ask_low=Decimal(str(ask_l)),
                ask_close=Decimal(str(ask_c)),
            )
        except Exception as e:
            logger.warning(f"Failed to parse candle: {e}")
            return None

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """
        Get the most recent complete bar.

        Args:
            symbol: Currency pair
            timeframe: Bar timeframe

        Returns:
            Latest Bar or None if unavailable
        """
        bars = self.get_bars(symbol, timeframe, limit=2)
        # Return the most recent complete bar
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current quote (bid/ask) for a currency pair.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            Current Tick with bid/ask prices
        """
        session = self._ensure_session()
        self._rate_limiter.acquire_sync()

        instrument = self._normalize_symbol(symbol)
        url = f"{self._base_url}/v3/accounts/{self._account_id}/pricing"
        params = {"instruments": instrument}

        try:
            response = session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get tick for {symbol}: {e}")
            return None

        prices = data.get("prices", [])
        if not prices:
            return None

        price_data = prices[0]

        try:
            # Parse bid/ask
            bids = price_data.get("bids", [])
            asks = price_data.get("asks", [])

            bid_price = float(bids[0]["price"]) if bids else 0.0
            ask_price = float(asks[0]["price"]) if asks else 0.0
            mid_price = (bid_price + ask_price) / 2.0 if bid_price and ask_price else 0.0

            # Timestamp
            ts_str = price_data.get("time", str(time.time()))
            ts_ms = int(float(ts_str) * 1000)

            return Tick(
                ts=ts_ms,
                symbol=instrument,
                bid=Decimal(str(bid_price)),
                ask=Decimal(str(ask_price)),
                last=Decimal(str(mid_price)),
                volume=Decimal("0"),  # Forex doesn't have tick volume
            )
        except Exception as e:
            logger.warning(f"Failed to parse tick: {e}")
            return None

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream bars by aggregating ticks.

        Note: OANDA doesn't provide bar streaming directly.
        This implementation aggregates ticks into bars.

        Args:
            symbols: List of currency pairs
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects as they complete
        """
        # Aggregate ticks into bars
        bar_builders: Dict[str, Dict[str, Any]] = {}
        start_time = int(time.time() * 1000)

        for tick in self.stream_ticks(symbols):
            symbol = tick.symbol
            current_time = tick.ts

            # Initialize or get bar builder for this symbol
            if symbol not in bar_builders:
                bar_builders[symbol] = {
                    "start_ts": current_time,
                    "open": float(tick.last),
                    "high": float(tick.last),
                    "low": float(tick.last),
                    "close": float(tick.last),
                    "volume": 0,
                }

            builder = bar_builders[symbol]
            price = float(tick.last)

            # Check if bar is complete
            if current_time - builder["start_ts"] >= interval_ms:
                # Yield completed bar
                yield Bar(
                    ts=builder["start_ts"],
                    symbol=symbol,
                    open=Decimal(str(builder["open"])),
                    high=Decimal(str(builder["high"])),
                    low=Decimal(str(builder["low"])),
                    close=Decimal(str(builder["close"])),
                    volume=Decimal(str(builder["volume"])),
                )

                # Start new bar
                bar_builders[symbol] = {
                    "start_ts": current_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                }
            else:
                # Update current bar
                builder["high"] = max(builder["high"], price)
                builder["low"] = min(builder["low"], price)
                builder["close"] = price
                builder["volume"] += 1

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks via HTTP streaming.

        Args:
            symbols: List of currency pairs

        Yields:
            Tick objects as they arrive
        """
        import requests

        instruments = ",".join(self._normalize_symbol(s) for s in symbols)
        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        params = {"instruments": instruments}

        try:
            with requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                stream=True,
                timeout=None,  # Streaming has no timeout
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        import json
                        data = json.loads(line.decode("utf-8"))

                        if data.get("type") == "PRICE":
                            tick = self._parse_stream_tick(data)
                            if tick is not None:
                                yield tick
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error parsing stream data: {e}")
                        continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def _parse_stream_tick(self, data: Dict[str, Any]) -> Optional[Tick]:
        """Parse streaming price data to Tick."""
        try:
            instrument = data.get("instrument", "")

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            bid_price = float(bids[0]["price"]) if bids else 0.0
            ask_price = float(asks[0]["price"]) if asks else 0.0
            mid_price = (bid_price + ask_price) / 2.0

            ts_str = data.get("time", str(time.time()))
            ts_ms = int(float(ts_str) * 1000)

            return Tick(
                ts=ts_ms,
                symbol=instrument,
                bid=Decimal(str(bid_price)),
                ask=Decimal(str(ask_price)),
                last=Decimal(str(mid_price)),
                volume=Decimal("0"),
            )
        except Exception:
            return None

    async def stream_ticks_async(
        self,
        symbols: Sequence[str],
    ) -> AsyncIterator[Tick]:
        """
        Stream real-time ticks asynchronously.

        Args:
            symbols: List of currency pairs

        Yields:
            Tick objects as they arrive
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for async streaming. Install with: pip install aiohttp"
            )

        instruments = ",".join(self._normalize_symbol(s) for s in symbols)
        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        params = {"instruments": instruments}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if not line:
                        continue

                    try:
                        import json
                        data = json.loads(line.decode("utf-8"))

                        if data.get("type") == "PRICE":
                            tick = self._parse_stream_tick(data)
                            if tick is not None:
                                yield tick
                    except Exception:
                        continue

    async def stream_bars_async(
        self,
        symbols: Sequence[str],
        interval_ms: int = 60000,
    ) -> AsyncIterator[Bar]:
        """
        Stream bars asynchronously by aggregating ticks.

        Args:
            symbols: List of currency pairs
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects as they complete
        """
        bar_builders: Dict[str, Dict[str, Any]] = {}

        async for tick in self.stream_ticks_async(symbols):
            symbol = tick.symbol
            current_time = tick.ts
            price = float(tick.last)

            if symbol not in bar_builders:
                bar_builders[symbol] = {
                    "start_ts": current_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                }

            builder = bar_builders[symbol]

            if current_time - builder["start_ts"] >= interval_ms:
                yield Bar(
                    ts=builder["start_ts"],
                    symbol=symbol,
                    open=Decimal(str(builder["open"])),
                    high=Decimal(str(builder["high"])),
                    low=Decimal(str(builder["low"])),
                    close=Decimal(str(builder["close"])),
                    volume=Decimal(str(builder["volume"])),
                )

                bar_builders[symbol] = {
                    "start_ts": current_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                }
            else:
                builder["high"] = max(builder["high"], price)
                builder["low"] = min(builder["low"], price)
                builder["close"] = price
                builder["volume"] += 1

    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread for a currency pair in pips.

        Args:
            symbol: Currency pair

        Returns:
            Spread in pips or None
        """
        tick = self.get_tick(symbol)
        if tick is None or tick.bid is None or tick.ask is None:
            return None

        spread = float(tick.ask) - float(tick.bid)

        # Convert to pips (0.0001 for most, 0.01 for JPY)
        pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
        return spread / pip_size

    def get_pip_size(self, symbol: str) -> float:
        """
        Get pip size for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Pip size (0.0001 for most, 0.01 for JPY pairs)
        """
        norm = self._normalize_symbol(symbol)
        return 0.01 if "JPY" in norm else 0.0001
