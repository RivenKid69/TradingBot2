# -*- coding: utf-8 -*-
"""
impl_alpaca_public.py
Market data source for Alpaca US Equities.

This module provides a `MarketDataSource` implementation that uses
the Alpaca adapter for fetching historical and real-time stock data.

Usage:
    source = AlpacaBarSource(timeframe="1h")
    for bar in source.stream_bars(["AAPL", "MSFT"], interval_ms=3600000):
        process(bar)

Requirements:
    pip install alpaca-py
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from core_models import Bar, Tick
from core_contracts import MarketDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AlpacaWSConfig:
    """Configuration for Alpaca WebSocket connection."""

    api_key: str = ""
    api_secret: str = ""
    paper: bool = True
    feed: str = "iex"  # "iex" (free) or "sip" (paid)
    reconnect_backoff_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    vendor: str = "alpaca"


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================


def timeframe_to_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    tf_lower = tf.lower()

    mappings = {
        "1m": 60_000,
        "1min": 60_000,
        "5m": 300_000,
        "5min": 300_000,
        "15m": 900_000,
        "15min": 900_000,
        "30m": 1_800_000,
        "30min": 1_800_000,
        "1h": 3_600_000,
        "1hour": 3_600_000,
        "4h": 14_400_000,
        "4hour": 14_400_000,
        "1d": 86_400_000,
        "1day": 86_400_000,
    }

    if tf_lower in mappings:
        return mappings[tf_lower]

    # Try parsing numeric format
    import re

    match = re.match(r"(\d+)([mhd])", tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "m":
            return value * 60_000
        elif unit == "h":
            return value * 3_600_000
        elif unit == "d":
            return value * 86_400_000

    raise ValueError(f"Unsupported timeframe: {tf}")


def ensure_timeframe(tf: str) -> None:
    """Validate timeframe string."""
    try:
        timeframe_to_ms(tf)
    except ValueError as e:
        raise ValueError(f"Invalid timeframe '{tf}': {e}")


# =============================================================================
# ALPACA BAR SOURCE
# =============================================================================


class AlpacaBarSource(MarketDataSource):
    """
    Market data source for Alpaca US Equities.

    Implements the MarketDataSource protocol to provide real-time
    and historical bar data from Alpaca.

    Args:
        timeframe: Bar timeframe (e.g., "1h", "15m", "1d")
        cfg: Optional configuration

    Example:
        source = AlpacaBarSource(timeframe="1h")
        for bar in source.stream_bars(["AAPL"], interval_ms=3600000):
            print(bar)
    """

    def __init__(
        self,
        timeframe: str,
        cfg: Optional[AlpacaWSConfig] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ensure_timeframe(timeframe)

        self._timeframe = timeframe
        self._interval_ms = timeframe_to_ms(timeframe)

        # Merge config sources
        if cfg is not None:
            self._cfg = cfg
        elif config is not None:
            self._cfg = AlpacaWSConfig(
                api_key=config.get("api_key", ""),
                api_secret=config.get("api_secret", ""),
                paper=config.get("paper", True),
                feed=config.get("feed", "iex"),
            )
        else:
            self._cfg = AlpacaWSConfig()

        self._symbols: List[str] = []
        self._q: "queue.Queue[Bar]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._adapter = None
        self._last_open_ts: Dict[str, int] = {}

    def _get_adapter(self):
        """Lazy initialization of Alpaca adapter."""
        if self._adapter is None:
            try:
                from adapters.alpaca import AlpacaMarketDataAdapter

                self._adapter = AlpacaMarketDataAdapter(
                    config={
                        "api_key": self._cfg.api_key,
                        "api_secret": self._cfg.api_secret,
                        "paper": self._cfg.paper,
                        "feed": self._cfg.feed,
                    }
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import Alpaca adapter: {e}. "
                    "Make sure alpaca-py is installed: pip install alpaca-py"
                )
        return self._adapter

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream real-time bars via WebSocket.

        Args:
            symbols: List of stock symbols to stream
            interval_ms: Bar interval in milliseconds (must match timeframe)

        Yields:
            Bar objects as they become available
        """
        if interval_ms != self._interval_ms:
            raise ValueError(
                f"Timeframe mismatch. Source={self._interval_ms}ms, requested={interval_ms}ms"
            )

        if not symbols:
            raise ValueError("No symbols provided")

        self._symbols = [s.upper() for s in symbols]
        self._stop.clear()

        # Start WebSocket thread
        self._thr = threading.Thread(
            target=self._run_websocket,
            name="alpaca-ws",
            daemon=True,
        )
        self._thr.start()

        try:
            while not self._stop.is_set():
                try:
                    bar = self._q.get(timeout=0.5)
                    yield bar
                except queue.Empty:
                    continue
        finally:
            self.close()

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        """
        Stream real-time ticks (quotes).

        Args:
            symbols: List of stock symbols

        Yields:
            Tick objects
        """
        if not symbols:
            return iter([])

        self._symbols = [s.upper() for s in symbols]
        self._stop.clear()

        # Use adapter's stream_ticks
        adapter = self._get_adapter()

        try:
            yield from adapter.stream_ticks(self._symbols)
        except Exception as e:
            logger.error(f"Error streaming ticks: {e}")
            return iter([])

    def get_historical_bars(
        self,
        symbol: str,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical bars.

        Args:
            symbol: Stock symbol
            limit: Maximum number of bars
            start_ts: Start timestamp (ms)
            end_ts: End timestamp (ms)

        Returns:
            List of Bar objects
        """
        adapter = self._get_adapter()
        return adapter.get_bars(
            symbol=symbol,
            timeframe=self._timeframe,
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def close(self) -> None:
        """Stop streaming and cleanup resources."""
        self._stop.set()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=5.0)
        if self._adapter:
            self._adapter.disconnect()

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _run_websocket(self) -> None:
        """Run WebSocket client in background thread."""
        try:
            asyncio.run(self._websocket_client())
        except RuntimeError:
            # Handle nested event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._websocket_client())
            finally:
                loop.close()

    async def _websocket_client(self) -> None:
        """Async WebSocket client for Alpaca data stream."""
        backoff = self._cfg.reconnect_backoff_s

        while not self._stop.is_set():
            try:
                await self._connect_and_stream()
                backoff = self._cfg.reconnect_backoff_s  # Reset on success
            except Exception as e:
                logger.warning(f"Alpaca WebSocket error: {e}, reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self._cfg.reconnect_backoff_max_s)

    async def _connect_and_stream(self) -> None:
        """Connect to Alpaca and stream bars."""
        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            logger.error("alpaca-py not installed")
            self._stop.set()
            return

        stream = StockDataStream(
            api_key=self._cfg.api_key,
            secret_key=self._cfg.api_secret,
            feed=self._cfg.feed,
        )

        # Handler for bar updates
        async def handle_bar(bar):
            try:
                converted = self._convert_bar(bar)
                if converted:
                    self._enqueue_bar(converted)
            except Exception as e:
                logger.debug(f"Bar conversion error: {e}")

        # Subscribe to bars
        stream.subscribe_bars(handle_bar, *self._symbols)

        # Run stream (blocking)
        try:
            await stream._run_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await stream.close()

    def _convert_bar(self, alpaca_bar: Any) -> Optional[Bar]:
        """Convert Alpaca bar to internal Bar model."""
        try:
            ts = int(alpaca_bar.timestamp.timestamp() * 1000)

            return Bar(
                ts=ts,
                symbol=str(alpaca_bar.symbol).upper(),
                open=Decimal(str(alpaca_bar.open)),
                high=Decimal(str(alpaca_bar.high)),
                low=Decimal(str(alpaca_bar.low)),
                close=Decimal(str(alpaca_bar.close)),
                volume_base=Decimal(str(alpaca_bar.volume)),
                volume_quote=(
                    Decimal(str(alpaca_bar.vwap * alpaca_bar.volume)) if alpaca_bar.vwap else None
                ),
                trades=getattr(alpaca_bar, "trade_count", None),
                vwap=Decimal(str(alpaca_bar.vwap)) if alpaca_bar.vwap else None,
                is_final=True,
            )
        except Exception as e:
            logger.debug(f"Bar conversion error: {e}")
            return None

    def _enqueue_bar(self, bar: Bar) -> None:
        """Add bar to queue with duplicate/gap detection."""
        bar_open_ms = bar.ts - self._interval_ms  # Approximate open time
        prev_open = self._last_open_ts.get(bar.symbol)

        # Check for duplicates
        if prev_open is not None:
            if bar_open_ms <= prev_open:
                logger.debug(f"Duplicate bar for {bar.symbol}")
                return

            # Check for gaps
            delta = bar_open_ms - prev_open
            if self._interval_ms > 0 and delta > self._interval_ms * 1.5:
                logger.warning(
                    f"Bar gap detected for {bar.symbol}: {delta}ms "
                    f"(expected {self._interval_ms}ms)"
                )

        self._last_open_ts[bar.symbol] = bar_open_ms

        try:
            self._q.put_nowait(bar)
        except queue.Full:
            # Drop oldest bar to make room
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(bar)


# =============================================================================
# OFFLINE BAR SOURCE FOR ALPACA
# =============================================================================


class AlpacaOfflineBarSource(MarketDataSource):
    """
    Offline bar source that loads historical data from Alpaca.

    Useful for backtesting with Alpaca historical data.

    Args:
        timeframe: Bar timeframe
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        config: Alpaca configuration
    """

    def __init__(
        self,
        timeframe: str,
        start_date: str,
        end_date: str,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._timeframe = timeframe
        self._interval_ms = timeframe_to_ms(timeframe)
        self._start_date = start_date
        self._end_date = end_date
        self._config = config or {}
        self._adapter = None
        self._bars_cache: Dict[str, List[Bar]] = {}

    def _get_adapter(self):
        """Lazy initialization of adapter."""
        if self._adapter is None:
            from adapters.alpaca import AlpacaMarketDataAdapter

            self._adapter = AlpacaMarketDataAdapter(config=self._config)
        return self._adapter

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream historical bars (offline mode).

        Args:
            symbols: List of stock symbols
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects in chronological order
        """
        if interval_ms != self._interval_ms:
            raise ValueError(
                f"Timeframe mismatch. Source={self._interval_ms}ms, requested={interval_ms}ms"
            )

        # Load all bars for all symbols
        all_bars: List[Bar] = []
        adapter = self._get_adapter()

        for symbol in symbols:
            symbol_upper = symbol.upper()

            if symbol_upper not in self._bars_cache:
                bars = adapter.get_bars(
                    symbol=symbol_upper,
                    timeframe=self._timeframe,
                    limit=10000,  # Fetch max available
                )
                self._bars_cache[symbol_upper] = bars

            all_bars.extend(self._bars_cache[symbol_upper])

        # Sort by timestamp and yield
        all_bars.sort(key=lambda b: b.ts)
        yield from all_bars

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        """Offline source doesn't support tick streaming."""
        return iter([])


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_alpaca_bar_source(
    timeframe: str = "1h",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    paper: bool = True,
    feed: str = "iex",
) -> AlpacaBarSource:
    """
    Factory function to create Alpaca bar source.

    Args:
        timeframe: Bar timeframe
        api_key: Alpaca API key (or use env var ALPACA_API_KEY)
        api_secret: Alpaca API secret (or use env var ALPACA_API_SECRET)
        paper: Use paper trading
        feed: Data feed ("iex" or "sip")

    Returns:
        AlpacaBarSource instance
    """
    import os

    config = AlpacaWSConfig(
        api_key=api_key or os.environ.get("ALPACA_API_KEY", ""),
        api_secret=api_secret or os.environ.get("ALPACA_API_SECRET", ""),
        paper=paper,
        feed=feed,
    )

    return AlpacaBarSource(timeframe=timeframe, cfg=config)
