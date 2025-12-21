# -*- coding: utf-8 -*-
"""
adapters/polygon/market_data.py
Polygon.io market data adapter implementation.

This module provides a MarketDataAdapter for Polygon.io, supporting:
- Historical OHLCV bars (stocks, crypto, forex)
- Real-time bar streaming via WebSocket
- Technical indicator data (SMA, EMA, RSI, MACD)

IMPLEMENTATION STATUS
=====================
- Historical bars: ✅ Implemented
- Bar streaming: ✅ Implemented
- Tick streaming: ⚠️ Returns empty iterator (see stream_ticks method)

Tick streaming via Polygon WebSocket "T.*" channels is not yet implemented.
Use bar streaming or historical bars for production use cases.

Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#adapter-polygon-tick-streaming

API Reference:
    https://polygon.io/docs/stocks/getting-started

Rate Limits:
    - Basic: 5 calls/minute
    - Starter: unlimited REST, delayed WebSocket
    - Developer+: unlimited REST, real-time WebSocket

Usage:
    adapter = PolygonMarketDataAdapter(config={"api_key": "..."})

    # Historical bars
    bars = adapter.get_bars("AAPL", "1h", limit=100)

    # Real-time streaming (bars only; tick streaming not implemented)
    for bar in adapter.stream_bars(["AAPL", "MSFT"], interval_ms=60000):
        process(bar)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from core_models import Bar, Tick

from ..base import MarketDataAdapter
from ..models import ExchangeVendor, MarketType

logger = logging.getLogger(__name__)


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================

def _timeframe_to_polygon(tf: str) -> tuple[int, str]:
    """
    Convert timeframe string to Polygon API format.

    Args:
        tf: Timeframe string (e.g., "1m", "1h", "1d")

    Returns:
        (multiplier, timespan) tuple for Polygon API

    Raises:
        ValueError: If timeframe not supported
    """
    tf_lower = tf.lower().strip()

    mappings = {
        "1m": (1, "minute"),
        "1min": (1, "minute"),
        "5m": (5, "minute"),
        "5min": (5, "minute"),
        "15m": (15, "minute"),
        "15min": (15, "minute"),
        "30m": (30, "minute"),
        "30min": (30, "minute"),
        "1h": (1, "hour"),
        "1hour": (1, "hour"),
        "4h": (4, "hour"),
        "4hour": (4, "hour"),
        "1d": (1, "day"),
        "1day": (1, "day"),
        "1w": (1, "week"),
        "1week": (1, "week"),
    }

    if tf_lower in mappings:
        return mappings[tf_lower]

    # Try parsing numeric format
    import re
    match = re.match(r"(\d+)([mhdw])", tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        unit_map = {"m": "minute", "h": "hour", "d": "day", "w": "week"}
        if unit in unit_map:
            return (value, unit_map[unit])

    raise ValueError(f"Unsupported timeframe: {tf}")


def _timeframe_to_ms(tf: str) -> int:
    """Convert timeframe to milliseconds."""
    mult, span = _timeframe_to_polygon(tf)

    ms_per_unit = {
        "minute": 60_000,
        "hour": 3_600_000,
        "day": 86_400_000,
        "week": 604_800_000,
    }

    return mult * ms_per_unit.get(span, 60_000)


# =============================================================================
# POLYGON MARKET DATA ADAPTER
# =============================================================================

class PolygonMarketDataAdapter(MarketDataAdapter):
    """
    Market data adapter for Polygon.io.

    Provides access to:
    - Historical OHLCV bars for US stocks, options, forex, crypto
    - Real-time streaming via WebSocket
    - Latest tick data

    Configuration:
        api_key: Polygon.io API key (or env POLYGON_API_KEY)
        timeout: Request timeout in seconds (default: 30)
        retries: Number of retry attempts (default: 3)

    Example:
        adapter = PolygonMarketDataAdapter(config={
            "api_key": "your_api_key",
        })

        # Get historical bars
        bars = adapter.get_bars("AAPL", "1h", limit=100)

        # Stream real-time bars
        for bar in adapter.stream_bars(["AAPL"], 60000):
            print(bar)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.POLYGON,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor=vendor, config=config)

        # Configuration
        self._api_key = self._config.get("api_key") or os.environ.get("POLYGON_API_KEY", "")
        self._timeout = float(self._config.get("timeout", 30))
        self._retries = int(self._config.get("retries", 3))
        self._base_url = self._config.get("base_url", "https://api.polygon.io")

        # Lazy-loaded client
        self._rest_client: Optional[Any] = None
        self._ws_client: Optional[Any] = None

        # Streaming state
        self._streaming_symbols: List[str] = []
        self._bar_queue: "queue.Queue[Bar]" = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None

    def _get_rest_client(self) -> Any:
        """Lazy initialization of REST client."""
        if self._rest_client is None:
            if not self._api_key:
                raise ValueError(
                    "Polygon API key required. Set POLYGON_API_KEY env var or pass in config."
                )

            try:
                from polygon import RESTClient
                self._rest_client = RESTClient(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "polygon-api-client not installed. Run: pip install polygon-api-client"
                )

        return self._rest_client

    def _do_connect(self) -> None:
        """Verify API connection."""
        client = self._get_rest_client()
        # Test connection with a simple request
        try:
            client.get_market_status()
            logger.info("Polygon API connection verified")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Polygon API: {e}")

    def _do_disconnect(self) -> None:
        """Cleanup resources."""
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        self._rest_client = None
        self._ws_client = None

    # -------------------------------------------------------------------------
    # MarketDataAdapter Interface
    # -------------------------------------------------------------------------

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
        Fetch historical OHLCV bars from Polygon.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")
            limit: Maximum bars to return (default 500, max 50000)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects, oldest first
        """
        client = self._get_rest_client()

        # Convert timeframe
        multiplier, timespan = _timeframe_to_polygon(timeframe)

        # Date range
        if end_ts is None:
            end_date = datetime.now(timezone.utc)
        else:
            end_date = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)

        if start_ts is None:
            # Calculate start based on limit and timeframe
            tf_ms = _timeframe_to_ms(timeframe)
            start_date = end_date - timedelta(milliseconds=tf_ms * limit * 1.5)
        else:
            start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        # Format dates for API
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        logger.debug(
            f"Fetching {symbol} bars: {timeframe} from {from_str} to {to_str} (limit={limit})"
        )

        try:
            # Fetch bars from Polygon
            aggs = client.get_aggs(
                ticker=symbol.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=from_str,
                to=to_str,
                limit=min(limit, 50000),
                sort="asc",
            )

            bars: List[Bar] = []
            for agg in aggs:
                bar = self._convert_agg_to_bar(agg, symbol)
                bars.append(bar)

            # Apply limit
            if len(bars) > limit:
                bars = bars[-limit:]

            logger.debug(f"Fetched {len(bars)} bars for {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return []

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """Get the most recent bar for a symbol."""
        bars = self.get_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current tick (last trade) for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Tick with last trade info
        """
        client = self._get_rest_client()

        try:
            # Get last trade
            trade = client.get_last_trade(symbol.upper())

            if trade:
                ts = int(trade.sip_timestamp / 1_000_000)  # Nanoseconds to milliseconds
                return Tick(
                    ts=ts,
                    symbol=symbol.upper(),
                    price=Decimal(str(trade.price)),
                    bid=None,  # Not available in trade
                    ask=None,
                    bid_qty=None,
                    ask_qty=None,
                )
        except Exception as e:
            logger.error(f"Failed to get tick for {symbol}: {e}")

        return None

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream real-time bars via WebSocket.

        Note: Requires Developer+ subscription for real-time data.
        Starter subscription provides 15-minute delayed data.

        Args:
            symbols: List of stock tickers
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects as they become available
        """
        if not symbols:
            return

        self._streaming_symbols = [s.upper() for s in symbols]
        self._stop_event.clear()

        # Clear queue
        while not self._bar_queue.empty():
            try:
                self._bar_queue.get_nowait()
            except queue.Empty:
                break

        # Start WebSocket in background thread
        self._ws_thread = threading.Thread(
            target=self._run_websocket,
            args=(self._streaming_symbols, interval_ms),
            name="polygon-ws",
            daemon=True,
        )
        self._ws_thread.start()

        try:
            while not self._stop_event.is_set():
                try:
                    bar = self._bar_queue.get(timeout=1.0)
                    yield bar
                except queue.Empty:
                    continue
        finally:
            self._stop_event.set()
            if self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=5.0)

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks (trades).

        Args:
            symbols: List of stock tickers

        Yields:
            Tick objects
        """
        # Tick streaming via Polygon WebSocket "T.*" channels is not yet implemented.
        # This is a known limitation - use bar streaming or historical data instead.
        #
        # Implementation would require:
        # 1. WebSocket subscription to "T.<symbol>" channels
        # 2. Message parsing for trade events
        # 3. Tick object construction from trade data
        #
        # Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#adapter-polygon-tick-streaming
        logger.warning(
            "Tick streaming not implemented for Polygon adapter. "
            "Use stream_bars() or get_bars() instead. Returns empty iterator."
        )
        return iter([])

    def get_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Dict[str, List[Bar]]:
        """Fetch historical bars for multiple symbols."""
        result: Dict[str, List[Bar]] = {}

        for symbol in symbols:
            result[symbol] = self.get_bars(
                symbol,
                timeframe,
                limit=limit,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        return result

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _convert_agg_to_bar(self, agg: Any, symbol: str) -> Bar:
        """Convert Polygon Agg to internal Bar model."""
        # agg.timestamp is in milliseconds
        ts = int(agg.timestamp)

        return Bar(
            ts=ts,
            symbol=symbol.upper(),
            open=Decimal(str(agg.open)),
            high=Decimal(str(agg.high)),
            low=Decimal(str(agg.low)),
            close=Decimal(str(agg.close)),
            volume_base=Decimal(str(agg.volume)),
            volume_quote=Decimal(str(agg.vwap * agg.volume)) if agg.vwap else None,
            trades=agg.transactions if hasattr(agg, "transactions") else None,
            vwap=Decimal(str(agg.vwap)) if agg.vwap else None,
            is_final=True,
        )

    def _run_websocket(self, symbols: List[str], interval_ms: int) -> None:
        """Run WebSocket client in background thread."""
        try:
            asyncio.run(self._websocket_client(symbols, interval_ms))
        except RuntimeError:
            # Handle nested event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._websocket_client(symbols, interval_ms))
            finally:
                loop.close()

    async def _websocket_client(self, symbols: List[str], interval_ms: int) -> None:
        """Async WebSocket client for Polygon data stream."""
        reconnect_delay = 1.0

        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream(symbols, interval_ms)
                reconnect_delay = 1.0  # Reset on success
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.warning(f"Polygon WebSocket error: {e}, reconnecting in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, 60.0)

    async def _connect_and_stream(self, symbols: List[str], interval_ms: int) -> None:
        """Connect to Polygon WebSocket and stream data."""
        try:
            from polygon import WebSocketClient
            from polygon.websocket.models import WebSocketMessage as PolygonWSMessage
        except ImportError:
            logger.error("polygon-api-client WebSocket not available")
            self._stop_event.set()
            return

        # Determine feed based on interval
        # AM.* for minute aggregates
        feed = "stocks"
        subscriptions = [f"AM.{s}" for s in symbols]

        ws = WebSocketClient(
            api_key=self._api_key,
            feed=feed,
            market="stocks",
        )

        # Track last bar timestamps to detect new bars
        last_bar_ts: Dict[str, int] = {}

        def handle_msg(messages: List[PolygonWSMessage]) -> None:
            """Handle incoming WebSocket messages."""
            for msg in messages:
                if hasattr(msg, "event_type") and msg.event_type == "AM":
                    try:
                        bar = self._convert_ws_bar(msg)
                        if bar:
                            # Check for duplicate
                            prev_ts = last_bar_ts.get(bar.symbol)
                            if prev_ts is None or bar.ts > prev_ts:
                                last_bar_ts[bar.symbol] = bar.ts
                                self._enqueue_bar(bar)
                    except Exception as e:
                        logger.debug(f"Bar conversion error: {e}")

        ws.subscribe(subscriptions)

        try:
            # Run WebSocket (this is blocking)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ws.run(handle_msg),
            )
        except Exception as e:
            if not self._stop_event.is_set():
                raise
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _convert_ws_bar(self, msg: Any) -> Optional[Bar]:
        """Convert Polygon WebSocket message to Bar."""
        try:
            # Extract symbol from msg.symbol (e.g., "AAPL")
            symbol = str(msg.symbol).upper()

            # Start time in milliseconds
            ts = int(msg.start_timestamp)

            return Bar(
                ts=ts,
                symbol=symbol,
                open=Decimal(str(msg.open)),
                high=Decimal(str(msg.high)),
                low=Decimal(str(msg.low)),
                close=Decimal(str(msg.close)),
                volume_base=Decimal(str(msg.volume)),
                volume_quote=Decimal(str(msg.vwap * msg.volume)) if msg.vwap else None,
                vwap=Decimal(str(msg.vwap)) if msg.vwap else None,
                is_final=True,
            )
        except Exception as e:
            logger.debug(f"WebSocket bar conversion error: {e}")
            return None

    def _enqueue_bar(self, bar: Bar) -> None:
        """Add bar to queue with overflow handling."""
        try:
            self._bar_queue.put_nowait(bar)
        except queue.Full:
            # Drop oldest to make room
            try:
                self._bar_queue.get_nowait()
            except queue.Empty:
                pass
            self._bar_queue.put_nowait(bar)
