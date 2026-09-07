# -*- coding: utf-8 -*-
"""
adapters/alpaca/market_data.py
Alpaca market data adapter for US equities.

Status: Tested and operational (Phase 2 Complete)

Alpaca provides:
- REST API for historical data (get_bars, get_latest_bar, get_tick)
- WebSocket for real-time quotes/trades:
  - Sync iterators: stream_bars(), stream_ticks()
  - Async generators: stream_bars_async(), stream_ticks_async()
- Both IEX (free) and SIP (paid) data feeds

Usage:
    # Sync streaming (blocking)
    for bar in adapter.stream_bars(["AAPL", "MSFT"], 60000):
        print(f"Bar: {bar}")

    # Async streaming (for live trading)
    async for bar in adapter.stream_bars_async(["AAPL", "MSFT"]):
        await process_bar(bar)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor

logger = logging.getLogger(__name__)


class AlpacaMarketDataAdapter(MarketDataAdapter):
    """
    Alpaca market data adapter for US equities.

    Configuration:
        api_key: Alpaca API key (required)
        api_secret: Alpaca API secret (required)
        base_url: API base URL (default: https://data.alpaca.markets)
        feed: Data feed type: "iex" (free) or "sip" (paid)
        paper: Use paper trading endpoint (default: False)

    Requirements:
        pip install alpaca-py

    Example:
        adapter = AlpacaMarketDataAdapter(config={
            "api_key": "your_key",
            "api_secret": "your_secret",
            "feed": "iex",
        })
        bars = adapter.get_bars("AAPL", "1h", limit=100)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        self._client = None
        self._historical_client = None
        self._stream_client = None

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            self._ensure_alpaca_sdk()

            from alpaca.data.historical import StockHistoricalDataClient

            api_key = self._config.get("api_key")
            api_secret = self._config.get("api_secret")

            if not api_key or not api_secret:
                raise ValueError("Alpaca API key and secret are required")

            self._historical_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret,
            )
            self._client = self._historical_client

        return self._client

    def _ensure_alpaca_sdk(self) -> None:
        """Ensure Alpaca SDK is installed."""
        try:
            import alpaca  # noqa: F401
        except ImportError:
            raise ImportError("Alpaca SDK not installed. Install with: pip install alpaca-py")

    def _do_connect(self) -> None:
        """Initialize client connection."""
        self._get_client()

    def _do_disconnect(self) -> None:
        """Close client connection."""
        # Alpaca SDK handles connection lifecycle internally
        self._client = None
        self._historical_client = None

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
        Fetch historical OHLCV bars from Alpaca.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT")
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")
            limit: Maximum number of bars (max 10000)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects, oldest first
        """
        client = self._get_client()

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Convert timeframe string to Alpaca TimeFrame
        tf = self._parse_timeframe(timeframe)

        # Build request
        request_params = {
            "symbol_or_symbols": symbol,
            "timeframe": tf,
            "limit": min(limit, 10000),
        }

        if start_ts:
            from datetime import datetime, timezone

            request_params["start"] = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        if end_ts:
            from datetime import datetime, timezone

            request_params["end"] = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)

        feed = self._config.get("feed", "iex")
        request_params["feed"] = feed

        adjustment = self._config.get("adjustment")
        if adjustment:
            from alpaca.data.enums import Adjustment

            if isinstance(adjustment, str):
                try:
                    # Map standard values to enum values
                    adj_str = adjustment.lower()
                    if adj_str == "raw":
                        adj_str = "raw"
                    elif adj_str == "split":
                        adj_str = "split"
                    elif adj_str == "all":
                        adj_str = "all"
                    request_params["adjustment"] = Adjustment(adj_str)
                except ValueError:
                    logger.warning(f"Invalid adjustment value: {adjustment}. Using default.")
            elif isinstance(adjustment, Adjustment):
                request_params["adjustment"] = adjustment

        request = StockBarsRequest(**request_params)
        bars_response = client.get_stock_bars(request)

        # Convert to Bar objects
        bars = []
        if symbol in bars_response:
            for alpaca_bar in bars_response[symbol]:
                bar = self._convert_bar(symbol, alpaca_bar)
                if bar:
                    bars.append(bar)

        return bars

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """Get the most recent bar."""
        bars = self.get_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current quote for symbol."""
        client = self._get_client()

        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = client.get_stock_latest_quote(request)

            if symbol not in quotes:
                return None

            quote = quotes[symbol]

            import time

            return Tick(
                ts=int(time.time() * 1000),
                symbol=symbol,
                bid=Decimal(str(quote.bid_price)) if quote.bid_price else None,
                ask=Decimal(str(quote.ask_price)) if quote.ask_price else None,
                bid_qty=Decimal(str(quote.bid_size)) if quote.bid_size else None,
                ask_qty=Decimal(str(quote.ask_size)) if quote.ask_size else None,
            )

        except Exception as e:
            logger.warning(f"Failed to get tick for {symbol}: {e}")
            return None

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream real-time bars via WebSocket (sync iterator).

        This is a blocking generator that yields bars as they arrive.
        For async usage, use stream_bars_async() instead.

        Note: Alpaca provides 1-minute bars via WebSocket.

        Args:
            symbols: Symbols to stream
            interval_ms: Bar interval in milliseconds (for reference only)

        Yields:
            Bar objects
        """
        import queue
        import threading

        self._ensure_alpaca_sdk()
        from alpaca.data.live import StockDataStream

        api_key = self._config.get("api_key")
        api_secret = self._config.get("api_secret")
        feed = self._config.get("feed", "iex")

        bar_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
            feed=feed,
        )

        async def handle_bar(bar):
            """Handle incoming bar from stream."""
            converted = self._convert_bar(bar.symbol, bar)
            if converted:
                bar_queue.put(converted)

        def run_stream():
            """Run stream in background thread."""
            stream.subscribe_bars(handle_bar, *symbols)
            try:
                stream.run()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                bar_queue.put(None)  # Signal end

        # Start stream in background thread
        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()

        # Yield bars from queue
        try:
            while not stop_event.is_set():
                try:
                    bar = bar_queue.get(timeout=1.0)
                    if bar is None:
                        break
                    yield bar
                except queue.Empty:
                    continue
        finally:
            stop_event.set()
            stream.stop()

    async def stream_bars_async(
        self,
        symbols: Sequence[str],
        interval_ms: int = 60000,
    ):
        """
        Stream real-time bars via WebSocket (async generator).

        This is an async generator for use with asyncio-based live trading.

        Args:
            symbols: Symbols to stream
            interval_ms: Bar interval in milliseconds (for reference only)

        Yields:
            Bar objects

        Example:
            async for bar in adapter.stream_bars_async(["AAPL", "MSFT"]):
                print(f"New bar: {bar.symbol} @ {bar.close}")
        """
        import asyncio

        self._ensure_alpaca_sdk()
        from alpaca.data.live import StockDataStream

        api_key = self._config.get("api_key")
        api_secret = self._config.get("api_secret")
        feed = self._config.get("feed", "iex")

        bar_queue: asyncio.Queue = asyncio.Queue()

        stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
            feed=feed,
        )

        async def handle_bar(bar):
            """Handle incoming bar from stream."""
            converted = self._convert_bar(bar.symbol, bar)
            if converted:
                await bar_queue.put(converted)

        stream.subscribe_bars(handle_bar, *symbols)

        # Run stream in background task
        async def run_stream():
            try:
                await stream._run_forever()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Async stream error: {e}")
                await bar_queue.put(None)

        stream_task = asyncio.create_task(run_stream())

        try:
            while True:
                bar = await bar_queue.get()
                if bar is None:
                    break
                yield bar
        finally:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time quotes via WebSocket (sync iterator).

        This is a blocking generator that yields ticks as they arrive.
        For async usage, use stream_ticks_async() instead.

        Args:
            symbols: Symbols to stream

        Yields:
            Tick objects
        """
        import queue
        import threading

        self._ensure_alpaca_sdk()
        from alpaca.data.live import StockDataStream

        api_key = self._config.get("api_key")
        api_secret = self._config.get("api_secret")
        feed = self._config.get("feed", "iex")

        tick_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
            feed=feed,
        )

        async def handle_quote(quote):
            """Handle incoming quote."""
            try:
                tick = Tick(
                    ts=int(quote.timestamp.timestamp() * 1000),
                    symbol=quote.symbol,
                    bid=Decimal(str(quote.bid_price)) if quote.bid_price else None,
                    ask=Decimal(str(quote.ask_price)) if quote.ask_price else None,
                    bid_qty=Decimal(str(quote.bid_size)) if quote.bid_size else None,
                    ask_qty=Decimal(str(quote.ask_size)) if quote.ask_size else None,
                )
                tick_queue.put(tick)
            except Exception as e:
                logger.debug(f"Quote conversion error: {e}")

        def run_stream():
            """Run stream in background thread."""
            stream.subscribe_quotes(handle_quote, *symbols)
            try:
                stream.run()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                tick_queue.put(None)

        # Start stream in background thread
        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()

        # Yield ticks from queue
        try:
            while not stop_event.is_set():
                try:
                    tick = tick_queue.get(timeout=1.0)
                    if tick is None:
                        break
                    yield tick
                except queue.Empty:
                    continue
        finally:
            stop_event.set()
            stream.stop()

    async def stream_ticks_async(
        self,
        symbols: Sequence[str],
    ):
        """
        Stream real-time quotes via WebSocket (async generator).

        This is an async generator for use with asyncio-based live trading.

        Args:
            symbols: Symbols to stream

        Yields:
            Tick objects

        Example:
            async for tick in adapter.stream_ticks_async(["AAPL", "MSFT"]):
                print(f"New tick: {tick.symbol} bid={tick.bid} ask={tick.ask}")
        """
        import asyncio

        self._ensure_alpaca_sdk()
        from alpaca.data.live import StockDataStream

        api_key = self._config.get("api_key")
        api_secret = self._config.get("api_secret")
        feed = self._config.get("feed", "iex")

        tick_queue: asyncio.Queue = asyncio.Queue()

        stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
            feed=feed,
        )

        async def handle_quote(quote):
            """Handle incoming quote."""
            try:
                tick = Tick(
                    ts=int(quote.timestamp.timestamp() * 1000),
                    symbol=quote.symbol,
                    bid=Decimal(str(quote.bid_price)) if quote.bid_price else None,
                    ask=Decimal(str(quote.ask_price)) if quote.ask_price else None,
                    bid_qty=Decimal(str(quote.bid_size)) if quote.bid_size else None,
                    ask_qty=Decimal(str(quote.ask_size)) if quote.ask_size else None,
                )
                await tick_queue.put(tick)
            except Exception as e:
                logger.debug(f"Quote conversion error: {e}")

        stream.subscribe_quotes(handle_quote, *symbols)

        # Run stream in background task
        async def run_stream():
            try:
                await stream._run_forever()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Async stream error: {e}")
                await tick_queue.put(None)

        stream_task = asyncio.create_task(run_stream())

        try:
            while True:
                tick = await tick_queue.get()
                if tick is None:
                    break
                yield tick
        finally:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    def _convert_bar(self, symbol: str, alpaca_bar: Any) -> Optional[Bar]:
        """Convert Alpaca bar to our Bar model."""
        try:
            ts = int(alpaca_bar.timestamp.timestamp() * 1000)

            return Bar(
                ts=ts,
                symbol=symbol,
                open=Decimal(str(alpaca_bar.open)),
                high=Decimal(str(alpaca_bar.high)),
                low=Decimal(str(alpaca_bar.low)),
                close=Decimal(str(alpaca_bar.close)),
                volume_base=Decimal(str(alpaca_bar.volume)),
                volume_quote=(
                    Decimal(str(alpaca_bar.vwap * alpaca_bar.volume)) if alpaca_bar.vwap else None
                ),
                trades=alpaca_bar.trade_count if hasattr(alpaca_bar, "trade_count") else None,
                vwap=Decimal(str(alpaca_bar.vwap)) if alpaca_bar.vwap else None,
                is_final=True,
            )
        except Exception as e:
            logger.debug(f"Bar conversion error: {e}")
            return None

    def _parse_timeframe(self, timeframe: str) -> Any:
        """Convert timeframe string to Alpaca TimeFrame."""
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_lower = timeframe.lower()

        # Parse common formats
        if tf_lower in ("1m", "1min"):
            return TimeFrame.Minute
        elif tf_lower in ("5m", "5min"):
            return TimeFrame(5, TimeFrameUnit.Minute)
        elif tf_lower in ("15m", "15min"):
            return TimeFrame(15, TimeFrameUnit.Minute)
        elif tf_lower in ("30m", "30min"):
            return TimeFrame(30, TimeFrameUnit.Minute)
        elif tf_lower in ("1h", "1hour"):
            return TimeFrame.Hour
        elif tf_lower in ("4h", "4hour"):
            return TimeFrame(4, TimeFrameUnit.Hour)
        elif tf_lower in ("1d", "1day"):
            return TimeFrame.Day
        elif tf_lower in ("1w", "1week"):
            return TimeFrame.Week
        elif tf_lower in ("1mo", "1month"):
            return TimeFrame.Month
        else:
            # Try to parse custom format
            import re

            match = re.match(r"(\d+)([mhd])", tf_lower)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if unit == "m":
                    return TimeFrame(value, TimeFrameUnit.Minute)
                elif unit == "h":
                    return TimeFrame(value, TimeFrameUnit.Hour)
                elif unit == "d":
                    return TimeFrame(value, TimeFrameUnit.Day)

            raise ValueError(f"Unsupported timeframe: {timeframe}")
