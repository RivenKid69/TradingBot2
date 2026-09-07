# -*- coding: utf-8 -*-
"""
adapters/binance/market_data.py
Binance market data adapter implementation.

Wraps existing BinancePublicClient functionality into the MarketDataAdapter interface.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor

logger = logging.getLogger(__name__)


class BinanceMarketDataAdapter(MarketDataAdapter):
    """
    Binance market data adapter.

    Wraps BinancePublicClient to implement the MarketDataAdapter interface.
    Supports both spot and futures market data.

    Configuration:
        base_url: API base URL (default: https://api.binance.com)
        futures_url: Futures API URL (default: https://fapi.binance.com)
        timeout: Request timeout in seconds (default: 20)
        use_futures: Whether to use futures endpoints (default: False)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Lazy import to avoid circular dependencies
        self._client = None
        self._use_futures = self._config.get("use_futures", False)

    def _get_client(self):
        """Lazy initialization of Binance client."""
        if self._client is None:
            from binance_public import BinancePublicClient, PublicEndpoints

            endpoints = PublicEndpoints(
                spot_base=self._config.get("base_url", "https://api.binance.com"),
                futures_base=self._config.get("futures_url", "https://fapi.binance.com"),
            )

            self._client = BinancePublicClient(
                endpoints=endpoints,
                timeout=int(self._config.get("timeout", 20)),
            )

        return self._client

    def _do_connect(self) -> None:
        """Initialize client connection."""
        self._get_client()

    def _do_disconnect(self) -> None:
        """Close client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

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
        Fetch historical OHLCV bars from Binance.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")
            limit: Maximum number of bars (max 1000)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects, oldest first
        """
        client = self._get_client()

        # Binance limits
        limit = min(limit, 1000)

        raw_klines = client.get_klines(
            market="futures" if self._use_futures else "spot",
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            start_ms=start_ts,
            end_ms=end_ts,
        )

        bars = []
        for kline in raw_klines:
            try:
                bar = self._parse_kline(symbol, kline)
                if bar is not None:
                    bars.append(bar)
            except Exception as e:
                logger.warning(f"Failed to parse kline for {symbol}: {e}")

        return bars

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """Get the most recent bar."""
        bars = self.get_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current BBO for symbol."""
        client = self._get_client()

        try:
            bid, ask = client.get_book_ticker(symbol)
            price = client.get_last_price(symbol)

            if bid is None and ask is None and price is None:
                return None

            spread_bps = None
            if bid is not None and ask is not None and bid > 0:
                spread = (ask - bid) / bid * 10000
                spread_bps = Decimal(str(spread))

            import time

            return Tick(
                ts=int(time.time() * 1000),
                symbol=symbol,
                price=Decimal(str(price)) if price is not None else None,
                bid=Decimal(str(bid)) if bid is not None else None,
                ask=Decimal(str(ask)) if ask is not None else None,
                spread_bps=spread_bps,
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
        Stream real-time bars.

        Note: This is a simplified implementation that polls.
        For true WebSocket streaming, use impl_binance_public.BinancePublicBarSource.

        Args:
            symbols: Symbols to stream
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects
        """
        # For proper WebSocket streaming, we delegate to the existing implementation
        try:
            from impl_binance_public import BinancePublicBarSource

            source = BinancePublicBarSource(
                cfg=self._config,
            )

            for bar in source.stream_bars(symbols, interval_ms):
                yield bar

        except ImportError:
            # Fallback to polling (not recommended for production)
            logger.warning("WebSocket streaming not available, falling back to polling")

            import time

            # Convert interval_ms to timeframe string
            timeframe = self._interval_ms_to_timeframe(interval_ms)
            last_ts: Dict[str, int] = {}

            while True:
                for symbol in symbols:
                    bar = self.get_latest_bar(symbol, timeframe)
                    if bar is not None:
                        if symbol not in last_ts or bar.ts > last_ts[symbol]:
                            last_ts[symbol] = bar.ts
                            yield bar

                time.sleep(interval_ms / 1000 / 2)  # Poll at half interval

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks.

        Note: This is a simplified polling implementation.
        For true WebSocket streaming, use the existing WebSocket implementation.

        Args:
            symbols: Symbols to stream

        Yields:
            Tick objects
        """
        import time

        poll_interval = float(self._config.get("tick_poll_interval", 1.0))

        while True:
            for symbol in symbols:
                tick = self.get_tick(symbol)
                if tick is not None:
                    yield tick

            time.sleep(poll_interval)

    def _parse_kline(self, symbol: str, kline: Any) -> Optional[Bar]:
        """Parse Binance kline data into Bar object."""
        if not isinstance(kline, (list, tuple)) or len(kline) < 6:
            return None

        try:
            return Bar(
                ts=int(kline[0]),
                symbol=symbol,
                open=Decimal(str(kline[1])),
                high=Decimal(str(kline[2])),
                low=Decimal(str(kline[3])),
                close=Decimal(str(kline[4])),
                volume_base=Decimal(str(kline[5])) if len(kline) > 5 else None,
                volume_quote=Decimal(str(kline[7])) if len(kline) > 7 else None,
                trades=int(kline[8]) if len(kline) > 8 else None,
                taker_buy_base=Decimal(str(kline[9])) if len(kline) > 9 else None,
                is_final=True,
            )
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Kline parse error: {e}")
            return None

    @staticmethod
    def _interval_ms_to_timeframe(interval_ms: int) -> str:
        """Convert interval milliseconds to Binance timeframe string."""
        minutes = interval_ms // 60000

        if minutes < 1:
            seconds = interval_ms // 1000
            return f"{seconds}s"
        elif minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:
            hours = minutes // 60
            return f"{hours}h"
        else:
            days = minutes // 1440
            return f"{days}d"
