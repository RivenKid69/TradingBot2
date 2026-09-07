# -*- coding: utf-8 -*-
"""
adapters/yahoo/market_data.py
Yahoo Finance market data adapter for indices and supplementary data.

This adapter provides access to:
- VIX (CBOE Volatility Index) - ^VIX
- Dollar Index (DXY) - DX-Y.NYB
- Treasury yields - ^TNX, ^TYX
- Sector ETFs and indices

VIX is critical for stock trading as a fear indicator.
Not available on Alpaca/Polygon, so Yahoo Finance is the primary source.

Usage:
    from adapters.yahoo.market_data import YahooMarketDataAdapter

    adapter = YahooMarketDataAdapter()
    bars = adapter.get_bars("^VIX", "1d", limit=365)

    # Or via download script:
    python scripts/download_stock_data.py --symbols ^VIX --provider yahoo

Dependencies:
    pip install yfinance>=0.2.0

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import logging
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor
from core_models import Bar, Tick

logger = logging.getLogger(__name__)


# Known Yahoo Finance symbols for macro indicators
YAHOO_INDICES = {
    # Volatility
    "^VIX": "CBOE Volatility Index",
    "^VXN": "CBOE Nasdaq Volatility",
    "^VIX3M": "CBOE 3-Month Volatility",
    # Currency
    "DX-Y.NYB": "US Dollar Index (DXY)",
    # Treasuries
    "^TNX": "10-Year Treasury Yield",
    "^TYX": "30-Year Treasury Yield",
    "^FVX": "5-Year Treasury Yield",
    "^IRX": "13-Week Treasury Bill",
    # Commodities
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "CL=F": "Crude Oil Futures",
    # Major Indices (for reference)
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "NASDAQ Composite",
    "^RUT": "Russell 2000",
}


class YahooMarketDataAdapter(MarketDataAdapter):
    """
    Yahoo Finance market data adapter for indices and macro indicators.

    Primary use case: VIX data for equity trading features.

    Features:
        - Historical OHLCV data for indices
        - Auto-retry on rate limits
        - Compatible with existing data pipeline

    Configuration:
        rate_limit_pause: Seconds to wait between requests (default: 0.5)
        max_retries: Maximum retry attempts (default: 3)

    Example:
        adapter = YahooMarketDataAdapter()
        vix_bars = adapter.get_bars("^VIX", "1d", limit=365)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.YAHOO,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config or {})

        self._yf = None
        self._rate_limit_pause = float(self._config.get("rate_limit_pause", 0.5))
        self._max_retries = int(self._config.get("max_retries", 3))

    def _ensure_yfinance(self) -> Any:
        """Lazy import of yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf

                self._yf = yf
            except ImportError:
                raise ImportError("yfinance not installed. Install with: pip install yfinance")
        return self._yf

    def _do_connect(self) -> None:
        """Initialize connection (lazy - no actual connection needed)."""
        self._ensure_yfinance()

    def _do_disconnect(self) -> None:
        """Close connection (no-op for yfinance)."""
        pass

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
        Fetch historical OHLCV bars from Yahoo Finance.

        Args:
            symbol: Yahoo symbol (e.g., "^VIX", "DX-Y.NYB")
            timeframe: Bar timeframe (e.g., "1d", "1h", "5m")
            limit: Maximum number of bars (approximate)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects, oldest first

        Note:
            Yahoo Finance has limitations on intraday data:
            - 1m data: last 7 days only
            - 5m/15m data: last 60 days
            - 1h data: last 730 days
            - 1d data: full history available
        """
        yf = self._ensure_yfinance()

        # Convert timeframe to Yahoo Finance interval
        interval = self._parse_timeframe(timeframe)

        # Calculate date range
        if end_ts:
            end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        if start_ts:
            start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
        else:
            # Estimate start based on limit and timeframe
            days_needed = self._estimate_days_for_limit(limit, timeframe)
            start_dt = end_dt - timedelta(days=days_needed)

        logger.info(
            f"Downloading {symbol} from Yahoo Finance: {start_dt.date()} to {end_dt.date()}"
        )

        try:
            ticker = yf.Ticker(symbol)

            # Download data
            df = ticker.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,  # Adjust for splits/dividends
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return []

            # Convert to Bar objects
            bars = []
            for idx, row in df.iterrows():
                try:
                    # idx is a DatetimeIndex
                    ts = int(idx.timestamp() * 1000)

                    bar = Bar(
                        ts=ts,
                        symbol=symbol,
                        open=Decimal(str(row["Open"])),
                        high=Decimal(str(row["High"])),
                        low=Decimal(str(row["Low"])),
                        close=Decimal(str(row["Close"])),
                        volume_base=Decimal(str(row.get("Volume", 0))),
                        is_final=True,
                    )
                    bars.append(bar)
                except Exception as e:
                    logger.debug(f"Error converting row: {e}")
                    continue

            logger.info(f"Downloaded {symbol}: {len(bars)} bars")
            return bars

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return []

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """Get the most recent bar."""
        bars = self.get_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current quote for symbol.

        Note: Yahoo Finance doesn't provide real-time bid/ask for indices.
        This returns the last traded price as both bid and ask.
        """
        yf = self._ensure_yfinance()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            price = info.get("regularMarketPrice") or info.get("previousClose")
            if not price:
                return None

            import time

            return Tick(
                ts=int(time.time() * 1000),
                symbol=symbol,
                bid=Decimal(str(price)),
                ask=Decimal(str(price)),
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
        Streaming not supported for Yahoo Finance.

        Raises:
            NotImplementedError: Yahoo Finance doesn't support streaming
        """
        raise NotImplementedError(
            "Yahoo Finance doesn't support real-time streaming. "
            "Use polling with get_latest_bar() instead."
        )

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Streaming not supported for Yahoo Finance.

        Raises:
            NotImplementedError: Yahoo Finance doesn't support streaming
        """
        raise NotImplementedError(
            "Yahoo Finance doesn't support real-time streaming. "
            "Use polling with get_tick() instead."
        )

    def _parse_timeframe(self, timeframe: str) -> str:
        """Convert timeframe string to Yahoo Finance interval."""
        tf_map = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # Note: Yahoo may not support all intervals
            "1d": "1d",
            "1day": "1d",
            "5d": "5d",
            "1w": "1wk",
            "1wk": "1wk",
            "1mo": "1mo",
            "1month": "1mo",
        }

        result = tf_map.get(timeframe.lower())
        if not result:
            # Try direct passthrough
            result = timeframe.lower()

        return result

    def _estimate_days_for_limit(self, limit: int, timeframe: str) -> int:
        """Estimate days needed to get `limit` bars."""
        bars_per_day = {
            "1m": 390,  # 6.5 hours * 60
            "5m": 78,
            "15m": 26,
            "30m": 13,
            "1h": 7,
            "4h": 2,
            "1d": 1,
            "1wk": 1 / 5,
            "1mo": 1 / 21,
        }

        bpd = bars_per_day.get(timeframe.lower(), 1)
        days = int(limit / max(bpd, 0.1)) + 30  # Add buffer
        return min(days, 365 * 10)  # Cap at 10 years

    @staticmethod
    def get_supported_indices() -> Dict[str, str]:
        """Return dict of supported Yahoo Finance indices."""
        return YAHOO_INDICES.copy()

    @staticmethod
    def is_yahoo_symbol(symbol: str) -> bool:
        """Check if symbol should use Yahoo Finance (indices, futures)."""
        # Yahoo symbols typically start with ^ or contain = or .
        return symbol.startswith("^") or "=" in symbol or symbol in YAHOO_INDICES
