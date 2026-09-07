# -*- coding: utf-8 -*-
"""
adapters/binance/futures_market_data.py
Binance Futures market data adapter.

Provides futures-specific market data including:
- Mark price and index price streams
- Funding rate information
- Open interest data
- Liquidation streams
- Premium index data

References:
    - Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
    - Mark Price: https://binance-docs.github.io/apidocs/futures/en/#mark-price
    - Funding Rate: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from core_models import Bar, Tick
from core_futures import (
    FundingRateInfo,
    FundingPayment,
    MarkPriceTick,
    OpenInterestInfo,
    LiquidationEvent,
)
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor, MarketType

logger = logging.getLogger(__name__)


class BinanceFuturesMarketDataAdapter(MarketDataAdapter):
    """
    Binance Futures market data adapter.

    Extends the base MarketDataAdapter with futures-specific functionality.

    Configuration:
        futures_url: API base URL (default: https://fapi.binance.com)
        timeout: Request timeout in seconds (default: 20)
        testnet: Use testnet URL (default: False)

    Example:
        >>> adapter = BinanceFuturesMarketDataAdapter()
        >>> adapter.connect()
        >>> mark = adapter.get_mark_price("BTCUSDT")
        >>> print(f"Mark: {mark.mark_price}, Index: {mark.index_price}")
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Lazy-initialized client
        self._client = None

        # URLs
        self._futures_url = self._config.get("futures_url", "https://fapi.binance.com")
        if self._config.get("testnet", False):
            self._futures_url = "https://testnet.binancefuture.com"

    def _get_client(self):
        """Lazy initialization of Binance client."""
        if self._client is None:
            from binance_public import BinancePublicClient, PublicEndpoints

            endpoints = PublicEndpoints(
                spot_base=self._config.get("base_url", "https://api.binance.com"),
                futures_base=self._futures_url,
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

    # ========================================================================
    # Standard MarketDataAdapter methods (futures implementation)
    # ========================================================================

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
        Fetch historical OHLCV bars from Binance Futures.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")
            limit: Maximum number of bars (max 1500 for futures)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects, oldest first
        """
        client = self._get_client()

        # Futures API allows up to 1500 klines
        limit = min(limit, 1500)

        raw_klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            start_ms=start_ts,
            end_ms=end_ts,
            use_futures=True,
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
            # Use futures endpoints
            bid, ask = client.get_book_ticker(symbol, use_futures=True)
            price = client.get_last_price(symbol, use_futures=True)

            if bid is None and ask is None and price is None:
                return None

            spread_bps = None
            if bid is not None and ask is not None and bid > 0:
                spread = (ask - bid) / bid * 10000
                spread_bps = Decimal(str(spread))

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
        Stream real-time bars (simplified polling implementation).

        For production WebSocket streaming, use impl_binance_public module.

        Args:
            symbols: Symbols to stream
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects
        """
        timeframe = self._interval_ms_to_timeframe(interval_ms)
        last_ts: Dict[str, int] = {}
        poll_interval = interval_ms / 1000 / 2

        while True:
            for symbol in symbols:
                try:
                    bar = self.get_latest_bar(symbol, timeframe)
                    if bar is not None:
                        if symbol not in last_ts or bar.ts > last_ts[symbol]:
                            last_ts[symbol] = bar.ts
                            yield bar
                except Exception as e:
                    logger.debug(f"Error streaming bar for {symbol}: {e}")

            time.sleep(poll_interval)

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks (simplified polling implementation).

        Args:
            symbols: Symbols to stream

        Yields:
            Tick objects
        """
        poll_interval = float(self._config.get("tick_poll_interval", 1.0))

        while True:
            for symbol in symbols:
                try:
                    tick = self.get_tick(symbol)
                    if tick is not None:
                        yield tick
                except Exception as e:
                    logger.debug(f"Error streaming tick for {symbol}: {e}")

            time.sleep(poll_interval)

    # ========================================================================
    # Futures-specific market data methods
    # ========================================================================

    def get_mark_price(self, symbol: str) -> Optional[MarkPriceTick]:
        """
        Get current mark price for a symbol.

        The mark price is used for:
        - P&L calculations
        - Liquidation price calculations
        - Funding rate calculation

        Args:
            symbol: Futures symbol (e.g., "BTCUSDT")

        Returns:
            MarkPriceTick with mark price, index price, and funding info
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}

            data = client._session_get(
                url,
                params=params,
                budget="markPrice",
                tokens=1.0,
            )

            if not isinstance(data, dict):
                return None

            return MarkPriceTick(
                symbol=str(data.get("symbol", symbol)),
                mark_price=Decimal(str(data.get("markPrice", "0"))),
                index_price=Decimal(str(data.get("indexPrice", "0"))),
                estimated_settle_price=Decimal(str(data.get("estimatedSettlePrice", "0"))),
                funding_rate=Decimal(str(data.get("lastFundingRate", "0"))),
                next_funding_time_ms=int(data.get("nextFundingTime", 0)),
                timestamp_ms=int(data.get("time", int(time.time() * 1000))),
            )

        except Exception as e:
            logger.warning(f"Failed to get mark price for {symbol}: {e}")
            return None

    def get_mark_prices(self, symbols: Optional[List[str]] = None) -> List[MarkPriceTick]:
        """
        Get mark prices for multiple symbols.

        Args:
            symbols: List of symbols (None for all)

        Returns:
            List of MarkPriceTick objects
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/premiumIndex"
            params = {}
            if symbols and len(symbols) == 1:
                params["symbol"] = symbols[0]

            data = client._session_get(
                url,
                params=params if params else None,
                budget="markPrice",
                tokens=2.0 if not symbols else 1.0,
            )

            results = []
            data_list = data if isinstance(data, list) else [data]

            for item in data_list:
                if not isinstance(item, dict):
                    continue

                sym = str(item.get("symbol", ""))
                if symbols and sym not in symbols:
                    continue

                results.append(
                    MarkPriceTick(
                        symbol=sym,
                        mark_price=Decimal(str(item.get("markPrice", "0"))),
                        index_price=Decimal(str(item.get("indexPrice", "0"))),
                        estimated_settle_price=Decimal(str(item.get("estimatedSettlePrice", "0"))),
                        funding_rate=Decimal(str(item.get("lastFundingRate", "0"))),
                        next_funding_time_ms=int(item.get("nextFundingTime", 0)),
                        timestamp_ms=int(item.get("time", int(time.time() * 1000))),
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Failed to get mark prices: {e}")
            return []

    def get_funding_rate(self, symbol: str) -> Optional[FundingRateInfo]:
        """
        Get current funding rate information.

        Funding rates are payments exchanged between longs and shorts
        every 8 hours to keep perpetual price close to index.

        Args:
            symbol: Futures symbol

        Returns:
            FundingRateInfo with current and predicted funding rate
        """
        mark_tick = self.get_mark_price(symbol)
        if mark_tick is None:
            return None

        return FundingRateInfo(
            symbol=mark_tick.symbol,
            funding_rate=mark_tick.funding_rate,
            next_funding_time_ms=mark_tick.next_funding_time_ms,
            mark_price=mark_tick.mark_price,
            index_price=mark_tick.index_price,
            estimated_rate=None,  # Would need separate API call
            funding_interval_hours=8,
        )

    def get_funding_rate_history(
        self,
        symbol: str,
        *,
        limit: int = 100,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[FundingPayment]:
        """
        Get historical funding rates.

        Args:
            symbol: Futures symbol
            limit: Maximum number of records (max 1000)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of FundingPayment objects (historical rates, not payments)
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/fundingRate"
            params: Dict[str, Any] = {
                "symbol": symbol,
                "limit": min(limit, 1000),
            }
            if start_ts is not None:
                params["startTime"] = start_ts
            if end_ts is not None:
                params["endTime"] = end_ts

            data = client._session_get(
                url,
                params=params,
                budget="fundingRate",
                tokens=1.0,
            )

            results = []
            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict):
                    continue

                # Note: This is funding rate history, not actual payments
                # For actual payment amounts, need account API
                results.append(
                    FundingPayment(
                        symbol=str(item.get("symbol", symbol)),
                        timestamp_ms=int(item.get("fundingTime", 0)),
                        funding_rate=Decimal(str(item.get("fundingRate", "0"))),
                        mark_price=Decimal(str(item.get("markPrice", "0"))),
                        position_qty=Decimal("0"),  # Unknown without account context
                        payment_amount=Decimal("0"),  # Unknown without account context
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Failed to get funding rate history for {symbol}: {e}")
            return []

    def get_open_interest(self, symbol: str) -> Optional[OpenInterestInfo]:
        """
        Get current open interest for a symbol.

        Open interest represents the total number of outstanding contracts.

        Args:
            symbol: Futures symbol

        Returns:
            OpenInterestInfo with total open interest
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/openInterest"
            params = {"symbol": symbol}

            data = client._session_get(
                url,
                params=params,
                budget="openInterest",
                tokens=1.0,
            )

            if not isinstance(data, dict):
                return None

            return OpenInterestInfo(
                symbol=str(data.get("symbol", symbol)),
                open_interest=Decimal(str(data.get("openInterest", "0"))),
                open_interest_value=Decimal("0"),  # Not provided by this endpoint
                timestamp_ms=int(data.get("time", int(time.time() * 1000))),
            )

        except Exception as e:
            logger.warning(f"Failed to get open interest for {symbol}: {e}")
            return None

    def get_open_interest_statistics(
        self,
        symbol: str,
        period: str = "5m",
        *,
        limit: int = 30,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[OpenInterestInfo]:
        """
        Get historical open interest statistics.

        Args:
            symbol: Futures symbol
            period: Period ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
            limit: Maximum records (max 500)
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            List of OpenInterestInfo objects
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/futures/data/openInterestHist"
            params: Dict[str, Any] = {
                "symbol": symbol,
                "period": period,
                "limit": min(limit, 500),
            }
            if start_ts is not None:
                params["startTime"] = start_ts
            if end_ts is not None:
                params["endTime"] = end_ts

            data = client._session_get(
                url,
                params=params,
                budget="openInterestHist",
                tokens=1.0,
            )

            results = []
            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict):
                    continue

                results.append(
                    OpenInterestInfo(
                        symbol=str(item.get("symbol", symbol)),
                        open_interest=Decimal(str(item.get("sumOpenInterest", "0"))),
                        open_interest_value=Decimal(str(item.get("sumOpenInterestValue", "0"))),
                        timestamp_ms=int(item.get("timestamp", 0)),
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Failed to get OI statistics for {symbol}: {e}")
            return []

    def get_mark_price_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Get mark price klines (OHLC based on mark price, not last traded).

        Useful for backtesting with mark price to avoid liquidation spikes.

        Args:
            symbol: Futures symbol
            timeframe: Kline interval
            limit: Maximum klines
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            List of Bar objects based on mark price
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/markPriceKlines"
            params: Dict[str, Any] = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": min(limit, 1500),
            }
            if start_ts is not None:
                params["startTime"] = start_ts
            if end_ts is not None:
                params["endTime"] = end_ts

            data = client._session_get(
                url,
                params=params,
                budget="markPriceKlines",
                tokens=max(1.0, limit / 100),
            )

            bars = []
            for kline in data if isinstance(data, list) else []:
                bar = self._parse_kline(symbol, kline)
                if bar is not None:
                    bars.append(bar)

            return bars

        except Exception as e:
            logger.warning(f"Failed to get mark price klines for {symbol}: {e}")
            return []

    def get_liquidation_orders(
        self,
        symbol: Optional[str] = None,
        *,
        limit: int = 100,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[LiquidationEvent]:
        """
        Get recent forced liquidation orders.

        Args:
            symbol: Futures symbol (optional)
            limit: Maximum records (max 1000)
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            List of LiquidationEvent objects
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/fapi/v1/forceOrders"
            params: Dict[str, Any] = {"limit": min(limit, 1000)}
            if symbol:
                params["symbol"] = symbol
            if start_ts is not None:
                params["startTime"] = start_ts
            if end_ts is not None:
                params["endTime"] = end_ts

            data = client._session_get(
                url,
                params=params,
                budget="forceOrders",
                tokens=5.0 if not symbol else 1.0,
            )

            results = []
            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict):
                    continue

                # Determine liquidation type based on status
                status = str(item.get("status", "")).upper()
                liq_type = "full" if status == "FILLED" else "partial"

                results.append(
                    LiquidationEvent(
                        symbol=str(item.get("symbol", "")),
                        timestamp_ms=int(item.get("time", 0)),
                        side=str(item.get("side", "")),
                        qty=Decimal(str(item.get("origQty", "0"))),
                        price=Decimal(str(item.get("avgPrice", item.get("price", "0")))),
                        liquidation_type=liq_type,
                        loss_amount=Decimal("0"),  # Not provided by API
                        order_id=str(item.get("orderId", "")),
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Failed to get liquidation orders: {e}")
            return []

    def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        *,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get top trader long/short ratio.

        Useful for sentiment analysis.

        Args:
            symbol: Futures symbol
            period: Period ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
            limit: Maximum records

        Returns:
            List of dictionaries with long/short ratios
        """
        client = self._get_client()

        try:
            url = f"{self._futures_url}/futures/data/topLongShortPositionRatio"
            params = {
                "symbol": symbol,
                "period": period,
                "limit": min(limit, 500),
            }

            data = client._session_get(
                url,
                params=params,
                budget="topLongShortRatio",
                tokens=1.0,
            )

            results = []
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict):
                    results.append(
                        {
                            "symbol": item.get("symbol", symbol),
                            "timestamp_ms": int(item.get("timestamp", 0)),
                            "long_short_ratio": float(item.get("longShortRatio", 0)),
                            "long_account": float(item.get("longAccount", 0)),
                            "short_account": float(item.get("shortAccount", 0)),
                        }
                    )

            return results

        except Exception as e:
            logger.warning(f"Failed to get long/short ratio for {symbol}: {e}")
            return []

    # ========================================================================
    # Streaming methods (futures-specific)
    # ========================================================================

    def stream_mark_prices(
        self,
        symbols: Optional[Sequence[str]] = None,
        update_speed: str = "1s",
    ) -> Iterator[MarkPriceTick]:
        """
        Stream mark price updates (polling implementation).

        For production use WebSocket: wss://fstream.binance.com/ws/!markPrice@arr@1s

        Args:
            symbols: Symbols to stream (None for all)
            update_speed: Update speed ("1s" or "3s")

        Yields:
            MarkPriceTick objects
        """
        poll_interval = 1.0 if update_speed == "1s" else 3.0

        while True:
            try:
                ticks = self.get_mark_prices(list(symbols) if symbols else None)
                for tick in ticks:
                    yield tick
            except Exception as e:
                logger.debug(f"Error streaming mark prices: {e}")

            time.sleep(poll_interval)

    def stream_liquidations(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Iterator[LiquidationEvent]:
        """
        Stream liquidation orders (polling implementation).

        For production use WebSocket: wss://fstream.binance.com/ws/!forceOrder@arr

        Args:
            symbols: Symbols to filter

        Yields:
            LiquidationEvent objects
        """
        poll_interval = float(self._config.get("liquidation_poll_interval", 5.0))
        last_seen: Dict[str, int] = {}

        while True:
            try:
                for symbol in symbols or [None]:
                    events = self.get_liquidation_orders(symbol, limit=50)
                    for event in events:
                        key = f"{event.symbol}_{event.timestamp_ms}_{event.order_id}"
                        if key not in last_seen:
                            last_seen[key] = event.timestamp_ms
                            if symbols is None or event.symbol in symbols:
                                yield event

                # Cleanup old entries
                cutoff = int(time.time() * 1000) - 3600000  # 1 hour
                last_seen = {k: v for k, v in last_seen.items() if v > cutoff}

            except Exception as e:
                logger.debug(f"Error streaming liquidations: {e}")

            time.sleep(poll_interval)

    # ========================================================================
    # Helper methods
    # ========================================================================

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

    @property
    def market_type(self) -> MarketType:
        """Return the market type this adapter serves."""
        return MarketType.CRYPTO_FUTURES
