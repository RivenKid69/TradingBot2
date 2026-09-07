# -*- coding: utf-8 -*-
"""
adapters/ib/market_data.py
Interactive Brokers market data adapter for CME Group futures.

Uses ib_insync library for TWS API connectivity.
Supports real-time quotes, historical bars, and contract details.

Rate Limits (IB TWS API):
─────────────────────────────────────────────────────────────────────────────
| Type                    | Limit                  | Window    | Action     |
|-------------------------|------------------------|-----------|------------|
| General messages        | 50 msg/sec             | 1 sec     | Block      |
| Historical data         | 60 requests            | 10 min    | Pacing     |
| Identical hist request  | 6 requests             | 10 min    | Cache      |
| Market data subscribe   | 1 subscription/sec     | 1 sec     | Block      |
| Market data lines       | 100 concurrent         | N/A       | Hard limit |
| Scanner subscriptions   | 10 concurrent          | N/A       | Hard limit |
| Account updates         | 1 request/sec          | 1 sec     | Block      |
─────────────────────────────────────────────────────────────────────────────

Reference:
- IB TWS API: https://interactivebrokers.github.io/tws-api/
- ib_insync: https://ib-insync.readthedocs.io/
- Historical limitations: https://interactivebrokers.github.io/tws-api/historical_limitations.html
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor
from core_models import Bar, Tick

logger = logging.getLogger(__name__)

# Try to import ib_insync, but don't fail if not installed
try:
    from ib_insync import IB, ContFuture, Future, util

    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    IB = None
    ContFuture = None
    Future = None
    util = None


# =========================
# Contract Mapping
# =========================

# Map common symbols to IB contract details (exchange, currency)
CONTRACT_MAP: Dict[str, Dict[str, str]] = {
    # Equity Index (CME)
    "ES": {"exchange": "CME", "currency": "USD", "description": "E-mini S&P 500"},
    "NQ": {"exchange": "CME", "currency": "USD", "description": "E-mini NASDAQ 100"},
    "RTY": {"exchange": "CME", "currency": "USD", "description": "E-mini Russell 2000"},
    "MES": {"exchange": "CME", "currency": "USD", "description": "Micro E-mini S&P 500"},
    "MNQ": {"exchange": "CME", "currency": "USD", "description": "Micro E-mini NASDAQ 100"},
    "M2K": {"exchange": "CME", "currency": "USD", "description": "Micro E-mini Russell 2000"},
    # Equity Index (CBOT)
    "YM": {"exchange": "CBOT", "currency": "USD", "description": "E-mini Dow Jones"},
    "MYM": {"exchange": "CBOT", "currency": "USD", "description": "Micro E-mini Dow Jones"},
    # Metals (COMEX)
    "GC": {"exchange": "COMEX", "currency": "USD", "description": "Gold Futures"},
    "SI": {"exchange": "COMEX", "currency": "USD", "description": "Silver Futures"},
    "HG": {"exchange": "COMEX", "currency": "USD", "description": "Copper Futures"},
    "MGC": {"exchange": "COMEX", "currency": "USD", "description": "Micro Gold Futures"},
    "SIL": {"exchange": "COMEX", "currency": "USD", "description": "Micro Silver Futures"},
    # Energy (NYMEX)
    "CL": {"exchange": "NYMEX", "currency": "USD", "description": "Crude Oil Futures"},
    "NG": {"exchange": "NYMEX", "currency": "USD", "description": "Natural Gas Futures"},
    "MCL": {"exchange": "NYMEX", "currency": "USD", "description": "Micro Crude Oil Futures"},
    "RB": {"exchange": "NYMEX", "currency": "USD", "description": "RBOB Gasoline Futures"},
    "HO": {"exchange": "NYMEX", "currency": "USD", "description": "Heating Oil Futures"},
    # Currencies (CME)
    "6E": {"exchange": "CME", "currency": "USD", "description": "Euro FX Futures"},
    "6J": {"exchange": "CME", "currency": "USD", "description": "Japanese Yen Futures"},
    "6B": {"exchange": "CME", "currency": "USD", "description": "British Pound Futures"},
    "6A": {"exchange": "CME", "currency": "USD", "description": "Australian Dollar Futures"},
    "6C": {"exchange": "CME", "currency": "USD", "description": "Canadian Dollar Futures"},
    "6S": {"exchange": "CME", "currency": "USD", "description": "Swiss Franc Futures"},
    "M6E": {"exchange": "CME", "currency": "USD", "description": "Micro Euro FX Futures"},
    # Bonds (CBOT)
    "ZB": {"exchange": "CBOT", "currency": "USD", "description": "30-Year Treasury Bond"},
    "ZN": {"exchange": "CBOT", "currency": "USD", "description": "10-Year Treasury Note"},
    "ZT": {"exchange": "CBOT", "currency": "USD", "description": "2-Year Treasury Note"},
    "ZF": {"exchange": "CBOT", "currency": "USD", "description": "5-Year Treasury Note"},
    # Grains (CBOT)
    "ZC": {"exchange": "CBOT", "currency": "USD", "description": "Corn Futures"},
    "ZS": {"exchange": "CBOT", "currency": "USD", "description": "Soybean Futures"},
    "ZW": {"exchange": "CBOT", "currency": "USD", "description": "Wheat Futures"},
    "ZM": {"exchange": "CBOT", "currency": "USD", "description": "Soybean Meal Futures"},
    "ZL": {"exchange": "CBOT", "currency": "USD", "description": "Soybean Oil Futures"},
}


# =========================
# IB Rate Limiter
# =========================


class IBRateLimiter:
    """
    Comprehensive IB TWS API rate limiter.

    IB has multiple rate limits that MUST be respected to avoid:
    - Temporary bans (automatic pacing violations)
    - Connection drops
    - "Max messages per second exceeded" errors

    Rate Limits:
    ─────────────────────────────────────────────────────────────────────────────
    | Type                    | Limit                  | Window    | Action     |
    |-------------------------|------------------------|-----------|------------|
    | General messages        | 50 msg/sec             | 1 sec     | Block      |
    | Historical data         | 60 requests            | 10 min    | Pacing     |
    | Identical hist request  | 6 requests             | 10 min    | Cache      |
    | Market data subscribe   | 1 subscription/sec     | 1 sec     | Block      |
    | Market data lines       | 100 concurrent         | N/A       | Hard limit |
    | Scanner subscriptions   | 10 concurrent          | N/A       | Hard limit |
    | Account updates         | 1 request/sec          | 1 sec     | Block      |
    ─────────────────────────────────────────────────────────────────────────────

    Reference:
    - https://interactivebrokers.github.io/tws-api/historical_limitations.html
    - https://interactivebrokers.github.io/tws-api/market_data.html
    """

    # Rate limit constants (with safety margins)
    MSG_PER_SEC = 45  # General: 50 limit, 45 for safety
    HIST_PER_10MIN = 55  # Historical: 60 limit, 55 for safety
    HIST_IDENTICAL_PER_10MIN = 5  # Identical requests: 6 limit
    SUBSCRIPTION_PER_SEC = 1  # Market data subscribe
    MAX_MARKET_DATA_LINES = 100  # Concurrent market data subscriptions
    MAX_SCANNER_SUBSCRIPTIONS = 10  # Concurrent scanner subscriptions

    def __init__(self) -> None:
        self._message_times: List[float] = []
        self._historical_times: List[float] = []
        self._historical_requests: Dict[str, List[float]] = {}  # For identical request tracking
        self._subscription_times: List[float] = []
        self._active_subscriptions: Set[str] = set()
        self._active_scanners: Set[str] = set()
        self._lock = threading.Lock()

    def can_send_message(self) -> bool:
        """Check if general message can be sent."""
        with self._lock:
            now = time.time()
            self._message_times = [t for t in self._message_times if now - t < 1.0]
            return len(self._message_times) < self.MSG_PER_SEC

    def record_message(self) -> None:
        """Record a message being sent."""
        with self._lock:
            self._message_times.append(time.time())

    def wait_for_message_slot(self, timeout: float = 5.0) -> bool:
        """Block until message can be sent or timeout."""
        start = time.time()
        while not self.can_send_message():
            if time.time() - start > timeout:
                return False
            time.sleep(0.02)  # 20ms poll
        self.record_message()
        return True

    def can_request_historical(self, request_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if historical data can be requested.

        Args:
            request_key: Unique key for this request (for identical request tracking)

        Returns:
            (can_request, reason_if_blocked)
        """
        now = time.time()
        window_10min = now - 600  # 10 minutes

        with self._lock:
            # Clean old entries
            self._historical_times = [t for t in self._historical_times if t > window_10min]

            # Check general historical limit
            if len(self._historical_times) >= self.HIST_PER_10MIN:
                wait_time = self._historical_times[0] - window_10min
                return False, f"Historical rate limit: wait {wait_time:.0f}s"

            # Check identical request limit
            if request_key:
                identical = self._historical_requests.get(request_key, [])
                identical = [t for t in identical if t > window_10min]
                self._historical_requests[request_key] = identical

                if len(identical) >= self.HIST_IDENTICAL_PER_10MIN:
                    wait_time = identical[0] - window_10min
                    return False, f"Identical request limit: wait {wait_time:.0f}s"

            return True, ""

    def record_historical_request(self, request_key: Optional[str] = None) -> None:
        """Record historical data request."""
        with self._lock:
            now = time.time()
            self._historical_times.append(now)
            if request_key:
                if request_key not in self._historical_requests:
                    self._historical_requests[request_key] = []
                self._historical_requests[request_key].append(now)

    def can_subscribe_market_data(self, symbol: str) -> Tuple[bool, str]:
        """Check if can subscribe to market data."""
        with self._lock:
            now = time.time()
            self._subscription_times = [t for t in self._subscription_times if now - t < 1.0]

            if len(self._subscription_times) >= self.SUBSCRIPTION_PER_SEC:
                return False, "Subscription rate limit: 1 per second"

            if len(self._active_subscriptions) >= self.MAX_MARKET_DATA_LINES:
                return False, f"Max market data lines reached ({self.MAX_MARKET_DATA_LINES})"

            return True, ""

    def record_subscription(self, symbol: str) -> None:
        """Record market data subscription."""
        with self._lock:
            self._subscription_times.append(time.time())
            self._active_subscriptions.add(symbol)

    def record_unsubscription(self, symbol: str) -> None:
        """Record market data unsubscription."""
        with self._lock:
            self._active_subscriptions.discard(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            now = time.time()
            return {
                "messages_this_second": len([t for t in self._message_times if now - t < 1.0]),
                "messages_per_sec_limit": self.MSG_PER_SEC,
                "historical_last_10min": len([t for t in self._historical_times if now - t < 600]),
                "historical_per_10min_limit": self.HIST_PER_10MIN,
                "active_subscriptions": len(self._active_subscriptions),
                "max_subscriptions": self.MAX_MARKET_DATA_LINES,
                "active_scanners": len(self._active_scanners),
                "max_scanners": self.MAX_SCANNER_SUBSCRIPTIONS,
            }


# =========================
# IB Connection Manager
# =========================


class IBConnectionManager:
    """
    Production-grade IB TWS connection lifecycle manager.

    Handles:
    - Automatic heartbeat every 30 seconds (IB requires activity)
    - Message rate limiting (IB limit: 50 msg/sec, we use 45 for safety)
    - Exponential backoff reconnection
    - Paper vs Live account routing
    - Connection state monitoring

    Port Reference:
    - 7496: TWS Live Trading
    - 7497: TWS Paper Trading
    - 4001: IB Gateway Live Trading
    - 4002: IB Gateway Paper Trading

    References:
    - IB TWS API: https://interactivebrokers.github.io/tws-api/
    - ib_insync: https://ib-insync.readthedocs.io/
    """

    HEARTBEAT_INTERVAL_SEC = 30  # IB requires activity every 60s, we use 30 for safety
    MAX_MESSAGES_PER_SEC = 45  # IB limit is 50, leave margin for safety
    RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60, 120]  # Exponential backoff (seconds)
    MAX_RECONNECT_ATTEMPTS = len(RECONNECT_DELAYS)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live
        client_id: int = 1,
        readonly: bool = False,
        account: Optional[str] = None,
    ) -> None:
        """
        Initialize connection manager.

        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port (7497=TWS Paper, 7496=TWS Live)
            client_id: Unique client ID (1-999)
            readonly: If True, no trading allowed (safer for data-only usage)
            account: Specific account to use (for multi-account setups)
        """
        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly
        self._account = account

        self._ib: Optional[IB] = None
        self._connected = False
        self._reconnect_count = 0
        self._last_heartbeat_ts: float = 0.0

        # Rate limiter
        self._rate_limiter = IBRateLimiter()

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to TWS/Gateway with retry logic.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully

        Raises:
            ConnectionError: If all connection attempts fail
            ImportError: If ib_insync is not installed
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IB adapters. " "Install with: pip install ib_insync"
            )

        for attempt, delay in enumerate(self.RECONNECT_DELAYS):
            try:
                self._ib = IB()
                self._ib.connect(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    readonly=self._readonly,
                    account=self._account,
                    timeout=timeout,
                )
                self._connected = True
                self._reconnect_count = 0
                self._last_heartbeat_ts = time.time()

                # Register disconnect handler
                self._ib.disconnectedEvent += self._on_disconnect

                logger.info(
                    f"Connected to IB TWS at {self._host}:{self._port} "
                    f"(client_id={self._client_id}, readonly={self._readonly})"
                )
                return True

            except Exception as e:
                self._reconnect_count = attempt + 1
                logger.warning(
                    f"IB connection attempt {attempt + 1}/{self.MAX_RECONNECT_ATTEMPTS} "
                    f"failed: {e}"
                )
                if attempt < len(self.RECONNECT_DELAYS) - 1:
                    time.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to IB after {self.MAX_RECONNECT_ATTEMPTS} "
                        f"attempts: {e}"
                    )

        return False

    def disconnect(self) -> None:
        """Safely disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB TWS")
        self._connected = False

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection - attempt reconnect."""
        self._connected = False
        logger.warning("IB connection lost, attempting reconnect...")
        try:
            self.connect()
        except ConnectionError as e:
            logger.error(f"Reconnection failed: {e}")

    def send_heartbeat(self) -> None:
        """Send heartbeat to keep connection alive."""
        if not self._connected or not self._ib:
            return

        now = time.time()
        if now - self._last_heartbeat_ts >= self.HEARTBEAT_INTERVAL_SEC:
            # Request current time as heartbeat
            try:
                self._ib.reqCurrentTime()
                self._last_heartbeat_ts = now
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                self._connected = False

    def wait_for_rate_limit(self) -> bool:
        """Block until rate limit allows sending."""
        return self._rate_limiter.wait_for_message_slot()

    @property
    def ib(self) -> IB:
        """Get IB instance, ensuring connected."""
        if not self._connected or not self._ib:
            raise ConnectionError("Not connected to IB")
        self.send_heartbeat()  # Opportunistic heartbeat
        return self._ib

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    @property
    def rate_limiter(self) -> IBRateLimiter:
        """Get rate limiter instance."""
        return self._rate_limiter

    def health_check(self) -> Dict[str, Any]:
        """Return connection health status."""
        return {
            "connected": self.is_connected,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "readonly": self._readonly,
            "reconnect_count": self._reconnect_count,
            "last_heartbeat": self._last_heartbeat_ts,
            "rate_limit_status": self._rate_limiter.get_status(),
        }


# =========================
# IB Market Data Adapter
# =========================


class IBMarketDataAdapter(MarketDataAdapter):
    """
    Interactive Brokers market data adapter.

    Provides access to CME Group futures market data via TWS API.

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live) or Gateway (4002 paper, 4001 live)
        client_id: Unique client ID (1-999)
        timeout: Connection timeout seconds
        readonly: If True, no trading allowed (safer for data-only usage)
        account: Specific account to use (for multi-account setups)
        default_exchange: Default exchange for symbol lookup (CME, CBOT, etc.)

    Rate Limits:
        - IB allows max 50 messages/second - we enforce 45 for safety
        - Historical data: max 60 requests per 10 minutes (pacing)
        - Market data lines: varies by subscription

    Usage:
        adapter = IBMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={"port": 7497, "client_id": 1}
        )
        adapter.connect()
        bars = adapter.get_bars("ES", "1h", limit=100)
        tick = adapter.get_tick("ES")
        adapter.disconnect()
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.IB,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._conn_manager: Optional[IBConnectionManager] = None
        self._host = self._config.get("host", "127.0.0.1")
        self._port = self._config.get("port", 7497)  # Paper trading default
        self._client_id = self._config.get("client_id", 1)
        self._readonly = self._config.get("readonly", True)
        self._account = self._config.get("account")
        self._default_exchange = self._config.get("default_exchange", "CME")

    def _do_connect(self) -> None:
        """Connect to TWS/Gateway with production-grade connection management."""
        self._conn_manager = IBConnectionManager(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            readonly=self._readonly,
            account=self._account,
        )
        self._conn_manager.connect(timeout=self._config.get("timeout", 10.0))

    def _do_disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._conn_manager:
            self._conn_manager.disconnect()
            self._conn_manager = None

    @property
    def _ib(self) -> IB:
        """Get IB instance with rate limiting."""
        if not self._conn_manager:
            raise ConnectionError("Not connected")
        self._conn_manager.wait_for_rate_limit()
        return self._conn_manager.ib

    def _create_contract(
        self,
        symbol: str,
        use_continuous: bool = True,
        exchange: Optional[str] = None,
    ):
        """
        Create IB Future contract.

        Args:
            symbol: Contract symbol (ES, NQ, GC, CL, 6E)
            use_continuous: Use continuous contract (auto-roll)
            exchange: Exchange override (CME, CBOT, NYMEX, COMEX)

        Returns:
            IB Future or ContFuture contract object
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync is required for IB adapters")

        # Get contract details from mapping
        details = CONTRACT_MAP.get(
            symbol.upper(), {"exchange": exchange or self._default_exchange, "currency": "USD"}
        )

        actual_exchange = exchange or details.get("exchange", self._default_exchange)

        if use_continuous:
            return ContFuture(symbol, exchange=actual_exchange)
        else:
            # Specific contract (need expiry)
            return Future(symbol, exchange=actual_exchange)

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
        Fetch historical bars from IB.

        Note: IB has pacing limits - max 60 requests per 10 minutes.

        Args:
            symbol: Futures symbol (ES, NQ, GC, etc.)
            timeframe: Bar timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Maximum number of bars to return
            start_ts: Start timestamp in milliseconds (inclusive)
            end_ts: End timestamp in milliseconds (exclusive)

        Returns:
            List of Bar objects, oldest first
        """
        # Check rate limit
        request_key = f"{symbol}:{timeframe}:{limit}"
        can_request, reason = self._conn_manager.rate_limiter.can_request_historical(request_key)
        if not can_request:
            logger.warning(f"Rate limit blocked: {reason}")
            return []

        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        # Convert timeframe to IB format
        bar_size = self._convert_timeframe(timeframe)
        duration = self._calculate_duration(limit, timeframe)

        # End datetime for historical request
        end_datetime = ""
        if end_ts:
            end_datetime = datetime.utcfromtimestamp(end_ts / 1000).strftime("%Y%m%d %H:%M:%S")

        bars_raw = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,  # Include extended trading hours
            formatDate=1,
        )

        # Record the request
        self._conn_manager.rate_limiter.record_historical_request(request_key)

        return [self._parse_ib_bar(symbol, b) for b in bars_raw]

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Futures symbol
            timeframe: Bar timeframe

        Returns:
            Latest Bar or None if unavailable
        """
        bars = self.get_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current quote for a futures contract.

        Args:
            symbol: Futures symbol

        Returns:
            Current Tick or None if unavailable
        """
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        # Request snapshot
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(1)  # Wait for data to arrive

        # Cancel subscription after snapshot
        self._ib.cancelMktData(contract)

        if ticker.bid is None or ticker.ask is None:
            return None

        # Convert IB timestamp
        ts_ms = int(time.time() * 1000)
        if ticker.time:
            try:
                ts_ms = int(util.dt(ticker.time).timestamp() * 1000)
            except Exception:
                pass

        return Tick(
            ts=ts_ms,
            symbol=symbol,
            price=Decimal(str(ticker.last)) if ticker.last else None,
            bid=Decimal(str(ticker.bid)),
            ask=Decimal(str(ticker.ask)),
            bid_qty=Decimal(str(ticker.bidSize)) if ticker.bidSize else None,
            ask_qty=Decimal(str(ticker.askSize)) if ticker.askSize else None,
        )

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream real-time bars.

        Note: IB does not provide direct bar streaming.
        This implementation polls for the latest bar at the specified interval.

        Args:
            symbols: List of symbols to stream
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects as they become available
        """
        timeframe = self._ms_to_timeframe(interval_ms)
        last_bars: Dict[str, Optional[Bar]] = {s: None for s in symbols}

        while True:
            for symbol in symbols:
                try:
                    bar = self.get_latest_bar(symbol, timeframe)
                    if bar and (last_bars[symbol] is None or bar.ts > last_bars[symbol].ts):
                        last_bars[symbol] = bar
                        yield bar
                except Exception as e:
                    logger.warning(f"Error streaming bar for {symbol}: {e}")

            # Sleep until next interval
            time.sleep(interval_ms / 1000.0)

    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks (BBO updates).

        Args:
            symbols: List of symbols to stream

        Yields:
            Tick objects as they become available
        """
        # Subscribe to all symbols
        contracts = []
        for symbol in symbols:
            can_sub, reason = self._conn_manager.rate_limiter.can_subscribe_market_data(symbol)
            if not can_sub:
                logger.warning(f"Cannot subscribe to {symbol}: {reason}")
                continue

            contract = self._create_contract(symbol)
            self._ib.qualifyContracts(contract)
            self._ib.reqMktData(contract)
            self._conn_manager.rate_limiter.record_subscription(symbol)
            contracts.append((symbol, contract))

        try:
            while True:
                self._ib.sleep(0.1)  # Small sleep to process callbacks

                for symbol, contract in contracts:
                    ticker = self._ib.ticker(contract)
                    if ticker and ticker.bid is not None and ticker.ask is not None:
                        ts_ms = int(time.time() * 1000)
                        yield Tick(
                            ts=ts_ms,
                            symbol=symbol,
                            price=Decimal(str(ticker.last)) if ticker.last else None,
                            bid=Decimal(str(ticker.bid)),
                            ask=Decimal(str(ticker.ask)),
                            bid_qty=Decimal(str(ticker.bidSize)) if ticker.bidSize else None,
                            ask_qty=Decimal(str(ticker.askSize)) if ticker.askSize else None,
                        )
        finally:
            # Cleanup subscriptions
            for symbol, contract in contracts:
                self._ib.cancelMktData(contract)
                self._conn_manager.rate_limiter.record_unsubscription(symbol)

    def _parse_ib_bar(self, symbol: str, ib_bar: Any) -> Bar:
        """Convert IB bar to our Bar model."""
        # IB returns datetime objects
        ts_ms = int(ib_bar.date.timestamp() * 1000) if hasattr(ib_bar.date, "timestamp") else 0

        return Bar(
            ts=ts_ms,
            symbol=symbol,
            open=Decimal(str(ib_bar.open)),
            high=Decimal(str(ib_bar.high)),
            low=Decimal(str(ib_bar.low)),
            close=Decimal(str(ib_bar.close)),
            volume=Decimal(str(ib_bar.volume)),
        )

    @staticmethod
    def _convert_timeframe(timeframe: str) -> str:
        """Convert timeframe to IB bar size format."""
        mapping = {
            "1m": "1 min",
            "2m": "2 mins",
            "3m": "3 mins",
            "5m": "5 mins",
            "10m": "10 mins",
            "15m": "15 mins",
            "20m": "20 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "2h": "2 hours",
            "3h": "3 hours",
            "4h": "4 hours",
            "8h": "8 hours",
            "1d": "1 day",
            "1w": "1 week",
            "1M": "1 month",
        }
        return mapping.get(timeframe, "1 hour")

    @staticmethod
    def _calculate_duration(limit: int, timeframe: str) -> str:
        """Calculate IB duration string based on limit and timeframe."""
        # Map timeframe to approximate minutes
        timeframe_minutes = {
            "1m": 1,
            "2m": 2,
            "3m": 3,
            "5m": 5,
            "10m": 10,
            "15m": 15,
            "20m": 20,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "3h": 180,
            "4h": 240,
            "8h": 480,
            "1d": 1440,
            "1w": 10080,
            "1M": 43200,
        }

        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = limit * minutes

        # Convert to appropriate duration unit
        if total_minutes <= 60:
            return f"{total_minutes} M"  # Minutes
        elif total_minutes <= 1440:
            return f"{(total_minutes // 60) + 1} H"  # Hours
        elif total_minutes <= 43200:
            return f"{(total_minutes // 1440) + 1} D"  # Days
        elif total_minutes <= 302400:
            return f"{(total_minutes // 10080) + 1} W"  # Weeks
        else:
            return f"{(total_minutes // 43200) + 1} M"  # Months

    @staticmethod
    def _ms_to_timeframe(interval_ms: int) -> str:
        """Convert milliseconds to timeframe string."""
        minutes = interval_ms // 60000
        if minutes <= 1:
            return "1m"
        elif minutes <= 5:
            return "5m"
        elif minutes <= 15:
            return "15m"
        elif minutes <= 30:
            return "30m"
        elif minutes <= 60:
            return "1h"
        elif minutes <= 240:
            return "4h"
        else:
            return "1d"

    def get_contract_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get contract details from IB.

        Args:
            symbol: Futures symbol

        Returns:
            Contract details dictionary
        """
        contract = self._create_contract(symbol, use_continuous=False)
        details = self._ib.reqContractDetails(contract)

        if not details:
            return {}

        d = details[0]
        return {
            "symbol": symbol,
            "exchange": d.contract.exchange,
            "currency": d.contract.currency,
            "multiplier": d.contract.multiplier,
            "tick_size": d.minTick,
            "trading_class": d.contract.tradingClass,
            "long_name": d.longName,
            "valid_exchanges": d.validExchanges,
        }
