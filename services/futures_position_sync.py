# -*- coding: utf-8 -*-
"""
services/futures_position_sync.py
Futures-specific position synchronization service.

Phase 9: Unified Live Trading (2025-12-02)

This module provides futures-specific position reconciliation:
- Sync local state with exchange (Binance Futures, CME via IB)
- Handle futures-specific events (ADL, liquidation, funding)
- Track leverage and margin state
- Support for both crypto perpetual and CME futures

Key Differences from Spot/Equity Position Sync:
1. Leverage tracking (position notional vs margin used)
2. Mark price vs Entry price divergence
3. ADL (Auto-Deleveraging) event detection
4. Liquidation event detection
5. Unrealized PnL tracking with mark-to-market
6. Funding payment tracking (crypto perpetual)
7. Settlement event handling (CME)

Design Principles:
- Extends base position_sync.py patterns
- Thread-safe for multi-symbol trading
- Designed for deployment with comprehensive logging
- Supports paper and live trading modes

References:
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- CME Group: https://www.cmegroup.com/
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

from core_futures import (
    FuturesPosition,
    FuturesAccountState,
    FuturesType,
    MarginMode,
    PositionSide,
    Exchange,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Sync intervals
DEFAULT_SYNC_INTERVAL_SEC = 10.0
MIN_SYNC_INTERVAL_SEC = 1.0
MAX_SYNC_INTERVAL_SEC = 300.0

# Tolerance for position comparison
DEFAULT_QTY_TOLERANCE_PCT = 0.001  # 0.1%
DEFAULT_PRICE_TOLERANCE_PCT = 0.01  # 1%
DEFAULT_LEVERAGE_TOLERANCE = 0  # Must match exactly

# ADL detection thresholds
ADL_WARNING_PERCENTILE = 70.0
ADL_DANGER_PERCENTILE = 85.0
ADL_CRITICAL_PERCENTILE = 95.0

# Retry configuration
MAX_SYNC_RETRIES = 3
RETRY_BACKOFF_SECONDS = [1.0, 2.0, 5.0]


# =============================================================================
# Enums
# =============================================================================


class FuturesSyncEventType(str, Enum):
    """Types of futures-specific sync events."""

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    QTY_MISMATCH = "qty_mismatch"
    LEVERAGE_MISMATCH = "leverage_mismatch"

    # Futures-specific events
    LIQUIDATION_DETECTED = "liquidation_detected"
    ADL_DETECTED = "adl_detected"
    FUNDING_RECEIVED = "funding_received"
    FUNDING_PAID = "funding_paid"
    SETTLEMENT_OCCURRED = "settlement_occurred"

    # Margin events
    MARGIN_CALL = "margin_call"
    MARGIN_RATIO_LOW = "margin_ratio_low"

    # Sync status
    SYNC_SUCCESS = "sync_success"
    SYNC_FAILURE = "sync_failure"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"


class ADLRiskLevel(str, Enum):
    """ADL (Auto-Deleveraging) risk level."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


# =============================================================================
# Protocols
# =============================================================================


class FuturesPositionProvider(Protocol):
    """Protocol for futures position data providers."""

    def get_futures_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[FuturesPosition]:
        """Get current futures positions from exchange."""
        ...


class FuturesAccountProvider(Protocol):
    """Protocol for futures account data providers."""

    def get_futures_account(self) -> FuturesAccountState:
        """Get futures account state from exchange."""
        ...


class FuturesOrderProvider(Protocol):
    """Protocol for futures order data providers."""

    def get_open_futures_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Any]:
        """Get open futures orders from exchange."""
        ...


class ADLIndicatorProvider(Protocol):
    """Protocol for ADL indicator data providers."""

    def get_adl_indicator(
        self,
        symbol: str,
        position_side: PositionSide,
    ) -> int:
        """Get ADL queue indicator (1-5 lights)."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FuturesPositionDiff:
    """Represents a difference between local and remote futures positions."""

    symbol: str
    event_type: FuturesSyncEventType

    # Position data
    local_qty: Optional[Decimal] = None
    remote_qty: Optional[Decimal] = None
    local_leverage: Optional[int] = None
    remote_leverage: Optional[int] = None
    local_margin_mode: Optional[MarginMode] = None
    remote_margin_mode: Optional[MarginMode] = None

    # PnL data
    local_unrealized_pnl: Optional[Decimal] = None
    remote_unrealized_pnl: Optional[Decimal] = None
    local_entry_price: Optional[Decimal] = None
    remote_entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None

    # ADL data
    adl_percentile: Optional[float] = None
    adl_risk_level: Optional[ADLRiskLevel] = None

    # Timestamp
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    details: str = ""

    @property
    def qty_diff(self) -> Decimal:
        """Quantity difference (remote - local)."""
        local = self.local_qty or Decimal("0")
        remote = self.remote_qty or Decimal("0")
        return remote - local

    @property
    def pnl_diff(self) -> Decimal:
        """Unrealized PnL difference (remote - local)."""
        local = self.local_unrealized_pnl or Decimal("0")
        remote = self.remote_unrealized_pnl or Decimal("0")
        return remote - local

    @property
    def is_position_closed_externally(self) -> bool:
        """Check if position was closed externally (liquidation/ADL)."""
        return self.event_type in (
            FuturesSyncEventType.LIQUIDATION_DETECTED,
            FuturesSyncEventType.ADL_DETECTED,
        )


@dataclass
class FuturesSyncResult:
    """Result of a futures position synchronization operation."""

    timestamp_ms: int
    success: bool
    exchange: Exchange
    futures_type: FuturesType

    # Position data
    local_positions: Dict[str, FuturesPosition] = field(default_factory=dict)
    remote_positions: Dict[str, FuturesPosition] = field(default_factory=dict)

    # Events detected
    events: List[FuturesPositionDiff] = field(default_factory=list)

    # Account state
    account_balance: Decimal = Decimal("0")
    total_margin_used: Decimal = Decimal("0")
    margin_ratio: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Order data
    open_orders_count: int = 0

    # Error info
    error: Optional[str] = None
    retry_count: int = 0

    @property
    def has_events(self) -> bool:
        """Whether any events were detected."""
        return len(self.events) > 0

    @property
    def has_liquidations(self) -> bool:
        """Whether any liquidations were detected."""
        return any(
            e.event_type == FuturesSyncEventType.LIQUIDATION_DETECTED
            for e in self.events
        )

    @property
    def has_adl_events(self) -> bool:
        """Whether any ADL events were detected."""
        return any(
            e.event_type == FuturesSyncEventType.ADL_DETECTED
            for e in self.events
        )

    @property
    def position_count_diff(self) -> int:
        """Difference in position count (remote - local)."""
        return len(self.remote_positions) - len(self.local_positions)


@dataclass
class FuturesSyncConfig:
    """Configuration for futures position synchronization."""

    # Exchange settings
    exchange: Exchange = Exchange.BINANCE
    futures_type: FuturesType = FuturesType.CRYPTO_PERPETUAL

    # Sync intervals
    sync_interval_sec: float = DEFAULT_SYNC_INTERVAL_SEC
    account_sync_interval_sec: float = 30.0  # Account state sync

    # Tolerances
    qty_tolerance_pct: float = DEFAULT_QTY_TOLERANCE_PCT
    price_tolerance_pct: float = DEFAULT_PRICE_TOLERANCE_PCT
    leverage_tolerance: int = DEFAULT_LEVERAGE_TOLERANCE

    # ADL monitoring
    enable_adl_monitoring: bool = True
    adl_warning_threshold: float = ADL_WARNING_PERCENTILE
    adl_danger_threshold: float = ADL_DANGER_PERCENTILE

    # Liquidation detection
    enable_liquidation_detection: bool = True

    # Funding tracking
    enable_funding_tracking: bool = True

    # Auto-reconciliation (DANGEROUS - use with caution!)
    auto_reconcile: bool = False
    max_auto_reconcile_qty: Decimal = Decimal("0.1")  # Max qty to auto-fix

    # Retry settings
    max_retries: int = MAX_SYNC_RETRIES

    # Symbol filtering
    include_symbols: Optional[List[str]] = None
    exclude_symbols: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.sync_interval_sec < MIN_SYNC_INTERVAL_SEC:
            raise ValueError(
                f"sync_interval_sec must be >= {MIN_SYNC_INTERVAL_SEC}"
            )
        if self.sync_interval_sec > MAX_SYNC_INTERVAL_SEC:
            raise ValueError(
                f"sync_interval_sec must be <= {MAX_SYNC_INTERVAL_SEC}"
            )


# =============================================================================
# Futures Position Synchronizer
# =============================================================================


class FuturesPositionSynchronizer:
    """
    Service for synchronizing futures positions between local state and exchange.

    Provides:
    - One-time sync operations
    - Periodic background sync
    - ADL risk monitoring
    - Liquidation detection
    - Funding payment tracking
    - Margin ratio monitoring

    Usage:
        sync = FuturesPositionSynchronizer(
            position_provider=binance_futures_adapter,
            account_provider=binance_futures_adapter,
            local_state_getter=lambda: my_state.futures_positions,
            config=FuturesSyncConfig(
                exchange=Exchange.BINANCE,
                futures_type=FuturesType.CRYPTO_PERPETUAL,
            ),
        )

        # One-time sync
        result = sync.sync_once()
        if result.has_liquidations:
            handle_liquidation_event(result)

        # Start background sync
        sync.start_background_sync()
    """

    def __init__(
        self,
        position_provider: FuturesPositionProvider,
        account_provider: FuturesAccountProvider,
        local_state_getter: Callable[[], Dict[str, FuturesPosition]],
        config: Optional[FuturesSyncConfig] = None,
        order_provider: Optional[FuturesOrderProvider] = None,
        adl_provider: Optional[ADLIndicatorProvider] = None,
        on_event: Optional[Callable[[FuturesPositionDiff], None]] = None,
        on_sync_complete: Optional[Callable[[FuturesSyncResult], None]] = None,
        on_liquidation: Optional[Callable[[FuturesPositionDiff], None]] = None,
        on_adl: Optional[Callable[[FuturesPositionDiff], None]] = None,
    ) -> None:
        """
        Initialize futures position synchronizer.

        Args:
            position_provider: Provider for exchange position data
            account_provider: Provider for exchange account data
            local_state_getter: Function returning local position state
            config: Sync configuration
            order_provider: Optional provider for open orders
            adl_provider: Optional provider for ADL indicators
            on_event: Callback for any sync event
            on_sync_complete: Callback when sync completes
            on_liquidation: Callback specifically for liquidation events
            on_adl: Callback specifically for ADL events
        """
        self._position_provider = position_provider
        self._account_provider = account_provider
        self._local_state_getter = local_state_getter
        self._config = config or FuturesSyncConfig()
        self._order_provider = order_provider
        self._adl_provider = adl_provider

        # Callbacks
        self._on_event = on_event
        self._on_sync_complete = on_sync_complete
        self._on_liquidation = on_liquidation
        self._on_adl = on_adl

        # Internal state
        self._lock = threading.RLock()
        self._last_sync_result: Optional[FuturesSyncResult] = None
        self._last_sync_time: float = 0.0
        self._last_account_state: Optional[FuturesAccountState] = None
        self._known_positions: Dict[str, FuturesPosition] = {}
        self._pending_liquidations: Set[str] = set()
        self._pending_adl: Set[str] = set()

        # Background sync state
        self._background_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Metrics
        self._sync_count = 0
        self._error_count = 0
        self._liquidation_count = 0
        self._adl_count = 0

    def sync_once(self) -> FuturesSyncResult:
        """
        Perform a single sync operation.

        Returns:
            FuturesSyncResult with sync status and detected events
        """
        with self._lock:
            timestamp_ms = int(time.time() * 1000)
            events: List[FuturesPositionDiff] = []
            error: Optional[str] = None
            retry_count = 0

            # Get local positions
            try:
                local_positions = self._local_state_getter()
            except Exception as e:
                logger.error(f"Failed to get local positions: {e}")
                local_positions = {}

            # Get remote positions with retry
            remote_positions: Dict[str, FuturesPosition] = {}
            account_state: Optional[FuturesAccountState] = None

            for attempt in range(self._config.max_retries):
                try:
                    # Fetch positions
                    raw_positions = self._position_provider.get_futures_positions(
                        symbols=self._config.include_symbols
                    )
                    remote_positions = {p.symbol: p for p in raw_positions}

                    # Fetch account state
                    account_state = self._account_provider.get_futures_account()

                    break  # Success

                except Exception as e:
                    retry_count = attempt + 1
                    error = str(e)
                    logger.warning(
                        f"Sync attempt {retry_count}/{self._config.max_retries} "
                        f"failed: {e}"
                    )
                    if attempt < self._config.max_retries - 1:
                        time.sleep(RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)])

            if retry_count == self._config.max_retries:
                logger.error(f"Sync failed after {retry_count} retries: {error}")
                result = FuturesSyncResult(
                    timestamp_ms=timestamp_ms,
                    success=False,
                    exchange=self._config.exchange,
                    futures_type=self._config.futures_type,
                    local_positions=local_positions,
                    error=error,
                    retry_count=retry_count,
                )
                self._last_sync_result = result
                self._error_count += 1
                return result

            # Filter excluded symbols
            if self._config.exclude_symbols:
                for sym in self._config.exclude_symbols:
                    remote_positions.pop(sym, None)

            # Compare positions and detect events
            events = self._compare_positions(
                local_positions,
                remote_positions,
                timestamp_ms
            )

            # Check for liquidation events
            if self._config.enable_liquidation_detection:
                liquidation_events = self._detect_liquidations(
                    local_positions,
                    remote_positions,
                    timestamp_ms
                )
                events.extend(liquidation_events)

            # Check ADL indicators
            if self._config.enable_adl_monitoring and self._adl_provider:
                adl_events = self._check_adl_indicators(
                    remote_positions,
                    timestamp_ms
                )
                events.extend(adl_events)

            # Get open orders count
            open_orders_count = 0
            if self._order_provider:
                try:
                    orders = self._order_provider.get_open_futures_orders()
                    open_orders_count = len(orders)
                except Exception as e:
                    logger.warning(f"Failed to get open orders: {e}")

            # Build result
            result = FuturesSyncResult(
                timestamp_ms=timestamp_ms,
                success=True,
                exchange=self._config.exchange,
                futures_type=self._config.futures_type,
                local_positions=local_positions,
                remote_positions=remote_positions,
                events=events,
                account_balance=account_state.total_wallet_balance if account_state else Decimal("0"),
                total_margin_used=account_state.total_initial_margin if account_state else Decimal("0"),
                margin_ratio=account_state.margin_ratio if account_state else Decimal("0"),
                unrealized_pnl=account_state.total_unrealized_pnl if account_state else Decimal("0"),
                open_orders_count=open_orders_count,
                retry_count=retry_count,
            )

            # Update internal state
            self._last_sync_result = result
            self._last_sync_time = time.time()
            self._last_account_state = account_state
            self._known_positions = dict(remote_positions)
            self._sync_count += 1

            # Fire callbacks
            self._fire_callbacks(result, events)

            return result

    def _compare_positions(
        self,
        local: Dict[str, FuturesPosition],
        remote: Dict[str, FuturesPosition],
        timestamp_ms: int,
    ) -> List[FuturesPositionDiff]:
        """Compare local and remote positions."""
        events: List[FuturesPositionDiff] = []

        all_symbols = set(local.keys()) | set(remote.keys())

        for symbol in all_symbols:
            local_pos = local.get(symbol)
            remote_pos = remote.get(symbol)

            if local_pos is None and remote_pos is not None:
                # New position opened (or missed opening)
                if abs(remote_pos.qty) > Decimal("0"):
                    events.append(FuturesPositionDiff(
                        symbol=symbol,
                        event_type=FuturesSyncEventType.POSITION_OPENED,
                        remote_qty=remote_pos.qty,
                        remote_leverage=remote_pos.leverage,
                        remote_entry_price=remote_pos.entry_price,
                        remote_unrealized_pnl=remote_pos.unrealized_pnl,
                        timestamp_ms=timestamp_ms,
                        details="Position opened on exchange but not tracked locally",
                    ))

            elif local_pos is not None and remote_pos is None:
                # Position closed (could be liquidation/ADL)
                events.append(FuturesPositionDiff(
                    symbol=symbol,
                    event_type=FuturesSyncEventType.POSITION_CLOSED,
                    local_qty=local_pos.qty,
                    local_leverage=local_pos.leverage,
                    local_entry_price=local_pos.entry_price,
                    local_unrealized_pnl=local_pos.unrealized_pnl,
                    timestamp_ms=timestamp_ms,
                    details="Position closed on exchange",
                ))

            elif local_pos is not None and remote_pos is not None:
                # Both exist - check for differences
                if abs(remote_pos.qty) == Decimal("0"):
                    # Position fully closed
                    events.append(FuturesPositionDiff(
                        symbol=symbol,
                        event_type=FuturesSyncEventType.POSITION_CLOSED,
                        local_qty=local_pos.qty,
                        remote_qty=Decimal("0"),
                        timestamp_ms=timestamp_ms,
                        details="Position quantity is zero on exchange",
                    ))
                    continue

                # Check quantity mismatch
                qty_diff = abs(remote_pos.qty - local_pos.qty)
                qty_tolerance = abs(local_pos.qty) * Decimal(str(self._config.qty_tolerance_pct))

                if qty_diff > qty_tolerance:
                    events.append(FuturesPositionDiff(
                        symbol=symbol,
                        event_type=FuturesSyncEventType.QTY_MISMATCH,
                        local_qty=local_pos.qty,
                        remote_qty=remote_pos.qty,
                        local_entry_price=local_pos.entry_price,
                        remote_entry_price=remote_pos.entry_price,
                        timestamp_ms=timestamp_ms,
                        details=f"Quantity mismatch: local={local_pos.qty}, remote={remote_pos.qty}",
                    ))

                # Check leverage mismatch
                if local_pos.leverage != remote_pos.leverage:
                    if abs(local_pos.leverage - remote_pos.leverage) > self._config.leverage_tolerance:
                        events.append(FuturesPositionDiff(
                            symbol=symbol,
                            event_type=FuturesSyncEventType.LEVERAGE_MISMATCH,
                            local_leverage=local_pos.leverage,
                            remote_leverage=remote_pos.leverage,
                            timestamp_ms=timestamp_ms,
                            details=f"Leverage mismatch: local={local_pos.leverage}x, remote={remote_pos.leverage}x",
                        ))

        return events

    def _detect_liquidations(
        self,
        local: Dict[str, FuturesPosition],
        remote: Dict[str, FuturesPosition],
        timestamp_ms: int,
    ) -> List[FuturesPositionDiff]:
        """Detect potential liquidation events."""
        events: List[FuturesPositionDiff] = []

        for symbol, local_pos in local.items():
            if symbol not in remote or abs(remote.get(symbol, FuturesPosition(symbol=symbol, side=PositionSide.BOTH, entry_price=Decimal("0"), qty=Decimal("0"), leverage=1, margin_mode=MarginMode.CROSS)).qty) == Decimal("0"):
                # Position disappeared - check if it was a forced liquidation
                # This is heuristic: sudden position close with high leverage
                # and significant unrealized loss suggests liquidation

                was_high_leverage = local_pos.leverage >= 10
                had_unrealized_loss = local_pos.unrealized_pnl < Decimal("0")
                was_tracked = symbol in self._known_positions

                # Check if we already processed this as regular close
                if was_tracked and was_high_leverage:
                    # Could be liquidation - mark for verification
                    if symbol not in self._pending_liquidations:
                        self._pending_liquidations.add(symbol)
                        events.append(FuturesPositionDiff(
                            symbol=symbol,
                            event_type=FuturesSyncEventType.LIQUIDATION_DETECTED,
                            local_qty=local_pos.qty,
                            local_leverage=local_pos.leverage,
                            local_entry_price=local_pos.entry_price,
                            local_unrealized_pnl=local_pos.unrealized_pnl,
                            timestamp_ms=timestamp_ms,
                            details="Potential liquidation detected (position disappeared with high leverage)",
                        ))
                        self._liquidation_count += 1

        return events

    def _check_adl_indicators(
        self,
        remote: Dict[str, FuturesPosition],
        timestamp_ms: int,
    ) -> List[FuturesPositionDiff]:
        """Check ADL (Auto-Deleveraging) indicators for positions."""
        events: List[FuturesPositionDiff] = []

        for symbol, pos in remote.items():
            if abs(pos.qty) == Decimal("0"):
                continue

            try:
                # Get ADL indicator (1-5 lights on Binance)
                adl_lights = self._adl_provider.get_adl_indicator(
                    symbol,
                    PositionSide.LONG if pos.qty > Decimal("0") else PositionSide.SHORT
                )

                # Convert to percentile (5 lights = 80-100%, 4 = 60-80%, etc.)
                adl_percentile = (adl_lights / 5.0) * 100.0

                # Determine risk level
                if adl_percentile >= self._config.adl_danger_threshold:
                    risk_level = ADLRiskLevel.CRITICAL if adl_percentile >= ADL_CRITICAL_PERCENTILE else ADLRiskLevel.DANGER

                    events.append(FuturesPositionDiff(
                        symbol=symbol,
                        event_type=FuturesSyncEventType.ADL_DETECTED,
                        remote_qty=pos.qty,
                        remote_leverage=pos.leverage,
                        adl_percentile=adl_percentile,
                        adl_risk_level=risk_level,
                        timestamp_ms=timestamp_ms,
                        details=f"High ADL risk: {adl_lights}/5 lights ({adl_percentile:.0f}%)",
                    ))

                    if symbol not in self._pending_adl:
                        self._pending_adl.add(symbol)
                        self._adl_count += 1

                elif adl_percentile >= self._config.adl_warning_threshold:
                    events.append(FuturesPositionDiff(
                        symbol=symbol,
                        event_type=FuturesSyncEventType.ADL_DETECTED,
                        remote_qty=pos.qty,
                        remote_leverage=pos.leverage,
                        adl_percentile=adl_percentile,
                        adl_risk_level=ADLRiskLevel.WARNING,
                        timestamp_ms=timestamp_ms,
                        details=f"ADL warning: {adl_lights}/5 lights ({adl_percentile:.0f}%)",
                    ))

            except Exception as e:
                logger.warning(f"Failed to get ADL indicator for {symbol}: {e}")

        return events

    def _fire_callbacks(
        self,
        result: FuturesSyncResult,
        events: List[FuturesPositionDiff],
    ) -> None:
        """Fire registered callbacks."""
        # General event callback
        if self._on_event:
            for event in events:
                try:
                    self._on_event(event)
                except Exception as e:
                    logger.error(f"Error in on_event callback: {e}")

        # Liquidation callback
        if self._on_liquidation:
            for event in events:
                if event.event_type == FuturesSyncEventType.LIQUIDATION_DETECTED:
                    try:
                        self._on_liquidation(event)
                    except Exception as e:
                        logger.error(f"Error in on_liquidation callback: {e}")

        # ADL callback
        if self._on_adl:
            for event in events:
                if event.event_type == FuturesSyncEventType.ADL_DETECTED:
                    try:
                        self._on_adl(event)
                    except Exception as e:
                        logger.error(f"Error in on_adl callback: {e}")

        # Sync complete callback
        if self._on_sync_complete:
            try:
                self._on_sync_complete(result)
            except Exception as e:
                logger.error(f"Error in on_sync_complete callback: {e}")

    def start_background_sync(self) -> None:
        """Start background synchronization loop."""
        if self._is_running:
            logger.warning("Background sync already running")
            return

        self._stop_event.clear()
        self._is_running = True

        # Run in thread for sync version
        thread = threading.Thread(
            target=self._background_sync_loop,
            name="FuturesPositionSync",
            daemon=True,
        )
        thread.start()
        logger.info("Started background futures position sync")

    def stop_background_sync(self) -> None:
        """Stop background synchronization loop."""
        if not self._is_running:
            return

        self._stop_event.set()
        self._is_running = False
        logger.info("Stopped background futures position sync")

    def _background_sync_loop(self) -> None:
        """Background sync loop (runs in thread)."""
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as e:
                logger.error(f"Error in background sync: {e}")
                self._error_count += 1

            # Wait for next interval
            self._stop_event.wait(self._config.sync_interval_sec)

    async def start_background_sync_async(self) -> None:
        """Start asynchronous background synchronization."""
        if self._is_running:
            logger.warning("Background sync already running")
            return

        self._stop_event.clear()
        self._is_running = True
        self._background_task = asyncio.create_task(
            self._background_sync_loop_async()
        )
        logger.info("Started async background futures position sync")

    async def stop_background_sync_async(self) -> None:
        """Stop asynchronous background synchronization."""
        if not self._is_running:
            return

        self._stop_event.set()
        self._is_running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped async background futures position sync")

    async def _background_sync_loop_async(self) -> None:
        """Async background sync loop."""
        while not self._stop_event.is_set():
            try:
                # Run sync in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.sync_once)
            except Exception as e:
                logger.error(f"Error in async background sync: {e}")
                self._error_count += 1

            await asyncio.sleep(self._config.sync_interval_sec)

    @property
    def last_sync_result(self) -> Optional[FuturesSyncResult]:
        """Get the last sync result."""
        return self._last_sync_result

    @property
    def last_account_state(self) -> Optional[FuturesAccountState]:
        """Get the last account state."""
        return self._last_account_state

    @property
    def is_running(self) -> bool:
        """Check if background sync is running."""
        return self._is_running

    @property
    def sync_count(self) -> int:
        """Total number of successful syncs."""
        return self._sync_count

    @property
    def error_count(self) -> int:
        """Total number of sync errors."""
        return self._error_count

    @property
    def liquidation_count(self) -> int:
        """Total number of liquidations detected."""
        return self._liquidation_count

    @property
    def adl_count(self) -> int:
        """Total number of ADL events detected."""
        return self._adl_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get synchronizer metrics."""
        return {
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "liquidation_count": self._liquidation_count,
            "adl_count": self._adl_count,
            "is_running": self._is_running,
            "last_sync_time": self._last_sync_time,
            "known_positions": len(self._known_positions),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_crypto_futures_sync(
    position_provider: FuturesPositionProvider,
    account_provider: FuturesAccountProvider,
    local_state_getter: Callable[[], Dict[str, FuturesPosition]],
    **kwargs,
) -> FuturesPositionSynchronizer:
    """
    Create a synchronizer for crypto futures (Binance).

    Args:
        position_provider: Binance futures adapter
        account_provider: Binance futures adapter
        local_state_getter: Function returning local positions
        **kwargs: Additional config options

    Returns:
        Configured FuturesPositionSynchronizer
    """
    config = FuturesSyncConfig(
        exchange=Exchange.BINANCE,
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        sync_interval_sec=kwargs.get("sync_interval_sec", 10.0),
        enable_adl_monitoring=kwargs.get("enable_adl_monitoring", True),
        enable_liquidation_detection=kwargs.get("enable_liquidation_detection", True),
        enable_funding_tracking=kwargs.get("enable_funding_tracking", True),
    )

    return FuturesPositionSynchronizer(
        position_provider=position_provider,
        account_provider=account_provider,
        local_state_getter=local_state_getter,
        config=config,
        **{k: v for k, v in kwargs.items() if k not in ("sync_interval_sec", "enable_adl_monitoring", "enable_liquidation_detection", "enable_funding_tracking")}
    )


def create_cme_futures_sync(
    position_provider: FuturesPositionProvider,
    account_provider: FuturesAccountProvider,
    local_state_getter: Callable[[], Dict[str, FuturesPosition]],
    futures_type: FuturesType = FuturesType.INDEX_FUTURES,
    **kwargs,
) -> FuturesPositionSynchronizer:
    """
    Create a synchronizer for CME futures (via IB).

    Args:
        position_provider: IB futures adapter
        account_provider: IB futures adapter
        local_state_getter: Function returning local positions
        futures_type: Type of CME futures
        **kwargs: Additional config options

    Returns:
        Configured FuturesPositionSynchronizer
    """
    config = FuturesSyncConfig(
        exchange=Exchange.CME,
        futures_type=futures_type,
        sync_interval_sec=kwargs.get("sync_interval_sec", 30.0),  # Less frequent for CME
        enable_adl_monitoring=False,  # No ADL for CME
        enable_liquidation_detection=True,
        enable_funding_tracking=False,  # No funding for CME
    )

    return FuturesPositionSynchronizer(
        position_provider=position_provider,
        account_provider=account_provider,
        local_state_getter=local_state_getter,
        config=config,
        **{k: v for k, v in kwargs.items() if k != "sync_interval_sec"}
    )
