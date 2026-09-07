# -*- coding: utf-8 -*-
"""
services/futures_live_runner.py
Main live trading runner for futures contracts.

Provides unified live trading for:
- Crypto perpetual futures (Binance USDT-M)
- CME futures (ES, NQ, GC, CL via Interactive Brokers)

Core features:
- Position synchronization with exchange
- Real-time funding rate tracking (crypto)
- Margin monitoring and alerts
- Auto-deleveraging detection
- Circuit breaker handling (CME)
- Settlement window handling (CME)

Architecture:
    FuturesLiveRunner
    ├── FuturesPositionSynchronizer (position sync)
    ├── FundingTrackerService (funding rates)
    ├── FuturesMarginMonitor (margin monitoring)
    ├── UnifiedFuturesRiskGuard (risk checks)
    └── FuturesOrderExecutionAdapter (order execution)

References:
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- IB TWS API: https://interactivebrokers.github.io/tws-api/
- CME Rule 80B: Circuit Breakers

Author: Trading Bot Team
Date: 2025-12-02
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
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

# Internal imports
from core_futures import (
    FuturesType,
    FuturesPosition,
    FuturesContractSpec,
    MarginMode,
    PositionSide,
    Exchange,
    OrderSide,
    OrderType,
    FundingRateInfo,
    FundingPayment,
)
from services.event_bus import EventBus
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
    FuturesSyncResult,
    FuturesSyncEventType,
    ADLRiskLevel,
    FuturesPositionDiff,
    create_crypto_futures_sync,
    create_cme_futures_sync,
)
from services.futures_funding_tracker import (
    FundingTrackerService,
    FundingTrackerConfig,
    create_funding_service,
)
from services.futures_risk_guards import (
    FuturesLeverageGuard,
    FuturesMarginGuard,
    FundingExposureGuard,
    ConcentrationGuard,
    ADLRiskGuard,
    MarginCallNotifier,
    MarginCheckResult,
    MarginStatus,
)
from services.unified_futures_risk import (
    UnifiedFuturesRiskGuard,
    UnifiedRiskConfig,
    UnifiedRiskEvent,
    create_unified_risk_guard,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Live trading loop intervals (seconds)
DEFAULT_MAIN_LOOP_INTERVAL_SEC = 1.0
DEFAULT_POSITION_SYNC_INTERVAL_SEC = 5.0
DEFAULT_MARGIN_CHECK_INTERVAL_SEC = 10.0
DEFAULT_FUNDING_CHECK_INTERVAL_SEC = 60.0
DEFAULT_HEARTBEAT_INTERVAL_SEC = 30.0

# Reconnection parameters
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_BACKOFF_BASE = 1.0
RECONNECT_BACKOFF_MAX = 60.0

# Order execution timeouts
ORDER_FILL_TIMEOUT_SEC = 30.0
ORDER_CANCEL_TIMEOUT_SEC = 10.0

# Health check thresholds
MAX_POSITION_SYNC_AGE_SEC = 30.0
MAX_MARKET_DATA_AGE_SEC = 10.0


# ============================================================================
# ENUMS
# ============================================================================


class LiveRunnerState(Enum):
    """Live runner states."""

    INITIALIZING = auto()
    CONNECTING = auto()
    SYNCING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class LiveRunnerEvent(Enum):
    """Events emitted by live runner."""

    STATE_CHANGED = auto()
    POSITION_SYNCED = auto()
    MARGIN_WARNING = auto()
    MARGIN_CRITICAL = auto()
    FUNDING_PAYMENT = auto()
    ADL_WARNING = auto()
    CIRCUIT_BREAKER = auto()
    ORDER_FILLED = auto()
    ORDER_REJECTED = auto()
    CONNECTION_LOST = auto()
    CONNECTION_RESTORED = auto()
    ERROR = auto()
    HEALTH_CHECK_FAILED = auto()


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers."""

    def get_mark_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mark price."""
        ...

    def get_index_price(self, symbol: str) -> Optional[Decimal]:
        """Get current index price."""
        ...

    def get_funding_rate(self, symbol: str) -> Optional[FundingRateInfo]:
        """Get current funding rate info."""
        ...

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[Dict]:
        """Get order book snapshot."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to data source."""
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Protocol for order execution."""

    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        reduce_only: bool = False,
    ) -> str:
        """Submit market order, return order ID."""
        ...

    def submit_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        price: Decimal,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> str:
        """Submit limit order, return order ID."""
        ...

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order, return success."""
        ...

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders, return count cancelled."""
        ...

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get order status."""
        ...


@runtime_checkable
class SignalProvider(Protocol):
    """Protocol for trading signal providers."""

    def get_signal(self, symbol: str, market_data: Dict) -> Dict:
        """Get trading signal for symbol."""
        ...


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class FuturesLiveRunnerConfig:
    """
    Configuration for FuturesLiveRunner.

    Attributes:
        futures_type: Type of futures (crypto/CME)
        symbols: Symbols to trade
        main_loop_interval_sec: Main loop interval
        position_sync_interval_sec: Position sync interval
        margin_check_interval_sec: Margin check interval
        funding_check_interval_sec: Funding rate check interval
        enable_position_sync: Enable background position sync
        enable_margin_monitoring: Enable margin monitoring
        enable_funding_tracking: Enable funding rate tracking
        enable_adl_monitoring: Enable ADL monitoring (crypto)
        enable_circuit_breaker_monitoring: Enable CB monitoring (CME)
        max_reconnect_attempts: Max reconnection attempts
        strict_mode: Enable strict risk checks
        paper_trading: Enable paper/simulation mode
        event_callbacks: Callbacks for events
    """

    futures_type: FuturesType = FuturesType.CRYPTO_PERPETUAL
    symbols: List[str] = field(default_factory=list)

    # Intervals
    main_loop_interval_sec: float = DEFAULT_MAIN_LOOP_INTERVAL_SEC
    position_sync_interval_sec: float = DEFAULT_POSITION_SYNC_INTERVAL_SEC
    margin_check_interval_sec: float = DEFAULT_MARGIN_CHECK_INTERVAL_SEC
    funding_check_interval_sec: float = DEFAULT_FUNDING_CHECK_INTERVAL_SEC
    heartbeat_interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC

    # Feature flags
    enable_position_sync: bool = True
    enable_margin_monitoring: bool = True
    enable_funding_tracking: bool = True
    enable_adl_monitoring: bool = True
    enable_circuit_breaker_monitoring: bool = True

    # Connection
    max_reconnect_attempts: int = MAX_RECONNECT_ATTEMPTS

    # Risk
    strict_mode: bool = True
    max_position_value: Optional[Decimal] = None
    max_leverage: int = 10

    # Mode
    paper_trading: bool = True

    # Event callbacks
    event_callbacks: Dict[LiveRunnerEvent, List[Callable]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesLiveRunnerConfig":
        """Create from dictionary."""
        futures_type_str = d.get("futures_type", "CRYPTO_PERPETUAL")
        if isinstance(futures_type_str, str):
            futures_type = FuturesType[futures_type_str.upper()]
        else:
            futures_type = futures_type_str

        return cls(
            futures_type=futures_type,
            symbols=list(d.get("symbols", [])),
            main_loop_interval_sec=float(
                d.get("main_loop_interval_sec", DEFAULT_MAIN_LOOP_INTERVAL_SEC)
            ),
            position_sync_interval_sec=float(
                d.get("position_sync_interval_sec", DEFAULT_POSITION_SYNC_INTERVAL_SEC)
            ),
            margin_check_interval_sec=float(
                d.get("margin_check_interval_sec", DEFAULT_MARGIN_CHECK_INTERVAL_SEC)
            ),
            funding_check_interval_sec=float(
                d.get("funding_check_interval_sec", DEFAULT_FUNDING_CHECK_INTERVAL_SEC)
            ),
            heartbeat_interval_sec=float(
                d.get("heartbeat_interval_sec", DEFAULT_HEARTBEAT_INTERVAL_SEC)
            ),
            enable_position_sync=bool(d.get("enable_position_sync", True)),
            enable_margin_monitoring=bool(d.get("enable_margin_monitoring", True)),
            enable_funding_tracking=bool(d.get("enable_funding_tracking", True)),
            enable_adl_monitoring=bool(d.get("enable_adl_monitoring", True)),
            enable_circuit_breaker_monitoring=bool(
                d.get("enable_circuit_breaker_monitoring", True)
            ),
            max_reconnect_attempts=int(d.get("max_reconnect_attempts", MAX_RECONNECT_ATTEMPTS)),
            strict_mode=bool(d.get("strict_mode", True)),
            max_position_value=(
                Decimal(str(d["max_position_value"])) if d.get("max_position_value") else None
            ),
            max_leverage=int(d.get("max_leverage", 10)),
            paper_trading=bool(d.get("paper_trading", True)),
        )

    @classmethod
    def for_crypto_perpetual(
        cls,
        symbols: List[str],
        paper_trading: bool = True,
        **kwargs,
    ) -> "FuturesLiveRunnerConfig":
        """Create config for crypto perpetual futures."""
        return cls(
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            symbols=symbols,
            paper_trading=paper_trading,
            enable_adl_monitoring=True,
            enable_circuit_breaker_monitoring=False,
            **kwargs,
        )

    @classmethod
    def for_cme_futures(
        cls,
        symbols: List[str],
        paper_trading: bool = True,
        **kwargs,
    ) -> "FuturesLiveRunnerConfig":
        """Create config for CME futures."""
        return cls(
            futures_type=FuturesType.INDEX_FUTURES,
            symbols=symbols,
            paper_trading=paper_trading,
            enable_adl_monitoring=False,
            enable_circuit_breaker_monitoring=True,
            enable_funding_tracking=False,  # CME doesn't have funding
            **kwargs,
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================


@dataclass
class HealthStatus:
    """Health status of the live runner."""

    is_healthy: bool
    state: LiveRunnerState
    last_position_sync_age_sec: float
    last_market_data_age_sec: float
    last_heartbeat_age_sec: float
    connection_status: Dict[str, bool]
    active_orders: int
    open_positions: int
    margin_status: MarginStatus
    warnings: List[str]
    errors: List[str]
    timestamp_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_healthy": self.is_healthy,
            "state": self.state.name,
            "last_position_sync_age_sec": self.last_position_sync_age_sec,
            "last_market_data_age_sec": self.last_market_data_age_sec,
            "last_heartbeat_age_sec": self.last_heartbeat_age_sec,
            "connection_status": self.connection_status,
            "active_orders": self.active_orders,
            "open_positions": self.open_positions,
            "margin_status": self.margin_status.name,
            "warnings": self.warnings,
            "errors": self.errors,
            "timestamp_ms": self.timestamp_ms,
        }


# ============================================================================
# LIVE RUNNER STATS
# ============================================================================


@dataclass
class LiveRunnerStats:
    """Statistics for live runner session."""

    session_start_time_ms: int = 0
    total_orders_submitted: int = 0
    total_orders_filled: int = 0
    total_orders_rejected: int = 0
    total_orders_cancelled: int = 0
    total_funding_payments: int = 0
    total_funding_paid: Decimal = Decimal("0")
    total_funding_received: Decimal = Decimal("0")
    total_position_syncs: int = 0
    total_position_discrepancies: int = 0
    total_margin_warnings: int = 0
    total_adl_warnings: int = 0
    total_circuit_breaker_events: int = 0
    total_reconnections: int = 0
    last_sync_result: Optional[FuturesSyncResult] = None

    def reset(self) -> None:
        """Reset all statistics."""
        self.session_start_time_ms = int(time.time() * 1000)
        self.total_orders_submitted = 0
        self.total_orders_filled = 0
        self.total_orders_rejected = 0
        self.total_orders_cancelled = 0
        self.total_funding_payments = 0
        self.total_funding_paid = Decimal("0")
        self.total_funding_received = Decimal("0")
        self.total_position_syncs = 0
        self.total_position_discrepancies = 0
        self.total_margin_warnings = 0
        self.total_adl_warnings = 0
        self.total_circuit_breaker_events = 0
        self.total_reconnections = 0
        self.last_sync_result = None


# ============================================================================
# MAIN LIVE RUNNER CLASS
# ============================================================================


class FuturesLiveRunner:
    """
    Main live trading runner for futures contracts.

    Coordinates:
    - Position synchronization with exchange
    - Real-time funding rate tracking (crypto)
    - Margin monitoring and alerts
    - ADL detection (crypto)
    - Circuit breaker handling (CME)
    - Signal processing and order execution

    Thread Safety:
    - Uses threading.Lock for state management
    - Async-compatible with asyncio support

    Example:
        >>> # Create runner for crypto perpetuals
        >>> config = FuturesLiveRunnerConfig.for_crypto_perpetual(
        ...     symbols=["BTCUSDT", "ETHUSDT"],
        ...     paper_trading=True,
        ... )
        >>> runner = FuturesLiveRunner(
        ...     config=config,
        ...     market_data=binance_md_adapter,
        ...     order_executor=binance_exec_adapter,
        ...     signal_provider=ml_model,
        ... )
        >>>
        >>> # Start and run
        >>> runner.start()
        >>> try:
        ...     runner.run()
        >>> finally:
        ...     runner.stop()
    """

    def __init__(
        self,
        config: FuturesLiveRunnerConfig,
        market_data: MarketDataProvider,
        order_executor: OrderExecutor,
        signal_provider: Optional[SignalProvider] = None,
        position_sync: Optional[FuturesPositionSynchronizer] = None,
        funding_tracker: Optional[FundingTrackerService] = None,
        risk_guard: Optional[UnifiedFuturesRiskGuard] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize futures live runner.

        Args:
            config: Runner configuration
            market_data: Market data provider
            order_executor: Order execution adapter
            signal_provider: Trading signal provider (optional)
            position_sync: Position synchronizer (created if None)
            funding_tracker: Funding tracker service (created if None)
            risk_guard: Risk guard (created if None)
            event_bus: Event bus for notifications
        """
        self._config = config
        self._market_data = market_data
        self._order_executor = order_executor
        self._signal_provider = signal_provider
        self._event_bus = event_bus

        # State management
        self._state = LiveRunnerState.INITIALIZING
        self._state_lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()

        # Local position state
        self._positions: Dict[str, FuturesPosition] = {}
        self._positions_lock = threading.Lock()
        self._active_orders: Dict[str, Dict] = {}
        self._orders_lock = threading.Lock()

        # Timing
        self._last_position_sync_time = 0.0
        self._last_margin_check_time = 0.0
        self._last_funding_check_time = 0.0
        self._last_heartbeat_time = 0.0
        self._last_market_data_time = 0.0

        # Statistics
        self._stats = LiveRunnerStats()

        # Initialize components
        self._position_sync = position_sync
        self._funding_tracker = funding_tracker
        self._risk_guard = risk_guard or create_unified_risk_guard()

        # Margin monitoring
        self._margin_guard = FuturesMarginGuard()
        self._margin_notifier = MarginCallNotifier(cooldown_seconds=300)
        self._last_margin_result: Optional[MarginCheckResult] = None

        # ADL monitoring (crypto only)
        self._adl_guard = ADLRiskGuard() if config.enable_adl_monitoring else None
        self._last_adl_level = ADLRiskLevel.SAFE

        # Reconnection state
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0

        logger.info(
            f"FuturesLiveRunner initialized: type={config.futures_type.name}, "
            f"symbols={config.symbols}, paper={config.paper_trading}"
        )

    # ------------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------------

    @property
    def state(self) -> LiveRunnerState:
        """Get current state."""
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: LiveRunnerState) -> None:
        """Set state with logging and event emission."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        if old_state != new_state:
            logger.info(f"State transition: {old_state.name} -> {new_state.name}")
            self._emit_event(
                LiveRunnerEvent.STATE_CHANGED,
                {
                    "old_state": old_state.name,
                    "new_state": new_state.name,
                    "timestamp_ms": int(time.time() * 1000),
                },
            )

    @property
    def is_running(self) -> bool:
        """Check if runner is active."""
        return self._running and self.state == LiveRunnerState.RUNNING

    # ------------------------------------------------------------------------
    # POSITION ACCESS
    # ------------------------------------------------------------------------

    def get_positions(self) -> Dict[str, FuturesPosition]:
        """Get current positions (copy)."""
        with self._positions_lock:
            return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Get position for symbol."""
        with self._positions_lock:
            return self._positions.get(symbol)

    def _update_local_positions(self, positions: Dict[str, FuturesPosition]) -> None:
        """Update local position state."""
        with self._positions_lock:
            self._positions = positions

    # ------------------------------------------------------------------------
    # ORDER ACCESS
    # ------------------------------------------------------------------------

    def get_active_orders(self) -> Dict[str, Dict]:
        """Get active orders (copy)."""
        with self._orders_lock:
            return dict(self._active_orders)

    def _add_order(self, order_id: str, order_info: Dict) -> None:
        """Add order to tracking."""
        with self._orders_lock:
            self._active_orders[order_id] = order_info

    def _remove_order(self, order_id: str) -> Optional[Dict]:
        """Remove order from tracking."""
        with self._orders_lock:
            return self._active_orders.pop(order_id, None)

    # ------------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------------

    def start(self) -> None:
        """Start the live runner."""
        if self._running:
            logger.warning("Runner already started")
            return

        logger.info("Starting futures live runner...")
        self._running = True
        self._stop_event.clear()
        self._stats.reset()

        # Initialize state
        self._set_state(LiveRunnerState.CONNECTING)

        # Initialize components
        self._initialize_components()

        # Initial sync
        self._set_state(LiveRunnerState.SYNCING)
        self._perform_initial_sync()

        self._set_state(LiveRunnerState.RUNNING)
        logger.info("Futures live runner started successfully")

    def stop(self, timeout_sec: float = 10.0) -> None:
        """
        Stop the live runner gracefully.

        Args:
            timeout_sec: Timeout for graceful shutdown
        """
        if not self._running:
            logger.warning("Runner not running")
            return

        logger.info("Stopping futures live runner...")
        self._set_state(LiveRunnerState.STOPPING)
        self._stop_event.set()
        self._running = False

        # Cancel all orders if configured
        try:
            cancelled = self._order_executor.cancel_all_orders()
            logger.info(f"Cancelled {cancelled} orders on shutdown")
        except Exception as e:
            logger.error(f"Error cancelling orders on shutdown: {e}")

        # Stop position sync background loop
        if self._position_sync:
            self._position_sync.stop_background_sync()

        self._set_state(LiveRunnerState.STOPPED)
        logger.info("Futures live runner stopped")

    def pause(self) -> None:
        """Pause trading (position sync continues)."""
        if self.state == LiveRunnerState.RUNNING:
            self._set_state(LiveRunnerState.PAUSED)
            logger.info("Runner paused")

    def resume(self) -> None:
        """Resume trading after pause."""
        if self.state == LiveRunnerState.PAUSED:
            self._set_state(LiveRunnerState.RUNNING)
            logger.info("Runner resumed")

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------

    def _initialize_components(self) -> None:
        """Initialize sub-components."""
        # Create position synchronizer if not provided
        if self._position_sync is None:
            if self._config.futures_type in (
                FuturesType.CRYPTO_PERPETUAL,
                FuturesType.CRYPTO_QUARTERLY,
            ):
                self._position_sync = create_crypto_futures_sync(
                    position_provider=self._market_data,
                    account_provider=self._market_data,
                    local_state_getter=self.get_positions,
                )
            else:
                self._position_sync = create_cme_futures_sync(
                    position_provider=self._market_data,
                    account_provider=self._market_data,
                    local_state_getter=self.get_positions,
                )

        # Create funding tracker if not provided (crypto only)
        if self._funding_tracker is None and self._config.enable_funding_tracking:
            self._funding_tracker = create_funding_service()

        logger.debug("Components initialized")

    def _perform_initial_sync(self) -> None:
        """Perform initial position synchronization."""
        if not self._position_sync:
            return

        try:
            result = self._position_sync.sync_once()
            self._stats.total_position_syncs += 1
            self._stats.last_sync_result = result

            if result.success:
                # Update local positions from exchange
                exchange_positions = {}
                for symbol in self._config.symbols:
                    pos = result.exchange_state.get(symbol)
                    if pos:
                        exchange_positions[symbol] = pos

                self._update_local_positions(exchange_positions)
                self._last_position_sync_time = time.time()

                logger.info(
                    f"Initial sync complete: {len(exchange_positions)} positions, "
                    f"{len(result.discrepancies)} discrepancies"
                )

                if result.discrepancies:
                    self._stats.total_position_discrepancies += len(result.discrepancies)
                    for disc in result.discrepancies:
                        logger.warning(f"Position discrepancy: {disc}")
            else:
                logger.error(f"Initial sync failed: {result.error_message}")

        except Exception as e:
            logger.error(f"Initial sync error: {e}")

    # ------------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the main trading loop (blocking).

        Call start() before run().
        """
        if not self._running:
            raise RuntimeError("Runner not started. Call start() first.")

        logger.info("Entering main trading loop")

        while self._running and not self._stop_event.is_set():
            try:
                self._loop_iteration()
                time.sleep(self._config.main_loop_interval_sec)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._handle_error(e)

        logger.info("Exited main trading loop")

    async def run_async(self) -> None:
        """
        Run the main trading loop (async).

        Call start() before run_async().
        """
        if not self._running:
            raise RuntimeError("Runner not started. Call start() first.")

        logger.info("Entering async main trading loop")

        while self._running and not self._stop_event.is_set():
            try:
                await self._loop_iteration_async()
                await asyncio.sleep(self._config.main_loop_interval_sec)
            except asyncio.CancelledError:
                logger.info("Async loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in async main loop: {e}", exc_info=True)
                self._handle_error(e)

        logger.info("Exited async main trading loop")

    def _loop_iteration(self) -> None:
        """Single iteration of the main loop."""
        now = time.time()

        # Skip if paused
        if self.state == LiveRunnerState.PAUSED:
            return

        # Check connection
        if not self._market_data.is_connected():
            self._handle_connection_lost()
            return

        # Position sync
        if (
            self._config.enable_position_sync
            and now - self._last_position_sync_time >= self._config.position_sync_interval_sec
        ):
            self._sync_positions()
            self._last_position_sync_time = now

        # Margin check
        if (
            self._config.enable_margin_monitoring
            and now - self._last_margin_check_time >= self._config.margin_check_interval_sec
        ):
            self._check_margin()
            self._last_margin_check_time = now

        # Funding check (crypto only)
        if (
            self._config.enable_funding_tracking
            and self._funding_tracker
            and now - self._last_funding_check_time >= self._config.funding_check_interval_sec
        ):
            self._check_funding()
            self._last_funding_check_time = now

        # Process signals if provider available
        if self._signal_provider and self.state == LiveRunnerState.RUNNING:
            self._process_signals()

        # Heartbeat
        if now - self._last_heartbeat_time >= self._config.heartbeat_interval_sec:
            self._heartbeat()
            self._last_heartbeat_time = now

    async def _loop_iteration_async(self) -> None:
        """Single iteration of the async main loop."""
        # Same logic as sync version, but could use async calls
        self._loop_iteration()

    # ------------------------------------------------------------------------
    # POSITION SYNC
    # ------------------------------------------------------------------------

    def _sync_positions(self) -> None:
        """Synchronize positions with exchange."""
        if not self._position_sync:
            return

        try:
            result = self._position_sync.sync_once()
            self._stats.total_position_syncs += 1
            self._stats.last_sync_result = result

            if result.success:
                # Check for discrepancies
                if result.discrepancies:
                    self._stats.total_position_discrepancies += len(result.discrepancies)
                    self._handle_position_discrepancies(result.discrepancies)

                # Check ADL warnings
                adl_warnings = [
                    d
                    for d in result.discrepancies
                    if d.event_type == FuturesSyncEventType.ADL_WARNING
                ]
                if adl_warnings:
                    self._handle_adl_warnings(adl_warnings)

                # Update local state
                exchange_positions = {}
                for symbol in self._config.symbols:
                    pos = result.exchange_state.get(symbol)
                    if pos:
                        exchange_positions[symbol] = pos
                self._update_local_positions(exchange_positions)

                self._emit_event(
                    LiveRunnerEvent.POSITION_SYNCED,
                    {
                        "positions": len(exchange_positions),
                        "discrepancies": len(result.discrepancies),
                        "timestamp_ms": result.timestamp_ms,
                    },
                )
            else:
                logger.warning(f"Position sync failed: {result.error_message}")

        except Exception as e:
            logger.error(f"Position sync error: {e}")

    def _handle_position_discrepancies(
        self,
        discrepancies: List[FuturesPositionDiff],
    ) -> None:
        """Handle detected position discrepancies."""
        for disc in discrepancies:
            if disc.event_type == FuturesSyncEventType.POSITION_MISMATCH:
                logger.warning(
                    f"Position mismatch for {disc.symbol}: "
                    f"local={disc.local_qty}, exchange={disc.exchange_qty}"
                )
            elif disc.event_type == FuturesSyncEventType.UNEXPECTED_POSITION:
                logger.warning(
                    f"Unexpected position on exchange: {disc.symbol}, " f"qty={disc.exchange_qty}"
                )
            elif disc.event_type == FuturesSyncEventType.MISSING_POSITION:
                logger.warning(
                    f"Position missing from exchange: {disc.symbol}, " f"local_qty={disc.local_qty}"
                )

    def _handle_adl_warnings(self, warnings: List[FuturesPositionDiff]) -> None:
        """Handle ADL warning events."""
        for warning in warnings:
            self._stats.total_adl_warnings += 1

            logger.warning(
                f"ADL warning for {warning.symbol}: "
                f"level={warning.adl_risk_level.name if warning.adl_risk_level else 'N/A'}"
            )

            self._emit_event(
                LiveRunnerEvent.ADL_WARNING,
                {
                    "symbol": warning.symbol,
                    "adl_level": warning.adl_risk_level.name if warning.adl_risk_level else None,
                    "timestamp_ms": warning.timestamp_ms,
                },
            )

    # ------------------------------------------------------------------------
    # MARGIN MONITORING
    # ------------------------------------------------------------------------

    def _check_margin(self) -> None:
        """Check margin status."""
        positions = self.get_positions()
        if not positions:
            return

        try:
            # Get account info for margin ratio
            # This would come from the account provider
            # For now, use a simplified check
            total_margin_used = Decimal("0")
            total_position_value = Decimal("0")

            for pos in positions.values():
                if pos.margin:
                    total_margin_used += pos.margin
                if pos.position_value:
                    total_position_value += abs(pos.position_value)

            # Check margin guard
            if total_margin_used > 0:
                # Simplified margin ratio check
                # In production, get actual account equity from adapter
                margin_ratio = Decimal("1.5")  # Placeholder

                result = self._margin_guard.check_margin_ratio(
                    margin_ratio=float(margin_ratio),
                    account_equity=float(total_position_value),
                    total_margin_used=float(total_margin_used),
                )

                self._last_margin_result = result

                if result.status in (MarginStatus.WARNING, MarginStatus.DANGER):
                    self._stats.total_margin_warnings += 1

                    event = (
                        LiveRunnerEvent.MARGIN_WARNING
                        if result.status == MarginStatus.WARNING
                        else LiveRunnerEvent.MARGIN_CRITICAL
                    )

                    self._emit_event(
                        event,
                        {
                            "status": result.status.name,
                            "margin_ratio": result.margin_ratio,
                            "timestamp_ms": int(time.time() * 1000),
                        },
                    )

                    # Notify via margin notifier
                    self._margin_notifier.check_and_notify(result)

        except Exception as e:
            logger.error(f"Margin check error: {e}")

    # ------------------------------------------------------------------------
    # FUNDING TRACKING
    # ------------------------------------------------------------------------

    def _check_funding(self) -> None:
        """Check funding rates and process payments."""
        if not self._funding_tracker:
            return

        positions = self.get_positions()
        if not positions:
            return

        try:
            now_ms = int(time.time() * 1000)

            for symbol, position in positions.items():
                if position.qty == 0:
                    continue

                # Get current funding rate from market data
                funding_info = self._market_data.get_funding_rate(symbol)
                if not funding_info:
                    continue

                # Check if we're at funding time
                if self._funding_tracker.is_funding_time(now_ms, tolerance_ms=60_000):
                    # Calculate funding payment
                    mark_price = self._market_data.get_mark_price(symbol)
                    if mark_price:
                        payment = self._funding_tracker.calculate_funding_payment(
                            position=position,
                            funding_rate=funding_info.rate,
                            mark_price=mark_price,
                            timestamp_ms=now_ms,
                        )

                        self._stats.total_funding_payments += 1
                        if payment.payment_amount > 0:
                            self._stats.total_funding_received += payment.payment_amount
                        else:
                            self._stats.total_funding_paid += abs(payment.payment_amount)

                        logger.info(f"Funding payment for {symbol}: {payment.payment_amount}")

                        self._emit_event(
                            LiveRunnerEvent.FUNDING_PAYMENT,
                            {
                                "symbol": symbol,
                                "payment": float(payment.payment_amount),
                                "rate": float(funding_info.rate),
                                "timestamp_ms": now_ms,
                            },
                        )

        except Exception as e:
            logger.error(f"Funding check error: {e}")

    # ------------------------------------------------------------------------
    # SIGNAL PROCESSING
    # ------------------------------------------------------------------------

    def _process_signals(self) -> None:
        """Process trading signals and execute orders."""
        if not self._signal_provider:
            return

        for symbol in self._config.symbols:
            try:
                # Get market data
                mark_price = self._market_data.get_mark_price(symbol)
                if not mark_price:
                    continue

                market_data = {
                    "symbol": symbol,
                    "mark_price": float(mark_price),
                    "timestamp_ms": int(time.time() * 1000),
                }

                # Get order book if available
                order_book = self._market_data.get_order_book(symbol)
                if order_book:
                    market_data["order_book"] = order_book

                # Get signal
                signal = self._signal_provider.get_signal(symbol, market_data)

                if signal and signal.get("action"):
                    self._execute_signal(symbol, signal)

            except Exception as e:
                logger.error(f"Signal processing error for {symbol}: {e}")

    def _execute_signal(self, symbol: str, signal: Dict) -> None:
        """Execute trading signal."""
        action = signal.get("action")
        qty = signal.get("qty")

        if not action or not qty:
            return

        # Check risk before execution
        position = self.get_position(symbol)
        mark_price = self._market_data.get_mark_price(symbol)

        if mark_price:
            risk_event = self._risk_guard.check_trade(
                symbol=symbol,
                side=action,
                quantity=float(qty),
            )

            if risk_event != UnifiedRiskEvent.NONE:
                logger.warning(
                    f"Trade blocked by risk guard: {symbol} {action} {qty}, "
                    f"event={risk_event.name}"
                )
                return

        # Execute order
        try:
            side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
            reduce_only = signal.get("reduce_only", False)

            if signal.get("order_type", "MARKET").upper() == "LIMIT":
                price = signal.get("price")
                if price:
                    order_id = self._order_executor.submit_limit_order(
                        symbol=symbol,
                        side=side,
                        qty=Decimal(str(qty)),
                        price=Decimal(str(price)),
                        reduce_only=reduce_only,
                    )
                else:
                    logger.warning(f"Limit order missing price: {signal}")
                    return
            else:
                order_id = self._order_executor.submit_market_order(
                    symbol=symbol,
                    side=side,
                    qty=Decimal(str(qty)),
                    reduce_only=reduce_only,
                )

            self._stats.total_orders_submitted += 1
            self._add_order(
                order_id,
                {
                    "symbol": symbol,
                    "side": side.name,
                    "qty": qty,
                    "time": time.time(),
                },
            )

            logger.info(f"Order submitted: {order_id} {symbol} {side.name} {qty}")

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            self._stats.total_orders_rejected += 1
            self._emit_event(
                LiveRunnerEvent.ORDER_REJECTED,
                {
                    "symbol": symbol,
                    "action": action,
                    "qty": qty,
                    "error": str(e),
                },
            )

    # ------------------------------------------------------------------------
    # CONNECTION HANDLING
    # ------------------------------------------------------------------------

    def _handle_connection_lost(self) -> None:
        """Handle lost connection to exchange."""
        logger.warning("Connection lost to exchange")

        self._emit_event(
            LiveRunnerEvent.CONNECTION_LOST,
            {
                "timestamp_ms": int(time.time() * 1000),
            },
        )

        # Attempt reconnection
        self._attempt_reconnection()

    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to exchange."""
        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._set_state(LiveRunnerState.ERROR)
            return

        # Exponential backoff
        backoff = min(
            RECONNECT_BACKOFF_BASE * (2**self._reconnect_attempts),
            RECONNECT_BACKOFF_MAX,
        )

        logger.info(f"Reconnection attempt {self._reconnect_attempts + 1} in {backoff}s")
        time.sleep(backoff)

        self._reconnect_attempts += 1
        self._stats.total_reconnections += 1

        # Check if connected
        if self._market_data.is_connected():
            logger.info("Reconnection successful")
            self._reconnect_attempts = 0
            self._emit_event(
                LiveRunnerEvent.CONNECTION_RESTORED,
                {
                    "timestamp_ms": int(time.time() * 1000),
                },
            )

    # ------------------------------------------------------------------------
    # ERROR HANDLING
    # ------------------------------------------------------------------------

    def _handle_error(self, error: Exception) -> None:
        """Handle runtime error."""
        self._emit_event(
            LiveRunnerEvent.ERROR,
            {
                "error": str(error),
                "timestamp_ms": int(time.time() * 1000),
            },
        )

        # Decide if we should stop or continue
        if isinstance(error, (ConnectionError, TimeoutError)):
            self._handle_connection_lost()
        else:
            # Log and continue for other errors
            logger.error(f"Runtime error: {error}")

    # ------------------------------------------------------------------------
    # HEARTBEAT & HEALTH
    # ------------------------------------------------------------------------

    def _heartbeat(self) -> None:
        """Perform heartbeat check."""
        logger.debug("Heartbeat")
        self._last_heartbeat_time = time.time()

    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        now = time.time()
        warnings = []
        errors = []

        # Check sync age
        sync_age = now - self._last_position_sync_time
        if sync_age > MAX_POSITION_SYNC_AGE_SEC:
            warnings.append(f"Position sync stale: {sync_age:.1f}s")

        # Check market data age
        md_age = now - self._last_market_data_time if self._last_market_data_time else float("inf")
        if md_age > MAX_MARKET_DATA_AGE_SEC:
            warnings.append(f"Market data stale: {md_age:.1f}s")

        # Check heartbeat
        hb_age = now - self._last_heartbeat_time

        # Check connection
        connected = self._market_data.is_connected()
        if not connected:
            errors.append("Not connected to exchange")

        # Determine health
        is_healthy = (
            len(errors) == 0
            and self.state in (LiveRunnerState.RUNNING, LiveRunnerState.PAUSED)
            and connected
        )

        return HealthStatus(
            is_healthy=is_healthy,
            state=self.state,
            last_position_sync_age_sec=sync_age,
            last_market_data_age_sec=md_age,
            last_heartbeat_age_sec=hb_age,
            connection_status={"exchange": connected},
            active_orders=len(self._active_orders),
            open_positions=len(self._positions),
            margin_status=(
                self._last_margin_result.status
                if self._last_margin_result
                else MarginStatus.HEALTHY
            ),
            warnings=warnings,
            errors=errors,
            timestamp_ms=int(now * 1000),
        )

    def perform_health_check(self) -> bool:
        """Perform health check and emit event if unhealthy."""
        status = self.get_health_status()

        if not status.is_healthy:
            self._emit_event(LiveRunnerEvent.HEALTH_CHECK_FAILED, status.to_dict())

        return status.is_healthy

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    def get_stats(self) -> LiveRunnerStats:
        """Get session statistics."""
        return self._stats

    # ------------------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------------------

    def _emit_event(self, event: LiveRunnerEvent, data: Dict) -> None:
        """Emit event to registered callbacks and event bus."""
        # Call registered callbacks
        callbacks = self._config.event_callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

        # Emit to event bus
        if self._event_bus:
            try:
                self._event_bus.emit(f"futures_live.{event.name.lower()}", data)
            except Exception as e:
                logger.error(f"Error emitting to event bus: {e}")

    def register_callback(
        self,
        event: LiveRunnerEvent,
        callback: Callable[[LiveRunnerEvent, Dict], None],
    ) -> None:
        """Register callback for event."""
        if event not in self._config.event_callbacks:
            self._config.event_callbacks[event] = []
        self._config.event_callbacks[event].append(callback)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_futures_live_runner(
    config: FuturesLiveRunnerConfig,
    market_data: MarketDataProvider,
    order_executor: OrderExecutor,
    signal_provider: Optional[SignalProvider] = None,
    **kwargs,
) -> FuturesLiveRunner:
    """
    Create futures live runner with provided components.

    Args:
        config: Runner configuration
        market_data: Market data provider
        order_executor: Order execution adapter
        signal_provider: Trading signal provider
        **kwargs: Additional arguments for FuturesLiveRunner

    Returns:
        FuturesLiveRunner instance
    """
    return FuturesLiveRunner(
        config=config,
        market_data=market_data,
        order_executor=order_executor,
        signal_provider=signal_provider,
        **kwargs,
    )


def create_crypto_futures_runner(
    symbols: List[str],
    market_data: MarketDataProvider,
    order_executor: OrderExecutor,
    signal_provider: Optional[SignalProvider] = None,
    paper_trading: bool = True,
    **kwargs,
) -> FuturesLiveRunner:
    """
    Create live runner for crypto perpetual futures.

    Args:
        symbols: Symbols to trade
        market_data: Market data provider (Binance adapter)
        order_executor: Order execution adapter (Binance adapter)
        signal_provider: Trading signal provider
        paper_trading: Enable paper trading mode
        **kwargs: Additional config options

    Returns:
        FuturesLiveRunner instance
    """
    config = FuturesLiveRunnerConfig.for_crypto_perpetual(
        symbols=symbols,
        paper_trading=paper_trading,
        **kwargs,
    )

    return create_futures_live_runner(
        config=config,
        market_data=market_data,
        order_executor=order_executor,
        signal_provider=signal_provider,
    )


def create_cme_futures_runner(
    symbols: List[str],
    market_data: MarketDataProvider,
    order_executor: OrderExecutor,
    signal_provider: Optional[SignalProvider] = None,
    paper_trading: bool = True,
    **kwargs,
) -> FuturesLiveRunner:
    """
    Create live runner for CME futures.

    Args:
        symbols: Symbols to trade (ES, NQ, GC, etc.)
        market_data: Market data provider (IB adapter)
        order_executor: Order execution adapter (IB adapter)
        signal_provider: Trading signal provider
        paper_trading: Enable paper trading mode
        **kwargs: Additional config options

    Returns:
        FuturesLiveRunner instance
    """
    config = FuturesLiveRunnerConfig.for_cme_futures(
        symbols=symbols,
        paper_trading=paper_trading,
        **kwargs,
    )

    return create_futures_live_runner(
        config=config,
        market_data=market_data,
        order_executor=order_executor,
        signal_provider=signal_provider,
    )
