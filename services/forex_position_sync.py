# -*- coding: utf-8 -*-
"""
services/forex_position_sync.py
Forex Position Synchronization Service

Phase 6: Forex Risk Management & Services (2025-11-30)

Syncs local position state with OANDA account.
Similar to services/position_sync.py for Alpaca.

Features:
- Background polling (configurable interval)
- Discrepancy detection and alerting
- Automatic reconciliation (optional)
- Swap cost tracking integration
- Financing/rollover tracking

Key Differences from Equity:
- Positions measured in units (not shares)
- Average price includes accumulated swap
- Financing (swap) is tracked separately
- Margin is calculated differently

References:
- OANDA REST API: https://developer.oanda.com/rest-live-v20/position-ep/
- OANDA Account Endpoints: https://developer.oanda.com/rest-live-v20/account-ep/
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .forex_risk_guards import SwapCostTracker

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class PositionDiscrepancyType(str, Enum):
    """Types of position discrepancies."""

    MISSING_LOCAL = "missing_local"  # Position on broker but not locally
    MISSING_REMOTE = "missing_remote"  # Position locally but not on broker
    UNITS_MISMATCH = "units_mismatch"  # Units quantity differs
    PRICE_MISMATCH = "price_mismatch"  # Average price differs (informational)
    SIDE_MISMATCH = "side_mismatch"  # Long/short direction differs


@dataclass
class ForexPosition:
    """
    Forex position from OANDA.

    Attributes:
        symbol: Currency pair (e.g., "EUR_USD")
        units: Position size (positive = long, negative = short)
        average_price: Average entry price
        unrealized_pnl: Current unrealized P&L
        margin_used: Margin used for this position
        financing: Accumulated swap/financing cost
    """

    symbol: str
    units: float  # Positive = long, Negative = short
    average_price: float
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    financing: float = 0.0  # Accumulated swap

    @property
    def is_long(self) -> bool:
        """Whether position is long."""
        return self.units > 0

    @property
    def is_short(self) -> bool:
        """Whether position is short."""
        return self.units < 0

    @property
    def abs_units(self) -> float:
        """Absolute position size."""
        return abs(self.units)

    @property
    def notional(self) -> float:
        """Notional value of position."""
        return self.abs_units * self.average_price


@dataclass
class PositionDiscrepancy:
    """Represents a discrepancy between local and remote positions."""

    symbol: str
    discrepancy_type: PositionDiscrepancyType
    local_units: Optional[float]
    remote_units: Optional[float]
    local_price: Optional[float] = None
    remote_price: Optional[float] = None
    remote_financing: Optional[float] = None
    details: str = ""

    @property
    def units_diff(self) -> float:
        """Unit difference (remote - local)."""
        local = self.local_units or 0.0
        remote = self.remote_units or 0.0
        return remote - local


@dataclass
class SyncConfig:
    """Position sync configuration."""

    sync_interval_sec: float = 30.0  # Interval between syncs
    position_tolerance_pct: float = 0.01  # 1% tolerance for unit comparison
    price_tolerance_pct: float = 0.01  # 1% tolerance for price comparison
    min_unit_diff: float = 1.0  # Minimum unit difference to report
    auto_reconcile: bool = False  # Auto-reconcile discrepancies
    max_reconcile_units: float = 100_000  # Max units to auto-reconcile
    alert_on_discrepancy: bool = True  # Alert on discrepancy
    track_financing: bool = True  # Track swap/financing
    include_symbols: Optional[List[str]] = None  # Only sync these symbols
    exclude_symbols: Optional[List[str]] = None  # Exclude these symbols


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    timestamp: float
    success: bool
    local_positions: Dict[str, float] = field(default_factory=dict)
    remote_positions: Dict[str, ForexPosition] = field(default_factory=dict)
    discrepancies: List[PositionDiscrepancy] = field(default_factory=list)
    total_financing: float = 0.0
    total_margin_used: float = 0.0  # Total margin across all positions
    long_exposure: float = 0.0  # Total long notional exposure
    short_exposure: float = 0.0  # Total short notional exposure
    reconciliation_results: List["ReconciliationOrderResult"] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_discrepancies(self) -> bool:
        """Whether any discrepancies were found."""
        return len(self.discrepancies) > 0

    @property
    def position_count(self) -> int:
        """Number of remote positions."""
        return len(self.remote_positions)

    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return self.long_exposure - self.short_exposure

    @property
    def reconciliation_count(self) -> int:
        """Number of reconciliation orders executed."""
        return len(self.reconciliation_results)

    @property
    def reconciliation_success_count(self) -> int:
        """Number of successful reconciliation orders."""
        return sum(1 for r in self.reconciliation_results if r.success)


# =============================================================================
# Protocols
# =============================================================================


class ForexPositionProvider(Protocol):
    """Protocol for forex position data providers."""

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, ForexPosition]:
        """Get current positions from broker."""
        ...


class ForexPositionProviderAsync(Protocol):
    """Async protocol for forex position data providers."""

    async def get_positions_async(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, ForexPosition]:
        """Get current positions from broker asynchronously."""
        ...


class ReconcileCallback(Protocol):
    """Protocol for reconciliation callback."""

    def __call__(
        self,
        symbol: str,
        remote_units: float,
        local_units: float,
    ) -> bool:
        """
        Handle reconciliation.

        Returns True if reconciliation was successful.
        """
        ...


class ForexOrderExecutor(Protocol):
    """
    Protocol for forex order execution.

    Used by auto-reconciliation to place corrective orders.

    References:
        - OANDA Order Endpoints: https://developer.oanda.com/rest-live-v20/order-ep/
        - Best Execution: MiFID II Article 27
    """

    def place_market_order(
        self,
        symbol: str,
        units: float,
    ) -> "ReconciliationOrderResult":
        """
        Place a market order for reconciliation.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            units: Number of units (positive=buy, negative=sell)

        Returns:
            ReconciliationOrderResult with execution details
        """
        ...


@dataclass
class ReconciliationOrderResult:
    """
    Result of a reconciliation order.

    Attributes:
        success: Whether order was executed successfully
        order_id: Broker order ID (if successful)
        symbol: Currency pair
        requested_units: Units requested
        filled_units: Units actually filled
        fill_price: Average fill price
        error: Error message (if failed)
        timestamp: Execution timestamp
    """

    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    requested_units: float = 0.0
    filled_units: float = 0.0
    fill_price: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReconciliationAction:
    """
    A reconciliation action to be taken.

    Attributes:
        symbol: Currency pair
        discrepancy: The discrepancy being reconciled
        units_to_trade: Units to trade (positive=buy, negative=sell)
        is_safe: Whether this reconciliation is safe (within limits)
        reason: Reason for action or why it's blocked
    """

    symbol: str
    discrepancy: "PositionDiscrepancy"
    units_to_trade: float
    is_safe: bool = True
    reason: str = ""


class ReconciliationExecutor:
    """
    Executes automatic position reconciliation.

    Implements safe reconciliation with:
    - Maximum units per reconciliation
    - Rate limiting (max reconciliations per period)
    - Side verification (prevent side flips)
    - Audit logging

    References:
        - CFTC Regulation 1.73: Risk Management
        - NFA Compliance Rule 2-36: Supervision
    """

    def __init__(
        self,
        order_executor: ForexOrderExecutor,
        max_units_per_order: float = 100_000.0,
        max_orders_per_hour: int = 10,
        prevent_side_flip: bool = True,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize reconciliation executor.

        Args:
            order_executor: Executor for placing orders
            max_units_per_order: Maximum units per reconciliation order
            max_orders_per_hour: Rate limit (orders per hour)
            prevent_side_flip: Prevent reconciliation that would flip position side
            dry_run: If True, log but don't execute orders
        """
        self._executor = order_executor
        self._max_units = max_units_per_order
        self._max_orders_per_hour = max_orders_per_hour
        self._prevent_side_flip = prevent_side_flip
        self._dry_run = dry_run

        # Rate limiting
        self._order_timestamps: List[float] = []
        self._lock = threading.Lock()

        # Audit log
        self._reconciliation_history: List[ReconciliationOrderResult] = []

    def can_reconcile(self, action: ReconciliationAction) -> Tuple[bool, str]:
        """
        Check if reconciliation can proceed.

        Args:
            action: Proposed reconciliation action

        Returns:
            (can_proceed, reason) tuple
        """
        # Check rate limit
        with self._lock:
            now = time.time()
            hour_ago = now - 3600
            self._order_timestamps = [ts for ts in self._order_timestamps if ts > hour_ago]

            if len(self._order_timestamps) >= self._max_orders_per_hour:
                return (False, f"Rate limit: {self._max_orders_per_hour} orders/hour exceeded")

        # Check units limit
        if abs(action.units_to_trade) > self._max_units:
            return (
                False,
                f"Units {abs(action.units_to_trade):.0f} exceeds max {self._max_units:.0f}",
            )

        # Check side flip
        if self._prevent_side_flip:
            disc = action.discrepancy
            if disc.local_units and disc.remote_units:
                local_side = disc.local_units > 0
                remote_side = disc.remote_units > 0
                if local_side != remote_side:
                    return (False, "Side flip detected - manual intervention required")

        return (True, "OK")

    def execute_reconciliation(
        self,
        action: ReconciliationAction,
    ) -> ReconciliationOrderResult:
        """
        Execute a reconciliation order.

        Args:
            action: Reconciliation action to execute

        Returns:
            ReconciliationOrderResult with execution details
        """
        can_proceed, reason = self.can_reconcile(action)

        if not can_proceed:
            logger.warning(f"Reconciliation blocked for {action.symbol}: {reason}")
            return ReconciliationOrderResult(
                success=False,
                symbol=action.symbol,
                requested_units=action.units_to_trade,
                error=reason,
            )

        if self._dry_run:
            logger.info(
                f"[DRY RUN] Would reconcile {action.symbol}: " f"{action.units_to_trade:+.0f} units"
            )
            return ReconciliationOrderResult(
                success=True,
                symbol=action.symbol,
                requested_units=action.units_to_trade,
                filled_units=action.units_to_trade,
                error="DRY_RUN",
            )

        try:
            # Execute order
            result = self._executor.place_market_order(
                symbol=action.symbol,
                units=action.units_to_trade,
            )

            # Update rate limiting
            with self._lock:
                self._order_timestamps.append(time.time())
                self._reconciliation_history.append(result)

            if result.success:
                logger.info(
                    f"Reconciliation order executed: {action.symbol} "
                    f"{result.filled_units:+.0f} units @ {result.fill_price}"
                )
            else:
                logger.error(f"Reconciliation order failed: {action.symbol} - {result.error}")

            return result

        except Exception as e:
            logger.error(f"Reconciliation execution error: {e}")
            return ReconciliationOrderResult(
                success=False,
                symbol=action.symbol,
                requested_units=action.units_to_trade,
                error=str(e),
            )

    def get_reconciliation_history(
        self,
        limit: int = 100,
    ) -> List[ReconciliationOrderResult]:
        """Get recent reconciliation history."""
        with self._lock:
            return list(self._reconciliation_history[-limit:])

    def clear_history(self) -> None:
        """Clear reconciliation history."""
        with self._lock:
            self._reconciliation_history.clear()


# =============================================================================
# Forex Position Synchronizer
# =============================================================================


class ForexPositionSynchronizer:
    """
    Synchronizes local state with OANDA positions.

    Provides:
    - One-time sync operations
    - Periodic background sync
    - Discrepancy detection and reporting
    - Swap cost tracking integration
    - Reconciliation callbacks

    Usage:
        sync = ForexPositionSynchronizer(
            position_provider=oanda_adapter,
            local_state_getter=lambda: my_state.positions,
            config=SyncConfig(sync_interval_sec=30),
        )

        # One-time sync
        result = sync.sync_once()
        if result.has_discrepancies:
            for d in result.discrepancies:
                print(f"Discrepancy: {d.symbol} {d.discrepancy_type}")

        # Start background sync
        sync.start_background_sync()

    References:
        - OANDA API: https://developer.oanda.com/rest-live-v20/
    """

    def __init__(
        self,
        position_provider: ForexPositionProvider,
        local_state_getter: Callable[[], Dict[str, float]],
        config: Optional[SyncConfig] = None,
        on_discrepancy: Optional[Callable[[PositionDiscrepancy], None]] = None,
        on_sync_complete: Optional[Callable[[SyncResult], None]] = None,
        swap_tracker: Optional[SwapCostTracker] = None,
        reconciliation_executor: Optional[ReconciliationExecutor] = None,
        on_reconciliation: Optional[Callable[[ReconciliationOrderResult], None]] = None,
    ) -> None:
        """
        Initialize forex position synchronizer.

        Args:
            position_provider: Adapter providing get_positions()
            local_state_getter: Callable returning local positions dict {symbol: units}
            config: Sync configuration
            on_discrepancy: Callback for each discrepancy found
            on_sync_complete: Callback after each sync completes
            swap_tracker: Optional swap cost tracker for financing tracking
            reconciliation_executor: Optional executor for auto-reconciliation
            on_reconciliation: Callback when reconciliation order is placed
        """
        self._provider = position_provider
        self._local_state_getter = local_state_getter
        self._config = config or SyncConfig()
        self._on_discrepancy = on_discrepancy
        self._on_sync_complete = on_sync_complete
        self._swap_tracker = swap_tracker
        self._reconciliation_executor = reconciliation_executor
        self._on_reconciliation = on_reconciliation

        # Background sync state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_sync: Optional[SyncResult] = None

        # Thread-based sync (for non-async usage)
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def config(self) -> SyncConfig:
        """Current configuration."""
        return self._config

    @property
    def last_sync(self) -> Optional[SyncResult]:
        """Result of last synchronization."""
        return self._last_sync

    @property
    def is_running(self) -> bool:
        """Whether background sync is running."""
        return self._running

    def sync_once(self) -> SyncResult:
        """
        Perform a single synchronization operation.

        Returns:
            SyncResult with comparison details
        """
        timestamp = time.time()

        try:
            # Get local positions
            local_positions = self._local_state_getter()
            logger.debug(f"Local forex positions: {len(local_positions)} symbols")

            # Get remote positions
            filter_symbols = self._config.include_symbols
            remote_raw = self._provider.get_positions(filter_symbols)

            # Convert to standard format
            remote_positions: Dict[str, ForexPosition] = {}

            for symbol, pos in remote_raw.items():
                # Skip excluded symbols
                if self._config.exclude_symbols and symbol in self._config.exclude_symbols:
                    continue

                # Handle different position formats
                if isinstance(pos, ForexPosition):
                    remote_positions[symbol] = pos
                elif isinstance(pos, dict):
                    remote_positions[symbol] = ForexPosition(
                        symbol=symbol,
                        units=float(pos.get("units", 0)),
                        average_price=float(pos.get("average_price", 0)),
                        unrealized_pnl=float(pos.get("unrealized_pnl", 0)),
                        margin_used=float(pos.get("margin_used", 0)),
                        financing=float(pos.get("financing", 0)),
                    )
                else:
                    # Try to extract from object attributes
                    remote_positions[symbol] = ForexPosition(
                        symbol=symbol,
                        units=float(getattr(pos, "units", 0)),
                        average_price=float(getattr(pos, "average_price", 0)),
                        unrealized_pnl=float(getattr(pos, "unrealized_pnl", 0)),
                        margin_used=float(getattr(pos, "margin_used", 0)),
                        financing=float(getattr(pos, "financing", 0)),
                    )

            logger.debug(f"Remote forex positions: {len(remote_positions)} symbols")

            # Calculate total financing
            total_financing = sum(p.financing for p in remote_positions.values())

            # Compare positions
            discrepancies = self._compare_positions(local_positions, remote_positions)

            # Calculate exposure metrics
            total_margin_used = 0.0
            long_exposure = 0.0
            short_exposure = 0.0
            for pos in remote_positions.values():
                total_margin_used += pos.margin_used
                if pos.is_long:
                    long_exposure += pos.notional
                else:
                    short_exposure += abs(pos.units) * pos.average_price

            # Create result
            result = SyncResult(
                timestamp=timestamp,
                success=True,
                local_positions=local_positions,
                remote_positions=remote_positions,
                discrepancies=discrepancies,
                total_financing=total_financing,
                total_margin_used=total_margin_used,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
            )

            # Invoke callbacks
            reconciliation_results: List[ReconciliationOrderResult] = []

            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} forex position discrepancies")
                for d in discrepancies:
                    logger.warning(f"  {d.symbol}: {d.discrepancy_type.value} - {d.details}")
                    if self._on_discrepancy:
                        try:
                            self._on_discrepancy(d)
                        except Exception as e:
                            logger.error(f"Discrepancy callback error: {e}")

                # Auto-reconciliation if enabled
                if self._config.auto_reconcile and self._reconciliation_executor is not None:
                    reconciliation_results = self._execute_auto_reconciliation(
                        discrepancies, remote_positions
                    )

            # Store reconciliation results in result
            result.reconciliation_results = reconciliation_results

            if self._on_sync_complete:
                try:
                    self._on_sync_complete(result)
                except Exception as e:
                    logger.error(f"Sync complete callback error: {e}")

            self._last_sync = result
            return result

        except Exception as e:
            logger.error(f"Forex position sync failed: {e}")
            result = SyncResult(
                timestamp=timestamp,
                success=False,
                error=str(e),
            )
            self._last_sync = result
            return result

    async def sync_once_async(self) -> SyncResult:
        """
        Perform a single async synchronization operation.

        Returns:
            SyncResult with comparison details
        """
        timestamp = time.time()

        try:
            # Get local positions
            local_positions = self._local_state_getter()

            # Get remote positions (async if supported)
            if hasattr(self._provider, "get_positions_async"):
                remote_raw = await self._provider.get_positions_async(self._config.include_symbols)
            else:
                # Fall back to sync
                remote_raw = self._provider.get_positions(self._config.include_symbols)

            # Convert to standard format
            remote_positions: Dict[str, ForexPosition] = {}

            for symbol, pos in remote_raw.items():
                if self._config.exclude_symbols and symbol in self._config.exclude_symbols:
                    continue

                if isinstance(pos, ForexPosition):
                    remote_positions[symbol] = pos
                elif isinstance(pos, dict):
                    remote_positions[symbol] = ForexPosition(
                        symbol=symbol,
                        units=float(pos.get("units", 0)),
                        average_price=float(pos.get("average_price", 0)),
                        unrealized_pnl=float(pos.get("unrealized_pnl", 0)),
                        margin_used=float(pos.get("margin_used", 0)),
                        financing=float(pos.get("financing", 0)),
                    )
                else:
                    remote_positions[symbol] = ForexPosition(
                        symbol=symbol,
                        units=float(getattr(pos, "units", 0)),
                        average_price=float(getattr(pos, "average_price", 0)),
                        unrealized_pnl=float(getattr(pos, "unrealized_pnl", 0)),
                        margin_used=float(getattr(pos, "margin_used", 0)),
                        financing=float(getattr(pos, "financing", 0)),
                    )

            total_financing = sum(p.financing for p in remote_positions.values())

            discrepancies = self._compare_positions(local_positions, remote_positions)

            result = SyncResult(
                timestamp=timestamp,
                success=True,
                local_positions=local_positions,
                remote_positions=remote_positions,
                discrepancies=discrepancies,
                total_financing=total_financing,
            )

            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} forex position discrepancies")
                for d in discrepancies:
                    logger.warning(f"  {d.symbol}: {d.discrepancy_type.value} - {d.details}")
                    if self._on_discrepancy:
                        try:
                            self._on_discrepancy(d)
                        except Exception as e:
                            logger.error(f"Discrepancy callback error: {e}")

            if self._on_sync_complete:
                try:
                    self._on_sync_complete(result)
                except Exception as e:
                    logger.error(f"Sync complete callback error: {e}")

            self._last_sync = result
            return result

        except Exception as e:
            logger.error(f"Forex position sync failed: {e}")
            result = SyncResult(
                timestamp=timestamp,
                success=False,
                error=str(e),
            )
            self._last_sync = result
            return result

    def _compare_positions(
        self,
        local: Dict[str, float],
        remote: Dict[str, ForexPosition],
    ) -> List[PositionDiscrepancy]:
        """Compare local and remote positions."""
        discrepancies: List[PositionDiscrepancy] = []

        all_symbols = set(local.keys()) | set(remote.keys())

        for symbol in all_symbols:
            local_units = local.get(symbol)
            remote_pos = remote.get(symbol)
            remote_units = remote_pos.units if remote_pos else None

            # Missing locally
            if local_units is None and remote_units is not None and remote_units != 0:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=PositionDiscrepancyType.MISSING_LOCAL,
                        local_units=None,
                        remote_units=remote_units,
                        remote_price=remote_pos.average_price if remote_pos else None,
                        remote_financing=remote_pos.financing if remote_pos else None,
                        details=(
                            f"Position exists on broker ({remote_units:.2f} units) "
                            f"but not in local state"
                        ),
                    )
                )
                continue

            # Missing remotely
            if remote_units is None and local_units is not None and local_units != 0:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=PositionDiscrepancyType.MISSING_REMOTE,
                        local_units=local_units,
                        remote_units=None,
                        details=(
                            f"Position exists locally ({local_units:.2f} units) "
                            f"but not on broker"
                        ),
                    )
                )
                continue

            # Both exist - check for mismatches
            if local_units is not None and remote_units is not None:
                # Side mismatch (one long, one short)
                if (
                    (local_units > 0) != (remote_units > 0)
                    and local_units != 0
                    and remote_units != 0
                ):
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=symbol,
                            discrepancy_type=PositionDiscrepancyType.SIDE_MISMATCH,
                            local_units=local_units,
                            remote_units=remote_units,
                            remote_price=remote_pos.average_price if remote_pos else None,
                            details=(
                                f"Side mismatch: local={'long' if local_units > 0 else 'short'}, "
                                f"remote={'long' if remote_units > 0 else 'short'}"
                            ),
                        )
                    )
                    continue

                # Units mismatch
                units_diff = abs(remote_units - local_units)
                max_units = max(abs(local_units), abs(remote_units))

                if max_units > 0:
                    pct_diff = units_diff / max_units
                else:
                    pct_diff = 0.0

                if (
                    pct_diff > self._config.position_tolerance_pct
                    and units_diff > self._config.min_unit_diff
                ):
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=symbol,
                            discrepancy_type=PositionDiscrepancyType.UNITS_MISMATCH,
                            local_units=local_units,
                            remote_units=remote_units,
                            remote_price=remote_pos.average_price if remote_pos else None,
                            remote_financing=remote_pos.financing if remote_pos else None,
                            details=(
                                f"Units mismatch: local={local_units:.2f}, "
                                f"remote={remote_units:.2f}, "
                                f"diff={units_diff:.2f} ({pct_diff:.2%})"
                            ),
                        )
                    )

        return discrepancies

    def _execute_auto_reconciliation(
        self,
        discrepancies: List[PositionDiscrepancy],
        remote_positions: Dict[str, ForexPosition],
    ) -> List[ReconciliationOrderResult]:
        """
        Execute automatic reconciliation for discrepancies.

        Only reconciles UNITS_MISMATCH and MISSING_LOCAL discrepancies.
        SIDE_MISMATCH and MISSING_REMOTE require manual intervention.

        Args:
            discrepancies: List of position discrepancies
            remote_positions: Current remote positions

        Returns:
            List of reconciliation order results
        """
        results: List[ReconciliationOrderResult] = []

        if self._reconciliation_executor is None:
            logger.warning("Auto-reconciliation enabled but no executor provided")
            return results

        for disc in discrepancies:
            # Only reconcile certain types
            if disc.discrepancy_type == PositionDiscrepancyType.SIDE_MISMATCH:
                logger.warning(
                    f"Skipping {disc.symbol}: side mismatch requires manual intervention"
                )
                continue

            if disc.discrepancy_type == PositionDiscrepancyType.MISSING_REMOTE:
                logger.warning(
                    f"Skipping {disc.symbol}: position missing on broker "
                    "(may have been closed externally)"
                )
                continue

            # Calculate units to trade to align with remote
            units_to_trade = 0.0

            if disc.discrepancy_type == PositionDiscrepancyType.MISSING_LOCAL:
                # Remote has position, local doesn't - skip (external position)
                # Or optionally close the remote position
                logger.info(
                    f"Skipping {disc.symbol}: external position on broker, "
                    "update local state or close manually"
                )
                continue

            elif disc.discrepancy_type == PositionDiscrepancyType.UNITS_MISMATCH:
                # Align local to remote by trading the difference
                # If remote > local, we need to sell (remote - local)
                # If remote < local, we need to buy (local - remote)
                local_units = disc.local_units or 0.0
                remote_units = disc.remote_units or 0.0

                # We adjust our local position to match remote
                # This means we need to trade: local_target - local_current
                # where local_target = remote_units
                units_to_trade = remote_units - local_units

            if abs(units_to_trade) < self._config.min_unit_diff:
                logger.debug(
                    f"Skipping {disc.symbol}: difference {units_to_trade:.2f} "
                    f"below minimum {self._config.min_unit_diff}"
                )
                continue

            # Check max reconcile units
            if abs(units_to_trade) > self._config.max_reconcile_units:
                logger.warning(
                    f"Reconciliation for {disc.symbol} ({units_to_trade:.0f} units) "
                    f"exceeds max ({self._config.max_reconcile_units:.0f}). "
                    f"Capping to max."
                )
                units_to_trade = (
                    self._config.max_reconcile_units
                    if units_to_trade > 0
                    else -self._config.max_reconcile_units
                )

            # Create reconciliation action
            action = ReconciliationAction(
                symbol=disc.symbol,
                discrepancy=disc,
                units_to_trade=units_to_trade,
            )

            # Execute reconciliation
            result = self._reconciliation_executor.execute_reconciliation(action)
            results.append(result)

            # Invoke callback
            if self._on_reconciliation:
                try:
                    self._on_reconciliation(result)
                except Exception as e:
                    logger.error(f"Reconciliation callback error: {e}")

        return results

    # =========================================================================
    # Background Sync (Async)
    # =========================================================================

    def start_background_sync(self) -> None:
        """Start background synchronization loop (async)."""
        if self._running:
            logger.warning("Background forex sync already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._background_sync_loop())
        logger.info(
            f"Started background forex position sync "
            f"(interval: {self._config.sync_interval_sec}s)"
        )

    def stop_background_sync(self) -> None:
        """Stop background synchronization loop."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
        if self._sync_thread and self._sync_thread.is_alive():
            self._stop_event.set()
            self._sync_thread.join(timeout=5.0)
            self._sync_thread = None
        logger.info("Stopped background forex position sync")

    async def _background_sync_loop(self) -> None:
        """Background async sync loop."""
        while self._running:
            try:
                await self.sync_once_async()
                await asyncio.sleep(self._config.sync_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background forex sync error: {e}")
                await asyncio.sleep(60.0)  # Wait before retry

    # =========================================================================
    # Background Sync (Thread-based for non-async contexts)
    # =========================================================================

    def start_background_sync_threaded(self) -> None:
        """Start background synchronization using threads (non-async)."""
        if self._running:
            logger.warning("Background forex sync already running")
            return

        self._running = True
        self._stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._background_sync_loop_threaded,
            daemon=True,
            name="ForexPositionSync",
        )
        self._sync_thread.start()
        logger.info(
            f"Started threaded background forex position sync "
            f"(interval: {self._config.sync_interval_sec}s)"
        )

    def _background_sync_loop_threaded(self) -> None:
        """Background threaded sync loop."""
        while self._running and not self._stop_event.is_set():
            try:
                self.sync_once()
                self._stop_event.wait(self._config.sync_interval_sec)
            except Exception as e:
                logger.error(f"Background forex sync error: {e}")
                self._stop_event.wait(60.0)


# =============================================================================
# OANDA-Specific Reconciliation
# =============================================================================


@dataclass
class OandaReconciliationResult:
    """Result of OANDA position reconciliation."""

    timestamp: float
    success: bool

    # Account info
    account_balance: Optional[float] = None
    account_nav: Optional[float] = None
    margin_used: Optional[float] = None
    margin_available: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    # Position summary
    position_count: int = 0
    total_financing: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    # Discrepancies
    position_discrepancies: List[PositionDiscrepancy] = field(default_factory=list)

    error: Optional[str] = None


def reconcile_oanda_state(
    adapter: Any,
    local_positions: Dict[str, float],
    config: Optional[SyncConfig] = None,
) -> OandaReconciliationResult:
    """
    Reconcile local state with OANDA exchange state.

    This function performs a comprehensive comparison between local trading
    state and the actual state on OANDA, including:
    - Positions (units, entry price, financing)
    - Account info (balance, NAV, margin)

    Args:
        adapter: OandaOrderExecutionAdapter instance
        local_positions: Dict mapping symbol to local position units
        config: Optional sync configuration

    Returns:
        OandaReconciliationResult with comprehensive comparison
    """
    config = config or SyncConfig()
    timestamp = time.time()

    try:
        # Get account info
        account_info = None
        try:
            account_info = adapter.get_account_summary()
        except Exception as e:
            logger.warning(f"Failed to get OANDA account info: {e}")

        # Get positions
        remote_positions_raw = adapter.get_positions()

        # Convert to ForexPosition format
        remote_positions: Dict[str, ForexPosition] = {}
        for symbol, pos in remote_positions_raw.items():
            if isinstance(pos, ForexPosition):
                remote_positions[symbol] = pos
            elif isinstance(pos, dict):
                remote_positions[symbol] = ForexPosition(
                    symbol=symbol,
                    units=float(pos.get("units", 0)),
                    average_price=float(pos.get("average_price", 0)),
                    unrealized_pnl=float(pos.get("unrealized_pnl", 0)),
                    margin_used=float(pos.get("margin_used", 0)),
                    financing=float(pos.get("financing", 0)),
                )
            else:
                remote_positions[symbol] = ForexPosition(
                    symbol=symbol,
                    units=float(getattr(pos, "units", 0)),
                    average_price=float(getattr(pos, "average_price", 0)),
                    unrealized_pnl=float(getattr(pos, "unrealized_pnl", 0)),
                    margin_used=float(getattr(pos, "margin_used", 0)),
                    financing=float(getattr(pos, "financing", 0)),
                )

        # Create synchronizer for position comparison
        def get_local() -> Dict[str, float]:
            return local_positions

        # Simple position provider wrapper
        class PositionProviderWrapper:
            def __init__(self, positions: Dict[str, ForexPosition]):
                self._positions = positions

            def get_positions(
                self, symbols: Optional[Sequence[str]] = None
            ) -> Dict[str, ForexPosition]:
                return self._positions

        sync = ForexPositionSynchronizer(
            position_provider=PositionProviderWrapper(remote_positions),
            local_state_getter=get_local,
            config=config,
        )

        sync_result = sync.sync_once()

        # Calculate exposure
        long_exposure = 0.0
        short_exposure = 0.0
        total_financing = 0.0

        for pos in remote_positions.values():
            if pos.units > 0:
                long_exposure += pos.notional
            else:
                short_exposure += abs(pos.units) * pos.average_price
            total_financing += pos.financing

        # Build result
        result = OandaReconciliationResult(
            timestamp=timestamp,
            success=True,
            position_count=len(remote_positions),
            total_financing=total_financing,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            position_discrepancies=sync_result.discrepancies,
        )

        # Add account info if available
        if account_info:
            if isinstance(account_info, dict):
                result.account_balance = float(account_info.get("balance", 0))
                result.account_nav = float(account_info.get("nav", 0))
                result.margin_used = float(account_info.get("margin_used", 0))
                result.margin_available = float(account_info.get("margin_available", 0))
                result.unrealized_pnl = float(account_info.get("unrealized_pnl", 0))
            else:
                result.account_balance = float(getattr(account_info, "balance", 0))
                result.account_nav = float(getattr(account_info, "nav", 0))
                result.margin_used = float(getattr(account_info, "margin_used", 0))
                result.margin_available = float(getattr(account_info, "margin_available", 0))
                result.unrealized_pnl = float(getattr(account_info, "unrealized_pnl", 0))

        return result

    except Exception as e:
        logger.error(f"OANDA reconciliation failed: {e}")
        return OandaReconciliationResult(
            timestamp=timestamp,
            success=False,
            error=str(e),
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_forex_position_sync(
    position_provider: ForexPositionProvider,
    local_state_getter: Callable[[], Dict[str, float]],
    config: Optional[SyncConfig] = None,
    swap_tracker: Optional[SwapCostTracker] = None,
) -> ForexPositionSynchronizer:
    """
    Create a ForexPositionSynchronizer with configuration.

    Args:
        position_provider: Adapter providing get_positions()
        local_state_getter: Callable returning local positions dict
        config: Optional sync configuration
        swap_tracker: Optional swap cost tracker

    Returns:
        Configured ForexPositionSynchronizer
    """
    return ForexPositionSynchronizer(
        position_provider=position_provider,
        local_state_getter=local_state_getter,
        config=config or SyncConfig(),
        swap_tracker=swap_tracker,
    )


def create_oanda_position_sync(
    oanda_adapter: Any,
    local_state_getter: Callable[[], Dict[str, float]],
    config: Optional[Dict[str, Any]] = None,
) -> ForexPositionSynchronizer:
    """
    Create a ForexPositionSynchronizer configured for OANDA.

    Args:
        oanda_adapter: OandaOrderExecutionAdapter instance
        local_state_getter: Callable returning local positions dict
        config: Optional configuration dict

    Returns:
        Configured ForexPositionSynchronizer
    """
    sync_config = SyncConfig(
        sync_interval_sec=config.get("sync_interval_sec", 30.0) if config else 30.0,
        position_tolerance_pct=config.get("position_tolerance_pct", 0.01) if config else 0.01,
        min_unit_diff=config.get("min_unit_diff", 1.0) if config else 1.0,
        auto_reconcile=config.get("auto_reconcile", False) if config else False,
        max_reconcile_units=config.get("max_reconcile_units", 100_000) if config else 100_000,
        alert_on_discrepancy=config.get("alert_on_discrepancy", True) if config else True,
        track_financing=config.get("track_financing", True) if config else True,
    )

    return ForexPositionSynchronizer(
        position_provider=oanda_adapter,
        local_state_getter=local_state_getter,
        config=sync_config,
    )
