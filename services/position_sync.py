# -*- coding: utf-8 -*-
"""
services/position_sync.py
Position synchronization service for live trading.

Phase 9: Live Trading Improvements (2025-11-27)

This module provides position reconciliation between local state and exchange:
- Fetch real positions from exchange (Alpaca, Binance)
- Compare with local state
- Detect and report discrepancies
- Support for periodic sync loops

Best Practices Applied:
- Defensive error handling to avoid disrupting trading
- Atomic state updates
- Comprehensive logging for audit trail
- Configurable tolerances for position comparison
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Types
# =============================================================================


class PositionProvider(Protocol):
    """Protocol for position data providers."""

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Get current positions from exchange."""
        ...


class AccountInfoProvider(Protocol):
    """Protocol for account info providers."""

    def get_account_info(self) -> Any:
        """Get account information."""
        ...


class OrderProvider(Protocol):
    """Protocol for order data providers."""

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Any]:
        """Get open orders from exchange."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


class DiscrepancyType(str, Enum):
    """Types of position discrepancies."""

    MISSING_LOCAL = "missing_local"  # Position exists on exchange but not locally
    MISSING_REMOTE = "missing_remote"  # Position exists locally but not on exchange
    QTY_MISMATCH = "qty_mismatch"  # Quantity differs
    PRICE_MISMATCH = "price_mismatch"  # Entry price differs (informational)


@dataclass
class PositionDiscrepancy:
    """Represents a discrepancy between local and remote positions."""

    symbol: str
    discrepancy_type: DiscrepancyType
    local_qty: Optional[Decimal]
    remote_qty: Optional[Decimal]
    local_price: Optional[Decimal] = None
    remote_price: Optional[Decimal] = None
    remote_market_value: Optional[Decimal] = None
    details: str = ""

    @property
    def qty_diff(self) -> Decimal:
        """Quantity difference (remote - local)."""
        local = self.local_qty or Decimal("0")
        remote = self.remote_qty or Decimal("0")
        return remote - local


@dataclass
class SyncResult:
    """Result of a position synchronization operation."""

    timestamp: float
    success: bool
    local_positions: Dict[str, Decimal] = field(default_factory=dict)
    remote_positions: Dict[str, Decimal] = field(default_factory=dict)
    discrepancies: List[PositionDiscrepancy] = field(default_factory=list)
    open_orders: int = 0
    error: Optional[str] = None
    # Auto-reconciliation outcome (P1 #9)
    halted: bool = False
    halt_reason: Optional[str] = None
    reconcile_actions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        """Whether any discrepancies were found."""
        return len(self.discrepancies) > 0

    @property
    def total_local_notional(self) -> Decimal:
        """Total notional value of local positions."""
        return sum(self.local_positions.values(), Decimal("0"))

    @property
    def total_remote_notional(self) -> Decimal:
        """Total notional value of remote positions."""
        return sum(self.remote_positions.values(), Decimal("0"))


@dataclass
class SyncConfig:
    """Configuration for position synchronization."""

    # Tolerance for quantity comparison (fraction of position)
    qty_tolerance_pct: float = 0.001  # 0.1% tolerance

    # Minimum absolute quantity difference to report
    min_qty_diff: float = 0.0001

    # Tolerance for entry price comparison (fraction)
    price_tolerance_pct: float = 0.01  # 1% tolerance

    # Interval for periodic sync (seconds)
    sync_interval_s: float = 300.0  # 5 minutes

    # Whether to auto-resolve discrepancies (dangerous!)
    auto_resolve: bool = False

    # --- Auto-reconciliation policy (P1 #9): act on persistent drift, don't just
    #     detect it. A discrepancy that survives N consecutive syncs is "persistent".
    halt_on_unreconciled: bool = True       # block new orders while drift persists
    consecutive_halt_threshold: int = 2     # syncs a discrepancy must persist before halt
    auto_flatten: bool = False              # also close the drifted position at the broker
    max_flatten_qty: float = 1e9            # safety cap on auto-flatten size

    # Symbols to include (None = all)
    include_symbols: Optional[List[str]] = None

    # Symbols to exclude
    exclude_symbols: Optional[List[str]] = None


# =============================================================================
# Position Synchronizer
# =============================================================================


class PositionSynchronizer:
    """
    Service for synchronizing positions between local state and exchange.

    Provides:
    - One-time sync operations
    - Periodic background sync
    - Discrepancy detection and reporting
    - Callbacks for discrepancy handling

    Usage:
        sync = PositionSynchronizer(
            position_provider=alpaca_adapter,
            local_state_getter=lambda: my_state.positions,
            config=SyncConfig(sync_interval_s=300),
        )

        # One-time sync
        result = sync.sync_once()
        if result.has_discrepancies:
            for d in result.discrepancies:
                print(f"Discrepancy: {d.symbol} {d.discrepancy_type}")

        # Start background sync
        sync.start_background_sync()
    """

    def __init__(
        self,
        position_provider: PositionProvider,
        local_state_getter: Callable[[], Dict[str, Decimal]],
        config: Optional[SyncConfig] = None,
        on_discrepancy: Optional[Callable[[PositionDiscrepancy], None]] = None,
        on_sync_complete: Optional[Callable[[SyncResult], None]] = None,
        order_provider: Optional[OrderProvider] = None,
        on_halt: Optional[Callable[[str], None]] = None,
        flatten_fn: Optional[Callable[[str, Decimal], Any]] = None,
    ) -> None:
        """
        Initialize position synchronizer.

        Args:
            position_provider: Adapter providing get_positions()
            local_state_getter: Callable returning local positions dict
            config: Sync configuration
            on_discrepancy: Callback for each discrepancy found
            on_sync_complete: Callback after each sync completes
            order_provider: Optional adapter for open orders
        """
        self._provider = position_provider
        self._local_state_getter = local_state_getter
        self._config = config or SyncConfig()
        self._on_discrepancy = on_discrepancy
        self._on_sync_complete = on_sync_complete
        self._order_provider = order_provider
        # Auto-reconciliation state (P1 #9)
        self._on_halt = on_halt
        self._flatten_fn = flatten_fn
        self._consecutive: Dict[str, int] = {}   # symbol -> consecutive unreconciled syncs
        self._halted = False
        self._halt_reason: Optional[str] = None

        # Background sync state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_sync: Optional[SyncResult] = None

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
            logger.debug(f"Local positions: {len(local_positions)} symbols")

            # Get remote positions
            filter_symbols = self._config.include_symbols
            remote_raw = self._provider.get_positions(filter_symbols)

            # Convert remote positions to standard format
            remote_positions: Dict[str, Decimal] = {}
            remote_meta: Dict[str, Any] = {}

            for symbol, pos in remote_raw.items():
                # Skip excluded symbols
                if self._config.exclude_symbols and symbol in self._config.exclude_symbols:
                    continue

                qty = self._extract_qty(pos)
                if qty and qty != Decimal("0"):
                    remote_positions[symbol] = qty
                    remote_meta[symbol] = pos

            logger.debug(f"Remote positions: {len(remote_positions)} symbols")

            # Get open orders count
            open_orders = 0
            if self._order_provider:
                try:
                    orders = self._order_provider.get_open_orders()
                    open_orders = len(orders)
                except Exception as e:
                    logger.warning(f"Failed to get open orders: {e}")

            # Compare positions
            discrepancies = self._compare_positions(
                local_positions, remote_positions, remote_meta
            )

            # Auto-reconciliation (P1 #9): act on PERSISTENT drift, not just detect.
            halted, halt_reason, actions = self._auto_reconcile(
                discrepancies, remote_positions
            )

            # Create result
            result = SyncResult(
                timestamp=timestamp,
                success=True,
                local_positions=local_positions,
                remote_positions=remote_positions,
                discrepancies=discrepancies,
                open_orders=open_orders,
                halted=halted,
                halt_reason=halt_reason,
                reconcile_actions=actions,
            )

            # Invoke callbacks
            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} position discrepancies")
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
            logger.error(f"Position sync failed: {e}")
            result = SyncResult(
                timestamp=timestamp,
                success=False,
                error=str(e),
            )
            self._last_sync = result
            return result

    # ------------------------------------------------------------------
    # Auto-reconciliation (P1 #9)
    # ------------------------------------------------------------------
    def _auto_reconcile(self, discrepancies, remote_positions):
        """Act on persistent drift: alert → halt new orders → optional auto-flatten.

        A discrepancy is "persistent" once it survives ``consecutive_halt_threshold``
        consecutive syncs. On persistence we halt new orders (always, if
        ``halt_on_unreconciled``) and, when ``auto_flatten`` is enabled, flatten the
        drifted symbol at the broker via ``flatten_fn``. Returns (halted, reason, actions).
        """
        cfg = self._config
        actions: List[Dict[str, Any]] = []
        current_syms = {d.symbol for d in discrepancies}

        # decay counters for symbols that are no longer drifting (self-healed)
        for sym in list(self._consecutive.keys()):
            if sym not in current_syms:
                self._consecutive.pop(sym, None)

        persistent: List[str] = []
        for d in discrepancies:
            self._consecutive[d.symbol] = self._consecutive.get(d.symbol, 0) + 1
            if self._consecutive[d.symbol] >= max(1, int(cfg.consecutive_halt_threshold)):
                persistent.append(d.symbol)

        # drift self-healed everywhere -> clear halt
        if not discrepancies:
            if self._halted:
                logger.info("position drift cleared — releasing reconciliation halt")
            self._halted = False
            self._halt_reason = None
            return False, None, actions

        if not persistent:
            return self._halted, self._halt_reason, actions

        # --- persistent drift: halt new orders ---
        if cfg.halt_on_unreconciled and not self._halted:
            self._halted = True
            self._halt_reason = (
                f"persistent position drift on {persistent} "
                f"(>= {cfg.consecutive_halt_threshold} consecutive syncs)"
            )
            logger.error("RECONCILIATION HALT: %s", self._halt_reason)
            actions.append({"action": "halt_new_orders", "symbols": list(persistent),
                            "reason": self._halt_reason})
            if self._on_halt:
                try:
                    self._on_halt(self._halt_reason)
                except Exception as e:  # pragma: no cover
                    logger.error("on_halt callback error: %s", e)

        # --- optional auto-flatten of the drifted broker position ---
        if cfg.auto_flatten and self._flatten_fn is not None:
            for sym in persistent:
                rq = remote_positions.get(sym, Decimal("0"))
                if rq == 0:
                    continue
                if abs(float(rq)) > float(cfg.max_flatten_qty):
                    actions.append({"action": "flatten_skipped_cap", "symbol": sym,
                                    "remote_qty": str(rq)})
                    continue
                try:
                    self._flatten_fn(sym, rq)
                    actions.append({"action": "auto_flatten", "symbol": sym, "remote_qty": str(rq)})
                    logger.warning("auto-flatten issued for %s (remote qty %s)", sym, rq)
                except Exception as e:  # pragma: no cover
                    logger.error("auto-flatten failed for %s: %s", sym, e)
                    actions.append({"action": "flatten_error", "symbol": sym, "error": str(e)})

        return self._halted, self._halt_reason, actions

    @property
    def is_halted(self) -> bool:
        """Whether reconciliation has halted new-order flow (check before trading)."""
        return self._halted

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def should_block_new_orders(self) -> bool:
        """The live loop calls this before submitting; True => do not send new orders."""
        return self._halted

    def clear_halt(self, approval: str = "") -> bool:
        """Manually clear the reconciliation halt (after operator review)."""
        self._halted = False
        self._halt_reason = None
        self._consecutive.clear()
        logger.info("reconciliation halt cleared (approval=%s)", approval or "n/a")
        return True

    def _extract_qty(self, position: Any) -> Optional[Decimal]:
        """Extract quantity from position object."""
        if isinstance(position, Decimal):
            return position
        if hasattr(position, "qty"):
            return Decimal(str(position.qty))
        if isinstance(position, dict):
            qty = position.get("qty") or position.get("quantity") or position.get("size")
            if qty is not None:
                return Decimal(str(qty))
        return None

    def _extract_price(self, position: Any) -> Optional[Decimal]:
        """Extract entry price from position object."""
        if hasattr(position, "avg_entry_price"):
            return Decimal(str(position.avg_entry_price))
        if isinstance(position, dict):
            price = position.get("avg_entry_price") or position.get("entry_price")
            if price is not None:
                return Decimal(str(price))
        return None

    def _extract_market_value(self, position: Any) -> Optional[Decimal]:
        """Extract market value from position object."""
        if hasattr(position, "market_value"):
            val = getattr(position, "market_value", None)
            if val is not None:
                return Decimal(str(val))
        if isinstance(position, dict):
            if "meta" in position and "market_value" in position["meta"]:
                return Decimal(str(position["meta"]["market_value"]))
            if "market_value" in position:
                return Decimal(str(position["market_value"]))
        return None

    def _compare_positions(
        self,
        local: Dict[str, Decimal],
        remote: Dict[str, Decimal],
        remote_meta: Dict[str, Any],
    ) -> List[PositionDiscrepancy]:
        """Compare local and remote positions."""
        discrepancies: List[PositionDiscrepancy] = []

        all_symbols = set(local.keys()) | set(remote.keys())

        for symbol in all_symbols:
            local_qty = local.get(symbol)
            remote_qty = remote.get(symbol)

            # Missing locally
            if local_qty is None and remote_qty is not None:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.MISSING_LOCAL,
                        local_qty=None,
                        remote_qty=remote_qty,
                        remote_price=self._extract_price(remote_meta.get(symbol)),
                        remote_market_value=self._extract_market_value(remote_meta.get(symbol)),
                        details=f"Position exists on exchange ({remote_qty}) but not in local state",
                    )
                )
                continue

            # Missing remotely
            if remote_qty is None and local_qty is not None:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.MISSING_REMOTE,
                        local_qty=local_qty,
                        remote_qty=None,
                        details=f"Position exists locally ({local_qty}) but not on exchange",
                    )
                )
                continue

            # Both exist - check quantity
            if local_qty is not None and remote_qty is not None:
                qty_diff = abs(remote_qty - local_qty)
                max_qty = max(abs(local_qty), abs(remote_qty))

                if max_qty > 0:
                    pct_diff = float(qty_diff / max_qty)
                else:
                    pct_diff = 0.0

                # Check if difference exceeds tolerance
                if (
                    pct_diff > self._config.qty_tolerance_pct
                    and float(qty_diff) > self._config.min_qty_diff
                ):
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=symbol,
                            discrepancy_type=DiscrepancyType.QTY_MISMATCH,
                            local_qty=local_qty,
                            remote_qty=remote_qty,
                            remote_price=self._extract_price(remote_meta.get(symbol)),
                            remote_market_value=self._extract_market_value(remote_meta.get(symbol)),
                            details=(
                                f"Quantity mismatch: local={local_qty}, remote={remote_qty}, "
                                f"diff={qty_diff} ({pct_diff:.2%})"
                            ),
                        )
                    )

        return discrepancies

    # ==========================================================================
    # Background Sync
    # ==========================================================================

    def start_background_sync(self) -> None:
        """Start background synchronization loop."""
        if self._running:
            logger.warning("Background sync already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._background_sync_loop())
        logger.info(
            f"Started background position sync (interval: {self._config.sync_interval_s}s)"
        )

    def stop_background_sync(self) -> None:
        """Stop background synchronization loop."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
        logger.info("Stopped background position sync")

    async def _background_sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                # Run sync
                self.sync_once()

                # Wait for next interval
                await asyncio.sleep(self._config.sync_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                # Wait before retry
                await asyncio.sleep(60.0)


# =============================================================================
# Alpaca-Specific Reconciliation
# =============================================================================


@dataclass
class AlpacaReconciliationResult:
    """Result of Alpaca position reconciliation."""

    timestamp: float
    success: bool

    # Account info
    account_equity: Optional[Decimal] = None
    buying_power: Optional[Decimal] = None
    cash: Optional[Decimal] = None
    pattern_day_trader: bool = False
    day_trade_count: int = 0

    # Position summary
    position_count: int = 0
    total_market_value: Decimal = Decimal("0")
    total_unrealized_pl: Decimal = Decimal("0")

    # Discrepancies
    position_discrepancies: List[PositionDiscrepancy] = field(default_factory=list)
    missing_orders: List[str] = field(default_factory=list)  # Order IDs on exchange not locally
    extra_orders: List[str] = field(default_factory=list)  # Order IDs locally not on exchange

    error: Optional[str] = None


def reconcile_alpaca_state(
    adapter: Any,
    local_positions: Dict[str, Decimal],
    local_orders: Optional[Dict[str, Any]] = None,
    config: Optional[SyncConfig] = None,
) -> AlpacaReconciliationResult:
    """
    Reconcile local state with Alpaca exchange state.

    This function performs a comprehensive comparison between local trading
    state and the actual state on Alpaca, including:
    - Positions (quantity, entry price)
    - Account info (equity, buying power, PDT status)
    - Open orders

    Args:
        adapter: AlpacaOrderExecutionAdapter instance
        local_positions: Dict mapping symbol to local position quantity
        local_orders: Optional dict mapping order_id to order info
        config: Optional sync configuration

    Returns:
        AlpacaReconciliationResult with comprehensive comparison
    """
    config = config or SyncConfig()
    timestamp = time.time()

    try:
        # Get account info
        account_info = None
        try:
            account_info = adapter.get_account_info()
        except Exception as e:
            logger.warning(f"Failed to get account info: {e}")

        # Get positions
        remote_positions = adapter.get_positions()

        # Get open orders
        open_orders = []
        try:
            open_orders = adapter.get_open_orders()
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")

        # Create synchronizer for position comparison
        def get_local() -> Dict[str, Decimal]:
            return local_positions

        sync = PositionSynchronizer(
            position_provider=adapter,
            local_state_getter=get_local,
            config=config,
        )

        # Compare positions
        sync_result = sync.sync_once()

        # Calculate totals from remote positions
        total_market_value = Decimal("0")
        total_unrealized_pl = Decimal("0")

        for symbol, pos in remote_positions.items():
            if hasattr(pos, "meta") and pos.meta:
                mv = pos.meta.get("market_value")
                if mv:
                    total_market_value += Decimal(str(mv))
                upl = pos.meta.get("unrealized_pl")
                if upl:
                    total_unrealized_pl += Decimal(str(upl))

        # Compare orders if local orders provided
        missing_orders: List[str] = []
        extra_orders: List[str] = []

        if local_orders is not None:
            local_order_ids = set(local_orders.keys())
            remote_order_ids = set()

            for order in open_orders:
                oid = getattr(order, "client_order_id", None) or str(order)
                remote_order_ids.add(oid)

            missing_orders = list(remote_order_ids - local_order_ids)
            extra_orders = list(local_order_ids - remote_order_ids)

        # Build result
        result = AlpacaReconciliationResult(
            timestamp=timestamp,
            success=True,
            position_count=len(remote_positions),
            total_market_value=total_market_value,
            total_unrealized_pl=total_unrealized_pl,
            position_discrepancies=sync_result.discrepancies,
            missing_orders=missing_orders,
            extra_orders=extra_orders,
        )

        # Add account info if available
        if account_info:
            result.account_equity = account_info.cash_balance + total_market_value
            result.buying_power = account_info.buying_power
            result.cash = account_info.cash_balance
            result.pattern_day_trader = getattr(account_info, "pattern_day_trader", False)

            raw = getattr(account_info, "raw_data", {}) or {}
            result.day_trade_count = raw.get("daytrade_count", 0)

        return result

    except Exception as e:
        logger.error(f"Alpaca reconciliation failed: {e}")
        return AlpacaReconciliationResult(
            timestamp=timestamp,
            success=False,
            error=str(e),
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_position_sync_for_alpaca(
    adapter: Any,
    local_state_getter: Callable[[], Dict[str, Decimal]],
    config: Optional[SyncConfig] = None,
) -> PositionSynchronizer:
    """
    Create a PositionSynchronizer configured for Alpaca.

    Args:
        adapter: AlpacaOrderExecutionAdapter instance
        local_state_getter: Callable returning local positions dict
        config: Optional sync configuration

    Returns:
        Configured PositionSynchronizer
    """
    return PositionSynchronizer(
        position_provider=adapter,
        local_state_getter=local_state_getter,
        config=config or SyncConfig(),
        order_provider=adapter,  # Alpaca adapter implements both protocols
    )


def create_position_sync_for_binance(
    adapter: Any,
    local_state_getter: Callable[[], Dict[str, Decimal]],
    config: Optional[SyncConfig] = None,
) -> PositionSynchronizer:
    """
    Create a PositionSynchronizer configured for Binance.

    Args:
        adapter: Binance adapter with get_positions method
        local_state_getter: Callable returning local positions dict
        config: Optional sync configuration

    Returns:
        Configured PositionSynchronizer
    """
    # Binance-specific default config
    binance_config = config or SyncConfig(
        qty_tolerance_pct=0.0001,  # Tighter tolerance for crypto
        min_qty_diff=1e-8,  # Satoshi-level precision
        sync_interval_s=60.0,  # More frequent for 24/7 market
    )

    return PositionSynchronizer(
        position_provider=adapter,
        local_state_getter=local_state_getter,
        config=binance_config,
    )
