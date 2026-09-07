# -*- coding: utf-8 -*-
"""
Position Reconciler - Syncs local state with broker.

AGENT ZONE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Callable

from .journal import OrderJournal, JournalEntry, JournalStatus


class MismatchAction(str, Enum):
    """Action to take on mismatch."""

    ACCEPT_BROKER = "accept_broker"
    ACCEPT_LOCAL = "accept_local"
    HALT = "halt"
    ALERT = "alert"


@dataclass
class PositionMismatch:
    """
    Represents a position mismatch between local and broker.
    """

    symbol: str
    local_quantity: Decimal
    broker_quantity: Decimal
    difference: Decimal
    severity: str = "warning"  # info, warning, error, critical
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_significant(self) -> bool:
        """Check if mismatch is significant."""
        # More than 1% difference is significant
        if self.broker_quantity == 0:
            return abs(self.local_quantity) > Decimal("0.01")
        return abs(self.difference / self.broker_quantity) > Decimal("0.01")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "local_quantity": str(self.local_quantity),
            "broker_quantity": str(self.broker_quantity),
            "difference": str(self.difference),
            "severity": self.severity,
            "is_significant": self.is_significant,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReconciliationResult:
    """
    Result of reconciliation.
    """

    success: bool = True
    mismatches: List[PositionMismatch] = field(default_factory=list)
    resolved: List[str] = field(default_factory=list)  # Symbols resolved
    halted: bool = False
    halt_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_significant_mismatches(self) -> bool:
        """Check if there are significant mismatches."""
        return any(m.is_significant for m in self.mismatches)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "mismatches": [m.to_dict() for m in self.mismatches],
            "resolved": self.resolved,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "has_significant_mismatches": self.has_significant_mismatches,
            "timestamp": self.timestamp.isoformat(),
        }


# Type for position fetch function
FetchPositionsFn = Callable[[], Dict[str, Decimal]]
FetchOrderStatusFn = Callable[[JournalEntry], Optional[JournalStatus]]


class PositionReconciler:
    """
    Reconciles local positions with broker positions.

    Used on startup and periodically to ensure consistency.

    Usage:
        reconciler = PositionReconciler(
            local_positions={"BTCUSDT": Decimal("1.5")},
            fetch_broker_positions=broker.get_positions,
        )

        result = reconciler.reconcile()
        if result.halted:
            halt_trading(result.halt_reason)
    """

    def __init__(
        self,
        local_positions: Optional[Dict[str, Decimal]] = None,
        fetch_broker_positions: Optional[FetchPositionsFn] = None,
        journal: Optional[OrderJournal] = None,
        fetch_broker_order_status: Optional[FetchOrderStatusFn] = None,
        mismatch_action: MismatchAction = MismatchAction.HALT,
        tolerance_pct: Decimal = Decimal("0.001"),  # 0.1% tolerance
    ):
        """
        Initialize reconciler.

        Args:
            local_positions: Current local position state
            fetch_broker_positions: Function to fetch broker positions
            journal: Order journal for pending order check
            mismatch_action: Action on significant mismatch
            tolerance_pct: Tolerance for position differences
        """
        self._local_positions = local_positions or {}
        self._fetch_broker_positions = fetch_broker_positions
        self._journal = journal
        self._fetch_broker_order_status = fetch_broker_order_status
        self._mismatch_action = mismatch_action
        self._tolerance_pct = tolerance_pct

    def reconcile(self) -> ReconciliationResult:
        """
        Perform position reconciliation.

        Returns:
            ReconciliationResult with any mismatches
        """
        result = ReconciliationResult()

        # Cannot reconcile without broker connection -> safe-halt on uncertainty
        if not self._fetch_broker_positions:
            result.success = False
            result.halted = True
            result.halt_reason = "No broker connection for reconciliation"
            return result

        # Fetch broker positions
        try:
            broker_positions = self._fetch_broker_positions()
        except Exception as e:
            result.success = False
            result.halted = True
            result.halt_reason = f"Failed to fetch broker positions: {e}"
            return result

        # Get all symbols
        all_symbols = set(self._local_positions.keys()) | set(broker_positions.keys())

        # Compare each symbol
        for symbol in all_symbols:
            local_qty = self._local_positions.get(symbol, Decimal("0"))
            broker_qty = broker_positions.get(symbol, Decimal("0"))
            difference = local_qty - broker_qty

            # Check if within tolerance
            if self._is_within_tolerance(local_qty, broker_qty):
                continue

            # Create mismatch record
            severity = "warning"
            if abs(difference) > abs(broker_qty) * Decimal("0.1"):
                severity = "error"

            mismatch = PositionMismatch(
                symbol=symbol,
                local_quantity=local_qty,
                broker_quantity=broker_qty,
                difference=difference,
                severity=severity,
            )
            result.mismatches.append(mismatch)

        # Handle mismatches
        if result.has_significant_mismatches:
            self._handle_mismatches(result)

        return result

    def reconcile_orders(self) -> ReconciliationResult:
        """
        Reconcile pending orders with broker.

        Returns:
            ReconciliationResult
        """
        result = ReconciliationResult()

        if not self._journal:
            return result

        unresolved = self._journal.get_unresolved_orders()
        if not unresolved:
            return result

        # If we have unresolved orders but no broker reconciliation capability,
        # we must safe-halt (restart uncertainty).
        if not self._fetch_broker_order_status:
            result.success = False
            result.halted = True
            sample_ids = ", ".join(e.client_order_id for e in unresolved[:5])
            more = "" if len(unresolved) <= 5 else f" (+{len(unresolved) - 5} more)"
            result.halt_reason = f"Unresolved orders present; cannot reconcile: {sample_ids}{more}"
            return result

        for entry in unresolved:
            try:
                broker_status = self._fetch_broker_order_status(entry)
            except Exception as e:
                # Exception while reconciling is uncertainty -> safe-halt
                result.success = False
                result.halted = True
                result.halt_reason = f"Order reconciliation failed for {entry.client_order_id}: {e}"
                return result

            if broker_status is None:
                self._journal.update_status(entry.entry_id, JournalStatus.UNKNOWN)
                result.success = False
                result.halted = True
                result.halt_reason = f"Order status unknown at broker for {entry.client_order_id}"
                return result

            if broker_status != entry.status:
                self._journal.update_status(
                    entry.entry_id, broker_status, broker_order_id=entry.broker_order_id
                )
                result.resolved.append(entry.client_order_id)

        # After reconciliation, if any unresolved remain, halt (still uncertain).
        remaining = self._journal.get_unresolved_orders()
        if remaining:
            result.success = False
            result.halted = True
            sample_ids = ", ".join(e.client_order_id for e in remaining[:5])
            more = "" if len(remaining) <= 5 else f" (+{len(remaining) - 5} more)"
            result.halt_reason = (
                f"Unresolved orders remain after reconciliation: {sample_ids}{more}"
            )
            return result

        return result

    def update_local_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update local position state."""
        self._local_positions = positions

    def get_local_position(self, symbol: str) -> Decimal:
        """Get local position for symbol."""
        return self._local_positions.get(symbol, Decimal("0"))

    def set_local_position(self, symbol: str, quantity: Decimal) -> None:
        """Set local position for symbol."""
        if quantity == Decimal("0"):
            self._local_positions.pop(symbol, None)
        else:
            self._local_positions[symbol] = quantity

    def _is_within_tolerance(
        self,
        local: Decimal,
        broker: Decimal,
    ) -> bool:
        """Check if difference is within tolerance."""
        if local == broker:
            return True

        if broker == Decimal("0"):
            return abs(local) < Decimal("0.0001")

        diff_pct = abs((local - broker) / broker)
        return diff_pct <= self._tolerance_pct

    def _handle_mismatches(self, result: ReconciliationResult) -> None:
        """Handle significant mismatches based on action policy."""
        if self._mismatch_action == MismatchAction.HALT:
            result.halted = True
            result.success = False
            symbols = [m.symbol for m in result.mismatches if m.is_significant]
            result.halt_reason = f"Significant position mismatch in: {', '.join(symbols)}"

        elif self._mismatch_action == MismatchAction.ACCEPT_BROKER:
            # Update local to match broker
            for mismatch in result.mismatches:
                self._local_positions[mismatch.symbol] = mismatch.broker_quantity
                result.resolved.append(mismatch.symbol)

        elif self._mismatch_action == MismatchAction.ALERT:
            # Just log, don't halt
            pass

        # ACCEPT_LOCAL - do nothing, keep local values
