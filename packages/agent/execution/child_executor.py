# -*- coding: utf-8 -*-
"""
Clock-driven child-order executor - AGENT ZONE ONLY.

Closes P0 blocker #5: the portfolio scheduler (``service_xs_execution.py``) only
produced a *plan* (TWAP/VWAP/POV slice schedule + cost estimate). Nothing actually
released child orders on a clock, tracked partial fills per child, or did
cancel-replace on stragglers.

This module turns a parent order (symbol, side, total qty) + a slice schedule into
live child orders:

  * **parent <-> child graph** (``ParentOrder`` / ``ChildOrder``);
  * **clock-driven release**: ``step(now_ts)`` releases the next due slice through
    the real ``LiveExecutionEngine`` (one durable, idempotent order per child);
  * **partial-fill tracking** per child via ``on_child_fill`` (wired from
    ``FillHandler``), which rolls up to the parent (filled / remaining);
  * **cancel-replace** of stragglers: a working child that hasn't filled within its
    time budget is cancelled at the broker and its leaves are re-sliced into a new
    child, bounded by ``max_replaces``.

Deterministic and unit-testable: ``step(now_ts)`` takes an explicit clock; the real
``run`` loop is a thin wrapper. PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
from packages.agent.execution.engine import LiveExecutionEngine, OrderStatus

logger = logging.getLogger(__name__)


# Cancel a working order at the broker: (client_order_id, broker_order_id) -> success.
BrokerCancelFn = Callable[[str, Optional[str]], bool]


class ChildState:
    PENDING = "pending"      # scheduled, not yet released
    WORKING = "working"      # released to engine/broker, awaiting fills
    FILLED = "filled"
    CANCELLED = "cancelled"
    REPLACED = "replaced"    # cancelled and rolled into a replacement child
    REJECTED = "rejected"


@dataclass
class ChildOrder:
    child_id: str
    parent_id: str
    qty: Decimal
    release_at: float                       # scheduled release timestamp (epoch s)
    slice_index: int
    status: str = ChildState.PENDING
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    released_at: Optional[float] = None
    replaces: int = 0                       # how many times this lineage was replaced

    @property
    def is_terminal(self) -> bool:
        return self.status in (ChildState.FILLED, ChildState.CANCELLED, ChildState.REJECTED, ChildState.REPLACED)

    @property
    def leaves(self) -> Decimal:
        rem = self.qty - self.filled_qty
        return rem if rem > 0 else Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "slice_index": self.slice_index,
            "qty": str(self.qty),
            "filled_qty": str(self.filled_qty),
            "leaves": str(self.leaves),
            "status": self.status,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "avg_fill_price": str(self.avg_fill_price) if self.avg_fill_price is not None else None,
            "release_at": self.release_at,
            "released_at": self.released_at,
            "replaces": self.replaces,
        }


@dataclass
class ParentOrder:
    parent_id: str
    symbol: str
    side: str                               # "buy" | "sell"
    total_qty: Decimal
    strategy_id: str
    limit_price: Optional[Decimal] = None
    children: List[ChildOrder] = field(default_factory=list)
    created_at: float = 0.0

    @property
    def filled_qty(self) -> Decimal:
        return sum((c.filled_qty for c in self.children), Decimal("0"))

    @property
    def released_qty(self) -> Decimal:
        # qty that has been (or is being) worked and not rolled away
        return sum(
            (c.qty for c in self.children if c.status in (ChildState.WORKING, ChildState.FILLED)),
            Decimal("0"),
        )

    @property
    def remaining_qty(self) -> Decimal:
        rem = self.total_qty - self.filled_qty
        return rem if rem > 0 else Decimal("0")

    @property
    def is_complete(self) -> bool:
        if self.total_qty <= 0:
            return True
        if self.filled_qty >= self.total_qty:
            return True
        # complete when no child can still produce a fill
        return all(c.is_terminal for c in self.children) and not self._has_unreleased()

    def _has_unreleased(self) -> bool:
        return any(c.status == ChildState.PENDING for c in self.children)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "symbol": self.symbol,
            "side": self.side,
            "total_qty": str(self.total_qty),
            "filled_qty": str(self.filled_qty),
            "remaining_qty": str(self.remaining_qty),
            "is_complete": self.is_complete,
            "children": [c.to_dict() for c in self.children],
        }


def _normalized_weights(n: int, weights: Optional[List[float]]) -> List[float]:
    if weights:
        w = [max(0.0, float(x)) for x in weights]
        s = sum(w)
        if s > 0:
            return [x / s for x in w]
    n = max(1, int(n))
    return [1.0 / n] * n


class ClockDrivenChildExecutor:
    """Releases child slices on a clock through the real execution engine."""

    def __init__(
        self,
        engine: LiveExecutionEngine,
        *,
        prices_provider: Any = None,          # .get_prices() -> {symbol: price}
        broker_cancel: Optional[BrokerCancelFn] = None,
        strategy_id: str = "xs_cross_sectional",
        slice_interval_s: float = 30.0,
        straggler_timeout_s: float = 60.0,
        max_replaces: int = 3,
        min_child_qty: Decimal = Decimal("0"),
        use_limit_orders: bool = False,
    ) -> None:
        self._engine = engine
        self._prices_provider = prices_provider
        self._broker_cancel = broker_cancel
        self._strategy_id = strategy_id
        self._slice_interval_s = float(slice_interval_s)
        self._straggler_timeout_s = float(straggler_timeout_s)
        self._max_replaces = int(max_replaces)
        self._min_child_qty = Decimal(str(min_child_qty))
        self._use_limit_orders = bool(use_limit_orders)

        self._parents: Dict[str, ParentOrder] = {}
        self._child_by_coid: Dict[str, ChildOrder] = {}

    # ------------------------------------------------------------------
    def submit_parent(
        self,
        *,
        symbol: str,
        side: str,
        total_qty: Decimal,
        n_slices: int,
        start_ts: float,
        weights: Optional[List[float]] = None,
        interval_s: Optional[float] = None,
        limit_price: Optional[Decimal] = None,
        parent_id: Optional[str] = None,
    ) -> ParentOrder:
        """Register a parent order and build its pending child schedule."""
        total_qty = Decimal(str(total_qty))
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy/sell, got {side!r}")
        pid = parent_id or f"parent_{uuid4().hex[:12]}"
        parent = ParentOrder(
            parent_id=pid, symbol=symbol, side=side, total_qty=total_qty,
            strategy_id=self._strategy_id, limit_price=limit_price, created_at=float(start_ts),
        )
        if total_qty > 0:
            interval = float(interval_s) if interval_s is not None else self._slice_interval_s
            w = _normalized_weights(n_slices, weights)
            # qty per slice with remainder on the last to preserve exact total
            qtys: List[Decimal] = [total_qty * Decimal(str(x)) for x in w]
            assigned = sum(qtys[:-1], Decimal("0"))
            qtys[-1] = total_qty - assigned
            for i, q in enumerate(qtys):
                if q <= 0:
                    continue
                parent.children.append(ChildOrder(
                    child_id=f"{pid}_c{i}", parent_id=pid, qty=q,
                    release_at=float(start_ts) + i * interval, slice_index=i,
                ))
        self._parents[pid] = parent
        return parent

    # ------------------------------------------------------------------
    def step(self, now_ts: float) -> Dict[str, Any]:
        """Advance the clock: release due slices and cancel-replace stragglers.

        Returns a summary of actions taken this tick.
        """
        released: List[str] = []
        cancelled: List[str] = []
        replaced: List[str] = []
        errors: List[str] = []

        prices = {}
        if self._prices_provider is not None:
            try:
                prices = self._prices_provider.get_prices() or {}
            except Exception as exc:  # pragma: no cover
                logger.warning("prices_provider failed: %s", exc)

        for parent in self._parents.values():
            if parent.is_complete:
                continue

            # 1) cancel-replace stragglers (working children past their time budget)
            for child in list(parent.children):
                if child.status != ChildState.WORKING or child.released_at is None:
                    continue
                if (now_ts - child.released_at) < self._straggler_timeout_s:
                    continue
                leaves = child.leaves
                if leaves <= 0:
                    continue
                ok = self._cancel_child(child)
                if not ok:
                    continue
                cancelled.append(child.child_id)
                # roll leaves into a replacement child released immediately
                if child.replaces + 1 <= self._max_replaces and leaves > self._min_child_qty:
                    repl = ChildOrder(
                        child_id=f"{child.child_id}_r{child.replaces + 1}",
                        parent_id=parent.parent_id, qty=leaves,
                        release_at=now_ts, slice_index=child.slice_index,
                        replaces=child.replaces + 1,
                    )
                    parent.children.append(repl)
                    child.status = ChildState.REPLACED
                    replaced.append(repl.child_id)
                else:
                    child.status = ChildState.CANCELLED

            # 2) release due pending children
            price = prices.get(parent.symbol)
            for child in parent.children:
                if child.status != ChildState.PENDING:
                    continue
                if child.release_at > now_ts:
                    continue
                if child.qty <= self._min_child_qty:
                    child.status = ChildState.CANCELLED
                    continue
                coid = self._release_child(parent, child, price, now_ts)
                if coid is None:
                    errors.append(child.child_id)
                else:
                    released.append(coid)

        return {
            "ts": now_ts,
            "released": released,
            "cancelled": cancelled,
            "replaced": replaced,
            "errors": errors,
            "complete": self.all_complete(),
        }

    def _release_child(
        self, parent: ParentOrder, child: ChildOrder, price: Any, now_ts: float
    ) -> Optional[str]:
        side = IntentSide.LONG if parent.side == "buy" else IntentSide.SHORT
        if self._use_limit_orders and parent.limit_price is not None:
            intent_type = IntentType.LIMIT_ENTRY
            limit_price = parent.limit_price
        else:
            intent_type = IntentType.MARKET_ENTRY
            limit_price = None
        intent = OrderIntent(
            strategy_id=parent.strategy_id,
            symbol=parent.symbol,
            intent_type=intent_type,
            side=side,
            target_quantity=child.qty,
            limit_price=limit_price,
            time_in_force="DAY",
            reason=f"child slice {child.slice_index} of parent {parent.parent_id}",
            metadata={"parent_id": parent.parent_id, "child_id": child.child_id},
        )
        cur_price = None
        if price is not None:
            try:
                cur_price = Decimal(str(price))
            except Exception:
                cur_price = None
        result = self._engine.execute(intent, current_price=cur_price, origin="runner")
        if not result.success or result.order is None:
            child.status = ChildState.REJECTED
            logger.warning("child release failed (%s): %s", child.child_id, result.error_message)
            return None
        child.client_order_id = result.order.client_order_id
        child.broker_order_id = result.order.broker_order_id
        child.status = ChildState.WORKING
        child.released_at = now_ts
        self._child_by_coid[child.client_order_id] = child
        return child.client_order_id

    def _cancel_child(self, child: ChildOrder) -> bool:
        if self._broker_cancel is None or child.client_order_id is None:
            # No broker cancel wired: optimistically treat as cancellable in sim.
            return True
        try:
            return bool(self._broker_cancel(child.client_order_id, child.broker_order_id))
        except Exception as exc:  # pragma: no cover
            logger.warning("broker cancel failed for %s: %s", child.client_order_id, exc)
            return False

    # ------------------------------------------------------------------
    def on_child_fill(self, client_order_id: str, cum_filled: Decimal, avg_price: Optional[Decimal]) -> None:
        """Wire this as FillHandler(on_child_fill=...). Updates child + parent rollup."""
        child = self._child_by_coid.get(client_order_id)
        if child is None:
            return
        child.filled_qty = Decimal(str(cum_filled))
        if avg_price is not None:
            child.avg_fill_price = Decimal(str(avg_price))
        if child.filled_qty >= child.qty and child.status == ChildState.WORKING:
            child.status = ChildState.FILLED

    def on_order_terminal(self, client_order_id: str, status: OrderStatus) -> None:
        """Optional hook: mark a child terminal from a non-fill status (cancel/reject)."""
        child = self._child_by_coid.get(client_order_id)
        if child is None:
            return
        if status == OrderStatus.FILLED:
            child.status = ChildState.FILLED
        elif status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
            if child.status == ChildState.WORKING:
                child.status = ChildState.CANCELLED
        elif status in (OrderStatus.REJECTED, OrderStatus.ERROR):
            child.status = ChildState.REJECTED

    # ------------------------------------------------------------------
    def all_complete(self) -> bool:
        return all(p.is_complete for p in self._parents.values())

    def get_parent(self, parent_id: str) -> Optional[ParentOrder]:
        return self._parents.get(parent_id)

    def parents(self) -> List[ParentOrder]:
        return list(self._parents.values())

    def snapshot(self) -> Dict[str, Any]:
        return {pid: p.to_dict() for pid, p in self._parents.items()}


__all__ = [
    "ChildState",
    "ChildOrder",
    "ParentOrder",
    "ClockDrivenChildExecutor",
    "BrokerCancelFn",
]
