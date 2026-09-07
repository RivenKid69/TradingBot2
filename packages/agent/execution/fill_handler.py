# -*- coding: utf-8 -*-
"""
Fill stream consumer - AGENT ZONE ONLY.

Closes P0 blocker #4: there was no consumer of broker fill / execution-report
events, so the OMS order lifecycle (NEW -> PARTIALLY_FILLED -> FILLED) was never
advanced by real fills and ``fills_count`` was incremented at *submission* time,
not when a fill actually happened.

This module:
  * defines a broker-agnostic ``FillEvent``;
  * provides ``FillHandler`` which consumes those events and drives the real
    ``LiveExecutionEngine`` OMS lifecycle, computing **cumulative filled qty,
    leaves (remaining), and notional-weighted average fill price**;
  * provides pluggable fill *sources*:
      - ``InMemoryFillSource`` (tests / replay),
      - ``PollingFillSource`` (diff broker order state into events),
      - ``CallbackFillSource`` (real websocket: broker pushes -> queue).

The handler is fully deterministic and unit-testable (no network, no sleeps).
A real broker websocket (e.g. Alpaca ``TradingStream`` ``trade_updates``) wires in
by translating its payloads into ``FillEvent`` and feeding ``CallbackFillSource``.

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from packages.agent.execution.engine import LiveExecutionEngine, Order, OrderStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------
def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


# Canonical broker event kinds -> internal handling.
_EVENT_NEW = "new"
_EVENT_ACCEPTED = "accepted"
_EVENT_PARTIAL = "partial_fill"
_EVENT_FILL = "fill"
_EVENT_CANCELED = "canceled"
_EVENT_REJECTED = "rejected"
_EVENT_EXPIRED = "expired"
_EVENT_REPLACED = "replaced"

# Normalize the many spellings brokers use into the canonical kinds above.
_EVENT_ALIASES: Dict[str, str] = {
    "new": _EVENT_NEW,
    "pending_new": _EVENT_NEW,
    "accepted": _EVENT_ACCEPTED,
    "open": _EVENT_ACCEPTED,
    "partially_filled": _EVENT_PARTIAL,
    "partial_fill": _EVENT_PARTIAL,
    "partial": _EVENT_PARTIAL,
    "fill": _EVENT_FILL,
    "filled": _EVENT_FILL,
    "done_for_day": _EVENT_FILL,
    "canceled": _EVENT_CANCELED,
    "cancelled": _EVENT_CANCELED,
    "pending_cancel": _EVENT_ACCEPTED,
    "rejected": _EVENT_REJECTED,
    "error": _EVENT_REJECTED,
    "expired": _EVENT_EXPIRED,
    "replaced": _EVENT_REPLACED,
}


@dataclass(frozen=True)
class FillEvent:
    """A single broker execution / order-status event (broker-agnostic).

    ``filled_qty`` is interpreted as the **cumulative** filled quantity for the
    order when ``cumulative=True`` (the default, matching most REST/WS feeds that
    report running totals). When a feed reports per-execution increments instead,
    set ``cumulative=False`` and the handler accumulates ``last_fill_qty``.
    """

    client_order_id: str
    event_type: str
    filled_qty: Optional[Decimal] = None
    avg_fill_price: Optional[Decimal] = None
    last_fill_qty: Optional[Decimal] = None
    last_fill_price: Optional[Decimal] = None
    broker_order_id: Optional[str] = None
    cumulative: bool = True
    ts: datetime = field(default_factory=datetime.utcnow)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return _EVENT_ALIASES.get(str(self.event_type).lower(), str(self.event_type).lower())

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "FillEvent":
        """Build from a generic broker payload dict.

        Recognized keys (any subset): client_order_id/client_id, event/status,
        filled_qty/cum_qty, avg_price/filled_avg_price, last_qty, last_price,
        broker_order_id/order_id.
        """
        coid = (
            payload.get("client_order_id")
            or payload.get("client_id")
            or payload.get("clientOrderId")
            or ""
        )
        ev = payload.get("event") or payload.get("status") or payload.get("type") or ""
        return cls(
            client_order_id=str(coid),
            event_type=str(ev),
            filled_qty=_to_decimal(payload.get("filled_qty", payload.get("cum_qty"))),
            avg_fill_price=_to_decimal(
                payload.get(
                    "avg_fill_price", payload.get("filled_avg_price", payload.get("avg_price"))
                )
            ),
            last_fill_qty=_to_decimal(payload.get("last_fill_qty", payload.get("last_qty"))),
            last_fill_price=_to_decimal(payload.get("last_fill_price", payload.get("last_price"))),
            broker_order_id=(
                str(payload["broker_order_id"])
                if payload.get("broker_order_id")
                else (str(payload["order_id"]) if payload.get("order_id") else None)
            ),
            cumulative=bool(payload.get("cumulative", True)),
            raw=dict(payload),
        )


@dataclass
class _FillState:
    """Internal cumulative tracking per client_order_id."""

    cum_qty: Decimal = Decimal("0")
    cum_notional: Decimal = Decimal("0")  # Σ price*qty for VWAP avg
    avg_price: Optional[Decimal] = None
    terminal: bool = False


# ---------------------------------------------------------------------------
# Fill sources
# ---------------------------------------------------------------------------
@runtime_checkable
class FillSource(Protocol):
    """A source of FillEvents. ``poll`` returns any events available now."""

    def poll(self) -> List[FillEvent]: ...


class InMemoryFillSource:
    """Deterministic source for tests / replay: events drained FIFO on poll()."""

    def __init__(self, events: Optional[List[FillEvent]] = None, *, batch: int = 0) -> None:
        self._events: List[FillEvent] = list(events or [])
        self._batch = int(batch)  # 0 = drain all on each poll

    def push(self, event: FillEvent) -> None:
        self._events.append(event)

    def poll(self) -> List[FillEvent]:
        if not self._events:
            return []
        if self._batch <= 0:
            out, self._events = self._events, []
            return out
        out = self._events[: self._batch]
        self._events = self._events[self._batch :]
        return out


class CallbackFillSource:
    """Thread-safe queue fed by a real broker websocket callback.

    Wire a broker stream like::

        src = CallbackFillSource()
        trading_stream.subscribe_trade_updates(lambda u: src.push(FillEvent.from_payload(_translate(u))))

    The consumer thread calls ``poll()`` to drain.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[FillEvent]" = queue.Queue()

    def push(self, event: FillEvent) -> None:
        self._q.put(event)

    def push_payload(self, payload: Dict[str, Any]) -> None:
        self._q.put(FillEvent.from_payload(payload))

    def poll(self) -> List[FillEvent]:
        out: List[FillEvent] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


# Fetch full order info from broker by client_order_id -> dict (or None).
FetchOrderFn = Callable[[str], Optional[Dict[str, Any]]]


class PollingFillSource:
    """Polls broker order state for tracked client_order_ids and emits a FillEvent
    only when the (status, cum_qty) changes — a diff-based execution feed for
    brokers without a push stream.
    """

    def __init__(self, fetch_order: FetchOrderFn) -> None:
        self._fetch = fetch_order
        self._tracked: Dict[str, Any] = {}  # client_order_id -> last (status, cum_qty)

    def track(self, client_order_id: str) -> None:
        self._tracked.setdefault(client_order_id, None)

    def untrack(self, client_order_id: str) -> None:
        self._tracked.pop(client_order_id, None)

    def poll(self) -> List[FillEvent]:
        out: List[FillEvent] = []
        for coid in list(self._tracked.keys()):
            try:
                info = self._fetch(coid)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("PollingFillSource fetch failed for %s: %s", coid, exc)
                continue
            if not info:
                continue
            status = str(info.get("status", info.get("event", ""))).lower()
            cum = _to_decimal(info.get("filled_qty", info.get("cum_qty")), Decimal("0"))
            signature = (status, str(cum))
            if self._tracked.get(coid) == signature:
                continue
            self._tracked[coid] = signature
            ev = FillEvent.from_payload({**info, "client_order_id": coid, "cumulative": True})
            out.append(ev)
            if (
                ev.kind in (_EVENT_FILL, _EVENT_CANCELED, _EVENT_REJECTED, _EVENT_EXPIRED)
                and cum is not None
            ):
                # stop tracking terminal orders after they're reported once
                self.untrack(coid)
        return out


# ---------------------------------------------------------------------------
# Fill handler (drives OMS)
# ---------------------------------------------------------------------------
# Callback signature for downstream notification of a processed fill.
FillCallback = Callable[[Dict[str, Any]], None]


class FillHandler:
    """Consumes ``FillEvent``s and advances the OMS lifecycle on the engine.

    Responsibilities:
      * map broker event kind -> ``OrderStatus``;
      * maintain cumulative filled qty + notional-weighted avg fill price;
      * decide PARTIALLY_FILLED vs FILLED by comparing cumulative fill to the
        order's quantity (leaves = quantity - filled);
      * call ``engine.update_order_status(...)`` so the durable journal advances;
      * fire an optional ``on_fill`` callback with a normalized dict.
    """

    _KIND_TO_STATUS = {
        _EVENT_NEW: OrderStatus.SUBMITTED,
        _EVENT_ACCEPTED: OrderStatus.ACCEPTED,
        _EVENT_REPLACED: OrderStatus.ACCEPTED,
        _EVENT_PARTIAL: OrderStatus.PARTIALLY_FILLED,
        _EVENT_FILL: OrderStatus.FILLED,
        _EVENT_CANCELED: OrderStatus.CANCELLED,
        _EVENT_REJECTED: OrderStatus.REJECTED,
        _EVENT_EXPIRED: OrderStatus.EXPIRED,
    }

    def __init__(
        self,
        engine: LiveExecutionEngine,
        *,
        on_fill: Optional[FillCallback] = None,
        on_child_fill: Optional[Callable[[str, Decimal, Optional[Decimal]], None]] = None,
    ) -> None:
        self._engine = engine
        self._on_fill = on_fill
        # Notifies a child executor of incremental fills: (client_order_id, cum_filled, avg_price)
        self._on_child_fill = on_child_fill
        self._state: Dict[str, _FillState] = {}
        self._lock = threading.RLock()

    # -- queries ------------------------------------------------------------
    def filled_quantity(self, client_order_id: str) -> Decimal:
        st = self._state.get(client_order_id)
        return st.cum_qty if st else Decimal("0")

    def leaves(self, client_order_id: str) -> Decimal:
        order = self._engine.get_order_by_client_id(client_order_id)
        if order is None:
            return Decimal("0")
        rem = Decimal(order.quantity) - self.filled_quantity(client_order_id)
        return rem if rem > 0 else Decimal("0")

    def avg_fill_price(self, client_order_id: str) -> Optional[Decimal]:
        st = self._state.get(client_order_id)
        return st.avg_price if st else None

    # -- core ---------------------------------------------------------------
    def handle_event(self, ev: FillEvent) -> Optional[Order]:
        with self._lock:
            return self._handle_event_locked(ev)

    def _handle_event_locked(self, ev: FillEvent) -> Optional[Order]:
        order = self._engine.get_order_by_client_id(ev.client_order_id)
        if order is None:
            logger.debug("FillHandler: event for unknown order %s (ignored)", ev.client_order_id)
            return None

        st = self._state.setdefault(ev.client_order_id, _FillState())
        if st.terminal:
            return order  # ignore late events after a terminal state

        kind = ev.kind
        inc = Decimal("0")  # incremental fill qty this event (for P&L ledger)

        # --- update cumulative fill bookkeeping ---
        if kind in (_EVENT_PARTIAL, _EVENT_FILL):
            if ev.cumulative and ev.filled_qty is not None:
                new_cum = ev.filled_qty
                inc = new_cum - st.cum_qty
            elif ev.last_fill_qty is not None:
                inc = ev.last_fill_qty
                new_cum = st.cum_qty + inc
            elif ev.filled_qty is not None:
                # treat as cumulative if that's all we got
                new_cum = ev.filled_qty
                inc = new_cum - st.cum_qty
            else:
                # a fill event with no qty: assume fully filled to order quantity
                new_cum = Decimal(order.quantity)
                inc = new_cum - st.cum_qty

            if inc < 0:
                inc = Decimal("0")
                new_cum = st.cum_qty

            # average price: prefer broker-reported avg; else accumulate VWAP
            price = ev.avg_fill_price
            if price is not None:
                st.cum_qty = new_cum
                st.avg_price = price
                st.cum_notional = price * new_cum
            else:
                fill_price = ev.last_fill_price
                if fill_price is not None and inc > 0:
                    st.cum_notional += fill_price * inc
                st.cum_qty = new_cum
                st.avg_price = (st.cum_notional / st.cum_qty) if st.cum_qty > 0 else st.avg_price

        # --- decide final status (partial vs full) ---
        status = self._KIND_TO_STATUS.get(kind)
        if status is None:
            logger.debug(
                "FillHandler: unmapped event kind %r for %s", ev.event_type, ev.client_order_id
            )
            return order

        order_qty = Decimal(order.quantity)
        if kind == _EVENT_FILL and order_qty > 0 and st.cum_qty < order_qty:
            # Broker said 'fill' but cumulative < ordered (e.g. done_for_day): treat
            # as terminal-partial only if cum>0, else keep accepted.
            status = OrderStatus.PARTIALLY_FILLED if st.cum_qty > 0 else OrderStatus.ACCEPTED
        if kind in (_EVENT_PARTIAL,) and order_qty > 0 and st.cum_qty >= order_qty:
            status = OrderStatus.FILLED  # cumulative reached full on a 'partial' event

        is_terminal = status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )
        st.terminal = is_terminal

        updated = self._engine.update_order_status(
            client_order_id=ev.client_order_id,
            status=status,
            filled_quantity=st.cum_qty if st.cum_qty > 0 else None,
            avg_fill_price=st.avg_price,
            broker_order_id=ev.broker_order_id,
            error_message=(str(ev.raw.get("error")) if ev.raw.get("error") else None),
        )

        leaves = (order_qty - st.cum_qty) if order_qty > st.cum_qty else Decimal("0")

        if self._on_child_fill is not None and kind in (_EVENT_PARTIAL, _EVENT_FILL):
            try:
                self._on_child_fill(ev.client_order_id, st.cum_qty, st.avg_price)
            except Exception as exc:  # pragma: no cover - never let a callback break the loop
                logger.warning("on_child_fill callback failed: %s", exc)

        if self._on_fill is not None:
            try:
                self._on_fill(
                    {
                        "client_order_id": ev.client_order_id,
                        "broker_order_id": ev.broker_order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "status": status.value,
                        "filled_qty": str(st.cum_qty),
                        "fill_increment": str(inc),  # incremental qty this event (P&L ledger)
                        "leaves_qty": str(leaves),
                        "avg_fill_price": str(st.avg_price) if st.avg_price is not None else None,
                        "ts": ev.ts.isoformat(),
                    }
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("on_fill callback failed: %s", exc)

        return updated

    def consume(self, source: FillSource, *, max_batches: int = 1) -> int:
        """Drain ``source`` up to ``max_batches`` polls; returns events handled.

        Returns the number of events processed. Used by a runner loop or tests;
        for real-time, call repeatedly (the source blocks/buffers as needed).
        """
        handled = 0
        for _ in range(max(1, int(max_batches))):
            events = source.poll()
            if not events:
                break
            for ev in events:
                if self.handle_event(ev) is not None:
                    handled += 1
        return handled


__all__ = [
    "FillEvent",
    "FillHandler",
    "FillSource",
    "InMemoryFillSource",
    "CallbackFillSource",
    "PollingFillSource",
    "FetchOrderFn",
]
