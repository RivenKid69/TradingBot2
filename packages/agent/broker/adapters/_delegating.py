# -*- coding: utf-8 -*-
"""
packages/agent/broker/adapters/_delegating.py
==============================================

Shared base for Agent broker connectors that delegate to an underlying adapter
(P2 #26). Implements the full ``BrokerConnector`` protocol by translating to/from a
small normalized ``Backend`` interface, so IB (futures) and OANDA (FX) order flow
can go through the CCEA Agent OMS just like Alpaca/Binance.

A ``Backend`` is duck-typed:
    place(symbol, side, qty, order_type, limit_price, stop_price, client_order_id) -> dict
    cancel(broker_order_id) -> bool
    positions() -> list[dict]            # {symbol, qty, avg_price, price, market_value}
    order(broker_order_id) -> dict|None  # {status, filled_qty, avg_price}
    account() -> dict                    # {equity, cash, buying_power}
    last_price(symbol) -> float|None

The real IB/OANDA adapters are wrapped into a Backend in each connector's
``_build_backend`` (lazy, SDK-optional). Tests inject a fake Backend.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from packages.agent.broker.protocol import (
    BaseBrokerConnector, BrokerCredentials, OrderRequest, OrderResult, OrderInfo,
    CancelResult, BulkCancelResult, Position, AccountInfo, OrderSide, OrderType,
    OrderStatus, PositionSide, TimeInForce,
)

_STATUS = {
    "filled": OrderStatus.FILLED, "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "partial": OrderStatus.PARTIALLY_FILLED, "submitted": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACCEPTED, "pending": OrderStatus.PENDING,
    "cancelled": OrderStatus.CANCELLED, "canceled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED, "expired": OrderStatus.EXPIRED, "error": OrderStatus.ERROR,
}


def _dec(x: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal(default)


class DelegatingConnector(BaseBrokerConnector):
    """Protocol-complete connector delegating to a normalized Backend."""

    _NAME = "delegating"

    def __init__(self, credentials: BrokerCredentials, *, sandbox: bool = False,
                 timeout_seconds: int = 30, backend: Any = None) -> None:
        super().__init__(credentials, sandbox=sandbox, timeout_seconds=timeout_seconds)
        self._backend = backend
        self._orders: Dict[str, OrderInfo] = {}   # broker_order_id -> info

    # -- subclasses build the real adapter-backed Backend here --
    def _build_backend(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def broker_name(self) -> str:
        return self._NAME

    def connect(self) -> bool:
        self._connect_error = None
        if self._backend is None:
            try:
                self._backend = self._build_backend()
            except Exception as exc:
                self._connected = False
                self._connect_error = str(exc)
                return False
        self._connected = self._backend is not None
        if self._connected and hasattr(self._backend, "connect"):
            try:
                self._connected = bool(self._backend.connect())
            except Exception as exc:
                self._connected = False
                self._connect_error = str(exc)
        if not self._connected and not getattr(self, "_connect_error", None):
            self._connect_error = "backend connection/health verification failed"
        if self._connected:
            self._last_heartbeat = datetime.utcnow()
        return self._connected

    def disconnect(self) -> None:
        self._connected = False
        try:
            if self._backend is not None and hasattr(self._backend, "disconnect"):
                self._backend.disconnect()
        except Exception:
            pass

    # -- orders -------------------------------------------------------------
    def submit_order(self, request: OrderRequest) -> OrderResult:
        if not self._connected:
            self.connect()
        if self._backend is None or not self._connected:
            return OrderResult(success=False, client_order_id=request.client_order_id,
                               status=OrderStatus.ERROR,
                               error_message=getattr(self, "_connect_error", None) or "not connected")
        res = self._backend.place(
            symbol=request.symbol,
            side=request.side.value if isinstance(request.side, OrderSide) else str(request.side),
            qty=float(request.quantity),
            order_type=request.order_type.value if isinstance(request.order_type, OrderType) else str(request.order_type),
            limit_price=float(request.limit_price) if request.limit_price is not None else None,
            stop_price=float(request.stop_price) if request.stop_price is not None else None,
            client_order_id=request.client_order_id,
        )
        ok = bool(res.get("success", True))
        bid = res.get("broker_order_id")
        info = OrderInfo(
            client_order_id=request.client_order_id, broker_order_id=bid,
            symbol=request.symbol, side=request.side, order_type=request.order_type,
            quantity=request.quantity, filled_quantity=_dec(res.get("filled_qty", 0)),
            limit_price=request.limit_price, stop_price=request.stop_price,
            avg_fill_price=_dec(res["avg_price"]) if res.get("avg_price") else None,
            status=_STATUS.get(str(res.get("status", "submitted")).lower(), OrderStatus.SUBMITTED),
            time_in_force=request.time_in_force, created_at=datetime.utcnow(),
        )
        if bid:
            self._orders[str(bid)] = info
        return OrderResult(
            success=ok, client_order_id=request.client_order_id, broker_order_id=bid,
            status=info.status, filled_quantity=info.filled_quantity,
            avg_fill_price=info.avg_fill_price, error_message=res.get("error"))

    def cancel_order(self, client_order_id: Optional[str] = None,
                     broker_order_id: Optional[str] = None) -> CancelResult:
        bid = broker_order_id or self._bid_for_client(client_order_id)
        if self._backend is None or not bid:
            return CancelResult(success=False, client_order_id=client_order_id or "",
                                error_message="not connected or unknown order")
        ok = bool(self._backend.cancel(bid))
        return CancelResult(success=ok, client_order_id=client_order_id or "",
                            broker_order_id=bid, cancelled_at=datetime.utcnow() if ok else None)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
        results = []
        for bid, info in list(self._orders.items()):
            if symbol and info.symbol != symbol:
                continue
            if info.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                results.append(self.cancel_order(broker_order_id=bid))
        ok = sum(1 for r in results if r.success)
        return BulkCancelResult(total_requested=len(results), total_cancelled=ok,
                                total_failed=len(results) - ok, results=results)

    def _bid_for_client(self, client_order_id: Optional[str]) -> Optional[str]:
        if not client_order_id:
            return None
        for bid, info in self._orders.items():
            if info.client_order_id == client_order_id:
                return bid
        return None

    def get_order(self, client_order_id: Optional[str] = None,
                  broker_order_id: Optional[str] = None) -> Optional[OrderInfo]:
        bid = broker_order_id or self._bid_for_client(client_order_id)
        if bid is None:
            return None
        info = self._orders.get(str(bid))
        if self._backend is not None:
            try:
                upd = self._backend.order(bid)
                if upd and info is not None:
                    info.status = _STATUS.get(str(upd.get("status", "")).lower(), info.status)
                    if upd.get("filled_qty") is not None:
                        info.filled_quantity = _dec(upd["filled_qty"])
                    if upd.get("avg_price"):
                        info.avg_fill_price = _dec(upd["avg_price"])
            except Exception:
                pass
        return info

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        return [self.get_order(broker_order_id=bid) for bid, info in self._orders.items()
                if (symbol is None or info.symbol == symbol)
                and info.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)]

    # -- positions ----------------------------------------------------------
    def _position_from_dict(self, d: Dict[str, Any]) -> Position:
        qty = _dec(d.get("qty", 0))
        side = PositionSide.LONG if qty > 0 else (PositionSide.SHORT if qty < 0 else PositionSide.FLAT)
        return Position(symbol=str(d.get("symbol", "")), side=side, quantity=qty,
                        avg_entry_price=_dec(d.get("avg_price", 0)),
                        current_price=_dec(d["price"]) if d.get("price") else None,
                        market_value=_dec(d["market_value"]) if d.get("market_value") else None)

    def get_positions(self) -> List[Position]:
        if self._backend is None:
            return []
        try:
            return [self._position_from_dict(p) for p in (self._backend.positions() or [])]
        except Exception:
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    def close_position(self, symbol: str, quantity: Optional[Decimal] = None) -> OrderResult:
        pos = self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            return OrderResult(success=False, client_order_id="", error_message="no position")
        q = abs(pos.quantity) if quantity is None else Decimal(str(quantity))
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        return self.submit_order(OrderRequest(
            client_order_id=f"close_{symbol}_{int(datetime.utcnow().timestamp())}",
            symbol=symbol, side=side, order_type=OrderType.MARKET, quantity=q))

    def close_all_positions(self) -> List[OrderResult]:
        return [self.close_position(p.symbol) for p in self.get_positions() if p.quantity != 0]

    # -- account / market data ---------------------------------------------
    def get_account(self) -> AccountInfo:
        if self._backend is None:
            return AccountInfo(account_id=self._NAME, equity=Decimal("0"),
                               cash=Decimal("0"), buying_power=Decimal("0"))
        a = self._backend.account() or {}
        return AccountInfo(account_id=str(a.get("account_id", self._NAME)),
                           equity=_dec(a.get("equity", 0)), cash=_dec(a.get("cash", 0)),
                           buying_power=_dec(a.get("buying_power", a.get("equity", 0))))

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        if self._backend is None:
            return None
        try:
            px = self._backend.last_price(symbol)
            return _dec(px) if px is not None else None
        except Exception:
            return None


__all__ = ["DelegatingConnector"]
