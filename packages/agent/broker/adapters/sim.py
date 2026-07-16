# -*- coding: utf-8 -*-
"""
SimBrokerConnector - in-process paper broker (AGENT ZONE).

A real-protocol ``BrokerConnector`` implementation that fills orders against a
last-price book without any external network/SDK. It lets the full live execution
stack (AgentClient -> LiveExecutionEngine -> broker -> fills -> OMS) run and be
exercised end-to-end in PAPER mode, honestly labeled as simulated.

Supports optional partial fills (``fill_ratio`` < 1) so child-order / cancel-replace
logic can be tested realistically. Maintains positions, orders, and an account so
``get_positions`` / ``get_order`` / ``get_account`` reconcile.

Implements the ``BrokerConnector`` protocol by duck typing (not inheriting the ABC,
to avoid pulling abstract members we don't need).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from packages.agent.broker.protocol import (
    OrderRequest, OrderResult, OrderInfo, CancelResult, BulkCancelResult,
    Position, AccountInfo, OrderSide, OrderType, OrderStatus, PositionSide,
    ConnectionStatus, TimeInForce,
)


class SimBrokerConnector:
    """Deterministic paper broker. Fills market orders at last price."""

    def __init__(
        self,
        prices: Optional[Dict[str, float]] = None,
        *,
        equity: float = 100_000.0,
        fill_ratio: float = 1.0,
        commission_bps: float = 0.0,
        broker_name: str = "sim_paper",
    ) -> None:
        self._prices: Dict[str, Decimal] = {
            str(k): Decimal(str(v)) for k, v in (prices or {}).items()
        }
        self._equity = Decimal(str(equity))
        self._cash = Decimal(str(equity))
        self._fill_ratio = Decimal(str(max(0.0, min(1.0, fill_ratio))))
        self._commission_bps = Decimal(str(commission_bps))
        self._broker_name = broker_name
        self._connected = True

        self._positions: Dict[str, Decimal] = {}        # symbol -> signed qty
        self._cost_basis: Dict[str, Decimal] = {}        # symbol -> avg entry
        self._orders: Dict[str, OrderInfo] = {}          # client_order_id -> info
        self._seq = 0

    # -- market data --------------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        self._prices[str(symbol)] = Decimal(str(price))

    def set_prices(self, prices: Dict[str, float]) -> None:
        for k, v in prices.items():
            self.set_price(k, v)

    def restore_state(
        self,
        *,
        cash: float,
        positions: List[Dict[str, Any]],
        sequence: int = 0,
    ) -> None:
        """Rehydrate the paper broker from the durable Agent ledger.

        The SimBroker itself is intentionally in-memory; the Agent ledger is the
        durable source of truth across desktop restarts.
        """
        self._cash = Decimal(str(cash))
        self._positions.clear()
        self._cost_basis.clear()
        for position in positions or []:
            symbol = str(position.get("symbol") or "")
            quantity = Decimal(str(position.get("quantity", position.get("qty", 0))))
            if not symbol or quantity == 0:
                continue
            avg_cost = Decimal(str(position.get("avg_cost", position.get("avg_entry", 0))))
            mark = Decimal(str(position.get("mark", position.get("price", avg_cost))))
            self._positions[symbol] = quantity
            self._cost_basis[symbol] = avg_cost
            self._prices[symbol] = mark
        self._seq = max(0, int(sequence))

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        return self._prices.get(str(symbol))

    # -- connection ---------------------------------------------------------
    @property
    def broker_name(self) -> str:
        return self._broker_name

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def connection_status(self) -> ConnectionStatus:
        return ConnectionStatus.CONNECTED if self._connected else ConnectionStatus.DISCONNECTED

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def health_check(self) -> Dict[str, Any]:
        return {"connected": self._connected, "broker": self._broker_name, "simulated": True}

    # -- orders -------------------------------------------------------------
    def submit_order(self, request: OrderRequest) -> OrderResult:
        # Idempotency: same client_order_id returns the prior result.
        if request.client_order_id in self._orders:
            existing = self._orders[request.client_order_id]
            return OrderResult(
                success=True, client_order_id=request.client_order_id,
                broker_order_id=existing.broker_order_id, status=existing.status,
                filled_quantity=existing.filled_quantity, avg_fill_price=existing.avg_fill_price,
            )

        price = self._prices.get(request.symbol)
        if price is None or price <= 0:
            return OrderResult(
                success=False, client_order_id=request.client_order_id,
                status=OrderStatus.REJECTED, error_message=f"no price for {request.symbol}",
            )

        self._seq += 1
        broker_id = f"SIM-{self._seq}"
        qty = Decimal(str(request.quantity))
        fill_qty = (qty * self._fill_ratio).quantize(Decimal("0.00000001"))
        fill_price = request.limit_price if (request.order_type == OrderType.LIMIT and request.limit_price) else price
        signed = fill_qty if request.side == OrderSide.BUY else -fill_qty

        # update position + cash with proper average-cost basis (not last-fill).
        prev = self._positions.get(request.symbol, Decimal("0"))
        new_pos = prev + signed
        prev_avg = self._cost_basis.get(request.symbol, fill_price)
        if prev == 0 or (prev > 0) == (signed > 0):
            # opening / increasing same direction -> weighted average
            self._cost_basis[request.symbol] = (
                (prev_avg * abs(prev) + fill_price * abs(signed)) / abs(new_pos)
                if new_pos != 0 else Decimal("0")
            )
        elif (prev > 0) != (new_pos > 0) and new_pos != 0:
            self._cost_basis[request.symbol] = fill_price  # flipped through zero
        elif new_pos == 0:
            self._cost_basis[request.symbol] = Decimal("0")
        # else: reducing same-side position keeps the existing average
        self._positions[request.symbol] = new_pos
        commission = abs(fill_qty * fill_price) * self._commission_bps / Decimal("10000")
        self._cash -= signed * fill_price + commission

        status = OrderStatus.FILLED if fill_qty >= qty else (
            OrderStatus.PARTIALLY_FILLED if fill_qty > 0 else OrderStatus.ACCEPTED)
        now = datetime.utcnow()
        info = OrderInfo(
            client_order_id=request.client_order_id, broker_order_id=broker_id,
            symbol=request.symbol, side=request.side, order_type=request.order_type,
            quantity=qty, filled_quantity=fill_qty, limit_price=request.limit_price,
            stop_price=request.stop_price, avg_fill_price=fill_price, status=status,
            time_in_force=request.time_in_force, commission=commission,
            created_at=now, updated_at=now, filled_at=now if status == OrderStatus.FILLED else None,
        )
        self._orders[request.client_order_id] = info
        return OrderResult(
            success=True, client_order_id=request.client_order_id, broker_order_id=broker_id,
            status=status, filled_quantity=fill_qty, avg_fill_price=fill_price, commission=commission,
        )

    def cancel_order(self, client_order_id: Optional[str] = None,
                     broker_order_id: Optional[str] = None) -> CancelResult:
        info = self._orders.get(client_order_id) if client_order_id else None
        if info is None:
            return CancelResult(success=False, client_order_id=client_order_id or "",
                                error_message="order not found")
        if info.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return CancelResult(success=False, client_order_id=info.client_order_id,
                                broker_order_id=info.broker_order_id,
                                error_message=f"cannot cancel {info.status.value}")
        info.status = OrderStatus.CANCELLED
        info.updated_at = datetime.utcnow()
        return CancelResult(success=True, client_order_id=info.client_order_id,
                            broker_order_id=info.broker_order_id, cancelled_at=info.updated_at)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
        results: List[CancelResult] = []
        for coid, info in self._orders.items():
            if symbol and info.symbol != symbol:
                continue
            if info.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                results.append(self.cancel_order(client_order_id=coid))
        ok = sum(1 for r in results if r.success)
        return BulkCancelResult(total_requested=len(results), total_cancelled=ok,
                                total_failed=len(results) - ok, results=results)

    def replace_order(self, client_order_id: str, *, quantity: Optional[Decimal] = None,
                      limit_price: Optional[Decimal] = None) -> OrderResult:
        """Amend a working order's qty/price (FIX 35=G semantics) — paper broker."""
        info = self._orders.get(client_order_id)
        if info is None:
            return OrderResult(success=False, client_order_id=client_order_id,
                               error_message="order not found")
        if info.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return OrderResult(success=False, client_order_id=client_order_id,
                               status=info.status, error_message=f"cannot amend {info.status.value}")
        if quantity is not None:
            info.quantity = Decimal(str(quantity))
        if limit_price is not None:
            info.limit_price = Decimal(str(limit_price))
        info.updated_at = datetime.utcnow()
        return OrderResult(success=True, client_order_id=client_order_id,
                           broker_order_id=info.broker_order_id, status=info.status,
                           filled_quantity=info.filled_quantity, avg_fill_price=info.avg_fill_price)

    def get_order(self, client_order_id: Optional[str] = None,
                  broker_order_id: Optional[str] = None) -> Optional[OrderInfo]:
        if client_order_id and client_order_id in self._orders:
            return self._orders[client_order_id]
        if broker_order_id:
            for info in self._orders.values():
                if info.broker_order_id == broker_order_id:
                    return info
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
            and (symbol is None or o.symbol == symbol)
        ]

    # -- positions ----------------------------------------------------------
    def _position_obj(self, symbol: str, qty: Decimal) -> Position:
        px = self._prices.get(symbol, Decimal("0"))
        side = PositionSide.LONG if qty > 0 else (PositionSide.SHORT if qty < 0 else PositionSide.FLAT)
        return Position(
            symbol=symbol, side=side, quantity=qty,
            avg_entry_price=self._cost_basis.get(symbol, px),
            current_price=px, market_value=qty * px,
            cost_basis=abs(qty) * self._cost_basis.get(symbol, px),
        )

    def get_positions(self) -> List[Position]:
        return [self._position_obj(s, q) for s, q in self._positions.items() if q != 0]

    def get_position(self, symbol: str) -> Optional[Position]:
        q = self._positions.get(symbol, Decimal("0"))
        return self._position_obj(symbol, q) if q != 0 else None

    def close_position(self, symbol: str, quantity: Optional[Decimal] = None) -> OrderResult:
        q = self._positions.get(symbol, Decimal("0"))
        if q == 0:
            return OrderResult(success=False, client_order_id="", error_message="no position")
        close_qty = abs(q) if quantity is None else Decimal(str(quantity))
        side = OrderSide.SELL if q > 0 else OrderSide.BUY
        return self.submit_order(OrderRequest(
            client_order_id=f"close_{symbol}_{self._seq + 1}", symbol=symbol, side=side,
            order_type=OrderType.MARKET, quantity=close_qty,
        ))

    def close_all_positions(self) -> List[OrderResult]:
        return [self.close_position(s) for s, q in list(self._positions.items()) if q != 0]

    # -- account ------------------------------------------------------------
    def get_account(self) -> AccountInfo:
        mv = sum((q * self._prices.get(s, Decimal("0")) for s, q in self._positions.items()), Decimal("0"))
        equity = self._cash + mv
        return AccountInfo(
            account_id="sim", equity=equity, cash=self._cash,
            buying_power=equity, currency="USD", status="active",
        )


__all__ = ["SimBrokerConnector"]
