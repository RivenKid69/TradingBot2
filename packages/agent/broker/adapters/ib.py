# -*- coding: utf-8 -*-
"""
packages/agent/broker/adapters/ib.py
=====================================

CCEA Agent broker connector for Interactive Brokers / CME futures (P2 #26).

Implements the full ``BrokerConnector`` protocol (via ``DelegatingConnector``) so IB
futures order flow goes through the Agent OMS (journaled, idempotent, risk-gated) —
previously only Alpaca + Binance had Agent connectors. Wraps the existing
``adapters/ib/order_execution.py`` into the normalized Backend on ``connect()``
(lazy; ib_insync is optional). A test/paper backend can be injected directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from packages.agent.broker.adapters._delegating import DelegatingConnector


class _IBBackend:
    """Normalize the IB order-execution adapter to the Backend interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        from adapters.ib.order_execution import IBOrderExecutionAdapter  # lazy (ib_insync)
        from adapters.models import ExchangeVendor

        self._a = IBOrderExecutionAdapter(vendor=ExchangeVendor.IB, config=config)
        try:
            from adapters.ib.market_data import IBMarketDataAdapter

            self._md = IBMarketDataAdapter(vendor=ExchangeVendor.IB, config=config)
        except Exception:
            self._md = None

    def connect(self) -> bool:
        """Connect and verify the real TWS/Gateway socket before activation."""
        if not bool(self._a.connect()):
            raise ConnectionError(
                getattr(self._a, "last_error", None) or "IB connect returned false"
            )
        connected = bool(getattr(self._a, "is_connected", False))
        raw = getattr(self._a, "_ib", None)
        if raw is not None and hasattr(raw, "isConnected"):
            connected = connected and bool(raw.isConnected())
        if not connected:
            self._a.disconnect()
            raise ConnectionError("IB adapter did not confirm an active TWS/Gateway session")
        return True

    def place(
        self,
        *,
        symbol,
        side,
        qty,
        order_type,
        limit_price=None,
        stop_price=None,
        client_order_id=None,
    ) -> Dict[str, Any]:
        try:
            if str(order_type).lower() == "limit" and limit_price is not None:
                r = self._a.submit_limit_order(symbol, side.upper(), qty, limit_price)
            else:
                r = self._a.submit_market_order(symbol, side.upper(), qty)
            return {
                "success": getattr(r, "success", True),
                "broker_order_id": str(getattr(r, "order_id", getattr(r, "broker_order_id", ""))),
                "status": getattr(r, "status", "submitted"),
                "filled_qty": float(getattr(r, "filled_qty", 0) or 0),
                "avg_price": getattr(r, "avg_fill_price", None),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "status": "error"}

    def cancel(self, broker_order_id) -> bool:
        try:
            return bool(self._a.cancel_order(broker_order_id))
        except Exception:
            return False

    def order(self, broker_order_id) -> Optional[Dict[str, Any]]:
        try:
            s = self._a.get_order_status(broker_order_id)
            return {
                "status": getattr(s, "status", str(s)),
                "filled_qty": float(getattr(s, "filled_qty", 0) or 0),
                "avg_price": getattr(s, "avg_fill_price", None),
            }
        except Exception:
            return None

    def positions(self) -> List[Dict[str, Any]]:
        out = []
        try:
            for p in self._a.get_positions() or []:
                out.append(
                    {
                        "symbol": getattr(p, "symbol", ""),
                        "qty": float(getattr(p, "qty", 0) or 0),
                        "avg_price": float(getattr(p, "avg_price", 0) or 0),
                        "market_value": getattr(p, "market_value", None),
                    }
                )
        except Exception:
            pass
        return out

    def account(self) -> Dict[str, Any]:
        try:
            a = self._a.get_account_info()
            return {
                "equity": float(getattr(a, "equity", 0) or 0),
                "cash": float(getattr(a, "cash", 0) or 0),
                "buying_power": float(getattr(a, "buying_power", 0) or 0),
            }
        except Exception:
            return {}

    def last_price(self, symbol) -> Optional[float]:
        if self._md is None:
            return None
        try:
            t = self._md.get_tick(symbol)
            px = getattr(t, "last", None) or getattr(t, "price", None)
            return float(px) if px is not None else None
        except Exception:
            return None

    def disconnect(self) -> None:
        try:
            if hasattr(self._a, "disconnect"):
                self._a.disconnect()
        except Exception:
            pass


class IBConnector(DelegatingConnector):
    _NAME = "ib"

    def __init__(
        self,
        credentials,
        *,
        sandbox: bool = True,
        config: Optional[Dict[str, Any]] = None,
        backend: Any = None,
        **kw,
    ) -> None:
        super().__init__(credentials, sandbox=sandbox, backend=backend, **kw)
        self._config = config or {
            "host": "127.0.0.1",
            "port": 7497 if sandbox else 7496,
            "client_id": 7,
        }

    def _build_backend(self) -> Any:
        return _IBBackend(self._config)


__all__ = ["IBConnector"]
