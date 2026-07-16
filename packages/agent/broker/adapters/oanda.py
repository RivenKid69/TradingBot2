# -*- coding: utf-8 -*-
"""
packages/agent/broker/adapters/oanda.py
========================================

CCEA Agent broker connector for OANDA / FX (P2 #26).

Implements the ``BrokerConnector`` protocol (via ``DelegatingConnector``) so FX order
flow goes through the Agent OMS. Wraps ``adapters/oanda/order_execution.py`` into the
normalized Backend on ``connect()`` (lazy; uses ``requests``). A test backend can be
injected directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from packages.agent.broker.adapters._delegating import DelegatingConnector


class _OANDABackend:
    """Normalize the OANDA order-execution adapter to the Backend interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        from adapters.oanda.order_execution import OANDAOrderExecutionAdapter  # lazy
        from adapters.models import ExchangeVendor
        self._a = OANDAOrderExecutionAdapter(vendor=ExchangeVendor.OANDA, config=config)
        try:
            from adapters.oanda.market_data import OANDAMarketDataAdapter
            self._md = OANDAMarketDataAdapter(vendor=ExchangeVendor.OANDA, config=config)
        except Exception:
            self._md = None

    def connect(self) -> bool:
        """Verify credentials/account through the adapter before activation."""
        if hasattr(self._a, "connect") and not bool(self._a.connect()):
            raise ConnectionError(getattr(self._a, "last_error", None) or "OANDA connect returned false")
        if hasattr(self._a, "is_connected") and not bool(self._a.is_connected):
            raise ConnectionError("OANDA adapter did not confirm an active session")
        # Account lookup is the authenticated heartbeat for the REST adapter.
        self._a.get_account_info()
        return True

    def disconnect(self) -> None:
        if hasattr(self._a, "disconnect"):
            self._a.disconnect()

    def place(self, *, symbol, side, qty, order_type, limit_price=None, stop_price=None,
              client_order_id=None) -> Dict[str, Any]:
        try:
            from core_models import Order, Side, OrderType as CMOrderType, TimeInForce
            side_enum = Side.BUY if side.upper() == "BUY" else Side.SELL
            ot = CMOrderType.LIMIT if str(order_type).lower() == "limit" else CMOrderType.MARKET
            order = Order(symbol=symbol, side=side_enum, quantity=qty, order_type=ot,
                          limit_price=limit_price, time_in_force=TimeInForce.GTC,
                          client_order_id=client_order_id or "")
            r = self._a.submit_order(order)
            return {"success": getattr(r, "success", True),
                    "broker_order_id": str(getattr(r, "order_id", getattr(r, "broker_order_id", ""))),
                    "status": getattr(r, "status", "submitted"),
                    "filled_qty": float(getattr(r, "filled_qty", 0) or 0),
                    "avg_price": getattr(r, "avg_fill_price", getattr(r, "fill_price", None))}
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
            return {"status": getattr(s, "status", str(s)),
                    "filled_qty": float(getattr(s, "filled_qty", 0) or 0),
                    "avg_price": getattr(s, "avg_fill_price", None)}
        except Exception:
            return None

    def positions(self) -> List[Dict[str, Any]]:
        out = []
        try:
            for p in self._a.get_positions() or []:
                out.append({"symbol": getattr(p, "symbol", getattr(p, "instrument", "")),
                            "qty": float(getattr(p, "qty", getattr(p, "units", 0)) or 0),
                            "avg_price": float(getattr(p, "avg_price", 0) or 0),
                            "market_value": getattr(p, "market_value", None)})
        except Exception:
            pass
        return out

    def account(self) -> Dict[str, Any]:
        try:
            a = self._a.get_account_info()
            return {"equity": float(getattr(a, "equity", 0) or 0), "cash": float(getattr(a, "cash", 0) or 0),
                    "buying_power": float(getattr(a, "buying_power", getattr(a, "margin_available", 0)) or 0)}
        except Exception:
            return {}

    def last_price(self, symbol) -> Optional[float]:
        if self._md is None:
            return None
        try:
            t = self._md.get_tick(symbol)
            px = getattr(t, "last", None) or getattr(t, "mid", None) or getattr(t, "price", None)
            return float(px) if px is not None else None
        except Exception:
            return None


class OANDAConnector(DelegatingConnector):
    _NAME = "oanda"

    def __init__(self, credentials, *, sandbox: bool = True, config: Optional[Dict[str, Any]] = None,
                 backend: Any = None, **kw) -> None:
        super().__init__(credentials, sandbox=sandbox, backend=backend, **kw)
        cfg = dict(config or {})
        cfg.setdefault("practice", sandbox)
        if credentials is not None:
            cfg.setdefault("api_key", getattr(credentials, "api_key", ""))
            cfg.setdefault("account_id", getattr(credentials, "subaccount", "") or
                           getattr(credentials, "extra", {}).get("account_id", ""))
        self._config = cfg

    def _build_backend(self) -> Any:
        return _OANDABackend(self._config)


__all__ = ["OANDAConnector"]
