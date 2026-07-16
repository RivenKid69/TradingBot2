# -*- coding: utf-8 -*-
"""Tests for the Binance **Spot** order-execution adapter (P0-C closure).

Before this, ``ExchangeVendor.BINANCE`` had no ``ORDER_EXECUTION`` adapter, so
``create_order_execution_adapter(ExchangeVendor.BINANCE, ...)`` raised and the
MVP crypto panic/holdings/close paths reported "unavailable". These tests lock
in that the spot adapter is registered and behaves correctly, with the HTTP
transport mocked (no network).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor, MarketType
from adapters.registry import create_order_execution_adapter
from adapters.binance.order_execution import BinanceOrderExecutionAdapter
from core_models import Order, OrderType, Side, TimeInForce, ExecStatus


# --------------------------------------------------------------------------- fixtures

def _make_adapter(**cfg):
    adapter = BinanceOrderExecutionAdapter(config={"api_key": "k", "api_secret": "s", **cfg})
    calls = []

    def fake_request(method, endpoint, params=None, signed=True):
        calls.append((method, endpoint, dict(params or {})))
        if endpoint == "/api/v3/order" and method == "POST":
            qty = params.get("quantity", "0")
            return {
                "orderId": 555, "clientOrderId": params.get("newClientOrderId", "x"),
                "status": "FILLED", "executedQty": qty,
                "cummulativeQuoteQty": str(float(qty) * 50000.0),
                "type": params.get("type"), "side": params.get("side"),
            }
        if endpoint == "/api/v3/order" and method == "DELETE":
            return {"orderId": 555, "status": "CANCELED"}
        if endpoint == "/api/v3/order" and method == "GET":
            return {"orderId": 555, "clientOrderId": "x", "symbol": "BTCUSDT",
                    "side": "BUY", "type": "LIMIT", "status": "FILLED",
                    "executedQty": "0.01", "cummulativeQuoteQty": "500", "price": "50000"}
        if endpoint == "/api/v3/account":
            return {"accountType": "SPOT", "balances": [
                {"asset": "BTC", "free": "0.5", "locked": "0.0"},
                {"asset": "ETH", "free": "2.0", "locked": "1.0"},
                {"asset": "USDT", "free": "10000", "locked": "0"},
                {"asset": "BNB", "free": "0", "locked": "0"},   # zero → skipped
            ]}
        if endpoint == "/api/v3/openOrders":
            if method == "GET":
                return [{"symbol": "BTCUSDT", "side": "SELL", "type": "LIMIT",
                         "origQty": "0.1", "price": "60000", "status": "NEW",
                         "orderId": 9, "clientOrderId": "c9", "timeInForce": "GTC"}]
            return {}
        return {}

    adapter._request = fake_request
    adapter._calls = calls
    return adapter


# --------------------------------------------------------------------------- registry

def test_spot_execution_adapter_is_registered():
    a = create_order_execution_adapter(ExchangeVendor.BINANCE, {"api_key": "k", "api_secret": "s"})
    assert isinstance(a, BinanceOrderExecutionAdapter)
    assert isinstance(a, OrderExecutionAdapter)
    assert a.market_type == MarketType.CRYPTO_SPOT


def test_spot_execution_adapter_registered_for_binance_us():
    a = create_order_execution_adapter(ExchangeVendor.BINANCE_US, {"api_key": "k", "api_secret": "s"})
    assert isinstance(a, BinanceOrderExecutionAdapter)
    assert a._spot_url == "https://api.binance.us"


def test_futures_adapter_still_resolves_separately():
    # No regression: the futures execution factory is unaffected.
    from adapters.registry import create_futures_order_execution_adapter
    fa = create_futures_order_execution_adapter(ExchangeVendor.BINANCE_FUTURES,
                                                {"api_key": "k", "api_secret": "s"})
    assert fa.market_type == MarketType.CRYPTO_FUTURES


# --------------------------------------------------------------------------- submit

def test_submit_market_order_parses_fill():
    a = _make_adapter()
    o = Order(ts=1, symbol="BTCUSDT", side=Side.BUY, order_type=OrderType.MARKET,
              quantity=Decimal("0.01"))
    r = a.submit_order(o)
    assert r.success and r.order_id == "555"
    assert r.filled_qty == Decimal("0.01")
    assert r.filled_price == Decimal("50000")  # cummulativeQuote / executedQty
    method, endpoint, params = a._calls[-1]
    assert (method, endpoint) == ("POST", "/api/v3/order")
    assert params["type"] == "MARKET" and params["side"] == "BUY"
    assert "reduceOnly" not in params and "positionSide" not in params  # spot


def test_submit_limit_order_sets_price_and_tif():
    a = _make_adapter()
    o = Order(ts=1, symbol="ETHUSDT", side=Side.SELL, order_type=OrderType.LIMIT,
              quantity=Decimal("2"), price=Decimal("3500"), time_in_force=TimeInForce.IOC)
    a.submit_order(o)
    params = a._calls[-1][2]
    assert params["type"] == "LIMIT" and params["price"] == "3500"
    assert params["timeInForce"] == "IOC"


def test_submit_order_error_code_is_failure():
    a = _make_adapter()
    a._request = lambda *args, **kw: {"code": -2010, "msg": "insufficient balance"}
    o = Order(ts=1, symbol="BTCUSDT", side=Side.BUY, order_type=OrderType.MARKET,
              quantity=Decimal("999"))
    r = a.submit_order(o)
    assert not r.success and r.error_code == "-2010"
    assert "insufficient" in r.error_message


def test_submit_spot_order_stop_limit():
    a = _make_adapter()
    a.submit_spot_order("BTCUSDT", "SELL", "STOP_LOSS_LIMIT", Decimal("0.1"),
                        price=Decimal("48000"), stop_price=Decimal("48500"),
                        time_in_force="GTC")
    params = a._calls[-1][2]
    assert params["type"] == "STOP_LOSS_LIMIT"
    assert params["stopPrice"] == "48500" and params["price"] == "48000"
    assert params["timeInForce"] == "GTC"


# --------------------------------------------------------------------------- cancel

def test_cancel_requires_symbol():
    a = _make_adapter()
    assert a.cancel_order(order_id="555") is False          # no symbol → refused
    assert a.cancel_order(order_id="555", symbol="BTCUSDT") is True


def test_cancel_all_orders_enumerates_symbols():
    a = _make_adapter()
    n = a.cancel_all_orders()   # no symbol → enumerate open orders, cancel per symbol
    assert n == 1
    # A DELETE on the openOrders endpoint was issued for the open order's symbol.
    assert any(m == "DELETE" and ep == "/api/v3/openOrders" and p.get("symbol") == "BTCUSDT"
               for m, ep, p in a._calls)


# --------------------------------------------------------------------------- positions

def test_positions_from_balances_map_to_pairs():
    a = _make_adapter()
    pos = a.get_positions()
    # BTC + ETH surfaced as *USDT; USDT (quote) and zero-balance BNB excluded.
    assert set(pos) == {"BTCUSDT", "ETHUSDT"}
    assert pos["ETHUSDT"].qty == Decimal("3.0")           # free + locked
    assert pos["BTCUSDT"].avg_entry_price == Decimal("0")  # spot: no cost basis
    assert pos["BTCUSDT"].meta["market"] == "spot"


def test_positions_custom_quote_asset():
    a = _make_adapter(quote_asset="USDC")
    pos = a.get_positions()
    assert set(pos) == {"BTCUSDC", "ETHUSDC"}


def test_get_positions_filter_by_symbol():
    a = _make_adapter()
    pos = a.get_positions(["BTCUSDT"])
    assert set(pos) == {"BTCUSDT"}


# --------------------------------------------------------------------------- close / flatten

def test_close_position_market_sells_full_balance():
    a = _make_adapter()
    r = a.close_position("BTCUSDT")
    assert r.success
    params = a._calls[-1][2]
    assert params["side"] == "SELL" and params["type"] == "MARKET"
    assert params["quantity"] == "0.5"


def test_close_position_noop_when_flat():
    a = _make_adapter()
    r = a.close_position("XRPUSDT")   # no XRP balance
    assert r.success and r.status == "NO_POSITION"


# --------------------------------------------------------------------------- status / account

def test_get_order_status_returns_exec_report():
    a = _make_adapter()
    rep = a.get_order_status(order_id="555", symbol="BTCUSDT")
    assert rep is not None
    assert rep.exec_status == ExecStatus.FILLED
    assert rep.symbol == "BTCUSDT"


def test_get_account_info_quote_cash():
    a = _make_adapter()
    info = a.get_account_info()
    assert info.cash_balance == Decimal("10000")
    assert info.margin_enabled is False


# --------------------------------------------------------------------------- app wiring (P0-C)
# Proves the MVP panic path now flattens crypto-spot through the registry
# adapter instead of the old fail-closed "no adapter" short-circuit. The
# broker adapter is stubbed so no network is touched.

def test_panic_halt_crypto_spot_flattens(monkeypatch):
    import os
    os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-spot")
    os.environ.setdefault("RIVEN_ENABLE_CCEA", "0")
    from fastapi.testclient import TestClient
    import app as app_module
    from app import api
    import adapters.registry as registry

    client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

    # Real-looking (non-placeholder) crypto creds + no CCEA supervisor.
    monkeypatch.setenv("BINANCE_API_KEY", "AKIASPOT000000000000")
    monkeypatch.setenv("BINANCE_API_SECRET", "spotsecret000000000000")
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    monkeypatch.setattr(app_module, "ACTIVE_ASSET", "crypto", raising=False)

    class _Pos:
        def __init__(self, qty, entry):
            self.qty = Decimal(str(qty)); self.avg_entry_price = Decimal(str(entry))

    class _StubAdapter:
        def __init__(self):
            self._pos = {"BTCUSDT": _Pos("0.5", "50000")}
            self.submitted = []
        def cancel_all_orders(self, symbol=None):
            return 0
        def get_positions(self, symbols=None):
            return dict(self._pos)
        def submit_order(self, order):
            self._pos.pop(order.symbol, None)   # flattened
            self.submitted.append(order.symbol)
            return OrderResult(success=True, order_id="1", status="FILLED")

    stub = _StubAdapter()

    def fake_create(vendor, config=None):
        assert str(getattr(vendor, "value", vendor)) == "binance"
        return stub
    monkeypatch.setattr(registry, "create_order_execution_adapter", fake_create)

    try:
        res = client.post("/api/panic_halt")
        assert res.status_code == 200, res.text
        data = res.json()
        # Wired to the real adapter path (NOT the old fail-closed short-circuit).
        assert data["execution_mode"] == "live_broker"
        assert data["status"] in ("success", "partial")
        assert data["kill_switch_tripped"] is True
        syms = [p["symbol"] for p in data["positions_liquidated"]]
        assert "BTCUSDT" in syms
        assert "нет order-execution" not in data["detail"]
        assert stub.submitted == ["BTCUSDT"]   # a market SELL was actually sent
    finally:
        client.post("/api/panic_reset")
