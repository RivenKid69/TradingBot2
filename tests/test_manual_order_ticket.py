"""Тесты ручного ордер-тикета трейдера (§5.27–28 gap-анализа).

Полноценный ордер (market/limit/stop/stop_limit, TIF, reduce-only), частичное
закрытие позиции, список рабочих ордеров и отмена — всё через НАСТОЯЩИЙ Agent OMS
(policy firewall → hash-chain журнал → fill → books), не мок.

Supervisor поднимается один раз на модуль (paper/SimBroker, ~5с boot).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-manual-order")
os.environ.setdefault("RIVEN_ENABLE_CCEA", "1")

from fastapi.testclient import TestClient

import app as app_module
from app import api

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})


@pytest.fixture(scope="module")
def sup():
    from ccea.desktop_supervisor import CCEASupervisor, SupervisorConfig

    s = CCEASupervisor(
        SupervisorConfig(data_dir=Path(tempfile.mkdtemp(prefix="ccea_order_test_")), paper=True)
    )
    s.start()
    # SimBroker нужна котировка для расчёта fill.
    s._broker.set_price("BTCUSDT", 50_000.0)
    s._broker.set_price("ETHUSDT", 2_000.0)
    yield s
    s.stop()


def _flatten(sup):
    # Закрыть все позиции до полной плоскости (bounded).
    for _ in range(5):
        holdings = sup.portfolio_snapshot().get("holdings") or []
        if not holdings:
            return
        for h in holdings:
            if abs(float(h.get("qty", 0))) > 1e-9:
                sup.close_position(h["symbol"])


@pytest.fixture(autouse=True)
def _flat(sup):
    _flatten(sup)
    yield
    _flatten(sup)


# ------------------------------------------------------------ market order


def test_market_buy_fills_and_books(sup):
    r = sup.submit_manual_order(symbol="BTCUSDT", side="buy", order_type="market", quantity=0.1)
    assert r["ok"] is True and r["state"] == "filled"
    assert r["order_type"] == "market" and r["side"] == "long"
    pos = sup.portfolio_snapshot()["holdings"]
    assert any(h["symbol"] == "BTCUSDT" and abs(h["qty"] - 0.1) < 1e-9 for h in pos)


def test_market_sell_opens_short(sup):
    r = sup.submit_manual_order(symbol="ETHUSDT", side="sell", order_type="market", quantity=1.0)
    assert r["ok"] is True and r["side"] == "short"
    pos = {h["symbol"]: h for h in sup.portfolio_snapshot()["holdings"]}
    assert pos["ETHUSDT"]["qty"] < 0


# ------------------------------------------------------------ limit / stop


def test_limit_order_carries_limit_price(sup):
    r = sup.submit_manual_order(
        symbol="BTCUSDT",
        side="buy",
        order_type="limit",
        quantity=0.1,
        limit_price=49_000.0,
        time_in_force="GTC",
    )
    assert r["ok"] is True and r["order_type"] == "limit"
    assert r["limit_price"] == 49_000.0


def test_stop_order_carries_stop_price(sup):
    r = sup.submit_manual_order(
        symbol="BTCUSDT", side="buy", order_type="stop", quantity=0.1, stop_price=51_000.0
    )
    assert r["ok"] is True and r["order_type"] == "stop" and r["stop_price"] == 51_000.0


def test_stop_limit_requires_both_prices(sup):
    assert (
        sup.submit_manual_order(
            symbol="BTCUSDT", side="buy", order_type="stop_limit", quantity=0.1, stop_price=51_000.0
        )["ok"]
        is False
    )  # нет limit
    r = sup.submit_manual_order(
        symbol="BTCUSDT",
        side="buy",
        order_type="stop_limit",
        quantity=0.1,
        stop_price=51_000.0,
        limit_price=51_100.0,
    )
    assert r["ok"] is True


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "kw,err",
    [
        (dict(symbol="BTCUSDT", side="up", order_type="market", quantity=0.1), "side"),
        (dict(symbol="BTCUSDT", side="buy", order_type="twap", quantity=0.1), "order_type"),
        (dict(symbol="BTCUSDT", side="buy", order_type="market", quantity=0), "quantity"),
        (dict(symbol="BTCUSDT", side="buy", order_type="limit", quantity=0.1), "limit_price"),
        (dict(symbol="BTCUSDT", side="buy", order_type="stop", quantity=0.1), "stop_price"),
        (
            dict(
                symbol="BTCUSDT", side="buy", order_type="market", quantity=0.1, time_in_force="XXX"
            ),
            "time_in_force",
        ),
    ],
)
def test_validation_rejections(sup, kw, err):
    r = sup.submit_manual_order(**kw)
    assert r["ok"] is False and err in r["error"]


# ------------------------------------------------------------ reduce-only


def test_reduce_only_without_position_rejected(sup):
    r = sup.submit_manual_order(
        symbol="BTCUSDT", side="sell", order_type="market", quantity=0.1, reduce_only=True
    )
    assert r["ok"] is False and "позиции нет" in r["error"]


def test_reduce_only_wrong_side_rejected(sup):
    sup.submit_manual_order(symbol="BTCUSDT", side="buy", order_type="market", quantity=0.2)
    # позиция LONG; reduce-only BUY (та же сторона) не уменьшает
    r = sup.submit_manual_order(
        symbol="BTCUSDT", side="buy", order_type="market", quantity=0.1, reduce_only=True
    )
    assert r["ok"] is False and "не уменьшает" in r["error"]


def test_reduce_only_caps_at_position_size(sup):
    sup.submit_manual_order(symbol="BTCUSDT", side="buy", order_type="market", quantity=0.2)
    # reduce-only SELL 0.5 при позиции 0.2 → ужимается до 0.2, не переворачивает
    r = sup.submit_manual_order(
        symbol="BTCUSDT", side="sell", order_type="market", quantity=0.5, reduce_only=True
    )
    assert r["ok"] is True and abs(r["quantity"] - 0.2) < 1e-9
    pos = {h["symbol"]: h for h in sup.portfolio_snapshot()["holdings"]}
    assert "BTCUSDT" not in pos or abs(pos["BTCUSDT"]["qty"]) < 1e-9  # плоская


# ------------------------------------------------------------ partial close


def test_partial_close(sup):
    sup.submit_manual_order(symbol="BTCUSDT", side="buy", order_type="market", quantity=0.4)
    r = sup.close_position("BTCUSDT", quantity=0.1)
    assert r["ok"] is True and r["partial"] is True
    assert abs(r["remaining_qty"] - 0.3) < 1e-9


def test_full_close_when_quantity_exceeds(sup):
    sup._broker.set_price("SOLUSDT", 100.0)  # изолированный символ
    sup.submit_manual_order(symbol="SOLUSDT", side="buy", order_type="market", quantity=2.0)
    r = sup.close_position("SOLUSDT", quantity=99.0)  # больше позиции → закрыть целиком
    assert r["ok"] is True and r["partial"] is False
    assert abs(r["remaining_qty"]) < 1e-9


def test_close_no_position(sup):
    r = sup.close_position("XRPUSDT")
    assert r["ok"] is False and "no active position" in r["error"]


# ------------------------------------------------------------ open orders / cancel


def test_open_orders_shape(sup):
    oo = sup.open_orders()
    assert oo["ok"] is True and isinstance(oo["orders"], list)
    assert oo["simulated"] is True


def test_cancel_unknown_order(sup):
    r = sup.cancel_order("does-not-exist")
    # SimBroker вернёт неуспех для несуществующего id.
    assert r["ok"] is False


# ------------------------------------------------------------------- REST


def _wire(monkeypatch, sup):
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", sup, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "running", raising=False)
    import services.ops_kill_switch as oks

    monkeypatch.setattr(oks, "tripped", lambda: False)


def test_api_order_submit_requires_confirm(monkeypatch, sup):
    _wire(monkeypatch, sup)
    res = client.post(
        "/api/ccea/order/submit",
        json={"symbol": "BTCUSDT", "side": "buy", "order_type": "market", "quantity": 0.1},
    )
    assert res.status_code == 409 and "confirm" in res.json()["detail"]


def test_api_order_submit_and_open_orders(monkeypatch, sup):
    _wire(monkeypatch, sup)
    res = client.post(
        "/api/ccea/order/submit",
        json={
            "symbol": "ETHUSDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.5,
            "limit_price": 1990.0,
            "confirm": True,
        },
    )
    assert res.status_code == 200 and res.json()["ok"] is True
    oo = client.get("/api/ccea/open_orders")
    assert oo.status_code == 200 and oo.json()["ok"] is True


def test_api_order_blocked_by_kill_switch(monkeypatch, sup):
    _wire(monkeypatch, sup)
    import services.ops_kill_switch as oks

    monkeypatch.setattr(oks, "tripped", lambda: True)
    res = client.post(
        "/api/ccea/order/submit",
        json={
            "symbol": "BTCUSDT",
            "side": "buy",
            "order_type": "market",
            "quantity": 0.1,
            "confirm": True,
        },
    )
    assert res.status_code == 400 and "Kill switch" in res.json()["detail"]


def test_api_partial_close(monkeypatch, sup):
    _wire(monkeypatch, sup)
    sup._broker.set_price("ADAUSDT", 100.0)  # изолированный символ
    sup.submit_manual_order(symbol="ADAUSDT", side="buy", order_type="market", quantity=3.0)
    res = client.post("/api/portfolio/close", json={"symbol": "ADAUSDT", "quantity": 1.0})
    assert res.status_code == 200
    body = res.json()
    assert body["partial"] is True and abs(body["remaining_qty"] - 2.0) < 1e-9


def test_api_order_validation_400(monkeypatch, sup):
    _wire(monkeypatch, sup)
    res = client.post(
        "/api/ccea/order/submit",
        json={
            "symbol": "BTCUSDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.1,
            "confirm": True,
        },
    )  # limit без цены
    assert res.status_code == 400 and "limit_price" in res.json()["detail"]


def test_api_503_without_ccea(monkeypatch):
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "stopped", raising=False)
    import services.ops_kill_switch as oks

    monkeypatch.setattr(oks, "tripped", lambda: False)
    assert (
        client.post(
            "/api/ccea/order/submit",
            json={
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_type": "market",
                "quantity": 0.1,
                "confirm": True,
            },
        ).status_code
        == 503
    )
    assert client.get("/api/ccea/open_orders").status_code == 503
