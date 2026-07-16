# -*- coding: utf-8 -*-
"""Tests for the Agent-zone live P&L ledger (packages.agent.accounting.pnl_ledger).

Covers: accounting identity, average-cost & FIFO inventory, reductions, full close,
sign flips through zero, fees, financing, day-P&L, EOD NAV snapshots, SQLite
crash-recovery, PortfolioState bridge, and the FillHandler cumulative->increment
callback.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.agent.accounting.pnl_ledger import PnLLedger, ledger_fill_callback


def _identity_holds(led: PnLLedger) -> bool:
    # equity == starting_cash + realized + unrealized − fees − financing
    lhs = led.equity
    rhs = (Decimal(str(led._starting_cash)) + led.realized_pnl + led.unrealized_pnl
           - led._fees_cum - led._financing_cum)
    return abs(lhs - rhs) < Decimal("1e-9")


# --------------------------------------------------------------------------- basics
def test_open_long_unrealized_and_identity():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    led.mark("BTC", 55_000)
    assert float(led.unrealized_pnl) == pytest.approx(500.0)
    assert float(led.equity) == pytest.approx(100_500.0)
    assert float(led.realized_pnl) == 0.0
    assert _identity_holds(led)


def test_round_trip_realized():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    out = led.on_fill("BTC", "sell", 0.1, 55_000)
    assert out["realized_delta"] == pytest.approx(500.0)
    assert float(led.realized_pnl) == pytest.approx(500.0)
    assert float(led.equity) == pytest.approx(100_500.0)
    assert led.position("BTC").quantity == 0
    assert _identity_holds(led)


def test_partial_reduction_average_cost():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("AAPL", "buy", 100, 100)   # avg 100
    led.on_fill("AAPL", "buy", 100, 120)   # avg 110
    p = led.position("AAPL")
    assert float(p.avg_cost) == pytest.approx(110.0)
    out = led.on_fill("AAPL", "sell", 50, 130)  # realized (130-110)*50 = 1000
    assert out["realized_delta"] == pytest.approx(1000.0)
    assert float(p.quantity) == pytest.approx(150.0)
    assert float(p.avg_cost) == pytest.approx(110.0)  # avg unchanged on reduce
    assert _identity_holds(led)


def test_short_position_realized():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("XOM", "sell", 100, 100)   # open short @100
    led.mark("XOM", 90)
    assert float(led.unrealized_pnl) == pytest.approx(1000.0)  # short gains as price falls
    out = led.on_fill("XOM", "buy", 100, 90)  # cover: realized (100-90)*100 = 1000
    assert out["realized_delta"] == pytest.approx(1000.0)
    assert led.position("XOM").quantity == 0
    assert _identity_holds(led)


def test_sign_flip_through_zero():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("ES", "buy", 10, 4000)            # long 10 @4000
    out = led.on_fill("ES", "sell", 15, 4100)      # close 10 (+1000), flip short 5 @4100
    assert out["realized_delta"] == pytest.approx(1000.0)
    p = led.position("ES")
    assert float(p.quantity) == pytest.approx(-5.0)
    assert float(p.avg_cost) == pytest.approx(4100.0)  # new short basis
    assert _identity_holds(led)


def test_fees_reduce_equity():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000, fee=10)
    led.mark("BTC", 50_000)
    assert float(led.equity) == pytest.approx(99_990.0)  # -10 fee
    assert float(led._fees_cum) == pytest.approx(10.0)
    assert _identity_holds(led)


def test_financing_accrual():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("EURUSD", "buy", 100_000, 1.0)
    led.mark("EURUSD", 1.0)
    led.accrue_financing("EURUSD", 25)  # 25$ carry cost
    assert float(led._financing_cum) == pytest.approx(25.0)
    assert float(led.equity) == pytest.approx(100_000 - 25)
    assert _identity_holds(led)


# --------------------------------------------------------------------------- FIFO
def test_fifo_realized_uses_oldest_lots():
    led = PnLLedger(starting_cash=100_000, method="fifo")
    led.on_fill("AAPL", "buy", 100, 100)
    led.on_fill("AAPL", "buy", 100, 120)
    # sell 150 @130: closes 100@100 (+3000) then 50@120 (+500) = 3500
    out = led.on_fill("AAPL", "sell", 150, 130)
    assert out["realized_delta"] == pytest.approx(3500.0)
    p = led.position("AAPL")
    assert float(p.quantity) == pytest.approx(50.0)
    assert float(p.avg_cost) == pytest.approx(120.0)  # remaining lot @120
    assert _identity_holds(led)


# --------------------------------------------------------------------------- day / EOD
def test_day_pnl_and_eod_close():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    led.mark("BTC", 55_000)
    assert float(led.day_pnl) == pytest.approx(500.0)
    snap = led.eod_close()
    assert snap.nav == pytest.approx(100_500.0)
    assert snap.day_pnl == pytest.approx(500.0)
    # after close, day resets; mark moves +500 more
    led.mark("BTC", 60_000)
    assert float(led.day_pnl) == pytest.approx(500.0)  # since last close
    assert float(led.equity) == pytest.approx(101_000.0)


# --------------------------------------------------------------------------- persistence
def test_crash_recovery_replays_fills(tmp_path):
    db = tmp_path / "pnl.db"
    led = PnLLedger(starting_cash=100_000, db_path=db)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    led.on_fill("ETH", "buy", 2, 3_000)
    led.mark("BTC", 55_000)
    led.mark("ETH", 3_100)
    led.eod_close()
    led.close()

    # reopen: state rebuilt from persisted fills
    led2 = PnLLedger(starting_cash=100_000, db_path=db)
    assert float(led2.position("BTC").quantity) == pytest.approx(0.1)
    assert float(led2.position("ETH").quantity) == pytest.approx(2.0)
    assert float(led2.position("BTC").mark) == pytest.approx(55_000.0)
    assert float(led2.position("ETH").mark) == pytest.approx(3_100.0)
    assert float(led2.equity) == pytest.approx(100_700.0)
    assert float(led2.unrealized_pnl) == pytest.approx(700.0)
    assert float(led2.day_pnl) == pytest.approx(0.0)
    assert len(led2.nav_history()) == 1
    assert _identity_holds(led2)


def test_starting_cash_persisted(tmp_path):
    db = tmp_path / "pnl2.db"
    led = PnLLedger(starting_cash=250_000, db_path=db)
    led.on_fill("BTC", "buy", 1, 40_000)
    led.close()
    # even if reopened with a different starting_cash arg, persisted value wins
    led2 = PnLLedger(starting_cash=999, db_path=db)
    assert float(led2._starting_cash) == pytest.approx(250_000.0)


# --------------------------------------------------------------------------- bridges
def test_to_portfolio_state():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    led.mark("BTC", 55_000)
    ps = led.to_portfolio_state()
    assert float(ps.equity) == pytest.approx(100_500.0)
    assert float(ps.get_position("BTC")) == pytest.approx(0.1)


def test_fillhandler_callback_increments():
    led = PnLLedger(starting_cash=100_000)
    cb = ledger_fill_callback(led)
    # cumulative partial fills for one order
    cb({"client_order_id": "o1", "symbol": "BTC", "side": "buy",
        "filled_qty": "0.05", "avg_fill_price": "50000"})
    cb({"client_order_id": "o1", "symbol": "BTC", "side": "buy",
        "filled_qty": "0.10", "avg_fill_price": "50000"})  # increment 0.05
    assert float(led.position("BTC").quantity) == pytest.approx(0.1)
    # a duplicate cumulative event applies no further increment
    cb({"client_order_id": "o1", "symbol": "BTC", "side": "buy",
        "filled_qty": "0.10", "avg_fill_price": "50000"})
    assert float(led.position("BTC").quantity) == pytest.approx(0.1)


def test_reconcile_against_broker():
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    rec = led.reconcile_against({"BTC": 0.1})
    assert rec["reconciled"] is True
    rec2 = led.reconcile_against({"BTC": 0.2})
    assert rec2["reconciled"] is False and rec2["breaks"][0]["symbol"] == "BTC"


def test_snapshot_serializable():
    import json
    led = PnLLedger(starting_cash=100_000)
    led.on_fill("BTC", "buy", 0.1, 50_000)
    led.mark("BTC", 55_000)
    blob = json.dumps(led.snapshot())
    assert "unrealized_pnl" in blob and "day_pnl" in blob
