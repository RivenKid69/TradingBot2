# -*- coding: utf-8 -*-
"""Tests for tamper-evident books-and-records: hash chain, trade blotter, cash
ledger, and the BooksAndRecords facade (incl. live surveillance wiring)."""

from __future__ import annotations

from datetime import date

import pytest

from packages.agent.audit.hash_chain import HashChain, verify_chain, ChainRecord, chain_hash
from packages.agent.accounting.blotter import TradeBlotter, CashLedger, settlement_date
from packages.agent.accounting.books import BooksAndRecords


# --------------------------------------------------------------------------- hash chain
def test_hash_chain_valid():
    hc = HashChain(key=b"secret")
    hc.append({"a": 1}); hc.append({"b": 2}); hc.append({"c": 3})
    res = hc.verify()
    assert res["valid"] and res["n"] == 3 and res["broken_at"] is None


def test_hash_chain_detects_mutation():
    hc = HashChain(key=b"secret")
    hc.append({"a": 1}); hc.append({"b": 2}); hc.append({"c": 3})
    hc.records[1].payload["b"] = 999   # tamper with record 2
    res = hc.verify()
    assert not res["valid"] and res["broken_at"] == 2


def test_hash_chain_detects_deletion():
    hc = HashChain()
    for i in range(5):
        hc.append({"i": i})
    del hc.records[2]                   # delete a middle record
    res = hc.verify()
    assert not res["valid"]


def test_hash_chain_keyed_vs_unkeyed():
    payload = {"x": 1}
    h_keyed = chain_hash("0" * 64, payload, 1, key=b"k")
    h_plain = chain_hash("0" * 64, payload, 1, key=None)
    assert h_keyed != h_plain and len(h_keyed) == 64


# --------------------------------------------------------------------------- blotter
def test_blotter_records_and_verifies(tmp_path):
    bl = TradeBlotter(db_path=tmp_path / "bl.db", hmac_key=b"k")
    bl.record_trade(symbol="AAPL", side="buy", quantity=100, price=150, fee=1.0,
                    figi="BBG000B9XRY4", asset_class="equity")
    bl.record_trade(symbol="AAPL", side="sell", quantity=50, price=160, fee=0.5,
                    figi="BBG000B9XRY4", asset_class="equity")
    v = bl.verify()
    assert v["valid"] and v["n"] == 2 and v["keyed"] is True
    s = bl.summary()
    assert s["n_trades"] == 2 and s["total_fees"] == pytest.approx(1.5)


def test_blotter_tamper_detected_in_db(tmp_path):
    db = tmp_path / "bl.db"
    bl = TradeBlotter(db_path=db, hmac_key=b"k")
    bl.record_trade(symbol="BTCUSDT", side="buy", quantity=1, price=50000, asset_class="crypto")
    bl.record_trade(symbol="BTCUSDT", side="buy", quantity=1, price=51000, asset_class="crypto")
    bl.close()
    # tamper directly in SQLite (mutate price of the first row)
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE trades SET price='1.0' WHERE seq=1")
    conn.commit(); conn.close()
    bl2 = TradeBlotter(db_path=db, hmac_key=b"k")
    v = bl2.verify()
    assert not v["valid"] and v["broken_at"] == 1


def test_blotter_persistence(tmp_path):
    db = tmp_path / "bl.db"
    bl = TradeBlotter(db_path=db, hmac_key=b"k")
    bl.record_trade(symbol="ES", side="buy", quantity=2, price=4000, asset_class="future")
    bl.close()
    bl2 = TradeBlotter(db_path=db, hmac_key=b"k")
    assert bl2.summary()["n_trades"] == 1
    assert bl2.verify()["valid"]


# --------------------------------------------------------------------------- cash ledger
def test_cash_ledger_running_balance_and_verify(tmp_path):
    cl = CashLedger(db_path=tmp_path / "cl.db", opening_balance=100_000, hmac_key=b"k")
    cl.post("TRADE", -5000, symbol="BTC")     # buy
    cl.post("FEE", -10)
    cl.post("TRADE", +5500, symbol="BTC")     # sell
    assert cl.balance == pytest.approx(100_490)
    v = cl.verify()
    assert v["valid"] and v["n_movements"] if "n_movements" in v else v["valid"]
    s = cl.summary()
    assert s["balance"] == pytest.approx(100_490)
    assert s["by_type"]["FEE"] == pytest.approx(-10)


def test_cash_ledger_bad_type():
    cl = CashLedger(opening_balance=0)
    with pytest.raises(ValueError):
        cl.post("BOGUS", 1)


def test_cash_ledger_tamper_detected():
    cl = CashLedger(opening_balance=1000)
    cl.post("DEPOSIT", 500)
    cl.post("WITHDRAWAL", -200)
    cl._mem[0].amount = 999999      # tamper in-memory
    assert not cl.verify()["valid"]


def test_settlement_date_by_asset_class():
    d = date(2026, 1, 12)  # a Monday
    from datetime import datetime, timezone
    dt = datetime(2026, 1, 12, tzinfo=timezone.utc)
    assert settlement_date("crypto", dt) == "2026-01-12"   # T+0
    assert settlement_date("equity", dt) == "2026-01-13"   # T+1
    assert settlement_date("fx", dt) == "2026-01-14"       # T+2


# --------------------------------------------------------------------------- books facade
def test_books_on_fill_updates_all(tmp_path):
    b = BooksAndRecords(starting_cash=100_000, data_dir=tmp_path, hmac_key=b"k")
    out = b.on_fill(symbol="BTCUSDT", side="buy", quantity=0.1, price=50_000, fee=5)
    assert out["figi"] is not None                     # resolved via instrument master
    assert out["trade"]["figi"] == out["figi"]
    b.mark("BTCUSDT", 55_000)
    snap = b.snapshot()
    assert snap["pnl"]["unrealized_pnl"] == pytest.approx(500.0)
    assert snap["blotter"]["n_trades"] == 1
    # cash ledger mirrors pnl ledger cash exactly
    assert snap["integrity"]["cash_ledger_matches_pnl_cash"] is True
    assert snap["integrity"]["all_valid"] is True


def test_books_cash_consistency_across_trades(tmp_path):
    b = BooksAndRecords(starting_cash=100_000, data_dir=tmp_path)
    b.on_fill(symbol="AAPL", side="buy", quantity=100, price=150, fee=1)
    b.on_fill(symbol="AAPL", side="sell", quantity=100, price=160, fee=1)
    assert b.cash.balance == pytest.approx(float(b.ledger.cash))
    assert b.verify_integrity()["all_valid"] is True
    # realized = (160-150)*100 - fees handled separately
    assert float(b.ledger.realized_pnl) == pytest.approx(1000.0)


def test_books_surveillance_wash_trade(tmp_path):
    b = BooksAndRecords(starting_cash=1_000_000, data_dir=tmp_path)
    # near-simultaneous buy & sell at same price by same account -> wash alert
    b.on_fill(symbol="AAPL", side="buy", quantity=100, price=150, client_order_id="o1")
    res = b.on_fill(symbol="AAPL", side="sell", quantity=100, price=150, client_order_id="o2")
    patterns = [a["pattern"] for a in res["alerts"]]
    assert "wash_trade" in patterns


def test_books_surveillance_spoofing_via_on_order(tmp_path):
    b = BooksAndRecords(starting_cash=1_000_000, data_dir=tmp_path)
    # 3 large orders placed far from mid then cancelled fast, never filled -> spoofing
    alerts = []
    for i in range(3):
        b.on_order(symbol="ES", side="buy", action="NEW", quantity=2000, price=3950,
                   order_id=f"s{i}", mid=4000, ts_ms=1_000_000 + i * 100)
        alerts = b.on_order(symbol="ES", side="buy", action="CANCEL", quantity=2000, price=3950,
                            order_id=f"s{i}", mid=4000, ts_ms=1_000_000 + i * 100 + 50)
    assert any(a["pattern"] == "spoofing" for a in alerts)


def test_books_integrity_serializable(tmp_path):
    import json
    b = BooksAndRecords(starting_cash=100_000, data_dir=tmp_path, hmac_key=b"k")
    b.on_fill(symbol="ETHUSDT", side="buy", quantity=2, price=3000)
    blob = json.dumps(b.snapshot())
    assert "blotter" in blob and "cash_ledger" in blob and "integrity" in blob
