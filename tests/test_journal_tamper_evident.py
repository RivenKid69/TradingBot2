# -*- coding: utf-8 -*-
"""Tests for the tamper-evident audit chain added to OrderJournal."""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.agent.reconciliation.journal import OrderJournal, JournalStatus


def _log(j, coid="o1"):
    return j.log_order(client_order_id=coid, intent_id="i1", symbol="BTCUSDT",
                       side="buy", quantity=Decimal("0.1"), order_type="market",
                       metadata={"strategy_id": "s"})


def test_audit_chain_valid_after_lifecycle(tmp_path):
    j = OrderJournal(db_path=tmp_path / "j.db", hmac_key=b"k")
    e = _log(j)
    j.update_status(e.entry_id, JournalStatus.SUBMITTED, broker_order_id="B1")
    j.update_status(e.entry_id, JournalStatus.CONFIRMED, broker_order_id="B1",
                    filled_quantity=Decimal("0.1"), avg_price=Decimal("50000"))
    v = j.verify_audit_chain()
    assert v["valid"] and v["n"] == 3 and v["keyed"] is True
    evs = j.get_audit_events()
    assert len(evs) == 3
    assert {e["event_type"] for e in evs} == {"order_logged", "status_submitted", "status_confirmed"}


def test_audit_chain_detects_db_tamper(tmp_path):
    db = tmp_path / "j.db"
    j = OrderJournal(db_path=db, hmac_key=b"k")
    e = _log(j)
    j.update_status(e.entry_id, JournalStatus.SUBMITTED, broker_order_id="B1")
    # tamper: mutate an audit payload directly in SQLite
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE order_audit SET payload = '{\"event_type\":\"order_logged\",\"hacked\":true}' WHERE seq=1")
    conn.commit(); conn.close()
    j2 = OrderJournal(db_path=db, hmac_key=b"k")
    v = j2.verify_audit_chain()
    assert not v["valid"] and v["broken_at"] == 1


def test_audit_chain_unkeyed_still_evident(tmp_path):
    j = OrderJournal(db_path=tmp_path / "j.db")   # no key
    _log(j)
    v = j.verify_audit_chain()
    assert v["valid"] and v["keyed"] is False


def test_journal_normal_ops_unaffected(tmp_path):
    # existing behaviour must be intact (duplicate detection, lookups)
    j = OrderJournal(db_path=tmp_path / "j.db", hmac_key=b"k")
    e = _log(j, "dup1")
    assert j.is_duplicate("dup1") is True
    assert j.get_by_client_id("dup1").entry_id == e.entry_id
    assert j.update_status(e.entry_id, JournalStatus.SUBMITTED) is True
