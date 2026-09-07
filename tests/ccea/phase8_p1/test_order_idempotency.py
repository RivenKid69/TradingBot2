# -*- coding: utf-8 -*-

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from packages.agent.execution.engine import LiveExecutionEngine
from packages.agent.reconciliation.journal import OrderJournal
from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide


def test_deterministic_client_order_id_and_persistent_dedup(tmp_path):
    journal_path = tmp_path / "order_journal.db"
    journal = OrderJournal(db_path=journal_path)

    submit_calls = {"count": 0}

    def broker_submit(order):
        submit_calls["count"] += 1
        return (True, "broker-1", None)

    intent_id = UUID("12345678-1234-5678-1234-567812345678")
    intent = OrderIntent(
        strategy_id="strat1",
        symbol="BTCUSDT",
        intent_type=IntentType.MARKET_ENTRY,
        side=IntentSide.LONG,
        target_quantity=Decimal("0.1"),
        intent_id=intent_id,
    )

    engine1 = LiveExecutionEngine(
        broker_submit=broker_submit, broker_name="demo", order_journal=journal
    )
    r1 = engine1.execute(intent)
    assert r1.success is True
    assert r1.order is not None
    cid1 = r1.order.client_order_id
    assert submit_calls["count"] == 1

    r2 = engine1.execute(intent)
    assert r2.success is True
    assert r2.order is not None
    assert r2.order.client_order_id == cid1
    assert submit_calls["count"] == 1  # no second submission

    submit_calls2 = {"count": 0}

    def broker_submit2(order):
        submit_calls2["count"] += 1
        return (True, "broker-2", None)

    # New engine (simulated restart) with same journal: must not resubmit
    engine2 = LiveExecutionEngine(
        broker_submit=broker_submit2,
        broker_name="demo",
        order_journal=OrderJournal(db_path=journal_path),
    )
    r3 = engine2.execute(intent)
    assert r3.success is True
    assert r3.order is not None
    assert r3.order.client_order_id == cid1
    assert submit_calls2["count"] == 0
