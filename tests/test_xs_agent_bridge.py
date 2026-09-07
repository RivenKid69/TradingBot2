# -*- coding: utf-8 -*-
"""
Tests for the XS -> Agent -> broker bridge (P0 blockers #1, #4, #5).

Covers:
  * AgentClient.send_intents: target weights -> deltas -> real OrderIntents ->
    LiveExecutionEngine (immediate market mode), incl. liquidating dropped names.
  * FillHandler: OMS lifecycle NEW -> PARTIALLY_FILLED -> FILLED with cumulative
    filled qty, leaves, and notional-weighted average fill price.
  * Idempotency: re-sending the same IntentBatch is de-duplicated by the journal.
  * ClockDrivenChildExecutor: sliced release over a clock (parent<->child graph).
  * Cancel-replace of a straggler child + leaves roll-forward.
  * CrossSectionalLiveRunner wired with a real AgentClient (no longer dry-run).
"""

import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from packages.agent.execution.engine import LiveExecutionEngine, OrderStatus
from packages.agent.reconciliation.journal import OrderJournal


def _engine(submit):
    """Engine with an ISOLATED on-disk journal (default journal persists to ~/.ccea
    and would make idempotency tests non-hermetic across runs)."""
    jpath = Path(tempfile.mkdtemp(prefix="oms_test_")) / "journal.db"
    return LiveExecutionEngine(
        broker_submit=submit, broker_name="paper", order_journal=OrderJournal(db_path=jpath)
    )


from packages.agent.execution.agent_client import AgentClient
from packages.agent.execution.fill_handler import (
    FillHandler,
    FillEvent,
    InMemoryFillSource,
    PollingFillSource,
)
from packages.agent.execution.child_executor import ClockDrivenChildExecutor, ChildState
from service_xs_live import Intent, IntentBatch, CrossSectionalLiveRunner


class _Prices:
    def __init__(self, px):
        self._px = px

    def get_prices(self):
        return dict(self._px)


class _Positions:
    def __init__(self, pos):
        self._pos = pos

    def get_positions(self):
        return dict(self._pos)


def _recording_broker():
    sent = []

    def submit(order):
        sent.append(order)
        return True, f"BRK-{len(sent)}", None

    return submit, sent


def _diversified_weights():
    # 5 names @ 0.15 each = 0.75 gross, each under the 25% concentration limit
    return pd.Series({"AAA": 0.15, "BBB": 0.15, "CCC": 0.15, "DDD": 0.15, "EEE": 0.15})


def _prices_for(weights):
    return _Prices({s: 100.0 for s in weights.index} | {"OLD": 50.0})


# ---------------------------------------------------------------------------
def test_immediate_rebalance_creates_orders_and_liquidates_dropped():
    submit, sent = _recording_broker()
    eng = _engine(submit)
    w = _diversified_weights()
    ac = AgentClient(
        eng,
        prices_provider=_prices_for(w),
        position_provider=_Positions({"OLD": 5000.0}),  # held but not in target -> liquidate
        min_trade_notional=1.0,
    )
    batch = IntentBatch(
        ts_ms=1,
        equity=100_000.0,
        idempotency_key="b1",
        intents=[Intent(s, float(v), float(v) * 100_000.0) for s, v in w.items()],
    )
    res = ac.send_intents(batch)
    assert len(res.errors) == 0, res.errors
    # 5 buys + 1 liquidation
    assert len(res.submitted) == 6
    by_symbol = {o.symbol: o for o in eng._orders_by_client_id.values()}
    assert by_symbol["OLD"].side == "sell"
    assert by_symbol["AAA"].side == "buy"
    # qty = notional/price = 15000/100 = 150
    assert by_symbol["AAA"].quantity == Decimal("150")
    # OLD liquidation qty = 5000/50 = 100
    assert by_symbol["OLD"].quantity == Decimal("100")


def test_fill_lifecycle_partial_then_full_with_avg_and_leaves():
    submit, _ = _recording_broker()
    eng = _engine(submit)
    w = _diversified_weights()
    src = InMemoryFillSource()
    fh = FillHandler(eng)
    ac = AgentClient(
        eng,
        prices_provider=_prices_for(w),
        position_provider=_Positions({}),
        fill_handler=fh,
        fill_source=src,
    )
    batch = IntentBatch(
        ts_ms=1, equity=100_000.0, idempotency_key="b2", intents=[Intent("AAA", 0.15, 15_000.0)]
    )
    ac.send_intents(batch)
    order = eng.get_order_by_client_id(next(iter(eng._orders_by_client_id)))
    coid = order.client_order_id
    assert order.quantity == Decimal("150")

    # partial fill 60 @ 100.0
    src.push(
        FillEvent(
            client_order_id=coid,
            event_type="partial_fill",
            filled_qty=Decimal("60"),
            avg_fill_price=Decimal("100.0"),
        )
    )
    ac.pump()
    o = eng.get_order_by_client_id(coid)
    assert o.status == OrderStatus.PARTIALLY_FILLED
    assert o.filled_quantity == Decimal("60")
    assert fh.leaves(coid) == Decimal("90")

    # final fill to 150 @ avg 100.5
    src.push(
        FillEvent(
            client_order_id=coid,
            event_type="fill",
            filled_qty=Decimal("150"),
            avg_fill_price=Decimal("100.5"),
        )
    )
    ac.pump()
    o = eng.get_order_by_client_id(coid)
    assert o.status == OrderStatus.FILLED
    assert o.filled_quantity == Decimal("150")
    assert o.avg_fill_price == Decimal("100.5")
    assert fh.leaves(coid) == Decimal("0")


def test_incremental_fills_accumulate_vwap():
    submit, _ = _recording_broker()
    eng = _engine(submit)
    w = pd.Series({"AAA": 0.10})
    src = InMemoryFillSource()
    fh = FillHandler(eng)
    ac = AgentClient(
        eng,
        prices_provider=_Prices({"AAA": 100.0}),
        position_provider=_Positions({}),
        fill_handler=fh,
        fill_source=src,
    )
    ac.send_intents(
        IntentBatch(
            ts_ms=1, equity=100_000.0, idempotency_key="b3", intents=[Intent("AAA", 0.10, 10_000.0)]
        )
    )  # qty=100
    coid = next(iter(eng._orders_by_client_id))
    # two incremental fills (cumulative=False): 40 @ 100, 60 @ 110 -> vwap 106
    src.push(
        FillEvent(
            client_order_id=coid,
            event_type="partial_fill",
            last_fill_qty=Decimal("40"),
            last_fill_price=Decimal("100"),
            cumulative=False,
        )
    )
    src.push(
        FillEvent(
            client_order_id=coid,
            event_type="fill",
            last_fill_qty=Decimal("60"),
            last_fill_price=Decimal("110"),
            cumulative=False,
        )
    )
    ac.pump()
    o = eng.get_order_by_client_id(coid)
    assert o.status == OrderStatus.FILLED
    assert o.filled_quantity == Decimal("100")
    assert o.avg_fill_price == Decimal("106")


def test_idempotent_resend_dedups():
    submit, sent = _recording_broker()
    eng = _engine(submit)
    w = _diversified_weights()
    ac = AgentClient(eng, prices_provider=_prices_for(w), position_provider=_Positions({}))
    batch = IntentBatch(
        ts_ms=1,
        equity=100_000.0,
        idempotency_key="same-key",
        intents=[Intent(s, float(v), float(v) * 100_000.0) for s, v in w.items()],
    )
    ac.send_intents(batch)
    n = len(sent)
    assert n == 5
    # resend identical batch -> deterministic intent ids -> journal dedups -> no new submits
    ac.send_intents(batch)
    assert len(sent) == n


def test_runner_wired_with_agent_client_is_not_dry_run():
    submit, sent = _recording_broker()
    eng = _engine(submit)
    w = _diversified_weights()
    ac = AgentClient(eng, prices_provider=_prices_for(w), position_provider=_Positions({}))
    runner = CrossSectionalLiveRunner(agent_client=ac, position_provider=_Positions({}))
    res = runner.rebalance(w, 100_000.0, ts_ms=1)
    assert res.approved and res.sent
    assert len(sent) == 5  # real orders went to the (fake) broker


# ---------------------------------------------------------------------------
# Sliced execution + cancel-replace
# ---------------------------------------------------------------------------
def test_sliced_release_over_clock():
    submit, sent = _recording_broker()
    eng = _engine(submit)
    child_exec = ClockDrivenChildExecutor(
        eng,
        prices_provider=_Prices({"AAA": 100.0}),
        slice_interval_s=10.0,
        straggler_timeout_s=999.0,
    )
    ac = AgentClient(
        eng,
        prices_provider=_Prices({"AAA": 100.0}),
        position_provider=_Positions({}),
        child_executor=child_exec,
        n_slices=4,
        clock=lambda: 1000.0,
    )
    # AAA target 0.10 of 100k = 10k -> qty 100, sliced into 4 of 25 each
    res = ac.send_intents(
        IntentBatch(
            ts_ms=1, equity=100_000.0, idempotency_key="s1", intents=[Intent("AAA", 0.10, 10_000.0)]
        )
    )
    assert res.sliced and len(res.parents) == 1
    parent = child_exec.get_parent(res.parents[0])
    assert len(parent.children) == 4
    assert sum(c.qty for c in parent.children) == Decimal("100")

    # t=1000: only slice 0 due -> 1 order
    ac.pump(now_ts=1000.0)
    assert len(sent) == 1
    # t=1025: slices 0,1,2 due (release_at 1000,1010,1020) -> 2 more
    ac.pump(now_ts=1025.0)
    assert len(sent) == 3
    # t=1035: slice 3 due -> all 4 released
    ac.pump(now_ts=1035.0)
    assert len(sent) == 4
    working = [c for c in parent.children if c.status == ChildState.WORKING]
    assert len(working) == 4


def test_cancel_replace_straggler_rolls_leaves():
    submit, sent = _recording_broker()
    cancels = []

    def broker_cancel(coid, broker_id):
        cancels.append(coid)
        return True

    eng = _engine(submit)
    child_exec = ClockDrivenChildExecutor(
        eng,
        prices_provider=_Prices({"AAA": 100.0}),
        broker_cancel=broker_cancel,
        slice_interval_s=10.0,
        straggler_timeout_s=30.0,
        max_replaces=2,
    )
    ac = AgentClient(
        eng,
        prices_provider=_Prices({"AAA": 100.0}),
        position_provider=_Positions({}),
        child_executor=child_exec,
        n_slices=2,
        clock=lambda: 0.0,
    )
    res = ac.send_intents(
        IntentBatch(
            ts_ms=1, equity=100_000.0, idempotency_key="s2", intents=[Intent("AAA", 0.10, 10_000.0)]
        )
    )  # qty 100 -> 2x50
    parent = child_exec.get_parent(res.parents[0])

    ac.pump(now_ts=0.0)  # release child 0 (50)
    assert len(sent) == 1
    c0 = parent.children[0]
    assert c0.status == ChildState.WORKING

    # child 0 never fills; advance past straggler_timeout -> cancel-replace
    ac.pump(now_ts=40.0)  # also releases child 1 (release_at=10)
    assert c0.client_order_id in cancels
    assert c0.status == ChildState.REPLACED
    # a replacement child with the 50 leaves exists and gets released
    repls = [c for c in parent.children if c.replaces == 1]
    assert len(repls) == 1 and repls[0].qty == Decimal("50")
    # replacement + child1 both eventually working
    ac.pump(now_ts=41.0)
    working = [c for c in parent.children if c.status == ChildState.WORKING]
    assert any(c.replaces == 1 for c in working)


def test_polling_fill_source_diffs_broker_state():
    submit, _ = _recording_broker()
    eng = _engine(submit)
    ac = AgentClient(eng, prices_provider=_Prices({"AAA": 100.0}), position_provider=_Positions({}))
    ac.send_intents(
        IntentBatch(
            ts_ms=1, equity=100_000.0, idempotency_key="p1", intents=[Intent("AAA", 0.10, 10_000.0)]
        )
    )
    coid = next(iter(eng._orders_by_client_id))

    broker_state = {"status": "accepted", "filled_qty": "0"}
    src = PollingFillSource(fetch_order=lambda c: dict(broker_state) if c == coid else None)
    src.track(coid)
    fh = FillHandler(eng)
    fh.consume(src)
    assert eng.get_order_by_client_id(coid).status == OrderStatus.ACCEPTED

    broker_state.update(status="filled", filled_qty="100", filled_avg_price="100.0")
    fh.consume(src)
    o = eng.get_order_by_client_id(coid)
    assert o.status == OrderStatus.FILLED and o.filled_quantity == Decimal("100")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
