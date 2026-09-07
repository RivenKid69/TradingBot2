# -*- coding: utf-8 -*-
"""P1 #10: live IS executor (Almgren-Chriss), FIX 35=G amend, engine cancel/replace,
and the fat-finger / price-collar pre-trade gate."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from packages.agent.execution.fix_protocol import (
    order_cancel_replace_request,
    parse_message,
    verify_checksum,
    MsgType,
    Tag,
    Side,
    OrdType,
)
from packages.agent.execution.engine import (
    LiveExecutionEngine,
    PriceCollarConfig,
    Order,
    OrderType,
    OrderStatus,
)
from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
from service_xs_execution import RebalanceScheduler


# --------------------------------------------------------------------------- FIX 35=G
def test_fix_cancel_replace_message():
    msg = order_cancel_replace_request(
        orig_cl_ord_id="O1",
        cl_ord_id="O2",
        symbol="AAPL",
        side=Side.BUY,
        qty=100,
        ord_type=OrdType.LIMIT,
        price=150.25,
    )
    assert verify_checksum(msg)
    d = parse_message(msg)
    assert d[Tag.MsgType] == MsgType.ORDER_CANCEL_REPLACE_REQUEST.value == "G"
    assert d[Tag.OrigClOrdID] == "O1" and d[Tag.ClOrdID] == "O2"
    assert d[Tag.OrderQty] == "100" and d[Tag.Price] == "150.25"


# --------------------------------------------------------------------------- IS profile
def test_is_profile_front_loaded_and_sums_to_one():
    sch = RebalanceScheduler(algo="IS", n_slices=8, urgency=3.0)
    w = sch._slice_weights(0.1)
    assert len(w) == 8
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[0] > w[-1]  # front-loaded
    assert all(w[i] >= w[i + 1] - 1e-9 for i in range(len(w) - 1))  # monotone decreasing


def test_is_zero_urgency_is_twap():
    sch = RebalanceScheduler(algo="IS", n_slices=6, urgency=0.0)
    w = sch._slice_weights(0.1)
    assert np.allclose(w, np.ones(6) / 6)


def test_is_algo_in_build_plan():
    import pandas as pd

    sch = RebalanceScheduler(algo="IS", n_slices=5, urgency=2.0)
    plan = sch.build_plan(
        pd.Series({"AAPL": 0.5}),
        pd.Series({"AAPL": 0.0}),
        pd.Series({"AAPL": 100.0}),
        equity=100_000,
    )
    assert plan.algo == "IS"
    assert plan.trades and len(plan.trades[0].slices) == 5
    # first slice notional > last (front-loaded)
    sl = plan.trades[0].slices
    assert sl[0].notional > sl[-1].notional


# --------------------------------------------------------------------------- price-collar
def _engine(collar, broker_submit=None, broker_cancel=None, broker_replace=None):
    import tempfile
    from pathlib import Path
    from packages.agent.reconciliation.journal import OrderJournal

    j = OrderJournal(db_path=Path(tempfile.mkdtemp()) / "j.db")
    return LiveExecutionEngine(
        broker_submit=broker_submit or (lambda o: (True, "B1", None)),
        broker_cancel=broker_cancel,
        broker_replace=broker_replace,
        order_journal=j,
        price_collar=collar,
        broker_name="sim",
    )


def _intent(symbol="AAPL", qty="100", limit=None):
    return OrderIntent(
        strategy_id="s",
        symbol=symbol,
        intent_type=IntentType.LIMIT_ENTRY if limit else IntentType.MARKET_ENTRY,
        side=IntentSide.LONG,
        target_quantity=Decimal(qty),
        limit_price=(Decimal(str(limit)) if limit else None),
    )


def test_price_collar_blocks_far_limit():
    eng = _engine(PriceCollarConfig(max_price_distance_pct=0.10))
    # limit 200 vs reference 100 -> 100% away -> blocked
    res = eng.execute(_intent(limit=200), current_price=Decimal("100"))
    assert not res.success and "Price-collar" in res.error_message


def test_price_collar_blocks_huge_notional():
    # qty 100 passes hard caps (<10k) but 100*100=10k notional > max_notional 1k.
    eng = _engine(PriceCollarConfig(max_price_distance_pct=None, max_notional=1_000))
    res = eng.execute(_intent(qty="100", limit=100), current_price=Decimal("100"))
    assert not res.success and "notional" in res.error_message


def test_price_collar_allows_normal_order():
    eng = _engine(PriceCollarConfig(max_price_distance_pct=0.10, max_notional=10_000_000))
    res = eng.execute(_intent(qty="10", limit=101), current_price=Decimal("100"))
    assert res.success


def test_price_collar_adv_participation():
    cfg = PriceCollarConfig(max_adv_participation=0.05, adv_provider=lambda s: 100_000.0)
    eng = _engine(cfg)
    # notional 10*100=1000? need > 5% of ADV 100k = 5000 -> use qty 100 @100 = 10000 > 5000
    res = eng.execute(_intent(qty="100", limit=100), current_price=Decimal("100"))
    assert not res.success and "ADV" in res.error_message


# --------------------------------------------------------------------------- cancel/replace
def test_engine_cancel_order():
    cancelled = {}
    eng = _engine(
        PriceCollarConfig(enabled=False),
        broker_cancel=lambda coid, bid: cancelled.setdefault("c", coid) or True,
    )
    res = eng.execute(_intent())
    coid = res.order.client_order_id
    cr = eng.cancel_order(coid)
    assert cr.success and cancelled["c"] == coid
    assert eng.get_order_by_client_id(coid).status == OrderStatus.CANCELLED


def test_engine_replace_order():
    eng = _engine(
        PriceCollarConfig(max_price_distance_pct=0.5),
        broker_replace=lambda coid, q, p: (True, "B2", None),
    )
    res = eng.execute(_intent(qty="100", limit=100))
    coid = res.order.client_order_id
    rr = eng.replace_order(
        coid,
        new_quantity=Decimal("150"),
        new_limit_price=Decimal("101"),
        current_price=Decimal("100"),
    )
    assert rr.success
    o = eng.get_order_by_client_id(coid)
    assert float(o.quantity) == 150.0 and float(o.limit_price) == 101.0


def test_engine_replace_blocked_by_collar():
    eng = _engine(
        PriceCollarConfig(max_price_distance_pct=0.10),
        broker_replace=lambda coid, q, p: (True, "B2", None),
    )
    res = eng.execute(_intent(qty="100", limit=100))
    coid = res.order.client_order_id
    # amend to a far price -> blocked
    rr = eng.replace_order(coid, new_limit_price=Decimal("200"), current_price=Decimal("100"))
    assert not rr.success and "Price-collar" in rr.error_message
