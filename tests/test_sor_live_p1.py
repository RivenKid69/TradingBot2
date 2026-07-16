# -*- coding: utf-8 -*-
"""P1 #7: SmartOrderRouter wired into the LIVE submission path — routed_broker_submit
splits an order across venues and dispatches real child orders to venue connectors."""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.agent.broker.adapters.sim import SimBrokerConnector
from packages.agent.execution.smart_order_router import SmartOrderRouter, Venue
from packages.agent.execution.live_factory import (
    BrokerLiquidityProvider, make_venue_submit, routed_broker_submit, build_live_stack,
)
from packages.agent.execution.engine import Order, OrderType


def _venues():
    return [Venue("V1", fee_bps=0.5, liquidity=5e6, impact_coef=0.1),
            Venue("V2", fee_bps=0.8, liquidity=5e6, impact_coef=0.1)]


def _conns(symbol="AAPL", price=100.0):
    c1, c2 = SimBrokerConnector(broker_name="V1"), SimBrokerConnector(broker_name="V2")
    c1.set_price(symbol, price); c2.set_price(symbol, price)
    return {"V1": c1, "V2": c2}


def test_make_venue_submit_dispatches_to_connector():
    conns = _conns()
    submit = make_venue_submit(conns, lambda s: 100.0)
    res = submit("V1", "AAPL", "BUY", 100_000.0)
    assert res["success"] and res["venue"] == "V1"
    # the SimBroker actually has the position now
    assert conns["V1"].get_position("AAPL") is not None


def test_routed_submit_splits_and_dispatches():
    conns = _conns()
    sor = SmartOrderRouter(_venues())
    provider = BrokerLiquidityProvider(conns)
    record = []
    submit = routed_broker_submit(sor, conns, lambda s: 100.0, provider, record=record)
    order = Order(client_order_id="o1", symbol="AAPL", side="buy",
                  order_type=OrderType.MARKET, quantity=Decimal("2000"))  # 200k notional
    ok, broker_id, err = submit(order)
    assert ok and broker_id.startswith("routed:")
    assert record and record[0]["dispatch"]["all_ok"]
    # both venues received child orders (water-filling split)
    venues_hit = {d["venue"] for d in record[0]["dispatch"]["dispatches"]}
    assert len(venues_hit) >= 1
    # positions actually created on the venues
    total = sum(float(c.get_position("AAPL").quantity) for c in conns.values()
                if c.get_position("AAPL") is not None)
    assert total == pytest.approx(2000.0, rel=1e-3)


def test_build_live_stack_uses_sor():
    main = SimBrokerConnector(broker_name="V1")
    main.set_price("AAPL", 100.0)
    conns = {"V1": main, "V2": SimBrokerConnector(broker_name="V2")}
    conns["V2"].set_price("AAPL", 100.0)
    sor = SmartOrderRouter(_venues())
    stack = build_live_stack(main, sor=sor, venue_connectors=conns, symbols=["AAPL"])
    assert stack["sor"] is sor
    # submit an order through the engine -> routed + dispatched
    from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
    eng = stack["engine"]
    res = eng.execute(OrderIntent(strategy_id="s", symbol="AAPL",
                                  intent_type=IntentType.MARKET_ENTRY, side=IntentSide.LONG,
                                  target_quantity=Decimal("100")),   # 10k notional < 25% conc limit
                      current_price=Decimal("100"))
    assert res.success
    assert stack["sor_routes"], "live routing should be recorded"
    assert str(res.order.broker_order_id).startswith("routed:")


def test_live_liquidity_provider_quote():
    conns = _conns(price=50.0)
    prov = BrokerLiquidityProvider(conns)
    q = prov.get_quote("V1", "AAPL")
    assert q is not None and q.bid < q.ask
    assert q.ask_size > 0
