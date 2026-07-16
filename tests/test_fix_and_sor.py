# -*- coding: utf-8 -*-
"""Тесты FIX 4.4 протокола + Smart Order Routing (P2)."""

from __future__ import annotations

import pytest

from packages.agent.execution.fix_protocol import (
    SOH, Tag, MsgType, OrdType, Side, encode_message, execution_report, new_order_single,
    order_cancel_request, parse_message, verify_checksum,
)
from packages.agent.execution.smart_order_router import SmartOrderRouter, Venue


# --- FIX ---
def test_new_order_single_roundtrip_and_checksum():
    msg = new_order_single(cl_ord_id="ORD1", symbol="AAPL", side=Side.BUY, qty=100,
                           ord_type=OrdType.LIMIT, price=150.0)
    assert verify_checksum(msg) is True
    d = parse_message(msg)
    assert d[Tag.MsgType] == "D"
    assert d[Tag.ClOrdID] == "ORD1"
    assert d[Tag.Symbol] == "AAPL"
    assert d[Tag.Side] == "1"             # BUY
    assert d[Tag.OrderQty] == "100"
    assert d[Tag.OrdType] == "2"          # LIMIT
    assert d[Tag.Price] == "150"


def test_body_length_correct():
    msg = new_order_single(cl_ord_id="X", symbol="ES", side=Side.SELL, qty=2)
    # BodyLength(9) = длина тела (от 35= до SOH перед 10=)
    body_start = msg.index(f"{SOH}{Tag.MsgType}=") + 1
    cs_start = msg.rfind(f"{SOH}{Tag.CheckSum}=") + 1
    body = msg[body_start:cs_start]
    d = parse_message(msg)
    assert int(d[Tag.BodyLength]) == len(body)


def test_checksum_detects_tamper():
    msg = new_order_single(cl_ord_id="X", symbol="ES", side=Side.BUY, qty=1)
    tampered = msg.replace("55=ES", "55=NQ")
    assert verify_checksum(tampered) is False


def test_execution_report_and_cancel():
    er = execution_report(order_id="B1", cl_ord_id="ORD1", exec_id="E1", symbol="AAPL",
                          side=Side.BUY, ord_status="2", exec_type="F", cum_qty=100, avg_px=150.25)
    assert verify_checksum(er)
    d = parse_message(er)
    assert d[Tag.MsgType] == "8" and d[Tag.OrdStatus] == "2" and d[Tag.AvgPx] == "150.25"

    cxl = order_cancel_request(orig_cl_ord_id="ORD1", cl_ord_id="ORD2", symbol="AAPL", side=Side.BUY)
    assert verify_checksum(cxl)
    assert parse_message(cxl)[Tag.MsgType] == "F"


# --- SOR ---
def _venues():
    return [
        Venue("CHEAP", fee_bps=0.5, latency_ms=20, liquidity=2e6, impact_coef=0.1),
        Venue("MID", fee_bps=1.0, latency_ms=50, liquidity=5e6, impact_coef=0.1),
        Venue("EXP", fee_bps=3.0, latency_ms=200, liquidity=1e7, impact_coef=0.1),
    ]


def test_best_venue_picks_lowest_cost():
    # при равной ликвидности (равный импакт) решает fee → CHEAP
    venues = [Venue("CHEAP", fee_bps=0.5, liquidity=1e7, impact_coef=0.1),
              Venue("MID", fee_bps=1.0, liquidity=1e7, impact_coef=0.1),
              Venue("EXP", fee_bps=3.0, liquidity=1e7, impact_coef=0.1)]
    sor = SmartOrderRouter(venues)
    assert sor.best_venue(100_000).name == "CHEAP"


def test_best_venue_prefers_liquidity_for_impact():
    # при равном fee решает ликвидность (меньше импакт) → DEEP
    venues = [Venue("THIN", fee_bps=1.0, liquidity=1e6, impact_coef=0.1),
              Venue("DEEP", fee_bps=1.0, liquidity=1e8, impact_coef=0.1)]
    sor = SmartOrderRouter(venues)
    assert sor.best_venue(500_000).name == "DEEP"


def test_split_allocates_full_and_cheaper_first():
    sor = SmartOrderRouter(_venues(), n_steps=100)
    res = sor.route("AAPL", "BUY", 3_000_000, split=True)
    total = sum(a.notional for a in res.allocations)
    assert total == pytest.approx(3_000_000, rel=1e-6)
    # multi-venue использовано (импакт распределён)
    assert len(res.allocations) >= 2
    # суммарная стоимость сплита ≤ стоимости одного venue на весь объём
    single = sor.route("AAPL", "BUY", 3_000_000, split=False)
    assert res.total_est_cost <= single.total_est_cost + 1e-6


def test_min_notional_respected():
    venues = [Venue("A", fee_bps=0.5, liquidity=1e6, min_notional=500_000),
              Venue("B", fee_bps=1.0, liquidity=1e6)]
    sor = SmartOrderRouter(venues, n_steps=20)
    res = sor.route("X", "BUY", 100_000, split=True)   # < min_notional у A
    names = {a.venue for a in res.allocations}
    assert "A" not in names or all(a.notional >= 500_000 for a in res.allocations if a.venue == "A")
