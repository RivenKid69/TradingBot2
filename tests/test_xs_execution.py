# -*- coding: utf-8 -*-
"""Тесты портфельного execution-scheduler (P1): trade-list → impact-aware TWAP/VWAP/POV slices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_xs_execution import RebalanceScheduler


def _inputs():
    syms = ["A", "B", "C"]
    wt = pd.Series([0.4, -0.3, 0.1], index=syms)
    w0 = pd.Series([0.0, 0.0, 0.0], index=syms)
    px = pd.Series([100.0, 50.0, 20.0], index=syms)
    adv = pd.Series([1e7, 1e7, 1e7], index=syms)
    return syms, wt, w0, px, adv


def test_trade_list_qty_side_notional():
    syms, wt, w0, px, adv = _inputs()
    sch = RebalanceScheduler(algo="TWAP", n_slices=4)
    plan = sch.build_plan(wt, w0, px, equity=1_000_000, adv=adv)
    by = {t.symbol: t for t in plan.trades}
    # A: +0.4 * 1e6 = +400k notional, BUY, qty = 400000/100 = 4000
    assert by["A"].side == "BUY"
    assert by["A"].notional == pytest.approx(400_000)
    assert by["A"].qty == pytest.approx(4000)
    # B: -0.3 → SELL, notional 300k, qty=6000
    assert by["B"].side == "SELL"
    assert by["B"].qty == pytest.approx(6000)
    # слайсы суммируются в родительский qty
    assert sum(s.qty for s in by["A"].slices) == pytest.approx(4000)
    assert all(s.n_slices == 4 for s in by["A"].slices)


def test_more_slices_lower_impact_cost():
    syms, wt, w0, px, adv = _inputs()
    sch1 = RebalanceScheduler(algo="TWAP", n_slices=1, impact_coef=0.2)
    sch9 = RebalanceScheduler(algo="TWAP", n_slices=9, impact_coef=0.2)
    c1 = sch1.build_plan(wt, w0, px, 1_000_000, adv=adv).total_est_cost
    c9 = sch9.build_plan(wt, w0, px, 1_000_000, adv=adv).total_est_cost
    assert c9 < c1            # нарезка снижает импакт (~1/√N)


def test_pov_more_slices_for_bigger_participation():
    syms, wt, w0, px, adv = _inputs()
    # маленький ADV → большое участие → POV режет на много слайсов
    small_adv = pd.Series([5e5, 5e5, 5e5], index=syms)
    sch = RebalanceScheduler(algo="POV", participation=0.05, impact_coef=0.2)
    plan = sch.build_plan(wt, w0, px, 1_000_000, adv=small_adv)
    a = next(t for t in plan.trades if t.symbol == "A")
    # участие A = 400k/500k = 0.8; при target 0.05 → ceil(0.8/0.05)=16 слайсов
    assert len(a.slices) == 16
    # каждый слайс участвует ≤ target
    assert all((s.notional / 5e5) <= 0.05 + 1e-9 for s in a.slices)


def test_vwap_weights_sum_to_one_and_ushape():
    syms, wt, w0, px, adv = _inputs()
    sch = RebalanceScheduler(algo="VWAP", n_slices=7)
    plan = sch.build_plan(wt, w0, px, 1_000_000, adv=adv)
    a = next(t for t in plan.trades if t.symbol == "A")
    ws = np.array([s.weight for s in a.slices])
    assert ws.sum() == pytest.approx(1.0)
    # U-форма: края тяжелее центра
    assert ws[0] > ws[len(ws) // 2] and ws[-1] > ws[len(ws) // 2]


def test_total_cost_aggregation_and_bps():
    syms, wt, w0, px, adv = _inputs()
    sch = RebalanceScheduler(algo="TWAP", n_slices=5, spread_bps=2.0, impact_coef=0.1)
    plan = sch.build_plan(wt, w0, px, 1_000_000, adv=adv)
    # total cost = сумма по символам; bps = cost/notional*1e4 ≥ half-spread floor
    assert plan.total_est_cost == pytest.approx(sum(t.est_cost for t in plan.trades))
    assert plan.total_est_cost_bps >= 1.0   # ≥ half-spread (1 bps)
    assert plan.total_notional == pytest.approx(400_000 + 300_000 + 100_000)


def test_min_trade_notional_filter():
    syms, wt, w0, px, adv = _inputs()
    sch = RebalanceScheduler(algo="TWAP", n_slices=4, min_trade_notional=150_000)
    plan = sch.build_plan(wt, w0, px, 1_000_000, adv=adv)
    # C: 0.1*1e6 = 100k < 150k → отфильтрован; остаются A(400k), B(300k)
    syms_traded = {t.symbol for t in plan.trades}
    assert syms_traded == {"A", "B"}
