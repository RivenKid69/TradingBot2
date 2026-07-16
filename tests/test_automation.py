# -*- coding: utf-8 -*-
"""Тесты автоматизации (P2): drift-driven retrain + авто-TCA отчёт."""

from __future__ import annotations

import pytest

from services.automation.drift_retrain import (
    DriftRetrainScheduler, RetrainDecision, psi_from_report,
)
from services.automation.tca_reporter import TCAReporter


# --- drift retrain ---
def test_psi_from_report_formats():
    assert psi_from_report({"f1": 0.3, "f2": 0.05}) == {"f1": 0.3, "f2": 0.05}
    assert psi_from_report({"features": {"f1": {"psi": 0.4}}}) == {"f1": 0.4}
    assert psi_from_report({}) == {}


def test_no_retrain_below_threshold():
    sch = DriftRetrainScheduler(psi_threshold=0.25)
    d = sch.check({"f1": 0.1, "f2": 0.2})
    assert d.should_retrain is False and "< threshold" in d.reason


def test_retrain_triggered_above_threshold():
    sch = DriftRetrainScheduler(psi_threshold=0.25)
    d = sch.check({"f1": 0.1, "f2": 0.4, "f3": 0.3})
    assert d.should_retrain is True
    assert d.triggering_features == ["f2", "f3"]   # отсортированы по PSI убыв.
    assert d.max_psi == pytest.approx(0.4)


def test_cooldown_blocks_repeat():
    clock = {"t": 0.0}
    sch = DriftRetrainScheduler(psi_threshold=0.25, cooldown_sec=100.0, time_fn=lambda: clock["t"])
    calls = []
    d1 = sch.run({"f": 0.5}, retrain_fn=lambda dec: calls.append(dec))
    assert d1.should_retrain and len(calls) == 1
    clock["t"] = 50.0
    d2 = sch.run({"f": 0.6}, retrain_fn=lambda dec: calls.append(dec))
    assert d2.should_retrain is False and d2.on_cooldown is True and len(calls) == 1
    clock["t"] = 150.0           # cooldown прошёл
    d3 = sch.run({"f": 0.6}, retrain_fn=lambda dec: calls.append(dec))
    assert d3.should_retrain and len(calls) == 2


# --- TCA reporter ---
def _trades():
    return [
        # BUY filled выше arrival → положительный implementation shortfall (хуже)
        {"symbol": "AAPL", "side": "BUY", "qty": 100, "arrival_price": 100.0,
         "fill_price": 100.10, "benchmark_price": 100.05, "venue": "NYSE"},
        # SELL filled ниже arrival → тоже положительный IS (хуже для продажи)
        {"symbol": "AAPL", "side": "SELL", "qty": 100, "arrival_price": 100.0,
         "fill_price": 99.90, "benchmark_price": 99.95, "venue": "NASDAQ"},
        {"symbol": "MSFT", "side": "BUY", "qty": 50, "arrival_price": 200.0,
         "fill_price": 200.0, "benchmark_price": 200.0, "venue": "NYSE"},
    ]


def test_tca_metrics_sign_correct():
    rep = TCAReporter().analyze(_trades())
    assert rep.n_trades == 3
    # обе «плохие» сделки → IS > 0
    assert rep.avg_impl_shortfall_bps > 0
    # AAPL BUY: (100.10-100.0)/100 *1e4 = 10 bps
    assert rep.by_symbol["AAPL"]["impl_shortfall_bps"] == pytest.approx(10.0, abs=0.1)
    assert set(rep.by_venue) == {"NYSE", "NASDAQ"}
    assert rep.by_venue["NYSE"]["n"] == 2


def test_tca_markdown_and_empty():
    rep = TCAReporter().analyze(_trades())
    md = TCAReporter().to_markdown(rep)
    assert "TCA" in md and "venue" in md.lower()
    empty = TCAReporter().analyze([])
    assert empty.n_trades == 0 and empty.total_cost == 0.0
