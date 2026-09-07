# -*- coding: utf-8 -*-
"""Интеграция P1 в cross-sectional live-путь: pre-trade VaR/стресс + execution-plan."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_pretrade_risk import PreTradeRiskAnalyzer, RiskLimits
from service_xs_execution import RebalanceScheduler
from service_xs_live import CrossSectionalLiveRunner


class _Prices:
    def __init__(self, d):
        self._d = d

    def get_prices(self):
        return self._d


class _Adv:
    def __init__(self, d):
        self._d = d

    def get_adv(self):
        return self._d


def _cov(syms, vol=0.3, rho=0.5):
    n = len(syms)
    C = np.full((n, n), rho)
    np.fill_diagonal(C, 1.0)
    D = np.diag([vol] * n)
    return pd.DataFrame(D @ C @ D, index=syms, columns=syms)


def test_pretrade_var_blocks_rebalance():
    syms = ["A", "B"]
    w = pd.Series([0.6, 0.6], index=syms)
    an = PreTradeRiskAnalyzer(_cov(syms))
    runner = CrossSectionalLiveRunner(
        pretrade_analyzer=an, risk_limits=RiskLimits(var_max=0.01)
    )  # очень жёсткий VaR
    res = runner.rebalance(w, equity=1_000_000, ts_ms=1)
    assert res.approved is False
    assert res.risk_report is not None
    assert any("VaR" in v for v in res.decision.violations)
    assert res.batch is None  # не отправлено


def test_pretrade_passes_and_builds_execution_plan():
    syms = ["A", "B", "C"]
    w = pd.Series([0.3, -0.3, 0.1], index=syms)
    an = PreTradeRiskAnalyzer(_cov(syms))
    sched = RebalanceScheduler(algo="TWAP", n_slices=5, impact_coef=0.1)
    runner = CrossSectionalLiveRunner(
        pretrade_analyzer=an,
        risk_limits=RiskLimits(var_max=10.0),  # мягко
        scheduler=sched,
        prices_provider=_Prices({"A": 100.0, "B": 50.0, "C": 20.0}),
        adv_provider=_Adv({"A": 1e7, "B": 1e7, "C": 1e7}),
    )
    res = runner.rebalance(w, equity=1_000_000, ts_ms=2)
    assert res.approved is True
    assert res.risk_report is not None and res.risk_report["approved"] is True
    # execution-plan построен с нарезкой
    plan = res.execution_plan
    assert plan is not None and plan["algo"] == "TWAP"
    assert len(plan["trades"]) == 3
    a = next(t for t in plan["trades"] if t["symbol"] == "A")
    assert a["n_slices"] == 5 and a["side"] == "BUY"
    assert plan["total_est_cost"] > 0


def test_scenario_grid_in_report():
    syms = ["A", "B"]
    w = pd.Series([0.5, 0.5], index=syms)
    B = pd.DataFrame({"market": [1.0, 1.0]}, index=syms)
    an = PreTradeRiskAnalyzer(_cov(syms), exposures=B)
    runner = CrossSectionalLiveRunner(pretrade_analyzer=an, risk_limits=RiskLimits())
    res = runner.rebalance(w, equity=1_000_000, ts_ms=3)
    scens = res.risk_report["scenarios"]
    names = {s["name"] for s in scens}
    assert any("market_shock" in n for n in names)
    assert any("vol_x" in n for n in names)
    assert any("corr_shift" in n for n in names)
    assert res.risk_report["worst_scenario"] is not None
