# -*- coding: utf-8 -*-
"""Тесты live pre-trade риск-контура (P1): VaR/CVaR/стресс/сценарии + factor-monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_pretrade_risk import (
    FactorExposureMonitor,
    PreTradeRiskAnalyzer,
    RiskLimits,
)


def _cov(symbols, vols, rho):
    n = len(symbols)
    C = np.full((n, n), rho)
    np.fill_diagonal(C, 1.0)
    D = np.diag(vols)
    S = D @ C @ D
    return pd.DataFrame(S, index=symbols, columns=symbols)


def test_portfolio_vol_and_var_cvar():
    syms = ["A", "B"]
    cov = _cov(syms, [0.2, 0.2], 0.0)  # независимые, σ=0.2
    w = pd.Series([0.5, 0.5], index=syms)
    an = PreTradeRiskAnalyzer(cov)
    vol = an.portfolio_vol(w)
    assert vol == pytest.approx(math_sqrt(0.5**2 * 0.04 + 0.5**2 * 0.04))
    var = an.parametric_var(w, 0.05)
    cvar = an.parametric_cvar(w, 0.05)
    assert var > 0 and cvar > var  # CVaR(ES) ≥ VaR для Gaussian
    assert var == pytest.approx(1.6448536 * vol, rel=1e-3)


def math_sqrt(x):
    import math

    return math.sqrt(x)


def test_historical_var_cvar():
    syms = ["A"]
    rng = np.random.RandomState(0)
    rets = pd.DataFrame({"A": rng.normal(0, 0.02, 5000)})
    an = PreTradeRiskAnalyzer(_cov(syms, [0.02], 0.0))
    w = pd.Series([1.0], index=syms)
    var, cvar = an.historical_var_cvar(w, rets, 0.05)
    assert var > 0 and cvar >= var
    assert var == pytest.approx(0.02 * 1.645, rel=0.15)  # ~Gaussian


def test_factor_exposures():
    syms = ["A", "B", "C"]
    B = pd.DataFrame({"market": [1.0, 1.0, 1.0], "size": [0.5, -0.5, 0.0]}, index=syms)
    an = PreTradeRiskAnalyzer(_cov(syms, [0.2] * 3, 0.1), exposures=B)
    w = pd.Series([0.4, -0.4, 0.2], index=syms)
    fexp = an.factor_exposures(w)
    assert fexp["market"] == pytest.approx(0.2)  # 0.4-0.4+0.2
    assert fexp["size"] == pytest.approx(0.4)  # 0.4*0.5 + (-0.4)*(-0.5)


def test_scenario_grid():
    syms = ["A", "B"]
    cov = _cov(syms, [0.2, 0.2], 0.3)
    B = pd.DataFrame({"market": [1.0, 1.0]}, index=syms)
    an = PreTradeRiskAnalyzer(cov, exposures=B)
    w = pd.Series([0.5, 0.5], index=syms)  # net-long
    scens = an.scenario_grid(w, market_shock=-0.10, vol_mult=1.5, corr_shift=0.2)
    names = {s.name: s for s in scens}
    # рыночный шок −10% для net-long beta=1 → P&L ≈ -10% * 1.0 = -0.10
    assert names["market_shock_-10pct"].pnl == pytest.approx(-0.10, rel=1e-6)
    # рост волатильности ×1.5 → VaR в 1.5 раза больше базового
    base_var = an.parametric_var(w, 0.05)
    assert names["vol_x1.5"].var == pytest.approx(1.5 * base_var, rel=1e-6)
    # сдвиг корреляций вверх → vol растёт (для одинаково-знаковых весов)
    assert names["corr_shift_+0.2"].var > base_var


def test_pretrade_gate_blocks_and_passes():
    syms = ["A", "B"]
    cov = _cov(syms, [0.3, 0.3], 0.5)
    B = pd.DataFrame({"market": [1.0, 1.0]}, index=syms)
    an = PreTradeRiskAnalyzer(cov, exposures=B)
    w = pd.Series([0.6, 0.6], index=syms)
    # жёсткие лимиты → блок
    rep = an.pretrade_check(w, limits=RiskLimits(var_max=0.01, factor_caps={"market": 0.5}))
    assert rep.approved is False
    assert any("VaR" in v for v in rep.violations)
    assert any("factor market" in v for v in rep.violations)
    assert rep.worst_scenario is not None
    # мягкие лимиты → проход
    rep2 = an.pretrade_check(w, limits=RiskLimits(var_max=10.0, factor_caps={"market": 10.0}))
    assert rep2.approved is True and rep2.violations == []


def test_factor_exposure_monitor():
    syms = ["A", "B"]
    B = pd.DataFrame({"market": [1.0, 1.0], "size": [1.0, -1.0]}, index=syms)
    mon = FactorExposureMonitor(B, {"market": 0.3, "size": 1.0})
    rec = mon.update(pd.Series([0.4, 0.4], index=syms), ts_ms=1)  # market=0.8 > 0.3
    assert rec["within_limits"] is False
    assert any("market" in b for b in rec["breaches"])
    rec2 = mon.update(pd.Series([0.1, -0.1], index=syms), ts_ms=2)  # market=0, size=0.2
    assert rec2["within_limits"] is True
    assert len(mon.history()) == 2
    assert mon.latest()["ts_ms"] == 2
