# -*- coding: utf-8 -*-
"""Tests for the firm-wide hierarchical risk aggregator (service_firm_risk).

Verifies the academic invariants:
  * Euler additivity: Σ component VaR(child) == parent VaR (homogeneity deg 1).
  * Euler additivity for CVaR (parametric and historical tail attribution).
  * Subadditivity / diversification benefit ≥ 0 for correlations < 1.
  * Hierarchy: firm = aggregate of desks = aggregate of strategies.
  * Limit breaches (hard/soft) and the approved verdict.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from service_firm_risk import (
    FirmPosition, FirmRiskAggregator, HierLimits, positions_from_books,
)


def _cov(units, vols, corr):
    d = np.diag(vols)
    C = np.array(corr, dtype="float64")
    S = d @ C @ d
    return pd.DataFrame(S, index=units, columns=units)


def _positions():
    return [
        FirmPosition("AAPL", 50_000, desk="equity", strategy="momentum", sector="tech"),
        FirmPosition("XOM", -30_000, desk="equity", strategy="meanrev", sector="energy"),
        FirmPosition("ES", 80_000, desk="futures", strategy="trend"),
        FirmPosition("EURUSD", -40_000, desk="fx", strategy="carry"),
    ]


def _agg(alpha=0.05):
    units = ["AAPL", "XOM", "ES", "EURUSD"]
    vols = [0.02, 0.025, 0.015, 0.008]
    corr = [
        [1.0, 0.3, 0.6, -0.1],
        [0.3, 1.0, 0.4, 0.0],
        [0.6, 0.4, 1.0, -0.2],
        [-0.1, 0.0, -0.2, 1.0],
    ]
    return FirmRiskAggregator(cov=_cov(units, vols, corr), alpha=alpha)


# --------------------------------------------------------------------------- core
def test_firm_var_positive_and_hierarchy():
    rep = _agg().aggregate(_positions())
    firm = rep.firm
    assert firm.var > 0 and firm.cvar >= firm.var
    assert firm.level == "firm"
    # firm has 3 desks
    assert len(firm.children) == 3
    # n_positions aggregates
    assert firm.n_positions == 4
    assert {c.name for c in firm.children} == {"equity", "futures", "fx"}


def test_euler_component_var_sums_to_parent_parametric():
    rep = _agg().aggregate(_positions())
    firm = rep.firm
    comp_sum = sum(c.component_var for c in firm.contributions)
    assert math.isclose(comp_sum, firm.var, rel_tol=1e-9, abs_tol=1e-6)
    # CVaR components also sum to parent CVaR
    comp_cvar_sum = sum(c.component_cvar for c in firm.contributions)
    assert math.isclose(comp_cvar_sum, firm.cvar, rel_tol=1e-9, abs_tol=1e-6)


def test_euler_components_recurse_each_desk():
    rep = _agg().aggregate(_positions())
    for desk in rep.firm.children:
        if desk.contributions:
            s = sum(c.component_var for c in desk.contributions)
            assert math.isclose(s, desk.var, rel_tol=1e-9, abs_tol=1e-6)


def test_diversification_benefit_nonnegative():
    rep = _agg().aggregate(_positions())
    # Σ standalone desk VaR ≥ firm VaR (subadditivity since |corr|<1)
    standalone = sum(c.standalone_var for c in rep.firm.contributions)
    assert standalone + 1e-9 >= rep.firm.var
    assert rep.firm.diversification_benefit >= -1e-9


def test_perfect_correlation_no_diversification():
    units = ["A", "B"]
    vols = [0.02, 0.02]
    corr = [[1.0, 1.0], [1.0, 1.0]]
    agg = FirmRiskAggregator(cov=_cov(units, vols, corr), alpha=0.05)
    pos = [
        FirmPosition("A", 10_000, desk="d1", strategy="s1"),
        FirmPosition("B", 10_000, desk="d2", strategy="s2"),
    ]
    rep = agg.aggregate(pos)
    # with corr=1 and same sign, VaR is additive => diversification ~ 0
    assert abs(rep.firm.diversification_benefit) < 1e-6


def test_gross_net_exposure():
    rep = _agg().aggregate(_positions())
    assert math.isclose(rep.firm.gross, 50_000 + 30_000 + 80_000 + 40_000)
    assert math.isclose(rep.firm.net, 50_000 - 30_000 + 80_000 - 40_000)


def test_sector_exposure_aggregation():
    rep = _agg().aggregate(_positions())
    sec = rep.firm.sector_exposure
    assert math.isclose(sec.get("tech", 0.0), 50_000)
    assert math.isclose(sec.get("energy", 0.0), -30_000)


def test_incremental_var_signs():
    rep = _agg().aggregate(_positions())
    # removing a book that adds risk should lower VaR => incremental > 0 for it
    incrementals = {c.name: c.incremental_var for c in rep.firm.contributions}
    assert any(v > 0 for v in incrementals.values())


# --------------------------------------------------------------------------- historical
def test_historical_engine_and_attribution():
    rng = np.random.default_rng(42)
    units = ["AAPL", "XOM", "ES", "EURUSD"]
    vols = np.array([0.02, 0.025, 0.015, 0.008])
    corr = np.array([
        [1.0, 0.3, 0.6, -0.1],
        [0.3, 1.0, 0.4, 0.0],
        [0.6, 0.4, 1.0, -0.2],
        [-0.1, 0.0, -0.2, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((4000, 4))
    R = (Z @ L.T) * vols
    returns = pd.DataFrame(R, columns=units)
    agg = FirmRiskAggregator(returns=returns, alpha=0.05)
    rep = agg.aggregate(_positions(), method="historical")
    assert rep.firm.var > 0 and rep.firm.cvar >= rep.firm.var
    # CVaR Euler components sum to parent CVaR exactly (tail-mean attribution)
    s = sum(c.component_cvar for c in rep.firm.contributions)
    assert math.isclose(s, rep.firm.cvar, rel_tol=1e-6, abs_tol=1e-4)


# --------------------------------------------------------------------------- limits
def test_hard_limit_breach_blocks_approval():
    limits = {"FIRM": HierLimits(var=1.0, hard=True)}  # absurdly low VaR cap
    rep = _agg().aggregate(_positions(), limits=limits)
    assert rep.approved is False
    assert any(b.node == "FIRM" and b.metric == "VaR" for b in rep.breaches)


def test_soft_limit_does_not_block():
    limits = {"FIRM": HierLimits(gross=1.0, hard=False)}
    rep = _agg().aggregate(_positions(), limits=limits)
    assert rep.approved is True
    assert any(not b.hard for b in rep.breaches)


def test_capital_var_pct():
    rep = _agg().aggregate(_positions(), capital={"FIRM": 1_000_000})
    assert rep.firm.var_pct is not None
    assert math.isclose(rep.firm.var_pct, rep.firm.var / 1_000_000)


def test_desk_level_limit():
    limits = {"equity": HierLimits(net=1.0, hard=True)}
    rep = _agg().aggregate(_positions(), limits=limits)
    assert rep.approved is False
    assert any(b.node == "equity" for b in rep.breaches)


# --------------------------------------------------------------------------- helpers
def test_positions_from_books():
    books = {
        "equity": {"momentum": [{"symbol": "AAPL", "exposure": 5000, "sector": "tech"}]},
        "fx": {"carry": [{"symbol": "EURUSD", "exposure": -3000}]},
    }
    pos = positions_from_books(books)
    assert len(pos) == 2
    aapl = [p for p in pos if p.symbol == "AAPL"][0]
    assert aapl.desk == "equity" and aapl.strategy == "momentum" and aapl.sector == "tech"


def test_empty_positions_safe():
    rep = _agg().aggregate([])
    assert rep.firm.var == 0.0 and rep.approved is True


def test_to_dict_serializable():
    import json
    rep = _agg().aggregate(_positions(), capital={"FIRM": 1_000_000})
    blob = json.dumps(rep.to_dict())
    assert "diversification_benefit" in blob
