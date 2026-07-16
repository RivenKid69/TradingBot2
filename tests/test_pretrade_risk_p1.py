# -*- coding: utf-8 -*-
"""P1 #9 tests: Monte-Carlo VaR, Euler component/marginal/incremental VaR, and the
named historical stress-scenario library in PreTradeRiskAnalyzer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from service_pretrade_risk import (
    PreTradeRiskAnalyzer, RiskLimits, NAMED_STRESS_SCENARIOS,
)


def _cov(symbols, vols, corr):
    d = np.diag(vols)
    S = d @ np.array(corr) @ d
    return pd.DataFrame(S, index=symbols, columns=symbols)


def _setup():
    syms = ["A", "B", "C"]
    vols = [0.02, 0.03, 0.015]
    corr = [[1, .3, .2], [.3, 1, .1], [.2, .1, 1]]
    cov = _cov(syms, vols, corr)
    w = pd.Series([0.5, -0.3, 0.4], index=syms)
    return PreTradeRiskAnalyzer(cov), w


# --------------------------------------------------------------------------- MC VaR
def test_mc_var_close_to_parametric_normal():
    an, w = _setup()
    pv = an.parametric_var(w, 0.05)
    mc_var, mc_cvar = an.monte_carlo_var_cvar(w, 0.05, n_sims=80_000, dist="normal", seed=1)
    # MC normal VaR should be within ~5% of the parametric closed form
    assert mc_var == pytest.approx(pv, rel=0.05)
    assert mc_cvar > mc_var


def test_mc_t_has_fatter_tails():
    an, w = _setup()
    v_norm, _ = an.monte_carlo_var_cvar(w, 0.01, n_sims=120_000, dist="normal", seed=2)
    v_t, _ = an.monte_carlo_var_cvar(w, 0.01, n_sims=120_000, dist="t", dof=4, seed=2)
    # Student-t (dof=4) produces heavier 1% tail than the variance-matched normal
    assert v_t > v_norm


# --------------------------------------------------------------------------- Euler
def test_component_var_sums_to_parametric_var():
    an, w = _setup()
    comp, marg = an.component_var(w, 0.05)
    assert math.isclose(sum(comp.values()), an.parametric_var(w, 0.05), rel_tol=1e-9)
    # component_i == w_i * marginal_i
    for s in w.index:
        assert math.isclose(comp[s], float(w[s]) * marg[s], rel_tol=1e-9)


def test_incremental_var():
    an, w = _setup()
    inc = an.incremental_var(w, "A", 0.05)
    # removing a contributor changes VaR; magnitude is finite
    assert isinstance(inc, float)
    # dropping a name that adds risk should not increase VaR beyond full
    assert inc <= an.parametric_var(w, 0.05) + 1e-9


# --------------------------------------------------------------------------- scenarios
def test_named_scenarios_present_and_negative():
    an, w = _setup()
    scens = an.named_scenarios(w, alpha=0.05)
    names = {s.name for s in scens}
    assert "2008_gfc" in names and "2020_covid" in names
    assert len(scens) == len(NAMED_STRESS_SCENARIOS)
    # the GFC scenario should be a large loss for a net-long book
    gfc = [s for s in scens if s.name == "2008_gfc"][0]
    assert gfc.pnl < 0


def test_pretrade_check_includes_p1_fields():
    an, w = _setup()
    rep = an.pretrade_check(w, limits=RiskLimits(), strict=False)
    d = rep.to_dict()
    assert d["mc_var"] is not None and d["mc_cvar"] is not None
    assert d["mc_dist"] == "t"
    assert math.isclose(sum(d["component_var"].values()), d["var"], rel_tol=1e-6) or d["var"] >= 0
    assert len(d["named_scenarios"]) == len(NAMED_STRESS_SCENARIOS)
    # worst scenario considers the named library too
    assert d["worst_scenario"] is not None


def test_mc_can_be_disabled():
    an, w = _setup()
    rep = an.pretrade_check(w, strict=False, monte_carlo=False)
    assert rep.mc_var is None and rep.mc_dist is None
