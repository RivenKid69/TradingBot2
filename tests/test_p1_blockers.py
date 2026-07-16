# -*- coding: utf-8 -*-
"""
Tests for P1 blockers #6, #9, #10, #11, #12.

  #6  sector caps enforced in scipy SLSQP AND numpy fallback (were silently ignored)
  #9  position auto-reconciliation: detect -> persist -> halt + auto-flatten -> heal
  #10 Sortino/Calmar + benchmark-relative metrics; rendered HTML tear-sheet
  #11 factor attribution tied to the fitted risk model (exact tie-out)
  #12 capacity (AUM->Sharpe) computed and surfaced in run_backtest
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# #6 — sector caps
# ---------------------------------------------------------------------------
def _opt_inputs():
    syms = [f"T{i}" for i in range(5)] + [f"F{i}" for i in range(5)]
    mu = pd.Series([0.5] * 5 + [0.01] * 5, index=syms)
    Sigma = pd.DataFrame(np.eye(10) * 0.04, index=syms, columns=syms)
    sm = {**{s: "tech" for s in syms[:5]}, **{s: "fin" for s in syms[5:]}}
    return syms, mu, Sigma, sm


def test_sector_cap_binds_scipy():
    from service_optimizer import PortfolioOptimizer, OptimizerConstraints
    syms, mu, Sigma, sm = _opt_inputs()
    base = OptimizerConstraints(gross_max=2.0, max_position=1.0)
    w_no = PortfolioOptimizer(objective="mean_variance", risk_aversion=0.5, constraints=base).solve(mu, Sigma)
    assert sum(abs(w_no[s]) for s in syms[:5]) > 0.30  # unconstrained piles into tech

    capd = OptimizerConstraints(gross_max=2.0, max_position=1.0,
                                sector_map=sm, sector_caps={"tech": 0.30})
    w = PortfolioOptimizer(objective="mean_variance", risk_aversion=0.5, constraints=capd).solve(mu, Sigma)
    assert sum(abs(w[s]) for s in syms[:5]) <= 0.30 + 1e-6


def test_sector_cap_numpy_fallback(monkeypatch):
    import service_optimizer as so
    from service_optimizer import PortfolioOptimizer, OptimizerConstraints
    syms, mu, Sigma, sm = _opt_inputs()
    monkeypatch.setattr(so, "_HAS_SCIPY", False)  # force numpy projection path
    capd = OptimizerConstraints(gross_max=2.0, max_position=1.0,
                                sector_map=sm, sector_caps={"tech": 0.30})
    w = PortfolioOptimizer(objective="mean_variance", risk_aversion=0.5, constraints=capd).solve(mu, Sigma)
    assert sum(abs(w[s]) for s in syms[:5]) <= 0.30 + 1e-6


# ---------------------------------------------------------------------------
# #10 — metrics
# ---------------------------------------------------------------------------
def test_metrics_have_sortino_calmar_and_benchmark():
    from core_xs_results import compute_metrics
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 200))
    b = pd.Series(rng.normal(0.0005, 0.008, 200))
    m = compute_metrics(r, benchmark=b)
    for k in ("sortino", "calmar", "information_ratio", "tracking_error", "beta", "alpha"):
        assert k in m
    # Sortino denominator (downside) <= total std -> |sortino| >= |sharpe| in general
    assert np.isfinite(m["sortino"]) and np.isfinite(m["calmar"])


# ---------------------------------------------------------------------------
# #9 — auto-reconciliation
# ---------------------------------------------------------------------------
def test_auto_reconciliation_halt_and_flatten_and_heal():
    from services.position_sync import PositionSynchronizer, SyncConfig
    local = {"AAPL": Decimal("100")}

    class P:
        def get_positions(self, symbols=None):
            return {"AAPL": {"qty": "60"}}

    halts, flats = [], []
    sync = PositionSynchronizer(
        position_provider=P(), local_state_getter=lambda: dict(local),
        config=SyncConfig(consecutive_halt_threshold=2, halt_on_unreconciled=True, auto_flatten=True),
        on_halt=lambda r: halts.append(r),
        flatten_fn=lambda s, q: flats.append((s, str(q))),
    )
    r1 = sync.sync_once()
    assert r1.has_discrepancies and not r1.halted          # detected, not yet persistent
    r2 = sync.sync_once()
    assert r2.halted and sync.should_block_new_orders()    # persistent -> halt
    assert halts and flats == [("AAPL", "60")]             # alert + auto-flatten
    local["AAPL"] = Decimal("60")                          # broker reconciled
    r3 = sync.sync_once()
    assert not r3.has_discrepancies and not sync.should_block_new_orders()  # heal clears halt


def test_no_halt_without_persistence():
    from services.position_sync import PositionSynchronizer, SyncConfig
    seq = [{"AAPL": {"qty": "60"}}, {"AAPL": {"qty": "100"}}]  # drift then heals

    class P:
        def __init__(self): self.i = 0
        def get_positions(self, symbols=None):
            v = seq[min(self.i, len(seq) - 1)]; self.i += 1; return v

    sync = PositionSynchronizer(
        position_provider=P(), local_state_getter=lambda: {"AAPL": Decimal("100")},
        config=SyncConfig(consecutive_halt_threshold=2),
    )
    sync.sync_once()                 # drift once
    r2 = sync.sync_once()            # healed before persistence threshold
    assert not r2.halted


# ---------------------------------------------------------------------------
# #11 / #12 / #10 — run_backtest factor attribution + capacity + tearsheet
# ---------------------------------------------------------------------------
def _syn_cfg():
    from service_xs_pipeline import XSConfig
    return XSConfig.model_validate({
        "data": {"source": "synthetic", "symbols": [f"S{i}" for i in range(10)], "synthetic_bars": 140},
        "universe": {"type": "static", "symbols": [f"S{i}" for i in range(10)]},
        "signals": [{"name": "m", "kind": "momentum", "lookback": 15}],
        "risk": {"type": "stat"},
        "optimizer": {"objective": "mean_variance", "gross_max": 1.0, "max_position": 0.2},
        "backtest": {"rebalance_every": 5, "cov_lookback": 40},
    })


def test_factor_attribution_tied_to_risk_model():
    from service_xs_pipeline import run_backtest
    out = run_backtest(_syn_cfg())
    fa = out["factor_attribution"]
    assert fa is not None and fa["risk_model"] == "stat"
    assert abs(fa["tie_out_residual"]) < 1e-6          # exact decomposition
    assert fa["factors"] and all(f.startswith("pc") for f in fa["factors"])


def test_capacity_in_output():
    from service_xs_pipeline import run_backtest
    out = run_backtest(_syn_cfg())
    cap = out["capacity"]
    assert cap is not None and len(cap["curve"]) > 0 and "capacity_aum" in cap


def test_tearsheet_renders_html():
    from service_xs_pipeline import run_backtest
    from service_tearsheet import render_html_tearsheet
    out = run_backtest(_syn_cfg())
    html = render_html_tearsheet(out)
    assert html.startswith("<!DOCTYPE html>")
    for section in ("Tear Sheet", "Capacity", "Factor attribution", "GIPS", "SYNTHETIC"):
        assert section in html


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
