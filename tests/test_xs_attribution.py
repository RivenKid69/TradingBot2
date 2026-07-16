# -*- coding: utf-8 -*-
"""
Stage A10 tests — service_attribution.

  * factor_attribution: сумма факторных вкладов + specific = полный P&L (tie-out)
  * без specific (r = B·f) → specific ≈ 0
  * signal_attribution: предсказательный сигнал → положительный P&L
  * brinson: allocation+selection+interaction = active return (tie-out)
  * tear_sheet → JSON-сериализуемый отчёт
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from service_attribution import (
    brinson_attribution,
    factor_attribution,
    signal_attribution,
    tear_sheet,
)

SYMS = ["A", "B", "C", "D", "E"]
TS = [1, 2]


def _exposures():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(5, 2)), index=SYMS, columns=["f1", "f2"])


# ---------------------------------------------------------------------------
# factor attribution
# ---------------------------------------------------------------------------
def test_factor_attribution_tie_out():
    rng = np.random.default_rng(1)
    B = _exposures()
    weights = pd.DataFrame(rng.normal(0, 0.3, (2, 5)), index=TS, columns=SYMS)
    asset_returns = pd.DataFrame(rng.normal(0, 0.02, (2, 5)), index=TS, columns=SYMS)
    rep = factor_attribution(weights, asset_returns, B)
    # Σ факторных вкладов + specific = total (точно)
    recombined = sum(rep["factor_pnl"].values()) + rep["specific_pnl"]
    assert abs(rep["total_pnl"] - recombined) < 1e-9
    assert abs(rep["tie_out_residual"]) < 1e-9
    assert set(rep["factor_pnl"].keys()) == {"f1", "f2"}


def test_factor_attribution_no_specific_when_returns_in_factor_span():
    B = _exposures()
    rng = np.random.default_rng(2)
    # r_t = B @ f_t  → доходности лежат в span факторов → specific = 0
    f = rng.normal(size=(2, 2))  # 2 периода × 2 фактора
    R = f @ B.to_numpy().T       # 2×5
    asset_returns = pd.DataFrame(R, index=TS, columns=SYMS)
    weights = pd.DataFrame(rng.normal(0, 0.3, (2, 5)), index=TS, columns=SYMS)
    rep = factor_attribution(weights, asset_returns, B)
    assert abs(rep["specific_pnl"]) < 1e-9
    assert abs(rep["tie_out_residual"]) < 1e-9


# ---------------------------------------------------------------------------
# signal attribution
# ---------------------------------------------------------------------------
def test_signal_attribution_positive_for_predictive():
    syms = ["A", "B", "C", "D"]
    rows = []
    rng = np.random.default_rng(3)
    for ts in (1, 2, 3):
        rets = rng.normal(0, 0.02, len(syms))
        for s, r in zip(syms, rets):
            rows.append((ts, s, r))
    idx = pd.MultiIndex.from_tuples([(t, s) for t, s, _ in rows], names=["ts_ms", "symbol"])
    fwd = pd.Series([r for _, _, r in rows], index=idx)
    signal_panel = pd.DataFrame({"good": fwd.to_numpy()}, index=idx)  # сигнал = будущий ретёрн
    att = signal_attribution(signal_panel, fwd)
    assert att["good"]["total_pnl"] > 0
    assert att["good"]["n_periods"] == 3


# ---------------------------------------------------------------------------
# brinson
# ---------------------------------------------------------------------------
def test_brinson_tie_out():
    groups = ["tech", "fin"]
    wp = pd.Series([0.6, 0.4], index=groups)
    wb = pd.Series([0.5, 0.5], index=groups)
    rp = pd.Series([0.10, 0.05], index=groups)
    rb = pd.Series([0.08, 0.06], index=groups)
    res = brinson_attribution(wp, wb, rp, rb)
    assert abs(res["tie_out_residual"]) < 1e-12
    recomb = res["allocation_total"] + res["selection_total"] + res["interaction_total"]
    assert abs(res["total_active"] - recomb) < 1e-12


# ---------------------------------------------------------------------------
# tear sheet
# ---------------------------------------------------------------------------
def test_tear_sheet_json():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.001, 0.01, 50))
    B = _exposures()
    weights = pd.DataFrame(rng.normal(0, 0.3, (2, 5)), index=TS, columns=SYMS)
    ar = pd.DataFrame(rng.normal(0, 0.02, (2, 5)), index=TS, columns=SYMS)
    fa = factor_attribution(weights, ar, B)
    sheet = tear_sheet(returns, factor=fa, trust={"deflated_sharpe": 0.8})
    assert "metrics" in sheet and "factor_attribution" in sheet
    assert "factor_pnl" in sheet["factor_attribution"]
    json.dumps(sheet)  # сериализуемо (per_period DataFrame не попадает в JSON)
