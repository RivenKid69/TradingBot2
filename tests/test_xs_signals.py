# -*- coding: utf-8 -*-
"""
Stage A4 tests — cross-sectional transforms, signal framework, IC-diagnostics.

  * zscore (mean≈0, std≈1), rank, winsorize
  * neutralize → остаток ортогонален фактору (corr≈0)
  * run_pipeline на панели
  * MomentumSignal / ColumnSignal / SignalLibrary (нормализованная панель сигналов)
  * IC (perfect=+1, anti=-1), quantile_spread, turnover, signal_report
  * устойчивость к пустому / одно-имённому юниверсу
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from impl_panel import PanelBuilder
import impl_cross_sectional as xs
import impl_signal_diagnostics as diag
from service_signals import ColumnSignal, MomentumSignal, SignalLibrary

T0_SEC = 1_700_000_000
STEP = 86_400


def _series(symbols, values):
    return pd.Series(list(values), index=list(symbols), dtype="float64")


def _mi_series(rows):
    """rows: list of (ts, symbol, value) -> MultiIndex Series."""
    idx = pd.MultiIndex.from_tuples([(t, s) for t, s, _ in rows], names=["ts_ms", "symbol"])
    return pd.Series([v for _, _, v in rows], index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def test_zscore_mean_std():
    s = _series("ABCDE", [1, 2, 3, 4, 5])
    z = xs.zscore(s)
    assert np.isclose(z.mean(), 0.0, atol=1e-12)
    assert np.isclose(z.std(ddof=0), 1.0, atol=1e-12)


def test_zscore_constant_and_single():
    assert (xs.zscore(_series("ABC", [7, 7, 7])) == 0.0).all()  # constant → нули, не NaN/inf
    assert (xs.zscore(_series("A", [5])) == 0.0).all()  # одно имя → 0


def test_rank_bounds():
    s = _series("ABCD", [10, 20, 30, 40])
    r = xs.rank(s, pct=True)
    assert r.min() > 0 and r.max() == pytest.approx(1.0)
    rc = xs.rank(s, pct=True, center=True)
    assert rc.min() >= -0.5 and rc.max() <= 0.5


def test_winsorize_clips_outlier():
    s = _series("ABCDEFGHIJ", list(range(9)) + [1000])  # выброс
    w = xs.winsorize(s, lower=0.0, upper=0.9)
    assert w.max() < 1000  # выброс обрезан


def test_neutralize_orthogonal_to_factor():
    rng = np.random.default_rng(0)
    syms = list("ABCDEFGHIJ")
    beta = _series(syms, np.linspace(-1.0, 2.0, 10))
    sig = 3.0 * beta + _series(syms, rng.normal(0, 0.1, 10))
    resid = xs.neutralize(sig, pd.DataFrame({"beta": beta}))
    corr = np.corrcoef(resid.to_numpy(), beta.to_numpy())[0, 1]
    assert abs(corr) < 1e-8  # OLS-остаток ортогонален регрессору


def test_neutralize_insufficient_points_returns_input():
    s = _series("AB", [1.0, 2.0])
    out = xs.neutralize(s, pd.DataFrame({"f": _series("AB", [1.0, 2.0])}))
    pd.testing.assert_series_equal(out, s)


# ---------------------------------------------------------------------------
# panel pipeline
# ---------------------------------------------------------------------------
def _two_ts_panel():
    frames = {}
    for sym, base in [("A", 10), ("B", 20), ("C", 30)]:
        frames[sym] = pd.DataFrame(
            {
                "timestamp": [T0_SEC, T0_SEC + STEP, T0_SEC + 2 * STEP],
                "symbol": sym,
                "close": [base, base * 1.1, base * 1.21],
                "feat": [base, base + 1, base + 2],
                "sector": ["tech", "tech", "tech"],
            }
        )
    return PanelBuilder.from_frames(frames)


def test_run_pipeline_applies_per_timestamp():
    panel = _two_ts_panel()
    out = xs.run_pipeline(panel["feat"], ["winsorize", "zscore"], factor_panel=panel)
    # zscore по каждой дате → среднее ≈ 0
    means = out.groupby(level="ts_ms").mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-9)


def test_momentum_signal_values():
    frame = pd.DataFrame(
        {
            "timestamp": [T0_SEC, T0_SEC + STEP, T0_SEC + 2 * STEP],
            "symbol": "A",
            "close": [100.0, 110.0, 121.0],
        }
    )
    panel = PanelBuilder.from_frames({"A": frame})
    mom = MomentumSignal("mom", lookback=1, skip=0).compute_panel(panel)
    vals = mom.to_numpy()
    assert np.isnan(vals[0])
    assert vals[1] == pytest.approx(0.10)
    assert vals[2] == pytest.approx(0.10)


def test_signal_library_normalized_panel_and_report():
    panel = _two_ts_panel()
    lib = SignalLibrary()
    lib.register(ColumnSignal("alpha", "feat"), transforms=["zscore"])
    sig_panel = lib.compute(panel)
    assert "alpha" in sig_panel.columns
    # нормализовано по дате
    means = sig_panel["alpha"].groupby(level="ts_ms").mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-9)
    # IC-отчёт считается (acceptance: нормализованный сигнал + IC)
    fwd = PanelBuilder.add_forward_returns(panel, price_col="close")["fwd_return"]
    rep = diag.signal_report(sig_panel["alpha"], fwd)
    assert set(rep) >= {"ic_mean", "ic_ir", "quantile_spread", "turnover"}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_ic_perfect_and_anti():
    sig = _mi_series([(1, "A", 1), (1, "B", 2), (1, "C", 3), (2, "A", 3), (2, "B", 1), (2, "C", 2)])
    ic_perfect = diag.information_coefficient(sig, sig)
    assert ic_perfect["ic_mean"] == pytest.approx(1.0)
    ic_anti = diag.information_coefficient(sig, -sig)
    assert ic_anti["ic_mean"] == pytest.approx(-1.0)


def test_quantile_spread_positive_when_predictive():
    rows = []
    for ts in (1, 2):
        for sym, sv, fv in [("A", 1, 10), ("B", 2, 20), ("C", 3, 30), ("D", 4, 40)]:
            rows.append((ts, sym, sv))
    sig = _mi_series(rows)
    fwd = _mi_series([(ts, s, v * 10) for ts, s, v in rows])
    qs = diag.quantile_spread(sig, fwd, n_quantiles=2)
    assert qs["spread"] > 0


def test_turnover_zero_when_ranking_stable():
    stable = _mi_series(
        [(1, "A", 1), (1, "B", 2), (1, "C", 3), (2, "A", 1), (2, "B", 2), (2, "C", 3)]
    )
    rev = _mi_series([(1, "A", 1), (1, "B", 2), (1, "C", 3), (2, "A", 3), (2, "B", 2), (2, "C", 1)])
    assert diag.turnover(stable)["turnover_mean"] == pytest.approx(0.0, abs=1e-9)
    assert diag.turnover(rev)["turnover_mean"] > diag.turnover(stable)["turnover_mean"]


def test_diagnostics_robust_to_empty_and_single_name():
    empty = pd.Series(
        dtype="float64", index=pd.MultiIndex.from_tuples([], names=["ts_ms", "symbol"])
    )
    ic = diag.information_coefficient(empty, empty)
    assert ic["n_periods"] == 0 and np.isnan(ic["ic_mean"])
    # один символ на дату → корреляция не определена, но без падения
    single = _mi_series([(1, "A", 1.0), (2, "A", 2.0)])
    ic2 = diag.information_coefficient(single, single)
    assert ic2["n_periods"] == 0
