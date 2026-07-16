# -*- coding: utf-8 -*-
"""
Stage A8 tests — cross-sectional backtest engine.

  * compute_metrics: корректные Sharpe / maxDD / total return
  * end-to-end: oracle-сигнал → положительная equity (long-short) [acceptance]
  * детерминизм: одинаковый прогон → идентичный результат
  * leakage-probe: панели, совпадающие до t*, дают ИДЕНТИЧНЫЕ веса на t*
    (causal momentum-сигнал → нет look-ahead)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from impl_universe import StaticUniverse
from service_signals import SignalLibrary, ColumnSignal, MomentumSignal
from service_alpha import EqualWeightAlpha
from service_risk_model import StatRiskModel
from service_optimizer import OptimizerConstraints, PortfolioOptimizer
from service_xs_backtest import CrossSectionalBacktest, XSBacktestConfig
from core_xs_results import compute_metrics

T0 = 1_700_000_000
STEP = 86_400


def _walk(rng, T, syms):
    closes = {}
    for s in syms:
        r = rng.normal(0.0, 0.02, T)
        closes[s] = 100.0 * np.cumprod(1.0 + r)
    return closes


def _panel(closes, *, with_oracle=False):
    T = len(next(iter(closes.values())))
    ts = [T0 + i * STEP for i in range(T)]
    frames = {}
    for s, c in closes.items():
        data = {"timestamp": ts, "symbol": s, "close": np.asarray(c, dtype="float64")}
        if with_oracle:
            c = np.asarray(c, dtype="float64")
            oracle = np.empty(T)
            oracle[:-1] = c[1:] / c[:-1] - 1.0  # доходность следующего бара (известна как фича)
            oracle[-1] = np.nan
            data["oracle"] = oracle
        frames[s] = pd.DataFrame(data)
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_compute_metrics():
    r = pd.Series([0.10, -0.05, 0.10])
    m = compute_metrics(r, periods_per_year=252)
    assert m["total_return"] == pytest.approx(1.1 * 0.95 * 1.1 - 1.0)
    assert m["max_drawdown"] == pytest.approx(-0.05, abs=1e-9)
    assert m["n_periods"] == 3
    assert np.isfinite(m["sharpe"])


def test_compute_metrics_empty():
    m = compute_metrics(pd.Series(dtype="float64"))
    assert m["n_periods"] == 0
    assert np.isnan(m["sharpe"])


# ---------------------------------------------------------------------------
# end-to-end (oracle → положительная equity)
# ---------------------------------------------------------------------------
def _build_engine(syms, *, signal):
    lib = SignalLibrary().register(signal, transforms=["zscore"])
    opt = PortfolioOptimizer(
        objective="mean_variance", use_cvxpy="never",
        constraints=OptimizerConstraints(net_target=0.0, gross_max=1.0),  # market-neutral
    )
    return CrossSectionalBacktest(
        universe=StaticUniverse(syms),
        alpha_model=EqualWeightAlpha(),
        risk_model=StatRiskModel(method="ledoit_wolf"),
        optimizer=opt,
        signal_library=lib,
        config=XSBacktestConfig(
            cov_lookback=10, min_cov_obs=5, cost_bps=0.0, rebalance_every=1
        ),
    )


def test_end_to_end_oracle_positive_equity():
    rng = np.random.default_rng(7)
    syms = [f"S{i}" for i in range(6)]
    panel = _panel(_walk(rng, 40, syms), with_oracle=True)
    eng = _build_engine(syms, signal=ColumnSignal("oracle", "oracle"))
    res = eng.run(panel)

    assert res.weights.shape[0] > 10          # были ребалансы
    assert res.nav.iloc[-1] > 1.0             # equity выросла (oracle предсказывает)
    assert res.metrics["total_return"] > 0
    assert res.metrics["sharpe"] > 0
    # market-neutral: net ≈ 0 на каждом ребалансе
    assert np.allclose(res.net.to_numpy(), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_determinism():
    rng = np.random.default_rng(11)
    syms = [f"S{i}" for i in range(5)]
    panel = _panel(_walk(rng, 35, syms), with_oracle=True)
    r1 = _build_engine(syms, signal=ColumnSignal("oracle", "oracle")).run(panel)
    r2 = _build_engine(syms, signal=ColumnSignal("oracle", "oracle")).run(panel)
    pd.testing.assert_series_equal(r1.returns, r2.returns)
    pd.testing.assert_frame_equal(r1.weights, r2.weights)


# ---------------------------------------------------------------------------
# leakage-probe: данные, совпадающие до t*, дают идентичные веса на t*
# ---------------------------------------------------------------------------
def test_no_lookahead_weights_identical_before_divergence():
    rng = np.random.default_rng(3)
    syms = [f"S{i}" for i in range(5)]
    base = _walk(rng, 30, syms)
    div_idx = 25  # цены расходятся ПОСЛЕ этого бара

    closes_A = {s: c.copy() for s, c in base.items()}
    closes_B = {s: c.copy() for s, c in base.items()}
    for s in syms:
        closes_B[s][div_idx + 1:] *= 1.5  # будущее B отличается

    panel_A = _panel(closes_A)
    panel_B = _panel(closes_B)

    def _engine():
        lib = SignalLibrary().register(MomentumSignal("mom", lookback=3), transforms=["zscore"])
        opt = PortfolioOptimizer(
            objective="mean_variance", use_cvxpy="never",
            constraints=OptimizerConstraints(net_target=0.0, gross_max=1.0),
        )
        return CrossSectionalBacktest(
            universe=StaticUniverse(syms),
            alpha_model=EqualWeightAlpha(),
            risk_model=StatRiskModel(method="ledoit_wolf"),
            optimizer=opt,
            signal_library=lib,
            config=XSBacktestConfig(cov_lookback=8, min_cov_obs=5, cost_bps=0.0),
        )

    res_A = _engine().run(panel_A)
    res_B = _engine().run(panel_B)

    t_star = (T0 + 20 * STEP) * 1000  # ребаланс ДО точки расхождения (панель в мс)
    assert t_star in res_A.weights.index and t_star in res_B.weights.index
    wa = res_A.weights.loc[t_star].reindex(syms).fillna(0.0)
    wb = res_B.weights.loc[t_star].reindex(syms).fillna(0.0)
    assert np.allclose(wa.to_numpy(), wb.to_numpy(), atol=1e-12)  # нет look-ahead
