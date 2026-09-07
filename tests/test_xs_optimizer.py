# -*- coding: utf-8 -*-
"""
Stage A7 tests — service_optimizer (analytic + projection fallback, без cvxpy).

  * аналитические кейсы: equal-weight, min-variance (inverse-var), tangency (∝Σ⁻¹μ),
    risk-parity (∝1/σ), market-neutral (Σw=0), Black-Litterman без views == MVO
  * жёсткие ограничения не нарушаются: gross / net / box / turnover / long-only
  * fallback работает без cvxpy (use_cvxpy='never')
  * соответствие core_portfolio.PortfolioConstructor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from service_optimizer import OptimizerConstraints, PortfolioOptimizer

SYMS = ["A", "B", "C"]


def _cov(diag):
    return pd.DataFrame(
        np.diag(diag).astype("float64"), index=SYMS[: len(diag)], columns=SYMS[: len(diag)]
    )


def _mu(vals, syms=SYMS):
    return pd.Series(np.asarray(vals, dtype="float64"), index=syms[: len(vals)])


def _opt(objective, **cons):
    return PortfolioOptimizer(
        objective=objective, use_cvxpy="never", constraints=OptimizerConstraints(**cons)
    )


# ---------------------------------------------------------------------------
# analytic cases
# ---------------------------------------------------------------------------
def test_equal_weight():
    opt = _opt("equal_weight", net_target=1.0, long_only=True)
    w = opt.solve(_mu([0, 0, 0]), _cov([1, 1, 1]))
    assert isinstance(w, pd.Series) and list(w.index) == SYMS
    assert np.allclose(w.to_numpy(), 1 / 3)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_inverse_var():
    opt = _opt("min_variance", net_target=1.0)
    w = opt.solve(_mu([0, 0, 0]), _cov([1, 4, 9]))
    inv = np.array([1.0, 0.25, 1.0 / 9])
    expected = inv / inv.sum()
    assert np.allclose(w.to_numpy(), expected, atol=1e-8)


def test_tangency_proportional_to_mu():
    opt = _opt("max_sharpe", net_target=1.0)
    w = opt.solve(_mu([1, 2, 3]), _cov([1, 1, 1]))  # Σ=I → w ∝ μ
    assert np.allclose(w.to_numpy(), np.array([1, 2, 3]) / 6.0, atol=1e-8)


def test_risk_parity_inverse_sigma():
    opt = _opt("risk_parity", net_target=1.0)
    w = opt.solve(_mu([0, 0, 0]), _cov([1, 4, 9]))  # σ=[1,2,3] → w ∝ 1/σ
    inv = np.array([1.0, 0.5, 1.0 / 3])
    assert np.allclose(w.to_numpy(), inv / inv.sum(), atol=1e-3)
    # равный вклад в риск (диагональ): w_i^2 σ_i^2
    rc = (w.to_numpy() ** 2) * np.array([1, 4, 9])
    assert np.allclose(rc, rc.mean(), rtol=1e-2)


def test_market_neutral_centering():
    opt = _opt("max_sharpe", net_target=0.0)
    w = opt.solve(_mu([1, 2, 3]), _cov([1, 1, 1]))
    assert w.sum() == pytest.approx(0.0, abs=1e-9)
    assert w["C"] > w["B"] > w["A"]  # порядок ∝ μ сохранён


def test_black_litterman_no_views_equals_mvo():
    mu, cov = _mu([1, 2, 3]), _cov([1, 1, 1])
    w_mvo = _opt("mean_variance", net_target=1.0).solve(mu, cov)
    bl = PortfolioOptimizer(
        objective="black_litterman",
        use_cvxpy="never",
        constraints=OptimizerConstraints(net_target=1.0),
        bl_views={},
    )
    w_bl = bl.solve(mu, cov)
    assert np.allclose(w_mvo.to_numpy(), w_bl.to_numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# hard constraints
# ---------------------------------------------------------------------------
def test_gross_cap_enforced():
    opt = _opt("mean_variance", gross_max=1.5)
    w = opt.solve(_mu([1, -2, 3]), _cov([1, 1, 1]))
    assert np.abs(w.to_numpy()).sum() <= 1.5 + 1e-9
    # направление ∝ [1,-2,3] сохранено
    ratios = w.to_numpy() / np.array([1, -2, 3])
    assert np.allclose(ratios, ratios[0], atol=1e-8)


def test_box_cap_enforced():
    opt = _opt("mean_variance", max_position=0.4, long_only=True)
    w = opt.solve(_mu([5, 10, 15]), _cov([1, 1, 1]))  # большой μ → бьёт по границе
    assert w.max() <= 0.4 + 1e-9
    assert (w.to_numpy() >= -1e-12).all()


def test_net_target_enforced():
    opt = _opt("equal_weight", net_target=0.8, long_only=True)
    w = opt.solve(_mu([0, 0, 0]), _cov([1, 1, 1]))
    assert w.sum() == pytest.approx(0.8, abs=1e-9)


def test_turnover_cap_enforced():
    opt = _opt("max_sharpe", net_target=1.0, max_turnover=0.03)
    w0 = pd.Series([0.2, 0.3, 0.5], index=SYMS)
    w = opt.solve(_mu([1, 2, 3]), _cov([1, 1, 1]), current_w=w0)
    turnover = np.abs(w.to_numpy() - w0.to_numpy()).sum()
    assert turnover <= 0.03 + 1e-9


def test_long_only_enforced():
    opt = _opt("mean_variance", long_only=True, net_target=1.0)
    w = opt.solve(_mu([1, -5, 2]), _cov([1, 1, 1]))
    assert (w.to_numpy() >= -1e-12).all()


def test_is_portfolio_constructor():
    opt = _opt("min_variance", net_target=1.0)
    assert isinstance(opt, cp.PortfolioConstructor)
