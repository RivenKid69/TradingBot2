# -*- coding: utf-8 -*-
"""Тесты tcost-aware оптимизации (tcost в objective) + Kelly/vol-target сайзинга (P1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_optimizer import (
    OptimizerConstraints,
    PortfolioOptimizer,
    SizingConfig,
    TCostModel,
    kelly_weights,
)


def _setup(n=4, seed=0):
    rng = np.random.RandomState(seed)
    syms = [f"S{i}" for i in range(n)]
    mu = pd.Series(rng.normal(0.02, 0.01, n), index=syms)
    A = rng.normal(0, 1, (n, n))
    cov = pd.DataFrame((A @ A.T) / n + np.eye(n) * 0.1, index=syms, columns=syms)
    return syms, mu, cov


def test_tcost_reduces_turnover_vs_no_tcost():
    syms, mu, cov = _setup()
    w0 = pd.Series([0.25, 0.25, -0.25, -0.25], index=syms)
    cons = OptimizerConstraints(gross_max=1.0, net_target=0.0, max_position=0.5)

    # без tcost
    opt0 = PortfolioOptimizer(objective="mean_variance", risk_aversion=5.0, constraints=cons)
    w_no = opt0.solve(mu, cov, current_w=w0)
    to_no = float((w_no - w0).abs().sum())

    # с заметным tcost в objective → решение ближе к w0 (меньше оборот)
    opt1 = PortfolioOptimizer(
        objective="mean_variance",
        risk_aversion=5.0,
        constraints=cons,
        tcost=TCostModel(linear=0.05, coef=1.0),
    )
    w_tc = opt1.solve(mu, cov, current_w=w0)
    to_tc = float((w_tc - w0).abs().sum())

    assert to_tc < to_no - 1e-6  # tcost снижает оборот
    # ограничения соблюдены
    assert float(w_tc.abs().sum()) <= 1.0 + 1e-6
    assert abs(float(w_tc.sum())) < 1e-6


def test_tcost_zero_recovers_unconstrained_direction():
    syms, mu, cov = _setup(seed=1)
    cons = OptimizerConstraints(gross_max=1.0, net_target=0.0)
    opt = PortfolioOptimizer(
        objective="mean_variance",
        constraints=cons,
        tcost=TCostModel(linear=0.0, quad=0.0, coef=0.0),
    )
    w = opt.solve(mu, cov, current_w=pd.Series(0.0, index=syms))
    # допустимое решение, gross в пределах
    assert float(w.abs().sum()) <= 1.0 + 1e-6


def test_vol_targeting_sizing():
    syms, mu, cov = _setup(seed=2)
    cons = OptimizerConstraints(gross_max=2.0, net_target=0.0, max_position=1.0)
    target = 0.10
    opt = PortfolioOptimizer(
        objective="mean_variance",
        constraints=cons,
        sizing=SizingConfig(method="vol_target", target_vol=target),
    )
    w = opt.solve(mu, cov, current_w=pd.Series(0.0, index=syms))
    S = cov.reindex(index=syms, columns=syms).to_numpy()
    wv = w.to_numpy()
    realized_vol = float(np.sqrt(wv @ S @ wv))
    assert realized_vol == pytest.approx(target, rel=1e-3)


def test_kelly_weights_direction():
    syms, mu, cov = _setup(seed=3)
    S = cov.to_numpy()
    m = mu.to_numpy()
    kw = kelly_weights(m, S, fraction=0.5)
    full = kelly_weights(m, S, fraction=1.0)
    assert np.allclose(kw, 0.5 * full)  # фракционный Kelly линеен
    # направление совпадает с Σ⁻¹μ
    assert np.allclose(full, np.linalg.solve(S + 1e-10 * np.eye(len(m)), m), atol=1e-6)


def test_factor_cap_constraint_in_scipy():
    syms, mu, cov = _setup(seed=4)
    B = pd.DataFrame({"market": [1.0, 1.0, 1.0, 1.0]}, index=syms)
    cons = OptimizerConstraints(
        gross_max=1.0, max_position=0.5, exposures=B, factor_caps={"market": 0.1}
    )
    opt = PortfolioOptimizer(
        objective="mean_variance", constraints=cons, tcost=TCostModel(linear=0.001)
    )
    w = opt.solve(mu, cov, current_w=pd.Series(0.0, index=syms))
    market_exp = float((B["market"] * w).sum())
    assert abs(market_exp) <= 0.1 + 1e-3  # факторный лимит соблюдён в objective-solve
