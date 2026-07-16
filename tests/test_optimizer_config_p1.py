# -*- coding: utf-8 -*-
"""P1 #6: OptimizerCfg/build_optimizer now expose sector/factor caps, robust μ,
Black-Litterman views and multi-period (Gârleanu–Pedersen) from YAML/config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_xs_pipeline import XSConfig, build_optimizer
from service_optimizer import MultiPeriodOptimizer


def _mu_cov(symbols):
    mu = pd.Series([0.02, 0.01, 0.015, -0.005], index=symbols)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    S = A @ A.T / 10 + np.eye(4) * 0.02
    cov = pd.DataFrame(S, index=symbols, columns=symbols)
    return mu, cov


SYMS = ["A", "B", "C", "D"]


def test_sector_caps_from_config():
    mu, cov = _mu_cov(SYMS)
    cfg = XSConfig.model_validate({
        "asset_class": "equity",
        "sectors": {"A": "tech", "B": "tech", "C": "energy", "D": "energy"},
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None,
                      "sector_caps": {"tech": 0.5, "energy": 0.5}},
    })
    opt = build_optimizer(cfg)
    w = opt.solve(mu, cov)
    tech = abs(w["A"]) + abs(w["B"])
    energy = abs(w["C"]) + abs(w["D"])
    assert tech <= 0.5 + 1e-4
    assert energy <= 0.5 + 1e-4


def test_beta_neutral_from_config():
    mu, cov = _mu_cov(SYMS)
    cfg = XSConfig.model_validate({
        "asset_class": "equity",
        "universe": {"symbols": SYMS},
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None,
                      "beta_neutral": True,
                      "exposures": {"A": {"market": 1.2}, "B": {"market": 0.8},
                                    "C": {"market": 1.5}, "D": {"market": 0.5}}},
    })
    opt = build_optimizer(cfg)
    w = opt.solve(mu, cov)
    betas = {"A": 1.2, "B": 0.8, "C": 1.5, "D": 0.5}
    beta_exp = sum(betas[s] * w[s] for s in SYMS)
    assert abs(beta_exp) < 1e-2   # βᵀw ≈ 0


def test_factor_caps_from_config():
    mu, cov = _mu_cov(SYMS)
    cfg = XSConfig.model_validate({
        "asset_class": "equity",
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None,
                      "factor_caps": {"value": 0.1},
                      "exposures": {"A": {"value": 1.0}, "B": {"value": -1.0},
                                    "C": {"value": 0.5}, "D": {"value": -0.5}}},
    })
    opt = build_optimizer(cfg)
    w = opt.solve(mu, cov)
    val_exp = 1.0 * w["A"] - 1.0 * w["B"] + 0.5 * w["C"] - 0.5 * w["D"]
    assert abs(val_exp) <= 0.1 + 1e-3


def test_robust_from_config_shrinks_vs_nonrobust():
    mu, cov = _mu_cov(SYMS)
    base = XSConfig.model_validate({"asset_class": "equity",
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None}})
    rob = XSConfig.model_validate({"asset_class": "equity",
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None,
                      "robust": {"enabled": True, "kind": "box", "kappa": 2.0}}})
    w_base = build_optimizer(base).solve(mu, cov)
    w_rob = build_optimizer(rob).solve(mu, cov)
    # robust (box) penalises weights -> typically lower or equal gross exposure
    assert w_rob.abs().sum() <= w_base.abs().sum() + 1e-6


def test_bl_views_from_config():
    mu, cov = _mu_cov(SYMS)
    cfg = XSConfig.model_validate({"asset_class": "equity",
        "optimizer": {"objective": "black_litterman", "gross_max": 2.0, "net_target": None,
                      "bl_views": {"P": [[1, -1, 0, 0]], "Q": [0.05], "tau": 0.05}}})
    opt = build_optimizer(cfg)
    w = opt.solve(mu, cov)
    assert len(w) == 4 and np.isfinite(w.to_numpy()).all()


def test_multi_period_from_config_damps_turnover():
    mu, cov = _mu_cov(SYMS)
    w0 = pd.Series([0.0, 0.0, 0.0, 0.0], index=SYMS)
    single = XSConfig.model_validate({"asset_class": "equity",
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None}})
    mp = XSConfig.model_validate({"asset_class": "equity",
        "optimizer": {"objective": "mean_variance", "gross_max": 2.0, "net_target": None,
                      "multi_period": {"enabled": True, "trade_rate": 0.3}}})
    opt_mp = build_optimizer(mp)
    assert isinstance(opt_mp, MultiPeriodOptimizer)
    w_single = build_optimizer(single).solve(mu, cov, current_w=w0)
    w_mp = opt_mp.solve(mu, cov, current_w=w0)
    # multi-period moves only ~30% of the way from w0 -> smaller turnover
    assert w_mp.abs().sum() < w_single.abs().sum()


def test_backward_compatible_defaults():
    # config without any new fields builds a plain PortfolioOptimizer
    cfg = XSConfig.model_validate({"asset_class": "crypto", "optimizer": {}})
    opt = build_optimizer(cfg)
    assert not isinstance(opt, MultiPeriodOptimizer)
    mu, cov = _mu_cov(SYMS)
    w = opt.solve(mu, cov)
    assert len(w) == 4
