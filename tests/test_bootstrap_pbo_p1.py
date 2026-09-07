# -*- coding: utf-8 -*-
"""P1 #8: block-bootstrap CIs (Politis–Romano) + CPCV/PBO wiring into trust_report."""

from __future__ import annotations

import numpy as np
import pytest

from research.bootstrap import block_bootstrap, bootstrap_report, sharpe, max_drawdown
from service_backtest_validation import trust_report


def _series(mu, sd, n, seed=0):
    return list(np.random.default_rng(seed).normal(mu, sd, n))


# --------------------------------------------------------------------------- bootstrap
def test_block_bootstrap_positive_edge_ci_excludes_zero():
    r = _series(0.0015, 0.01, 1500, seed=1)  # strong positive drift
    bs = block_bootstrap(r, lambda x: sharpe(x, 252.0), n_boot=1500)
    assert bs["ci_low"] > 0.0  # significant edge
    assert bs["p_value"] < 0.05  # P[sharpe<=0] small
    assert bs["ci_low"] < bs["point"] < bs["ci_high"]


def test_block_bootstrap_noise_not_significant():
    r = _series(0.0, 0.01, 1500, seed=2)  # zero drift
    bs = block_bootstrap(r, lambda x: sharpe(x, 252.0), n_boot=1500)
    # noise must NOT register a significant POSITIVE edge (lower CI bound not > 0)
    assert bs["ci_low"] <= 0.0
    assert bs["p_value"] > 0.05  # not significantly positive


def test_bootstrap_report_keys():
    r = _series(0.001, 0.012, 800, seed=3)
    rep = bootstrap_report(r, n_boot=800)
    assert set(rep) == {"sharpe", "cagr", "max_drawdown"}
    for k in rep:
        assert {"point", "ci_low", "ci_high", "p_value", "se"} <= set(rep[k])


def test_circular_method():
    r = _series(0.001, 0.01, 600, seed=4)
    bs = block_bootstrap(r, max_drawdown, n_boot=500, method="circular")
    assert bs["n_boot"] > 0 and bs["ci_high"] >= bs["ci_low"]


def test_short_series_safe():
    bs = block_bootstrap([0.01, 0.02], sharpe, n_boot=100)
    assert bs["n_boot"] == 0


# --------------------------------------------------------------------------- trust_report
def test_trust_report_includes_bootstrap():
    r = _series(0.0012, 0.01, 1000, seed=5)
    rep = trust_report(r, n_trials=4, bootstrap=True, bootstrap_n=800)
    assert "bootstrap" in rep and "sharpe" in rep["bootstrap"]
    assert "sharpe_ci_excludes_zero" in rep
    assert rep["sharpe_bootstrap_pvalue"] is not None


def test_trust_report_pbo_from_matrix():
    # T×N OOS path matrix: one column has real edge, rest are noise
    rng = np.random.default_rng(7)
    T, N = 600, 8
    mat = rng.normal(0.0, 0.01, (T, N))
    mat[:, 0] += 0.0015  # variant 0 has a real edge
    rep = trust_report(list(mat[:, 0]), n_trials=N, trial_performance=mat, bootstrap=False)
    assert rep["pbo"] is not None and 0.0 <= rep["pbo"] <= 1.0


def test_trust_report_bootstrap_can_be_off():
    r = _series(0.001, 0.01, 500, seed=8)
    rep = trust_report(r, bootstrap=False)
    assert rep.get("bootstrap") in (None,) or "bootstrap" not in rep
