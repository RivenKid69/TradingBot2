# -*- coding: utf-8 -*-
"""
Stage A9 tests — anti-overfit validation + capacity + Trust Report.

  * DSR убывает с числом испытаний; PSR>0.5 для положительного Sharpe
  * expected_max_sharpe / haircut растут/убывают монотонно
  * PBO: реальный скилл → низкий; чистый шум → ~0.5
  * purged K-fold: purge/embargo исключают граничные индексы
  * capacity монотонно ухудшается с ростом AUM
  * trust_report → JSON с DSR, PBO, capacity, verdict
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from service_backtest_validation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    is_oos_degradation,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    purged_kfold_indices,
    sharpe_haircut,
    trust_report,
)
from impl_capacity import capacity_curve


# ---------------------------------------------------------------------------
# DSR / PSR
# ---------------------------------------------------------------------------
def test_deflated_sharpe_decreases_with_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 250)
    d1 = deflated_sharpe_ratio(r, n_trials=1)
    d100 = deflated_sharpe_ratio(r, n_trials=100)
    d10000 = deflated_sharpe_ratio(r, n_trials=10000)
    for d in (d1, d100, d10000):
        assert 0.0 <= d <= 1.0
    assert d1 > d100 > d10000
    assert probabilistic_sharpe_ratio(r) > 0.5  # положительный Sharpe


def test_expected_max_sharpe_monotonic():
    s2 = expected_max_sharpe(2, 1.0)
    s100 = expected_max_sharpe(100, 1.0)
    s10000 = expected_max_sharpe(10000, 1.0)
    assert s2 < s100 < s10000
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_sharpe_haircut_monotonic():
    h1 = sharpe_haircut(5.0, 1, 1.0)
    h100 = sharpe_haircut(5.0, 100, 1.0)
    h10000 = sharpe_haircut(5.0, 10000, 1.0)
    assert h1 > h100 > h10000


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------
def test_pbo_low_for_genuine_skill():
    rng = np.random.default_rng(1)
    M = rng.normal(0.0, 0.01, (80, 10))
    M[:, 0] = 0.02 + rng.normal(0.0, 0.005, 80)  # стратегия 0 реально хороша
    res = pbo_cscv(M, n_splits=8)
    assert res["pbo"] < 0.2


def test_pbo_around_half_for_noise():
    rng = np.random.default_rng(2)
    M = rng.normal(0.0, 0.01, (80, 10))  # чистый шум
    res = pbo_cscv(M, n_splits=8)
    assert 0.25 < res["pbo"] < 0.75


# ---------------------------------------------------------------------------
# Purged K-fold
# ---------------------------------------------------------------------------
def test_purged_kfold_excludes_boundary():
    folds = purged_kfold_indices(20, 5, purge=2, embargo=2)
    train, test = folds[2]  # test = [8,9,10,11]
    assert set(test.tolist()) == {8, 9, 10, 11}
    train_set = set(train.tolist())
    for i in (6, 7, 8, 9, 10, 11, 12, 13):  # purge(6,7)+test+embargo(12,13)
        assert i not in train_set
    assert 5 in train_set and 14 in train_set


def test_purged_kfold_no_purge_is_complement():
    folds = purged_kfold_indices(20, 5, purge=0, embargo=0)
    all_test = np.concatenate([t for _, t in folds])
    assert sorted(all_test.tolist()) == list(range(20))
    train, test = folds[0]
    assert set(train.tolist()) == set(range(20)) - set(test.tolist())


def test_is_oos_degradation():
    # IS: высокий Sharpe (стабильно ~0.02), OOS: явно ниже (шумно вокруг нуля)
    d = is_oos_degradation([0.02, 0.019, 0.021], [0.002, -0.001, 0.001])
    assert d["oos_sharpe"] < d["is_sharpe"]  # деградация
    assert 0 < d["degradation_ratio"] < 1


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
def test_capacity_monotonic_degradation():
    rng = np.random.default_rng(5)
    gross = rng.normal(0.001, 0.01, 60)
    turnover = np.full(60, 0.2)
    grid = [1e5, 1e6, 5e6, 1e7, 5e7, 1e8]
    res = capacity_curve(gross, turnover, adv_usd=1e7, aum_grid=grid, impact_coef=0.1)
    sharpes = [p["sharpe"] for p in res["curve"]]
    assert np.all(np.diff(sharpes) <= 1e-9)  # монотонно не растёт
    assert "capacity_aum" in res and res["capacity_aum"] >= 0
    assert np.isfinite(res["base_sharpe"])
    # avg cost растёт с AUM
    costs = [p["avg_cost_bps"] for p in res["curve"]]
    assert np.all(np.diff(costs) >= -1e-9)


# ---------------------------------------------------------------------------
# Trust Report
# ---------------------------------------------------------------------------
def test_trust_report_json():
    rng = np.random.default_rng(6)
    r = rng.normal(0.001, 0.01, 250)
    M = rng.normal(0.0, 0.01, (80, 10))
    cap = capacity_curve(r[:60], np.full(60, 0.2), adv_usd=1e7, aum_grid=[1e6, 1e8])
    rep = trust_report(r, n_trials=50, trial_performance=M, capacity=cap)

    assert {
        "deflated_sharpe",
        "probabilistic_sharpe",
        "pbo",
        "capacity",
        "verdict",
        "sharpe_annual",
    } <= set(rep)
    assert 0.0 <= rep["deflated_sharpe"] <= 1.0
    assert isinstance(rep["pbo"], float)
    assert rep["verdict"] in {"strong", "moderate", "weak", "likely_overfit", "insufficient_data"}
    json.dumps(rep)  # JSON-сериализуем
