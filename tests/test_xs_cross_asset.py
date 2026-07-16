# -*- coding: utf-8 -*-
"""
Stage C1 tests — unified cross-asset portfolio (service_cross_asset).

  * валютная нормализация: r_base = (1+r_local)(1+r_fx)−1 (корректно)
  * joint Σ симметрична и PSD (eigenvalues ≥ 0)
  * combine ≥2 классов: веса охватывают оба, class-allocations Σ=1, общий vol-target достигнут
  * risk_parity даёт больше низковолатильному классу
  * block_from_xs_config + combine_from_configs end-to-end (crypto+equity вертикали)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from service_cross_asset import (
    normalize_returns_to_base, AssetClassBlock, build_cross_asset_cov,
    combine_cross_asset, block_from_xs_config, combine_from_configs,
)
from service_xs_pipeline import XSConfig


def _rw(symbols, n, scale, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    return pd.DataFrame({s: rng.normal(0.0, scale, n) for s in symbols}, index=idx)


# ---------------------------------------------------------------------------
# currency normalization
# ---------------------------------------------------------------------------
def test_currency_normalization_formula():
    idx = [0, 1, 2]
    rw = pd.DataFrame({"BUND": [0.01, -0.02, 0.005]}, index=idx)  # local EUR returns
    fx = {"EUR": pd.Series([0.003, 0.001, -0.002], index=idx)}    # EUR vs USD returns
    out = normalize_returns_to_base(rw, currency_map={"BUND": "EUR"}, fx_returns=fx, base="USD")
    exp = (1 + rw["BUND"]) * (1 + fx["EUR"]) - 1.0
    assert np.allclose(out["BUND"].to_numpy(), exp.to_numpy())


def test_currency_base_passthrough():
    rw = pd.DataFrame({"SPY": [0.01, 0.02]}, index=[0, 1])
    out = normalize_returns_to_base(rw, currency_map={"SPY": "USD"},
                                    fx_returns={"EUR": pd.Series([0.5, 0.5], index=[0, 1])}, base="USD")
    assert np.allclose(out["SPY"].to_numpy(), rw["SPY"].to_numpy())  # base → без изменений


# ---------------------------------------------------------------------------
# joint covariance PSD
# ---------------------------------------------------------------------------
def test_joint_cov_psd_and_symmetric():
    b1 = AssetClassBlock("crypto", pd.Series({"BTC": 0.6, "ETH": -0.4}), _rw(["BTC", "ETH"], 120, 0.03, 1))
    b2 = AssetClassBlock("equity", pd.Series({"AAPL": 0.5, "XOM": -0.5}), _rw(["AAPL", "XOM"], 120, 0.01, 2))
    cov = build_cross_asset_cov([b1, b2])
    assert set(cov.index) == {"BTC", "ETH", "AAPL", "XOM"}
    C = cov.to_numpy()
    assert np.allclose(C, C.T, atol=1e-10)                 # симметрична
    eig = np.linalg.eigvalsh(C)
    assert eig.min() >= -1e-8                                # PSD


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------
def test_combine_hits_vol_target_and_spans_classes():
    b1 = AssetClassBlock("crypto", pd.Series({"BTC": 0.6, "ETH": -0.4}), _rw(["BTC", "ETH"], 150, 0.03, 3))
    b2 = AssetClassBlock("equity", pd.Series({"AAPL": 0.5, "XOM": -0.5}), _rw(["AAPL", "XOM"], 150, 0.012, 4))
    res = combine_cross_asset([b1, b2], target_vol=0.10, periods_per_year=252)
    assert set(res.weights.index) == {"BTC", "ETH", "AAPL", "XOM"}      # оба класса
    assert sum(res.class_allocations.values()) == pytest.approx(1.0)
    assert res.port_vol_annual == pytest.approx(0.10, abs=1e-6)         # общий vol-target


def test_risk_parity_favors_low_vol_class():
    low = AssetClassBlock("equity", pd.Series({"A": 0.5, "B": -0.5}), _rw(["A", "B"], 150, 0.005, 5))
    high = AssetClassBlock("crypto", pd.Series({"C": 0.5, "D": -0.5}), _rw(["C", "D"], 150, 0.05, 6))
    res = combine_cross_asset([low, high], target_vol=0.10, class_weighting="risk_parity")
    assert res.class_allocations["equity"] > res.class_allocations["crypto"]  # inverse-vol


def test_equal_class_weighting():
    b1 = AssetClassBlock("crypto", pd.Series({"BTC": 1.0}), _rw(["BTC"], 120, 0.03, 7))
    b2 = AssetClassBlock("equity", pd.Series({"AAPL": 1.0}), _rw(["AAPL"], 120, 0.01, 8))
    res = combine_cross_asset([b1, b2], class_weighting="equal")
    assert res.class_allocations["crypto"] == pytest.approx(0.5)
    assert res.class_allocations["equity"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# end-to-end from vertical configs
# ---------------------------------------------------------------------------
def _vert_cfg(asset_class, symbols, kind, seed):
    return XSConfig.model_validate({
        "asset_class": asset_class,
        "data": {"source": "synthetic", "symbols": symbols, "synthetic_bars": 160, "synthetic_seed": seed},
        "universe": {"type": "static", "symbols": symbols},
        "signals": [{"name": "m", "kind": kind, "lookback": 30, "transforms": ["zscore"]}],
        "alpha": {"method": "ic_weighted"}, "risk": {"type": "stat", "method": "ledoit_wolf"},
        "optimizer": {"objective": "mean_variance", "risk_aversion": 5.0, "gross_max": 1.0,
                      "net_target": 0.0, "long_only": False, "max_position": 0.5},
        "backtest": {"rebalance_every": 5, "cov_lookback": 40, "min_cov_obs": 20,
                     "alpha_refit_every": 5, "cost_bps": 3.0, "price_col": "close", "periods_per_year": 252},
    })


def test_block_from_config_and_combine():
    crypto = _vert_cfg("crypto", ["BTC", "ETH", "SOL", "BNB"], "crypto_momentum", 11)
    equity = _vert_cfg("equity", ["AAPL", "MSFT", "NVDA", "XOM"], "equity_momentum", 12)
    b_c = block_from_xs_config(crypto, name="crypto")
    assert b_c.name == "crypto" and not b_c.returns_wide.empty
    res = combine_from_configs({"crypto": crypto, "equity": equity},
                               target_vol=0.12, class_weighting="risk_parity")
    assert sum(res.class_allocations.values()) == pytest.approx(1.0)
    # ≥2 классов в портфеле, общий риск-таргет (acceptance)
    assert len(res.class_allocations) == 2
    if res.port_vol_annual > 0:
        assert res.port_vol_annual == pytest.approx(0.12, abs=1e-6)


def test_api_cross_asset_demo():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xs_api import register_xs_routes

    app = FastAPI()
    register_xs_routes(app)
    client = TestClient(app)
    r = client.post("/api/xs/cross_asset", json={"demo": True, "target_vol": 0.10, "class_weighting": "risk_parity"})
    assert r.status_code == 200
    data = r.json()
    assert sum(data["class_allocations"].values()) == pytest.approx(1.0, abs=1e-6)
    assert data["n_names"] > 0
    assert len(data["class_allocations"]) == 4  # crypto+equity+futures+forex
