# -*- coding: utf-8 -*-
"""
Stage B4 tests — forex-вертикаль (signals/forex_signals + xs_risk/forex_factors + pipeline).

  * carry из rate_diff / rate_base-rate_quote (+ приоритет готовой колонки); graceful (BYO)
  * fx_momentum / fx_value (PPP-колонка ИЛИ прокси mean-reversion); знаки корректны
  * usd_beta восстанавливает известную β; build_forex_exposures — корректная B (минимум факторов)
  * МАЛЫЙ юниверс (3-4 пары) НЕ ломает оптимизатор/факторную Σ
  * end-to-end carry/momentum контур по пресету config_xs_forex.yaml → Trust Report (acceptance)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from signals.forex_signals import FXCarry, FXMomentum, FXValue, TermsOfTrade
from xs_risk.forex_factors import usd_beta, build_forex_exposures
from service_xs_pipeline import XSConfig, run_backtest, build_signal_library

T0, STEP = 1_700_000_000, 86_400
FOREX_CFG = os.path.join("configs", "config_xs_forex.yaml")


def _fx_frame(sym, base, n, rate_base=None, rate_quote=None):
    ts = [T0 + i * STEP for i in range(n)]
    close = base * (1.0 + 0.003 * np.arange(n))
    d = {"timestamp": ts, "symbol": sym, "close": close}
    if rate_base is not None:
        d["rate_base"] = rate_base
    if rate_quote is not None:
        d["rate_quote"] = rate_quote
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def test_fx_carry_from_rate_diff_and_ready_column():
    # rate_base − rate_quote
    panel = PanelBuilder.from_frames({"AUDUSD": _fx_frame("AUDUSD", 0.66, 4, rate_base=0.045, rate_quote=0.01)})
    c = FXCarry("c").compute_panel(panel)
    assert c.iloc[0] == pytest.approx(0.045 - 0.01)
    # готовая rate_diff имеет приоритет
    f = _fx_frame("AUDUSD", 0.66, 4); f["rate_diff"] = 0.02
    panel2 = PanelBuilder.from_frames({"AUDUSD": f})
    assert FXCarry("c").compute_panel(panel2).iloc[0] == pytest.approx(0.02)


def test_fx_momentum_and_value_proxy_signs():
    panel = PanelBuilder.from_frames({"EURUSD": _fx_frame("EURUSD", 1.10, 30)})
    mom = FXMomentum("m", lookback=10).compute_panel(panel)
    val = FXValue("v", lookback=10).compute_panel(panel)   # нет ppp → прокси −long-return
    bar = (T0 + 12 * STEP) * 1000
    close0, close10, close12 = 1.10, 1.10 * (1 + 0.003 * 10), 1.10 * (1 + 0.003 * 12)
    exp = close12 / (1.10 * (1 + 0.003 * 2)) - 1.0
    assert mom.loc[(bar, "EURUSD")] == pytest.approx(exp)
    assert val.loc[(bar, "EURUSD")] == pytest.approx(-exp)


def test_fx_value_prefers_ppp_column():
    f = _fx_frame("EURUSD", 1.10, 4); f["ppp"] = 0.07
    panel = PanelBuilder.from_frames({"EURUSD": f})
    assert FXValue("v").compute_panel(panel).iloc[0] == pytest.approx(0.07)


def test_signals_graceful_missing_column():
    panel = PanelBuilder.from_frames({"EURUSD": _fx_frame("EURUSD", 1.10, 3)})
    assert FXCarry("c").compute_panel(panel).isna().all()      # нет ставок → BYO пуст
    assert TermsOfTrade("t").compute_panel(panel).isna().all()  # нет ToT → BYO пуст


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def test_usd_beta_recovers_known():
    rng = np.random.default_rng(0)
    usd = rng.normal(0, 0.005, 150)
    rw = pd.DataFrame({"DXY": usd, "HI2X": 2.0 * usd, "NOISE": rng.normal(0, 0.005, 150)})
    beta = usd_beta(rw, usd_symbol="DXY")
    assert beta["DXY"] == pytest.approx(1.0, abs=1e-9)
    assert beta["HI2X"] == pytest.approx(2.0, abs=1e-9)


def test_build_forex_exposures_minimal():
    rng = np.random.default_rng(1)
    rw = pd.DataFrame({s: rng.normal(0, 0.005, 100) for s in ["EURUSD", "GBPUSD", "AUDUSD"]})
    B = build_forex_exposures(
        rw, carries={"EURUSD": 0.0, "GBPUSD": 0.01, "AUDUSD": 0.04},
        values={"EURUSD": 0.05, "GBPUSD": 0.02, "AUDUSD": 0.08},
        usd_symbol=None,
    )
    for col in ("usd_beta", "carry", "value"):
        assert col in B.columns
    assert list(B.index) == ["EURUSD", "GBPUSD", "AUDUSD"]


# ---------------------------------------------------------------------------
# малый юниверс не ломает оптимизатор/факторную Σ
# ---------------------------------------------------------------------------
def test_small_universe_does_not_break_optimizer():
    cfg = XSConfig.model_validate({
        "asset_class": "forex",
        "data": {"source": "synthetic", "symbols": ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"], "synthetic_bars": 200},
        "signals": [{"name": "m", "kind": "fx_momentum", "lookback": 60, "transforms": ["zscore"]},
                    {"name": "v", "kind": "fx_value", "lookback": 120, "transforms": ["zscore"]}],
        "alpha": {"method": "ic_weighted"},
        "risk": {"type": "forex_factor", "method": "ledoit_wolf"},
        "optimizer": {"objective": "mean_variance", "risk_aversion": 5.0, "gross_max": 1.0,
                      "net_target": 0.0, "long_only": False, "max_position": 0.5},
        "backtest": {"rebalance_every": 5, "cov_lookback": 40, "min_cov_obs": 20,
                     "alpha_refit_every": 5, "cost_bps": 1.0, "price_col": "close", "periods_per_year": 252},
    })
    out = run_backtest(cfg)
    assert out["n_rebalances"] > 0
    assert np.allclose(out["result"].net.to_numpy(), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_builds_forex_signals():
    cfg = XSConfig.model_validate({
        "asset_class": "forex",
        "data": {"source": "synthetic", "symbols": ["EURUSD", "GBPUSD"]},
        "signals": [
            {"name": "carry", "kind": "fx_carry"},
            {"name": "mom", "kind": "fx_momentum", "lookback": 90},
            {"name": "val", "kind": "fx_value", "lookback": 250},
        ],
    })
    lib = build_signal_library(cfg)
    assert lib.names == ["carry", "mom", "val"]


# ---------------------------------------------------------------------------
# end-to-end (acceptance)
# ---------------------------------------------------------------------------
def test_forex_preset_end_to_end():
    import yaml
    with open(FOREX_CFG, "r", encoding="utf-8") as fh:
        cfg = XSConfig.model_validate(yaml.safe_load(fh))
    assert cfg.risk.type == "forex_factor"
    out = run_backtest(cfg)
    assert out["n_rebalances"] > 0
    assert "deflated_sharpe" in out["trust_report"]
    assert np.allclose(out["result"].net.to_numpy(), 0.0, atol=1e-6)  # USD-neutral
