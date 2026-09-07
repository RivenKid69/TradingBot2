# -*- coding: utf-8 -*-
"""
Stage B3 tests — futures-вертикаль (impl_continuous_futures + signals/futures_signals +
xs_risk/futures_factors + pipeline).

  * back-adjust: НЕТ искусственного скачка на роллах (ratio сохраняет доходности, diff — уровни)
  * stitch_contracts склеивает контракты в непрерывную серию
  * futures-сигналы (trend/carry/value/inv_vol) считаются; graceful к отсутствующим колонкам (BYO)
  * market_beta восстанавливает известную β; build_futures_exposures — корректная B (asset-class)
  * end-to-end CTA-бэктест по пресету config_xs_futures.yaml → Trust Report (acceptance)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from impl_continuous_futures import (
    back_adjust,
    stitch_contracts,
    build_continuous_panel,
    synthetic_continuous_frames,
)
from signals.futures_signals import Trend, Carry, FuturesValue, RealizedVolInv
from xs_risk.futures_factors import market_beta, build_futures_exposures
from service_xs_pipeline import XSConfig, run_backtest, build_signal_library, load_panel

T0, STEP = 1_600_000_000, 86_400
FUTURES_CFG = os.path.join("configs", "config_xs_futures.yaml")


# ---------------------------------------------------------------------------
# back-adjust (нет скачка на роллах)
# ---------------------------------------------------------------------------
def test_back_adjust_ratio_preserves_returns_no_jump():
    # «истинный» путь P_t; контракт A = P, контракт B = P * 1.10 (10% уровневый сдвиг).
    rng = np.random.default_rng(0)
    n = 40
    ts = [T0 + i * STEP for i in range(n)]
    P = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    roll = 20
    # сырой стич: A до ролла (P), B с ролла (P*1.10) → искусственный скачок на roll
    raw_vals = np.where(np.arange(n) < roll, P, P * 1.10)
    raw = pd.Series(raw_vals, index=[t * 1000 for t in ts])
    roll_ts = ts[roll] * 1000
    # gap (ratio) = B/A на перекрытии = 1.10
    cont = back_adjust(raw, [(roll_ts, 1.10)], method="ratio")
    rc = cont.pct_change().to_numpy()
    # доходность на роллe должна равняться ИСТИННОЙ (P[roll]/P[roll-1]-1), без 10% скачка
    true_ret = P[roll] / P[roll - 1] - 1.0
    assert rc[roll] == pytest.approx(true_ret, abs=1e-12)
    # последний сегмент не тронут (= реальные цены B)
    assert cont.iloc[-1] == pytest.approx(raw.iloc[-1])


def test_back_adjust_diff_continuous_levels():
    n = 20
    ts = [T0 + i * STEP for i in range(n)]
    P = 100.0 + np.arange(n, dtype="float64")  # +1 за бар
    roll = 10
    offset = 5.0
    raw_vals = np.where(np.arange(n) < roll, P, P + offset)
    raw = pd.Series(raw_vals, index=[t * 1000 for t in ts])
    cont = back_adjust(raw, [(ts[roll] * 1000, offset)], method="diff")
    # приращение уровня на роллe = истинное (+1), без offset-скачка
    assert (cont.iloc[roll] - cont.iloc[roll - 1]) == pytest.approx(1.0)


def test_stitch_contracts_builds_continuous():
    n = 30
    ts = np.array([(T0 + i * STEP) * 1000 for i in range(n)])
    P = 100.0 * np.cumprod(1.0 + 0.001 * np.ones(n))
    a = pd.Series(P[:22], index=ts[:22])  # контракт A (с перекрытием)
    b = pd.Series(P[18:] * 1.05, index=ts[18:])  # контракт B, +5% уровень
    # A активен до ts[20], затем B
    cont, rolls = stitch_contracts([(ts[0], a), (ts[20], b)], method="ratio")
    assert len(rolls) == 1
    r = cont.pct_change().to_numpy()
    # на склейке нет 5% скачка
    assert abs(r[np.where(cont.index.to_numpy() == ts[20])[0][0]]) < 0.02


def test_build_continuous_panel_smoke():
    n = 25
    ts = np.array([(T0 + i * STEP) * 1000 for i in range(n)])
    P = 100.0 * np.cumprod(1.0 + 0.001 * np.ones(n))
    a = pd.Series(P[:18], index=ts[:18])
    b = pd.Series(P[15:] * 1.03, index=ts[15:])
    panel, meta = build_continuous_panel({"ES": [(ts[0], a), (ts[16], b)]}, method="ratio")
    assert "close" in panel.columns
    assert meta["ES"].n_rolls == 1 and meta["ES"].method == "ratio"


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def _fut_panel(n: int = 40):
    frames = synthetic_continuous_frames(["ES", "CL", "GC"], n_bars=n, seed=3)
    return PanelBuilder.from_frames(frames)


def test_trend_and_value_signs():
    n = 30
    ts = [T0 + i * STEP for i in range(n)]
    close = 100.0 * (1.0 + 0.01 * np.arange(n))  # монотонный рост
    frame = pd.DataFrame({"timestamp": ts, "symbol": "ES", "close": close})
    panel = PanelBuilder.from_frames({"ES": frame})
    tr = Trend("t", lookback=10).compute_panel(panel)
    val = FuturesValue("v", lookback=10).compute_panel(panel)
    bar = (T0 + 12 * STEP) * 1000
    exp = close[12] / close[2] - 1.0
    assert tr.loc[(bar, "ES")] == pytest.approx(exp)
    assert val.loc[(bar, "ES")] == pytest.approx(-exp)  # value = −trend


def test_carry_from_front_back_and_ready_column():
    n = 4
    ts = [T0 + i * STEP for i in range(n)]
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "CL",
            "close": [70.0] * n,
            "front": [70.0] * n,
            "back": [68.0] * n,  # backwardation: (70-68)/68 > 0
        }
    )
    panel = PanelBuilder.from_frames({"CL": frame})
    c = Carry("c").compute_panel(panel)
    assert c.iloc[0] == pytest.approx((70.0 - 68.0) / 68.0)
    # готовая carry-колонка имеет приоритет
    frame2 = pd.DataFrame(
        {"timestamp": ts, "symbol": "CL", "close": [70.0] * n, "carry": [0.05] * n}
    )
    panel2 = PanelBuilder.from_frames({"CL": frame2})
    assert Carry("c").compute_panel(panel2).iloc[0] == pytest.approx(0.05)


def test_signals_graceful_missing_column():
    frame = pd.DataFrame({"timestamp": [T0, T0 + STEP], "symbol": "ES", "close": [1.0, 2.0]})
    panel = PanelBuilder.from_frames({"ES": frame})
    assert Carry("c").compute_panel(panel).isna().all()  # нет front/back/carry → BYO пуст


def test_inv_vol_positive():
    panel = _fut_panel(80)
    iv = RealizedVolInv("iv", window=20).compute_panel(panel)
    finite = iv.dropna()
    assert len(finite) > 0 and (finite > 0).all()


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def test_market_beta_recovers_known():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, 150)
    rw = pd.DataFrame({"ES": mkt, "HI2X": 2.0 * mkt, "NOISE": rng.normal(0, 0.01, 150)})
    beta = market_beta(rw, market_symbol="ES")
    assert beta["ES"] == pytest.approx(1.0, abs=1e-9)
    assert beta["HI2X"] == pytest.approx(2.0, abs=1e-9)


def test_build_futures_exposures_asset_class():
    rng = np.random.default_rng(1)
    rw = pd.DataFrame({s: rng.normal(0, 0.01, 100) for s in ["ES", "CL", "GC", "ZN"]})
    B = build_futures_exposures(
        rw,
        asset_classes={"ES": "equity_index", "CL": "energy", "GC": "metals", "ZN": "rates"},
    )
    assert "market_beta" in B.columns and "vol" in B.columns
    assert any(c.startswith("ac_") for c in B.columns)
    assert list(B.index) == ["ES", "CL", "GC", "ZN"]


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_builds_futures_signals():
    cfg = XSConfig.model_validate(
        {
            "asset_class": "futures",
            "data": {"source": "synthetic", "symbols": ["ES", "CL"]},
            "signals": [
                {"name": "t50", "kind": "trend", "lookback": 50, "vol_normalize": True},
                {"name": "val", "kind": "futures_value", "lookback": 250},
                {"name": "c", "kind": "carry"},
            ],
        }
    )
    lib = build_signal_library(cfg)
    assert lib.names == ["t50", "val", "c"]


def test_futures_synthetic_panel_is_continuous():
    cfg = XSConfig.model_validate(
        {
            "asset_class": "futures",
            "data": {"source": "synthetic", "symbols": ["ES", "CL", "GC"], "synthetic_bars": 60},
        }
    )
    panel = load_panel(cfg)
    assert "close" in panel.columns
    assert panel.index.get_level_values("symbol").nunique() == 3


# ---------------------------------------------------------------------------
# end-to-end (acceptance)
# ---------------------------------------------------------------------------
def test_futures_preset_end_to_end():
    import yaml

    with open(FUTURES_CFG, "r", encoding="utf-8") as fh:
        cfg = XSConfig.model_validate(yaml.safe_load(fh))
    assert cfg.risk.type == "futures_factor"
    out = run_backtest(cfg)
    assert out["n_rebalances"] > 0
    assert "deflated_sharpe" in out["trust_report"]
    assert np.allclose(out["result"].net.to_numpy(), 0.0, atol=1e-6)  # market-neutral
