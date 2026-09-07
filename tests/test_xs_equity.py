# -*- coding: utf-8 -*-
"""
Stage B2 tests — equity-вертикаль (signals/equity_signals + xs_risk/equity_factors + pipeline).

  * equity-сигналы считаются; graceful к отсутствующим колонкам (BYO-слот)
  * earnings_yield/book_to_price — из фундаментал/цена, корректный знак/значение
  * low_vol = −rolling std (неположителен), equity_momentum = 12-1
  * market_beta восстанавливает известную β; build_equity_exposures — корректная B
  * end-to-end бэктест по пресету config_xs_equity.yaml → Trust Report (acceptance)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from signals.equity_signals import (
    Accruals,
    BookToPrice,
    EarningsYield,
    EquityMomentum,
    EquitySize,
    FCFYield,
    LowVolatility,
    ReturnOnEquity,
)
from xs_risk.equity_factors import market_beta, build_equity_exposures
from service_xs_pipeline import XSConfig, run_backtest, build_signal_library

T0, STEP = 1_700_000_000, 86_400
EQUITY_CFG = os.path.join("configs", "config_xs_equity.yaml")


def _equity_panel(n: int = 60):
    ts = [T0 + i * STEP for i in range(n)]
    # base_price, earnings, book_value, market_cap
    spec = {
        "AAA": (100.0, 8.0, 40.0, 3000.0),
        "BBB": (50.0, 2.0, 60.0, 800.0),
        "CCC": (200.0, 5.0, 20.0, 120.0),
    }
    frames = {}
    for sym, (base, earn, book, mcap) in spec.items():
        close = base * (1.0 + 0.005 * np.arange(n))
        frames[sym] = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": sym,
                "close": close,
                "earnings": earn,
                "book_value": book,
                "market_cap": mcap,
                "roe": earn / book,
                "accruals": 0.1,
            }
        )
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def test_equity_momentum():
    panel = _equity_panel(40)
    mom = EquityMomentum("m", lookback=10, skip=2).compute_panel(panel)
    bar = (T0 + 12 * STEP) * 1000  # t=12: price[t-skip]/price[t-lookback] = price[10]/price[2]
    expected = (1 + 0.005 * 10) / (1 + 0.005 * 2) - 1.0
    assert mom.loc[(bar, "AAA")] == pytest.approx(expected)


def test_earnings_book_yield_from_fundamental():
    panel = _equity_panel(5)
    ey = EarningsYield("ey").compute_panel(panel)
    bp = BookToPrice("bp").compute_panel(panel)
    t0 = T0 * 1000
    # AAA: earnings=8 / close=100 → E/P=0.08 ; book=40/100 → B/P=0.40
    assert ey.loc[(t0, "AAA")] == pytest.approx(8.0 / 100.0)
    assert bp.loc[(t0, "AAA")] == pytest.approx(40.0 / 100.0)


def test_earnings_yield_prefers_ready_column():
    frame = pd.DataFrame({"timestamp": [T0], "symbol": "AAA", "close": [100.0], "ep": [0.123]})
    panel = PanelBuilder.from_frames({"AAA": frame})
    ey = EarningsYield("ey", yield_col="ep").compute_panel(panel)
    assert ey.loc[(T0 * 1000, "AAA")] == pytest.approx(0.123)


def test_roe_and_accruals_sign():
    panel = _equity_panel(5)
    roe = ReturnOnEquity("roe").compute_panel(panel)
    acc = Accruals("acc").compute_panel(panel)
    t0 = T0 * 1000
    assert roe.loc[(t0, "AAA")] == pytest.approx(8.0 / 40.0)  # earnings/book
    assert acc.loc[(t0, "AAA")] == pytest.approx(-0.1)  # −accruals


def test_low_vol_nonpositive_and_size():
    panel = _equity_panel(80)
    lv = LowVolatility("lv", window=20).compute_panel(panel)
    finite = lv.dropna()
    assert len(finite) > 0
    assert (finite <= 1e-12).all()  # −std ≤ 0
    sz = EquitySize("sz").compute_panel(panel)
    t0 = T0 * 1000
    assert sz.loc[(t0, "CCC")] == pytest.approx(-np.log(120.0))


def test_signals_graceful_missing_column():
    # панель только с close → фундаментальные → NaN (BYO-слот пуст), без падения
    frame = pd.DataFrame({"timestamp": [T0, T0 + STEP], "symbol": "AAA", "close": [1.0, 2.0]})
    panel = PanelBuilder.from_frames({"AAA": frame})
    for sig in (
        EarningsYield("e"),
        BookToPrice("b"),
        FCFYield("f"),
        ReturnOnEquity("r"),
        Accruals("a"),
        EquitySize("s"),
    ):
        out = sig.compute_panel(panel)
        assert out.isna().all()


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def test_market_beta_recovers_known():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.02, 150)
    rw = pd.DataFrame({"SPY": mkt, "HI2X": 2.0 * mkt, "NOISE": rng.normal(0, 0.02, 150)})
    beta = market_beta(rw, market_symbol="SPY")
    assert beta["SPY"] == pytest.approx(1.0, abs=1e-9)
    assert beta["HI2X"] == pytest.approx(2.0, abs=1e-9)


def test_market_beta_proxy_when_no_index():
    rng = np.random.default_rng(2)
    rw = pd.DataFrame({s: rng.normal(0, 0.02, 120) for s in ["A", "B", "C"]})
    beta = market_beta(rw, market_symbol=None)  # равновзвешенный прокси
    assert set(beta.index) == {"A", "B", "C"}
    assert np.isfinite(beta.to_numpy()).all()


def test_build_equity_exposures():
    rng = np.random.default_rng(1)
    rw = pd.DataFrame({s: rng.normal(0, 0.02, 100) for s in ["AAA", "BBB", "CCC", "SPY"]})
    B = build_equity_exposures(
        rw,
        sectors={"AAA": "Tech", "BBB": "Financials", "CCC": "Energy", "SPY": "Index"},
        mcaps={"AAA": 3000, "BBB": 800, "CCC": 120, "SPY": 0.001},
        values={"AAA": 0.05, "BBB": 0.09, "CCC": 0.02, "SPY": 0.04},
        market_symbol="SPY",
    )
    for col in ("market_beta", "size", "value", "momentum"):
        assert col in B.columns
    assert any(c.startswith("sector_") for c in B.columns)
    assert list(B.index) == ["AAA", "BBB", "CCC", "SPY"]


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_builds_equity_signals():
    cfg = XSConfig.model_validate(
        {
            "asset_class": "equity",
            "data": {"source": "synthetic", "symbols": ["AAA", "BBB"]},
            "signals": [
                {"name": "mom", "kind": "equity_momentum", "lookback": 252, "skip": 21},
                {"name": "lv", "kind": "low_vol", "vol_window": 60},
                {"name": "ey", "kind": "earnings_yield"},
            ],
        }
    )
    lib = build_signal_library(cfg)
    assert lib.names == ["mom", "lv", "ey"]


# ---------------------------------------------------------------------------
# end-to-end (acceptance)
# ---------------------------------------------------------------------------
def test_equity_preset_end_to_end():
    import yaml

    with open(EQUITY_CFG, "r", encoding="utf-8") as fh:
        cfg = XSConfig.model_validate(yaml.safe_load(fh))
    assert cfg.risk.type == "equity_factor"
    out = run_backtest(cfg)
    assert out["n_rebalances"] > 0
    assert "deflated_sharpe" in out["trust_report"]
    # market-neutral long-short
    assert np.allclose(out["result"].net.to_numpy(), 0.0, atol=1e-6)
