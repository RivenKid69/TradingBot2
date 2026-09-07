# -*- coding: utf-8 -*-
"""
Stage B5 tests — options-вертикаль (ОТДЕЛЬНЫЙ greeks-space оптимизатор).

  * OptionsPortfolioConstructor: greeks-нейтральность (residual delta/vega/gamma ≈ 0)
    методом null-space проекции; gross масштабируется; объектив = альфа·w
  * options-сигналы (VRP/skew/dispersion/term) считаются; graceful к отсутствующим (BYO)
  * vol_level_beta восстанавливает известную β; build_options_exposures — корректная B
  * pipeline строит options-сигналы; API /api/xs/options/construct отдаёт нейтральный портфель
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from service_options_portfolio import (
    OptionLeg,
    GreeksNeutralConstraints,
    OptionsPortfolioConstructor,
    construct_options_portfolio,
    synthetic_option_book,
)
from signals.options_signals import VolRiskPremium, Skew, Dispersion, TermStructure
from xs_risk.options_factors import vol_level_beta, build_options_exposures
from service_xs_pipeline import XSConfig, build_signal_library

T0, STEP = 1_700_000_000, 86_400


# ---------------------------------------------------------------------------
# greeks-space constructor
# ---------------------------------------------------------------------------
def test_delta_vega_neutral():
    legs = synthetic_option_book(spot=100.0)
    port = construct_options_portfolio(legs, neutralize=["delta", "vega"], gross_max=1.0)
    assert port.is_neutral
    assert abs(port.net_greeks["delta"]) < 1e-6
    assert abs(port.net_greeks["vega"]) < 1e-6
    assert port.gross == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(port.objective)


def test_delta_gamma_vega_neutral():
    legs = synthetic_option_book(spot=100.0)
    cons = GreeksNeutralConstraints(neutralize=["delta", "gamma", "vega"], gross_max=2.0)
    port = OptionsPortfolioConstructor(cons).construct(legs)
    for g in ("delta", "gamma", "vega"):
        assert abs(port.net_greeks[g]) < 1e-5
    assert port.is_neutral
    assert port.gross == pytest.approx(2.0, abs=1e-6)


def test_objective_captures_alpha_direction():
    # альфа коррелирует с весом → объектив положителен (захватываем edge)
    legs = synthetic_option_book(spot=100.0)
    port = construct_options_portfolio(legs, neutralize=["delta"], gross_max=1.0)
    assert port.objective > 0.0  # стандартизованная альфа·w > 0 после проекции


def test_too_few_legs_collapses_to_neutral_zero():
    # 1 нога, нейтрализуем delta → null-space тривиален → w≈0 (корректно)
    legs = [
        OptionLeg("X", spot=100, strike=100, time_to_expiry=0.25, iv=0.2, is_call=True, alpha=1.0)
    ]
    port = construct_options_portfolio(legs, neutralize=["delta"], gross_max=1.0)
    assert port.gross < 1e-9
    assert abs(port.net_greeks["delta"]) < 1e-9


def test_max_position_clip_diversifies():
    legs = synthetic_option_book(spot=100.0)
    free = construct_options_portfolio(legs, neutralize=["delta", "vega"], gross_max=1.0)
    capped = construct_options_portfolio(
        legs, neutralize=["delta", "vega"], gross_max=1.0, max_position=0.05
    )
    # клип → больше активных ног (диверсификация), нейтральность сохранена
    assert (capped.weights.abs() > 1e-6).sum() >= (free.weights.abs() > 1e-6).sum()
    assert abs(capped.net_greeks["delta"]) < 1e-5 and abs(capped.net_greeks["vega"]) < 1e-5


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def _opt_panel(n=4):
    ts = [T0 + i * STEP for i in range(n)]
    frames = {}
    for sym, iv, rv in [("SPX", 0.20, 0.14), ("AAPL", 0.30, 0.28)]:
        frames[sym] = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": sym,
                "close": 100.0,
                "iv": iv,
                "realized_vol": rv,
                "iv_put_25": iv + 0.03,
                "iv_call_25": iv - 0.01,
                "iv_front": iv + 0.02,
                "iv_back": iv,
            }
        )
    return PanelBuilder.from_frames(frames)


def test_vrp_skew_term_signs():
    panel = _opt_panel()
    vrp = VolRiskPremium("vrp").compute_panel(panel)
    sk = Skew("sk").compute_panel(panel)
    ts0 = T0 * 1000
    assert vrp.loc[(ts0, "SPX")] == pytest.approx(0.20 - 0.14)  # IV − RV
    assert sk.loc[(ts0, "SPX")] == pytest.approx((0.20 + 0.03) - (0.20 - 0.01))  # put − call
    term = TermStructure("t").compute_panel(panel)
    assert term.loc[(ts0, "SPX")] == pytest.approx(0.02)  # front − back


def test_signals_graceful_missing_column():
    frame = pd.DataFrame({"timestamp": [T0, T0 + STEP], "symbol": "SPX", "close": [1.0, 2.0]})
    panel = PanelBuilder.from_frames({"SPX": frame})
    for sig in (VolRiskPremium("v"), Skew("s"), Dispersion("d"), TermStructure("t")):
        assert sig.compute_panel(panel).isna().all()


# ---------------------------------------------------------------------------
# vol factors
# ---------------------------------------------------------------------------
def test_vol_level_beta_recovers_known():
    rng = np.random.default_rng(0)
    vix = rng.normal(0, 0.02, 120)
    ivc = pd.DataFrame({"VIX": vix, "HI2X": 2.0 * vix, "NOISE": rng.normal(0, 0.02, 120)})
    beta = vol_level_beta(ivc, vol_index_symbol="VIX")
    assert beta["VIX"] == pytest.approx(1.0, abs=1e-9)
    assert beta["HI2X"] == pytest.approx(2.0, abs=1e-9)


def test_build_options_exposures():
    rng = np.random.default_rng(1)
    ivc = pd.DataFrame({s: rng.normal(0, 0.02, 100) for s in ["SPX", "AAPL", "MSFT"]})
    B = build_options_exposures(
        ivc,
        skews={"SPX": 0.05, "AAPL": 0.02, "MSFT": 0.01},
        terms={"SPX": 0.01, "AAPL": -0.01, "MSFT": 0.0},
    )
    for col in ("vol_level_beta", "skew", "term"):
        assert col in B.columns
    assert list(B.index) == ["SPX", "AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# pipeline + API
# ---------------------------------------------------------------------------
def test_pipeline_builds_options_signals():
    cfg = XSConfig.model_validate(
        {
            "asset_class": "options",
            "signals": [
                {"name": "vrp", "kind": "vrp"},
                {"name": "skew", "kind": "skew"},
                {"name": "term", "kind": "term_structure"},
            ],
        }
    )
    lib = build_signal_library(cfg)
    assert lib.names == ["vrp", "skew", "term"]


def test_api_options_construct_demo():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xs_api import register_xs_routes

    app = FastAPI()
    register_xs_routes(app)
    client = TestClient(app)
    r = client.post(
        "/api/xs/options/construct",
        json={"demo": True, "neutralize": ["delta", "vega"], "gross_max": 1.0},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_neutral"] is True
    assert abs(data["net_greeks"]["delta"]) < 1e-5
    assert abs(data["net_greeks"]["vega"]) < 1e-5
    assert data["n_legs"] > 0
