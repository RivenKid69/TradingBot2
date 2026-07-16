# -*- coding: utf-8 -*-
"""
Stage D5 tests — options free + IV (loaders/options_enrich).

  * IVSummaryEnricher/DeribitIVEnricher: iv/skew/term_slope as-of PIT; YFinance=pit-none honest
  * RealizedVolEnricher: realized_vol из close (PIT-true)
  * VRP/Skew/Term сигналы «оживают» на собранной панели
  * OptionsBookLoader.chain_to_legs → legs → greeks-нейтральный портфель
  * API /api/xs/options/construct принимает реальный chain; реестр build_options_enricher
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import DataAssembler
from signals.options_signals import VolRiskPremium, Skew, TermStructure
from loaders.options_enrich import (
    IVSummaryEnricher, DeribitIVEnricher, YFinanceChainEnricher,
    RealizedVolEnricher, OptionsBookLoader, build_options_enricher,
)

T0, STEP = 1_700_000_000, 86_400


class FakePriceSource:
    def __init__(self, n=30, vendor="deribit"):
        self.meta = DataSourceMeta(name=f"free:{vendor}", vendor=vendor, kind="price")
        self.n = n

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        rng = np.random.default_rng(0)
        out = {}
        for s in symbols:
            close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, self.n))
            out[s] = pd.DataFrame({"timestamp": [T0 + i * STEP for i in range(self.n)], "symbol": s, "close": close})
        return out


def _panel(n=10, syms=("BTC",)):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": 100.0 + np.arange(n)})
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# IV enrichers
# ---------------------------------------------------------------------------
def test_iv_summary_pit():
    pub = (T0 + 4 * STEP) * 1000
    def prov(symbols):
        return pd.DataFrame({"publish_ts": [pub], "symbol": ["BTC"],
                             "iv": [0.55], "skew": [0.03], "term_slope": [0.01]})
    enr = DeribitIVEnricher(provider=prov)
    out = enr.enrich(_panel(8))
    iv = out.xs("BTC", level=SYMBOL_LEVEL)["iv"].to_numpy()
    assert np.isnan(iv[:4]).all() and np.allclose(iv[4:], 0.55)
    assert "skew" in out.columns and "term_slope" in out.columns
    assert enr.meta.pit_quality == "approx"


def test_yfinance_chain_pit_none():
    enr = YFinanceChainEnricher(provider=lambda s: pd.DataFrame(columns=["publish_ts", "symbol", "iv", "skew", "term_slope"]))
    assert enr.meta.pit_quality == "none"      # снимок → не backtest-safe (honest)


def test_realized_vol_from_close():
    panel = _panel(40)
    out = RealizedVolEnricher(window=10, periods_per_year=365).enrich(panel)
    rv = out.xs("BTC", level=SYMBOL_LEVEL)["realized_vol"].dropna()
    assert len(rv) > 0 and (rv >= 0).all()
    assert RealizedVolEnricher().meta.pit_quality == "true"


# ---------------------------------------------------------------------------
# signals come alive (VRP = IV − RV)
# ---------------------------------------------------------------------------
def test_vrp_skew_term_come_alive():
    src = FakePriceSource(n=30)
    pub = T0 * 1000
    def iv_prov(symbols):
        rows = [{"publish_ts": pub, "symbol": s, "iv": 0.55, "skew": 0.04, "term_slope": 0.02} for s in symbols]
        return pd.DataFrame(rows)
    res = DataAssembler(src, enrichers=[DeribitIVEnricher(provider=iv_prov),
                                        RealizedVolEnricher(window=10)]).assemble(["BTC", "ETH"], "1d")
    assert {"iv", "realized_vol", "skew", "term_slope"} <= set(res.panel.columns)
    vrp = VolRiskPremium("vrp").compute_panel(res.panel)
    assert not vrp.isna().all()                 # ожил (iv − realized_vol)
    assert not Skew("sk").compute_panel(res.panel).isna().all()
    assert not TermStructure("t").compute_panel(res.panel).isna().all()


# ---------------------------------------------------------------------------
# book loader → greeks-neutral
# ---------------------------------------------------------------------------
def test_chain_to_legs_and_construct():
    from service_options_portfolio import construct_options_portfolio
    chain = []
    for K in (80, 90, 100, 110, 120):
        for is_call in (True, False):
            chain.append({"strike": K, "time_to_expiry": 0.25, "iv": 0.2 + 0.001 * abs(K - 100),
                          "is_call": is_call, "alpha": abs(K - 100) * 0.5})
    legs = OptionsBookLoader.chain_to_legs(chain, spot=100.0)
    assert len(legs) == 10 and all(l.spot == 100.0 for l in legs)
    port = construct_options_portfolio(legs, neutralize=["delta", "vega"], gross_max=1.0)
    assert port.is_neutral
    assert abs(port.net_greeks["delta"]) < 1e-5 and abs(port.net_greeks["vega"]) < 1e-5


def test_chain_to_legs_expiry_days():
    legs = OptionsBookLoader.chain_to_legs(
        [{"strike": 100, "expiry_days": 73, "iv": 0.2, "is_call": True}], spot=100.0)
    assert len(legs) == 1
    assert legs[0].time_to_expiry == pytest.approx(73 / 365.0)


# ---------------------------------------------------------------------------
# API + registry
# ---------------------------------------------------------------------------
def test_api_options_construct_from_chain():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xs_api import register_xs_routes

    chain = []
    for K in (80, 90, 100, 110, 120):
        for c in (True, False):
            chain.append({"strike": K, "time_to_expiry": 0.25, "iv": 0.22, "is_call": c, "alpha": abs(K - 100)})
    app = FastAPI(); register_xs_routes(app)
    client = TestClient(app)
    r = client.post("/api/xs/options/construct",
                    json={"chain": chain, "spot": 100.0, "neutralize": ["delta", "vega"]})
    assert r.status_code == 200
    data = r.json()
    assert data["is_neutral"] is True and data["n_legs"] == 10


def test_build_options_enricher_registry():
    from service_xs_pipeline import XSConfig, build_enrichers
    cfg = XSConfig.model_validate({
        "asset_class": "options",
        "data": {"source": "free", "vendor": "deribit", "symbols": ["BTC"], "enrich": ["iv", "realized_vol"]},
        "iv_vendor": "deribit",
    })
    enrichers = build_enrichers(cfg)
    cols = sorted(sum([e.columns() for e in enrichers], []))
    assert "iv" in cols and "realized_vol" in cols
