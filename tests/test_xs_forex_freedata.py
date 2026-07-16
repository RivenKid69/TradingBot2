# -*- coding: utf-8 -*-
"""
Stage D3 tests — forex free + дифференциал ставок (loaders/forex_enrich).

  * parse_pair: EURUSD/EUR_USD/EUR/USD → (base, quote)
  * RateDiffEnricher static: rate_base/rate_quote/rate_diff корректны; pit=approx
  * RateDiffEnricher history: PIT as-of (NaN до publish_ts); pit=true
  * интеграция: fx_carry «оживает» на собранной панели; малый юниверс не ломается
  * build_enrichers: rate_diff с policy_rates → обогатитель; без ставок → пропуск
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import DataAssembler
from signals.forex_signals import FXCarry
from loaders.forex_enrich import parse_pair, RateDiffEnricher, build_forex_enricher

T0, STEP = 1_700_000_000, 86_400


class FakePriceSource:
    def __init__(self, n=8, vendor="oanda"):
        self.meta = DataSourceMeta(name=f"free:{vendor}", vendor=vendor, kind="price")
        self.n = n

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        return {s: pd.DataFrame({"timestamp": [T0 + i * STEP for i in range(self.n)], "symbol": s,
                                 "close": 1.1 + 0.001 * np.arange(self.n)}) for s in symbols}


def _panel(n=6, syms=("EURUSD", "USDJPY")):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": 1.1 + 0.001 * np.arange(n)})
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# parse_pair
# ---------------------------------------------------------------------------
def test_parse_pair():
    assert parse_pair("EURUSD") == ("EUR", "USD")
    assert parse_pair("EUR_USD") == ("EUR", "USD")
    assert parse_pair("EUR/USD") == ("EUR", "USD")
    assert parse_pair("USDJPY") == ("USD", "JPY")


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
def test_rate_diff_static():
    rates = {"USD": 0.045, "EUR": 0.035, "JPY": 0.001}
    enr = RateDiffEnricher(rates)
    out = enr.enrich(_panel(4, syms=("EURUSD", "USDJPY")))
    eur = out.xs("EURUSD", level=SYMBOL_LEVEL)
    assert (eur["rate_base"] == 0.035).all() and (eur["rate_quote"] == 0.045).all()
    assert np.allclose(eur["rate_diff"].to_numpy(), 0.035 - 0.045)   # EUR − USD
    usdjpy = out.xs("USDJPY", level=SYMBOL_LEVEL)
    assert np.allclose(usdjpy["rate_diff"].to_numpy(), 0.045 - 0.001)  # USD − JPY (положит.)
    assert enr.meta.pit_quality == "approx"


def test_rate_diff_missing_currency_nan():
    out = RateDiffEnricher({"USD": 0.045}).enrich(_panel(3, syms=("EURUSD",)))
    eur = out.xs("EURUSD", level=SYMBOL_LEVEL)
    assert np.isnan(eur["rate_base"].to_numpy()).all()   # EUR нет в карте
    assert np.isnan(eur["rate_diff"].to_numpy()).all()


# ---------------------------------------------------------------------------
# history (PIT)
# ---------------------------------------------------------------------------
def test_rate_diff_history_pit():
    # ставки публикуются на баре 3 → до этого rate_diff NaN (PIT)
    pub = (T0 + 3 * STEP) * 1000   # панель в мс
    def hist(currencies):
        rows = []
        for c, r in {"EUR": 0.035, "USD": 0.045}.items():
            if c in currencies:
                rows.append({"publish_ts": pub, "currency": c, "rate": r})
        return pd.DataFrame(rows)
    enr = RateDiffEnricher(history_fn=hist)
    out = enr.enrich(_panel(8, syms=("EURUSD",)))
    d = out.xs("EURUSD", level=SYMBOL_LEVEL)["rate_diff"].to_numpy()
    assert np.isnan(d[:3]).all()
    assert np.allclose(d[3:], 0.035 - 0.045)
    assert enr.meta.pit_quality == "true"


# ---------------------------------------------------------------------------
# integration: fx_carry оживает + малый юниверс
# ---------------------------------------------------------------------------
def test_fx_carry_comes_alive_small_universe():
    src = FakePriceSource(n=8)
    enr = RateDiffEnricher({"USD": 0.045, "EUR": 0.035, "AUD": 0.0435})
    res = DataAssembler(src, enrichers=[enr]).assemble(["EURUSD", "AUDUSD"], "1d")
    assert "rate_diff" in res.panel.columns
    carry = FXCarry("c").compute_panel(res.panel)
    assert not carry.isna().all()                       # сигнал ожил
    # FXCarry приоритет rate_diff: EURUSD → 0.035−0.045 = −0.01
    eur = carry.xs("EURUSD", level=SYMBOL_LEVEL).iloc[0]
    assert eur == pytest.approx(0.035 - 0.045)
    prov = {c.column: c for c in res.report.columns}
    assert prov["rate_diff"].pit_quality == "approx"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_build_forex_enricher_with_and_without_rates():
    from service_xs_pipeline import XSConfig, build_enrichers
    cfg = XSConfig.model_validate({
        "asset_class": "forex",
        "data": {"source": "free", "vendor": "oanda", "symbols": ["EURUSD"], "enrich": ["rate_diff"]},
        "policy_rates": {"USD": 0.045, "EUR": 0.035},
    })
    enrichers = build_enrichers(cfg)
    assert len(enrichers) == 1 and "rate_diff" in enrichers[0].columns()
    # без policy_rates → пропуск (honest)
    cfg2 = XSConfig.model_validate({"asset_class": "forex",
                                    "data": {"source": "free", "enrich": ["rate_diff"]}})
    assert build_enrichers(cfg2) == []
