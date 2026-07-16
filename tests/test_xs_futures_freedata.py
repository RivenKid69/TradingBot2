# -*- coding: utf-8 -*-
"""
Stage D4 tests — futures continuous free + roll-accurate (loaders/futures_enrich).

  * ContinuousProxySource: ES→ES=F трансляция + возврат исходного символа; pit=approx
  * build_roll_accurate_panel: BYO контракты → continuous без скачка на роллах; pit=true meta
  * CarryEnricher: front/back → carry=(front−back)/back; PIT as-of; graceful empty
  * Carry-сигнал «оживает»; _price_source_for: futures free → ContinuousProxySource
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from signals.futures_signals import Carry
from loaders.futures_enrich import (
    ContinuousProxySource, build_roll_accurate_panel, CarryEnricher, DEFAULT_CME_YAHOO_MAP,
)

T0, STEP = 1_600_000_000, 86_400


# ---------------------------------------------------------------------------
# ContinuousProxySource
# ---------------------------------------------------------------------------
class FakeInner:
    def __init__(self):
        self.meta = DataSourceMeta(name="free:yahoo", vendor="yahoo", kind="price")
        self.calls = []

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        self.calls.append(list(symbols))
        out = {}
        for tk in symbols:
            out[tk] = pd.DataFrame({"timestamp": [T0, T0 + STEP], "symbol": tk, "close": [100.0, 101.0]})
        return out


def test_continuous_proxy_translates_symbols():
    inner = FakeInner()
    src = ContinuousProxySource(inner=inner)
    out = src.get_bars(["ES", "CL"], "1d")
    assert inner.calls == [["ES=F"], ["CL=F"]]      # транслировано в yahoo-тикеры
    assert set(out.keys()) == {"ES", "CL"}           # возвращены исходные символы
    assert (out["ES"]["symbol"] == "ES").all()
    assert src.meta.pit_quality == "approx"


def test_default_map_covers_majors():
    for s in ("ES", "NQ", "CL", "GC", "6E", "ZN", "ZC"):
        assert DEFAULT_CME_YAHOO_MAP[s].endswith("=F")


# ---------------------------------------------------------------------------
# roll-accurate (BYO)
# ---------------------------------------------------------------------------
def test_roll_accurate_no_jump():
    n = 30
    ts = np.array([(T0 + i * STEP) * 1000 for i in range(n)])
    P = 100.0 * np.cumprod(1.0 + 0.001 * np.ones(n))
    a = pd.Series(P[:22], index=ts[:22])
    b = pd.Series(P[18:] * 1.05, index=ts[18:])       # контракт B на +5% уровне
    panel, metas = build_roll_accurate_panel({"ES": [(ts[0], a), (ts[20], b)]}, method="ratio")
    assert "close" in panel.columns
    assert metas["ES"].pit_quality == "true" and metas["ES"].n_rolls == 1
    cont = panel.xs("ES", level=SYMBOL_LEVEL)["close"]
    r = cont.pct_change().to_numpy()
    roll_pos = int(np.where(cont.index.to_numpy() == ts[20])[0][0])
    assert abs(r[roll_pos]) < 0.02                    # нет 5% скачка на роллe


# ---------------------------------------------------------------------------
# CarryEnricher
# ---------------------------------------------------------------------------
def _panel(n=6, syms=("ES",)):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": 100.0 + np.arange(n)})
    return PanelBuilder.from_frames(frames)


def test_carry_enricher_pit():
    pub = (T0 + 3 * STEP) * 1000
    def fb(symbols):
        return pd.DataFrame({"publish_ts": [pub], "symbol": ["ES"], "front": [70.0], "back": [68.0]})
    enr = CarryEnricher(fb_provider=fb)
    out = enr.enrich(_panel(8))
    c = out.xs("ES", level=SYMBOL_LEVEL)["carry"].to_numpy()
    assert np.isnan(c[:3]).all()
    assert np.allclose(c[3:], (70.0 - 68.0) / 68.0)
    assert enr.meta.pit_quality == "true"


def test_carry_enricher_graceful_empty():
    out = CarryEnricher(fb_provider=lambda syms: pd.DataFrame()).enrich(_panel(4))
    assert out["carry"].isna().all()


def test_carry_signal_comes_alive():
    pub = T0 * 1000
    def fb(symbols):
        rows = [{"publish_ts": pub, "symbol": s, "front": 70.0, "back": 68.0} for s in symbols]
        return pd.DataFrame(rows)
    out = CarryEnricher(fb_provider=fb).enrich(_panel(4, syms=("ES", "CL")))
    sig = Carry("c").compute_panel(out)
    assert not sig.isna().all()
    assert sig.xs("ES", level=SYMBOL_LEVEL).iloc[0] == pytest.approx((70.0 - 68.0) / 68.0)


# ---------------------------------------------------------------------------
# price source routing
# ---------------------------------------------------------------------------
def test_price_source_routing():
    from service_xs_pipeline import XSConfig, _price_source_for
    fut = XSConfig.model_validate({"asset_class": "futures",
                                   "data": {"source": "free", "vendor": "yahoo", "symbols": ["ES"]}})
    assert isinstance(_price_source_for(fut), ContinuousProxySource)
    cry = XSConfig.model_validate({"asset_class": "crypto",
                                   "data": {"source": "free", "vendor": "binance", "symbols": ["BTC"]}})
    assert not isinstance(_price_source_for(cry), ContinuousProxySource)
