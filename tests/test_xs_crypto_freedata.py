# -*- coding: utf-8 -*-
"""
Stage D1 tests — crypto free end-to-end (loaders/crypto_enrich + adapter fix).

  * FundingEnricher: funding_rate появляется, PIT (NaN до первой fundingTime), значение корректно
  * BasisEnricher: basis = perp/spot − 1
  * MarketCapEnricher: snapshot (approx) + history (PIT)
  * интеграция: DataAssembler(fake price + FundingEnricher) → FundingCarry «оживает» (не NaN)
  * build_enrichers реестр: enrich=[funding, basis, mcap] → 3 обогатителя
  * регрессия адаптера: BinanceMarketDataAdapter.get_bars зовёт get_klines с market= (НЕ use_futures)
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import DataAssembler
from signals.crypto_signals import FundingCarry, Size
from loaders.crypto_enrich import (
    FundingEnricher,
    BasisEnricher,
    MarketCapEnricher,
    build_crypto_enricher,
)

T0, STEP = 1_700_000_000, 86_400
FP = namedtuple("FP", ["timestamp_ms", "funding_rate"])


class FakePriceSource:
    def __init__(self, n=10, vendor="binance"):
        self.meta = DataSourceMeta(name=f"free:{vendor}", vendor=vendor, kind="price")
        self.n = n

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        out = {}
        for s in symbols:
            ts = [T0 + i * STEP for i in range(self.n)]
            out[s] = pd.DataFrame(
                {"timestamp": ts, "symbol": s, "close": 100.0 * (1 + 0.01 * np.arange(self.n))}
            )
        return out


def _panel(n=10, syms=("BTC",)):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": 100.0 + np.arange(n)})
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# FundingEnricher (PIT)
# ---------------------------------------------------------------------------
def test_funding_enricher_pit():
    # funding опубликован на баре 4 → бары 0..3 NaN, 4.. = значение
    def hist(symbol, limit=1000):
        return [FP(T0 + 4 * STEP, 0.0003)]

    enr = FundingEnricher(history_fn=hist)
    out = enr.enrich(_panel(10))
    f = out.xs("BTC", level=SYMBOL_LEVEL)["funding_rate"].to_numpy()
    assert np.isnan(f[:4]).all()
    assert np.allclose(f[4:], 0.0003)
    assert enr.meta.pit_quality == "true"


def test_funding_enricher_multiple_settlements():
    def hist(symbol, limit=1000):
        return [FP(T0 + 2 * STEP, 0.0001), FP(T0 + 6 * STEP, -0.0002)]

    out = FundingEnricher(history_fn=hist).enrich(_panel(10))
    f = out.xs("BTC", level=SYMBOL_LEVEL)["funding_rate"].to_numpy()
    assert np.isnan(f[:2]).all()
    assert np.allclose(f[2:6], 0.0001)
    assert np.allclose(f[6:], -0.0002)


# ---------------------------------------------------------------------------
# BasisEnricher
# ---------------------------------------------------------------------------
def test_basis_enricher():
    panel = _panel(5)  # spot close = 100..104

    def perp(symbols, timeframe, limit=1000):
        rows = []
        for s in symbols:
            for i in range(5):
                rows.append(
                    {"publish_ts": T0 + i * STEP, "symbol": s, "perp_close": (100.0 + i) * 1.01}
                )  # perp на 1% выше spot
        return pd.DataFrame(rows)

    out = BasisEnricher(perp_provider=perp).enrich(panel)
    b = out.xs("BTC", level=SYMBOL_LEVEL)["basis"].to_numpy()
    assert np.allclose(b, 0.01)  # (perp/spot - 1) = 1%
    assert "perp_close" not in out.columns


def test_basis_graceful_empty():
    out = BasisEnricher(perp_provider=lambda syms, tf, limit=1000: pd.DataFrame()).enrich(_panel(4))
    assert out["basis"].isna().all()


# ---------------------------------------------------------------------------
# MarketCapEnricher
# ---------------------------------------------------------------------------
def test_mcap_snapshot_approx():
    enr = MarketCapEnricher({"BTC": 1300.0})
    out = enr.enrich(_panel(4))
    assert (out.xs("BTC", level=SYMBOL_LEVEL)["mcap"] == 1300.0).all()
    assert enr.meta.pit_quality == "approx"


def test_mcap_history_pit():
    def hist(symbols):
        return pd.DataFrame({"publish_ts": [T0 + 3 * STEP], "symbol": ["BTC"], "mcap": [1500.0]})

    enr = MarketCapEnricher(history_fn=hist)
    out = enr.enrich(_panel(6))
    m = out.xs("BTC", level=SYMBOL_LEVEL)["mcap"].to_numpy()
    assert np.isnan(m[:3]).all() and (m[3:] == 1500.0).all()
    assert enr.meta.pit_quality == "true"


# ---------------------------------------------------------------------------
# integration: signal оживает
# ---------------------------------------------------------------------------
def test_funding_carry_signal_comes_alive():
    src = FakePriceSource(n=10)

    def hist(symbol, limit=1000):
        return [FP(T0 + 2 * STEP, 0.0005 if symbol == "BTC" else -0.0003)]

    res = DataAssembler(src, enrichers=[FundingEnricher(history_fn=hist)]).assemble(
        ["BTC", "ETH"], "1d"
    )
    assert "funding_rate" in res.panel.columns
    sig = FundingCarry("fc").compute_panel(res.panel)
    assert not sig.isna().all()  # сигнал ожил
    # FundingCarry = −funding → BTC negative, ETH positive после публикации
    last_btc = sig.xs("BTC", level=SYMBOL_LEVEL).dropna().iloc[-1]
    assert last_btc == pytest.approx(-0.0005)
    prov = {c.column: c for c in res.report.columns}
    assert prov["funding_rate"].pit_quality == "true"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_build_enrichers_registry():
    from service_xs_pipeline import XSConfig, build_enrichers

    cfg = XSConfig.model_validate(
        {
            "asset_class": "crypto",
            "data": {
                "source": "free",
                "vendor": "binance",
                "symbols": ["BTC", "ETH"],
                "enrich": ["funding", "basis", "mcap"],
            },
            "mcaps": {"BTC": 1300, "ETH": 400},
        }
    )
    enrichers = build_enrichers(cfg)
    cols = sorted(sum([e.columns() for e in enrichers], []))
    assert cols == ["basis", "funding_rate", "mcap"]


def test_build_enrichers_skips_mcap_without_data():
    from service_xs_pipeline import XSConfig, build_enrichers

    cfg = XSConfig.model_validate(
        {
            "asset_class": "crypto",
            "data": {"source": "free", "enrich": ["mcap"]},  # нет cfg.mcaps → пропуск
        }
    )
    assert build_enrichers(cfg) == []


# ---------------------------------------------------------------------------
# adapter regression: get_klines(market=...) not use_futures
# ---------------------------------------------------------------------------
def test_binance_adapter_get_klines_market_kwarg():
    from adapters.binance.market_data import BinanceMarketDataAdapter
    from adapters.models import ExchangeVendor

    captured = {}

    class StubClient:
        def get_klines(self, **kwargs):
            captured.update(kwargs)
            return []

    adapter = BinanceMarketDataAdapter(vendor=ExchangeVendor.BINANCE, config={"use_futures": False})
    adapter._client = StubClient()
    adapter.get_bars("BTCUSDT", "1d", limit=10)
    assert captured.get("market") == "spot"
    assert "use_futures" not in captured  # баг исправлен
    # futures режим → market=futures
    captured.clear()
    fadapter = BinanceMarketDataAdapter(vendor=ExchangeVendor.BINANCE, config={"use_futures": True})
    fadapter._client = StubClient()
    fadapter.get_bars("BTCUSDT", "1d", limit=10)
    assert captured.get("market") == "futures"
