# -*- coding: utf-8 -*-
"""
Stage D0 tests — unified data-assembly (core_xs_data + impl_data_cache + service_xs_data).

  * DataAssembler собирает prices + enrichers (DI, без сети); провенанс/pit_quality по колонкам
  * AsofEnricher PIT-безопасен (publish_lag → НЕ тянет будущее)
  * ParquetCache: put/get hit + TTL miss (атомарно)
  * DataQualityReport: coverage/verdict корректны; pit_quality=none → warn
  * интеграция: data_quality_for_config (synthetic) + API /api/xs/data_quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from core_xs_data import ColumnProvenance, DataQualityReport, PIT_NONE, PIT_TRUE
from impl_data_cache import ParquetCache
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import (
    DataAssembler, ColumnMapEnricher, AsofEnricher, FunctionEnricher, build_quality_report,
)

T0, STEP = 1_700_000_000, 86_400


# ---------------------------------------------------------------------------
# fake price source (DI, без сети)
# ---------------------------------------------------------------------------
class FakePriceSource:
    def __init__(self, vendor="fake", n=12, pit="true"):
        self.meta = DataSourceMeta(name=f"free:{vendor}", vendor=vendor, kind="price", pit_quality=pit)
        self.n = n
        self.calls = []

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        self.calls.append(list(symbols))
        out = {}
        for s in symbols:
            ts = [T0 + i * STEP for i in range(self.n)]
            out[s] = pd.DataFrame({"timestamp": ts, "symbol": s,
                                   "close": 100.0 * (1 + 0.01 * np.arange(self.n)),
                                   "volume": 1000.0})
        return out


# ---------------------------------------------------------------------------
# assembler + provenance
# ---------------------------------------------------------------------------
def test_assemble_prices_only_provenance():
    src = FakePriceSource(vendor="binance")
    res = DataAssembler(src).assemble(["BTC", "ETH"], "1d")
    assert "close" in res.panel.columns
    assert res.report.n_symbols == 2
    cols = {c.column: c for c in res.report.columns}
    assert cols["close"].vendor == "binance" and cols["close"].pit_quality == "true"
    assert res.report.verdict() == "ok"


def test_assemble_with_column_map_enricher():
    src = FakePriceSource()
    enr = ColumnMapEnricher({"BTC": 1300.0, "ETH": 400.0}, "mcap", pit_quality="approx")
    res = DataAssembler(src, enrichers=[enr]).assemble(["BTC", "ETH"], "1d")
    assert "mcap" in res.panel.columns
    btc = res.panel.xs("BTC", level=SYMBOL_LEVEL)["mcap"]
    assert (btc == 1300.0).all()
    prov = {c.column: c for c in res.report.columns}
    assert prov["mcap"].pit_quality == "approx" and prov["mcap"].source == "static:mcap"


def test_function_enricher_di():
    src = FakePriceSource()
    def add_col(panel):
        out = panel.copy(); out["feat"] = out["close"] * 2.0; return out
    enr = FunctionEnricher(add_col, columns=["feat"], name="x2")
    res = DataAssembler(src, enrichers=[enr]).assemble(["BTC"], "1d")
    assert np.allclose(res.panel["feat"].to_numpy(), res.panel["close"].to_numpy() * 2.0)


# ---------------------------------------------------------------------------
# PIT safety
# ---------------------------------------------------------------------------
def test_asof_enricher_is_pit_safe():
    src = FakePriceSource(n=10)
    # фундаментал публикуется на баре 5 → бары 0..4 ДОЛЖНЫ быть NaN (нет look-ahead)
    pub_ts = T0 + 5 * STEP
    def provider(symbols):
        return pd.DataFrame({"publish_ts": [pub_ts], "symbol": ["BTC"], "earnings": [8.0]})
    enr = AsofEnricher(provider, columns=["earnings"], publish_ts_col="publish_ts", pit_quality="true")
    res = DataAssembler(src, enrichers=[enr]).assemble(["BTC"], "1d")
    e = res.panel.xs("BTC", level=SYMBOL_LEVEL)["earnings"].to_numpy()
    assert np.isnan(e[:5]).all()        # до публикации — NaN (PIT)
    assert (e[5:] == 8.0).all()         # после — значение


def test_asof_enricher_publish_lag():
    src = FakePriceSource(n=10)
    pub_ts = T0 + 3 * STEP
    def provider(symbols):
        return pd.DataFrame({"publish_ts": [pub_ts], "symbol": ["BTC"], "roe": [0.2]})
    # лаг в 2 бара → значение доступно только с бара 5
    enr = AsofEnricher(provider, columns=["roe"], publish_lag_ms=2 * STEP * 1000)
    res = DataAssembler(src, enrichers=[enr]).assemble(["BTC"], "1d")
    r = res.panel.xs("BTC", level=SYMBOL_LEVEL)["roe"].to_numpy()
    assert np.isnan(r[:5]).all() and (r[5:] == 0.2).all()


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_parquet_cache_hit_and_ttl(tmp_path):
    cache = ParquetCache(root=str(tmp_path))
    df = pd.DataFrame({"timestamp": [T0], "symbol": "BTC", "close": [100.0]})
    assert cache.put("binance", "BTC", "1d", df) is True
    got = cache.get("binance", "BTC", "1d")
    assert got is not None and len(got) == 1
    # TTL: now далеко в будущем → устарел → miss
    future = int((T0 + 10**9) * 1000)
    assert cache.get("binance", "BTC", "1d", ttl_ms=1, now_ms=future) is None


def test_assembler_uses_cache(tmp_path):
    src = FakePriceSource(vendor="binance")
    cache = ParquetCache(root=str(tmp_path))
    asm = DataAssembler(src, cache=cache)
    asm.assemble(["BTC"], "1d")                 # первый раз → fetch + put
    n_after_first = len(src.calls)
    asm.assemble(["BTC"], "1d")                 # второй раз → из кэша, без fetch
    assert len(src.calls) == n_after_first      # повторного get_bars не было


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------
def test_quality_report_coverage_and_verdict():
    frame = pd.DataFrame({"timestamp": [T0, T0 + STEP, T0 + 2 * STEP], "symbol": "BTC",
                          "close": [1.0, 2.0, 3.0], "iv": [np.nan, np.nan, 0.2]})
    panel = PanelBuilder.from_frames({"BTC": frame})
    prov = [ColumnProvenance("close", "p", "binance", PIT_TRUE),
            ColumnProvenance("iv", "e", "byo", PIT_NONE)]
    rep = build_quality_report(panel, prov, price_col="close")
    assert rep.coverage["close"] == pytest.approx(1.0)
    assert rep.coverage["iv"] == pytest.approx(1 / 3)
    assert rep.worst_pit == "none"
    assert rep.verdict() == "warn"                       # none-колонка + низкое покрытие
    assert any("pit_quality=none" in w for w in rep.warnings)


# ---------------------------------------------------------------------------
# pipeline + API integration
# ---------------------------------------------------------------------------
def test_data_quality_for_config_synthetic():
    from service_xs_pipeline import XSConfig, data_quality_for_config
    cfg = XSConfig.model_validate({"asset_class": "crypto",
                                   "data": {"source": "synthetic", "symbols": ["BTC", "ETH"], "synthetic_bars": 30}})
    rep = data_quality_for_config(cfg)
    assert rep.n_symbols == 2
    assert rep.worst_pit == "none"                       # синтетика честно помечена
    assert "close" in rep.coverage


def test_api_data_quality():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xs_api import register_xs_routes
    app = FastAPI(); register_xs_routes(app)
    client = TestClient(app)
    r = client.post("/api/xs/data_quality",
                    json={"asset_class": "crypto", "data": {"source": "synthetic", "symbols": ["BTC", "ETH"]}})
    assert r.status_code == 200
    data = r.json()
    assert data["n_symbols"] == 2 and "verdict" in data and "columns" in data
