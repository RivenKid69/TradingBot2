# -*- coding: utf-8 -*-
"""Тесты alt-data enrichers (P2): COT positioning + economic calendar в пайплайне."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL
from loaders.altdata_enrich import (
    ALTDATA_ENRICHERS, COTEnricher, EconCalendarEnricher, make_cot_enricher,
)

DAY = 86_400_000


def _panel(symbols=("EUR_USD", "GBP_USD"), n=30, t0=1_600_000_000_000):
    ts = [t0 + i * DAY for i in range(n)]
    frames = []
    for s in symbols:
        idx = pd.MultiIndex.from_arrays([ts, [s] * n], names=[TS_LEVEL, SYMBOL_LEVEL])
        frames.append(pd.DataFrame({"close": np.linspace(1.0, 1.1, n)}, index=idx))
    return pd.concat(frames).sort_index(), ts


def test_cot_asof_join_with_lag():
    panel, ts = _panel(symbols=("EUR_USD",))
    # COT-отчёт опубликован на ts[10] (publish_ts), lag=0 для простоты
    prov = lambda syms: pd.DataFrame({"publish_ts": [ts[10]], "symbol": ["EUR_USD"], "cot_net": [0.42]})
    enr = COTEnricher(prov, publish_lag_days=0)
    out = enr.enrich(panel)
    assert "cot_net" in out.columns
    col = out["cot_net"]
    # до публикации — NaN, после — 0.42
    assert pd.isna(col.iloc[5])
    assert col.iloc[15] == pytest.approx(0.42)


def test_cot_publish_lag_blocks_lookahead():
    panel, ts = _panel(symbols=("EUR_USD",))
    prov = lambda syms: pd.DataFrame({"publish_ts": [ts[10]], "symbol": ["EUR_USD"], "cot_net": [0.5]})
    enr = COTEnricher(prov, publish_lag_days=3)   # доступно только с ts[10]+3д
    out = enr.enrich(panel)
    assert pd.isna(out["cot_net"].iloc[11])       # ts[11] < publish+lag
    assert out["cot_net"].iloc[14] == pytest.approx(0.5)   # ts[13]=publish+3д → доступно


def test_econ_calendar_flag():
    panel, ts = _panel(symbols=("EUR_USD",))
    # high-impact USD событие на ts[20]
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp(ts[20], unit="ms")],
        "impact": ["High"], "currency": ["USD"], "event_name": ["NFP"],
    })
    enr = EconCalendarEnricher(events, window_days=2)
    out = enr.enrich(panel)
    assert "high_impact_soon" in out.columns
    f = out["high_impact_soon"]
    assert f.iloc[19] == 1.0 or f.iloc[18] == 1.0   # за ≤2 дня до события
    assert f.iloc[5] == 0.0                          # далеко


def test_pipeline_wires_altdata_enrichers():
    from service_xs_pipeline import XSConfig, build_enrichers
    cfg = XSConfig.model_validate({
        "mode": "cross_sectional", "asset_class": "forex",
        "data": {"source": "synthetic", "symbols": ["EUR_USD"], "enrich": ["cot", "econ_calendar"]},
        "universe": {"type": "static", "symbols": ["EUR_USD"]},
    })
    enrichers = build_enrichers(cfg)
    # оба обогатителя сконструированы (не пропущены)
    names = [getattr(getattr(e, "meta", None), "name", "") for e in enrichers]
    assert "cot" in names and "econ_calendar" in names
    assert set(ALTDATA_ENRICHERS) == {"cot", "econ_calendar"}
