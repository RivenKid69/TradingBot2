# -*- coding: utf-8 -*-
"""Тесты расширенного каталога сигналов (P2): residual mom / seasonality / sentiment / 52w / idio-vol / cot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL
from signals.common_signals import (
    COMMON_SIGNAL_KINDS, COTPositioning, IdiosyncraticVol, ResidualMomentum,
    Seasonality, Sentiment, Week52High, build_common_signal,
)


def _panel(n_bars=400, symbols=("A", "B", "C"), seed=1, extra=None):
    rng = np.random.RandomState(seed)
    t0 = 1_600_000_000_000
    step = 86_400_000
    ts = [t0 + i * step for i in range(n_bars)]
    frames = []
    for j, s in enumerate(symbols):
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, n_bars))
        idx = pd.MultiIndex.from_arrays([ts, [s] * n_bars], names=[TS_LEVEL, SYMBOL_LEVEL])
        d = {"close": px}
        if extra:
            for col, val in extra.items():
                d[col] = val(rng, n_bars, j)
        frames.append(pd.DataFrame(d, index=idx))
    return pd.concat(frames).sort_index()


def test_residual_momentum_finite():
    p = _panel()
    sig = ResidualMomentum(lookback=120, skip=10, beta_window=40).compute_panel(p)
    assert sig.index.equals(p.index)
    # последние строки должны быть конечны (достаточно истории)
    last = sig.groupby(level=SYMBOL_LEVEL).tail(5)
    assert np.isfinite(last.to_numpy()).any()


def test_seasonality_pit_safe_finite():
    p = _panel(n_bars=800)   # >2 года → есть прошлые те же месяцы
    sig = Seasonality().compute_panel(p)
    assert sig.index.equals(p.index)
    assert np.isfinite(sig.dropna().to_numpy()).all()
    # первые наблюдения месяца (нет прошлого) → NaN (PIT)
    assert sig.isna().any()


def test_sentiment_byo_slot():
    # без колонки → NaN
    p = _panel()
    assert Sentiment().compute_panel(p).isna().all()
    # с колонкой → значения
    p2 = _panel(extra={"sentiment": lambda rng, n, j: rng.normal(0, 1, n)})
    s = Sentiment().compute_panel(p2)
    assert np.isfinite(s.to_numpy()).all()


def test_week52high_non_positive():
    p = _panel()
    sig = Week52High(window=200).compute_panel(p)
    vals = sig.dropna().to_numpy()
    assert (vals <= 1e-9).all()       # цена ≤ rolling-max → прокси ≤ 0
    assert np.isfinite(vals).all()


def test_idio_vol_negative():
    p = _panel()
    sig = IdiosyncraticVol(window=40).compute_panel(p)
    vals = sig.dropna().to_numpy()
    assert (vals <= 1e-12).all()      # −std ≤ 0
    assert np.isfinite(vals).all()


def test_cot_byo_slot():
    p = _panel()
    assert COTPositioning().compute_panel(p).isna().all()
    p2 = _panel(extra={"cot_net": lambda rng, n, j: rng.normal(0, 1, n)})
    assert np.isfinite(COTPositioning().compute_panel(p2).to_numpy()).all()


def test_factory_and_kinds():
    assert set(COMMON_SIGNAL_KINDS) == {
        "residual_momentum", "seasonality", "sentiment", "high_52w", "idio_vol", "cot"}
    sig = build_common_signal("residual_momentum", "rm", lookback=60, skip=5)
    assert isinstance(sig, ResidualMomentum) and sig.name == "rm"


def test_pipeline_integration():
    from service_xs_pipeline import XSConfig, build_signal_library, load_panel
    cfg = XSConfig.model_validate({
        "mode": "cross_sectional", "asset_class": "equity",
        "data": {"source": "synthetic", "symbols": ["A", "B", "C", "D"], "synthetic_bars": 320},
        "universe": {"type": "static", "symbols": ["A", "B", "C", "D"]},
        "signals": [
            {"name": "rm", "kind": "residual_momentum", "lookback": 120, "skip": 10, "transforms": ["zscore"]},
            {"name": "seas", "kind": "seasonality", "transforms": ["zscore"]},
            {"name": "h52", "kind": "high_52w", "lookback": 200, "transforms": ["zscore"]},
            {"name": "iv", "kind": "idio_vol", "vol_window": 40, "transforms": ["zscore"]},
        ],
    })
    panel = load_panel(cfg)
    lib = build_signal_library(cfg)
    out = lib.compute(panel)
    assert set(["rm", "seas", "h52", "iv"]).issubset(out.columns)
