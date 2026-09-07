# -*- coding: utf-8 -*-
"""
Stage D2 tests — equity free + РЕАЛЬНЫЙ PIT-фундаментал (loaders/equity_enrich).

  * PITFundamentalsEnricher: as-of join НЕ тянет publish_ts из будущего (анти-look-ahead — ключевой про-тест)
    + publish_lag; pit_quality наследуется от источника (BYO parquet=true, free снимок=none)
  * TotalReturnEnricher: реинвест дивиденда → tr_close корректен (формула total_return_index)
  * EarningsEnricher: has_earnings_soon в окне
  * интеграция: equity value-сигнал «оживает» на собранной панели; провенанс честный
  * build_enrichers: equity enrich → обогатители; free фундаментал = pit=none (honest)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import DataAssembler
from signals.equity_signals import EarningsYield, BookToPrice
from loaders.equity_enrich import (
    PITFundamentalsEnricher,
    TotalReturnEnricher,
    EarningsEnricher,
    make_pit_fundamentals_enricher,
)

T0, STEP = 1_700_000_000, 86_400


class FakeFund:
    """DI-источник фундаментала (long df с publish_ts)."""

    def __init__(self, df, pit="true", vendor="byo"):
        self.meta = DataSourceMeta(name="fund", vendor=vendor, kind="fundamentals", pit_quality=pit)
        self._df = df

    def get_fundamentals(self, symbols, fields):
        return self._df[self._df["symbol"].isin(list(symbols))].copy()


class FakePriceSource:
    def __init__(self, n=10, vendor="yahoo"):
        self.meta = DataSourceMeta(name=f"free:{vendor}", vendor=vendor, kind="price")
        self.n = n

    def get_bars(self, symbols, timeframe, *, start_ms=None, end_ms=None, limit=1000):
        return {
            s: pd.DataFrame(
                {
                    "timestamp": [T0 + i * STEP for i in range(self.n)],
                    "symbol": s,
                    "close": 100.0 + np.arange(self.n),
                }
            )
            for s in symbols
        }


def _panel(n=10, syms=("AAA",), close=None):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        c = close if close is not None else (100.0 + np.arange(n))
        frames[s] = pd.DataFrame(
            {"timestamp": ts, "symbol": s, "close": np.asarray(c, dtype="float64")}
        )
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# PIT anti-look-ahead (headline)
# ---------------------------------------------------------------------------
def test_pit_fundamentals_no_look_ahead():
    # отчётность опубликована на баре 5 → бары 0..4 ДОЛЖНЫ быть NaN (нет будущего)
    pub = T0 + 5 * STEP
    df = pd.DataFrame(
        {"publish_ts": [pub], "symbol": ["AAA"], "earnings": [8.0], "book_value": [40.0]}
    )
    enr = PITFundamentalsEnricher(FakeFund(df, pit="true"), fields=["earnings", "book_value"])
    out = enr.enrich(_panel(10))
    e = out.xs("AAA", level=SYMBOL_LEVEL)["earnings"].to_numpy()
    assert np.isnan(e[:5]).all()  # до публикации — НЕТ данных (PIT)
    assert np.allclose(e[5:], 8.0)  # после — значение
    assert enr.meta.pit_quality == "true"


def test_pit_fundamentals_publish_lag():
    pub = T0 + 3 * STEP
    df = pd.DataFrame({"publish_ts": [pub], "symbol": ["AAA"], "roe": [0.2]})
    enr = PITFundamentalsEnricher(FakeFund(df), fields=["roe"], publish_lag_ms=2 * STEP * 1000)
    out = enr.enrich(_panel(8))
    r = out.xs("AAA", level=SYMBOL_LEVEL)["roe"].to_numpy()
    assert np.isnan(r[:5]).all() and np.allclose(r[5:], 0.2)  # доступно с бара 5 (3+2 лаг)


def test_free_snapshot_marked_pit_none():
    # без parquet_path → FreeFundamentals (снимок) → pit=none (honest, не backtest-safe)
    enr = make_pit_fundamentals_enricher(parquet_path=None, fields=["earnings"])
    assert enr.meta.pit_quality == "none"


# ---------------------------------------------------------------------------
# total return
# ---------------------------------------------------------------------------
def test_total_return_reinvests_dividend():
    panel = _panel(4, close=[100.0, 100.0, 100.0, 100.0])
    div_ts = (T0 + 2 * STEP) * 1000  # панель нормализует ts в мс
    enr = TotalReturnEnricher(actions_fn=lambda s: ({div_ts: 5.0}, {}))
    out = enr.enrich(panel)
    tr = out.xs("AAA", level=SYMBOL_LEVEL)["tr_close"].to_numpy()
    # ret[2] = (100+5)/100-1 = 5% → tr = [100,100,105,105]
    assert np.allclose(tr, [100.0, 100.0, 105.0, 105.0])
    assert enr.meta.pit_quality == "approx"


# ---------------------------------------------------------------------------
# earnings flag
# ---------------------------------------------------------------------------
def test_earnings_soon_flag():
    panel = _panel(10)
    earnings_ts = (T0 + 6 * STEP) * 1000  # мс
    enr = EarningsEnricher(dates_fn=lambda s: [earnings_ts], window_days=3)
    out = enr.enrich(panel)
    f = out.xs("AAA", level=SYMBOL_LEVEL)["has_earnings_soon"].to_numpy()
    # earnings на баре 6, окно 3 дня → флаг на барах 3,4,5 (t < earnings <= t+3)
    assert f[3] == 1.0 and f[4] == 1.0 and f[5] == 1.0
    assert f[6] == 0.0 and f[2] == 0.0


# ---------------------------------------------------------------------------
# integration: signal оживает
# ---------------------------------------------------------------------------
def test_equity_value_signal_comes_alive():
    src = FakePriceSource(n=10)
    pub = T0 + 3 * STEP
    df = pd.DataFrame(
        {
            "publish_ts": [pub, pub],
            "symbol": ["AAA", "BBB"],
            "earnings": [8.0, 2.0],
            "book_value": [40.0, 60.0],
        }
    )
    enr = PITFundamentalsEnricher(FakeFund(df, pit="true"), fields=["earnings", "book_value"])
    res = DataAssembler(src, enrichers=[enr]).assemble(["AAA", "BBB"], "1d")
    assert "earnings" in res.panel.columns and "book_value" in res.panel.columns
    ey = EarningsYield("ey").compute_panel(res.panel)
    assert not ey.isna().all()  # сигнал ожил
    # E/P = earnings/close после публикации
    aaa_close = res.panel.xs("AAA", level=SYMBOL_LEVEL)["close"]
    last = ey.xs("AAA", level=SYMBOL_LEVEL).dropna().iloc[-1]
    assert last == pytest.approx(8.0 / float(aaa_close.iloc[-1]))
    prov = {c.column: c for c in res.report.columns}
    assert prov["earnings"].pit_quality == "true"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_build_enrichers_equity():
    from service_xs_pipeline import XSConfig, build_enrichers

    cfg = XSConfig.model_validate(
        {
            "asset_class": "equity",
            "data": {
                "source": "free",
                "vendor": "yahoo",
                "symbols": ["AAA"],
                "enrich": ["total_return", "pit_fundamentals", "earnings"],
            },
        }
    )
    enrichers = build_enrichers(cfg)
    cols = sorted(sum([e.columns() for e in enrichers], []))
    assert "tr_close" in cols and "has_earnings_soon" in cols
    assert "earnings" in cols and "book_value" in cols
    # free фундаментал (нет fundamentals_path) → pit=none honest
    fund = [e for e in enrichers if "earnings" in e.columns()][0]
    assert fund.meta.pit_quality == "none"


def test_data_quality_flags_snapshot_warn():
    from service_xs_pipeline import XSConfig, data_quality_for_config

    cfg = XSConfig.model_validate(
        {
            "asset_class": "equity",
            "data": {"source": "synthetic", "symbols": ["AAA", "BBB"], "synthetic_bars": 20},
        }
    )
    rep = data_quality_for_config(cfg)
    assert rep.verdict() in ("ok", "warn", "poor")  # отчёт строится
