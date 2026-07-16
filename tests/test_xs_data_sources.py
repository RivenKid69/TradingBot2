# -*- coding: utf-8 -*-
"""
Stage A2 tests — impl_data_sources (free + BYO + total-return + PIT leakage-guard).

Сетевых вызовов нет: free-источники тестируются через внедрённые (DI) фейковые
адаптеры/фетчеры; BYO — через временные parquet/CSV.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from impl_panel import PanelBuilder
import impl_data_sources as ds

T0_SEC = 1_700_000_000
STEP_SEC = 3_600
T0_MS = T0_SEC * 1000
H_MS = STEP_SEC * 1000


# ---------------------------------------------------------------------------
# Метаданные / PIT-флаги
# ---------------------------------------------------------------------------
def test_meta_pit_quality_validation():
    m = ds.DataSourceMeta(name="x", vendor="v", kind="price", pit_quality="true")
    assert m.to_dict()["pit_quality"] == "true"
    with pytest.raises(ValueError):
        ds.DataSourceMeta(name="x", vendor="v", kind="price", pit_quality="kinda")


# ---------------------------------------------------------------------------
# bars_to_frame
# ---------------------------------------------------------------------------
def _bar(ts, o, h, l, c, v):
    return SimpleNamespace(ts=ts, open=o, high=h, low=l, close=c, volume_base=v)


def test_bars_to_frame():
    bars = [_bar(T0_MS, 10, 11, 9, 10.5, 100), _bar(T0_MS + H_MS, 10.5, 12, 10, 11, 200)]
    frame = ds.bars_to_frame(bars, "AAA")
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert frame["symbol"].unique().tolist() == ["AAA"]
    assert frame["close"].tolist() == [10.5, 11.0]
    assert frame["volume"].tolist() == [100.0, 200.0]


# ---------------------------------------------------------------------------
# AdapterPriceSource (free, DI fake adapter)
# ---------------------------------------------------------------------------
class _FakeAdapter:
    def __init__(self, by_symbol, raise_for=None):
        self._b = by_symbol
        self._raise_for = set(raise_for or ())

    def get_bars(self, symbol, timeframe, *, limit=500, start_ts=None, end_ts=None):
        if symbol in self._raise_for:
            raise RuntimeError("boom")
        return self._b.get(symbol, [])


def test_adapter_price_source_happy_path():
    fake = _FakeAdapter(
        {
            "AAA": [_bar(T0_MS, 10, 11, 9, 10, 1), _bar(T0_MS + H_MS, 10, 12, 10, 11, 2)],
            "BBB": [_bar(T0_MS, 20, 21, 19, 20, 1)],
        }
    )
    src = ds.AdapterPriceSource(vendor="fake", adapter=fake)
    assert src.meta.pit_quality == "true"
    assert src.available() is True
    out = src.get_bars(["AAA", "BBB"], "1h")
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"]["close"].tolist() == [10.0, 11.0]


def test_adapter_price_source_graceful_failures():
    fake = _FakeAdapter({"AAA": [_bar(T0_MS, 10, 11, 9, 10, 1)]}, raise_for={"BBB"})
    src = ds.AdapterPriceSource(vendor="fake", adapter=fake)
    out = src.get_bars(["AAA", "BBB"], "1h")
    assert "AAA" in out and "BBB" not in out  # сбой по BBB не валит весь запрос


# ---------------------------------------------------------------------------
# build_price_panel: интеграция с A1
# ---------------------------------------------------------------------------
def test_build_price_panel_from_source():
    fake = _FakeAdapter(
        {
            "AAA": [_bar(T0_MS, 10, 11, 9, 10, 1), _bar(T0_MS + H_MS, 10, 12, 10, 11, 2)],
            "BBB": [_bar(T0_MS, 20, 21, 19, 20, 1), _bar(T0_MS + H_MS, 20, 22, 20, 21, 2)],
        }
    )
    src = ds.AdapterPriceSource(vendor="fake", adapter=fake)
    panel = ds.build_price_panel(src, ["AAA", "BBB"], "1h")
    cp.validate_panel(panel, allow_empty=False)
    assert cp.panel_symbols(panel) == ["AAA", "BBB"]
    assert "close" in panel.columns


# ---------------------------------------------------------------------------
# ParquetPriceSource (BYO)
# ---------------------------------------------------------------------------
def test_parquet_price_source_byo(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": np.array([T0_MS, T0_MS + H_MS, T0_MS + 2 * H_MS], dtype="int64"),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [100, 110, 120],
        }
    )
    p = tmp_path / "AAA.parquet"
    df.to_parquet(p)
    src = ds.ParquetPriceSource(root=str(tmp_path))
    assert src.meta.free is False and src.meta.pit_quality == "true"
    out = src.get_bars(["AAA"])
    assert "AAA" in out and len(out["AAA"]) == 3
    # фильтр диапазона
    out2 = src.get_bars(["AAA"], start_ms=T0_MS + H_MS, end_ms=T0_MS + 2 * H_MS)
    assert len(out2["AAA"]) == 1


# ---------------------------------------------------------------------------
# ParquetFundamentals (BYO) + PIT leakage-guard через asof_join
# ---------------------------------------------------------------------------
def test_parquet_fundamentals_pit_leakage_guard(tmp_path):
    # панель цен по AAA (3 бара)
    pf = pd.DataFrame(
        {
            "timestamp": np.array([T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC], dtype="int64"),
            "symbol": "AAA",
            "close": [100.0, 101.0, 102.0],
        }
    )
    panel = PanelBuilder.from_frames({"AAA": pf})
    bar0, bar1, bar2 = T0_MS, T0_MS + H_MS, T0_MS + 2 * H_MS

    # фундаментал опубликован МЕЖДУ bar0 и bar1
    fdf = pd.DataFrame(
        {"publish_ts": [bar0 + 1_800_000], "symbol": ["AAA"], "ep": [0.05]}
    )
    fpath = tmp_path / "fund.parquet"
    fdf.to_parquet(fpath)

    fsrc = ds.ParquetFundamentals(str(fpath))
    assert fsrc.meta.pit_quality == "true"
    got = fsrc.get_fundamentals(["AAA"], fields=["ep"])
    assert list(got.columns) == ["publish_ts", "symbol", "ep"]

    joined = PanelBuilder.asof_join(panel, got, value_cols=["ep"], ts_col="publish_ts")
    # bar0 — публикации ещё не было → NaN (нет look-ahead)
    assert np.isnan(joined.loc[(bar0, "AAA"), "ep"])
    # bar1, bar2 — фундаментал доступен
    assert joined.loc[(bar1, "AAA"), "ep"] == pytest.approx(0.05)
    assert joined.loc[(bar2, "AAA"), "ep"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# FreeFundamentals (snapshot, PIT none, DI fetcher)
# ---------------------------------------------------------------------------
def test_free_fundamentals_snapshot_is_pit_none():
    fetched = {
        "AAA": {"pe": 15.0, "pb": 2.0},
        "BBB": {"pe": 25.0, "pb": 4.0},
    }
    src = ds.FreeFundamentals(asof_ms=T0_MS, fetcher=lambda s: fetched[s])
    assert src.meta.pit_quality == "none"  # снимок, не для бэктеста
    out = src.get_fundamentals(["AAA", "BBB"], fields=["pe", "pb"])
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert (out["publish_ts"] == T0_MS).all()
    assert out.loc[out["symbol"] == "AAA", "pe"].iloc[0] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------
def test_total_return_index_split():
    # сплит 2:1 на втором баре: сырая цена 50 (пост-сплит), компенсируем ×2 → не «обвал»
    close = pd.Series([100.0, 50.0, 55.0], index=[10, 20, 30])
    tr = ds.total_return_index(close, splits={20: 2.0})
    assert tr.tolist() == pytest.approx([100.0, 100.0, 110.0])


def test_total_return_index_dividend():
    # дивиденд $5 на плоской цене → +5% тотал-ретёрн
    close = pd.Series([100.0, 100.0, 100.0], index=[10, 20, 30])
    tr = ds.total_return_index(close, dividends={20: 5.0})
    assert tr.tolist() == pytest.approx([100.0, 105.0, 105.0])


def test_add_total_return_dataframe():
    df = pd.DataFrame(
        {
            "timestamp": np.array([T0_MS, T0_MS + H_MS, T0_MS + 2 * H_MS], dtype="int64"),
            "close": [100.0, 100.0, 100.0],
        }
    )
    out = ds.add_total_return(df, dividends={T0_MS + H_MS: 5.0})
    assert "tr_close" in out.columns
    assert out["tr_close"].tolist() == pytest.approx([100.0, 105.0, 105.0])
