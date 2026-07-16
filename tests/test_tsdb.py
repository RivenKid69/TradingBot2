# -*- coding: utf-8 -*-
"""Тесты TS-DB абстракции (P2): partitioned parquet backend + facade + backend factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.tsdb import (
    ParquetTSBackend, SYM_COL, TS_COL, TimeSeriesStore, make_backend,
)


def _long(symbols=("A", "B", "C"), n=20, t0=1000):
    rows = []
    for s in symbols:
        for i in range(n):
            rows.append({TS_COL: t0 + i, SYM_COL: s, "close": 100.0 + i, "vol": 1.0 + i})
    return pd.DataFrame(rows)


def test_parquet_roundtrip(tmp_path):
    be = ParquetTSBackend(root=str(tmp_path / "db"))
    be.write("prices", _long())
    out = be.read("prices")
    assert len(out) == 60
    assert set(out[SYM_COL].unique()) == {"A", "B", "C"}
    assert "prices" in be.tables()


def test_partitioned_symbol_read(tmp_path):
    be = ParquetTSBackend(root=str(tmp_path / "db"))
    be.write("prices", _long())
    out = be.read("prices", symbols=["A"])
    assert set(out[SYM_COL].unique()) == {"A"}
    assert len(out) == 20


def test_time_range_and_columns(tmp_path):
    be = ParquetTSBackend(root=str(tmp_path / "db"))
    be.write("prices", _long(n=20, t0=1000))
    out = be.read("prices", symbols=["A"], start_ms=1005, end_ms=1010, columns=["close"])
    assert out[TS_COL].min() == 1005 and out[TS_COL].max() == 1010
    assert set(out.columns) >= {TS_COL, SYM_COL, "close"}
    assert "vol" not in out.columns


def test_upsert_dedup(tmp_path):
    be = ParquetTSBackend(root=str(tmp_path / "db"))
    be.write("p", _long(symbols=("A",), n=5, t0=1000))
    # перезапись того же ts с новым значением → dedup keep last
    upd = pd.DataFrame([{TS_COL: 1002, SYM_COL: "A", "close": 999.0, "vol": 9.0}])
    be.write("p", upd)
    out = be.read("p", symbols=["A"])
    assert len(out) == 5                       # без дублей
    assert out[out[TS_COL] == 1002]["close"].iloc[0] == 999.0


def test_facade_panel_roundtrip(tmp_path):
    store = TimeSeriesStore(ParquetTSBackend(root=str(tmp_path / "db")))
    long = _long(symbols=("A", "B"), n=10)
    panel = long.set_index([TS_COL, SYM_COL]).sort_index()
    store.write_panel("feat", panel)
    back = store.read_panel("feat", symbols=["A"])
    assert isinstance(back.index, pd.MultiIndex)
    assert set(back.index.get_level_values(1).unique()) == {"A"}
    assert "close" in back.columns


def test_backend_factory_fallback(tmp_path):
    # нет драйвера clickhouse/timescale → graceful fallback на parquet
    be1 = make_backend("clickhouse", root=str(tmp_path / "ch"))
    be2 = make_backend("timescale", root=str(tmp_path / "ts"))
    be3 = make_backend("parquet", root=str(tmp_path / "pq"))
    assert all(b.available() for b in (be1, be2, be3))
    assert isinstance(be3, ParquetTSBackend)
    # все умеют write/read
    be1.write("t", _long(symbols=("A",), n=3))
    assert len(be1.read("t")) == 3
