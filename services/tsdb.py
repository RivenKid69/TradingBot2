# -*- coding: utf-8 -*-
"""
services/tsdb.py
================

Абстракция time-series хранилища (P2): единый API поверх бэкендов, чтобы уйти от
плоского parquet к масштабируемому хранилищу (ClickHouse/TimescaleDB) для 10³ символов /
суб-минутных данных — без переписывания пайплайна.

  * ``ParquetTSBackend`` — **партиционирование по символу** (файл на символ) → быстрый
    выбор подмножества символов + time-range без полного скана. Работает «из коробки».
  * ``ClickHouseTSBackend`` / ``TimescaleTSBackend`` — реальные SQL-адаптеры через
    DI-драйвер (``clickhouse_driver`` / ``psycopg2``); при отсутствии драйвера
    ``available()=False`` (graceful) — код пайплайна не меняется, бэкенд переключается конфигом.
  * ``TimeSeriesStore`` — фасад: ``write_panel`` / ``read_panel`` (MultiIndex (ts_ms, symbol)).

Формат: long (``ts_ms``, ``symbol``, <feature-колонки>). Слой services.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

TS_COL = "ts_ms"
SYM_COL = "symbol"


class TSBackend(Protocol):
    def available(self) -> bool: ...
    def write(self, table: str, df: pd.DataFrame) -> None: ...
    def read(
        self,
        table: str,
        *,
        symbols: Optional[Sequence[str]] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame: ...
    def tables(self) -> List[str]: ...


# ---------------------------------------------------------------------------
# Parquet backend (partitioned by symbol)
# ---------------------------------------------------------------------------
class ParquetTSBackend:
    """Партиционирование по символу: ``<root>/<table>/symbol=<S>.parquet``."""

    def __init__(self, root: str = "data/tsdb") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def available(self) -> bool:
        return True

    def _tdir(self, table: str) -> str:
        return os.path.join(self.root, table)

    @staticmethod
    def _safe(sym: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(sym))

    def write(self, table: str, df: pd.DataFrame) -> None:
        if TS_COL not in df.columns or SYM_COL not in df.columns:
            raise ValueError(f"df must have '{TS_COL}' and '{SYM_COL}' columns")
        tdir = self._tdir(table)
        os.makedirs(tdir, exist_ok=True)
        for sym, g in df.groupby(SYM_COL, sort=False):
            path = os.path.join(tdir, f"symbol={self._safe(sym)}.parquet")
            g2 = g.drop(columns=[SYM_COL]).sort_values(TS_COL)
            if os.path.exists(path):
                old = pd.read_parquet(path)
                merged = pd.concat([old, g2], ignore_index=True)
                merged = merged.drop_duplicates(subset=[TS_COL], keep="last").sort_values(TS_COL)
                merged.to_parquet(path, index=False)
            else:
                g2.to_parquet(path, index=False)

    def read(
        self,
        table: str,
        *,
        symbols: Optional[Sequence[str]] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        tdir = self._tdir(table)
        if not os.path.isdir(tdir):
            return pd.DataFrame(columns=[TS_COL, SYM_COL])
        if symbols is not None:
            files = [(s, os.path.join(tdir, f"symbol={self._safe(s)}.parquet")) for s in symbols]
            files = [(s, p) for s, p in files if os.path.exists(p)]
        else:
            files = []
            for fn in os.listdir(tdir):
                if fn.startswith("symbol=") and fn.endswith(".parquet"):
                    files.append((fn[len("symbol=") : -len(".parquet")], os.path.join(tdir, fn)))
        frames = []
        for sym, path in files:
            cols = None
            if columns is not None:
                cols = [TS_COL] + [c for c in columns if c != SYM_COL]
            d = pd.read_parquet(path, columns=cols)
            if start_ms is not None:
                d = d[d[TS_COL] >= int(start_ms)]
            if end_ms is not None:
                d = d[d[TS_COL] <= int(end_ms)]
            if len(d):
                d[SYM_COL] = sym
                frames.append(d)
        if not frames:
            return pd.DataFrame(columns=[TS_COL, SYM_COL])
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values([SYM_COL, TS_COL]).reset_index(drop=True)

    def tables(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted([d for d in os.listdir(self.root) if os.path.isdir(self._tdir(d))])


# ---------------------------------------------------------------------------
# ClickHouse / Timescale backends (DI-driver, graceful if absent)
# ---------------------------------------------------------------------------
class ClickHouseTSBackend:
    """ClickHouse-адаптер. Драйвер ``clickhouse_driver`` (lazy). MergeTree ORDER BY (symbol, ts_ms)."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 9000,
        database: str = "default",
        client: Any = None,
    ) -> None:
        self._client = client
        self._cfg = {"host": host, "port": port, "database": database}

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from clickhouse_driver import Client  # type: ignore

            self._client = Client(**self._cfg)
        except Exception:
            self._client = None
        return self._client

    def available(self) -> bool:
        return self._ensure() is not None

    def write(self, table: str, df: pd.DataFrame) -> None:
        c = self._ensure()
        if c is None:
            raise RuntimeError("clickhouse driver unavailable")
        cols = list(df.columns)
        ddl_cols = ", ".join(
            f"{col} " + ("Int64" if col == TS_COL else "String" if col == SYM_COL else "Float64")
            for col in cols
        )
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ({ddl_cols}) "
            f"ENGINE = MergeTree ORDER BY ({SYM_COL}, {TS_COL})"
        )
        c.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES", df.to_dict("records"))

    def read(
        self, table: str, *, symbols=None, start_ms=None, end_ms=None, columns=None
    ) -> pd.DataFrame:
        c = self._ensure()
        if c is None:
            raise RuntimeError("clickhouse driver unavailable")
        sel = "*" if not columns else ", ".join([TS_COL, SYM_COL] + list(columns))
        where = []
        if symbols is not None:
            lst = ", ".join("'%s'" % s for s in symbols)
            where.append(f"{SYM_COL} IN ({lst})")
        if start_ms is not None:
            where.append(f"{TS_COL} >= {int(start_ms)}")
        if end_ms is not None:
            where.append(f"{TS_COL} <= {int(end_ms)}")
        q = (
            f"SELECT {sel} FROM {table}"
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" ORDER BY {SYM_COL}, {TS_COL}"
        )
        rows, meta = c.execute(q, with_column_types=True)
        return pd.DataFrame(rows, columns=[m[0] for m in meta])

    def tables(self) -> List[str]:
        c = self._ensure()
        if c is None:
            return []
        return [r[0] for r in c.execute("SHOW TABLES")]


class TimescaleTSBackend:
    """TimescaleDB (PostgreSQL hypertable). Драйвер ``psycopg2`` (lazy)."""

    def __init__(self, *, dsn: str = "", conn: Any = None) -> None:
        self._conn = conn
        self._dsn = dsn

    def _ensure(self) -> Any:
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2  # type: ignore

            self._conn = psycopg2.connect(self._dsn)
        except Exception:
            self._conn = None
        return self._conn

    def available(self) -> bool:
        return self._ensure() is not None

    def write(self, table: str, df: pd.DataFrame) -> None:  # pragma: no cover - нужен live PG
        conn = self._ensure()
        if conn is None:
            raise RuntimeError("psycopg2 unavailable")
        cols = list(df.columns)
        ddl = ", ".join(
            f"{c} " + ("BIGINT" if c == TS_COL else "TEXT" if c == SYM_COL else "DOUBLE PRECISION")
            for c in cols
        )
        cur = conn.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")
        try:
            cur.execute(f"SELECT create_hypertable('{table}', '{TS_COL}', if_not_exists => TRUE)")
        except Exception:
            pass
        args = ",".join(
            cur.mogrify("(" + ",".join(["%s"] * len(cols)) + ")", tuple(r)).decode()
            for r in df.itertuples(index=False)
        )
        cur.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES " + args)
        conn.commit()

    def read(
        self, table: str, *, symbols=None, start_ms=None, end_ms=None, columns=None
    ) -> pd.DataFrame:  # pragma: no cover
        conn = self._ensure()
        if conn is None:
            raise RuntimeError("psycopg2 unavailable")
        sel = "*" if not columns else ", ".join([TS_COL, SYM_COL] + list(columns))
        where, params = [], []
        if symbols is not None:
            where.append(f"{SYM_COL} = ANY(%s)")
            params.append(list(symbols))
        if start_ms is not None:
            where.append(f"{TS_COL} >= %s")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append(f"{TS_COL} <= %s")
            params.append(int(end_ms))
        q = (
            f"SELECT {sel} FROM {table}"
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" ORDER BY {SYM_COL}, {TS_COL}"
        )
        return pd.read_sql(q, conn, params=params or None)

    def tables(self) -> List[str]:  # pragma: no cover
        conn = self._ensure()
        if conn is None:
            return []
        return pd.read_sql("SELECT tablename FROM pg_tables WHERE schemaname='public'", conn)[
            "tablename"
        ].tolist()


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
class TimeSeriesStore:
    """Фасад: panel (MultiIndex (ts_ms, symbol)) ↔ long-хранилище."""

    def __init__(self, backend: Optional[TSBackend] = None) -> None:
        self.backend = backend or ParquetTSBackend()

    def write_panel(self, table: str, panel: pd.DataFrame) -> None:
        long = panel.reset_index()
        # нормализуем имена уровней индекса к ts_ms/symbol
        cols = list(long.columns)
        if TS_COL not in cols or SYM_COL not in cols:
            lvl = panel.index
            names = list(lvl.names) if isinstance(lvl, pd.MultiIndex) else [lvl.name]
            ren = {}
            if len(names) >= 2:
                ren[names[0]] = TS_COL
                ren[names[1]] = SYM_COL
            long = long.rename(columns=ren)
        self.backend.write(table, long)

    def read_panel(
        self, table: str, *, symbols=None, start_ms=None, end_ms=None, columns=None
    ) -> pd.DataFrame:
        long = self.backend.read(
            table, symbols=symbols, start_ms=start_ms, end_ms=end_ms, columns=columns
        )
        if not len(long):
            return pd.DataFrame()
        return long.set_index([TS_COL, SYM_COL]).sort_index()


def make_backend(kind: str = "parquet", **kwargs: Any) -> TSBackend:
    """Фабрика бэкенда по имени. CH/TS недоступны (нет драйвера) → graceful fallback на parquet."""
    k = (kind or "parquet").lower()
    root = kwargs.pop("root", "data/tsdb")  # root — только для parquet (fallback)
    if k == "clickhouse":
        b = ClickHouseTSBackend(**kwargs)
        return b if b.available() else ParquetTSBackend(root=root)
    if k in ("timescale", "timescaledb", "postgres"):
        b = TimescaleTSBackend(**kwargs)
        return b if b.available() else ParquetTSBackend(root=root)
    return ParquetTSBackend(root=root)


__all__ = [
    "TS_COL",
    "SYM_COL",
    "TSBackend",
    "ParquetTSBackend",
    "ClickHouseTSBackend",
    "TimescaleTSBackend",
    "TimeSeriesStore",
    "make_backend",
]
