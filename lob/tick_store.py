# -*- coding: utf-8 -*-
"""
lob/tick_store.py
=================

Persistent tick / L1 / L2 / L3 store for microstructure research (P2 #19).

Previously the platform had live streaming (`stream_ticks`) and an L3 *simulator*,
but nothing **persisted** market microstructure for research — you couldn't replay
or query historical ticks/quotes/depth. This module provides a durable, queryable
store:

  * append trades (L1 prints), top-of-book quotes (L1), and depth snapshots (L2/L3);
  * partitioned parquet on disk (``<root>/<kind>/<symbol>/<YYYYMMDD>.parquet``) with a
    buffered writer (flush by size/explicit), so high-rate streams don't thrash disk;
  * query back by symbol + time range + kind for alpha/decay/impact research;
  * a ``record_from_stream`` helper to wire any adapter ``stream_ticks`` into the store.

Depends only on pandas/pyarrow (already used project-wide). Falls back to CSV if
parquet engine is unavailable.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

TRADE = "trade"
QUOTE = "quote"  # L1 top of book
DEPTH = "depth"  # L2/L3 book snapshot (levels flattened)


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y%m%d")


@dataclass
class TickStore:
    """Durable, buffered, queryable microstructure store."""

    root: str = "data/ticks"
    flush_every: int = 1000  # rows buffered per (kind,symbol) before auto-flush
    fmt: str = "parquet"  # parquet | csv

    _buffers: Dict[tuple, List[Dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # -- recording ----------------------------------------------------------
    def _append(self, kind: str, symbol: str, row: Dict[str, Any]) -> None:
        with self._lock:
            key = (kind, str(symbol))
            buf = self._buffers.setdefault(key, [])
            buf.append(row)
            if len(buf) >= self.flush_every:
                self._flush_key(key)

    def record_trade(
        self, symbol: str, ts_ms: int, price: float, size: float, side: str = "", **extra: Any
    ) -> None:
        self._append(
            TRADE,
            symbol,
            {
                "ts_ms": int(ts_ms),
                "price": float(price),
                "size": float(size),
                "side": str(side),
                **extra,
            },
        )

    def record_quote(
        self,
        symbol: str,
        ts_ms: int,
        bid: float,
        ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
        **extra: Any,
    ) -> None:
        self._append(
            QUOTE,
            symbol,
            {
                "ts_ms": int(ts_ms),
                "bid": float(bid),
                "ask": float(ask),
                "bid_size": float(bid_size),
                "ask_size": float(ask_size),
                "mid": (float(bid) + float(ask)) / 2.0,
                **extra,
            },
        )

    def record_depth(
        self,
        symbol: str,
        ts_ms: int,
        bids: List[tuple],
        asks: List[tuple],
        *,
        levels: int = 10,
        **extra: Any,
    ) -> None:
        """Record an L2/L3 snapshot. ``bids``/``asks`` = list of (price, size) best-first."""
        row: Dict[str, Any] = {"ts_ms": int(ts_ms), **extra}
        for i in range(levels):
            bp, bs = bids[i] if i < len(bids) else (None, None)
            ap, as_ = asks[i] if i < len(asks) else (None, None)
            row[f"bid_px_{i}"] = float(bp) if bp is not None else None
            row[f"bid_sz_{i}"] = float(bs) if bs is not None else None
            row[f"ask_px_{i}"] = float(ap) if ap is not None else None
            row[f"ask_sz_{i}"] = float(as_) if as_ is not None else None
        self._append(DEPTH, symbol, row)

    # -- persistence --------------------------------------------------------
    def _path(self, kind: str, symbol: str, ts_ms: int) -> str:
        d = os.path.join(self.root, kind, str(symbol))
        os.makedirs(d, exist_ok=True)
        ext = "parquet" if self.fmt == "parquet" else "csv"
        return os.path.join(d, f"{_day(ts_ms)}.{ext}")

    def _write(self, path: str, df: pd.DataFrame) -> None:
        if os.path.exists(path):
            try:
                prev = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
                df = pd.concat([prev, df], ignore_index=True)
            except Exception:
                pass
        try:
            if path.endswith(".parquet"):
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
        except Exception:
            # parquet engine missing → degrade to CSV
            df.to_csv(path.replace(".parquet", ".csv"), index=False)

    def _flush_key(self, key: tuple) -> None:
        kind, symbol = key
        buf = self._buffers.get(key) or []
        if not buf:
            return
        # group by day to write into the right partition
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for r in buf:
            by_day.setdefault(_day(int(r["ts_ms"])), []).append(r)
        for day, rows in by_day.items():
            df = pd.DataFrame(rows)
            self._write(self._path(kind, symbol, int(rows[0]["ts_ms"])), df)
        self._buffers[key] = []

    def flush(self) -> None:
        with self._lock:
            for key in list(self._buffers.keys()):
                self._flush_key(key)

    # -- query --------------------------------------------------------------
    def query(
        self,
        symbol: str,
        *,
        kind: str = TRADE,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read back records for a symbol/kind within [start_ms, end_ms]."""
        self.flush()
        d = os.path.join(self.root, kind, str(symbol))
        if not os.path.isdir(d):
            return pd.DataFrame()
        frames = []
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            try:
                df = pd.read_parquet(p) if fn.endswith(".parquet") else pd.read_csv(p)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).sort_values("ts_ms")
        if start_ms is not None:
            out = out[out["ts_ms"] >= int(start_ms)]
        if end_ms is not None:
            out = out[out["ts_ms"] <= int(end_ms)]
        return out.reset_index(drop=True)

    def record_from_stream(self, symbol: str, stream, *, max_events: int = 0) -> int:
        """Drain an adapter tick stream (iterable of dict/obj with ts/price/size) into
        the store. Returns events recorded. Bounded by ``max_events`` (0 = until end)."""
        n = 0
        for ev in stream:
            ts = int(
                getattr(ev, "ts_ms", None) or (ev.get("ts_ms") if isinstance(ev, dict) else 0) or 0
            )
            px = getattr(ev, "price", None) or (ev.get("price") if isinstance(ev, dict) else None)
            sz = getattr(ev, "size", None) or (ev.get("size") if isinstance(ev, dict) else 0.0)
            if px is not None:
                self.record_trade(symbol, ts, float(px), float(sz or 0.0))
                n += 1
            if max_events and n >= max_events:
                break
        self.flush()
        return n


__all__ = ["TickStore", "TRADE", "QUOTE", "DEPTH"]
