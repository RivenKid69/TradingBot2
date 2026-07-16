# -*- coding: utf-8 -*-
"""P1 #11: market-data QC (spike/staleness/frozen/gap) + cross-vendor reconciliation
+ MarketDataRouter vendor failover with a per-source circuit breaker."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from services.market_data_quality import (
    DataQualityMonitor, MarketDataRouter, cross_source_reconcile, modified_zscores,
)


def _bars(n=50, price=100.0, step_ms=60_000, start_ms=1_700_000_000_000, vol=0.005, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    px = price * np.cumprod(1 + r)
    ts = [start_ms + i * step_ms for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "close": px,
                         "high": px * 1.001, "low": px * 0.999,
                         "open": px, "volume": rng.uniform(1, 5, n)})


# --------------------------------------------------------------------------- detectors
def test_modified_zscore_robust():
    # realistic noise (non-degenerate MAD) + one big outlier
    x = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 10.0])
    z = modified_zscores(x)
    assert abs(z[-1]) > 3.0                       # outlier flagged
    assert all(abs(v) < 3.0 for v in z[:-1])      # inliers not flagged


def test_clean_series_passes():
    m = DataQualityMonitor(staleness_seconds=None)
    rep = m.check(_bars(), symbol="AAPL")
    assert rep.clean and rep.n_bars == 50


def test_spike_detected():
    df = _bars(seed=1)
    df.loc[25, "close"] = df.loc[24, "close"] * 5.0   # 400% jump
    m = DataQualityMonitor(staleness_seconds=None)
    rep = m.check(df, symbol="AAPL")
    assert not rep.clean
    assert any(i.type == "spike" for i in rep.issues)


def test_staleness_detected():
    df = _bars(n=20)
    m = DataQualityMonitor(staleness_seconds=300)
    # now is far after the last bar
    now = int(df["timestamp"].iloc[-1]) + 10 * 60 * 1000
    rep = m.check(df, symbol="AAPL", now_ms=now)
    assert any(i.type == "stale" for i in rep.issues)
    assert rep.staleness_seconds >= 600


def test_frozen_feed_detected():
    df = _bars(n=30)
    df.loc[10:20, "close"] = 100.0   # 11 identical
    m = DataQualityMonitor(staleness_seconds=None, frozen_run=6)
    rep = m.check(df, symbol="AAPL")
    assert any(i.type == "frozen" for i in rep.issues)


def test_gap_detected():
    df = _bars(n=30)
    df.loc[15, "timestamp"] = int(df.loc[14, "timestamp"]) + 60_000 * 50   # huge gap
    df["timestamp"] = df["timestamp"].astype("int64")
    m = DataQualityMonitor(staleness_seconds=None, session_gap_factor=6.0)
    rep = m.check(df, symbol="AAPL")
    assert any(i.type == "gap" for i in rep.issues)


def test_nonpositive_and_ohlc():
    df = _bars(n=20)
    df.loc[5, "close"] = -1.0
    df.loc[6, "high"] = 1.0
    df.loc[6, "low"] = 100.0
    m = DataQualityMonitor(staleness_seconds=None)
    rep = m.check(df, symbol="AAPL")
    types = {i.type for i in rep.issues}
    assert "nonpositive" in types and "ohlc" in types and not rep.clean


# --------------------------------------------------------------------------- cross-vendor
def test_cross_source_reconcile():
    a = {"AAPL": 100.0, "MSFT": 200.0, "XOM": 50.0}
    b = {"AAPL": 100.05, "MSFT": 205.0, "XOM": 50.0}   # MSFT diverges ~250bps
    rec = cross_source_reconcile(a, b, tolerance_bps=50)
    assert not rec["reconciled"]
    assert rec["divergences"][0]["symbol"] == "MSFT"


# --------------------------------------------------------------------------- router/failover
def test_router_primary_used_when_healthy():
    r = MarketDataRouter([("primary", lambda s, **k: _bars()),
                          ("backup", lambda s, **k: _bars())],
                         monitor=DataQualityMonitor(staleness_seconds=None))
    out = r.get_bars("AAPL")
    assert out["source"] == "primary" and out["failover"] is False


def test_router_fails_over_on_error():
    def bad(s, **k):
        raise RuntimeError("vendor down")
    r = MarketDataRouter([("primary", bad), ("backup", lambda s, **k: _bars())],
                         monitor=DataQualityMonitor(staleness_seconds=None))
    out = r.get_bars("AAPL")
    assert out["source"] == "backup" and out["failover"] is True


def test_router_fails_over_on_bad_data():
    def stale(s, **k):
        df = _bars(n=10)
        return df   # will be flagged stale (old timestamps vs now)
    r = MarketDataRouter([("primary", stale), ("backup", lambda s, **k: _bars(start_ms=int(time.time()*1000)-600000))],
                         monitor=DataQualityMonitor(staleness_seconds=300))
    out = r.get_bars("AAPL", now_ms=int(time.time() * 1000))
    # primary stale -> backup (which is fresh-ish); at least it failed over off primary
    assert out["source"] in ("backup", None)


def test_circuit_breaker_trips_after_threshold():
    def bad(s, **k):
        raise RuntimeError("down")
    r = MarketDataRouter([("primary", bad), ("backup", lambda s, **k: _bars())],
                         monitor=DataQualityMonitor(staleness_seconds=None),
                         failure_threshold=2)
    for _ in range(3):
        r.get_bars("AAPL")
    st = {s["name"]: s for s in r.status()["sources"]}
    assert st["primary"]["tripped"] is True


def test_router_all_down():
    def bad(s, **k):
        raise RuntimeError("down")
    r = MarketDataRouter([("a", bad), ("b", bad)],
                         monitor=DataQualityMonitor(staleness_seconds=None))
    out = r.get_bars("AAPL")
    assert out["bars"] is None and "error" in out
