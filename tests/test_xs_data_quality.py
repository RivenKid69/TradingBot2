# -*- coding: utf-8 -*-
"""
Stage D7 tests — Data-Trust gate (service_data_quality + интеграция).

  * pit_leak_scan: ловит look-ahead (значение раньше публикации); чистый as-of проходит
  * signal_columns: lineage сигнал → колонки
  * data_trust_report: verdict trusted/caution/untrusted; PIT-violations при none-колонке
  * интеграция: run_backtest несёт data_trust; API /api/xs/data_trust
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL
from core_xs_data import ColumnProvenance, PIT_NONE, PIT_APPROX, PIT_TRUE
from impl_panel import PanelBuilder
from service_signals import SignalLibrary, ColumnSignal, MomentumSignal
from service_data_quality import pit_leak_scan, signal_columns, data_trust_report

T0, STEP = 1_700_000_000, 86_400


def _panel(n=8, syms=("AAA",), extra=None):
    frames = {}
    for s in syms:
        ts = [T0 + i * STEP for i in range(n)]
        d = {"timestamp": ts, "symbol": s, "close": 100.0 + np.arange(n)}
        if extra:
            for col, vals in extra.items():
                d[col] = vals
        frames[s] = pd.DataFrame(d)
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# pit_leak_scan
# ---------------------------------------------------------------------------
def test_pit_leak_scan_clean():
    # значение появляется на баре 4 (как при честном as-of, publish на баре 4)
    iv = [np.nan] * 4 + [0.2] * 4
    panel = _panel(8, extra={"iv": iv})
    long = pd.DataFrame({"publish_ts": [(T0 + 4 * STEP) * 1000], "symbol": ["AAA"], "iv": [0.2]})
    leaks = pit_leak_scan(panel, long, value_col="iv")
    assert leaks == []                                  # чисто


def test_pit_leak_scan_catches_lookahead():
    # значение присутствует с бара 0, а публикация только на баре 4 → УТЕЧКА
    iv = [0.2] * 8
    panel = _panel(8, extra={"iv": iv})
    long = pd.DataFrame({"publish_ts": [(T0 + 4 * STEP) * 1000], "symbol": ["AAA"], "iv": [0.2]})
    leaks = pit_leak_scan(panel, long, value_col="iv")
    assert len(leaks) == 1 and "look-ahead" in leaks[0]["reason"]


# ---------------------------------------------------------------------------
# signal_columns lineage
# ---------------------------------------------------------------------------
def test_signal_columns_lineage():
    panel = _panel(5, extra={"funding_rate": -0.001})
    from signals.crypto_signals import FundingCarry
    assert signal_columns(FundingCarry("fc"), panel) == ["funding_rate"]
    assert signal_columns(MomentumSignal("m", price_col="close"), panel) == ["close"]


# ---------------------------------------------------------------------------
# data_trust_report verdict
# ---------------------------------------------------------------------------
def test_trust_trusted_when_all_pit_true():
    panel = _panel(6, extra={"funding_rate": -0.001})
    prov = [ColumnProvenance("close", "binance", "binance", PIT_TRUE),
            ColumnProvenance("funding_rate", "binance:funding", "binance", PIT_TRUE)]
    from signals.crypto_signals import FundingCarry
    lib = SignalLibrary(); lib.register(FundingCarry("fc"))
    rep = data_trust_report(panel, prov, signal_library=lib)
    assert rep["trust_verdict"] == "trusted"
    assert rep["pit_violations"] == []
    assert rep["signal_lineage"]["fc"]["backtest_safe"] is True


def test_trust_untrusted_when_signal_uses_none_column():
    panel = _panel(6, extra={"iv": 0.2, "realized_vol": 0.1})
    prov = [ColumnProvenance("close", "p", "yahoo", PIT_TRUE),
            ColumnProvenance("iv", "yfinance:chain", "yfinance", PIT_NONE),   # снимок!
            ColumnProvenance("realized_vol", "computed", "computed", PIT_TRUE)]
    from signals.options_signals import VolRiskPremium
    lib = SignalLibrary(); lib.register(VolRiskPremium("vrp"))
    rep = data_trust_report(panel, prov, signal_library=lib)
    assert rep["trust_verdict"] == "untrusted"          # VRP читает iv (none)
    assert "vrp" in rep["pit_violations"]
    assert rep["signal_lineage"]["vrp"]["backtest_safe"] is False


def test_trust_caution_when_approx():
    panel = _panel(6, extra={"mcap": 1000.0})
    prov = [ColumnProvenance("close", "p", "binance", PIT_TRUE),
            ColumnProvenance("mcap", "static:mcap", "static", PIT_APPROX)]
    from signals.crypto_signals import Size
    lib = SignalLibrary(); lib.register(Size("sz"))
    rep = data_trust_report(panel, prov, signal_library=lib)
    assert rep["trust_verdict"] == "caution"            # mcap=approx


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------
def test_run_backtest_carries_data_trust():
    from service_xs_pipeline import XSConfig, run_backtest
    cfg = XSConfig.model_validate({
        "asset_class": "crypto",
        "data": {"source": "synthetic", "symbols": ["BTC", "ETH", "SOL"], "synthetic_bars": 60},
        "signals": [{"name": "mom", "kind": "crypto_momentum", "lookback": 20, "skip": 1}],
        "backtest": {"rebalance_every": 5, "cov_lookback": 20, "min_cov_obs": 10},
    })
    out = run_backtest(cfg)
    assert "data_trust" in out and out["data_trust"] is not None
    # синтетика → close pit=none → momentum зависит → untrusted (honest)
    assert out["data_trust"]["trust_verdict"] == "untrusted"
    assert "mom" in out["data_trust"]["signal_lineage"]


def test_api_data_trust():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xs_api import register_xs_routes
    app = FastAPI(); register_xs_routes(app)
    client = TestClient(app)
    r = client.post("/api/xs/data_trust", json={
        "asset_class": "crypto",
        "data": {"source": "synthetic", "symbols": ["BTC", "ETH"]},
        "signals": [{"name": "mom", "kind": "crypto_momentum", "lookback": 20}],
    })
    assert r.status_code == 200
    data = r.json()
    assert "trust_verdict" in data and "signal_lineage" in data and "pit_violations" in data
