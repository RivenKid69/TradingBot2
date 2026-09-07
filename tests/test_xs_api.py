# -*- coding: utf-8 -*-
"""
Stage A12 tests — config schema, /api/xs/* router (изолированно), CLI smoke.

Роутер тестируется на свежем FastAPI (без импорта тяжёлого app.py).
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from service_xs_pipeline import XSConfig, run_backtest
from xs_api import make_xs_router

TEMPLATE = os.path.join("configs", "config_xs_template.yaml")


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(make_xs_router())
    return TestClient(app)


_SMALL_CFG = {
    "data": {
        "source": "synthetic",
        "symbols": list("ABCDEF"),
        "synthetic_bars": 60,
        "synthetic_seed": 1,
    },
    "universe": {"symbols": list("ABCDEF")},
    "signals": [
        {"name": "mom", "kind": "momentum", "lookback": 5, "skip": 0, "transforms": ["zscore"]}
    ],
    "alpha": {"method": "equal_weight"},
    "risk": {"type": "stat", "method": "ledoit_wolf"},
    "optimizer": {
        "objective": "mean_variance",
        "gross_max": 1.0,
        "net_target": 0.0,
        "max_position": 0.4,
    },
    "backtest": {"cov_lookback": 10, "min_cov_obs": 5, "cost_bps": 5.0, "periods_per_year": 365},
    "n_trials": 2,
}


# ---------------------------------------------------------------------------
# config schema
# ---------------------------------------------------------------------------
def test_config_validates():
    cfg = XSConfig.model_validate(_SMALL_CFG)
    assert cfg.mode == "cross_sectional"
    assert len(cfg.signals) == 1


def test_template_config_parses_and_runs():
    import yaml

    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = XSConfig.model_validate(data)
    out = run_backtest(cfg)
    assert "deflated_sharpe" in out["trust_report"]
    assert out["n_rebalances"] > 0


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
def test_endpoint_config(client):
    r = client.post("/api/xs/config", json=_SMALL_CFG)
    assert r.status_code == 200 and r.json()["valid"] is True
    bad = client.post("/api/xs/config", json={"signals": "notalist"})
    assert bad.status_code == 422


def test_endpoint_optimize(client):
    r = client.post(
        "/api/xs/optimize",
        json={
            "mu": {"A": 1, "B": 2, "C": 3},
            "objective": "max_sharpe",
            "constraints": {"net_target": 1.0},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["weights"]) == {"A", "B", "C"}
    assert body["net"] == pytest.approx(1.0, abs=1e-6)


def test_endpoint_trust_report(client):
    import numpy as np

    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 200).tolist()
    r = client.post("/api/xs/trust_report", json={"returns": rets, "n_trials": 100})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["deflated_sharpe"] <= 1.0
    assert "verdict" in body


def test_endpoint_universe(client):
    r = client.post("/api/xs/universe", json={"universe": {"symbols": ["A", "B"]}, "asof_ms": 0})
    assert r.status_code == 200
    assert r.json()["constituents"] == ["A", "B"]


def test_endpoint_backtest(client):
    r = client.post("/api/xs/backtest", json=_SMALL_CFG)
    assert r.status_code == 200
    body = r.json()
    assert "trust_report" in body and "summary" in body
    assert body["n_rebalances"] > 0


def test_endpoint_live_rebalance(client):
    r = client.post(
        "/api/xs/live/rebalance",
        json={
            "target_weights": {"A": 0.5, "B": -0.5},
            "equity": 100.0,
            "limits": {"gross_max": 1.0},
            "ts_ms": 1,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is True
    assert len(body["batch"]["intents"]) == 2
    # CCEA: интент не содержит order-полей
    assert set(body["batch"]["intents"][0]) == {"symbol", "target_weight", "target_notional"}


def test_endpoint_weights(client):
    r = client.post("/api/xs/weights", json=_SMALL_CFG)
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body and body["n_names"] >= 1
    assert abs(body["gross"]) >= 0


def test_pro_integration_in_index_html():
    """Cross-sectional встроен в Pro-режим как native под-вкладки (не отдельно)."""
    with open("index.html", "r", encoding="utf-8") as fh:
        html = fh.read()
    # native под-вкладки в нужных Pro-модулях
    for panel in (
        "pro-model-panel-xsec",
        "pro-research-panel-xsec",
        "pro-backtest-panel-xsec",
        "pro-risk-panel-xsec",
        "pro-oms-panel-xsec",
    ):
        assert f'id="{panel}"' in html
    for tab in (
        "tab-pro-model-xsec",
        "tab-pro-research-xsec",
        "tab-pro-backtest-xsec",
        "tab-pro-risk-xsec",
        "tab-pro-oms-xsec",
    ):
        assert f'id="{tab}"' in html
    # каждый switcher знает про новую вкладку 'xsec'
    assert html.count("'xsec'") >= 10  # 5 массивов + 5 onclick
    # общий драйвер + ключевые действия подключены к /api/xs/*
    for fn in ("xsOptimize", "xsSignals", "xsBacktest", "xsPreTrade", "xsRebalance"):
        assert fn in html
    assert "/api/xs/weights" in html and "/api/xs/backtest" in html
    # отдельной консоли больше нет
    assert "/api/xs/ui" not in html


# ---------------------------------------------------------------------------
# CLI smoke (acceptance)
# ---------------------------------------------------------------------------
def test_cli_backtest_smoke():
    import script_xs_backtest as cli

    rc = cli.main(["--config", TEMPLATE])
    assert rc == 0
