# -*- coding: utf-8 -*-
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-pro-risk")

import app as app_module
from app import api

# The global auth middleware only whitelists loopback peers; the TestClient
# peer is "testclient", so authenticate explicitly with the API token.
client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

def test_get_risk_summary():
    response = client.get("/api/risk/summary")
    assert response.status_code == 200
    data = response.json()
    assert "kill_switch_active" in data
    assert "leak_guard" in data
    assert "alerts" in data
    assert "compliance_clock" in data
    assert data["compliance_clock"]["status"] == "SYNCHRONIZED"

def test_pdt_check_no_restriction():
    payload = {
        "position_value": 100000.0,
        "account_equity": 30000.0,
        "day_trades": 2
    }
    response = client.post("/api/risk/pdt_check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pdt_status" in data
    assert "margin_status" in data
    assert "margin_used" in data
    assert data["pdt_status"] == "OK"

def test_pdt_check_restricted():
    payload = {
        "position_value": 100000.0,
        "account_equity": 20000.0,  # Below Reg T requirement or PDT 25k threshold with day trades
        "day_trades": 5
    }
    response = client.post("/api/risk/pdt_check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pdt_status"] == "BLOCKED"

def test_options_greeks():
    payload = {
        "spot": 180.0,
        "strike": 180.0,
        "dte": 30.0,
        "vol": 0.20,
        "rate": 0.05
    }
    response = client.post("/api/risk/options_greeks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "call_price" in data
    assert "put_price" in data
    assert "delta" in data
    assert "gamma" in data
    assert "vega" in data
    assert "theta" in data
    assert "rho" in data
    assert data["call_price"] > 0
    assert data["put_price"] > 0

def test_futures_span():
    payload = {
        "positions": [
            {"symbol": "ESM6", "qty": 5},
            {"symbol": "NQM6", "qty": 2}
        ]
    }
    response = client.post("/api/risk/futures_span", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "span_requirement" in data
    assert "maintenance_margin" in data
    assert "escalation_status" in data
    # ES: 12400 * 5 = 62000
    # NQ: 18600 * 2 = 37200
    # Total: 99200
    assert data["span_requirement"] == 99200.0

def test_kill_switch_lifecycle():
    # Trip the switch
    payload = {
        "scope": "ALL",
        "reason": "TEST_RUN",
        "active": True
    }
    response = client.post("/api/risk/kill_switch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["kill_switch_active"] is True

    # Check summary
    summary_resp = client.get("/api/risk/summary")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["kill_switch_active"] is True

    # Reset the switch
    payload["active"] = False
    response = client.post("/api/risk/kill_switch", json=payload)
    assert response.status_code == 200
    assert response.json()["kill_switch_active"] is False

    # Check summary again
    summary_resp = client.get("/api/risk/summary")
    assert summary_resp.json()["kill_switch_active"] is False

def test_dynamic_no_trade():
    response = client.get("/api/risk/dynamic_no_trade?symbol=BTCUSDT")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTCUSDT"
    assert "blocked" in data
    assert "metrics" in data

def test_dynamic_no_trade_tune():
    payload = {
        "sigma_window": 42,
        "sigma_upper": 3.0,
        "spread_upper": 90.0,
        "spread_abs_bps": 5.0,
        "hysteresis": 0.1,
        "cooldown_bars": 3
    }
    response = client.post("/api/risk/dynamic_no_trade/tune", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
