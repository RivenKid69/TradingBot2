# -*- coding: utf-8 -*-
"""Pro-mode honesty closure (Pro audit) — no element pretends to work.

Locks in the Pro audit fixes: real backend bugs fixed, unflagged fixtures now
flagged, and the Pro frontend no longer shows unlabeled fakes or false-success
toasts (demos self-label; no-ops say DEMO/not-persisted).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-pro-honesty")
os.environ.setdefault("RIVEN_ENABLE_CCEA", "0")

from fastapi.testclient import TestClient

import app as app_module
from app import api

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})
HTML = (Path(__file__).resolve().parents[1] / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------------------ backend real bugs

def test_clock_drift_is_real_not_fabricated_constant():
    d = client.get("/api/compliance/clock/status").json()
    # the fake constant 12.4 (from the offset_ns-never-existed bug) must be gone
    assert d["drift_microseconds"] != 12.4
    assert "drift_measured" in d and d["data_source"] in ("real_ntp", "unavailable")


def test_killswitch_has_no_fabricated_cancel_count():
    k = client.post("/api/compliance/killswitch/trigger", json={"scope": "ALL"}).json()
    assert "cancelled_orders_count" not in k       # was hardcoded 8
    assert k.get("tripped") is True


def test_gdpr_download_route_serves_real_zip():
    exp = client.post("/api/gdpr/export", json={"client_id": "u1"}).json()
    assert exp["download_url"].startswith("/api/gdpr/download?request_id=")
    dl = client.get(exp["download_url"])
    assert dl.status_code == 200 and "zip" in dl.headers.get("content-type", "")


def test_adapters_status_reports_registration_not_fake_auth():
    rows = client.get("/api/adapters/status").json()
    assert all(r["status"] in ("REGISTERED", "NOT_REGISTERED") for r in rows)
    assert all(r["ping_ms"] is None for r in rows)   # no fabricated ping
    tc = client.post("/api/adapters/test_connection", json={"vendor": "binance"}).json()
    assert tc["status"] == "registered" and tc.get("simulated") is True


def test_fixture_endpoints_now_flagged_demo():
    for path in ("/api/treasury/balances", "/api/post_trade/clearing_status"):
        assert client.get(path).json().get("simulated") is True
    assert client.post("/api/forex/reconcile").json().get("simulated") is True


def test_config_endpoints_persist_for_real():
    # Real persistence (was an echo): the endpoint reports persisted_to and the
    # YAML actually appears on disk. Clean up the runtime config afterwards.
    r = client.post("/api/execution/algo_config",
                    json={"algorithm": "TWAP", "max_participation": 10, "window": 30, "offset": 2}).json()
    assert r.get("persisted_to")
    p = Path(r["persisted_to"])
    try:
        assert p.exists() and "TWAP" in p.read_text(encoding="utf-8")
    finally:
        p.unlink(missing_ok=True)


# ------------------------------------------------------------ frontend honesty

def test_no_false_success_toasts_remain():
    for lie in (
        "Locate approved for ${symbol}. Borrow secured.",
        "Master Key перегенерирован и сохранен",
        "Conformance тесты пройдены успешно.",
        "Ликвидация ESM6 и покупка ESU6 завершена успешно",
        "через notebook_service.py'",
        'Биржевые фильтры обновлены."',
    ):
        assert lie not in HTML, f"false-success string still present: {lie}"


def test_unlabeled_client_fakes_now_self_label():
    # each previously-unbadged fake renderer now calls showSimBadge
    for marker in (
        "иллюстративный реестр",            # renderProModelRegistry
        "иллюстративные прогоны",           # renderProModelExperiments
        "иллюстративные факторные беты",    # renderProRiskFactor
        "иллюстративный стресс-сценарий",   # renderProRiskScenario
        "иллюстративные торговые алерты",   # renderProDashAlerts
        "иллюстративные фильтры Binance",   # refreshProExchangeFilters
    ):
        assert marker in HTML, f"missing DEMO self-label: {marker}"


def test_position_sync_verdict_is_honest():
    assert "не сверено" in HTML
    # the always-green SYNCED/OK reconciliation verdict is gone from those rows
    assert HTML.count(">\n                                                    SYNCED") == 0


def test_backend_honesty_flags_surfaced_in_frontend():
    # OMS portfolio + compliance attestations now surface the backend flag
    assert "flagFromPayload(data, 'Портфель — paper/simulated" in HTML
    assert "mock-executor" in HTML or "mock_executor" in HTML          # conformance disclaimer
    assert "DEMO чек-лист" in HTML                                      # AI-Act disclaimer


def test_no_op_buttons_relabeled_demo():
    # a representative set of the relabeled no-op buttons
    assert "rate-лимиты" in HTML and "НЕ применены" in HTML
    assert "PDF не создаётся" in HTML                                   # tearsheet export
    assert "промо" not in HTML.lower() or "DEMO" in HTML               # sanity
