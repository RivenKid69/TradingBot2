"""Тесты регулярного XS-ребаланса (service_xs_rebalance + REST-проводка).

Закрывает P1-C (§4.9) из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: путь
«веса → гардрейлы → Intents → CCEA Agent» с журналом решений.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-xs-rebalance")

from fastapi.testclient import TestClient

import app as app_module
from app import api
from service_xs_rebalance import (
    PlannedOrder,
    RebalanceLimits,
    plan_rebalance,
    run_rebalance,
)

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

LIM = RebalanceLimits(max_turnover=0.5, min_trade_notional=10.0,
                      drift_band_bps=10.0, max_position_weight=0.5, max_orders=50)


# ------------------------------------------------------------------- планнер

def test_plan_basic_deltas_and_sells_first():
    plan = plan_rebalance(
        target_weights={"BTC": 0.30, "ETH": 0.10},
        holdings_qty={"BTC": 0.001, "SOL": 10.0},     # SOL вне цели → закрыть
        prices={"BTC": 50_000.0, "ETH": 2_000.0, "SOL": 100.0},
        equity=10_000.0,
        limits=LIM,
    )
    orders = {o.symbol: o for o in plan["orders"]}
    # SOL: цель 0 → продать всё (10 шт · $100 = $1000)
    assert orders["SOL"].qty == pytest.approx(-10.0)
    # BTC: цель $3000, факт $50 → купить $2950 → 0.059 BTC
    assert orders["BTC"].qty == pytest.approx(2950.0 / 50_000.0)
    # ETH: цель $1000, факта нет → купить 0.5 ETH
    assert orders["ETH"].qty == pytest.approx(0.5)
    # Продажи раньше покупок (высвобождают кэш).
    sides = [o.qty < 0 for o in plan["orders"]]
    assert sides == sorted(sides, reverse=True)


def test_plan_no_trade_band_and_min_notional():
    plan = plan_rebalance(
        target_weights={"BTC": 0.1001},               # дрейф 1 bps от факта 0.10
        holdings_qty={"BTC": 0.02},
        prices={"BTC": 50_000.0},
        equity=10_000.0,
        limits=RebalanceLimits(drift_band_bps=25.0, min_trade_notional=10.0),
    )
    assert plan["orders"] == []
    assert any(s["reason"] == "drift_band" for s in plan["skipped"])

    plan2 = plan_rebalance(
        target_weights={"BTC": 0.006},                # дельта $60 > band(25bps=$25), но < min $100
        holdings_qty={},
        prices={"BTC": 50_000.0},
        equity=10_000.0,
        limits=RebalanceLimits(drift_band_bps=25.0, min_trade_notional=100.0),
    )
    assert plan2["orders"] == []
    assert any(s["reason"] == "min_notional" for s in plan2["skipped"])


def test_plan_turnover_cap_scales_proportionally():
    plan = plan_rebalance(
        target_weights={"A": 0.4, "B": 0.4},
        holdings_qty={},
        prices={"A": 10.0, "B": 10.0},
        equity=10_000.0,
        limits=RebalanceLimits(max_turnover=0.2, min_trade_notional=1.0,
                               drift_band_bps=1.0, max_position_weight=0.5),
    )
    # raw turnover 0.8 → scale 0.25; каждый ордер $4000 → $1000.
    assert plan["scale"] == pytest.approx(0.25)
    for o in plan["orders"]:
        assert abs(o.notional) == pytest.approx(1000.0)
    assert plan["turnover_planned"] == pytest.approx(0.2)


def test_plan_clips_position_weight():
    plan = plan_rebalance(
        target_weights={"A": 0.9},
        holdings_qty={},
        prices={"A": 10.0},
        equity=10_000.0,
        limits=RebalanceLimits(max_position_weight=0.2, min_trade_notional=1.0, drift_band_bps=1.0),
    )
    assert plan["clipped"] and plan["clipped"][0]["clipped_to"] == pytest.approx(0.2)
    assert plan["orders"][0].notional == pytest.approx(2000.0)


def test_plan_no_price_position_untouched():
    plan = plan_rebalance(
        target_weights={"A": 0.2},
        holdings_qty={"GHOST": 5.0},
        prices={"A": 10.0},                           # у GHOST цены нет
        equity=10_000.0,
        limits=LIM,
    )
    assert all(o.symbol != "GHOST" for o in plan["orders"])   # вслепую не торгуем
    assert any(s["symbol"] == "GHOST" and s["reason"] == "no_price" for s in plan["skipped"])


def test_plan_zero_crossing_split_into_two_legs():
    plan = plan_rebalance(
        target_weights={"A": -0.2},                    # из лонга в шорт
        holdings_qty={"A": 100.0},
        prices={"A": 10.0},
        equity=10_000.0,
        limits=RebalanceLimits(max_turnover=2.0, min_trade_notional=1.0,
                               drift_band_bps=1.0, max_position_weight=0.5),
    )
    kinds = [o.kind for o in plan["orders"]]
    assert kinds == ["close_leg", "open_leg"]
    close, open_ = plan["orders"]
    assert close.qty == pytest.approx(-100.0)          # закрыть лонг целиком
    assert open_.qty == pytest.approx(-200.0)          # открыть шорт на -$2000


def test_plan_max_orders_cap_journals_dropped():
    plan = plan_rebalance(
        target_weights={f"S{i}": 0.05 for i in range(10)},
        holdings_qty={},
        prices={f"S{i}": 10.0 for i in range(10)},
        equity=10_000.0,
        limits=RebalanceLimits(max_orders=3, min_trade_notional=1.0,
                               drift_band_bps=1.0, max_turnover=5.0),
    )
    assert len(plan["orders"]) == 3
    assert len(plan["dropped_by_max_orders"]) == 7


# ------------------------------------------------------------------- runner

class FakeSupervisor:
    def __init__(self, *, simulated=True, equity=10_000.0, holdings=None, fail_symbols=()):
        self.simulated = simulated
        self.equity = equity
        self.holdings = holdings or []
        self.fail_symbols = set(fail_symbols)
        self.submitted = []

    def portfolio_snapshot(self):
        return {"ok": True, "holdings": self.holdings,
                "metrics": {"net_liquidation_value": self.equity},
                "simulated": self.simulated,
                "data_source": "paper_broker" if self.simulated else "live_broker"}

    def submit_rebalance_order(self, symbol, qty, price, **kw):
        self.submitted.append((symbol, qty, price))
        if symbol in self.fail_symbols:
            return {"ok": False, "error": "rejected by firewall"}
        return {"ok": True, "client_order_id": f"oid-{symbol}", "error": None}


@pytest.fixture()
def xs_env(tmp_path, monkeypatch):
    """Мини-окружение: рабочая директория tmp, фиктивный пайплайн весов."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "xs.yaml"
    cfg.write_text("name: test\n", encoding="utf-8")

    import pandas as pd

    def fake_load_config_dict(data):
        class _RL:  # noqa: N801
            checkpoint = None
        class _Cfg:  # noqa: N801
            rl = _RL()
        return _Cfg()

    def fake_load_panel(cfg_obj):
        idx = pd.MultiIndex.from_product(
            [[1_700_000_000_000, 1_700_000_060_000], ["BTC", "ETH"]],
            names=["ts_ms", "symbol"])
        return pd.DataFrame({"close": [49_000.0, 1_900.0, 50_000.0, 2_000.0]}, index=idx)

    def fake_weights(cfg_obj, panel=None):
        return pd.Series({"BTC": 0.30, "ETH": 0.10})

    import service_xs_pipeline as xsp
    monkeypatch.setattr(xsp, "load_config_dict", fake_load_config_dict)
    monkeypatch.setattr(xsp, "load_panel", fake_load_panel)
    monkeypatch.setattr(xsp, "latest_target_weights", fake_weights)
    return cfg


def test_runner_executes_plan_and_journals(xs_env, tmp_path):
    sup = FakeSupervisor()
    rec = run_rebalance(str(xs_env), sup, limits=LIM, out_dir=str(tmp_path / "journal"))
    assert rec["status"] == "ok"
    assert len(sup.submitted) == 2                     # BTC + ETH покупки
    # Журнал решений записан и содержит полный план + результаты.
    last = json.loads((tmp_path / "journal" / "last.json").read_text(encoding="utf-8"))
    assert last["status"] == "ok"
    assert last["weights"] == {"BTC": 0.3, "ETH": 0.1}
    assert all(e["ok"] for e in last["executions"])
    assert last["equity"] == 10_000.0


def test_runner_dry_run_sends_nothing(xs_env, tmp_path):
    sup = FakeSupervisor()
    rec = run_rebalance(str(xs_env), sup, dry_run=True, limits=LIM,
                        out_dir=str(tmp_path / "journal"))
    assert rec["status"] == "dry_run"
    assert sup.submitted == []                         # ни одного ордера
    assert rec["plan"]["orders"]                       # но план построен и записан


def test_runner_blocked_by_kill_switch(xs_env, tmp_path, monkeypatch):
    import services.ops_kill_switch as oks
    monkeypatch.setattr(oks, "tripped", lambda: True)
    sup = FakeSupervisor()
    rec = run_rebalance(str(xs_env), sup, limits=LIM, out_dir=str(tmp_path / "j"))
    assert rec["status"] == "blocked" and "kill switch" in rec["reason"]
    assert sup.submitted == []


def test_runner_paper_only_blocks_live_broker(xs_env, tmp_path):
    sup = FakeSupervisor(simulated=False)              # live-брокер
    rec = run_rebalance(str(xs_env), sup, paper_only=True, limits=LIM,
                        out_dir=str(tmp_path / "j"))
    assert rec["status"] == "blocked" and "live" in rec["reason"]
    assert sup.submitted == []


def test_runner_partial_when_oms_rejects_some(xs_env, tmp_path):
    sup = FakeSupervisor(fail_symbols={"ETH"})
    rec = run_rebalance(str(xs_env), sup, limits=LIM, out_dir=str(tmp_path / "j"))
    assert rec["status"] == "partial"
    failed = [e for e in rec["executions"] if not e["ok"]]
    assert failed and failed[0]["symbol"] == "ETH" and "firewall" in failed[0]["error"]


def test_runner_noop_when_inside_bands(xs_env, tmp_path):
    # Портфель уже соответствует цели → все дельты в бэнде.
    sup = FakeSupervisor(holdings=[
        {"symbol": "BTC", "qty": 3000.0 / 50_000.0},
        {"symbol": "ETH", "qty": 1000.0 / 2_000.0},
    ])
    rec = run_rebalance(str(xs_env), sup, limits=LIM, out_dir=str(tmp_path / "j"))
    assert rec["status"] == "noop"
    assert sup.submitted == []


def test_runner_blocked_without_supervisor(xs_env, tmp_path):
    rec = run_rebalance(str(xs_env), None, limits=LIM, out_dir=str(tmp_path / "j"))
    assert rec["status"] == "blocked"


def test_runner_signature_gate_blocks_rl_checkpoint(xs_env, tmp_path, monkeypatch):
    """RL-чекпоинт в конфиге + enforce ⇒ незарегистрированная модель блокирует ребаланс."""
    import service_xs_pipeline as xsp

    ckpt = tmp_path / "rogue.zip"
    ckpt.write_bytes(b"unsigned")

    def cfg_with_rl(data):
        class _RL:  # noqa: N801
            checkpoint = str(ckpt)
        class _Cfg:  # noqa: N801
            rl = _RL()
        return _Cfg()

    monkeypatch.setattr(xsp, "load_config_dict", cfg_with_rl)
    monkeypatch.setenv("RIVEN_MODEL_SIGNATURE_POLICY", "enforce")
    sup = FakeSupervisor()
    rec = run_rebalance(str(xs_env), sup, limits=LIM, out_dir=str(tmp_path / "j"))
    assert rec["status"] == "blocked" and "подпись" in rec["reason"]
    assert sup.submitted == []
    assert rec["signature"] and rec["signature"]["ok"] is False


# ------------------------------------------------------------------- REST

def test_api_run_requires_confirmation_for_real_send(monkeypatch):
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", object(), raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "running", raising=False)
    res = client.post("/api/xs/rebalance/run",
                      json={"config": "configs/xs.yaml", "dry_run": False})
    assert res.status_code == 409
    assert "confirm_trading" in res.json()["detail"]


def test_api_run_503_without_ccea(monkeypatch):
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "stopped", raising=False)
    res = client.post("/api/xs/rebalance/run", json={"config": "x.yaml", "dry_run": True})
    assert res.status_code == 503


def test_api_last_404_then_payload(tmp_path, monkeypatch):
    import service_xs_rebalance as sxr
    monkeypatch.setattr(sxr, "load_last_record", lambda: None)
    assert client.get("/api/xs/rebalance/last").status_code == 404
    monkeypatch.setattr(sxr, "load_last_record", lambda: {"status": "ok"})
    assert client.get("/api/xs/rebalance/last").json() == {"status": "ok"}


def test_api_models_verify_endpoint_returns_verdict(tmp_path):
    rogue = tmp_path / "m.zip"
    rogue.write_bytes(b"x")
    res = client.get("/api/models/verify_for_live", params={"path": str(rogue)})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False and body["registered"] is False
    assert body["effective_live_policy"] == "enforce"


# ------------------------------------------------------- планировщик-действие

def test_scheduler_action_maps_statuses(monkeypatch, tmp_path):
    import service_xs_rebalance as sxr
    from services.scheduler import ScheduledJob, STATUS_SKIPPED, STATUS_SUCCEEDED

    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", object(), raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "running", raising=False)
    monkeypatch.setattr(sxr, "run_rebalance",
                        lambda *a, **k: {"status": "ok", "reason": "исполнено 2/2"})
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(id="x", title="x", action="trade.xs_rebalance",
                       daily_utc="13:45", trading_impacting=True,
                       params={"config": "configs/whatever.yaml"})
    res = actions["trade.xs_rebalance"](job)
    assert res.status == STATUS_SUCCEEDED and "исполнено 2/2" in res.detail

    # Без конфига — честный skip ещё до запуска.
    job2 = ScheduledJob(id="x2", title="x", action="trade.xs_rebalance",
                        daily_utc="13:45", trading_impacting=True, params={})
    assert actions["trade.xs_rebalance"](job2).status == STATUS_SKIPPED
