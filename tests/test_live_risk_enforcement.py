"""Тесты enforcement риск-лимитов в реальном торговом контуре (P0-B).

Закрывает §3.6: форма лимитов сохраняла daily loss / max DD / leverage, но
никто их не применял. Теперь:
  * pre-trade: RiskChecker с leverage/drawdown/daily-loss из lite_limits;
  * intra-day: LiveRiskMonitor — circuit breaker при пробое дневного убытка/
    просадки → auto-halt.
"""

from __future__ import annotations

import os
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-risk-enforce")
os.environ.setdefault("RIVEN_ENABLE_CCEA", "1")

from fastapi.testclient import TestClient

import app as app_module
from app import api
from packages.agent.policy.risk_checker import PortfolioState, RiskChecker
from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent
from services.live_risk_limits import (
    BREACH_DAILY_LOSS,
    BREACH_DRAWDOWN,
    LiveRiskLimits,
    LiveRiskMonitor,
    build_risk_checker,
    load_live_risk_limits,
)

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})


def _entry(symbol="BTCUSDT", qty="1", side=IntentSide.LONG, itype=IntentType.MARKET_ENTRY):
    return OrderIntent(strategy_id="t", symbol=symbol, intent_type=itype, side=side,
                       target_quantity=Decimal(qty), reason="test")


# --------------------------------------------------- RiskChecker pre-trade

def test_leverage_cap_blocks_order():
    rc = RiskChecker(max_leverage=Decimal("2.0"))
    # equity 10k, уже gross 15k → плечо 1.5x; ордер +10k notional → 2.5x > 2.0
    pf = PortfolioState(equity=Decimal("10000"), gross_exposure=Decimal("15000"))
    res = rc.check(_entry(qty="1"), pf, price=Decimal("10000"))
    assert not res.passed
    assert any(f.check_type.value == "leverage" for f in res.failed_checks)


def test_leverage_cap_allows_within():
    rc = RiskChecker(max_leverage=Decimal("3.0"), max_concentration_pct=Decimal("0.90"))
    pf = PortfolioState(equity=Decimal("10000"), gross_exposure=Decimal("5000"))
    res = rc.check(_entry(qty="0.4"), pf, price=Decimal("10000"))  # +4k → 9k/10k=0.9x
    assert res.passed, [f.message for f in res.failed_checks]


def test_leverage_skipped_for_exit():
    rc = RiskChecker(max_leverage=Decimal("1.0"))
    pf = PortfolioState(equity=Decimal("10000"), gross_exposure=Decimal("50000"))  # 5x
    # close position — уменьшает риск, leverage-check пропускается
    res = rc.check(_entry(itype=IntentType.CLOSE_POSITION, side=IntentSide.SHORT),
                   pf, price=Decimal("10000"))
    lev = [f for f in res.checks if f.check_type.value == "leverage"]
    assert lev and lev[0].passed


def test_drawdown_blocks_new_risk():
    rc = RiskChecker(max_drawdown_pct=Decimal("0.10"))   # 10%
    pf = PortfolioState(equity=Decimal("8500"), peak_equity=Decimal("10000"))  # -15%
    res = rc.check(_entry(qty="0.1"), pf, price=Decimal("10000"))
    assert not res.passed
    assert any(f.check_type.value == "max_drawdown" for f in res.failed_checks)


def test_drawdown_allows_exit_to_recover():
    rc = RiskChecker(max_drawdown_pct=Decimal("0.10"))
    pf = PortfolioState(equity=Decimal("8000"), peak_equity=Decimal("10000"))  # -20%
    res = rc.check(_entry(itype=IntentType.CLOSE_POSITION, side=IntentSide.SHORT),
                   pf, price=Decimal("10000"))
    dd = [f for f in res.checks if f.check_type.value == "max_drawdown"]
    assert dd and dd[0].passed


# --------------------------------------------------- limits loader / builder

def test_load_limits_from_yaml(tmp_path):
    p = tmp_path / "risk.yaml"
    p.write_text(
        "lite_limits:\n"
        "  daily_loss_limit_usd: 500\n"
        "  max_drawdown_pct: 15\n"
        "  max_leverage: 2.0\n"
        "  max_concentration_pct: 25\n",
        encoding="utf-8")
    lim = load_live_risk_limits(str(p))
    assert lim.daily_loss_limit_usd == 500.0 and lim.max_drawdown_pct == 15.0
    assert lim.max_leverage == 2.0 and lim.any_enforced


def test_build_risk_checker_maps_limits():
    lim = LiveRiskLimits(daily_loss_limit_usd=1000, max_leverage=2.0,
                         max_drawdown_pct=15, max_concentration_pct=25)
    rc = build_risk_checker(lim, equity=50_000)
    assert rc.max_daily_loss == Decimal("1000")
    assert rc.max_leverage == Decimal("2.0")
    assert rc.max_drawdown_pct == Decimal("0.15")
    assert rc.max_concentration_pct == Decimal("0.25")


def test_empty_limits_no_enforcement():
    lim = load_live_risk_limits("no/such/file.yaml")
    assert not lim.any_enforced
    rc = build_risk_checker(lim)   # дефолтный RiskChecker, leverage/dd не проверяются
    assert rc.max_leverage is None and rc.max_drawdown_pct is None


# --------------------------------------------------- intra-day monitor

def _limits(**kw):
    base = dict(daily_loss_limit_usd=None, max_drawdown_pct=None, max_leverage=None)
    base.update(kw)
    return lambda: LiveRiskLimits(**base)


def test_monitor_daily_loss_breach_triggers_halt(tmp_path):
    halts = []
    m = LiveRiskMonitor(limits_loader=_limits(daily_loss_limit_usd=500),
                        halt_callback=lambda p: halts.append(p),
                        peak_state_path=str(tmp_path / "peak.json"))
    # день -$300 → armed, не пробит
    st = m.evaluate({"equity": 9700, "day_pnl": -300, "gross_exposure": 0})
    assert st["status"] == "armed" and not st["breaches"] and not halts
    # день -$600 → пробой → halt
    st = m.evaluate({"equity": 9400, "day_pnl": -600, "gross_exposure": 0})
    assert BREACH_DAILY_LOSS in st["breaches"] and st.get("halt_triggered")
    assert len(halts) == 1 and "убыт" in halts[0]["reason"].lower()


def test_monitor_drawdown_breach_triggers_halt(tmp_path):
    halts = []
    m = LiveRiskMonitor(limits_loader=_limits(max_drawdown_pct=10),
                        halt_callback=lambda p: halts.append(p),
                        peak_state_path=str(tmp_path / "peak.json"))
    m.evaluate({"equity": 10000, "day_pnl": 0, "gross_exposure": 0})   # peak=10k
    st = m.evaluate({"equity": 8500, "day_pnl": -1500, "gross_exposure": 0})  # -15% dd
    assert BREACH_DRAWDOWN in st["breaches"] and len(halts) == 1


def test_monitor_idempotent_single_halt(tmp_path):
    halts = []
    m = LiveRiskMonitor(limits_loader=_limits(daily_loss_limit_usd=500),
                        halt_callback=lambda p: halts.append(p),
                        peak_state_path=str(tmp_path / "peak.json"))
    m.evaluate({"equity": 9000, "day_pnl": -600, "gross_exposure": 0})
    m.evaluate({"equity": 8800, "day_pnl": -800, "gross_exposure": 0})
    assert len(halts) == 1   # не триггерит повторно
    m.reset_breach()
    m.evaluate({"equity": 8500, "day_pnl": -900, "gross_exposure": 0})
    assert len(halts) == 2   # после reset снова вооружён


def test_monitor_peak_survives_reload(tmp_path):
    peak_path = str(tmp_path / "peak.json")
    m1 = LiveRiskMonitor(limits_loader=_limits(max_drawdown_pct=10), peak_state_path=peak_path)
    m1.evaluate({"equity": 12000, "day_pnl": 2000, "gross_exposure": 0})   # peak=12k
    m2 = LiveRiskMonitor(limits_loader=_limits(max_drawdown_pct=10), peak_state_path=peak_path)
    st = m2.evaluate({"equity": 10600, "day_pnl": -1400, "gross_exposure": 0})  # -11.7% от 12k
    assert BREACH_DRAWDOWN in st["breaches"]   # peak пережил рестарт


def test_monitor_reset_day():
    m = LiveRiskMonitor(limits_loader=_limits(max_drawdown_pct=10))
    m.evaluate({"equity": 12000, "day_pnl": 2000, "gross_exposure": 0})
    m.reset_day(equity=12000)
    st = m.evaluate({"equity": 11500, "day_pnl": -500, "gross_exposure": 0})  # -4.2% от нового пика
    assert not st["breaches"]


def test_monitor_usage_percentages(tmp_path):
    m = LiveRiskMonitor(limits_loader=_limits(daily_loss_limit_usd=1000, max_leverage=4.0),
                        peak_state_path=str(tmp_path / "peak.json"))
    st = m.evaluate({"equity": 10000, "day_pnl": -400, "gross_exposure": 20000})
    assert st["usage"]["daily_loss_pct"] == pytest.approx(40.0)
    assert st["usage"]["leverage_pct_of_cap"] == pytest.approx(50.0)  # 2x / 4x


# --------------------------------------------------- REST (без супервизора)
# Полная supervisor-интеграция (reload применяет лимиты к Agent, circuit
# breaker → auto-halt на живых fill'ах) проверяется live smoke на реальном
# CCEA-сервере — два реальных control-plane в одном pytest-процессе
# конфликтуют (общий SQLAlchemy state), поэтому supervisor-фикстуру здесь не
# поднимаем; ядро полностью покрыто unit-тестами выше.

def test_api_enforcement_agent_offline(monkeypatch):
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "stopped", raising=False)
    res = client.get("/api/risk/enforcement")
    assert res.status_code == 200 and res.json()["status"] == "agent_offline"
    assert res.json()["enforced"] is False


def test_api_save_limits_offline_persists(monkeypatch, tmp_path):
    risk_yaml = tmp_path / "risk.yaml"
    risk_yaml.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_module, "RISK_LIMITS_CONFIG_PATH", str(risk_yaml))
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_STATE", "stopped", raising=False)
    res = client.post("/api/risk/limits", json={
        "daily_loss_limit_usd": 2000, "max_leverage": 2.0, "max_drawdown_pct": 15,
        "max_concentration_pct": 20})
    assert res.status_code == 200
    body = res.json()
    # Без Agent — сохранено, но не applied live (честно).
    assert body["applied_to_agent"] is False
    import yaml as _yaml
    on_disk = _yaml.safe_load(risk_yaml.read_text(encoding="utf-8"))
    assert on_disk["lite_limits"]["max_leverage"] == 2.0
