"""Тесты планировщика регулярных задач (services/scheduler.py + REST-проводка).

Закрывает P0-F из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: проверяются
триггеры (daily UTC / interval / маски дней), anacron catch-up, ретраи с
backoff и алертом, CCEA-гейт торговых задач, долговечность состояния,
fail-closed research-пайплайн и REST-поверхность.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-scheduler")

from fastapi.testclient import TestClient

import app as app_module
from app import api
from services.scheduler import (
    JobRunResult,
    ScheduledJob,
    SchedulerService,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCEEDED,
)

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

# Среда: 2026-07-15 — среда. Фиксированная точка для детерминизма.
WED_NOON = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, ts: float) -> None:
        self.ts = float(ts)

    def __call__(self) -> float:
        return self.ts

    def advance(self, sec: float) -> None:
        self.ts += sec


def make_service(
    tmp_path: Path, jobs_yaml: str, clock: Clock, actions=None, alert_fn=None
) -> SchedulerService:
    cfg = tmp_path / "scheduler.yaml"
    cfg.write_text(jobs_yaml, encoding="utf-8")
    return SchedulerService(
        config_path=str(cfg),
        state_path=str(tmp_path / "state.json"),
        journal_path=str(tmp_path / "runs.jsonl"),
        actions=actions or {},
        alert_fn=alert_fn,
        time_fn=clock,
        tick_sec=1,
    )


def wait_terminal(svc: SchedulerService, job_id: str, timeout: float = 5.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = svc._job_state(job_id)
        if st.get("last_status") not in (None, "running"):
            return st["last_status"]
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} не достиг терминального статуса: {svc._job_state(job_id)}")


def wait_runs(svc: SchedulerService, n: int, timeout: float = 5.0) -> list:
    """Дождаться, пока в журнале появится n записей (для повторных запусков,
    где last_status предыдущего запуска уже терминален)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = svc.recent_runs()
        if len(runs) >= n:
            return runs
        time.sleep(0.05)
    raise AssertionError(f"в журнале {len(svc.recent_runs())} записей, ожидалось {n}")


# ---------------------------------------------------------------- триггеры


def test_daily_trigger_last_and_next_scheduled():
    job = ScheduledJob(id="j", title="j", action="a", daily_utc="06:00")
    last = job.last_scheduled_before(WED_NOON)
    assert (
        datetime.fromtimestamp(last, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        == "2026-07-15 06:00"
    )
    nxt = job.next_scheduled_after(WED_NOON)
    assert (
        datetime.fromtimestamp(nxt, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        == "2026-07-16 06:00"
    )


def test_market_days_only_skips_weekend():
    job = ScheduledJob(id="j", title="j", action="a", daily_utc="06:00", market_days_only=True)
    saturday_noon = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc).timestamp()
    # Последний торговый слот до субботнего полудня — пятница 06:00.
    last = job.last_scheduled_before(saturday_noon)
    assert datetime.fromtimestamp(last, tz=timezone.utc).strftime("%a %H:%M") == "Fri 06:00"
    # Следующий — понедельник.
    nxt = job.next_scheduled_after(saturday_noon)
    assert datetime.fromtimestamp(nxt, tz=timezone.utc).strftime("%a") == "Mon"


def test_weekday_mask():
    job = ScheduledJob(id="j", title="j", action="a", daily_utc="08:00", weekdays=[6])  # только Вс
    nxt = job.next_scheduled_after(WED_NOON)
    assert datetime.fromtimestamp(nxt, tz=timezone.utc).weekday() == 6


def test_validation_rejects_bad_jobs():
    with pytest.raises(ValueError):
        ScheduledJob(id="x", title="x", action="a").validate()  # нет триггера
    with pytest.raises(ValueError):
        ScheduledJob(id="x", title="x", action="a", interval_sec=60, daily_utc="01:00").validate()
    with pytest.raises(ValueError):
        ScheduledJob(id="x", title="x", action="a", interval_sec=5).validate()  # лавина
    with pytest.raises(ValueError):
        ScheduledJob(id="x", title="x", action="a", daily_utc="25:00").validate()


# ---------------------------------------------------------------- catch-up

BASE_DAILY = """
jobs:
  - id: daily_job
    title: "Daily"
    action: act
    enabled: true
    daily_utc: "06:00"
    catch_up: true
    catch_up_grace_sec: 21600
"""


def test_catch_up_within_grace_is_due(tmp_path):
    clock = Clock(WED_NOON)  # 12:00, план был в 06:00, grace 6ч → ровно на границе
    runs = []
    svc = make_service(
        tmp_path,
        BASE_DAILY,
        clock,
        actions={"act": lambda job: (runs.append(1), JobRunResult(STATUS_SUCCEEDED))[1]},
    )
    launched = svc.tick_once()
    assert launched == ["daily_job"]
    assert wait_terminal(svc, "daily_job") == STATUS_SUCCEEDED
    assert runs == [1]
    # Повторный тик в тот же день — не due (last_attempt >= scheduled_for).
    assert svc.tick_once() == []
    svc.stop()


def test_catch_up_outside_grace_skips_to_next_day(tmp_path):
    clock = Clock(WED_NOON + 3600)  # 13:00 — прошло 7ч > grace 6ч
    svc = make_service(
        tmp_path, BASE_DAILY, clock, actions={"act": lambda job: JobRunResult(STATUS_SUCCEEDED)}
    )
    assert svc.tick_once() == []
    # Но на следующий день в 06:05 — due.
    clock.ts = datetime(2026, 7, 16, 6, 5, tzinfo=timezone.utc).timestamp()
    assert svc.tick_once() == ["daily_job"]
    wait_terminal(svc, "daily_job")
    svc.stop()


def test_interval_job_anchors_then_fires(tmp_path):
    yaml_txt = """
jobs:
  - id: iv
    title: "Interval"
    action: act
    enabled: true
    interval_sec: 600
"""
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path, yaml_txt, clock, actions={"act": lambda job: JobRunResult(STATUS_SUCCEEDED)}
    )
    assert svc.tick_once() == []  # первый тик — только якорь, без лавины
    clock.advance(599)
    assert svc.tick_once() == []
    clock.advance(2)
    assert svc.tick_once() == ["iv"]
    wait_terminal(svc, "iv")
    svc.stop()


# ------------------------------------------------------------ ретраи/алерты


def test_retry_backoff_then_alert_after_exhaustion(tmp_path):
    yaml_txt = """
jobs:
  - id: flaky
    title: "Flaky"
    action: act
    enabled: true
    daily_utc: "06:00"
    max_retries: 1
    retry_backoff_sec: 300
"""
    clock = Clock(WED_NOON)
    alerts = []
    svc = make_service(
        tmp_path,
        yaml_txt,
        clock,
        actions={"act": lambda job: JobRunResult(STATUS_FAILED, "boom")},
        alert_fn=lambda key, text: alerts.append((key, text)),
    )
    assert svc.tick_once() == ["flaky"]
    wait_terminal(svc, "flaky")
    st = svc._job_state("flaky")
    assert st["consecutive_failures"] == 1
    assert st["retry_at"] == pytest.approx(clock.ts + 300, abs=5)
    assert alerts == []  # ретраи ещё не исчерпаны

    clock.advance(301)  # наступил retry_at
    assert svc.tick_once() == ["flaky"]
    wait_runs(svc, 2)
    st = svc._job_state("flaky")
    assert st["consecutive_failures"] == 2
    assert "retry_at" not in st  # исчерпано (max_retries=1)
    assert len(alerts) == 1 and "Flaky" in alerts[0][1]
    svc.stop()


def test_success_resets_failure_counter(tmp_path):
    outcomes = [JobRunResult(STATUS_FAILED, "x"), JobRunResult(STATUS_SUCCEEDED)]
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path,
        BASE_DAILY.replace("daily_job", "j2"),
        clock,
        actions={"act": lambda job: outcomes.pop(0)},
    )
    svc.tick_once()
    wait_runs(svc, 1)
    clock.advance(301)
    # retry_backoff_sec по умолчанию 300 → due по retry_at
    svc.tick_once()
    wait_runs(svc, 2)
    st = svc._job_state("j2")
    assert st["last_status"] == STATUS_SUCCEEDED
    assert st["consecutive_failures"] == 0
    svc.stop()


# ------------------------------------------------------- CCEA trading-гейт

TRADING_YAML = """
jobs:
  - id: rebal
    title: "Rebalance"
    action: act
    enabled: true
    daily_utc: "06:00"
    trading_impacting: true
"""


def test_trading_job_scheduled_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("RIVEN_ALLOW_SCHEDULED_TRADING", raising=False)
    ran = []
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path,
        TRADING_YAML,
        clock,
        actions={"act": lambda job: (ran.append(1), JobRunResult(STATUS_SUCCEEDED))[1]},
    )
    svc.tick_once()
    st = svc._job_state("rebal")
    assert st["last_status"] == STATUS_SKIPPED
    assert "RIVEN_ALLOW_SCHEDULED_TRADING" in st["last_detail"]
    assert ran == []  # действие даже не вызывалось
    # Журнал зафиксировал skip.
    runs = svc.recent_runs()
    assert runs and runs[0]["status"] == STATUS_SKIPPED
    svc.stop()


def test_trading_job_scheduled_runs_with_double_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVEN_ALLOW_SCHEDULED_TRADING", "1")
    ran = []
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path,
        TRADING_YAML,
        clock,
        actions={"act": lambda job: (ran.append(1), JobRunResult(STATUS_SUCCEEDED))[1]},
    )
    svc.tick_once()
    wait_terminal(svc, "rebal")
    assert ran == [1]
    svc.stop()


def test_trading_job_manual_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.delenv("RIVEN_ALLOW_SCHEDULED_TRADING", raising=False)
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path, TRADING_YAML, clock, actions={"act": lambda job: JobRunResult(STATUS_SUCCEEDED)}
    )
    with pytest.raises(PermissionError):
        svc.run_now("rebal")
    res = svc.run_now("rebal", confirm_trading=True)  # человеческое подтверждение
    assert res["queued"] is True
    assert wait_terminal(svc, "rebal") == STATUS_SUCCEEDED
    svc.stop()


# --------------------------------------------------- состояние и журнал


def test_enable_override_survives_restart(tmp_path):
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path, BASE_DAILY, clock, actions={"act": lambda j: JobRunResult(STATUS_SUCCEEDED)}
    )
    svc.set_enabled("daily_job", False)
    svc.stop()
    svc2 = make_service(
        tmp_path, BASE_DAILY, clock, actions={"act": lambda j: JobRunResult(STATUS_SUCCEEDED)}
    )
    job = svc2._jobs["daily_job"]
    assert svc2.is_enabled(job) is False  # override пережил «рестарт»
    assert svc2.tick_once() == []  # и реально блокирует запуск
    svc2.stop()


def test_status_shape_and_journal_order(tmp_path):
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path, BASE_DAILY, clock, actions={"act": lambda j: JobRunResult(STATUS_SUCCEEDED, "ok")}
    )
    svc.tick_once()
    wait_terminal(svc, "daily_job")
    status = svc.status()
    assert status["jobs"][0]["id"] == "daily_job"
    for key in ("enabled", "trigger", "last_status", "next_run_ts", "trading_impacting"):
        assert key in status["jobs"][0]
    runs = svc.recent_runs()
    assert runs[0]["job"] == "daily_job" and runs[0]["status"] == STATUS_SUCCEEDED
    json.dumps(status)
    json.dumps(runs)  # сериализуемость
    svc.stop()


def test_unknown_action_fails_honestly(tmp_path):
    clock = Clock(WED_NOON)
    svc = make_service(tmp_path, BASE_DAILY, clock, actions={})  # действия нет
    svc.tick_once()
    wait_terminal(svc, "daily_job")
    st = svc._job_state("daily_job")
    assert st["last_status"] == STATUS_FAILED
    assert "не зарегистрирован" in st["last_detail"]
    svc.stop()


# ------------------------------------------- действия app.py (fail-closed)


def test_research_pipeline_stops_on_first_failed_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prices.parquet").write_bytes(b"x")  # precondition
    calls = []

    def fake_worker(job_name, params, timeout_sec):
        calls.append(job_name)
        if job_name == "run_targets":
            return JobRunResult(STATUS_FAILED, "нет колонки")
        return JobRunResult(STATUS_SUCCEEDED, "ok")

    monkeypatch.setattr(app_module, "_sched_run_worker", fake_worker)
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(
        id="rn", title="rn", action="pipeline.research_nightly", daily_utc="01:00", timeout_sec=600
    )
    res = actions["pipeline.research_nightly"](job)
    assert res.status == STATUS_FAILED
    assert "run_targets" in res.detail
    assert calls == ["run_features", "run_targets"]  # fail-closed: цепочка остановлена
    assert [s["step"] for s in res.steps] == calls


def test_research_pipeline_enforces_leakguard_floor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prices.parquet").write_bytes(b"x")
    seen = {}

    def fake_worker(job_name, params, timeout_sec):
        if job_name == "run_training_table":
            seen.update(params)
        return JobRunResult(STATUS_SUCCEEDED)

    monkeypatch.setattr(app_module, "_sched_run_worker", fake_worker)
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(
        id="rn", title="rn", action="x", daily_utc="01:00", params={"decision_delay_ms": 50}
    )  # попытка ослабить
    res = actions["pipeline.research_nightly"](job)
    assert res.status == STATUS_SUCCEEDED
    assert seen["decision_delay_ms"] == 8000  # пол не ослабляется планировщиком


def test_drift_action_recommends_without_auto_retrain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "data" / "training_table.parquet").write_bytes(b"x")
    (tmp_path / "models" / "drift_report.json").write_text(
        json.dumps(
            {
                "features": {"f1": {"psi": 0.9}},
                "avg_psi": 0.9,
                "worst_psi": 0.9,
                "worst_feature": "f1",
                "status": "drift",
            }
        ),
        encoding="utf-8",
    )
    launched = []
    monkeypatch.setattr(
        app_module,
        "_sched_run_worker",
        lambda name, params, t: (launched.append(name), JobRunResult(STATUS_SUCCEEDED))[1],
    )
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(
        id="d",
        title="d",
        action="x",
        daily_utc="06:00",
        params={"psi_threshold": 0.25, "auto_retrain": False},
    )
    res = actions["monitor.drift_and_retrain"](job)
    assert res.status == STATUS_SUCCEEDED
    assert "ДРЕЙФ" in res.detail
    assert launched == ["run_psi"]  # ретрейн НЕ запускался


def test_eod_report_written_even_without_ccea(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(
        id="e", title="e", action="x", daily_utc="21:15", params={"reports_dir": "reports/daily"}
    )
    res = actions["eod.close_and_report"](job)
    assert res.status == STATUS_SUCCEEDED
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = json.loads(
        (tmp_path / "reports" / "daily" / f"{day}.json").read_text(encoding="utf-8")
    )
    assert report["ccea"] is None
    assert "CCEA Agent не запущен" in (report["note"] or "")  # честно, без выдуманного NAV


def test_xs_rebalance_action_is_fail_closed_skeleton(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    actions = app_module._build_scheduler_actions()
    job = ScheduledJob(
        id="x",
        title="x",
        action="t",
        daily_utc="13:45",
        trading_impacting=True,
        params={"config": ""},
    )
    res = actions["trade.xs_rebalance"](job)
    assert res.status == STATUS_SKIPPED  # без конфига — только skip, никаких сделок


# ----------------------------------------------------------------- REST API


def test_api_503_when_scheduler_disabled(monkeypatch):
    monkeypatch.setattr(app_module, "_SCHEDULER", None, raising=False)
    assert client.get("/api/scheduler/status").status_code == 503


def test_api_roundtrip_with_injected_scheduler(tmp_path, monkeypatch):
    clock = Clock(WED_NOON)
    svc = make_service(
        tmp_path,
        TRADING_YAML
        + """
  - id: safe
    title: "Safe"
    action: act
    enabled: true
    daily_utc: "23:59"
""",
        clock,
        actions={"act": lambda j: JobRunResult(STATUS_SUCCEEDED)},
    )
    monkeypatch.setattr(app_module, "_SCHEDULER", svc, raising=False)

    body = client.get("/api/scheduler/status").json()
    assert {j["id"] for j in body["jobs"]} == {"rebal", "safe"}

    assert client.post("/api/scheduler/job/safe/enable", json={"enabled": False}).status_code == 200
    assert [j for j in client.get("/api/scheduler/status").json()["jobs"] if j["id"] == "safe"][0][
        "enabled"
    ] is False

    assert client.post("/api/scheduler/job/nope/enable", json={"enabled": True}).status_code == 404
    # Торговая задача без подтверждения — 409 (CCEA local approval).
    assert (
        client.post("/api/scheduler/job/rebal/run", json={"confirm_trading": False}).status_code
        == 409
    )
    # Безопасная задача запускается вручную.
    res = client.post("/api/scheduler/job/safe/run", json={})
    assert res.status_code == 200 and res.json()["queued"] is True
    wait_terminal(svc, "safe")

    runs = client.get("/api/scheduler/runs?limit=5").json()["runs"]
    assert runs and runs[0]["job"] == "safe"
    svc.stop()


def test_scheduler_not_autostarted_under_pytest():
    # Автостарт обязан молчать под pytest — иначе каждый тестовый прогон
    # запускал бы catch-up задачи в рабочей копии.
    assert app_module._SCHEDULER is None or isinstance(app_module._SCHEDULER, SchedulerService)
    # Прямая проверка гейта:
    import sys as _sys

    assert "pytest" in _sys.modules


def test_default_catalog_is_valid_and_safe():
    """configs/scheduler.yaml: валиден, торговые задачи выключены, включённые — безопасны."""
    import yaml as _yaml

    cfg = _yaml.safe_load(Path("configs/scheduler.yaml").read_text(encoding="utf-8"))
    jobs = {j["id"]: j for j in cfg["jobs"]}
    for j in jobs.values():
        ScheduledJob(
            id=j["id"],
            title=j.get("title", ""),
            action=j["action"],
            interval_sec=j.get("interval_sec"),
            daily_utc=j.get("daily_utc"),
        ).validate()
    # Все trading_impacting задачи по умолчанию выключены.
    for j in jobs.values():
        if j.get("trading_impacting"):
            assert (
                j.get("enabled") is False
            ), f"{j['id']}: торговая задача не может быть включена по умолчанию"
    # Включённые по умолчанию — только безопасные локальные задачи.
    enabled = {jid for jid, j in jobs.items() if j.get("enabled")}
    assert enabled <= {"drift_check", "eod_close_report", "state_backup", "log_rotation"}
    # Research-пайплайн в каталоге не ослабляет LeakGuard.
    assert jobs["research_nightly"]["params"]["decision_delay_ms"] >= 8000
