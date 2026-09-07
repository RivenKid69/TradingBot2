"""Планировщик регулярных задач платформы (data → research → risk → post-trade).

Закрывает P0-F из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: до него обновление
данных, drift-retrain и EOD-переоценка запускались только руками.

Дизайн следует практике реальных квант-пайплайнов, адаптированной к desktop-среде
(машина НЕ работает 24/7):

* **Anacron-семантика (catch-up).** Ежедневная задача, пропущенная из-за
  выключенного приложения, выполняется при следующем старте, если с планового
  времени прошло не больше ``catch_up_grace_sec``. Это стандартное поведение
  anacron/launchd, а не cron: cron-семантика на десктопе просто теряет запуски.
* **Fail-closed пайплайны.** Составная задача (ingest → QC → features → …)
  останавливается на первом упавшем шаге; смежные задачи не запускаются на
  заведомо битых данных.
* **Ретраи с экспоненциальным backoff** и алертом после исчерпания попыток
  (Telegram/webhook через services.alerts, если сконфигурированы).
* **Глобальная сериализация тяжёлых задач**: один воркер-поток — обучение и
  инжест никогда не толкаются локтями (та же дисциплина, что и pid-файлы
  фоновых джобов приложения).
* **CCEA: TRADING_IMPACTING задачи никогда не автостартуют.** Автозапуск
  требует двойного opt-in (``enabled`` в YAML **и** env
  ``RIVEN_ALLOW_SCHEDULED_TRADING=1``); ручной запуск — явного подтверждения
  человеком (``confirm_trading=True``). По умолчанию — skipped, не failure.
* **Долговечность и наблюдаемость.** Состояние (последний запуск, статусы,
  счётчики ретраев) — в ``state/scheduler_state.json`` (атомарная запись);
  журнал запусков — ``logs/scheduler_runs.jsonl``; всё видно через REST/UI.

Ядро не импортирует ``app`` — действия (actions) инжектируются снаружи, чтобы
не создавать циклических импортов и чтобы сервис тестировался в изоляции.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time as _time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import yaml

from services.utils_app import atomic_write_json, read_json

logger = logging.getLogger(__name__)

# Статусы завершения одного запуска задачи.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"  # precondition не выполнен / trading-гейт: НЕ ошибка
STATUS_TIMEOUT = "timeout"

_TRUTHY = ("1", "true", "yes", "on")


def _scheduled_trading_allowed() -> bool:
    return os.environ.get("RIVEN_ALLOW_SCHEDULED_TRADING", "").strip().lower() in _TRUTHY


@dataclass
class JobRunResult:
    """Итог одного выполнения действия задачи."""

    status: str
    detail: str = ""
    exit_code: Optional[int] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScheduledJob:
    """Декларативное описание задачи из configs/scheduler.yaml."""

    id: str
    title: str
    action: str
    enabled: bool = False
    # Триггер: ровно один из двух.
    interval_sec: Optional[int] = None
    daily_utc: Optional[str] = None  # "HH:MM"
    weekdays: Optional[List[int]] = None  # 0=Пн … 6=Вс; None = все дни
    market_days_only: bool = False  # пропускать Сб/Вс (v1; праздники — v2)
    catch_up: bool = True
    catch_up_grace_sec: int = 6 * 3600
    max_retries: int = 1
    retry_backoff_sec: int = 300
    timeout_sec: int = 3600
    trading_impacting: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if bool(self.interval_sec) == bool(self.daily_utc):
            raise ValueError(
                f"job '{self.id}': нужен ровно один триггер (interval_sec ИЛИ daily_utc)"
            )
        if self.daily_utc is not None:
            hh, mm = self.daily_utc.split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError(f"job '{self.id}': daily_utc вне диапазона: {self.daily_utc!r}")
        if self.interval_sec is not None and self.interval_sec < 30:
            raise ValueError(f"job '{self.id}': interval_sec < 30 запрещён (лавина запусков)")

    def _day_allowed(self, dt: datetime) -> bool:
        if self.market_days_only and dt.weekday() >= 5:
            return False
        if self.weekdays is not None and dt.weekday() not in self.weekdays:
            return False
        return True

    def last_scheduled_before(self, now_ts: float) -> Optional[float]:
        """Последнее плановое время (epoch) <= now для daily-триггера."""
        if self.daily_utc is None:
            return None
        hh, mm = (int(x) for x in self.daily_utc.split(":"))
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        for _ in range(14):  # максимум две недели назад (маски дней)
            if cand <= now and self._day_allowed(cand):
                return cand.timestamp()
            cand -= timedelta(days=1)
        return None

    def next_scheduled_after(self, now_ts: float) -> Optional[float]:
        """Следующее плановое время (epoch) > now — для отображения в UI."""
        if self.interval_sec is not None:
            return None  # считается от last_attempt в SchedulerService
        hh, mm = (int(x) for x in self.daily_utc.split(":"))
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand <= now:
            cand += timedelta(days=1)
        for _ in range(14):
            if self._day_allowed(cand):
                return cand.timestamp()
            cand += timedelta(days=1)
        return None


class SchedulerService:
    """Тик-цикл + долговечное состояние + журнал + один воркер исполнения."""

    def __init__(
        self,
        *,
        config_path: str = os.path.join("configs", "scheduler.yaml"),
        state_path: str = os.path.join("state", "scheduler_state.json"),
        journal_path: str = os.path.join("logs", "scheduler_runs.jsonl"),
        actions: Optional[Dict[str, Callable[[ScheduledJob], JobRunResult]]] = None,
        alert_fn: Optional[Callable[[str, str], None]] = None,
        time_fn: Callable[[], float] = _time.time,
        tick_sec: float = 30.0,
    ) -> None:
        self.config_path = config_path
        self.state_path = state_path
        self.journal_path = journal_path
        self.actions = dict(actions or {})
        self.alert_fn = alert_fn
        self.time_fn = time_fn
        self.tick_sec = float(tick_sec)

        self._jobs: Dict[str, ScheduledJob] = {}
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._running_ids: set = set()
        # Один воркер = глобальная сериализация тяжёлых задач (см. док-стринг).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sched-job")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.started_at = self.time_fn()

        self.reload_config()
        self._load_state()

    # ------------------------------------------------------------- config/state

    def reload_config(self) -> None:
        jobs: Dict[str, ScheduledJob] = {}
        raw: Dict[str, Any] = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.error("scheduler: не удалось прочитать %s: %s", self.config_path, exc)
        for item in raw.get("jobs") or []:
            try:
                job = ScheduledJob(
                    id=str(item["id"]),
                    title=str(item.get("title", item["id"])),
                    action=str(item["action"]),
                    enabled=bool(item.get("enabled", False)),
                    interval_sec=item.get("interval_sec"),
                    daily_utc=item.get("daily_utc"),
                    weekdays=item.get("weekdays"),
                    market_days_only=bool(item.get("market_days_only", False)),
                    catch_up=bool(item.get("catch_up", True)),
                    catch_up_grace_sec=int(item.get("catch_up_grace_sec", 6 * 3600)),
                    max_retries=int(item.get("max_retries", 1)),
                    retry_backoff_sec=int(item.get("retry_backoff_sec", 300)),
                    timeout_sec=int(item.get("timeout_sec", 3600)),
                    trading_impacting=bool(item.get("trading_impacting", False)),
                    params=dict(item.get("params") or {}),
                )
                job.validate()
                jobs[job.id] = job
            except Exception as exc:
                logger.error("scheduler: пропускаю некорректную задачу %r: %s", item, exc)
        with self._lock:
            self._jobs = jobs

    def _load_state(self) -> None:
        with self._lock:
            data = read_json(self.state_path)
            self._state = data if isinstance(data, dict) else {}
            self._state.setdefault("jobs", {})

    def _save_state(self) -> None:
        with self._lock:
            self._state["updated_at"] = datetime.fromtimestamp(
                self.time_fn(), tz=timezone.utc
            ).isoformat(timespec="seconds")
            atomic_write_json(self.state_path, self._state)

    def _job_state(self, job_id: str) -> Dict[str, Any]:
        return self._state.setdefault("jobs", {}).setdefault(job_id, {})

    # ---------------------------------------------------------------- журнал

    def _journal(self, entry: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.journal_path) or ".", exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("scheduler: journal write failed: %s", exc)

    def recent_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.journal_path):
            return []
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-max(1, int(limit)) :]
            out = []
            for line in lines:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
            return list(reversed(out))
        except Exception:
            return []

    def notify(self, key: str, text: str) -> None:
        """Алерт во внешний канал (если сконфигурирован) — не роняет планировщик."""
        if self.alert_fn is None:
            return
        try:
            self.alert_fn(key, text)
        except Exception as exc:
            logger.warning("scheduler: alert failed: %s", exc)

    # ------------------------------------------------------------- расписание

    def is_enabled(self, job: ScheduledJob) -> bool:
        override = self._job_state(job.id).get("enabled_override")
        return bool(job.enabled if override is None else override)

    def set_enabled(self, job_id: str, enabled: bool) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._job_state(job_id)["enabled_override"] = bool(enabled)
            self._save_state()
        return {"job": job_id, "enabled": bool(enabled)}

    def _effective_grace(self, job: ScheduledJob) -> float:
        return float(job.catch_up_grace_sec) if job.catch_up else 900.0

    def _due(self, job: ScheduledJob, now: float) -> bool:
        st = self._job_state(job.id)
        if job.id in self._running_ids:
            return False
        retry_at = st.get("retry_at")
        if retry_at:
            return now >= float(retry_at)
        last_attempt = st.get("last_attempt_ts")
        if job.interval_sec is not None:
            if last_attempt is None:
                # Первый запуск через полный интервал после старта приложения —
                # без лавины задач в момент включения.
                st["last_attempt_ts"] = self.started_at
                return False
            return (now - float(last_attempt)) >= float(job.interval_sec)
        scheduled_for = job.last_scheduled_before(now)
        if scheduled_for is None:
            return False
        if last_attempt is not None and float(last_attempt) >= scheduled_for:
            return False
        return (now - scheduled_for) <= self._effective_grace(job)

    def next_run_ts(self, job: ScheduledJob, now: float) -> Optional[float]:
        st = self._job_state(job.id)
        retry_at = st.get("retry_at")
        if retry_at:
            return float(retry_at)
        if job.interval_sec is not None:
            last = st.get("last_attempt_ts") or self.started_at
            return float(last) + float(job.interval_sec)
        if self._due(job, now):
            return now
        return job.next_scheduled_after(now)

    # ------------------------------------------------------------- исполнение

    def run_now(self, job_id: str, *, confirm_trading: bool = False) -> Dict[str, Any]:
        """Ручной запуск. Для trading-impacting задач требует подтверждения человеком."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.trading_impacting and not confirm_trading:
                raise PermissionError(
                    "trading-impacting задача: ручной запуск требует confirm_trading=true (CCEA local approval)"
                )
            if job_id in self._running_ids:
                return {"job": job_id, "queued": False, "detail": "уже выполняется"}
            self._running_ids.add(job_id)
        self._executor.submit(self._execute, job, True)
        return {"job": job_id, "queued": True}

    def _submit_scheduled(self, job: ScheduledJob) -> None:
        # Fail-closed гейт CCEA: автозапуск торговых задач — только при явном
        # двойном opt-in. Иначе — журналируемый skip, не ошибка.
        if job.trading_impacting and not _scheduled_trading_allowed():
            now = self.time_fn()
            with self._lock:
                st = self._job_state(job.id)
                st["last_attempt_ts"] = now
                st["last_status"] = STATUS_SKIPPED
                st["last_detail"] = (
                    "trading-impacting: автозапуск запрещён (нет RIVEN_ALLOW_SCHEDULED_TRADING=1)"
                )
                st.pop("retry_at", None)
                self._save_state()
            self._journal(
                {
                    "job": job.id,
                    "run_id": uuid.uuid4().hex[:8],
                    "manual": False,
                    "started_at": now,
                    "finished_at": now,
                    "status": STATUS_SKIPPED,
                    "detail": st["last_detail"],
                    "attempt": 0,
                }
            )
            return
        with self._lock:
            if job.id in self._running_ids:
                return
            self._running_ids.add(job.id)
        self._executor.submit(self._execute, job, False)

    def _execute(self, job: ScheduledJob, manual: bool) -> None:
        run_id = uuid.uuid4().hex[:8]
        started = self.time_fn()
        with self._lock:
            st = self._job_state(job.id)
            attempt = int(st.get("consecutive_failures", 0)) + 1
            st["last_attempt_ts"] = started
            st["last_status"] = "running"
            st["last_detail"] = ""
            st.pop("retry_at", None)
            self._save_state()

        action = self.actions.get(job.action)
        if action is None:
            result = JobRunResult(STATUS_FAILED, f"action '{job.action}' не зарегистрирован")
        else:
            try:
                result = action(job)
                if not isinstance(result, JobRunResult):
                    result = JobRunResult(STATUS_SUCCEEDED, detail=str(result))
            except Exception as exc:  # действие не должно ронять воркер
                logger.exception("scheduler: job %s crashed", job.id)
                result = JobRunResult(STATUS_FAILED, f"exception: {exc}")

        finished = self.time_fn()
        with self._lock:
            st = self._job_state(job.id)
            st["last_finish_ts"] = finished
            st["last_status"] = result.status
            st["last_detail"] = result.detail[:500]
            if result.status in (STATUS_FAILED, STATUS_TIMEOUT):
                fails = int(st.get("consecutive_failures", 0)) + 1
                st["consecutive_failures"] = fails
                if fails <= job.max_retries:
                    backoff = job.retry_backoff_sec * (2 ** (fails - 1))
                    st["retry_at"] = finished + backoff
                else:
                    st.pop("retry_at", None)
                    self.notify(
                        f"scheduler:{job.id}",
                        f"⛔ Задача «{job.title}» упала {fails} раз(а) подряд и исчерпала ретраи: {result.detail[:300]}",
                    )
            else:
                st["consecutive_failures"] = 0
                st.pop("retry_at", None)
            self._running_ids.discard(job.id)
            self._save_state()

        self._journal(
            {
                "job": job.id,
                "run_id": run_id,
                "manual": manual,
                "started_at": started,
                "finished_at": finished,
                "status": result.status,
                "detail": result.detail[:500],
                "exit_code": result.exit_code,
                "steps": result.steps,
                "attempt": attempt,
            }
        )

    # ------------------------------------------------------------------- цикл

    def tick_once(self) -> List[str]:
        """Один проход планирования; возвращает id поставленных задач (для тестов)."""
        now = self.time_fn()
        launched: List[str] = []
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if not self.is_enabled(job):
                continue
            with self._lock:
                due = self._due(job, now)
                if due:
                    self._save_state()  # _due мог проинициализировать interval-якорь
            if due:
                self._submit_scheduled(job)
                launched.append(job.id)
        return launched

    def _loop(self) -> None:
        while not self._stop_event.wait(self.tick_sec):
            try:
                self.tick_once()
            except Exception:
                logger.exception("scheduler: tick failed")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="SchedulerThread", daemon=True)
        self._thread.start()
        logger.info("scheduler: запущен (%d задач, tick=%ss)", len(self._jobs), self.tick_sec)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -------------------------------------------------------------------- API

    def status(self) -> Dict[str, Any]:
        now = self.time_fn()
        out = []
        with self._lock:
            for job in self._jobs.values():
                st = self._job_state(job.id)
                nxt = self.next_run_ts(job, now)
                out.append(
                    {
                        "id": job.id,
                        "title": job.title,
                        "action": job.action,
                        "enabled": self.is_enabled(job),
                        "trading_impacting": job.trading_impacting,
                        "trigger": (
                            f"каждые {job.interval_sec}с"
                            if job.interval_sec
                            else f"ежедневно {job.daily_utc} UTC"
                            + (" (только торговые дни)" if job.market_days_only else "")
                        ),
                        "running": job.id in self._running_ids,
                        "last_attempt_ts": st.get("last_attempt_ts"),
                        "last_status": st.get("last_status"),
                        "last_detail": st.get("last_detail", ""),
                        "consecutive_failures": st.get("consecutive_failures", 0),
                        "next_run_ts": nxt,
                    }
                )
        return {
            "enabled_env": os.environ.get("RIVEN_ENABLE_SCHEDULER", "1"),
            "scheduled_trading_allowed": _scheduled_trading_allowed(),
            "tick_sec": self.tick_sec,
            "jobs": out,
            "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
        }


__all__ = [
    "JobRunResult",
    "ScheduledJob",
    "SchedulerService",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "STATUS_SUCCEEDED",
    "STATUS_TIMEOUT",
]
