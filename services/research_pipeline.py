# -*- coding: utf-8 -*-
"""DAG-оркестратор research-пайплайна (закрывает «планировщик пайплайна», §5.21+).

Планировщик задач (``services/scheduler.py``, P0-F) отвечает за «КОГДА»
(cron/anacron). Этот модуль отвечает за «ЧТО И В КАКОМ ПОРЯДКЕ»: DAG шагов с
зависимостями — то, что в проф. пайплайнах делает Airflow/Dagster/Prefect,
но легковесно и без внешних демонов:

* декларативные YAML-спеки в ``configs/pipelines/*.yaml`` (шаги: worker из
  реестра джобов приложения + params + depends_on + retries + timeout);
* топологическое выполнение (Kahn, детерминированный порядок), шаги с упавшей
  зависимостью помечаются ``blocked`` (fail-closed, не выполняются);
* долговечный журнал запуска ``state/pipeline_runs/<run_id>.json``
  (пишется после каждого шага) → ``resume`` докатывает упавший прогон, не
  повторяя успешные шаги;
* cancel между шагами;
* **LeakGuard-пол**: любой шаг с ``decision_delay_ms`` клампится движком к
  ≥ 8000 мс — YAML пользователя не может ослабить защиту от утечки.

Исполнение воркера инжектится (``worker_fn(name, params, timeout) →
(status, detail, exit_code)``) — движок не знает про FastAPI/subprocess и
полностью тестируем; в приложении воркером служит тот же bridge, что у
планировщика (``_sched_run_worker``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

PIPELINES_DIR = os.path.join("configs", "pipelines")
RUNS_DIR = os.path.join("state", "pipeline_runs")

LEAKGUARD_MIN_DELAY_MS = 8000

STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_SUCCEEDED = "succeeded"
STEP_FAILED = "failed"
STEP_BLOCKED = "blocked"  # зависимость упала — шаг не выполнялся (fail-closed)
STEP_CANCELLED = "cancelled"

RUN_RUNNING = "running"
RUN_SUCCEEDED = "succeeded"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"

WorkerFn = Callable[[str, Dict[str, Any], int], Tuple[str, str, Optional[int]]]


@dataclass
class StepSpec:
    id: str
    worker: str
    title: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout_sec: int = 1800
    retries: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StepSpec":
        return StepSpec(
            id=str(d["id"]),
            worker=str(d["worker"]),
            title=str(d.get("title", d["id"])),
            params=dict(d.get("params", {}) or {}),
            depends_on=[str(x) for x in (d.get("depends_on") or [])],
            timeout_sec=int(d.get("timeout_sec", 1800)),
            retries=max(0, int(d.get("retries", 0))),
        )


@dataclass
class PipelineSpec:
    name: str
    title: str = ""
    description: str = ""
    steps: List[StepSpec] = field(default_factory=list)
    path: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "PipelineSpec":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        spec = PipelineSpec(
            name=str(data.get("name") or os.path.splitext(os.path.basename(path))[0]),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            steps=[StepSpec.from_dict(s) for s in (data.get("steps") or [])],
            path=path,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"pipeline {self.name}: дублирующиеся id шагов")
        known = set(ids)
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in known:
                    raise ValueError(
                        f"pipeline {self.name}: шаг {s.id} зависит от неизвестного {dep!r}"
                    )
        self.topo_order()  # бросит на цикле

    def topo_order(self) -> List[str]:
        """Kahn с сортированным ready-множеством — детерминированный порядок."""
        indeg = {s.id: len(s.depends_on) for s in self.steps}
        children: Dict[str, List[str]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                children[dep].append(s.id)
        ready = sorted([i for i, d in indeg.items() if d == 0])
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for ch in sorted(children[node]):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    ready.append(ch)
            ready.sort()
        if len(order) != len(self.steps):
            raise ValueError(f"pipeline {self.name}: цикл в зависимостях")
        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "path": self.path,
            "steps": [
                {
                    "id": s.id,
                    "worker": s.worker,
                    "title": s.title,
                    "params": s.params,
                    "depends_on": s.depends_on,
                    "timeout_sec": s.timeout_sec,
                    "retries": s.retries,
                }
                for s in self.steps
            ],
        }


def list_specs(directory: str = PIPELINES_DIR) -> List[PipelineSpec]:
    specs: List[PipelineSpec] = []
    if not os.path.isdir(directory):
        return specs
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(directory, fname)
        try:
            specs.append(PipelineSpec.from_yaml(path))
        except Exception as exc:
            logger.warning("pipeline spec %s не загрузился: %s", path, exc)
    return specs


def load_spec(name: str, directory: str = PIPELINES_DIR) -> Optional[PipelineSpec]:
    for spec in list_specs(directory):
        if spec.name == name:
            return spec
    return None


def _apply_leakguard_floor(step: StepSpec) -> StepSpec:
    """Движковый LeakGuard-пол: YAML не может ослабить задержку решения."""
    if "decision_delay_ms" in step.params:
        try:
            requested = int(step.params["decision_delay_ms"])
        except (TypeError, ValueError):
            requested = LEAKGUARD_MIN_DELAY_MS
        if requested < LEAKGUARD_MIN_DELAY_MS:
            logger.warning(
                "pipeline: шаг %s запросил decision_delay_ms=%s < пола %s — клампим",
                step.id,
                requested,
                LEAKGUARD_MIN_DELAY_MS,
            )
        step.params["decision_delay_ms"] = max(requested, LEAKGUARD_MIN_DELAY_MS)
    return step


class PipelineRunner:
    """Исполнитель DAG с долговечным состоянием и resume."""

    def __init__(self, worker_fn: WorkerFn, *, runs_dir: str = RUNS_DIR) -> None:
        self._worker = worker_fn
        self._runs_dir = runs_dir
        self._cancel_flags: Dict[str, bool] = {}
        self._lock = threading.RLock()
        os.makedirs(runs_dir, exist_ok=True)

    # ------------------------------------------------------------- state io

    def _run_path(self, run_id: str) -> str:
        return os.path.join(self._runs_dir, f"{run_id}.json")

    def _save(self, state: Dict[str, Any]) -> None:
        tmp = self._run_path(state["run_id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._run_path(state["run_id"]))

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._run_path(run_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.isdir(self._runs_dir):
            return []
        files = sorted(
            (f for f in os.listdir(self._runs_dir) if f.endswith(".json")),
            key=lambda f: os.path.getmtime(os.path.join(self._runs_dir, f)),
            reverse=True,
        )[:limit]
        out = []
        for f in files:
            st = self.load_run(f[:-5])
            if st:
                out.append(
                    {
                        k: st.get(k)
                        for k in ("run_id", "pipeline", "status", "started_utc", "finished_utc")
                    }
                )
        return out

    def cancel(self, run_id: str) -> None:
        with self._lock:
            self._cancel_flags[run_id] = True

    # ------------------------------------------------------------- execute

    def run(
        self,
        spec: PipelineSpec,
        *,
        run_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Выполнить DAG. ``resume=True`` с существующим run_id докатывает
        прогон: succeeded-шаги не повторяются."""
        run_id = run_id or f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        prev: Dict[str, Dict[str, Any]] = {}
        if resume:
            prior = self.load_run(run_id)
            if prior:
                prev = {s["id"]: s for s in prior.get("steps", [])}

        state: Dict[str, Any] = {
            "run_id": run_id,
            "pipeline": spec.name,
            "status": RUN_RUNNING,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_utc": None,
            "resumed": bool(resume and prev),
            "steps": [],
        }
        step_state: Dict[str, Dict[str, Any]] = {}
        for s in spec.steps:
            carried = prev.get(s.id, {})
            st = {
                "id": s.id,
                "worker": s.worker,
                "title": s.title,
                "status": (
                    STEP_SUCCEEDED if carried.get("status") == STEP_SUCCEEDED else STEP_PENDING
                ),
                "detail": (
                    carried.get("detail", "") if carried.get("status") == STEP_SUCCEEDED else ""
                ),
                "attempts": 0,
            }
            step_state[s.id] = st
            state["steps"].append(st)
        self._save(state)

        specs_by_id = {s.id: s for s in spec.steps}
        failed_any = False

        for step_id in spec.topo_order():
            st = step_state[step_id]
            if st["status"] == STEP_SUCCEEDED:  # resume-скип
                continue

            with self._lock:
                if self._cancel_flags.pop(run_id, False):
                    st["status"] = STEP_CANCELLED
                    state["status"] = RUN_CANCELLED
                    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    self._save(state)
                    return state

            sspec = _apply_leakguard_floor(specs_by_id[step_id])

            # fail-closed: зависимость не succeeded → blocked, не выполняем
            broken = [d for d in sspec.depends_on if step_state[d]["status"] != STEP_SUCCEEDED]
            if broken:
                st["status"] = STEP_BLOCKED
                st["detail"] = f"зависимости не выполнены: {', '.join(broken)}"
                failed_any = True
                self._save(state)
                continue

            st["status"] = STEP_RUNNING
            self._save(state)
            final_status, detail = STEP_FAILED, ""
            for attempt in range(sspec.retries + 1):
                st["attempts"] = attempt + 1
                try:
                    status, detail, exit_code = self._worker(
                        sspec.worker, dict(sspec.params), sspec.timeout_sec
                    )
                except Exception as exc:
                    status, detail, exit_code = "failed", f"worker exception: {exc}", None
                if status == "succeeded":
                    final_status = STEP_SUCCEEDED
                    break
                final_status = STEP_FAILED
            st["status"] = final_status
            st["detail"] = detail
            if final_status != STEP_SUCCEEDED:
                failed_any = True
            self._save(state)

        state["status"] = RUN_FAILED if failed_any else RUN_SUCCEEDED
        state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save(state)
        return state


__all__ = [
    "LEAKGUARD_MIN_DELAY_MS",
    "PIPELINES_DIR",
    "RUNS_DIR",
    "PipelineRunner",
    "PipelineSpec",
    "StepSpec",
    "list_specs",
    "load_spec",
    "STEP_BLOCKED",
    "STEP_CANCELLED",
    "STEP_FAILED",
    "STEP_PENDING",
    "STEP_RUNNING",
    "STEP_SUCCEEDED",
    "RUN_CANCELLED",
    "RUN_FAILED",
    "RUN_RUNNING",
    "RUN_SUCCEEDED",
]
