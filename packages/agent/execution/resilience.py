# -*- coding: utf-8 -*-
"""
packages/agent/execution/resilience.py
======================================

Авто-recovery исполнения (P1, Agent-зона): устойчивость к сбоям брокера.

  * ``RetryPolicy`` — экспоненциальный backoff + jitter.
  * ``CircuitBreaker`` — CLOSED/OPEN/HALF_OPEN: после N подряд ошибок «размыкает»
    цепь (не бьёт по брокеру), через cooldown пробует half-open.
  * ``ResilientExecutor`` — обёртка вызова брокера: retry(backoff) под защитой breaker.
  * ``OrderStatusPoller`` — авто-poll статусов ордеров (детект терминальных/филлов).
  * ``StartupReconciler`` — сверка локального состояния (journal) с брокером ПРИ СТАРТЕ
    после сбоя: расхождения по ордерам и позициям → список действий.

Время инъектируется (``time_fn``/``sleep_fn``) → тесты детерминированы без реального сна.
Без тяжёлых зависимостей (stdlib). Слой Agent (ордера/секреты остаются локально, CCEA).
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """Брокерская цепь разомкнута — вызов отклонён без обращения к брокеру."""


class MaxRetriesExceeded(RuntimeError):
    """Превышено число попыток retry."""


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.0  # доля (0.1 = ±10%); детерминированный по умолчанию

    def delay(self, attempt: int, *, rand: float = 0.0) -> float:
        """Задержка перед попыткой ``attempt`` (1-based). ``rand`` ∈ [0,1) для jitter (DI)."""
        d = self.base_delay * (self.multiplier ** max(0, attempt - 1))
        d = min(self.max_delay, d)
        if self.jitter > 0:
            d = d * (1.0 + self.jitter * (2.0 * rand - 1.0))
        return max(0.0, d)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Размыкатель цепи по подряд идущим ошибкам брокера."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        time_fn: Callable[[], float] = _time.time,
    ) -> None:
        self.failure_threshold = int(failure_threshold)
        self.reset_timeout = float(reset_timeout)
        self.half_open_max_calls = int(half_open_max_calls)
        self._time = time_fn
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        self._maybe_half_open()
        return self._state

    def _maybe_half_open(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self._time() - self._opened_at >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def allow(self) -> bool:
        """Можно ли сейчас вызывать брокера."""
        self._maybe_half_open()
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False  # OPEN

    def record_attempt(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_calls = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._time()
        logger.warning("CircuitBreaker OPEN after %d failures", self._failures)


# ---------------------------------------------------------------------------
# Resilient executor (retry + circuit breaker)
# ---------------------------------------------------------------------------
class ResilientExecutor:
    """Вызов брокера с retry(backoff) под защитой circuit breaker."""

    def __init__(
        self,
        *,
        retry: Optional[RetryPolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        retryable: Tuple[type, ...] = (Exception,),
        sleep_fn: Callable[[float], None] = _time.sleep,
        rand_fn: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self.retryable = retryable
        self._sleep = sleep_fn
        self._rand = rand_fn
        self.attempts_made = 0
        self.sleeps: List[float] = []

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retry.max_attempts + 1):
            if not self.breaker.allow():
                raise CircuitOpenError("circuit breaker is OPEN — broker call rejected")
            self.breaker.record_attempt()
            self.attempts_made += 1
            try:
                result = fn(*args, **kwargs)
                self.breaker.record_success()
                return result
            except self.retryable as exc:
                last_exc = exc
                self.breaker.record_failure()
                logger.warning(
                    "broker call failed (attempt %d/%d): %s", attempt, self.retry.max_attempts, exc
                )
                if attempt < self.retry.max_attempts:
                    d = self.retry.delay(attempt, rand=self._rand())
                    self.sleeps.append(d)
                    self._sleep(d)
        raise MaxRetriesExceeded(str(last_exc)) from last_exc


# ---------------------------------------------------------------------------
# Order status auto-poll
# ---------------------------------------------------------------------------
_TERMINAL = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED", "ERROR"}


@dataclass
class OrderStatus:
    order_id: str
    status: str
    filled_qty: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return str(self.status).upper() in _TERMINAL

    @property
    def is_filled(self) -> bool:
        return str(self.status).upper() == "FILLED"


class OrderStatusPoller:
    """Авто-poll статусов ордеров через ``status_fn(order_id)->dict``."""

    def __init__(self, status_fn: Callable[[str], Dict[str, Any]]) -> None:
        self._status_fn = status_fn

    def poll(self, order_ids: Sequence[str]) -> Dict[str, OrderStatus]:
        out: Dict[str, OrderStatus] = {}
        for oid in order_ids:
            try:
                raw = self._status_fn(oid) or {}
            except Exception as exc:  # pragma: no cover - брокерский сбой
                logger.warning("status poll failed for %s: %s", oid, exc)
                out[oid] = OrderStatus(oid, "ERROR", raw={"error": str(exc)})
                continue
            out[oid] = OrderStatus(
                order_id=oid,
                status=str(raw.get("status", "UNKNOWN")).upper(),
                filled_qty=float(raw.get("filled_qty", raw.get("filled", 0.0)) or 0.0),
                raw=raw,
            )
        return out

    def split_terminal(self, statuses: Dict[str, OrderStatus]):
        """(terminal, pending) словари по терминальности."""
        terminal = {k: v for k, v in statuses.items() if v.is_terminal}
        pending = {k: v for k, v in statuses.items() if not v.is_terminal}
        return terminal, pending


# ---------------------------------------------------------------------------
# Startup reconciliation (after crash/restart)
# ---------------------------------------------------------------------------
@dataclass
class ReconcileReport:
    untracked_broker_orders: List[str] = field(default_factory=list)  # есть у брокера, нет локально
    missing_at_broker: List[str] = field(default_factory=list)  # локально pending, нет у брокера
    position_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.untracked_broker_orders or self.missing_at_broker or self.position_mismatches
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean": self.clean,
            "untracked_broker_orders": list(self.untracked_broker_orders),
            "missing_at_broker": list(self.missing_at_broker),
            "position_mismatches": list(self.position_mismatches),
            "actions": list(self.actions),
        }


class StartupReconciler:
    """Сверка локального состояния (journal) с брокером при старте после сбоя."""

    def __init__(self, *, qty_tolerance: float = 1e-6) -> None:
        self.qty_tolerance = float(qty_tolerance)

    def reconcile(
        self,
        *,
        local_open_order_ids: Sequence[str],
        broker_open_order_ids: Sequence[str],
        local_positions: Dict[str, float],
        broker_positions: Dict[str, float],
    ) -> ReconcileReport:
        rep = ReconcileReport()
        local_set, broker_set = set(local_open_order_ids), set(broker_open_order_ids)

        for oid in sorted(broker_set - local_set):
            rep.untracked_broker_orders.append(oid)
            rep.actions.append(f"adopt-or-cancel untracked broker order {oid}")
        for oid in sorted(local_set - broker_set):
            rep.missing_at_broker.append(oid)
            rep.actions.append(f"mark/resubmit local order {oid} (not at broker)")

        syms = set(local_positions) | set(broker_positions)
        for s in sorted(syms):
            lq = float(local_positions.get(s, 0.0))
            bq = float(broker_positions.get(s, 0.0))
            if abs(lq - bq) > self.qty_tolerance:
                rep.position_mismatches.append(
                    {"symbol": s, "local": lq, "broker": bq, "delta": bq - lq}
                )
                rep.actions.append(f"reconcile position {s}: local={lq} broker={bq} (trust broker)")
        return rep


__all__ = [
    "CircuitOpenError",
    "MaxRetriesExceeded",
    "RetryPolicy",
    "CircuitState",
    "CircuitBreaker",
    "ResilientExecutor",
    "OrderStatus",
    "OrderStatusPoller",
    "ReconcileReport",
    "StartupReconciler",
]
