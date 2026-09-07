# -*- coding: utf-8 -*-
"""Тесты авто-recovery исполнения (P1): retry/backoff, circuit-breaker, poll, reconcile."""

from __future__ import annotations

import pytest

from packages.agent.execution.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    MaxRetriesExceeded,
    OrderStatusPoller,
    ResilientExecutor,
    RetryPolicy,
    StartupReconciler,
)


# --- RetryPolicy ---
def test_retry_delay_exponential_capped():
    rp = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=10.0, jitter=0.0)
    assert rp.delay(1) == pytest.approx(1.0)
    assert rp.delay(2) == pytest.approx(2.0)
    assert rp.delay(3) == pytest.approx(4.0)
    assert rp.delay(10) == pytest.approx(10.0)  # capped


def test_retry_jitter_bounds():
    rp = RetryPolicy(base_delay=1.0, multiplier=1.0, jitter=0.1)
    assert rp.delay(1, rand=0.0) == pytest.approx(0.9)  # -10%
    assert rp.delay(1, rand=1.0) == pytest.approx(1.1)  # +10%


# --- CircuitBreaker ---
def test_breaker_opens_and_half_opens():
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, time_fn=lambda: clock["t"])
    assert cb.allow() and cb.state == CircuitState.CLOSED
    for _ in range(3):
        cb.record_attempt()
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False
    clock["t"] = 31.0  # после cooldown
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow() is True  # одна пробная
    cb.record_attempt()
    assert cb.allow() is False  # half_open_max_calls=1 исчерпан
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_breaker_halfopen_failure_reopens():
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0, time_fn=lambda: clock["t"])
    cb.record_attempt()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    clock["t"] = 11.0
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_attempt()
    cb.record_failure()  # провал в half-open → снова OPEN
    assert cb.state == CircuitState.OPEN


# --- ResilientExecutor ---
def test_executor_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("broker down")
        return "ok"

    ex = ResilientExecutor(
        retry=RetryPolicy(max_attempts=5, base_delay=1.0), sleep_fn=lambda d: None
    )
    assert ex.call(flaky) == "ok"
    assert calls["n"] == 3
    assert ex.sleeps == [1.0, 2.0]  # backoff между 2 ретраями (без реального сна)


def test_executor_raises_after_max():
    def always_fail():
        raise TimeoutError("nope")

    ex = ResilientExecutor(
        retry=RetryPolicy(max_attempts=3, base_delay=0.1), sleep_fn=lambda d: None
    )
    with pytest.raises(MaxRetriesExceeded):
        ex.call(always_fail)


def test_executor_circuit_open_rejects():
    cb = CircuitBreaker(failure_threshold=2)
    ex = ResilientExecutor(
        retry=RetryPolicy(max_attempts=10, base_delay=0.0), breaker=cb, sleep_fn=lambda d: None
    )

    def always_fail():
        raise ConnectionError("x")

    with pytest.raises(CircuitOpenError):
        ex.call(always_fail)  # 2 провала → OPEN → отклонение
    assert cb.state == CircuitState.OPEN


# --- OrderStatusPoller ---
def test_poller_classifies_terminal_and_filled():
    db = {
        "o1": {"status": "FILLED", "filled_qty": 100},
        "o2": {"status": "NEW"},
        "o3": {"status": "PARTIALLY_FILLED", "filled": 40},
    }
    poller = OrderStatusPoller(lambda oid: db[oid])
    st = poller.poll(["o1", "o2", "o3"])
    assert st["o1"].is_terminal and st["o1"].is_filled and st["o1"].filled_qty == 100
    assert not st["o2"].is_terminal
    assert st["o3"].filled_qty == 40 and not st["o3"].is_terminal
    terminal, pending = poller.split_terminal(st)
    assert set(terminal) == {"o1"} and set(pending) == {"o2", "o3"}


# --- StartupReconciler ---
def test_reconcile_detects_all_divergences():
    rec = StartupReconciler(qty_tolerance=1e-6)
    rep = rec.reconcile(
        local_open_order_ids=["a", "b"],
        broker_open_order_ids=["b", "c"],  # c untracked, a missing-at-broker
        local_positions={"AAPL": 100, "MSFT": 50},
        broker_positions={"AAPL": 100, "MSFT": 60, "TSLA": 10},
    )
    assert rep.untracked_broker_orders == ["c"]
    assert rep.missing_at_broker == ["a"]
    syms = {m["symbol"] for m in rep.position_mismatches}
    assert syms == {"MSFT", "TSLA"}  # AAPL совпадает
    assert rep.clean is False
    assert len(rep.actions) >= 4


def test_reconcile_clean_when_matched():
    rec = StartupReconciler()
    rep = rec.reconcile(
        local_open_order_ids=["a"],
        broker_open_order_ids=["a"],
        local_positions={"AAPL": 100},
        broker_positions={"AAPL": 100},
    )
    assert rep.clean is True and rep.actions == []
