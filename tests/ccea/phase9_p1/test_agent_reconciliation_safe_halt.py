# -*- coding: utf-8 -*-
from __future__ import annotations

from decimal import Decimal

import pytest

from packages.agent.reconciliation.journal import OrderJournal
from packages.agent.reconciliation.reconciler import PositionReconciler
from packages.agent.runner.live import LiveRunner, LiveRunnerConfig
from packages.shared.contracts.config import ExecutionMode
from packages.shared.contracts.strategy import BaseStrategy, StrategyContext, StrategyResult


class NoopStrategy(BaseStrategy):
    _strategy_id = "noop"
    _version = "1.0.0"

    def on_data(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult()


def test_position_reconciler_no_broker_fn_halts():
    reconciler = PositionReconciler(
        local_positions={"BTCUSDT": Decimal("1")},
        fetch_broker_positions=None,
    )
    result = reconciler.reconcile()

    assert result.success is False
    assert result.halted is True
    assert result.halt_reason


def test_position_reconciler_significant_mismatch_halts():
    reconciler = PositionReconciler(
        local_positions={"BTCUSDT": Decimal("1")},
        fetch_broker_positions=lambda: {"BTCUSDT": Decimal("0")},
    )
    result = reconciler.reconcile()

    assert result.success is False
    assert result.halted is True
    assert "BTCUSDT" in (result.halt_reason or "")


def test_order_reconciliation_unresolved_without_fetcher_halts(tmp_path):
    journal = OrderJournal(db_path=tmp_path / "orders.db")
    journal.log_order(
        client_order_id="ccea_test_1",
        intent_id="intent-1",
        symbol="BTCUSDT",
        side="buy",
        quantity=Decimal("0.1"),
        order_type="market",
    )

    reconciler = PositionReconciler(
        local_positions={},
        fetch_broker_positions=lambda: {},
        journal=journal,
        fetch_broker_order_status=None,
    )
    result = reconciler.reconcile_orders()

    assert result.success is False
    assert result.halted is True
    assert "Unresolved orders" in (result.halt_reason or "")


def test_live_runner_start_safe_halts_when_positions_uncertain(tmp_path):
    config = LiveRunnerConfig(
        run_id="r1",
        strategy_id="noop",
        symbols=["BTCUSDT"],
        mode=ExecutionMode.LIVE,
        enable_reconciliation=True,
        order_journal_path=tmp_path / "orders.db",
        fetch_broker_positions_fn=None,  # uncertainty -> safe-halt
    )
    runner = LiveRunner(config)
    assert runner.initialize(NoopStrategy()) is True

    assert runner.start() is False
    assert runner.is_kill_switch_triggered() is True
    assert runner.get_kill_switch_reason()


def test_live_runner_restart_safe_halts_on_unresolved_orders(tmp_path):
    journal_path = tmp_path / "orders.db"
    journal = OrderJournal(db_path=journal_path)
    journal.log_order(
        client_order_id="ccea_test_2",
        intent_id="intent-2",
        symbol="BTCUSDT",
        side="buy",
        quantity=Decimal("0.1"),
        order_type="market",
    )
    journal.close()

    config = LiveRunnerConfig(
        run_id="r2",
        strategy_id="noop",
        symbols=["BTCUSDT"],
        mode=ExecutionMode.LIVE,
        enable_reconciliation=True,
        order_journal_path=journal_path,
        fetch_broker_positions_fn=lambda: {},  # positions ok
        fetch_broker_order_status_fn=None,  # but unresolved orders -> safe-halt
    )
    runner = LiveRunner(config)
    assert runner.initialize(NoopStrategy()) is True

    assert runner.start() is False
    assert runner.is_kill_switch_triggered() is True
    assert "Unresolved orders" in (runner.get_kill_switch_reason() or "")
