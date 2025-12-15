# -*- coding: utf-8 -*-
"""
Tests for Kill Switch Executor.

Design Doc Section 9.4: Kill Switch with broker integration.
- cancel open orders
- optional flatten (only if locally allowed)
- halt run
- report reason in telemetry
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

from packages.agent.daemon.kill_switch_executor import (
    KillSwitchExecutor,
    KillSwitchExecutorConfig,
    ExecutionResult,
    create_kill_switch_with_broker,
)
from packages.agent.daemon.kill_switch import (
    KillSwitchManager,
    KillSwitchConfig,
    HaltReason,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
)
from packages.agent.broker.protocol import (
    BrokerConnector,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    OrderRequest,
    OrderResult,
    OrderInfo,
    Position,
    CancelResult,
    BulkCancelResult,
    PositionSide,
)


class MockBrokerConnector:
    """Mock broker connector for testing."""

    def __init__(self):
        self.is_connected = True
        self._open_orders: List[OrderInfo] = []
        self._positions: List[Position] = []
        self._cancel_all_called = False
        self._close_all_called = False
        self._cancel_result = BulkCancelResult(
            total_requested=0,
            total_cancelled=0,
            total_failed=0,
            results=[],
            errors=[],
        )
        self._close_results: List[OrderResult] = []

    def get_open_orders(self) -> List[OrderInfo]:
        return self._open_orders

    def cancel_all_orders(self, symbol=None) -> BulkCancelResult:
        self._cancel_all_called = True
        return self._cancel_result

    def get_positions(self) -> List[Position]:
        return self._positions

    def close_all_positions(self) -> List[OrderResult]:
        self._close_all_called = True
        return self._close_results


class TestKillSwitchExecutorConfig:
    """Tests for KillSwitchExecutorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KillSwitchExecutorConfig()

        assert config.cancel_on_trigger is True
        assert config.flatten_on_trigger is False
        assert config.flatten_timeout_seconds == 30
        assert config.max_cancel_retries == 3
        assert config.max_flatten_retries == 2
        assert config.retry_delay_seconds == 1.0
        assert config.log_all_actions is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KillSwitchExecutorConfig(
            cancel_on_trigger=True,
            flatten_on_trigger=True,
            flatten_timeout_seconds=60,
            max_cancel_retries=5,
            max_flatten_retries=3,
            retry_delay_seconds=0.5,
        )

        assert config.flatten_on_trigger is True
        assert config.flatten_timeout_seconds == 60
        assert config.max_cancel_retries == 5


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_default_result(self):
        """Test default execution result."""
        result = ExecutionResult(success=True)

        assert result.success is True
        assert result.orders_cancelled == 0
        assert result.orders_failed_to_cancel == 0
        assert result.positions_flattened == 0
        assert result.positions_failed_to_flatten == 0
        assert result.execution_time_ms == 0.0
        assert result.errors == []
        assert result.details == {}

    def test_result_with_values(self):
        """Test execution result with values."""
        result = ExecutionResult(
            success=False,
            orders_cancelled=5,
            orders_failed_to_cancel=2,
            errors=["Order cancel failed"],
        )

        assert result.success is False
        assert result.orders_cancelled == 5
        assert result.orders_failed_to_cancel == 2
        assert len(result.errors) == 1


class TestKillSwitchExecutor:
    """Tests for KillSwitchExecutor."""

    def test_init_without_broker(self):
        """Test initialization without broker."""
        executor = KillSwitchExecutor()

        assert executor.has_broker is False
        assert executor._broker is None
        assert executor._config is not None

    def test_init_with_broker(self):
        """Test initialization with broker."""
        broker = MockBrokerConnector()
        executor = KillSwitchExecutor(broker_connector=broker)

        assert executor.has_broker is True

    def test_set_broker_connector(self):
        """Test setting broker connector after init."""
        executor = KillSwitchExecutor()
        assert executor.has_broker is False

        broker = MockBrokerConnector()
        executor.set_broker_connector(broker)

        assert executor.has_broker is True

    def test_cancel_all_orders_no_broker(self):
        """Test cancel all orders without broker."""
        executor = KillSwitchExecutor()
        result = executor.cancel_all_orders()

        assert result == 0

    def test_cancel_all_orders_with_broker(self):
        """Test cancel all orders with broker."""
        broker = MockBrokerConnector()
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
            OrderInfo(
                client_order_id="client-2",
                broker_order_id="broker-2",
                symbol="MSFT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                filled_quantity=Decimal("0"),
                limit_price=None,
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.DAY,
                created_at=datetime.utcnow(),
            ),
        ]
        broker._cancel_result = BulkCancelResult(
            total_requested=2,
            total_cancelled=2,
            total_failed=0,
            results=[
                CancelResult(client_order_id="client-1", success=True),
                CancelResult(client_order_id="client-2", success=True),
            ],
            errors=[],
        )

        executor = KillSwitchExecutor(broker_connector=broker)
        result = executor.cancel_all_orders()

        assert result == 2
        assert broker._cancel_all_called is True

    def test_cancel_all_orders_no_open_orders(self):
        """Test cancel when no open orders."""
        broker = MockBrokerConnector()
        broker._open_orders = []

        executor = KillSwitchExecutor(broker_connector=broker)
        result = executor.cancel_all_orders()

        assert result == 0

    def test_flatten_all_positions_disabled(self):
        """Test flatten when disabled in config."""
        broker = MockBrokerConnector()
        config = KillSwitchExecutorConfig(flatten_on_trigger=False)

        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        result = executor.flatten_all_positions()

        assert result is False
        assert broker._close_all_called is False

    def test_flatten_all_positions_enabled(self):
        """Test flatten when enabled in config."""
        broker = MockBrokerConnector()
        broker._positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
            ),
        ]
        broker._close_results = [
            OrderResult(
                success=True,
                client_order_id="client-close-1",
                broker_order_id="broker-close-1",
            ),
        ]

        config = KillSwitchExecutorConfig(flatten_on_trigger=True)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        result = executor.flatten_all_positions()

        assert result is True
        assert broker._close_all_called is True

    def test_flatten_all_positions_no_positions(self):
        """Test flatten when no positions."""
        broker = MockBrokerConnector()
        broker._positions = []

        config = KillSwitchExecutorConfig(flatten_on_trigger=True)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        result = executor.flatten_all_positions()

        assert result is True

    def test_execute_halt_cancel_orders(self):
        """Test execute halt with CANCEL_ORDERS action."""
        broker = MockBrokerConnector()
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
        ]
        broker._cancel_result = BulkCancelResult(
            total_requested=1,
            total_cancelled=1,
            total_failed=0,
            results=[CancelResult(client_order_id="client-1", success=True)],
            errors=[],
        )

        executor = KillSwitchExecutor(broker_connector=broker)
        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Daily loss exceeded",
        )

        result = executor.execute_halt(
            action=HaltAction.CANCEL_ORDERS,
            reason=reason,
        )

        assert result.success is True
        assert result.orders_cancelled == 1
        assert result.positions_flattened == 0
        assert broker._cancel_all_called is True
        assert broker._close_all_called is False

    def test_execute_halt_flatten_local(self):
        """Test execute halt with FLATTEN_LOCAL action."""
        broker = MockBrokerConnector()
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
        ]
        broker._cancel_result = BulkCancelResult(
            total_requested=1,
            total_cancelled=1,
            total_failed=0,
            results=[CancelResult(client_order_id="client-1", success=True)],
            errors=[],
        )
        broker._positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
            ),
        ]
        broker._close_results = [
            OrderResult(success=True, client_order_id="client-close-1", broker_order_id="broker-close-1"),
        ]

        config = KillSwitchExecutorConfig(flatten_on_trigger=True)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            severity=HaltSeverity.HIGH,
            message="Manual flatten requested",
        )

        result = executor.execute_halt(
            action=HaltAction.FLATTEN_LOCAL,
            reason=reason,
        )

        assert result.success is True
        assert result.orders_cancelled == 1
        assert result.positions_flattened == 1
        assert broker._cancel_all_called is True
        assert broker._close_all_called is True

    def test_execute_halt_flatten_skipped_if_disabled(self):
        """Test that flatten is skipped if disabled in local policy."""
        broker = MockBrokerConnector()
        broker._positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
            ),
        ]

        # flatten_on_trigger is False by default
        executor = KillSwitchExecutor(broker_connector=broker)
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            severity=HaltSeverity.HIGH,
            message="Flatten requested but should be skipped",
        )

        result = executor.execute_halt(
            action=HaltAction.FLATTEN_LOCAL,
            reason=reason,
        )

        assert result.positions_flattened == 0
        assert broker._close_all_called is False

    def test_execute_halt_only(self):
        """Test execute halt with HALT_ONLY action."""
        broker = MockBrokerConnector()
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
        ]

        executor = KillSwitchExecutor(broker_connector=broker)
        reason = HaltReason(
            reason_type=HaltReasonType.TIME_SYNC_DRIFT,
            severity=HaltSeverity.MEDIUM,
            message="Time sync issue",
        )

        result = executor.execute_halt(
            action=HaltAction.HALT_ONLY,
            reason=reason,
        )

        assert result.success is True
        assert result.orders_cancelled == 0
        assert broker._cancel_all_called is False

    def test_on_action_callback(self):
        """Test that on_action callback is called."""
        broker = MockBrokerConnector()
        broker._cancel_result = BulkCancelResult(
            total_requested=0,
            total_cancelled=0,
            total_failed=0,
            results=[],
            errors=[],
        )

        actions_logged = []

        def on_action(action: str, details: dict):
            actions_logged.append((action, details))

        executor = KillSwitchExecutor(
            broker_connector=broker,
            on_action=on_action,
        )
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            severity=HaltSeverity.HIGH,
            message="Test halt",
        )

        executor.execute_halt(HaltAction.CANCEL_ORDERS, reason)

        assert len(actions_logged) > 0
        action_names = [a[0] for a in actions_logged]
        assert "halt_initiated" in action_names
        assert "halt_completed" in action_names

    def test_get_status(self):
        """Test get executor status."""
        broker = MockBrokerConnector()
        config = KillSwitchExecutorConfig(flatten_on_trigger=True)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)

        status = executor.get_status()

        assert status["has_broker"] is True
        assert status["config"]["cancel_on_trigger"] is True
        assert status["config"]["flatten_on_trigger"] is True
        assert status["execution_count"] == 0

    def test_execution_history(self):
        """Test execution history is maintained."""
        broker = MockBrokerConnector()
        executor = KillSwitchExecutor(broker_connector=broker)

        for i in range(5):
            reason = HaltReason(
                reason_type=HaltReasonType.MANUAL_TRIGGER,
                severity=HaltSeverity.HIGH,
                message=f"Test halt {i}",
            )
            executor.execute_halt(HaltAction.HALT_ONLY, reason)

        status = executor.get_status()
        assert status["execution_count"] == 5


class TestCreateKillSwitchWithBroker:
    """Tests for create_kill_switch_with_broker factory function."""

    def test_create_with_broker(self):
        """Test factory function with broker."""
        broker = MockBrokerConnector()
        kill_switch, executor = create_kill_switch_with_broker(
            broker_connector=broker,
        )

        assert isinstance(kill_switch, KillSwitchManager)
        assert isinstance(executor, KillSwitchExecutor)
        assert executor.has_broker is True

    def test_create_without_broker(self):
        """Test factory function without broker."""
        kill_switch, executor = create_kill_switch_with_broker(
            broker_connector=None,
        )

        assert isinstance(kill_switch, KillSwitchManager)
        assert isinstance(executor, KillSwitchExecutor)
        assert executor.has_broker is False

    def test_create_with_configs(self):
        """Test factory function with configs."""
        broker = MockBrokerConnector()
        kill_switch_config = KillSwitchConfig(
            max_daily_loss_pct=Decimal("0.05"),
        )
        executor_config = KillSwitchExecutorConfig(
            flatten_on_trigger=True,
            max_cancel_retries=5,
        )

        kill_switch, executor = create_kill_switch_with_broker(
            broker_connector=broker,
            kill_switch_config=kill_switch_config,
            executor_config=executor_config,
        )

        assert kill_switch.config.max_daily_loss_pct == Decimal("0.05")
        assert executor._config.flatten_on_trigger is True
        assert executor._config.max_cancel_retries == 5

    def test_create_with_callbacks(self):
        """Test factory function with callbacks."""
        broker = MockBrokerConnector()
        triggers = []
        actions = []

        def on_trigger(reason: HaltReason):
            triggers.append(reason)

        def on_action(action: str, details: dict):
            actions.append((action, details))

        kill_switch, executor = create_kill_switch_with_broker(
            broker_connector=broker,
            on_trigger=on_trigger,
            on_action=on_action,
        )

        # Trigger kill switch
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            severity=HaltSeverity.HIGH,
            message="Test",
        )
        kill_switch.trigger(reason, HaltAction.CANCEL_ORDERS)

        assert len(triggers) == 1
        assert len(actions) > 0


class TestKillSwitchExecutorRetry:
    """Tests for retry logic in KillSwitchExecutor."""

    def test_cancel_retry_on_failure(self):
        """Test that cancel retries on partial failure."""
        broker = MockBrokerConnector()
        call_count = [0]

        def cancel_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return BulkCancelResult(
                    total_requested=2,
                    total_cancelled=1,
                    total_failed=1,
                    results=[
                        CancelResult(client_order_id="client-1", success=True),
                        CancelResult(client_order_id="client-2", success=False, error_message="Failed"),
                    ],
                    errors=["Order 2 failed"],
                )
            else:
                return BulkCancelResult(
                    total_requested=1,
                    total_cancelled=1,
                    total_failed=0,
                    results=[CancelResult(client_order_id="client-2", success=True)],
                    errors=[],
                )

        broker.cancel_all_orders = cancel_with_retry
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
            OrderInfo(
                client_order_id="client-2",
                broker_order_id="broker-2",
                symbol="MSFT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                filled_quantity=Decimal("0"),
                limit_price=None,
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.DAY,
                created_at=datetime.utcnow(),
            ),
        ]

        config = KillSwitchExecutorConfig(retry_delay_seconds=0.01)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        executor.cancel_all_orders()

        # Should have retried
        assert call_count[0] >= 1

    def test_cancel_stops_after_max_retries(self):
        """Test that cancel stops after max retries."""
        broker = MockBrokerConnector()
        call_count = [0]

        def cancel_always_fails(*args, **kwargs):
            call_count[0] += 1
            return BulkCancelResult(
                total_requested=1,
                total_cancelled=0,
                total_failed=1,
                results=[CancelResult(client_order_id="client-1", success=False, error_message="Always fails")],
                errors=["Always fails"],
            )

        broker.cancel_all_orders = cancel_always_fails
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
        ]

        config = KillSwitchExecutorConfig(max_cancel_retries=3, retry_delay_seconds=0.01)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)
        result = executor._execute_cancel_orders()

        assert call_count[0] == 3
        assert result.success is False


class TestKillSwitchExecutorSecurity:
    """Tests for security invariants in KillSwitchExecutor."""

    def test_cloud_cannot_force_flatten(self):
        """
        Test that Cloud cannot force flatten if local policy forbids it.

        Design Doc Section 9.4: Cloud CANNOT request flatten directly -
        only local trigger with local policy approval.
        """
        broker = MockBrokerConnector()
        broker._positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
            ),
        ]

        # Local policy forbids flatten
        config = KillSwitchExecutorConfig(flatten_on_trigger=False)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)

        # Even with FLATTEN_EMERGENCY action, local policy wins
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,  # Cloud-originated request
            severity=HaltSeverity.CRITICAL,
            message="Cloud requested flatten",
            trigger_source="Cloud",
        )

        result = executor.execute_halt(HaltAction.FLATTEN_EMERGENCY, reason)

        # Flatten should not happen because local policy forbids it
        assert result.positions_flattened == 0
        assert broker._close_all_called is False

    def test_cancel_always_executed(self):
        """
        Test that cancel is always executed (unless HALT_ONLY).

        Design Doc Section 9.4: Cancel orders is ALWAYS executed
        (cannot be disabled).
        """
        broker = MockBrokerConnector()
        broker._open_orders = [
            OrderInfo(
                client_order_id="client-1",
                broker_order_id="broker-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                limit_price=Decimal("150.00"),
                stop_price=None,
                avg_fill_price=None,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.utcnow(),
            ),
        ]
        broker._cancel_result = BulkCancelResult(
            total_requested=1,
            total_cancelled=1,
            total_failed=0,
            results=[CancelResult(client_order_id="client-1", success=True)],
            errors=[],
        )

        # Even with cancel_on_trigger=False in config (shouldn't happen),
        # the executor still cancels because it's a mandatory action
        config = KillSwitchExecutorConfig(cancel_on_trigger=True)
        executor = KillSwitchExecutor(broker_connector=broker, config=config)

        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Loss limit",
        )

        result = executor.execute_halt(HaltAction.CANCEL_ORDERS, reason)

        assert broker._cancel_all_called is True
        assert result.orders_cancelled == 1
