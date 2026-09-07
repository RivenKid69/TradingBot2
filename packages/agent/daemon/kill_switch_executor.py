# -*- coding: utf-8 -*-
"""
Kill Switch Executor - Executes real broker actions on kill switch trigger.

Design Doc Section 9.4:
- cancel open orders
- optional flatten (только если локально разрешено)
- halt run
- отправка телеметрии о причине

This module bridges KillSwitchManager with BrokerConnector to execute
real trading actions (cancel orders, flatten positions) when kill switch triggers.

AGENT ZONE ONLY - Never import in Cloud zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from packages.agent.daemon.kill_switch import (
    KillSwitchManager,
    KillSwitchConfig,
    HaltReason,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
    HaltEvent,
)

if TYPE_CHECKING:
    from packages.agent.broker.protocol import BrokerConnector, BulkCancelResult


logger = logging.getLogger(__name__)


@dataclass
class KillSwitchExecutorConfig:
    """
    Kill switch executor configuration.

    Design Doc Section 9.4: Kill Switch configuration.
    """

    # Action settings
    cancel_on_trigger: bool = True  # Always cancel open orders
    flatten_on_trigger: bool = False  # Only if local policy allows
    flatten_timeout_seconds: int = 30  # Timeout for flatten operation

    # Retry settings
    max_cancel_retries: int = 3
    max_flatten_retries: int = 2
    retry_delay_seconds: float = 1.0

    # Logging
    log_all_actions: bool = True


@dataclass
class ExecutionResult:
    """
    Result of kill switch execution.
    """

    success: bool
    orders_cancelled: int = 0
    orders_failed_to_cancel: int = 0
    positions_flattened: int = 0
    positions_failed_to_flatten: int = 0
    execution_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class KillSwitchExecutor:
    """
    Executes kill switch actions via BrokerConnector.

    Design Doc Section 9.4:
    - cancel open orders -> First priority
    - optional flatten -> Only if local policy allows
    - halt run -> Always
    - report reason in telemetry -> Always

    SECURITY INVARIANTS:
    1. Cancel orders is ALWAYS executed (cannot be disabled)
    2. Flatten is ONLY executed if local policy allows (config.flatten_on_trigger)
    3. Cloud CANNOT request flatten directly - only local trigger
    4. All actions are logged for audit trail

    Usage:
        executor = KillSwitchExecutor(broker_connector, config)

        # Register with KillSwitchManager
        kill_switch = KillSwitchManager(
            cancel_orders_fn=executor.cancel_all_orders,
            flatten_fn=executor.flatten_all_positions,
        )

        # Or use execute_halt directly
        result = executor.execute_halt(
            action=HaltAction.CANCEL_ORDERS,
            reason=halt_reason,
        )
    """

    def __init__(
        self,
        broker_connector: Optional["BrokerConnector"] = None,
        config: Optional[KillSwitchExecutorConfig] = None,
        on_action: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize kill switch executor.

        Args:
            broker_connector: Broker connector for order/position management
            config: Executor configuration
            on_action: Callback for action logging/telemetry
        """
        self._broker = broker_connector
        self._config = config or KillSwitchExecutorConfig()
        self._on_action = on_action

        # Execution tracking
        self._last_execution: Optional[ExecutionResult] = None
        self._execution_history: List[ExecutionResult] = []

    def set_broker_connector(self, connector: "BrokerConnector") -> None:
        """
        Set broker connector.

        Args:
            connector: BrokerConnector instance
        """
        self._broker = connector

    @property
    def has_broker(self) -> bool:
        """Check if broker connector is set."""
        return self._broker is not None and self._broker.is_connected

    def execute_halt(
        self,
        action: HaltAction,
        reason: HaltReason,
    ) -> ExecutionResult:
        """
        Execute kill switch halt action.

        This is the main entry point for executing halt actions.

        Args:
            action: Halt action to execute
            reason: Reason for halt

        Returns:
            ExecutionResult with action outcomes
        """
        start_time = datetime.utcnow()

        result = ExecutionResult(success=True)

        self._log_action(
            "halt_initiated",
            {
                "action": action.value,
                "reason_type": reason.reason_type.name,
                "severity": reason.severity.value,
                "message": reason.message,
            },
        )

        # Step 1: Cancel all open orders (ALWAYS, unless HALT_ONLY)
        if action != HaltAction.HALT_ONLY:
            cancel_result = self._execute_cancel_orders()
            result.orders_cancelled = cancel_result.orders_cancelled
            result.orders_failed_to_cancel = cancel_result.orders_failed_to_cancel
            result.errors.extend(cancel_result.errors)

            if cancel_result.orders_failed_to_cancel > 0:
                result.success = False

        # Step 2: Flatten positions (only if allowed by local policy AND requested)
        if action in (HaltAction.FLATTEN_LOCAL, HaltAction.FLATTEN_EMERGENCY):
            if self._config.flatten_on_trigger:
                flatten_result = self._execute_flatten_positions()
                result.positions_flattened = flatten_result.positions_flattened
                result.positions_failed_to_flatten = flatten_result.positions_failed_to_flatten
                result.errors.extend(flatten_result.errors)

                if flatten_result.positions_failed_to_flatten > 0:
                    result.success = False
            else:
                self._log_action(
                    "flatten_skipped",
                    {
                        "reason": "flatten_on_trigger is disabled in local policy",
                    },
                )

        # Calculate execution time
        end_time = datetime.utcnow()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Store result
        self._last_execution = result
        self._execution_history.append(result)

        # Trim history
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

        self._log_action(
            "halt_completed",
            {
                "success": result.success,
                "orders_cancelled": result.orders_cancelled,
                "positions_flattened": result.positions_flattened,
                "execution_time_ms": result.execution_time_ms,
                "errors": result.errors,
            },
        )

        return result

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        This method is designed to be passed to KillSwitchManager as cancel_orders_fn.

        Returns:
            Number of orders cancelled
        """
        result = self._execute_cancel_orders()
        return result.orders_cancelled

    def flatten_all_positions(self) -> bool:
        """
        Flatten all positions.

        This method is designed to be passed to KillSwitchManager as flatten_fn.

        Returns:
            True if all positions flattened successfully
        """
        if not self._config.flatten_on_trigger:
            logger.warning("Flatten requested but disabled in local policy")
            return False

        result = self._execute_flatten_positions()
        return result.positions_failed_to_flatten == 0

    def _execute_cancel_orders(self) -> ExecutionResult:
        """
        Execute cancel all orders with retries.

        Returns:
            ExecutionResult for cancel operation
        """
        result = ExecutionResult(success=True)

        if not self.has_broker:
            result.errors.append("No broker connector available")
            result.success = False
            return result

        self._log_action("cancel_orders_started", {})

        for attempt in range(self._config.max_cancel_retries):
            try:
                # Get open orders first
                open_orders = self._broker.get_open_orders()
                total_open = len(open_orders)

                if total_open == 0:
                    self._log_action("cancel_orders_none_open", {})
                    return result

                # Cancel all orders
                cancel_result = self._broker.cancel_all_orders()

                result.orders_cancelled = cancel_result.total_cancelled
                result.orders_failed_to_cancel = cancel_result.total_failed
                result.errors.extend(cancel_result.errors)

                self._log_action(
                    "cancel_orders_result",
                    {
                        "attempt": attempt + 1,
                        "total_open": total_open,
                        "cancelled": cancel_result.total_cancelled,
                        "failed": cancel_result.total_failed,
                    },
                )

                # Check if all cancelled
                if cancel_result.total_failed == 0:
                    result.success = True
                    return result

                # Retry if there were failures
                if attempt < self._config.max_cancel_retries - 1:
                    import time

                    time.sleep(self._config.retry_delay_seconds)

            except Exception as e:
                error_msg = f"Cancel orders attempt {attempt + 1} failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

                if attempt < self._config.max_cancel_retries - 1:
                    import time

                    time.sleep(self._config.retry_delay_seconds)

        result.success = result.orders_failed_to_cancel == 0
        return result

    def _execute_flatten_positions(self) -> ExecutionResult:
        """
        Execute flatten all positions with retries.

        Returns:
            ExecutionResult for flatten operation
        """
        result = ExecutionResult(success=True)

        if not self.has_broker:
            result.errors.append("No broker connector available")
            result.success = False
            return result

        self._log_action("flatten_positions_started", {})

        for attempt in range(self._config.max_flatten_retries):
            try:
                # Get all positions
                positions = self._broker.get_positions()

                if not positions:
                    self._log_action("flatten_positions_none_open", {})
                    return result

                # Close all positions
                close_results = self._broker.close_all_positions()

                successful = sum(1 for r in close_results if r.success)
                failed = sum(1 for r in close_results if not r.success)

                result.positions_flattened = successful
                result.positions_failed_to_flatten = failed

                for r in close_results:
                    if not r.success and r.error_message:
                        result.errors.append(r.error_message)

                self._log_action(
                    "flatten_positions_result",
                    {
                        "attempt": attempt + 1,
                        "total_positions": len(positions),
                        "flattened": successful,
                        "failed": failed,
                    },
                )

                # Check if all flattened
                if failed == 0:
                    result.success = True
                    return result

                # Retry if there were failures
                if attempt < self._config.max_flatten_retries - 1:
                    import time

                    time.sleep(self._config.retry_delay_seconds)

            except Exception as e:
                error_msg = f"Flatten positions attempt {attempt + 1} failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

                if attempt < self._config.max_flatten_retries - 1:
                    import time

                    time.sleep(self._config.retry_delay_seconds)

        result.success = result.positions_failed_to_flatten == 0
        return result

    def _log_action(self, action: str, details: Dict[str, Any]) -> None:
        """
        Log action for audit/telemetry.

        Args:
            action: Action name
            details: Action details
        """
        if self._config.log_all_actions:
            logger.info(f"KillSwitchExecutor: {action} - {details}")

        if self._on_action:
            try:
                self._on_action(action, details)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get executor status.

        Returns:
            Status dictionary
        """
        return {
            "has_broker": self.has_broker,
            "config": {
                "cancel_on_trigger": self._config.cancel_on_trigger,
                "flatten_on_trigger": self._config.flatten_on_trigger,
            },
            "last_execution": self._last_execution.details if self._last_execution else None,
            "execution_count": len(self._execution_history),
        }


def create_kill_switch_with_broker(
    broker_connector: Optional["BrokerConnector"],
    kill_switch_config: Optional[KillSwitchConfig] = None,
    executor_config: Optional[KillSwitchExecutorConfig] = None,
    on_trigger: Optional[Callable[[HaltReason], None]] = None,
    on_action: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> tuple[KillSwitchManager, KillSwitchExecutor]:
    """
    Factory function to create KillSwitchManager with BrokerConnector integration.

    Design Doc Section 9.4: Kill Switch with broker integration.

    Args:
        broker_connector: Broker connector for order/position management
        kill_switch_config: Kill switch configuration
        executor_config: Executor configuration
        on_trigger: Callback when kill switch triggers
        on_action: Callback for action logging

    Returns:
        (KillSwitchManager, KillSwitchExecutor) tuple

    Usage:
        from packages.agent.broker import BrokerConnectorFactory, BrokerCredentials

        # Create broker connector
        creds = BrokerCredentials(...)
        broker = BrokerConnectorFactory.create("alpaca", creds)
        broker.connect()

        # Create kill switch with broker integration
        kill_switch, executor = create_kill_switch_with_broker(
            broker_connector=broker,
            kill_switch_config=KillSwitchConfig(max_daily_loss_pct=Decimal("0.05")),
            executor_config=KillSwitchExecutorConfig(flatten_on_trigger=False),
        )

        # Kill switch will now actually cancel orders on trigger
        if kill_switch.check_daily_loss(daily_pnl, equity):
            kill_switch.trigger(reason, action=HaltAction.CANCEL_ORDERS)
    """
    executor = KillSwitchExecutor(
        broker_connector=broker_connector,
        config=executor_config,
        on_action=on_action,
    )

    kill_switch = KillSwitchManager(
        config=kill_switch_config,
        on_trigger=on_trigger,
        cancel_orders_fn=executor.cancel_all_orders,
        flatten_fn=executor.flatten_all_positions,
    )

    return kill_switch, executor
