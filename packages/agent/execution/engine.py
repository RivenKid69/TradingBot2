# -*- coding: utf-8 -*-
"""
Live Execution Engine - AGENT ZONE ONLY.

Converts OrderIntents into actual orders and submits to brokers.
This module is PROHIBITED in Cloud zone.

Key Features:
- Intent to order conversion
- Policy validation before submission
- Broker connector integration
- Order tracking and status management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Callable
from uuid import UUID, uuid4

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
from packages.agent.policy.firewall import PolicyFirewall, PolicyResult
from packages.agent.policy.hard_caps import HardCapEnforcer
from packages.agent.policy.risk_checker import RiskChecker, PortfolioState


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class OrderType(str, Enum):
    """Order type for broker submission."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """
    Live order for broker submission.

    Created from OrderIntent after policy validation.
    """

    order_id: UUID = field(default_factory=uuid4)
    client_order_id: str = ""  # Deterministic for idempotency
    intent_id: UUID = field(default_factory=uuid4)

    # Order details
    symbol: str = ""
    side: str = "buy"  # buy, sell
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal("0")
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")

    # Broker info
    broker: str = ""
    broker_order_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": str(self.order_id),
            "client_order_id": self.client_order_id,
            "intent_id": str(self.intent_id),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "avg_fill_price": str(self.avg_fill_price) if self.avg_fill_price else None,
            "commission": str(self.commission),
            "broker": self.broker,
            "broker_order_id": self.broker_order_id,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "error_message": self.error_message,
        }


@dataclass
class ExecutionResult:
    """
    Result of execution attempt.
    """

    success: bool = False
    order: Optional[Order] = None
    policy_result: Optional[PolicyResult] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "policy_passed": self.policy_result.allowed if self.policy_result else None,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


# Type for broker submit function
BrokerSubmitFn = Callable[[Order], tuple[bool, Optional[str], Optional[str]]]


class LiveExecutionEngine:
    """
    Live Execution Engine - Converts intents to orders.

    AGENT ZONE ONLY - Never in Cloud.

    Flow:
    1. Receive OrderIntent from strategy
    2. Validate against Policy Firewall
    3. Validate against Hard Caps
    4. Run pre-trade risk checks
    5. Convert to Order
    6. Submit to broker via connector
    7. Track status and report

    Usage:
        engine = LiveExecutionEngine(
            policy_firewall=firewall,
            broker_submit=broker.submit_order,
        )

        result = engine.execute(intent)
        if result.success:
            print(f"Order submitted: {result.order.broker_order_id}")
    """

    def __init__(
        self,
        policy_firewall: Optional[PolicyFirewall] = None,
        hard_cap_enforcer: Optional[HardCapEnforcer] = None,
        risk_checker: Optional[RiskChecker] = None,
        broker_submit: Optional[BrokerSubmitFn] = None,
        broker_name: str = "default",
    ):
        """
        Initialize execution engine.

        Args:
            policy_firewall: Policy validation
            hard_cap_enforcer: Hard cap validation
            risk_checker: Pre-trade risk checks
            broker_submit: Function to submit orders to broker
            broker_name: Name of broker for orders
        """
        self._policy = policy_firewall or PolicyFirewall()
        self._hard_caps = hard_cap_enforcer or HardCapEnforcer()
        self._risk_checker = risk_checker or RiskChecker()
        self._broker_submit = broker_submit
        self._broker_name = broker_name

        # Order tracking
        self._orders: Dict[UUID, Order] = {}
        self._orders_by_client_id: Dict[str, Order] = {}

        # Portfolio state (should be updated from broker)
        self._portfolio = PortfolioState()

        # Counters for idempotency
        self._order_counter = 0

    def execute(
        self,
        intent: OrderIntent,
        current_price: Optional[Decimal] = None,
    ) -> ExecutionResult:
        """
        Execute an OrderIntent.

        Args:
            intent: OrderIntent to execute
            current_price: Current market price

        Returns:
            ExecutionResult with order status
        """
        # Skip passive intents
        if intent.is_passive:
            return ExecutionResult(
                success=True,
                error_message="Passive intent - no action required",
            )

        # 1. Policy validation
        current_position = self._portfolio.get_position(intent.symbol)
        policy_result = self._policy.check_intent(
            intent,
            current_position=current_position,
            current_exposure=self._portfolio.gross_exposure,
            account_equity=self._portfolio.equity,
        )

        if not policy_result.allowed:
            return ExecutionResult(
                success=False,
                policy_result=policy_result,
                error_message=f"Policy violation: {policy_result.violations[0].message}",
            )

        # 2. Hard cap validation
        quantity = intent.target_quantity or Decimal("0")
        cap_violation = self._hard_caps.check_order_size(quantity)
        if cap_violation:
            return ExecutionResult(
                success=False,
                error_message=f"Hard cap violation: {cap_violation.message}",
            )

        # 3. Pre-trade risk checks
        risk_result = self._risk_checker.check(
            intent, self._portfolio, current_price
        )
        if not risk_result.passed:
            failed = risk_result.failed_checks[0]
            return ExecutionResult(
                success=False,
                error_message=f"Risk check failed: {failed.message}",
            )

        # 4. Convert to order
        order = self._intent_to_order(intent, current_price)

        # 5. Check idempotency
        if order.client_order_id in self._orders_by_client_id:
            existing = self._orders_by_client_id[order.client_order_id]
            return ExecutionResult(
                success=True,
                order=existing,
                error_message="Duplicate order - returning existing",
            )

        # 6. Submit to broker
        if self._broker_submit:
            success, broker_id, error = self._broker_submit(order)
            if success:
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.utcnow()
                order.broker_order_id = broker_id
            else:
                order.status = OrderStatus.ERROR
                order.error_message = error
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=f"Broker submission failed: {error}",
                )
        else:
            # No broker connected - mark as pending
            order.status = OrderStatus.PENDING

        # 7. Track order
        self._orders[order.order_id] = order
        self._orders_by_client_id[order.client_order_id] = order

        # Record for rate limiting
        self._policy.record_order()
        self._hard_caps.record_order()

        return ExecutionResult(
            success=True,
            order=order,
            policy_result=policy_result,
        )

    def _intent_to_order(
        self,
        intent: OrderIntent,
        current_price: Optional[Decimal],
    ) -> Order:
        """Convert OrderIntent to Order."""
        # Generate deterministic client order ID
        self._order_counter += 1
        client_order_id = f"{intent.strategy_id}_{intent.intent_id}_{self._order_counter}"

        # Determine side
        side = "buy" if intent.side == IntentSide.LONG else "sell"

        # Determine order type
        if intent.is_limit:
            order_type = OrderType.LIMIT
        elif intent.is_stop:
            order_type = OrderType.STOP
        else:
            order_type = OrderType.MARKET

        # Determine quantity
        quantity = intent.target_quantity or Decimal("0")
        if quantity == 0 and intent.target_notional and current_price:
            quantity = intent.target_notional / current_price

        return Order(
            client_order_id=client_order_id,
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            time_in_force=intent.time_in_force,
            broker=self._broker_name,
        )

    def update_order_status(
        self,
        client_order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[Decimal] = None,
        avg_fill_price: Optional[Decimal] = None,
        broker_order_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Update order status from broker callback.

        Args:
            client_order_id: Client order ID
            status: New status
            filled_quantity: Filled quantity
            avg_fill_price: Average fill price
            broker_order_id: Broker's order ID
            error_message: Error message if any

        Returns:
            Updated order or None if not found
        """
        order = self._orders_by_client_id.get(client_order_id)
        if not order:
            return None

        order.status = status

        if filled_quantity is not None:
            order.filled_quantity = filled_quantity

        if avg_fill_price is not None:
            order.avg_fill_price = avg_fill_price

        if broker_order_id is not None:
            order.broker_order_id = broker_order_id

        if error_message is not None:
            order.error_message = error_message

        if status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()

        return order

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        """Update portfolio state."""
        self._portfolio = portfolio

    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        return self._orders_by_client_id.get(client_order_id)

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED)
        ]

    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]
