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

import hashlib
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
from packages.agent.reconciliation.journal import OrderJournal, JournalStatus


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
class PriceCollarConfig:
    """Fat-finger / price-collar pre-trade gate (P1 #10).

    A last-line sanity check BEFORE submission, independent of the risk stack:
      * ``max_price_distance_pct`` — reject a limit order whose price is more than
        this fraction away from the reference (last/mid) price (fat-finger price);
      * ``max_notional`` — reject an order whose notional exceeds this absolute cap;
      * ``max_adv_participation`` + ``adv_provider`` — reject if the order would be a
        larger fraction of the name's ADV than this (oversized-vs-liquidity).
    Any ``None`` field disables that check. Enabled by default with wide bounds.
    """

    enabled: bool = True
    max_price_distance_pct: Optional[float] = 0.10  # 10% from reference
    max_notional: Optional[float] = None
    max_adv_participation: Optional[float] = None
    adv_provider: Optional[Callable[[str], Optional[float]]] = None


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
    CLOUD INJECTION PROTECTION - Intents must originate from local strategy execution.

    Flow:
    1. Receive OrderIntent from strategy (LOCAL ONLY)
    2. Validate intent origin (REJECT if from Cloud)
    3. Validate against Policy Firewall
    4. Validate against Hard Caps
    5. Run pre-trade risk checks
    6. Convert to Order
    7. Submit to broker via connector
    8. Track status and report

    SECURITY:
    - Cloud CANNOT create or modify OrderIntents
    - Cloud CANNOT submit orders directly
    - All intents must pass through local policy stack
    - All orders are logged in durable journal BEFORE submission

    IDEMPOTENCY (Design Doc Phase 8 WI-AGENT-06):
    - client_order_id is computed deterministically from stable identifiers
    - deployment_id: Fixed per deployment, survives restarts
    - run_id: Identifies the current run, increments on restart
    - sequence: Monotonic counter per run, recovered from journal on restart
    - This ensures the same logical order gets the same client_order_id

    Usage:
        engine = LiveExecutionEngine(
            policy_firewall=firewall,
            broker_submit=broker.submit_order,
            deployment_id="deploy_123",
            run_id="run_456",
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
        order_journal: Optional[OrderJournal] = None,
        deployment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        broker_cancel: Optional[Callable[[str, Optional[str]], bool]] = None,
        broker_replace: Optional[
            Callable[[str, Optional[Decimal], Optional[Decimal]], tuple]
        ] = None,
        price_collar: Optional[PriceCollarConfig] = None,
    ):
        """
        Initialize execution engine.

        Args:
            policy_firewall: Policy validation
            hard_cap_enforcer: Hard cap validation
            risk_checker: Pre-trade risk checks
            broker_submit: Function to submit orders to broker
            broker_name: Name of broker for orders
            order_journal: Durable order journal
            deployment_id: Stable deployment identifier (survives restarts)
            run_id: Run identifier (increments on restart)
        """
        self._policy = policy_firewall or PolicyFirewall()
        self._hard_caps = hard_cap_enforcer or HardCapEnforcer()
        self._risk_checker = risk_checker or RiskChecker()
        self._broker_submit = broker_submit
        self._broker_cancel = broker_cancel
        self._broker_replace = broker_replace
        self._price_collar = price_collar or PriceCollarConfig()
        self._broker_name = broker_name

        # Idempotency identifiers (Design Doc Phase 8 WI-AGENT-06)
        self._deployment_id = deployment_id or str(uuid4())
        self._run_id = run_id or str(uuid4())
        self._sequence: int = 0  # Monotonic counter per run

        # Order tracking
        self._orders: Dict[UUID, Order] = {}
        self._orders_by_client_id: Dict[str, Order] = {}
        self._journal = order_journal or OrderJournal()
        self._journal_entry_by_client_id: Dict[str, str] = {}

        # Portfolio state (should be updated from broker)
        self._portfolio = PortfolioState()

        # Recover sequence from journal on restart
        self._recover_sequence_from_journal()

    def _recover_sequence_from_journal(self) -> None:
        """
        Recover sequence counter from journal on restart.

        Design Doc Phase 8 WI-AGENT-06:
        On restart, we need to find the highest sequence number used
        in the current run to continue from there.
        """
        # Get all entries for this deployment and run
        entries = self._journal.get_all_entries()
        max_seq = 0

        for entry in entries:
            # Check if entry belongs to this deployment/run
            metadata = entry.metadata or {}
            entry_deployment = metadata.get("deployment_id", "")
            entry_run = metadata.get("run_id", "")

            if entry_deployment == self._deployment_id and entry_run == self._run_id:
                entry_seq = metadata.get("sequence", 0)
                if isinstance(entry_seq, int) and entry_seq > max_seq:
                    max_seq = entry_seq

        self._sequence = max_seq

    def _compute_client_order_id(self, intent: OrderIntent) -> str:
        """
        Compute a deterministic client_order_id.

        Design Doc Phase 8 WI-AGENT-06: stable across retries/restarts.

        The client_order_id is computed from:
        - broker_name: Identifies the broker
        - strategy_id: Identifies the strategy
        - intent_id: Unique identifier for the intent (stable for same intent object)

        This ensures:
        1. Same intent -> same client_order_id (idempotency/duplicate detection)
        2. After restart, same intent produces same ID (journal lookup works)
        3. Different strategies produce different IDs

        NOTE: deployment_id and run_id are NOT included in client_order_id.
        They are tracked in journal metadata for audit/monitoring purposes.
        This allows the same intent to be detected as duplicate across restarts.
        """
        # Build deterministic client_order_id from stable identifiers
        # Using intent_id ensures same intent object -> same client_order_id
        raw = f"ccea|{self._broker_name}|{intent.strategy_id}|{intent.intent_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"ccea_{digest}"

    def _record_sequence_for_order(self) -> int:
        """
        Record sequence number for the current order.

        This is called AFTER client_order_id is computed and order is accepted.
        Sequence is for audit/monitoring, not for ID computation.

        Returns:
            The sequence number for this order
        """
        self._sequence += 1
        return self._sequence

    def execute(
        self,
        intent: OrderIntent,
        current_price: Optional[Decimal] = None,
        origin: str = "local",
    ) -> ExecutionResult:
        """
        Execute an OrderIntent.

        Args:
            intent: OrderIntent to execute
            current_price: Current market price
            origin: Origin of intent (must be "local" or "strategy")

        Returns:
            ExecutionResult with order status
        """
        # SECURITY: Reject intents from Cloud
        # Cloud can send commands, but NEVER intents directly
        if origin not in ("local", "strategy", "runner"):
            return ExecutionResult(
                success=False,
                error_message=f"Intent origin '{origin}' not allowed - Cloud injection blocked",
            )

        # SECURITY: Validate intent has local strategy_id
        if not intent.strategy_id:
            return ExecutionResult(
                success=False,
                error_message="Intent missing strategy_id - origin cannot be verified",
            )

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
        risk_result = self._risk_checker.check(intent, self._portfolio, current_price)
        if not risk_result.passed:
            failed = risk_result.failed_checks[0]
            return ExecutionResult(
                success=False,
                error_message=f"Risk check failed: {failed.message}",
            )

        # 4. Convert to order
        order = self._intent_to_order(intent, current_price)

        # 4b. Fat-finger / price-collar gate (P1 #10) — last sanity check pre-submit.
        collar_err = self._check_price_collar(order, current_price)
        if collar_err is not None:
            return ExecutionResult(
                success=False,
                order=order,
                error_message=f"Price-collar / fat-finger block: {collar_err}",
            )

        # 5. Check idempotency
        if self._journal.is_duplicate(order.client_order_id):
            existing = self._orders_by_client_id.get(order.client_order_id)
            if existing is not None:
                return ExecutionResult(
                    success=True,
                    order=existing,
                    error_message="Duplicate order - returning existing",
                )
            entry = self._journal.get_by_client_id(order.client_order_id)
            if entry:
                recovered = Order(
                    client_order_id=entry.client_order_id,
                    intent_id=UUID(entry.intent_id),
                    symbol=entry.symbol,
                    side=entry.side,
                    order_type=OrderType(entry.order_type),
                    quantity=Decimal(entry.quantity),
                    broker=self._broker_name,
                    broker_order_id=entry.broker_order_id,
                )
                self._orders[recovered.order_id] = recovered
                self._orders_by_client_id[recovered.client_order_id] = recovered
                return ExecutionResult(
                    success=True,
                    order=recovered,
                    error_message="Duplicate order - recovered from journal",
                )
            return ExecutionResult(
                success=True, error_message="Duplicate order - journal entry exists"
            )

        # 5b. Log before submission (durable)
        # Record sequence for this order (for audit/monitoring)
        order_sequence = self._record_sequence_for_order()

        # Include idempotency fields for recovery on restart (Design Doc Phase 8 WI-AGENT-06)
        entry = self._journal.log_order(
            client_order_id=order.client_order_id,
            intent_id=str(order.intent_id),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type.value,
            metadata={
                "strategy_id": intent.strategy_id,
                "intent_type": intent.intent_type.value,
                # Idempotency fields for sequence recovery
                "deployment_id": self._deployment_id,
                "run_id": self._run_id,
                "sequence": order_sequence,
            },
        )
        self._journal_entry_by_client_id[order.client_order_id] = entry.entry_id

        # 6. Submit to broker
        if self._broker_submit:
            success, broker_id, error = self._broker_submit(order)
            if success:
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.utcnow()
                order.broker_order_id = broker_id
                self._journal.update_status(
                    entry.entry_id,
                    JournalStatus.SUBMITTED,
                    broker_order_id=broker_id,
                )
            else:
                order.status = OrderStatus.ERROR
                order.error_message = error
                self._journal.update_status(entry.entry_id, JournalStatus.REJECTED)
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
        client_order_id = self._compute_client_order_id(intent)

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

    def _check_price_collar(self, order: Order, current_price: Optional[Decimal]) -> Optional[str]:
        """Return a rejection reason if the order trips the fat-finger / price-collar
        gate, else None. Reference price = order limit price's neighbourhood vs
        ``current_price``; notional and ADV-participation are absolute sanity caps."""
        c = self._price_collar
        if c is None or not c.enabled:
            return None
        try:
            ref = current_price if current_price is not None else order.limit_price
            # 1) limit price far from reference (fat-finger price)
            if (
                c.max_price_distance_pct is not None
                and order.limit_price is not None
                and ref is not None
                and float(ref) > 0
            ):
                dist = abs(float(order.limit_price) - float(ref)) / float(ref)
                if dist > float(c.max_price_distance_pct) + 1e-12:
                    return (
                        f"limit {order.limit_price} is {dist:.1%} from reference {ref} "
                        f"(> {c.max_price_distance_pct:.0%})"
                    )
            # 2) absolute notional cap (fat-finger size)
            px = order.limit_price if order.limit_price is not None else ref
            notional = abs(float(order.quantity) * float(px)) if px is not None else None
            if (
                c.max_notional is not None
                and notional is not None
                and notional > float(c.max_notional)
            ):
                return f"notional ${notional:,.0f} exceeds cap ${float(c.max_notional):,.0f}"
            # 3) ADV participation (oversized vs liquidity)
            if (
                c.max_adv_participation is not None
                and c.adv_provider is not None
                and notional is not None
            ):
                adv = c.adv_provider(order.symbol)
                if adv and float(adv) > 0:
                    part = notional / float(adv)
                    if part > float(c.max_adv_participation) + 1e-12:
                        return f"order is {part:.1%} of ADV (> {c.max_adv_participation:.0%})"
        except Exception:  # pragma: no cover - never let the gate crash execution
            return None
        return None

    def cancel_order(self, client_order_id: str) -> ExecutionResult:
        """Cancel a working order via the broker; advance the OMS + journal."""
        order = self._orders_by_client_id.get(client_order_id)
        if order is None:
            return ExecutionResult(success=False, error_message="order not found")
        if order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            return ExecutionResult(
                success=False,
                order=order,
                error_message=f"cannot cancel a {order.status.value} order",
            )
        if self._broker_cancel is None:
            return ExecutionResult(
                success=False, order=order, error_message="no broker_cancel wired"
            )
        try:
            ok = bool(self._broker_cancel(client_order_id, order.broker_order_id))
        except Exception as exc:  # pragma: no cover
            return ExecutionResult(
                success=False, order=order, error_message=f"broker cancel error: {exc}"
            )
        if ok:
            self.update_order_status(
                client_order_id, OrderStatus.CANCELLED, broker_order_id=order.broker_order_id
            )
        return ExecutionResult(
            success=ok, order=order, error_message=None if ok else "broker rejected cancel"
        )

    def replace_order(
        self,
        client_order_id: str,
        *,
        new_quantity: Optional[Decimal] = None,
        new_limit_price: Optional[Decimal] = None,
        current_price: Optional[Decimal] = None,
    ) -> ExecutionResult:
        """Amend a working order's quantity and/or limit price (FIX 35=G semantics).

        Runs the price-collar gate on the amended terms before sending."""
        order = self._orders_by_client_id.get(client_order_id)
        if order is None:
            return ExecutionResult(success=False, error_message="order not found")
        if order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            return ExecutionResult(
                success=False,
                order=order,
                error_message=f"cannot amend a {order.status.value} order",
            )
        amended = Order(
            client_order_id=order.client_order_id,
            intent_id=order.intent_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=(new_quantity if new_quantity is not None else order.quantity),
            limit_price=(new_limit_price if new_limit_price is not None else order.limit_price),
            broker=self._broker_name,
            broker_order_id=order.broker_order_id,
        )
        collar_err = self._check_price_collar(amended, current_price)
        if collar_err is not None:
            return ExecutionResult(
                success=False,
                order=order,
                error_message=f"Price-collar block on amend: {collar_err}",
            )
        if self._broker_replace is None:
            return ExecutionResult(
                success=False, order=order, error_message="no broker_replace wired"
            )
        try:
            ok, broker_id, err = self._broker_replace(
                client_order_id, new_quantity, new_limit_price
            )
        except Exception as exc:  # pragma: no cover
            return ExecutionResult(
                success=False, order=order, error_message=f"broker replace error: {exc}"
            )
        if ok:
            order.quantity = amended.quantity
            order.limit_price = amended.limit_price
            if broker_id:
                order.broker_order_id = broker_id
            self.update_order_status(
                client_order_id, OrderStatus.ACCEPTED, broker_order_id=order.broker_order_id
            )
        return ExecutionResult(
            success=ok,
            order=order,
            error_message=None if ok else (err or "broker rejected replace"),
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

        entry_id = self._journal_entry_by_client_id.get(client_order_id)
        if entry_id:
            if status in (
                OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                self._journal.update_status(
                    entry_id, JournalStatus.SUBMITTED, broker_order_id=broker_order_id
                )
            elif status in (OrderStatus.FILLED,):
                self._journal.update_status(
                    entry_id, JournalStatus.CONFIRMED, broker_order_id=broker_order_id
                )
            elif status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
                self._journal.update_status(
                    entry_id, JournalStatus.CANCELLED, broker_order_id=broker_order_id
                )
            elif status in (OrderStatus.REJECTED, OrderStatus.ERROR):
                self._journal.update_status(
                    entry_id, JournalStatus.REJECTED, broker_order_id=broker_order_id
                )

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
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED)
        ]

    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    # ===== Idempotency State (Design Doc Phase 8 WI-AGENT-06) =====

    @property
    def deployment_id(self) -> str:
        """Get deployment ID."""
        return self._deployment_id

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self._run_id

    @property
    def sequence(self) -> int:
        """Get current sequence number."""
        return self._sequence

    def get_idempotency_state(self) -> Dict[str, Any]:
        """
        Get current idempotency state for monitoring/debugging.

        Returns:
            Dictionary with deployment_id, run_id, sequence
        """
        return {
            "deployment_id": self._deployment_id,
            "run_id": self._run_id,
            "sequence": self._sequence,
            "broker_name": self._broker_name,
        }

    def set_deployment_id(self, deployment_id: str) -> None:
        """
        Set deployment ID.

        WARNING: Only use for initialization - changing this mid-run
        will break idempotency guarantees.
        """
        self._deployment_id = deployment_id

    def set_run_id(self, run_id: str) -> None:
        """
        Set run ID and reset sequence.

        Call this when starting a new run to ensure proper isolation.
        Sequence is reset to 0 for the new run.
        """
        self._run_id = run_id
        self._sequence = 0
        # Re-recover sequence in case there are existing entries for this run
        self._recover_sequence_from_journal()
