# -*- coding: utf-8 -*-
"""
Live Runner - AGENT ZONE ONLY.

Runs strategies with real order execution through LiveExecutionEngine.
Enforces all risk controls before every order.

Key Features:
- Processes OrderIntents through full risk stack
- Policy Firewall validation
- Hard Cap enforcement
- Pre-trade risk checks
- Order execution via broker connectors
- Position reconciliation
- Kill switch support

IMPORTANT:
    This module is PROHIBITED in Cloud zone.
    All orders go through full risk validation.
    Cloud cannot override local hard caps.

Flow:
    Strategy -> OrderIntent -> PolicyFirewall -> HardCapEnforcer
    -> RiskChecker -> LiveExecutionEngine -> Order -> Broker
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from packages.shared.contracts.intent import OrderIntent, IntentList, IntentType, IntentSide
from packages.shared.contracts.strategy import (
    StrategyContract,
    StrategyContext,
    StrategyResult,
    MarketSnapshot,
)
from packages.shared.contracts.config import ExecutionMode, RiskConfig
from packages.shared.runner.base import (
    BaseRunner,
    RunnerConfig,
    RunnerResult,
    RunnerState,
    RunnerZone,
    IntentProcessingResult,
)
from packages.agent.execution.engine import (
    LiveExecutionEngine,
    ExecutionResult,
    Order,
    OrderStatus,
    BrokerSubmitFn,
)
from packages.agent.reconciliation.journal import OrderJournal
from packages.agent.reconciliation.reconciler import (
    PositionReconciler,
    MismatchAction,
    FetchPositionsFn,
    FetchOrderStatusFn,
)
from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig, PolicyResult
from packages.agent.policy.hard_caps import HardCapEnforcer, HardCaps, HardCapViolation
from packages.agent.policy.risk_checker import RiskChecker, RiskCheckResult, PortfolioState


@dataclass
class LiveRunnerConfig(RunnerConfig):
    """
    Configuration for live runner.

    Extends base config with live execution settings.
    Agent zone ONLY.
    """

    # Risk configuration (from Cloud, validated against local hard caps)
    risk_config: Optional[RiskConfig] = None

    # Local policy (has priority over Cloud)
    local_policy: Optional[PolicyConfig] = None

    # Hard caps (NEVER overridable by Cloud)
    hard_caps: Optional[HardCaps] = None

    # Broker settings
    broker_name: str = "default"
    broker_submit_fn: Optional[BrokerSubmitFn] = None
    fetch_broker_positions_fn: Optional[FetchPositionsFn] = None
    fetch_broker_order_status_fn: Optional[FetchOrderStatusFn] = None

    # Persistent order journal
    order_journal_path: Optional[Path] = None

    # Kill switch settings
    enable_kill_switch: bool = True
    kill_switch_callback: Optional[Callable[[str], None]] = None

    # Reconciliation
    enable_reconciliation: bool = True
    reconciliation_interval_seconds: int = 60

    # Approval settings (for TRADING_IMPACTING changes)
    require_local_approval: bool = True
    auto_approve_threshold: Decimal = Decimal("0")  # 0 = never auto-approve

    def __post_init__(self):
        """Ensure zone is AGENT and mode is LIVE."""
        self.zone = RunnerZone.AGENT
        if self.mode == ExecutionMode.BACKTEST:
            self.mode = ExecutionMode.PAPER  # Never backtest in live runner

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "risk_config": self.risk_config.to_dict() if self.risk_config else None,
            "local_policy": self.local_policy.to_dict() if self.local_policy else None,
            "hard_caps": self.hard_caps.to_dict() if self.hard_caps else None,
            "broker_name": self.broker_name,
            "order_journal_path": str(self.order_journal_path) if self.order_journal_path else None,
            "enable_kill_switch": self.enable_kill_switch,
            "enable_reconciliation": self.enable_reconciliation,
            "require_local_approval": self.require_local_approval,
        })
        return base


class LiveRunner(BaseRunner):
    """
    Live Strategy Runner - AGENT ZONE ONLY.

    Executes strategies with full risk controls and real order submission.
    NEVER deployed in Cloud zone.

    Security Design Commitments:
    - All intents pass through PolicyFirewall
    - All intents validated against HardCaps
    - All intents pass pre-trade RiskChecker
    - Cloud cannot override local hard caps
    - Kill switch triggers on critical violations

    Usage:
        config = LiveRunnerConfig(
            run_id="live_001",
            strategy_id="momentum",
            symbols=["BTCUSDT"],
            mode=ExecutionMode.LIVE,
            broker_submit_fn=broker.submit_order,
        )

        runner = LiveRunner(config)
        runner.initialize(strategy)
        runner.start()

        # Process market data
        for tick in live_data_feed:
            result = runner.process_tick(tick)

        runner.stop()
    """

    def __init__(self, config: LiveRunnerConfig):
        """
        Initialize live runner.

        Args:
            config: Live runner configuration
        """
        # Ensure zone is AGENT
        config.zone = RunnerZone.AGENT

        super().__init__(config)
        self._live_config = config if isinstance(config, LiveRunnerConfig) else LiveRunnerConfig(**config.to_dict())

        # Initialize risk components
        self._policy_firewall = PolicyFirewall(
            local_policy=self._live_config.local_policy or PolicyConfig(),
            cloud_config=self._live_config.risk_config,
        )
        self._hard_cap_enforcer = HardCapEnforcer(
            hard_caps=self._live_config.hard_caps or HardCaps(),
        )
        self._risk_checker = RiskChecker()

        # Persistent order journal shared with execution + reconciliation
        self._order_journal = OrderJournal(db_path=self._live_config.order_journal_path)

        # Initialize execution engine
        self._execution_engine = LiveExecutionEngine(
            policy_firewall=self._policy_firewall,
            hard_cap_enforcer=self._hard_cap_enforcer,
            risk_checker=self._risk_checker,
            broker_submit=self._live_config.broker_submit_fn,
            broker_name=self._live_config.broker_name,
            order_journal=self._order_journal,
        )

        # Strategy
        self._strategy: Optional[StrategyContract] = None

        # Portfolio state
        self._portfolio = PortfolioState(
            equity=self._config.initial_capital,
            buying_power=self._config.initial_capital,
        )

        # Reconciliation (safe-halt on uncertainty)
        self._reconciler = PositionReconciler(
            local_positions=dict(self._portfolio.positions),
            fetch_broker_positions=self._live_config.fetch_broker_positions_fn,
            journal=self._order_journal,
            fetch_broker_order_status=self._live_config.fetch_broker_order_status_fn,
            mismatch_action=MismatchAction.HALT,
        )
        self._last_reconcile_ts: Optional[float] = None

        # Tracking
        self._orders: List[Order] = []
        self._tick_count = 0
        self._kill_switch_triggered = False
        self._kill_switch_reason: Optional[str] = None

        # Prices
        self._prices: Dict[str, Decimal] = {}

    def _maybe_reconcile(self, now_ts: float, force: bool = False) -> None:
        """
        Run reconciliation on start/periodic.

        Safety property (Phase 9 / WI-AGENT-05): on any uncertainty we safe-halt.
        """
        if not self._live_config.enable_reconciliation:
            return

        if not force and self._last_reconcile_ts is not None:
            interval = max(0, int(self._live_config.reconciliation_interval_seconds))
            if interval > 0 and (now_ts - self._last_reconcile_ts) < interval:
                return

        # Keep local positions current
        self._reconciler.update_local_positions(dict(self._portfolio.positions))

        # Position reconciliation is mandatory only for LIVE runs; for PAPER runs it is best-effort.
        if self._config.mode == ExecutionMode.LIVE or self._live_config.fetch_broker_positions_fn is not None:
            pos_result = self._reconciler.reconcile()
            if not pos_result.success or pos_result.halted:
                self._trigger_kill_switch(f"Reconciliation failed: {pos_result.halt_reason}")
                return

        # Order reconciliation is mandatory only for LIVE runs; for PAPER runs it is best-effort.
        if self._config.mode == ExecutionMode.LIVE or self._live_config.fetch_broker_order_status_fn is not None:
            order_result = self._reconciler.reconcile_orders()
            if not order_result.success or order_result.halted:
                self._trigger_kill_switch(f"Reconciliation failed: {order_result.halt_reason}")
                return

        self._last_reconcile_ts = now_ts

    def initialize(self, strategy: StrategyContract) -> bool:
        """
        Initialize runner with strategy.

        Args:
            strategy: Strategy to run

        Returns:
            True if initialization successful
        """
        try:
            self._state = RunnerState.INITIALIZING

            # Store strategy
            self._strategy = strategy

            # Initialize strategy
            strategy.initialize({
                "symbols": self._config.symbols,
                "mode": self._config.mode.value,
                "run_id": self._config.run_id,
                "is_live": self._config.mode == ExecutionMode.LIVE,
            })

            # Update result
            self._result.start_time = datetime.utcnow()
            self._result.initial_capital = self._config.initial_capital

            self._state = RunnerState.IDLE
            return True

        except Exception as e:
            self._state = RunnerState.ERROR
            self._result.errors.append(f"Initialization failed: {str(e)}")
            return False

    def process_tick(self, market_data: Dict[str, Any]) -> StrategyResult:
        """
        Process a single market data tick.

        Args:
            market_data: Current market data

        Returns:
            Strategy result with intents
        """
        # Check kill switch
        if self._kill_switch_triggered:
            return StrategyResult(
                warnings=[f"Kill switch triggered: {self._kill_switch_reason}"]
            )

        if self._state != RunnerState.RUNNING:
            return StrategyResult()

        if self._strategy is None:
            return StrategyResult()

        # Periodic reconciliation before any strategy execution/trading decisions
        self._maybe_reconcile(time.time(), force=False)
        if self._kill_switch_triggered:
            return StrategyResult(
                warnings=[f"Kill switch triggered: {self._kill_switch_reason}"]
            )

        self._tick_count += 1

        # Update prices
        symbol = market_data.get("symbol", self._config.symbols[0] if self._config.symbols else "UNKNOWN")
        price = Decimal(str(market_data.get("close", market_data.get("last", 0))))
        self._prices[symbol] = price

        # Update portfolio state with position info
        market_data["position_qty"] = str(self._portfolio.get_position(symbol))
        market_data["position_avg_price"] = str(self._portfolio.position_values.get(symbol, Decimal("0")) / self._portfolio.get_position(symbol)) if self._portfolio.get_position(symbol) else None

        # Create context
        context = self._create_context(market_data)

        # Get strategy decision
        start_time = time.time()
        strategy_result = self._strategy.on_data(context)
        execution_time_ms = (time.time() - start_time) * 1000
        strategy_result.execution_time_ms = execution_time_ms

        # Process intents
        if strategy_result.has_intents:
            self._result.intents_received += len(strategy_result.intents)
            intent_results = self.process_intents(strategy_result.intents)
            self._result.intent_results.extend(intent_results)

        # Update stats
        self._result.ticks_processed = self._tick_count

        return strategy_result

    def process_intents(self, intents: IntentList) -> List[IntentProcessingResult]:
        """
        Process OrderIntents through full risk stack.

        Args:
            intents: List of OrderIntents to process

        Returns:
            List of processing results
        """
        results = []

        for intent in intents:
            result = IntentProcessingResult(intent_id=str(intent.intent_id))

            try:
                # Skip passive intents
                if intent.is_passive:
                    result.processed = True
                    result.fill_info = {"status": "skipped", "reason": "passive_intent"}
                    results.append(result)
                    continue

                # Get current price
                current_price = self._prices.get(intent.symbol)

                # Execute through engine (includes all risk checks)
                exec_result = self._execution_engine.execute(intent, current_price)

                if exec_result.success:
                    if exec_result.order:
                        self._orders.append(exec_result.order)
                        result.processed = True
                        result.fill_info = exec_result.order.to_dict()
                        self._result.fills_count += 1
                        self._result.intents_processed += 1

                        # Notify strategy of order
                        if self._strategy:
                            self._strategy.on_fill({
                                "order_id": str(exec_result.order.order_id),
                                "client_order_id": exec_result.order.client_order_id,
                                "symbol": exec_result.order.symbol,
                                "status": exec_result.order.status.value,
                            })

                        # Callback if provided
                        if self._config.on_fill_callback:
                            self._config.on_fill_callback(exec_result.order.to_dict())
                    else:
                        result.processed = True
                        result.fill_info = {"status": "no_order", "message": exec_result.error_message}
                else:
                    result.processed = False
                    result.error = exec_result.error_message

                    # Check for kill switch violations
                    if exec_result.policy_result and not exec_result.policy_result.allowed:
                        for violation in exec_result.policy_result.violations:
                            if violation.severity == "critical":
                                self._trigger_kill_switch(violation.message)
                                break

            except Exception as e:
                result.error = str(e)
                self._result.errors.append(f"Intent processing error: {str(e)}")

                # Error callback if provided
                if self._config.on_error_callback:
                    self._config.on_error_callback(str(e))

                # Track errors for kill switch
                self._hard_cap_enforcer.record_error()
                error_violation = self._hard_cap_enforcer.check_error_count(
                    self._hard_cap_enforcer._errors_today
                )
                if error_violation:
                    self._trigger_kill_switch(error_violation.message)

            results.append(result)

        return results

    def start(self) -> bool:
        """
        Start the runner.

        Returns:
            True if started successfully
        """
        if self._state not in (RunnerState.IDLE, RunnerState.STOPPED):
            return False

        if self._strategy is None:
            self._result.errors.append("Cannot start: no strategy initialized")
            return False

        if self._kill_switch_triggered:
            self._result.errors.append(f"Cannot start: kill switch triggered ({self._kill_switch_reason})")
            return False

        # Reconcile on start (safe-halt on uncertainty)
        self._maybe_reconcile(time.time(), force=True)
        if self._kill_switch_triggered:
            self._result.errors.append(f"Cannot start: kill switch triggered ({self._kill_switch_reason})")
            return False

        self._state = RunnerState.RUNNING
        self._result.start_time = datetime.utcnow()
        return True

    def stop(self, reason: str = "user_requested") -> RunnerResult:
        """
        Stop the runner.

        Args:
            reason: Reason for stopping

        Returns:
            Final runner result
        """
        if self._state == RunnerState.STOPPED:
            return self._result

        self._state = RunnerState.STOPPING

        # Shutdown strategy
        if self._strategy:
            try:
                self._strategy.shutdown()
            except Exception as e:
                self._result.errors.append(f"Strategy shutdown error: {str(e)}")

        # Finalize result
        self._result.end_time = datetime.utcnow()
        self._result.duration_ms = int(
            (self._result.end_time - self._result.start_time).total_seconds() * 1000
            if self._result.start_time else 0
        )
        self._result.final_equity = self._portfolio.equity
        self._result.total_pnl = self._portfolio.daily_pnl
        self._result.success = len(self._result.errors) == 0 and not self._kill_switch_triggered
        self._result.state = RunnerState.STOPPED

        self._state = RunnerState.STOPPED
        return self._result

    def pause(self) -> bool:
        """
        Pause the runner.

        Returns:
            True if paused successfully
        """
        if self._state != RunnerState.RUNNING:
            return False

        self._state = RunnerState.PAUSED
        return True

    def resume(self) -> bool:
        """
        Resume paused runner.

        Returns:
            True if resumed successfully
        """
        if self._state != RunnerState.PAUSED:
            return False

        if self._kill_switch_triggered:
            return False

        self._state = RunnerState.RUNNING
        return True

    def _trigger_kill_switch(self, reason: str) -> None:
        """
        Trigger kill switch.

        Args:
            reason: Reason for triggering
        """
        if self._kill_switch_triggered:
            return

        self._kill_switch_triggered = True
        self._kill_switch_reason = reason
        self._state = RunnerState.ERROR

        self._result.errors.append(f"KILL SWITCH: {reason}")

        # Callback if provided
        if self._live_config.kill_switch_callback:
            self._live_config.kill_switch_callback(reason)

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        """
        Update portfolio state from broker.

        Args:
            portfolio: Current portfolio state
        """
        self._portfolio = portfolio
        self._execution_engine.update_portfolio(portfolio)
        self._reconciler.update_local_positions(dict(self._portfolio.positions))

        # Check for kill switch conditions
        if self._live_config.enable_kill_switch:
            # Check daily loss
            loss_violation = self._hard_cap_enforcer.check_daily_loss(
                portfolio.daily_pnl, portfolio.equity
            )
            if loss_violation:
                self._trigger_kill_switch(loss_violation.message)

            # Check drawdown
            drawdown_violation = self._hard_cap_enforcer.check_drawdown(
                portfolio.equity, portfolio.peak_equity
            )
            if drawdown_violation:
                self._trigger_kill_switch(drawdown_violation.message)

    def update_order_status(
        self,
        client_order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[Decimal] = None,
        avg_fill_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """
        Update order status from broker callback.

        Args:
            client_order_id: Client order ID
            status: New status
            filled_quantity: Filled quantity
            avg_fill_price: Average fill price

        Returns:
            Updated order or None
        """
        return self._execution_engine.update_order_status(
            client_order_id=client_order_id,
            status=status,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
        )

    def apply_cloud_config(self, cloud_config: RiskConfig) -> bool:
        """
        Apply Cloud configuration (if it doesn't violate local policy).

        Args:
            cloud_config: Risk config from Cloud

        Returns:
            True if applied, False if rejected
        """
        return self._policy_firewall.apply_cloud_config(cloud_config)

    def get_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders)

    def get_pending_orders(self) -> List[Order]:
        """Get pending orders."""
        return self._execution_engine.get_pending_orders()

    def get_portfolio(self) -> PortfolioState:
        """Get current portfolio state."""
        return self._portfolio

    def is_kill_switch_triggered(self) -> bool:
        """Check if kill switch was triggered."""
        return self._kill_switch_triggered

    def get_kill_switch_reason(self) -> Optional[str]:
        """Get kill switch reason."""
        return self._kill_switch_reason

    def reset_kill_switch(self, approval_code: str) -> bool:
        """
        Reset kill switch (requires manual approval).

        Args:
            approval_code: Approval code for reset

        Returns:
            True if reset, False if denied
        """
        # In production, this would verify the approval code
        # For now, require a specific format
        if not approval_code.startswith("APPROVE_RESET_"):
            return False

        self._kill_switch_triggered = False
        self._kill_switch_reason = None
        return True
