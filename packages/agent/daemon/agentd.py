# -*- coding: utf-8 -*-
"""
Agent Daemon (agentd) - Core daemon with lifecycle management.

Design Doc Phase 5:
- Agent Daemon: Local Vault + Sandbox + Policy Firewall + Reconciliation + Safe-degraded
- Агент автономно держит live-loop, хранит ключи, enforce'ит hard caps,
  восстанавливается и безопасно деградирует без cloud

CCEA Phase 5 Component.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple
from uuid import uuid4

from packages.agent.approval.manager import ApprovalManager
from packages.agent.cloud.client import CloudClient, CloudClientConfig
from packages.agent.cloud.types import PendingCommand
from packages.shared.contracts.config import ChangeClass

from packages.agent.daemon.kill_switch import (
    KillSwitchManager,
    KillSwitchConfig,
    HaltReason,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
)
from packages.agent.daemon.kill_switch_executor import (
    KillSwitchExecutor,
    KillSwitchExecutorConfig,
    ExecutionResult,
)
from packages.agent.daemon.time_sync import TimeSyncChecker, TimeSyncConfig
from packages.agent.daemon.preflight import PreflightChecker, PreflightConfig, PreflightResult
from packages.agent.daemon.degraded_mode import (
    DegradedModeManager,
    DegradedModeConfig,
    DegradedMode,
    DegradedModeAction,
)
from packages.agent.daemon.telemetry_buffer import (
    TelemetryBuffer,
    TelemetryBufferConfig,
    TelemetryEvent,
    TelemetryEventType,
)
from packages.agent.daemon.keychain import KeychainManager, KeychainConfig
from packages.agent.daemon.sandbox import Sandbox, SandboxConfig, SandboxType
from packages.agent.daemon.sandbox_enforcer import SandboxPermissionsEnforcer, EnforcedPermissions
from packages.agent.daemon.artifact_manager import (
    ArtifactManager,
    Artifact,
    ArtifactManifest,
    ArtifactVerificationError,
    ArtifactDownloadError,
)
from packages.agent.runner.live import LiveRunner, LiveRunnerConfig
from packages.agent.execution.engine import LiveExecutionEngine, OrderStatus
from packages.agent.reconciliation.reconciler import PositionReconciler, MismatchAction
from packages.agent.reconciliation.journal import (
    OrderJournal,
    JournalStatus,
    CommandJournal,
    CommandStatus as CommandJournalStatus,
)
from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig
from packages.agent.policy.hard_caps import HardCapEnforcer, HardCaps
from packages.agent.policy.risk_checker import RiskChecker, PortfolioState
from packages.shared.contracts.config import ExecutionMode


# Constants
HEARTBEAT_INTERVAL: Final[int] = 30  # seconds
RECONCILIATION_INTERVAL: Final[int] = 60  # seconds
VERSION: Final[str] = "1.0.0"


class DaemonState(Enum):
    """Agent daemon state."""
    CREATED = auto()
    INITIALIZING = auto()
    ENROLLING = auto()
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    HALTED = auto()
    ERROR = auto()


@dataclass
class DaemonConfig:
    """
    Agent daemon configuration.
    """
    # Identity
    agent_id: Optional[str] = None
    agent_name: str = "ccea-agent"
    agent_version: str = VERSION

    # Cloud connection
    cloud_endpoint: Optional[str] = None
    cloud_timeout_seconds: int = 30
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL
    cloud_enrollment_token: Optional[str] = None
    cloud_access_token: Optional[str] = None

    # Local storage
    data_dir: Path = field(default_factory=lambda: Path.home() / ".ccea")
    state_file: str = "daemon_state.json"

    # Component configs
    kill_switch_config: Optional[KillSwitchConfig] = None
    kill_switch_executor_config: Optional[KillSwitchExecutorConfig] = None
    time_sync_config: Optional[TimeSyncConfig] = None
    preflight_config: Optional[PreflightConfig] = None
    degraded_mode_config: Optional[DegradedModeConfig] = None
    telemetry_config: Optional[TelemetryBufferConfig] = None
    keychain_config: Optional[KeychainConfig] = None
    sandbox_config: Optional[SandboxConfig] = None

    # Behavior
    auto_recover: bool = True
    require_preflight: bool = True
    enable_telemetry: bool = True
    safe_shutdown_timeout_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "cloud_endpoint": self.cloud_endpoint,
            "cloud_enrollment_token": "***" if self.cloud_enrollment_token else None,
            "cloud_access_token": "***" if self.cloud_access_token else None,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "data_dir": str(self.data_dir),
            "auto_recover": self.auto_recover,
            "require_preflight": self.require_preflight,
        }


@dataclass
class DaemonStatus:
    """
    Agent daemon status snapshot.
    """
    agent_id: str = ""
    state: DaemonState = DaemonState.CREATED
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Uptime
    started_at: Optional[datetime] = None
    uptime_seconds: float = 0.0

    # Components
    vault_unlocked: bool = False
    cloud_connected: bool = False
    broker_connected: bool = False

    # Kill switch
    kill_switch_triggered: bool = False
    kill_switch_reason: Optional[str] = None

    # Degraded mode
    degraded_mode: DegradedMode = DegradedMode.NORMAL
    degraded_action: DegradedModeAction = DegradedModeAction.CONTINUE

    # Run info
    active_run_id: Optional[str] = None
    active_strategy: Optional[str] = None

    # Metrics
    orders_today: int = 0
    fills_today: int = 0
    pnl_today: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.name,
            "timestamp": self.timestamp.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": self.uptime_seconds,
            "vault_unlocked": self.vault_unlocked,
            "cloud_connected": self.cloud_connected,
            "broker_connected": self.broker_connected,
            "kill_switch_triggered": self.kill_switch_triggered,
            "kill_switch_reason": self.kill_switch_reason,
            "degraded_mode": self.degraded_mode.name,
            "degraded_action": self.degraded_action.value,
            "active_run_id": self.active_run_id,
            "active_strategy": self.active_strategy,
            "orders_today": self.orders_today,
            "fills_today": self.fills_today,
            "pnl_today": str(self.pnl_today),
        }


@dataclass
class RunControllerConfig:
    """
    Configuration for RunController.

    Design Doc Section 4.2/5.1/9.3: Live run controller configuration.
    """
    # Execution mode
    execution_mode: ExecutionMode = ExecutionMode.LIVE

    # Sandbox
    sandbox_enabled: bool = True
    sandbox_type: SandboxType = SandboxType.PROCESS
    deny_outbound_by_default: bool = True
    readonly_fs: bool = True

    # Reconciliation
    reconcile_on_start: bool = True
    reconcile_interval_seconds: int = RECONCILIATION_INTERVAL
    reconcile_on_restart: bool = True
    safe_halt_on_mismatch: bool = True

    # Policy
    require_policy_validation: bool = True
    require_risk_checks: bool = True

    # Idempotency (Design Doc 9.5)
    deployment_id: Optional[str] = None
    run_id: Optional[str] = None


class RunController:
    """
    Live Run Controller - Orchestrates live trading execution.

    Design Doc Sections 4.2/5.1/9.3:
    - Raises sandbox with policy enforcement
    - Loads artifact/manifest via ArtifactManager
    - Runs strategy in isolated process/container
    - Routes OrderIntents through PolicyFirewall + HardCapEnforcer + RiskChecker
    - Executes via LiveExecutionEngine
    - Journals via OrderJournal
    - Performs mandatory reconciliation

    AGENT ZONE ONLY - Never in Cloud zone.

    Usage:
        controller = RunController(config)

        # Initialize with artifact
        controller.initialize(artifact_manager, artifact_id)

        # Start live run
        success, error = controller.start_run()

        # Tick loop
        while running:
            controller.tick(market_data)

        # Stop
        controller.stop_run()
    """

    def __init__(
        self,
        config: Optional[RunControllerConfig] = None,
        policy_firewall: Optional[PolicyFirewall] = None,
        hard_cap_enforcer: Optional[HardCapEnforcer] = None,
        risk_checker: Optional[RiskChecker] = None,
        order_journal: Optional[OrderJournal] = None,
        reconciler: Optional[PositionReconciler] = None,
        broker_connector: Optional[Any] = None,
        kill_switch: Optional[KillSwitchManager] = None,
        on_intent: Optional[Callable[[Any], None]] = None,
        on_order: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize run controller.

        Args:
            config: Controller configuration
            policy_firewall: Policy validation
            hard_cap_enforcer: Hard cap enforcement
            risk_checker: Risk checking
            order_journal: Order journaling
            reconciler: Position reconciliation
            broker_connector: Broker for order execution
            kill_switch: Kill switch manager
            on_intent: Callback for intents
            on_order: Callback for orders
            on_error: Callback for errors
        """
        self.config = config or RunControllerConfig()

        # Components
        self._policy_firewall = policy_firewall
        self._hard_cap_enforcer = hard_cap_enforcer
        self._risk_checker = risk_checker
        self._order_journal = order_journal or OrderJournal()
        self._reconciler = reconciler
        self._broker_connector = broker_connector
        self._kill_switch = kill_switch

        # Sandbox and artifact management
        self._sandbox: Optional[Sandbox] = None
        self._sandbox_enforcer: Optional[SandboxPermissionsEnforcer] = None
        self._artifact_manager: Optional[ArtifactManager] = None
        self._current_artifact: Optional[Artifact] = None
        self._current_manifest: Optional[ArtifactManifest] = None

        # Execution engine
        self._execution_engine: Optional[LiveExecutionEngine] = None
        self._live_runner: Optional[LiveRunner] = None

        # Callbacks
        self._on_intent = on_intent
        self._on_order = on_order
        self._on_error = on_error

        # State
        self._is_running = False
        self._is_initialized = False
        self._run_id: Optional[str] = None
        self._portfolio_state = PortfolioState()

        # IPC queues for sandbox communication
        self._intent_queue: Optional[Any] = None  # multiprocessing.Queue
        self._command_queue: Optional[Any] = None

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._reconcile_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Check if run controller is active."""
        return self._is_running

    @property
    def is_initialized(self) -> bool:
        """Check if run controller is initialized."""
        return self._is_initialized

    def initialize(
        self,
        artifact_manager: ArtifactManager,
        artifact_id: Optional[str] = None,
        manifest: Optional[ArtifactManifest] = None,
    ) -> bool:
        """
        Initialize run controller with artifact.

        Args:
            artifact_manager: Artifact manager instance
            artifact_id: Artifact ID to use (if None, uses active artifact)
            manifest: Optional manifest override

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._is_initialized:
                return True

            try:
                self._artifact_manager = artifact_manager

                # Get artifact
                if artifact_id:
                    self._current_artifact = artifact_manager.get_artifact(artifact_id)
                else:
                    self._current_artifact = artifact_manager.get_active_artifact()

                if self._current_artifact:
                    self._current_manifest = self._current_artifact.manifest
                elif manifest:
                    self._current_manifest = manifest
                else:
                    self._current_manifest = ArtifactManifest.default_restrictive()

                # Initialize sandbox enforcer
                self._sandbox_enforcer = SandboxPermissionsEnforcer(
                    on_violation=self._handle_sandbox_violation,
                )

                # Initialize sandbox from manifest
                if self.config.sandbox_enabled:
                    self._init_sandbox()

                # Initialize execution engine
                self._init_execution_engine()

                # Initialize live runner if artifact available
                if self._current_artifact and self._current_artifact.extracted_path:
                    self._init_live_runner()

                self._is_initialized = True
                return True

            except Exception as e:
                if self._on_error:
                    self._on_error(f"RunController initialization failed: {e}")
                return False

    def _init_sandbox(self) -> None:
        """Initialize sandbox with enforced permissions from manifest."""
        if not self._sandbox_enforcer or not self._current_manifest:
            return

        # Compute enforced permissions from manifest + local policy
        permissions = self._sandbox_enforcer.compute_permissions(
            manifest=self._current_manifest,
            artifact_id=self._current_artifact.artifact_id if self._current_artifact else "",
        )

        # Apply Design Doc 9.2: deny-by-default egress and read-only FS
        sandbox_config = SandboxConfig(
            sandbox_type=self.config.sandbox_type,
            cpu_limit=permissions.max_cpu_percent / 100.0,
            memory_limit_mb=permissions.max_memory_mb,
            timeout_seconds=permissions.max_execution_time_seconds,
            network_enabled=permissions.network_enabled,
            egress_allowlist=permissions.egress_allowlist,
            deny_outbound_by_default=self.config.deny_outbound_by_default,
            readonly_fs=self.config.readonly_fs or permissions.filesystem_readonly,
            allowed_paths=[Path(p) for p in permissions.allowed_paths],
        )

        self._sandbox = Sandbox(
            config=sandbox_config,
            on_error=self._handle_sandbox_error,
        )

    def _init_execution_engine(self) -> None:
        """Initialize live execution engine."""
        # Generate deployment_id and run_id for idempotency
        deployment_id = self.config.deployment_id or str(uuid4())
        run_id = self.config.run_id or str(uuid4())

        # Broker submit function
        broker_submit = None
        if self._broker_connector and hasattr(self._broker_connector, "submit_order"):
            broker_submit = self._create_broker_submit_fn()

        self._execution_engine = LiveExecutionEngine(
            policy_firewall=self._policy_firewall or PolicyFirewall(),
            hard_cap_enforcer=self._hard_cap_enforcer or HardCapEnforcer(),
            risk_checker=self._risk_checker or RiskChecker(),
            broker_submit=broker_submit,
            broker_name=self._broker_connector.name if self._broker_connector and hasattr(self._broker_connector, "name") else "default",
            order_journal=self._order_journal,
            deployment_id=deployment_id,
            run_id=run_id,
        )

    def _create_broker_submit_fn(self) -> Callable:
        """Create broker submit function."""
        def submit(order) -> Tuple[bool, Optional[str], Optional[str]]:
            try:
                result = self._broker_connector.submit_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type.value,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    client_order_id=order.client_order_id,
                )
                return (result.success, result.broker_order_id, result.error_message if not result.success else None)
            except Exception as e:
                return (False, None, str(e))
        return submit

    def _init_live_runner(self) -> None:
        """Initialize live runner for strategy execution."""
        if not self._current_artifact or not self._current_artifact.extracted_path:
            return

        runner_config = LiveRunnerConfig(
            mode=self.config.execution_mode,
        )

        self._live_runner = LiveRunner(config=runner_config)

    def start_run(
        self,
        run_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Start a live trading run.

        Design Doc 9.3/9.5:
        - Performs mandatory reconciliation before first tick
        - Starts sandbox
        - Initializes strategy in isolated process
        - Begins execution loop

        Args:
            run_id: Optional run ID

        Returns:
            (success, error_message)
        """
        with self._lock:
            if self._is_running:
                return True, None

            if not self._is_initialized:
                return False, "RunController not initialized"

            try:
                self._run_id = run_id or str(uuid4())
                self._stop_event.clear()

                # Design Doc 9.5: Mandatory reconciliation on start
                if self.config.reconcile_on_start:
                    reconcile_result = self._perform_reconciliation()
                    if not reconcile_result.success and self.config.safe_halt_on_mismatch:
                        return False, f"Reconciliation failed: {reconcile_result.error} - safe halt"

                # Start sandbox
                if self._sandbox:
                    if not self._sandbox.start():
                        return False, "Failed to start sandbox"

                # Start reconciliation thread
                if self.config.reconcile_interval_seconds > 0:
                    self._start_reconciliation_thread()

                self._is_running = True
                return True, None

            except Exception as e:
                return False, str(e)

    def stop_run(self, reason: str = "user_requested") -> bool:
        """
        Stop the current run.

        Args:
            reason: Reason for stopping

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if not self._is_running:
                return True

            self._stop_event.set()

            # Stop reconciliation thread
            if self._reconcile_thread:
                self._reconcile_thread.join(timeout=5)

            # Stop sandbox
            if self._sandbox:
                self._sandbox.stop()

            # Final reconciliation
            if self.config.reconcile_on_restart:
                self._perform_reconciliation()

            self._is_running = False
            return True

    def tick(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[Any]:
        """
        Execute one tick of the trading loop.

        Design Doc 5.1: Main execution tick
        - Receives market data
        - Gets intents from strategy (via sandbox IPC)
        - Validates intents through policy stack
        - Executes valid intents

        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state

        Returns:
            List of execution results
        """
        if not self._is_running:
            return []

        results = []

        # Update portfolio state
        if portfolio_state:
            self._portfolio_state = portfolio_state
            if self._execution_engine:
                self._execution_engine.update_portfolio(portfolio_state)

        # Check kill switch
        if self._kill_switch and self._kill_switch.is_triggered:
            return []

        # Get intents from strategy via IPC (if sandbox running)
        intents = self._get_strategy_intents(market_data)

        # Execute each intent
        for intent in intents:
            if self._on_intent:
                self._on_intent(intent)

            result = self._execute_intent(intent)
            results.append(result)

            if self._on_order and result.order:
                self._on_order(result.order)

        return results

    def _get_strategy_intents(self, market_data: Optional[Dict[str, Any]]) -> List[Any]:
        """Get intents from strategy via sandbox IPC."""
        # This is a simplified implementation
        # Full implementation would communicate with strategy process via IPC
        intents = []

        if self._intent_queue:
            try:
                while not self._intent_queue.empty():
                    intent = self._intent_queue.get_nowait()
                    intents.append(intent)
            except Exception:
                pass

        return intents

    def _execute_intent(self, intent: Any) -> Any:
        """Execute a single intent through the policy stack."""
        if not self._execution_engine:
            from packages.agent.execution.engine import ExecutionResult
            return ExecutionResult(
                success=False,
                error_message="Execution engine not initialized",
            )

        # Execute through engine (handles policy validation)
        return self._execution_engine.execute(
            intent=intent,
            current_price=None,  # Would come from market data
            origin="runner",  # Local origin
        )

    def _perform_reconciliation(self) -> Any:
        """
        Perform position reconciliation.

        Design Doc 9.5: Mandatory reconciliation
        - Reconciles broker positions with local state
        - Safe-halts on unrecoverable mismatches
        """
        @dataclass
        class ReconcileResult:
            success: bool = True
            error: Optional[str] = None
            mismatches: List[Any] = field(default_factory=list)

        if not self._reconciler:
            return ReconcileResult()

        try:
            # Fetch broker positions
            broker_positions = {}
            if self._broker_connector and hasattr(self._broker_connector, "fetch_positions"):
                broker_positions = self._broker_connector.fetch_positions()

            # Fetch broker open orders
            broker_orders = {}
            if self._broker_connector and hasattr(self._broker_connector, "fetch_order_status"):
                broker_orders = self._broker_connector.get_open_orders()

            # Reconcile positions
            result = self._reconciler.reconcile(
                local_positions=self._portfolio_state.positions,
                broker_positions=broker_positions,
            )

            # Reconcile orders
            order_result = self._reconciler.reconcile_orders(
                local_orders={},  # From order journal
                broker_orders={str(o.order_id): o for o in broker_orders} if broker_orders else {},
            )

            if not result.is_consistent or not order_result.is_consistent:
                # Check if safe-halt is required
                if self.config.safe_halt_on_mismatch:
                    return ReconcileResult(
                        success=False,
                        error="Position mismatch detected - safe halt",
                        mismatches=result.mismatches + order_result.mismatches,
                    )

            return ReconcileResult(success=True)

        except Exception as e:
            return ReconcileResult(success=False, error=str(e))

    def _start_reconciliation_thread(self) -> None:
        """Start background reconciliation thread."""
        self._reconcile_thread = threading.Thread(
            target=self._reconciliation_loop,
            daemon=True,
        )
        self._reconcile_thread.start()

    def _reconciliation_loop(self) -> None:
        """Background reconciliation loop."""
        while not self._stop_event.is_set():
            try:
                result = self._perform_reconciliation()
                if not result.success and self.config.safe_halt_on_mismatch:
                    # Trigger kill switch on reconciliation failure
                    if self._kill_switch:
                        reason = HaltReason(
                            reason_type=HaltReasonType.RECONCILIATION_MISMATCH,
                            severity=HaltSeverity.CRITICAL,
                            message=f"Reconciliation failed: {result.error}",
                            trigger_source="RunController",
                        )
                        self._kill_switch.trigger(reason, HaltAction.HALT_ONLY)

            except Exception:
                pass

            self._stop_event.wait(self.config.reconcile_interval_seconds)

    def _handle_sandbox_violation(self, violation: Any) -> None:
        """Handle sandbox permission violation."""
        if self._on_error:
            self._on_error(f"Sandbox violation: {violation.details}")

    def _handle_sandbox_error(self, error: str) -> None:
        """Handle sandbox error."""
        if self._on_error:
            self._on_error(f"Sandbox error: {error}")

    def get_status(self) -> Dict[str, Any]:
        """Get run controller status."""
        return {
            "is_running": self._is_running,
            "is_initialized": self._is_initialized,
            "run_id": self._run_id,
            "has_sandbox": self._sandbox is not None,
            "sandbox_running": self._sandbox.is_running if self._sandbox else False,
            "has_execution_engine": self._execution_engine is not None,
            "has_reconciler": self._reconciler is not None,
            "artifact_id": self._current_artifact.artifact_id if self._current_artifact else None,
            "artifact_name": self._current_artifact.name if self._current_artifact else None,
            "execution_mode": self.config.execution_mode.value if hasattr(self.config.execution_mode, "value") else str(self.config.execution_mode),
        }

    def set_broker_connector(self, connector: Any) -> None:
        """Set broker connector."""
        self._broker_connector = connector
        # Update execution engine's broker submit function
        if self._execution_engine and connector:
            self._execution_engine._broker_submit = self._create_broker_submit_fn()

    def set_reconciler(self, reconciler: PositionReconciler) -> None:
        """Set position reconciler."""
        self._reconciler = reconciler

    def set_kill_switch(self, kill_switch: KillSwitchManager) -> None:
        """Set kill switch manager."""
        self._kill_switch = kill_switch


class AgentDaemon:
    """
    CCEA Agent Daemon (agentd) - Core lifecycle manager.

    Design Doc Phase 5:
    - Manages agent lifecycle (start, stop, pause, resume)
    - Integrates all Phase 5 components
    - Autonomous operation (works without cloud)
    - Safe degradation and recovery

    Architecture:
        AgentDaemon
        ├── LocalVault (credentials)
        ├── KeychainManager (master key)
        ├── PolicyFirewall (risk limits)
        ├── HardCapEnforcer (absolute limits)
        ├── KillSwitchManager (emergency stop)
        ├── DegradedModeManager (safe modes)
        ├── PreflightChecker (validation)
        ├── TimeSyncChecker (time sync)
        ├── TelemetryBuffer (events)
        ├── Sandbox (strategy isolation)
        ├── LiveRunner (execution)
        └── PositionReconciler (state sync)

    Usage:
        daemon = AgentDaemon(config)
        daemon.initialize()
        daemon.start()

        # Autonomous operation
        # Daemon handles:
        # - Heartbeats to cloud
        # - Kill switch monitoring
        # - Degraded mode transitions
        # - Position reconciliation
        # - Telemetry buffering

        daemon.stop()
    """

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
    ):
        """
        Initialize agent daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or DaemonConfig()

        # Generate agent ID if not provided
        if not self.config.agent_id:
            self.config.agent_id = str(uuid4())

        # State
        self._state = DaemonState.CREATED
        self._started_at: Optional[datetime] = None
        self._status = DaemonStatus(agent_id=self.config.agent_id)

        # Components
        self._keychain: Optional[KeychainManager] = None
        self._vault: Optional[Any] = None  # LocalVault
        self._kill_switch: Optional[KillSwitchManager] = None
        self._kill_switch_executor: Optional[KillSwitchExecutor] = None
        self._time_checker: Optional[TimeSyncChecker] = None
        self._preflight_checker: Optional[PreflightChecker] = None
        self._degraded_manager: Optional[DegradedModeManager] = None
        self._telemetry_buffer: Optional[TelemetryBuffer] = None
        self._sandbox: Optional[Sandbox] = None
        self._approval_manager: ApprovalManager = ApprovalManager()
        self._cloud_client: Optional[CloudClient] = None

        # Cloud command tracking (Design Doc 10.4.2: durable journal for idempotency)
        self._command_journal: Optional[CommandJournal] = None
        self._executed_command_ids: set[str] = set()  # In-memory cache backed by journal
        self._pending_cloud_approval_by_command_id: Dict[str, Any] = {}
        self._submitted_cloud_approvals: set[str] = set()

        # External components (set via setters)
        self._policy_firewall: Optional[PolicyFirewall] = None
        self._hard_cap_enforcer: Optional[HardCapEnforcer] = None
        self._live_runner: Optional[LiveRunner] = None
        self._reconciler: Optional[PositionReconciler] = None
        self._broker_connector: Optional[Any] = None

        # Run controller (Design Doc 4.2/5.1/9.3)
        self._run_controller: Optional[RunController] = None
        self._run_controller_config: Optional[RunControllerConfig] = None
        self._artifact_manager: Optional[ArtifactManager] = None
        self._order_journal: Optional[OrderJournal] = None

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_state_change: Optional[Callable[[DaemonState], None]] = None
        self._on_kill_switch: Optional[Callable[[HaltReason], None]] = None
        self._on_degraded_mode: Optional[Callable[[DegradedMode, DegradedModeAction], None]] = None

        # Register cleanup
        atexit.register(self._cleanup)

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id or ""

    @property
    def state(self) -> DaemonState:
        """Get current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._state in (DaemonState.RUNNING, DaemonState.PAUSED)

    @property
    def is_halted(self) -> bool:
        """Check if daemon is halted."""
        return self._state == DaemonState.HALTED

    @property
    def status(self) -> DaemonStatus:
        """Get current status snapshot."""
        self._update_status()
        return self._status

    # ===== Lifecycle =====

    def initialize(self) -> bool:
        """
        Initialize daemon and all components.

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._state != DaemonState.CREATED:
                return self._state != DaemonState.ERROR

            self._state = DaemonState.INITIALIZING
            self._emit_state_change()

            try:
                # Ensure data directory exists
                self.config.data_dir.mkdir(parents=True, exist_ok=True)

                # Initialize components
                self._init_keychain()
                self._init_kill_switch()
                self._init_time_checker()
                self._init_degraded_manager()
                self._init_telemetry_buffer()
                self._init_preflight_checker()
                self._init_cloud_client()

                # Initialize sandbox from config (Design Doc 9.2)
                self._init_sandbox()

                # Initialize artifact manager (Design Doc Phase 4)
                self._init_artifact_manager()

                # Initialize run controller (Design Doc 4.2/5.1/9.3)
                self._init_run_controller()

                # Initialize command journal (Design Doc 10.4.2)
                self._init_command_journal()

                # Load saved state
                self._load_state()

                # Mark as idle
                self._state = DaemonState.IDLE
                self._emit_state_change()

                # Log initialization
                self._log_event(TelemetryEventType.STATE_CHANGE, {
                    "old_state": "CREATED",
                    "new_state": "IDLE",
                    "message": "Agent daemon initialized",
                })

                return True

            except Exception as e:
                self._state = DaemonState.ERROR
                self._log_event(TelemetryEventType.ERROR, {
                    "error_type": "InitializationError",
                    "message": str(e),
                })
                return False

    def start(
        self,
        run_id: Optional[str] = None,
        strategy: Optional[Any] = None,
        manifest: Optional[Dict[str, Any]] = None,
        broker_name: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Start the daemon.

        Args:
            run_id: Optional run ID
            strategy: Optional strategy to start
            manifest: Optional artifact manifest
            broker_name: Broker to use

        Returns:
            (success, error_message)
        """
        with self._lock:
            if self._state == DaemonState.RUNNING:
                return True, None

            if self._state not in (DaemonState.IDLE, DaemonState.STOPPED, DaemonState.PAUSED):
                return False, f"Cannot start from state {self._state.name}"

            # Check kill switch
            if self._kill_switch and self._kill_switch.is_triggered:
                return False, f"Kill switch triggered: {self._kill_switch.current_halt.halt_reason.message if self._kill_switch.current_halt else 'unknown'}"

            # Run preflight checks
            if self.config.require_preflight:
                preflight_result = self._run_preflight(manifest, broker_name, run_id)
                if not preflight_result.passed:
                    return False, f"Preflight failed: {'; '.join(preflight_result.errors)}"

            # Start
            try:
                self._started_at = datetime.utcnow()
                self._state = DaemonState.RUNNING
                self._status.active_run_id = run_id or str(uuid4())
                self._emit_state_change()

                # Start background threads
                self._start_heartbeat_thread()
                self._start_monitor_thread()

                # Start sandbox if configured
                if self._sandbox:
                    self._sandbox.start()

                # Log start
                self._log_event(TelemetryEventType.STATE_CHANGE, {
                    "old_state": "IDLE",
                    "new_state": "RUNNING",
                    "run_id": self._status.active_run_id,
                })

                return True, None

            except Exception as e:
                self._state = DaemonState.ERROR
                return False, str(e)

    def stop(self, reason: str = "user_requested") -> bool:
        """
        Stop the daemon gracefully.

        Args:
            reason: Reason for stopping

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self._state in (DaemonState.STOPPED, DaemonState.CREATED):
                return True

            self._state = DaemonState.STOPPING
            self._emit_state_change()

            try:
                # Stop background threads
                self._stop_event.set()

                if self._heartbeat_thread:
                    self._heartbeat_thread.join(timeout=5)
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)

                # Stop sandbox
                if self._sandbox:
                    self._sandbox.stop()

                # Flush telemetry
                if self._telemetry_buffer:
                    self._telemetry_buffer.flush()

                # Save state
                self._save_state()

                # Update state
                self._state = DaemonState.STOPPED
                self._status.active_run_id = None
                self._emit_state_change()

                # Log stop
                self._log_event(TelemetryEventType.STATE_CHANGE, {
                    "old_state": "RUNNING",
                    "new_state": "STOPPED",
                    "reason": reason,
                })

                return True

            except Exception as e:
                self._state = DaemonState.ERROR
                return False

    def pause(self) -> bool:
        """
        Pause the daemon.

        Returns:
            True if paused successfully
        """
        with self._lock:
            if self._state != DaemonState.RUNNING:
                return False

            self._state = DaemonState.PAUSED
            self._emit_state_change()

            self._log_event(TelemetryEventType.STATE_CHANGE, {
                "old_state": "RUNNING",
                "new_state": "PAUSED",
            })

            return True

    def resume(self) -> bool:
        """
        Resume paused daemon.

        Returns:
            True if resumed successfully
        """
        with self._lock:
            if self._state != DaemonState.PAUSED:
                return False

            # Check kill switch
            if self._kill_switch and self._kill_switch.is_triggered:
                return False

            self._state = DaemonState.RUNNING
            self._emit_state_change()

            self._log_event(TelemetryEventType.STATE_CHANGE, {
                "old_state": "PAUSED",
                "new_state": "RUNNING",
            })

            return True

    def halt(self, reason: HaltReason, action: HaltAction = HaltAction.CANCEL_ORDERS) -> bool:
        """
        Trigger emergency halt.

        Args:
            reason: Halt reason
            action: Action to take

        Returns:
            True if halted
        """
        with self._lock:
            if self._kill_switch:
                self._kill_switch.trigger(reason, action)

            self._state = DaemonState.HALTED
            self._emit_state_change()

            self._log_event(TelemetryEventType.KILL_SWITCH, {
                "reason_type": reason.reason_type.name,
                "severity": reason.severity.value,
                "message": reason.message,
                "action": action.value,
            })

            if self._on_kill_switch:
                self._on_kill_switch(reason)

            return True

    def acknowledge_halt(self, acknowledged_by: str, approval_code: str) -> bool:
        """
        Acknowledge halt event.

        Args:
            acknowledged_by: Who acknowledged
            approval_code: Approval code

        Returns:
            True if acknowledged
        """
        if self._kill_switch:
            return self._kill_switch.acknowledge(acknowledged_by, approval_code)
        return False

    def reset_halt(self) -> bool:
        """
        Reset from halted state.

        Returns:
            True if reset successful
        """
        with self._lock:
            if self._state != DaemonState.HALTED:
                return True

            if self._kill_switch:
                if not self._kill_switch.reset():
                    return False

            self._state = DaemonState.IDLE
            self._emit_state_change()

            return True

    # ===== Component Initialization =====

    def _init_keychain(self) -> None:
        """Initialize keychain manager."""
        self._keychain = KeychainManager(
            config=self.config.keychain_config
        )

    def _init_kill_switch(self) -> None:
        """
        Initialize kill switch manager with executor integration.

        Design Doc Section 9.4:
        - KillSwitchExecutor bridges KillSwitchManager with BrokerConnector
        - Executor handles actual order cancellation and position flattening
        - Manager handles trigger logic and state management
        """
        # Create executor (broker connector set later via set_broker_connector)
        self._kill_switch_executor = KillSwitchExecutor(
            broker_connector=self._broker_connector,
            config=self.config.kill_switch_executor_config,
            on_action=self._handle_kill_switch_action,
        )

        # Create manager with executor callbacks
        self._kill_switch = KillSwitchManager(
            config=self.config.kill_switch_config,
            on_trigger=self._handle_kill_switch,
            cancel_orders_fn=self._kill_switch_executor.cancel_all_orders,
            flatten_fn=self._kill_switch_executor.flatten_all_positions,
        )

    def _handle_kill_switch_action(self, action: str, details: Dict[str, Any]) -> None:
        """
        Handle kill switch executor action for telemetry.

        Args:
            action: Action name (e.g., 'cancel_orders_started', 'flatten_positions_result')
            details: Action details
        """
        self._log_event(TelemetryEventType.KILL_SWITCH, {
            "executor_action": action,
            **details,
        })

    def _init_time_checker(self) -> None:
        """Initialize time sync checker."""
        self._time_checker = TimeSyncChecker(
            config=self.config.time_sync_config,
            on_drift=self._handle_time_drift,
        )

    def _init_degraded_manager(self) -> None:
        """Initialize degraded mode manager."""
        self._degraded_manager = DegradedModeManager(
            config=self.config.degraded_mode_config,
            on_mode_change=self._handle_degraded_mode,
        )

    def _init_telemetry_buffer(self) -> None:
        """Initialize telemetry buffer."""
        if self.config.enable_telemetry:
            self._telemetry_buffer = TelemetryBuffer(
                config=self.config.telemetry_config,
                agent_id=self.agent_id,
            )

    def _init_preflight_checker(self) -> None:
        """Initialize preflight checker."""
        self._preflight_checker = PreflightChecker(
            config=self.config.preflight_config,
            vault=self._vault,
            policy_firewall=self._policy_firewall,
            hard_cap_enforcer=self._hard_cap_enforcer,
            time_checker=self._time_checker,
        )

    def _init_sandbox(self) -> None:
        """
        Initialize sandbox from config.

        Design Doc Section 9.2:
        - deny-by-default egress
        - read-only filesystem
        - process/container isolation
        """
        if not self.config.sandbox_config:
            # Create default sandbox config with secure defaults
            self.config.sandbox_config = SandboxConfig(
                sandbox_type=SandboxType.PROCESS,
                deny_outbound_by_default=True,
                readonly_fs=True,
            )

        self._sandbox = Sandbox(
            config=self.config.sandbox_config,
            on_error=self._handle_sandbox_error,
        )

    def _handle_sandbox_error(self, error: str) -> None:
        """Handle sandbox error."""
        self._log_event(TelemetryEventType.ERROR, {
            "error_type": "SandboxError",
            "message": error,
        })

    def _init_artifact_manager(self) -> None:
        """
        Initialize artifact manager.

        Design Doc Phase 4:
        - Manages artifact download/verification
        - Supports OCI and ZIP formats
        - Pinned digest enforcement
        """
        artifact_cache_dir = self.config.data_dir / "artifacts"
        self._artifact_manager = ArtifactManager(
            cache_dir=artifact_cache_dir,
            on_progress=self._handle_artifact_progress,
        )

    def _handle_artifact_progress(self, artifact_id: str, progress: float) -> None:
        """Handle artifact download progress."""
        self._log_event(TelemetryEventType.STATE_CHANGE, {
            "message": f"Artifact download progress: {progress:.1f}%",
            "artifact_id": artifact_id,
            "progress_pct": progress,
        })

    def _init_run_controller(self) -> None:
        """
        Initialize run controller for live execution.

        Design Doc Sections 4.2/5.1/9.3:
        - Orchestrates live trading execution
        - Integrates sandbox, policy firewall, execution engine
        - Mandatory reconciliation
        """
        # Create run controller config
        self._run_controller_config = RunControllerConfig(
            execution_mode=ExecutionMode.LIVE,
            sandbox_enabled=self.config.sandbox_config is not None,
            sandbox_type=self.config.sandbox_config.sandbox_type if self.config.sandbox_config else SandboxType.PROCESS,
            deny_outbound_by_default=True,
            readonly_fs=True,
            reconcile_on_start=True,
            reconcile_interval_seconds=RECONCILIATION_INTERVAL,
            safe_halt_on_mismatch=True,
        )

        # Initialize order journal
        journal_path = self.config.data_dir / "order_journal.db"
        self._order_journal = OrderJournal(db_path=journal_path)

        # Create run controller
        self._run_controller = RunController(
            config=self._run_controller_config,
            policy_firewall=self._policy_firewall,
            hard_cap_enforcer=self._hard_cap_enforcer,
            order_journal=self._order_journal,
            reconciler=self._reconciler,
            broker_connector=self._broker_connector,
            kill_switch=self._kill_switch,
            on_intent=self._handle_run_controller_intent,
            on_order=self._handle_run_controller_order,
            on_error=self._handle_run_controller_error,
        )

        # Initialize run controller with artifact manager
        if self._artifact_manager:
            self._run_controller.initialize(self._artifact_manager)

    def _handle_run_controller_intent(self, intent: Any) -> None:
        """Handle intent from run controller."""
        self._log_event(TelemetryEventType.TRADE, {
            "event": "intent_received",
            "intent_id": str(intent.intent_id) if hasattr(intent, "intent_id") else "unknown",
            "symbol": intent.symbol if hasattr(intent, "symbol") else "unknown",
        })

    def _handle_run_controller_order(self, order: Any) -> None:
        """Handle order from run controller."""
        self._log_event(TelemetryEventType.TRADE, {
            "event": "order_submitted",
            "order_id": str(order.order_id) if hasattr(order, "order_id") else "unknown",
            "symbol": order.symbol if hasattr(order, "symbol") else "unknown",
            "side": order.side if hasattr(order, "side") else "unknown",
        })
        self._status.orders_today += 1

    def _handle_run_controller_error(self, error: str) -> None:
        """Handle error from run controller."""
        self._log_event(TelemetryEventType.ERROR, {
            "error_type": "RunControllerError",
            "message": error,
        })

    def _init_command_journal(self) -> None:
        """
        Initialize command journal for idempotent command delivery.

        Design Doc 10.4.2:
        - Durable journal persists command execution state across restarts
        - Prevents duplicate execution of the same command
        - Recovers executed command IDs on startup
        """
        journal_path = self.config.data_dir / "command_journal.db"
        self._command_journal = CommandJournal(db_path=journal_path)

        # Restore executed command IDs from journal (Design Doc 10.4.2)
        self._executed_command_ids = self._command_journal.get_executed_command_ids()

        self._log_event(TelemetryEventType.STATE_CHANGE, {
            "message": "Command journal initialized",
            "restored_command_count": len(self._executed_command_ids),
        })

    # ===== Callbacks =====

    def _handle_kill_switch(self, reason: HaltReason) -> None:
        """Handle kill switch trigger."""
        with self._lock:
            self._state = DaemonState.HALTED
            self._emit_state_change()

        self._log_event(TelemetryEventType.KILL_SWITCH, reason.to_dict())

        if self._on_kill_switch:
            try:
                self._on_kill_switch(reason)
            except Exception:
                pass

    def _handle_time_drift(self, result: Any) -> None:
        """Handle time drift detection."""
        if abs(result.drift_seconds) > 5:
            reason = HaltReason(
                reason_type=HaltReasonType.TIME_SYNC_DRIFT,
                severity=HaltSeverity.HIGH,
                message=f"Time drift of {result.drift_seconds:.1f}s detected",
                details={"drift_seconds": result.drift_seconds},
                trigger_source="TimeSyncChecker",
            )
            self.halt(reason, HaltAction.HALT_ONLY)

    def _handle_degraded_mode(self, mode: DegradedMode, action: DegradedModeAction) -> None:
        """Handle degraded mode change."""
        self._log_event(TelemetryEventType.DEGRADED_MODE, {
            "mode": mode.name,
            "action": action.value,
        })

        if self._on_degraded_mode:
            try:
                self._on_degraded_mode(mode, action)
            except Exception:
                pass

    def _emit_state_change(self) -> None:
        """Emit state change event."""
        if self._on_state_change:
            try:
                self._on_state_change(self._state)
            except Exception:
                pass

    # ===== Preflight =====

    def _run_preflight(
        self,
        manifest: Optional[Dict[str, Any]],
        broker_name: Optional[str],
        run_id: Optional[str],
    ) -> PreflightResult:
        """Run preflight checks."""
        if not self._preflight_checker:
            self._init_preflight_checker()

        return self._preflight_checker.run_preflight(
            manifest=manifest,
            broker_name=broker_name,
            run_id=run_id,
        )

    # ===== Background Threads =====

    def _start_heartbeat_thread(self) -> None:
        """Start heartbeat thread."""
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _start_monitor_thread(self) -> None:
        """Start monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                # Send heartbeat telemetry
                if self._telemetry_buffer:
                    self._telemetry_buffer.add_heartbeat(
                        status=self._state.name,
                        details={
                            "agent_id": self.agent_id,
                            "uptime_seconds": self._get_uptime_seconds(),
                        },
                    )

                # Report cloud status to degraded manager
                if self._degraded_manager:
                    self._degraded_manager.report_cloud_status(
                        connected=bool(self._cloud_client and self._cloud_client.is_connected)
                    )

                # Cloud lifecycle heartbeat + command poll (outbound-only)
                if self._cloud_client and self.config.cloud_access_token:
                    self._cloud_client.access_token = self.config.cloud_access_token
                    hb = self._cloud_client.heartbeat(
                        agent_version=self.config.agent_version,
                        current_state=self._state.name,
                        last_run_id=None,
                        health_metrics={
                            "uptime_seconds": self._get_uptime_seconds(),
                            "state": self._state.name,
                        },
                    )
                    self._status.cloud_connected = True

                    if hb.pending_commands > 0:
                        poll = self._cloud_client.poll_commands(limit=10)
                        if poll.commands:
                            self._process_cloud_commands(poll.commands)

                    # Design Doc Phase 5: Flush telemetry to Cloud
                    self._flush_telemetry_to_cloud()
                else:
                    self._status.cloud_connected = False

            except Exception:
                self._status.cloud_connected = False
                pass

            self._stop_event.wait(self.config.heartbeat_interval_seconds)

    def _flush_telemetry_to_cloud(self) -> None:
        """
        Flush buffered telemetry events to Cloud.

        Design Doc Phase 5: End-to-end telemetry pipeline.
        Uses CloudClient.send_telemetry() to upload buffered events.
        """
        if not self._telemetry_buffer or not self._cloud_client:
            return

        def send_fn(events: List[Dict[str, Any]]) -> bool:
            """Send function for telemetry buffer flush."""
            try:
                return self._cloud_client.send_telemetry(events)
            except Exception as e:
                self._log_event(TelemetryEventType.ERROR, {
                    "error_type": "TelemetryUploadError",
                    "message": str(e),
                })
                return False

        try:
            sent_count = self._telemetry_buffer.flush(send_fn=send_fn)
            if sent_count > 0:
                self._log_event(TelemetryEventType.STATE_CHANGE, {
                    "message": f"Flushed {sent_count} telemetry events to Cloud",
                    "events_sent": sent_count,
                })
        except Exception as e:
            self._log_event(TelemetryEventType.ERROR, {
                "error_type": "TelemetryFlushError",
                "message": str(e),
            })

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Check time sync
                if self._time_checker:
                    self._time_checker.check()

                # Update status
                self._update_status()

            except Exception:
                pass

            self._stop_event.wait(60)  # Check every minute

    # ===== State Persistence =====

    def _save_state(self) -> None:
        """Save daemon state to disk."""
        state_file = self.config.data_dir / self.config.state_file

        state = {
            "version": VERSION,
            "agent_id": self.agent_id,
            "cloud_access_token": self.config.cloud_access_token,
            "saved_at": datetime.utcnow().isoformat(),
            "state": self._state.name,
            "status": self._status.to_dict(),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load daemon state from disk."""
        state_file = self.config.data_dir / self.config.state_file

        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            # Restore agent ID if configured
            if not self.config.agent_id and state.get("agent_id"):
                self.config.agent_id = state["agent_id"]
                self._status.agent_id = state["agent_id"]

            if not self.config.cloud_access_token and state.get("cloud_access_token"):
                self.config.cloud_access_token = state["cloud_access_token"]

        except Exception:
            pass  # Start fresh

    def _init_cloud_client(self) -> None:
        """Initialize outbound-only CloudClient (if configured)."""
        if not self.config.cloud_endpoint:
            return

        self._cloud_client = CloudClient(
            CloudClientConfig(
                base_url=self.config.cloud_endpoint,
                timeout_seconds=self.config.cloud_timeout_seconds,
                user_agent=f"ccea-agentd/{self.config.agent_version}",
            ),
            access_token=self.config.cloud_access_token,
        )

        # Best-effort enroll if token provided and no access token
        if not self.config.cloud_access_token and self.config.cloud_enrollment_token:
            enroll = self._cloud_client.enroll(
                enrollment_token=self.config.cloud_enrollment_token,
                agent_name=self.config.agent_name,
                agent_version=self.config.agent_version,
                capabilities=[],
                attestation=None,
            )
            self.config.cloud_access_token = enroll.access_token
            # Align local id with cloud agent UUID (string form)
            self.config.agent_id = str(enroll.agent_id)
            self._status.agent_id = str(enroll.agent_id)

    def _process_cloud_commands(self, commands: List[PendingCommand]) -> None:
        """
        Process polled commands from Cloud.

        Design Doc 10.4.2: Uses CommandJournal for idempotent command delivery.

        Security invariant:
        - TRADING_IMPACTING commands MUST require local approval; Cloud cannot bypass by flipping flags.
        """
        if not self._cloud_client:
            return

        for cmd in commands:
            cmd_id = str(cmd.id)

            # Check both in-memory cache and durable journal (Design Doc 10.4.2)
            if cmd_id in self._executed_command_ids:
                continue
            if self._command_journal and self._command_journal.is_executed(cmd_id, cmd.idempotency_key):
                self._executed_command_ids.add(cmd_id)  # Update in-memory cache
                continue

            # Fail-closed change class parsing
            try:
                change_class = ChangeClass(cmd.change_class)
            except Exception:
                change_class = ChangeClass.TRADING_IMPACTING

            trading_impacting = change_class == ChangeClass.TRADING_IMPACTING

            # If Cloud tries to mark trading-impacting as not requiring approval -> refuse execution.
            if trading_impacting and not cmd.requires_approval:
                try:
                    # Log to journal as received then failed (Design Doc 10.4.2)
                    if self._command_journal:
                        self._command_journal.log_received(
                            cmd_id, cmd.idempotency_key, cmd.command_type, cmd.payload_ref
                        )
                        self._command_journal.mark_failed(
                            cmd_id, "Refused: TRADING_IMPACTING without requires_approval"
                        )

                    self._cloud_client.acknowledge_command(cmd.id)
                    self._cloud_client.submit_command_result(
                        command_id=cmd.id,
                        success=False,
                        error_message="Refused: TRADING_IMPACTING command without requires_approval (fail-closed).",
                    )
                    self._executed_command_ids.add(cmd_id)
                except Exception:
                    pass
                continue

            # Approval phase
            if cmd.status.lower() == "pending_approval":
                if cmd_id in self._submitted_cloud_approvals:
                    continue

                artifact_digest = cmd.payload_ref if cmd.payload_ref.startswith("sha256:") else None
                req_id = self._pending_cloud_approval_by_command_id.get(cmd_id)
                req = self._approval_manager.get_request(req_id) if req_id else None
                if req is None:
                    req = self._approval_manager.create_request(
                        command_type=cmd.command_type,
                        description=f"Cloud command {cmd.command_type} requires local approval",
                        change_class=ChangeClass.TRADING_IMPACTING if trading_impacting else change_class,
                        details={
                            "command_id": cmd_id,
                            "idempotency_key": cmd.idempotency_key,
                            "payload_ref": cmd.payload_ref,
                        },
                        artifact_digest=artifact_digest,
                    )
                    self._pending_cloud_approval_by_command_id[cmd_id] = req.request_id

                if req.status.name.lower() == "approved" and req.evidence_hash:
                    try:
                        self._cloud_client.submit_local_approval(
                            command_id=cmd.id,
                            approved=True,
                            evidence_hash=f"sha256:{req.evidence_hash}",
                            diff_summary={
                                "command_type": cmd.command_type,
                                "payload_ref": cmd.payload_ref,
                            },
                            reason=req.decision_reason or "Approved by local policy",
                        )
                        self._submitted_cloud_approvals.add(cmd_id)
                    except Exception:
                        pass
                continue

            # Execution phase
            try:
                self._cloud_client.acknowledge_command(cmd.id)
            except Exception:
                continue

            # Log to journal as RECEIVED before execution (Design Doc 10.4.2)
            if self._command_journal:
                self._command_journal.log_received(
                    cmd_id, cmd.idempotency_key, cmd.command_type, cmd.payload_ref
                )
                self._command_journal.mark_in_progress(cmd_id)

            ok, result, err = self._execute_cloud_command(cmd)

            # Update journal with result (Design Doc 10.4.2)
            if self._command_journal:
                if ok:
                    self._command_journal.mark_completed(cmd_id, result)
                else:
                    self._command_journal.mark_failed(cmd_id, err or "Unknown error")

            try:
                self._cloud_client.submit_command_result(
                    command_id=cmd.id,
                    success=ok,
                    result=result if ok else None,
                    error_message=err if not ok else None,
                )
                self._executed_command_ids.add(cmd_id)
            except Exception:
                pass

    def _execute_cloud_command(self, cmd: PendingCommand) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Execute a cloud command locally (Agent-side)."""
        try:
            if cmd.command_type == "REQUEST_START_RUN":
                success, error = self.start(run_id=str(cmd.run_id) if cmd.run_id else None)
                return (success, {"action": "started", "run_id": self._status.active_run_id}, error)
            if cmd.command_type == "REQUEST_STOP_RUN":
                stopped = self.stop(reason="cloud_requested")
                return (stopped, {"action": "stopped"}, None if stopped else "Failed to stop")
            if cmd.command_type == "REQUEST_PAUSE_RUN":
                paused = self.pause()
                return (paused, {"action": "paused"}, None if paused else "Failed to pause")
            if cmd.command_type == "REQUEST_UPGRADE_ARTIFACT":
                # Design Doc Phase 4: Full artifact upgrade with cryptographic verification
                return self._handle_upgrade_artifact(cmd)
            if cmd.command_type == "REQUEST_UPDATE_CONFIG":
                # Design Doc Phase 4: Config update with validation
                return self._handle_update_config(cmd)
            if cmd.command_type == "REQUEST_ROTATE_AGENT_SESSION":
                return (True, {"action": "rotate_session_acknowledged"}, None)
            if cmd.command_type == "REQUEST_EXPORT_LOGS":
                return (True, {"action": "export_logs_acknowledged"}, None)
            return (False, {}, f"Unknown command_type: {cmd.command_type}")
        except Exception as e:
            return (False, {}, str(e))

    def decide_cloud_command_approval(
        self,
        command_id: str,
        *,
        approved: bool,
        reason: str = "",
        decided_by: str = "local_user",
    ) -> bool:
        """
        Decide a pending cloud approval and submit it back to Cloud.

        This is the local operator/CLI integration point.
        """
        if not self._cloud_client:
            return False
        req_id = self._pending_cloud_approval_by_command_id.get(command_id)
        if not req_id:
            return False

        finalized = self._approval_manager.decide(
            req_id,
            approved=approved,
            reason=reason,
            decided_by=decided_by,
        )
        if finalized is None:
            return False

        try:
            from uuid import UUID

            self._cloud_client.submit_local_approval(
                command_id=UUID(command_id),
                approved=approved,
                evidence_hash=f"sha256:{finalized.evidence_hash}" if finalized.evidence_hash else None,
                diff_summary={
                    "command_type": finalized.command_type,
                    "diff_summary": finalized.diff_summary,
                    "details": finalized.details,
                },
                reason=finalized.decision_reason or reason,
            )
            self._submitted_cloud_approvals.add(command_id)
            del self._pending_cloud_approval_by_command_id[command_id]
            return True
        except Exception:
            return False

    # ===== Status =====

    def _update_status(self) -> None:
        """Update status snapshot."""
        self._status.state = self._state
        self._status.timestamp = datetime.utcnow()
        self._status.started_at = self._started_at
        self._status.uptime_seconds = self._get_uptime_seconds()

        # Vault status
        if self._vault and hasattr(self._vault, "is_locked"):
            self._status.vault_unlocked = not self._vault.is_locked

        # Kill switch
        if self._kill_switch:
            self._status.kill_switch_triggered = self._kill_switch.is_triggered
            if self._kill_switch.current_halt:
                self._status.kill_switch_reason = self._kill_switch.current_halt.halt_reason.message

        # Degraded mode
        if self._degraded_manager:
            self._status.degraded_mode = self._degraded_manager.current_mode
            self._status.degraded_action = self._degraded_manager.current_action

    def _get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self._started_at:
            return (datetime.utcnow() - self._started_at).total_seconds()
        return 0.0

    # ===== Artifact & Config Handlers (Design Doc Phase 4) =====

    def _handle_upgrade_artifact(
        self,
        cmd: PendingCommand,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Handle REQUEST_UPGRADE_ARTIFACT command.

        Design Doc Phase 4 compliant:
        - Downloads artifact from presigned URL
        - Verifies SHA-256 digest
        - Performs REAL cryptographic signature verification via ArtifactVerifier
        - Prepares for atomic swap

        Args:
            cmd: Pending command with artifact info

        Returns:
            (success, result_data, error_message)
        """
        from packages.agent.daemon.artifact_manager import (
            ArtifactManager,
            ArtifactVerificationError,
            ArtifactDownloadError,
        )

        self._log_event(TelemetryEventType.STATE_CHANGE, {
            "message": "Starting artifact upgrade",
            "command_id": cmd.command_id,
            "payload_ref": cmd.payload_ref,
        })

        try:
            # Extract artifact info from command payload
            payload = cmd.payload_ref or {}
            if isinstance(payload, str):
                import json
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"url": payload}

            artifact_url = payload.get("download_url") or payload.get("url")
            expected_digest = payload.get("digest") or payload.get("expected_digest")
            deployment_id = payload.get("deployment_id") or cmd.deployment_id
            artifact_name = payload.get("name") or payload.get("artifact_name")
            signature_info = payload.get("signature")

            if not artifact_url:
                return (False, {}, "Missing artifact download URL")

            if not expected_digest:
                return (False, {}, "Missing expected digest (Design Doc requires digest verification)")

            # Initialize artifact manager
            artifact_cache_dir = self.config.data_dir / "artifacts"
            artifact_manager = ArtifactManager(cache_dir=artifact_cache_dir)

            # Try to get ArtifactVerifier for crypto verification
            # Per Design Doc 8.3: Agent verifies digest + signature + allowlist registry
            verifier = None
            try:
                from ccea.artifact.verifier import ArtifactVerifier as CCEAVerifier
                from ccea.artifact.verifier import create_verifier_from_key_manager
                from ccea.crypto.key_manager import KeyManager

                # Load trusted keys from config/keychain per Design Doc 8.3
                # Priority: 1) KeyManager from trusted_keys dir, 2) Strict mode (reject unsigned)
                trusted_keys_path = self.config.data_dir / "trusted_keys"

                if trusted_keys_path.exists():
                    try:
                        key_manager = KeyManager(keys_dir=trusted_keys_path)
                        verifier = create_verifier_from_key_manager(
                            key_manager=key_manager,
                            allowed_registries={"local", "ccea-registry"},
                            require_sbom=True,
                        )
                        self._log_event(TelemetryEventType.STATE_CHANGE, {
                            "message": "ArtifactVerifier initialized with KeyManager",
                            "trusted_keys_path": str(trusted_keys_path),
                            "key_count": len(key_manager.list_keys()),
                        })
                    except Exception as km_err:
                        self._log_event(TelemetryEventType.WARNING, {
                            "message": f"KeyManager init failed, using strict verifier: {km_err}",
                        })
                        verifier = CCEAVerifier(strict_mode=True)
                else:
                    # No trusted keys configured - strict mode rejects unsigned artifacts
                    verifier = CCEAVerifier(strict_mode=True)
                    self._log_event(TelemetryEventType.WARNING, {
                        "message": "No trusted_keys directory - strict mode (unsigned rejected)",
                        "expected_path": str(trusted_keys_path),
                    })
            except ImportError:
                self._log_event(TelemetryEventType.WARNING, {
                    "message": "ArtifactVerifier not available - signature verification limited",
                })

            # Download and verify artifact
            artifact, verification_report = artifact_manager.download_verify_and_prepare(
                url=artifact_url,
                expected_digest=expected_digest,
                deployment_id=deployment_id,
                artifact_name=artifact_name,
                signature_info=signature_info,
                verifier=verifier,
            )

            # Prepare for upgrade (don't activate yet - may need approval)
            current_artifact = artifact_manager.get_active_artifact()
            current_artifact_id = current_artifact.artifact_id if current_artifact else None

            ready = artifact_manager.prepare_upgrade(current_artifact_id, artifact)
            if not ready:
                return (False, {}, "Artifact not ready for upgrade")

            # Log success
            self._log_event(TelemetryEventType.STATE_CHANGE, {
                "message": "Artifact downloaded and verified",
                "artifact_id": artifact.artifact_id,
                "digest_verified": artifact.verified,
                "crypto_verified": verification_report.signature_verified if verification_report else False,
            })

            return (True, {
                "action": "upgrade_prepared",
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "version": artifact.version,
                "digest_verified": artifact.verified,
                "crypto_verified": verification_report.signature_verified if verification_report else False,
                "ready_for_swap": ready,
            }, None)

        except ArtifactVerificationError as e:
            self._log_event(TelemetryEventType.ERROR, {
                "error_type": "ArtifactVerificationError",
                "message": str(e),
            })
            return (False, {}, f"Artifact verification failed: {e}")

        except ArtifactDownloadError as e:
            self._log_event(TelemetryEventType.ERROR, {
                "error_type": "ArtifactDownloadError",
                "message": str(e),
            })
            return (False, {}, f"Artifact download failed: {e}")

        except Exception as e:
            self._log_event(TelemetryEventType.ERROR, {
                "error_type": "ArtifactUpgradeError",
                "message": str(e),
            })
            return (False, {}, f"Artifact upgrade failed: {e}")

    def _handle_update_config(
        self,
        cmd: PendingCommand,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Handle REQUEST_UPDATE_CONFIG command.

        Design Doc Phase 4 compliant:
        - Validates config against schema
        - Checks change class (TRADING_IMPACTING requires local approval)
        - Applies config via policy firewall validation

        Args:
            cmd: Pending command with config info

        Returns:
            (success, result_data, error_message)
        """
        self._log_event(TelemetryEventType.STATE_CHANGE, {
            "message": "Processing config update",
            "command_id": cmd.command_id,
            "payload_ref": cmd.payload_ref,
        })

        try:
            # Extract config from command payload
            payload = cmd.payload_ref or {}
            if isinstance(payload, str):
                import json
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    return (False, {}, "Invalid config payload format")

            config_blob = payload.get("config") or payload.get("config_blob") or payload
            change_class = payload.get("change_class", "NON_IMPACTING")

            # Validate change class
            if change_class == "TRADING_IMPACTING":
                # This should have been caught by approval flow, but double-check
                if not cmd.requires_approval:
                    self._log_event(TelemetryEventType.WARNING, {
                        "message": "TRADING_IMPACTING config update without approval flag",
                    })

            # Validate config through policy firewall if available
            if self._policy_firewall:
                if hasattr(self._policy_firewall, "check_config_change"):
                    result = self._policy_firewall.check_config_change(config_blob, change_class)
                    if not result.allowed:
                        self._log_event(TelemetryEventType.ERROR, {
                            "error_type": "ConfigValidationError",
                            "message": f"Config rejected by policy firewall: {result.violations}",
                        })
                        return (False, {
                            "action": "config_rejected",
                            "violations": result.violations,
                        }, f"Config rejected by policy: {result.violations}")

            # Apply config (store for runtime use)
            config_path = self.config.data_dir / "runtime_config.json"
            import json
            with open(config_path, "w") as f:
                json.dump({
                    "config": config_blob,
                    "change_class": change_class,
                    "applied_at": datetime.utcnow().isoformat(),
                    "command_id": cmd.command_id,
                }, f, indent=2)

            self._log_event(TelemetryEventType.STATE_CHANGE, {
                "message": "Config update applied",
                "change_class": change_class,
            })

            return (True, {
                "action": "config_applied",
                "change_class": change_class,
                "config_path": str(config_path),
            }, None)

        except Exception as e:
            self._log_event(TelemetryEventType.ERROR, {
                "error_type": "ConfigUpdateError",
                "message": str(e),
            })
            return (False, {}, f"Config update failed: {e}")

    # ===== Logging =====

    def _log_event(
        self,
        event_type: TelemetryEventType,
        data: Dict[str, Any],
    ) -> None:
        """Log event to telemetry buffer."""
        if self._telemetry_buffer:
            event = TelemetryEvent(
                event_type=event_type,
                data=data,
                run_id=self._status.active_run_id,
            )
            self._telemetry_buffer.add(event)

    # ===== Setters =====

    def set_vault(self, vault: Any) -> None:
        """Set local vault."""
        self._vault = vault
        if self._preflight_checker:
            self._preflight_checker._vault = vault

    def set_policy_firewall(self, firewall: Any) -> None:
        """Set policy firewall."""
        self._policy_firewall = firewall
        if self._preflight_checker:
            self._preflight_checker._policy_firewall = firewall

    def set_hard_cap_enforcer(self, enforcer: Any) -> None:
        """Set hard cap enforcer."""
        self._hard_cap_enforcer = enforcer
        if self._preflight_checker:
            self._preflight_checker._hard_cap_enforcer = enforcer

    def set_live_runner(self, runner: Any) -> None:
        """Set live runner."""
        self._live_runner = runner

    def set_broker_connector(self, connector: Any) -> None:
        """
        Set broker connector.

        Also updates the kill switch executor to use the new connector
        for order cancellation and position flattening.
        """
        self._broker_connector = connector
        if self._preflight_checker:
            self._preflight_checker._broker_connector = connector
        if self._kill_switch_executor:
            self._kill_switch_executor.set_broker_connector(connector)
        if self._run_controller:
            self._run_controller.set_broker_connector(connector)

    def set_reconciler(self, reconciler: PositionReconciler) -> None:
        """Set position reconciler."""
        self._reconciler = reconciler
        if self._run_controller:
            self._run_controller.set_reconciler(reconciler)

    def set_on_state_change(self, callback: Callable[[DaemonState], None]) -> None:
        """Set state change callback."""
        self._on_state_change = callback

    def set_on_kill_switch(self, callback: Callable[[HaltReason], None]) -> None:
        """Set kill switch callback."""
        self._on_kill_switch = callback

    def set_on_degraded_mode(self, callback: Callable[[DegradedMode, DegradedModeAction], None]) -> None:
        """Set degraded mode callback."""
        self._on_degraded_mode = callback

    # ===== Cleanup =====

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        if self._state in (DaemonState.RUNNING, DaemonState.PAUSED):
            self.stop(reason="process_exit")
        if self._cloud_client:
            try:
                self._cloud_client.close()
            except Exception:
                pass

    # ===== API =====

    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary."""
        return self.status.to_dict()

    def get_kill_switch_status(self) -> Dict[str, Any]:
        """Get kill switch status."""
        if self._kill_switch:
            return {
                "triggered": self._kill_switch.is_triggered,
                "current_halt": self._kill_switch.current_halt.to_dict() if self._kill_switch.current_halt else None,
                "history_count": len(self._kill_switch.halt_history),
            }
        return {"triggered": False}

    def get_degraded_mode_status(self) -> Dict[str, Any]:
        """Get degraded mode status."""
        if self._degraded_manager:
            return self._degraded_manager.get_status()
        return {"is_degraded": False}

    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        if self._telemetry_buffer:
            return self._telemetry_buffer.get_statistics()
        return {}

    def get_time_sync_status(self) -> Dict[str, Any]:
        """Get time sync status."""
        if self._time_checker:
            return self._time_checker.get_statistics()
        return {}

# (No extra aliases)
