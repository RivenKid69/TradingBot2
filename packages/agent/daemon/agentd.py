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
        self._time_checker: Optional[TimeSyncChecker] = None
        self._preflight_checker: Optional[PreflightChecker] = None
        self._degraded_manager: Optional[DegradedModeManager] = None
        self._telemetry_buffer: Optional[TelemetryBuffer] = None
        self._sandbox: Optional[Sandbox] = None
        self._approval_manager: ApprovalManager = ApprovalManager()
        self._cloud_client: Optional[CloudClient] = None
        self._executed_command_ids: set[str] = set()
        self._pending_cloud_approval_by_command_id: Dict[str, Any] = {}
        self._submitted_cloud_approvals: set[str] = set()

        # External components (set via setters)
        self._policy_firewall: Optional[Any] = None
        self._hard_cap_enforcer: Optional[Any] = None
        self._live_runner: Optional[Any] = None
        self._reconciler: Optional[Any] = None
        self._broker_connector: Optional[Any] = None

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
        """Initialize kill switch manager."""
        self._kill_switch = KillSwitchManager(
            config=self.config.kill_switch_config,
            on_trigger=self._handle_kill_switch,
        )

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
                else:
                    self._status.cloud_connected = False

            except Exception:
                self._status.cloud_connected = False
                pass

            self._stop_event.wait(self.config.heartbeat_interval_seconds)

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

        Security invariant:
        - TRADING_IMPACTING commands MUST require local approval; Cloud cannot bypass by flipping flags.
        """
        if not self._cloud_client:
            return

        for cmd in commands:
            cmd_id = str(cmd.id)
            if cmd_id in self._executed_command_ids:
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

            ok, result, err = self._execute_cloud_command(cmd)
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
                # Placeholder: artifact download/verification is Phase 9 (integrity) and runner integration.
                self._log_event(TelemetryEventType.STATE_CHANGE, {"message": "Upgrade requested", "payload_ref": cmd.payload_ref})
                return (True, {"action": "upgrade_acknowledged"}, None)
            if cmd.command_type == "REQUEST_UPDATE_CONFIG":
                # Placeholder: config blob retrieval + validation is handled by higher-level runner/config subsystems.
                self._log_event(TelemetryEventType.STATE_CHANGE, {"message": "Config update requested", "payload_ref": cmd.payload_ref})
                return (True, {"action": "config_update_acknowledged"}, None)
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
        """Set broker connector."""
        self._broker_connector = connector
        if self._preflight_checker:
            self._preflight_checker._broker_connector = connector

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
